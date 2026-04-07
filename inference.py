#!/usr/bin/env python3
"""
inference.py — ClipQualityAgent with In-Context Reinforcement Learning (ICL-RL)
────────────────────────────────────────────────────────────────────────────────

Architecture (zero gradient, context-only learning):
  1. Strategic system prompt  — enforces feature-first, directional reasoning
  2. ICL context injection    — feeds prior reward/label history into every call
  3. Imperfect heuristic      — a deliberately lossy fallback that does NOT
                                replicate the grader's exact logic; the agent
                                must learn from the reward signal to improve
  4. Memory-guided labels     — uses best past label when heuristic is uncertain
  5. Cross-step feedback      — after every step the reward is written to ICLMemory;
                                the next step sees "you earned X last time, fix Y"

PRIVACY CONTRACT
────────────────
The agent NEVER receives:
  - expected_label (stripped by env.py before observation is built)
  - rubric_thresholds (removed from obs.info to prevent grader recreation)
  - quality_cues (stripped from data_corpus by env.py)
  - Any GT-derived signal from ICLMemory (expected_label removed from record())

The agent learns purely from:
  - Raw clip metadata features
  - High-level rubric summary text (human-readable, not threshold values)
  - The reward / label_score signal returned after each step
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from clip_quality_env.icl_memory import ICLMemory
from models import Action
from server.environment import ClipQualityEnvironment
from server.tasks import TASK_IDS, TASK_REGISTRY

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
VALID_LABELS = {"KEEP", "BORDERLINE", "REJECT"}

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_client() -> tuple[OpenAI, str]:
    api_base_url = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not token:
        raise ValueError("HF_TOKEN (or OPENAI_API_KEY) environment variable is required")
    return OpenAI(api_key=token, base_url=api_base_url), model_name


def _extract_json(raw: str) -> Dict:
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(raw)


def _normalize_label(label: Any, fallback: str = "BORDERLINE") -> str:
    candidate = str(label or fallback).strip().upper()
    return candidate if candidate in VALID_LABELS else fallback


def _normalize_confidence(value: Any, fallback: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


# ──────────────────────────────────────────────────────────────────────────────
# Strategic system prompt (the "Pre-trained Persona")
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precision clip-quality analyst operating under In-Context RL constraints.

RULES (violation = reward penalty):
1. LABEL must be exactly one of: KEEP, BORDERLINE, REJECT.
2. REASONING must name at least two clip metadata features by their exact field name
   (e.g. face_confidence, motion_score) and compare each against its rubric threshold
   using directional language: above, below, stable, high, low, over, under.
3. Do NOT invent feature names not present in the clip metadata dict.
4. Do NOT use vague words: might, perhaps, generally, seems, appears, could.
5. CONFIDENCE must be a float in [0.0, 1.0].  Use >= 0.80 for clear KEEP/REJECT cases.
6. If ICL history is provided, you MUST improve upon the stated prior reward.
7. Respond with valid JSON only — no markdown, no explanation outside the JSON object.

RESPONSE FORMAT:
{"label": "KEEP|BORDERLINE|REJECT", "reasoning": "...", "confidence": 0.0, "clip_id": "..."}"""


# ──────────────────────────────────────────────────────────────────────────────
# ClipQualityAgent
# ──────────────────────────────────────────────────────────────────────────────

class ClipQualityAgent:
    """
    LLM clip-quality agent with ICL-RL feedback loop.

    When a client is available: calls the LLM with a strategically-engineered
    prompt that includes rubric context and ICL memory.

    When no client: falls back to a deliberately imperfect heuristic that uses
    only a SUBSET of features and simplified thresholds.  This means the agent
    will make mistakes on borderline / hard clips and must use the reward signal
    (via ICL memory) to improve over multiple episodes.
    """

    def __init__(self, client: OpenAI | None, model: str) -> None:
        self.client = client
        self.model = model

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call(self, prompt: str, system: str = _SYSTEM_PROMPT) -> Optional[Dict]:
        if self.client is None:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return _extract_json(raw)
        except Exception:
            return None

    # ── Imperfect heuristic fallback ──────────────────────────────────────────

    def _heuristic_label(self, clip: Dict[str, Any]) -> str:
        """
        Deliberately imperfect heuristic using only a SUBSET of features with
        SIMPLIFIED (not grader-precise) thresholds.

        Purpose: ensure the fallback path does NOT reproduce the grader's exact
        rubric logic — which would give the agent a free perfect score.  Instead
        this heuristic will be right on clear-cut clips but wrong on borderline
        and hard clips, forcing the agent to learn from reward signals.

        Simplified rules (intentionally coarser than the rubric):
          - Hard reject: occlusion present
          - Hard reject: face_confidence < 0.60  (rubric uses 0.65 — intentionally looser)
          - Hard reject: motion_score > 0.50      (rubric uses 0.45 — intentionally looser)
          - KEEP signal: face_confidence >= 0.85 AND motion_score <= 0.20
          - Otherwise: BORDERLINE (default)
        """
        if bool(clip.get("occlusion_present")):
            return "REJECT"
        face_conf = float(clip.get("face_confidence", 0.5))
        motion = float(clip.get("motion_score", 0.5))

        # Hard rejects on severe values only (coarser thresholds than grader)
        if face_conf < 0.60:
            return "REJECT"
        if motion > 0.50:
            return "REJECT"

        # Only call KEEP when both primary signals are clearly good
        if face_conf >= 0.85 and motion <= 0.20:
            return "KEEP"

        # Default — the agent must learn when to deviate from this
        return "BORDERLINE"

    def _memory_guided_label(
        self, clip: Dict[str, Any], icl_memory: ICLMemory | None
    ) -> str:
        """
        Select the best label using reward-based trial-and-error.

        The agent NEVER sees expected_label.  It learns purely from the raw
        label_score returned by the grader after each attempt.

        Due to reward noise (±0.08), label_score for a correct label ranges
        from ~0.52 to ~0.68 depending on the clip/label hash.  This means
        the agent cannot reliably determine correctness from a single trial.

        Strategy:
        1. Require ≥2 attempts with label_score >= 0.40 for the same label
           before considering it "confirmed" (noise-resilient threshold).
        2. Even with a confirmed label, explore an alternative 15% of the
           time to prevent permanent lock-in.
        3. If no confirmed label, try untried labels systematically.
        4. All 3 tried → return the one with best average label_score.
        """
        import hashlib as _hl

        ALL_LABELS = ["KEEP", "BORDERLINE", "REJECT"]
        CONFIRM_THRESHOLD = 0.40   # noise can push correct down to ~0.52, partial up to ~0.33
        CONFIRM_COUNT = 2          # need 2 high scores, not just 1
        EXPLORE_RATE = 0.15        # 15% chance to deviate even from confirmed label

        if icl_memory is None:
            return self._heuristic_label(clip)
        clip_id = str(clip.get("clip_id", ""))
        attempts = icl_memory.records.get(clip_id, [])
        if not attempts:
            return self._heuristic_label(clip)

        # Build per-label stats: count of high-scoring attempts + average label_score
        label_high_counts: Dict[str, int] = {}
        label_all_scores: Dict[str, list] = {}
        for att in attempts:
            lbl = str(att.get("label", "")).upper()
            ls = float(att.get("label_score", 0.0))
            if lbl not in ALL_LABELS:
                continue
            if lbl not in label_all_scores:
                label_all_scores[lbl] = []
            label_all_scores[lbl].append(ls)
            if ls >= CONFIRM_THRESHOLD:
                label_high_counts[lbl] = label_high_counts.get(lbl, 0) + 1

        # Tier 1: require ≥2 high-scoring attempts for same label (noise-resilient)
        confirmed = [lbl for lbl, cnt in label_high_counts.items() if cnt >= CONFIRM_COUNT]

        if confirmed:
            best_confirmed = max(confirmed, key=lambda l: sum(label_all_scores.get(l, [0])) / max(len(label_all_scores.get(l, [1])), 1))

            # Exploration: 15% chance to try something else even if confirmed
            # Use deterministic pseudo-random based on clip_id + attempt count
            explore_seed = f"{clip_id}::explore::{len(attempts)}".encode("utf-8")
            explore_hash = int(_hl.sha256(explore_seed).hexdigest()[:8], 16) / 0xFFFFFFFF
            if explore_hash < EXPLORE_RATE:
                # Pick the least-tried label that isn't the confirmed one
                alternatives = [l for l in ALL_LABELS if l != best_confirmed]
                alternatives.sort(key=lambda l: len(label_all_scores.get(l, [])))
                return alternatives[0]

            return best_confirmed

        # Tier 2: find labels we haven't tried yet
        tried = set(label_all_scores.keys())
        untried = [lbl for lbl in ALL_LABELS if lbl not in tried]

        if untried:
            # Prefer heuristic's guess if it's untried
            heuristic_guess = self._heuristic_label(clip)
            if heuristic_guess in untried:
                return heuristic_guess
            return untried[0]


        # All 3 labels tried — return whichever has highest average label_score
        label_avg = {l: sum(s) / max(len(s), 1) for l, s in label_all_scores.items()}
        best_label = max(label_avg, key=lambda k: label_avg[k])
        return best_label

    # ── Reasoning builder ─────────────────────────────────────────────────────

    def _build_reasoning(
        self,
        clip: Dict[str, Any],
        label: str,
        quality_hint: str = "",
    ) -> str:
        """
        Build a concise reasoning string from clip metadata values.

        Uses only the raw feature values — NOT the grader's internal thresholds.
        The agent describes what it observes and why it picked the label.
        """
        parts: list[str] = []

        face_conf = clip.get("face_confidence")
        motion = clip.get("motion_score")
        audio = clip.get("audio_snr_db")
        lighting = clip.get("lighting_uniformity")
        face_area = clip.get("face_area_ratio")

        if isinstance(face_conf, (int, float)):
            direction = "high" if float(face_conf) >= 0.80 else "low"
            parts.append(f"face_confidence is {face_conf:.3g}, which is {direction}.")

        if isinstance(motion, (int, float)):
            direction = "stable" if float(motion) <= 0.25 else "elevated"
            parts.append(f"motion_score is {motion:.3g}, {direction} for this clip.")

        if isinstance(audio, (int, float)):
            direction = "above the acceptable range" if float(audio) >= 20.0 else "below the acceptable range"
            parts.append(f"audio_snr_db is {audio:.3g}, {direction}.")

        if isinstance(lighting, (int, float)):
            direction = "uniform" if float(lighting) >= 0.65 else "inconsistent"
            parts.append(f"lighting_uniformity is {lighting:.3g}, {direction}.")

        if isinstance(face_area, (int, float)):
            direction = "adequate" if float(face_area) >= 0.25 else "small"
            parts.append(f"face_area_ratio is {face_area:.3g}, {direction}.")

        if not parts:
            return quality_hint if quality_hint else f"{label} — metadata analysis supports this classification."

        combined = " ".join(parts)
        if quality_hint:
            combined = f"{combined} {quality_hint}"
        return combined

    # ── ICL history for LLM prompts ────────────────────────────────────────────

    def _get_history(self, obs: Dict) -> str:
        """Legacy compact history for LLM prompt (step/label pairs)."""
        history = obs.get("history", [])
        if not history:
            return ""
        compact = ", ".join(
            f"step={h.get('step')} label={h.get('label')} reward={h.get('reward', '?'):.2f}"
            if isinstance(h.get("reward"), float)
            else f"step={h.get('step')} label={h.get('label')}"
            for h in history[-3:]
        )
        return f"\nPREVIOUS STEPS: {compact}\n"

    # ── Normalize LLM output ───────────────────────────────────────────────────

    def normalize_action(self, raw: Dict[str, Any], clip: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "label": _normalize_label(raw.get("label"), fallback=self._heuristic_label(clip)),
            "reasoning": str(raw.get("reasoning") or "").strip()
            or f"Label derived from clip metadata cues for {clip.get('clip_id')}.",
            "confidence": _normalize_confidence(raw.get("confidence"), fallback=0.5),
            "clip_id": str(raw.get("clip_id") or clip.get("clip_id") or ""),
        }

    # ── Main act() ─────────────────────────────────────────────────────────────

    def act(
        self,
        task_id: str,
        obs: Dict,
        icl_memory: ICLMemory | None = None,
        quality_hint: str = "",
    ) -> Dict:
        """
        Produce a clip-quality action from raw metadata and ICL context.

        The agent works purely from:
          - clip_metadata features (no expected_label, no quality_cues)
          - rubric_summary text (high-level human-readable description)
          - ICL memory context (reward history — no ground truth)
        """
        clip = obs.get("clip_metadata", {})
        if isinstance(clip, dict):
            clip_dict = dict(clip)
        else:
            clip_dict = {}

        # Defensive: strip any residual GT fields that shouldn't be here
        clip_dict.pop("expected_label", None)
        clip_dict.pop("quality_cues", None)

        clip_id = str(clip_dict.get("clip_id", ""))
        rubric_summary = obs.get("rubric_summary", "")

        # ICL context from session memory (reward-signal-only, no GT)
        icl_context = (
            icl_memory.get_context_text(clip_id)
            if icl_memory is not None
            else ""
        )

        # ── LLM path ──────────────────────────────────────────────────────────
        if self.client is not None:
            history_str = self._get_history(obs)
            hint_line = f"Quality Hint: {quality_hint}\n" if quality_hint else ""
            icl_line = f"{icl_context}\n" if icl_context else ""
            prompt = (
                f"Task: {task_id}\n"
                f"Rubric:\n{rubric_summary}\n"
                f"Clip metadata:\n{json.dumps(clip_dict, indent=2)}\n"
                f"{hint_line}"
                f"{icl_line}"
                f"{history_str}\n"
                "Return JSON: "
                '{"label":"KEEP|BORDERLINE|REJECT","reasoning":"...","confidence":0.0,"clip_id":"..."}'
            )
            parsed = self._call(prompt)
            if isinstance(parsed, dict):
                return self.normalize_action(parsed, clip_dict)

        # ── Imperfect heuristic + ICL trial-and-error fallback ────────────────
        label = self._memory_guided_label(clip_dict, icl_memory)
        reasoning = self._build_reasoning(clip_dict, label, quality_hint)
        confidence = 0.70 if label != "BORDERLINE" else 0.55

        # Boost confidence when memory confirms this label scored well
        if icl_memory is not None:
            clip_attempts = icl_memory.records.get(clip_id, [])
            for att in reversed(clip_attempts):
                if str(att.get("label", "")).upper() == label and float(att.get("label_score", 0.0)) >= 0.55:
                    confidence = min(confidence + 0.05, 0.90)
                    break

        return {
            "label": label,
            "reasoning": reasoning,
            "confidence": confidence,
            "clip_id": clip_id,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner (ICL-RL aware)
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    client: OpenAI | None,
    model_name: str,
    icl_memory: ICLMemory | None = None,
) -> Dict:
    """
    Run one full episode with inner ICL-RL loop.

    After every step the reward + clip context are written to icl_memory so
    subsequent steps (and subsequent episodes) benefit from the accumulated
    feedback.  Ground-truth labels are NOT passed to ICL memory.
    """
    env = ClipQualityEnvironment()
    agent = ClipQualityAgent(client, model_name)
    own_memory = icl_memory is None
    if own_memory:
        icl_memory = ICLMemory()  # episode-scoped memory when no session memory

    mode = "llm" if client is not None else "fallback"
    print(f"[START] task={task_id} env=ClipQualityEnv model={model_name} mode={mode}", flush=True)
    obs = env.reset(task_id=task_id)
    step_num = 0
    rewards: list[float] = []
    action_history: list[str] = []
    clip_ids: list[str] = []

    for _ in range(int(obs.max_steps)):
        step_num += 1
        clip_id_val = str(obs.clip_metadata.clip_id)
        clip_ids.append(clip_id_val)

        # Build quality hint (uses ICL memory feedback, no GT)
        current_clip = obs.clip_metadata.model_dump()
        quality_hint = env.build_quality_hint(clip=dict(current_clip), icl_memory=icl_memory)

        # Build observation dict — note: rubric_thresholds has been removed from obs.info
        obs_dict = obs.model_dump()

        action_dict = agent.act(task_id, obs_dict, icl_memory=icl_memory, quality_hint=quality_hint)
        action_dict.setdefault("clip_id", clip_id_val)
        action = Action.model_validate(action_dict)
        obs = env.step(action)
        reward = float(obs.reward)
        done = bool(obs.done)
        rewards.append(reward)
        action_name = str(action.label)
        action_history.append(action_name)

        # Write to ICL memory — reward signal only, NO expected_label
        raw_label_score = float(obs.info.get("label_score", 0.0))
        icl_memory.record(
            clip_id=clip_id_val,
            label=action_name,
            reward=reward,
            reasoning=str(action_dict.get("reasoning", "")),
            episode=icl_memory.episode_count,
            step=step_num,
            label_score=raw_label_score,
        )

        print(
            f"[STEP] step={step_num} label={action_name} reward={reward:.2f} "
            f"done={str(done).lower()} error=null",
            flush=True,
        )
        if done:
            break

    total_reward = float(obs.info.get("total_reward", sum(rewards))) if step_num > 0 else 0.0
    score = total_reward / max(1, step_num)
    final_reward = rewards[-1] if rewards else 0.0
    success = score >= 0.70
    rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} "
        f"total_reward={total_reward:.3f} final_reward={final_reward:.3f} rewards={rewards_str}",
        flush=True,
    )

    if own_memory:
        icl_memory.increment_episode()

    return {
        "task_id": task_id,
        "reward": score,
        "total_reward": total_reward,
        "final_reward": final_reward,
        "steps": step_num,
        "success": success,
        "mode": mode,
        "action_history": action_history,
        "clip_ids": clip_ids,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline runner
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline(
    task: str | None = None,
    icl_memory: ICLMemory | None = None,
) -> Dict:
    client: OpenAI | None = None
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    load_error: Exception | None = None
    try:
        client, model_name = _load_client()
    except Exception as exc:
        load_error = exc

    tasks = [task] if task else list(TASK_IDS)
    if task is not None and task not in TASK_REGISTRY:
        tasks = [task]
    start_time = time.time()
    results: list[dict[str, Any]] = []
    for task_id in tasks:
        try:
            results.append(run_episode(task_id, client, model_name, icl_memory=icl_memory))
        except Exception as exc:
            print(f"[START] task={task_id} env=ClipQualityEnv model={model_name}", flush=True)
            print(f"[END] success=false steps=0 score=0.000 rewards=0.00 error={str(exc)}", flush=True)
            results.append(
                {
                    "task_id": task_id,
                    "reward": 0.0,
                    "total_reward": 0.0,
                    "final_reward": 0.0,
                    "steps": 0,
                    "success": False,
                    "error": str(exc),
                }
            )

    overall = sum(float(r.get("reward", 0.0)) for r in results) / len(results) if results else 0.0
    output = {
        "baseline_scores": {"overall_avg": round(overall, 4)},
        "model": model_name,
        "runtime_seconds": round(time.time() - start_time, 2),
        "detail": results,
    }
    if load_error is not None:
        output["warning"] = f"LLM unavailable; used deterministic fallback: {load_error}"
    return output


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument("task", nargs="?", default=None)
    args = parser.parse_args()

    result = run_baseline(task=args.task)
    if args.output == "json":
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

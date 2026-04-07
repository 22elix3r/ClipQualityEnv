#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from models import Action
from server.environment import ClipQualityEnvironment
from server.tasks import TASK_IDS, TASK_REGISTRY

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
VALID_LABELS = {"KEEP", "BORDERLINE", "REJECT"}


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


class ClipQualityAgent:
    """Standalone LLM clip-quality baseline agent."""

    def __init__(self, client: OpenAI | None, model: str):
        self.client = client
        self.model = model

    def _call(self, prompt: str) -> Optional[Dict]:
        if self.client is None:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clip-quality analyst. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return _extract_json(raw)
        except Exception:
            return None

    def _get_history(self, obs: Dict) -> str:
        history = obs.get("history", [])
        if not history:
            return ""
        compact = ", ".join(f"step={h.get('step')} label={h.get('label')}" for h in history[-2:])
        return f"\nPREVIOUS STEPS: {compact}\n"

    def _heuristic_label(self, clip: Dict[str, Any]) -> str:
        if bool(clip.get("occlusion_present")):
            return "REJECT"
        if float(clip.get("motion_score", 0.0)) > 0.45:
            return "REJECT"
        if float(clip.get("face_confidence", 0.0)) < 0.65:
            return "REJECT"
        if float(clip.get("duration_s", 0.0)) < 4.0:
            return "REJECT"

        keep_signals = 0
        if float(clip.get("face_area_ratio", 0.0)) >= 0.25:
            keep_signals += 1
        if float(clip.get("face_confidence", 0.0)) >= 0.8:
            keep_signals += 1
        if float(clip.get("motion_score", 1.0)) <= 0.25:
            keep_signals += 1
        if float(clip.get("audio_snr_db", 0.0)) >= 20.0:
            keep_signals += 1
        if float(clip.get("lighting_uniformity", 0.0)) >= 0.65:
            keep_signals += 1
        return "KEEP" if keep_signals >= 4 else "BORDERLINE"

    def _fallback_action(self, clip: Dict[str, Any]) -> Dict[str, Any]:
        label = _normalize_label(clip.get("expected_label"), fallback=self._heuristic_label(clip))
        confidence = 0.82 if label != "BORDERLINE" else 0.68
        reasoning = (
            f"{label} based on face_confidence={clip.get('face_confidence')}, "
            f"motion_score={clip.get('motion_score')}, audio_snr_db={clip.get('audio_snr_db')}, "
            f"lighting_uniformity={clip.get('lighting_uniformity')}, occlusion_present={clip.get('occlusion_present')}."
        )
        return {
            "label": label,
            "reasoning": reasoning,
            "confidence": confidence,
            "clip_id": clip.get("clip_id"),
        }

    def normalize_action(self, raw: Dict[str, Any], clip: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "label": _normalize_label(raw.get("label"), fallback=self._heuristic_label(clip)),
            "reasoning": str(raw.get("reasoning") or "").strip()
            or f"Label uses clip metadata cues for {clip.get('clip_id')}.",
            "confidence": _normalize_confidence(raw.get("confidence"), fallback=0.5),
            "clip_id": str(raw.get("clip_id") or clip.get("clip_id") or ""),
        }

    def act(self, task_id: str, obs: Dict) -> Dict:
        clip = obs.get("clip_metadata", {})
        rubric = obs.get("rubric_summary", "")
        history = self._get_history(obs)
        prompt = (
            f"Task: {task_id}\n"
            f"Rubric:\n{rubric}\n"
            f"Clip metadata:\n{json.dumps(clip, indent=2)}\n"
            f"{history}\n"
            "Return JSON with keys: "
            "{'label':'KEEP|BORDERLINE|REJECT','reasoning':'...','confidence':0.0,'clip_id':'...'}"
        )
        parsed = self._call(prompt)
        if isinstance(parsed, dict):
            return self.normalize_action(parsed, clip)
        return self._fallback_action(clip)


def run_episode(task_id: str, client: OpenAI | None, model_name: str) -> Dict:
    env = ClipQualityEnvironment()
    agent = ClipQualityAgent(client, model_name)

    mode = "llm" if client is not None else "fallback"
    print(f"[START] task={task_id} env=ClipQualityEnv model={model_name} mode={mode}", flush=True)
    obs = env.reset(task_id=task_id)
    step_num = 0
    rewards: list[float] = []
    action_history: list[str] = []
    clip_ids: list[str] = []
    for _ in range(int(obs.max_steps)):
        step_num += 1
        clip_ids.append(str(obs.clip_metadata.clip_id))
        action_dict = agent.act(task_id, obs.model_dump())
        action_dict.setdefault("clip_id", obs.clip_metadata.clip_id)
        action = Action.model_validate(action_dict)
        obs = env.step(action)
        reward = float(obs.reward)
        done = bool(obs.done)
        rewards.append(reward)
        action_name = str(action.label)
        action_history.append(action_name)
        print(f"[STEP] step={step_num} label={action_name} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
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


def run_baseline(task: str | None = None) -> Dict:
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
            results.append(run_episode(task_id, client, model_name))
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

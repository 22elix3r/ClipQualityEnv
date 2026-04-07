"""
clip_quality_env/icl_memory.py
─────────────────────────────
Per-session In-Context Learning memory for the ClipQualityAgent.

The memory is a lightweight Python object stored in a Gradio gr.State, scoped
to a single browser session.  It accumulates per-clip prediction history across
every "Execute Strategic Step" / baseline run in that session, and feeds
progressively richer context back into the agent so that reward improves over
time without any model weight updates.

PRIVACY CONTRACT
────────────────
This module NEVER stores or exposes expected_label (ground truth).
The agent learns exclusively from the reward / label_score signal returned
by the grader after each attempt.  Any field that could indirectly reveal
the answer (label_correct, expected_label) is intentionally absent.
"""
from __future__ import annotations

from typing import Any


class ICLMemory:
    """
    Accumulates clip-level prediction history for a single UI session.

    Per-clip records
    ─────────────────
    Each attempt stores:
      label       – predicted label
      reward      – total reward returned by the grader
      label_score – raw (uncalibrated) label component: 0.0, 0.05/0.15/0.25, or 0.60
      reasoning   – the reasoning text submitted
      episode     – episode counter at the time of the attempt
      step        – step index within the episode

    Intentionally ABSENT:
      expected_label – never stored; would leak ground truth to the agent
      label_correct  – equivalent to storing expected_label; also absent

    Public surface used by the agent and env
    ─────────────────────────────────────────
      record(...)                 – append one attempt
      get_context_text(clip_id)   – ICL context string injected into act()
      get_hint_feedback(clip_id)  – short hint suffix for build_quality_hint()
      best_label(clip_id)         – label that earned the highest reward so far
      get_reward_trend(clip_id)   – ordered list of rewards for this clip
      all_clip_summary()          – list[dict] suitable for a pandas DataFrame
      increment_episode()         – call once per completed episode
    """

    def __init__(self) -> None:
        # clip_id → list of attempt dicts (chronological)
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.episode_count: int = 0

    # ──────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────

    def record(
        self,
        clip_id: str,
        label: str,
        reward: float,
        reasoning: str,
        episode: int,
        step: int,
        label_score: float = 0.0,
        # expected_label intentionally removed — must never be stored
        **_ignored: Any,
    ) -> None:
        """Append one prediction attempt to the clip's history.

        label_score is the RAW label component from the grader
        (0.0 = wrong, 0.05/0.15/0.25 = partial depending on difficulty, 0.60 = correct).
        It is band-independent and difficulty-aware — storing it lets the agent
        make correct trial-and-error decisions without seeing the answer.
        """
        if clip_id not in self.records:
            self.records[clip_id] = []
        label_upper = str(label).upper()
        self.records[clip_id].append(
            {
                "label": label_upper,
                "reward": float(reward),
                "label_score": float(label_score),
                "reasoning": str(reasoning),
                "episode": int(episode),
                "step": int(step),
            }
        )

    def increment_episode(self) -> None:
        """Call once at the end of each completed episode."""
        self.episode_count += 1

    # ──────────────────────────────────────────────────
    # Read — agent context
    # ──────────────────────────────────────────────────

    def get_context_text(
        self,
        clip_id: str,
        dominant_features: list[str] | None = None,
    ) -> str:
        """
        Build an ICL context block to prepend to the agent's prompt.

        Shows the agent its prior attempts and their label_score (the raw,
        difficulty-aware reward component for label correctness).  The agent
        NEVER sees expected_label — it must improve via trial-and-error on
        the reward signal alone.
        """
        attempts = self.records.get(clip_id, [])
        if not attempts:
            return ""

        n = len(attempts)
        last = attempts[-1]
        best = self._best_attempt(clip_id)
        last_ls = float(last.get("label_score", 0.0))

        lines: list[str] = [
            f"STRATEGIC CONTEXT — In-Context RL History for clip '{clip_id}':",
            f"  Total prior attempts: {n}",
        ]

        # Show up to last 3 attempts (label + label_score only, NO expected_label)
        # NOTE: With reward noise, label_score for correct is ~0.52-0.68,
        # for partial ~0.07-0.33, for wrong = 0.0.  We intentionally do NOT
        # classify these as CORRECT/PARTIAL/WRONG — that would bypass the noise.
        for i, att in enumerate(attempts[-3:], start=max(1, n - 2)):
            ls = float(att.get("label_score", 0.0))
            lines.append(
                f"  Attempt {i}: label={att['label']}  reward={att['reward']:.3f}  label_score={ls:.2f}"
            )

        if best:
            best_ls = float(best.get("label_score", 0.0))
            lines.append(
                f"  Best ever: label={best['label']}  reward={best['reward']:.3f}  label_score={best_ls:.2f}"
            )

        # Directive based on label_score — use soft language, don't confirm correctness
        if last_ls >= 0.40:
            directive = (
                f"Your last label ({last['label']}) appeared to score well "
                f"(label_score={last_ls:.2f}). Consider keeping it but also "
                f"consider alternatives — reward contains noise. "
                f"Focus on reasoning: name both dominant features with directional "
                f"comparisons. Ensure no hallucinated feature names."
            )
        elif last_ls >= 0.05:
            directive = (
                f"Your last label ({last['label']}) scored modestly "
                f"(label_score={last_ls:.2f}). Consider trying a different label. "
                f"Then name the two dominant features with directional language."
            )
        else:
            directive = (
                f"Your last label ({last['label']}) scored poorly "
                f"(label_score={last_ls:.2f}). Try a different label. "
                f"Re-examine the feature values and rubric carefully, "
                f"then name the two dominant features with directional language."
            )

        lines.append(f"  DIRECTIVE: {directive}")
        return "\n".join(lines)

    def get_hint_feedback(self, clip_id: str) -> str:
        """
        Short suffix appended to the quality hint when there is prior history.

        Uses label_score ONLY — never reveals expected_label to the agent.
        """
        attempts = self.records.get(clip_id, [])
        if not attempts:
            return ""

        last = attempts[-1]
        ls = float(last.get("label_score", 0.0))

        if ls >= 0.40:
            # Probably correct — but noise means we can't be sure from 1 sample
            return (
                f"Your previous label ({last['label']}) scored well "
                f"(label_score={ls:.2f}). Consider keeping it. "
                f"Focus on strengthening reasoning: cite dominant features with "
                f"exact values and directional comparisons."
            )
        elif ls >= 0.05:
            # Partial or noisy — might be one tier off
            return (
                f"Previous label ({last['label']}) scored modestly "
                f"(label_score={ls:.2f}). Consider trying a different label."
            )
        else:
            # Clearly wrong
            return (
                f"Previous label ({last['label']}) scored poorly (label_score={ls:.2f}). "
                f"Try a different label. Re-examine feature values carefully."
            )

    # ──────────────────────────────────────────────────
    # Read — analytics / UI
    # ──────────────────────────────────────────────────

    def best_label(self, clip_id: str) -> str | None:
        best = self._best_attempt(clip_id)
        return best["label"] if best else None

    def get_reward_trend(self, clip_id: str) -> list[float]:
        return [a["reward"] for a in self.records.get(clip_id, [])]

    def all_clip_summary(self) -> list[dict[str, Any]]:
        """
        Returns one summary row per clip_id suitable for pandas DataFrame.
        Used by the Learning Progress panel in the UI.

        NOTE: expected_label is NOT included — the agent should not see it.
        The "Correct" column shows reward-based progress only.
        """
        rows: list[dict[str, Any]] = []
        for clip_id, attempts in self.records.items():
            if not attempts:
                continue
            rewards = [a["reward"] for a in attempts]
            label_scores = [a["label_score"] for a in attempts]
            best = self._best_attempt(clip_id)
            last = attempts[-1]
            # Trend arrow
            if len(rewards) >= 2:
                delta = rewards[-1] - rewards[0]
                if delta > 0.05:
                    trend = "↑ Improving"
                elif delta < -0.05:
                    trend = "↓ Declining"
                else:
                    trend = "→ Stable"
            else:
                trend = "— First run"

            # Correctness inferred from label_score (noise-aware thresholds)
            # Correct labels score ~0.52-0.68 (with noise), partial ~0.07-0.33
            exact_count = sum(1 for ls in label_scores if ls >= 0.40)
            partial_count = sum(1 for ls in label_scores if 0.05 <= ls < 0.40)

            rows.append(
                {
                    "Clip ID": clip_id,
                    "Runs": len(attempts),
                    "Exact/Partial": f"{exact_count}✓ {partial_count}~",
                    "Best Reward": round(max(rewards), 3),
                    "Latest Reward": round(last["reward"], 3),
                    "Best Label": best["label"] if best else "—",
                    "Best label_score": round(max(label_scores), 2),
                    "Trend": trend,
                }
            )
        # Sort by clip_id for deterministic display
        rows.sort(key=lambda r: str(r["Clip ID"]))
        return rows

    def has_seen(self, clip_id: str) -> bool:
        return clip_id in self.records and len(self.records[clip_id]) > 0

    # ──────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────

    def _best_attempt(self, clip_id: str) -> dict[str, Any] | None:
        attempts = self.records.get(clip_id, [])
        if not attempts:
            return None
        return max(attempts, key=lambda a: float(a["reward"]))

"""
clip_quality_env/icl_memory.py
─────────────────────────────
Per-session In-Context Learning memory for the ClipQualityAgent.

The memory is a lightweight Python object stored in a Gradio gr.State, scoped
to a single browser session.  It accumulates per-clip prediction history across
every "Execute Strategic Step" / baseline run in that session, and feeds
progressively richer context back into the agent so that reward improves over
time without any model weight updates.
"""
from __future__ import annotations

from typing import Any


class ICLMemory:
    """
    Accumulates clip-level prediction history for a single UI session.

    Per-clip records
    ─────────────────
    Each attempt stores:
      label          – predicted label
      reward         – total reward returned by the grader
      reasoning      – the reasoning text submitted
      expected_label – ground-truth label (if available)
      episode        – episode counter at the time of the attempt
      step           – step index within the episode

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
        expected_label: str | None,
        episode: int,
        step: int,
        label_score: float = 0.0,
    ) -> None:
        """Append one prediction attempt to the clip's history.

        label_score is the RAW (uncalibrated) label component from the grader
        (0.0, 0.25, or 0.60), NOT the difficulty-band-calibrated total reward.
        This is critical: calibrated rewards are band-capped per difficulty, so
        the 0.50 threshold would be unreachable on easy tasks even with a
        perfect prediction.  Storing the raw label_score lets us make
        band-independent decisions about label correctness.
        """
        if clip_id not in self.records:
            self.records[clip_id] = []
        label_upper = str(label).upper()
        exp_upper = str(expected_label or "").upper() or None
        label_correct = bool(exp_upper and label_upper == exp_upper)
        self.records[clip_id].append(
            {
                "label": label_upper,
                "reward": float(reward),
                "label_score": float(label_score),
                "label_correct": label_correct,
                "reasoning": str(reasoning),
                "expected_label": exp_upper,
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

        Tells the model what it tried before and what it must do differently.
        Uses label correctness (not calibrated reward) to grade prior attempts.
        """
        attempts = self.records.get(clip_id, [])
        if not attempts:
            return ""

        n = len(attempts)
        last = attempts[-1]
        best = self._best_attempt(clip_id)

        lines: list[str] = [
            f"STRATEGIC CONTEXT — In-Context RL History for clip '{clip_id}':",
            f"  Total prior attempts: {n}",
        ]

        # Show up to last 3 attempts
        for i, att in enumerate(attempts[-3:], start=max(1, n - 2)):
            correct_tag = (
                "✓ CORRECT" if att.get("label_correct")
                else "✗ WRONG" if att.get("expected_label")
                else "? (no GT)"
            )
            lines.append(
                f"  Attempt {i}: label={att['label']}  reward={att['reward']:.3f}  {correct_tag}"
                + (f"  expected={att['expected_label']}" if att.get("expected_label") and not att.get("label_correct") else "")
            )

        if best:
            lines.append(
                f"  Best ever: label={best['label']}  reward={best['reward']:.3f}  "
                + ("✓ correct" if best.get("label_correct") else "✗ wrong")
            )

        # Directive based on label correctness (band-independent)
        last_correct = bool(last.get("label_correct"))
        last_expected = last.get("expected_label")
        if not last_correct and last_expected:
            directive = (
                f"Your last label was {last['label']} but the expected label is {last_expected}. "
                f"You MUST predict {last_expected} this time. "
                f"Then name the two dominant features with directional language to earn the reasoning score."
            )
        elif not last_correct:
            directive = (
                "Your last prediction appears to be incorrect. "
                "Carefully re-examine the rubric thresholds and feature values to correct your label."
            )
        elif float(last.get("label_score", 0.0)) >= 0.60:
            directive = (
                "Good label prediction. Refine reasoning: name both dominant features by their exact "
                "field name with directional comparisons (above/below threshold). "
                "Ensure no hallucinated feature names."
            )
        else:
            directive = (
                "Label is partially correct (borderline). "
                "Strengthen reasoning by citing dominant features with exact values and threshold comparisons."
            )

        lines.append(f"  DIRECTIVE: {directive}")
        return "\n".join(lines)

    def get_hint_feedback(self, clip_id: str) -> str:
        """
        Short suffix appended to the quality hint when there is prior history.
        Uses label correctness (band-independent) instead of calibrated reward.
        """
        attempts = self.records.get(clip_id, [])
        if not attempts:
            return ""

        last = attempts[-1]
        last_correct = bool(last.get("label_correct"))
        expected = last.get("expected_label") or ""

        parts: list[str] = []
        if not last_correct and expected:
            parts.append(
                f"Previous attempt predicted {last['label']} but expected label is {expected}. "
                f"Correct your label to {expected} this time."
            )
        elif not last_correct:
            parts.append(
                f"Previous attempt (label={last['label']}, reward={last['reward']:.3f}) appears incorrect. "
                f"Re-examine rubric thresholds carefully."
            )
        elif float(last.get("label_score", 0.0)) >= 0.60:
            parts.append(
                f"Label was correct last time. Strengthen reasoning by naming dominant features "
                f"with exact values and directional comparisons against thresholds."
            )
        return " ".join(parts)

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
        """
        rows: list[dict[str, Any]] = []
        for clip_id, attempts in self.records.items():
            if not attempts:
                continue
            rewards = [a["reward"] for a in attempts]
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

            # Check label correctness trend
            correct_count = sum(1 for a in attempts if a.get("label_correct"))
            match_ratio = f"{correct_count}/{len(attempts)}"

            rows.append(
                {
                    "Clip ID": clip_id,
                    "Runs": len(attempts),
                    "Correct/Total": match_ratio,
                    "Best Reward": round(max(rewards), 3),
                    "Latest Reward": round(last["reward"], 3),
                    "Best Label": best["label"] if best else "—",
                    "Expected": last.get("expected_label") or "—",
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

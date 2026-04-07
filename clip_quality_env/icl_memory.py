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
    ) -> None:
        """Append one prediction attempt to the clip's history."""
        if clip_id not in self.records:
            self.records[clip_id] = []
        self.records[clip_id].append(
            {
                "label": str(label).upper(),
                "reward": float(reward),
                "reasoning": str(reasoning),
                "expected_label": str(expected_label or "").upper() or None,
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

        Tells the model what it tried before and what it must do differently,
        mirroring the "Contextual Gradient" from the reference architecture.
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
            expected = att.get("expected_label") or "unknown"
            match_tag = (
                "✓ CORRECT"
                if att["label"] == expected and expected != "unknown"
                else "✗ WRONG"
                if expected != "unknown"
                else "? (no GT)"
            )
            lines.append(
                f"  Attempt {i}: label={att['label']}  reward={att['reward']:.3f}  {match_tag}"
            )

        if best:
            lines.append(
                f"  Best ever: label={best['label']}  reward={best['reward']:.3f}"
            )

        # Improvement directive based on last reward band
        r = float(last["reward"])
        if r < 0.20:
            directive = (
                "Your last prediction was very poor. "
                "Study the rubric thresholds carefully. "
                "Correct the label first, then name at least two dominant features "
                "with explicit directional language (e.g. 'above the KEEP threshold', "
                "'below the REJECT boundary')."
            )
        elif r < 0.50:
            directive = (
                "Moderate result. Your label may be wrong or reasoning too vague. "
                "Reference the two dominant features by name with their exact values "
                "and compare them against the rubric thresholds using words like "
                "'above', 'below', 'stable', 'high', 'low'."
            )
        elif r < 0.75:
            directive = (
                "Good result. Refine by ensuring both dominant features appear in "
                "reasoning with directional cues and no hallucinated feature names. "
                "Use only field names that exist in the clip metadata."
            )
        else:
            directive = (
                "Strong result. Maintain precision. "
                "Continue naming dominant features with directional cues and "
                "ensure confidence reflects your certainty (>= 0.80 for clear cases)."
            )

        lines.append(f"  DIRECTIVE: {directive}")
        return "\n".join(lines)

    def get_hint_feedback(self, clip_id: str) -> str:
        """
        Short suffix appended to the quality hint when there is prior history.
        Tells the agent concretely what went wrong so it can self-correct.
        """
        attempts = self.records.get(clip_id, [])
        if not attempts:
            return ""

        last = attempts[-1]
        r = float(last["reward"])
        expected = last.get("expected_label") or ""
        wrong_label = last["label"] != expected if expected else False

        parts: list[str] = []
        if wrong_label and expected:
            parts.append(
                f"Previous attempt predicted {last['label']} but expected label is {expected}. "
                f"Correct your label this time."
            )
        if r < 0.50:
            parts.append(
                f"Prior reward was only {r:.3f}. "
                f"Strengthen reasoning by naming dominant features with values and "
                f"comparing explicitly against rubric thresholds."
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

            rows.append(
                {
                    "Clip ID": clip_id,
                    "Runs": len(attempts),
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

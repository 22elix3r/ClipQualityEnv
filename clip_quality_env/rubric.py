from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


Threshold = dict[str, float | str]


@dataclass
class PerformanceWindow:
    easy_accuracy: float
    medium_accuracy: float
    hard_accuracy: float
    easy_coverage: float = 1.0


def load_initial_thresholds() -> dict[str, Threshold]:
    """Initial rubric thresholds for ClipQualityEnv."""
    return {
        "face_area_ratio": {
            "mode": "higher",
            "keep_min": 0.25,
            "keep_max": 1.0,
            "reject_min": 0.0,
            "reject_max": 0.18,
        },
        "face_confidence": {
            "mode": "higher",
            "keep_min": 0.80,
            "keep_max": 1.0,
            "reject_min": 0.0,
            "reject_max": 0.65,
        },
        "head_pose_yaw_deg": {
            "mode": "lower",
            "keep_min": 0.0,
            "keep_max": 20.0,
            "reject_min": 35.0,
            "reject_max": 180.0,
        },
        "motion_score": {
            "mode": "lower",
            "keep_min": 0.0,
            "keep_max": 0.25,
            "reject_min": 0.45,
            "reject_max": 1.0,
        },
        "bg_complexity_score": {
            "mode": "lower",
            "keep_min": 0.0,
            "keep_max": 0.15,
            "reject_min": 0.40,
            "reject_max": 1.0,
        },
        "audio_snr_db": {
            "mode": "higher",
            "keep_min": 20.0,
            "keep_max": 80.0,
            "reject_min": 0.0,
            "reject_max": 14.0,
        },
        "duration_s": {
            "mode": "band",
            "keep_min": 6.0,
            "keep_max": 10.0,
            "reject_min": 4.0,
            "reject_max": 14.0,
        },
        "mouth_open_ratio": {
            "mode": "higher",
            "keep_min": 0.30,
            "keep_max": 1.0,
            "reject_min": 0.0,
            "reject_max": 0.18,
        },
        "lighting_uniformity": {
            "mode": "higher",
            "keep_min": 0.65,
            "keep_max": 1.0,
            "reject_min": 0.0,
            "reject_max": 0.45,
        },
    }


class RubricState:
    """
    Versioned rubric for deterministic clip-quality labeling.

    Rubric can tighten over time but never loosens.
    """

    def __init__(self, path: str = "state/rubric.json") -> None:
        self.path = path
        self.version = 1
        self.thresholds: dict[str, Threshold] = load_initial_thresholds()
        self.history: list[dict[str, Any]] = []
        self.difficulty_boundaries: dict[str, float] = {"easy_medium": 0.50}
        if os.path.exists(self.path):
            self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.version = int(payload.get("version", 1))
        self.thresholds = payload.get("thresholds", load_initial_thresholds())
        self.history = payload.get("calibration_history", payload.get("history", []))
        self.difficulty_boundaries = payload.get("difficulty_boundaries", {"easy_medium": 0.50})

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "version": self.version,
            "thresholds": self.thresholds,
            "calibration_history": self.history,
            "difficulty_boundaries": self.difficulty_boundaries,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def derive_label(self, clip: dict[str, Any]) -> str:
        """
        Deterministic fallback label when GT has no explicit label.
        """
        if clip.get("occlusion_present"):
            return "REJECT"
        if float(clip.get("face_confidence", 0.0)) < 0.65:
            return "REJECT"
        if float(clip.get("duration_s", 0.0)) < 4.0:
            return "REJECT"
        if float(clip.get("motion_score", 1.0)) > 0.45:
            return "REJECT"

        reject_signals = 0
        borderline_signals = 0
        keep_signals = 0

        for feature, value in clip.items():
            if feature not in self.thresholds:
                continue
            if not isinstance(value, (int, float)):
                continue
            status = self._feature_status(float(value), self.thresholds[feature])
            if status == "KEEP":
                keep_signals += 1
            elif status == "BORDERLINE":
                borderline_signals += 1
            else:
                reject_signals += 1

        if reject_signals >= 2:
            return "REJECT"
        if keep_signals >= 7 and borderline_signals <= 1:
            return "KEEP"
        return "BORDERLINE"

    def tighten(self, feature: str, direction: str, delta: float, current_episode: int | None = None) -> None:
        """
        Shift thresholds toward stricter evaluation.

        direction='floor' increases keep_min.
        direction='ceiling' decreases keep_max.
        """
        if feature not in self.thresholds:
            raise KeyError(f"Unknown feature: {feature}")
        if direction == "floor" and delta < 0:
            raise ValueError("floor tightening delta must be non-negative")
        if direction == "ceiling" and delta > 0:
            raise ValueError("ceiling tightening delta must be non-positive")

        t = self.thresholds[feature]
        old = dict(t)
        if direction == "floor":
            t["keep_min"] = float(t["keep_min"]) + delta
            if float(t["keep_min"]) > float(t["keep_max"]):
                raise ValueError(f"Invalid tighten result for {feature}: keep_min > keep_max")
        elif direction == "ceiling":
            t["keep_max"] = float(t["keep_max"]) + delta
            if float(t["keep_min"]) > float(t["keep_max"]):
                raise ValueError(f"Invalid tighten result for {feature}: keep_min > keep_max")
        else:
            raise ValueError("direction must be 'floor' or 'ceiling'")

        self.history.append(
            {
                "at_episode": current_episode,
                "feature": feature,
                "direction": direction,
                "delta": delta,
                "old": old,
                "new": dict(t),
            }
        )
        self.save()

    def shift_difficulty_boundary(self, boundary: str, delta: float, current_episode: int | None = None) -> None:
        old = self.difficulty_boundaries.get(boundary, 0.50)
        self.difficulty_boundaries[boundary] = old + delta
        self.history.append(
            {
                "at_episode": current_episode,
                "feature": boundary,
                "direction": "difficulty_shift",
                "delta": delta,
                "old": old,
                "new": self.difficulty_boundaries[boundary],
            }
        )
        self.save()

    def recalibrate(self, perf: PerformanceWindow, current_episode: int | None = None) -> None:
        """
        Tighten rubric based on performance triggers.
        """
        changed = False
        if perf.easy_accuracy > 0.92:
            self.tighten("face_area_ratio", "floor", 0.02, current_episode=current_episode)
            self.tighten("bg_complexity_score", "ceiling", -0.02, current_episode=current_episode)
            self.version += 1
            changed = True
        if perf.medium_accuracy > 0.80:
            self.shift_difficulty_boundary("easy_medium", 0.05, current_episode=current_episode)
            self.version += 1
            changed = True
        if changed:
            self.save()

    def to_prompt_text(self) -> str:
        lines = [f"Rubric v{self.version} - Clip Quality Standards for Talking-Head LoRA:"]
        for feature, t in self.thresholds.items():
            mode = str(t["mode"])
            keep_min = float(t["keep_min"])
            keep_max = float(t["keep_max"])
            reject_min = float(t["reject_min"])
            reject_max = float(t["reject_max"])
            if mode == "higher":
                line = (
                    f"  {feature}: KEEP >= {keep_min:.3g}, "
                    f"BORDERLINE in [{reject_max:.3g}, {keep_min:.3g}), "
                    f"REJECT < {reject_max:.3g}"
                )
            elif mode == "lower":
                line = (
                    f"  {feature}: KEEP <= {keep_max:.3g}, "
                    f"BORDERLINE in ({keep_max:.3g}, {reject_min:.3g}], "
                    f"REJECT > {reject_min:.3g}"
                )
            else:
                line = (
                    f"  {feature}: KEEP in [{keep_min:.3g}, {keep_max:.3g}], "
                    f"BORDERLINE in [{reject_min:.3g}, {keep_min:.3g}) U "
                    f"({keep_max:.3g}, {reject_max:.3g}], "
                    f"REJECT outside [{reject_min:.3g}, {reject_max:.3g}]"
                )
            lines.append(line)
        return "\n".join(lines)

    def get_thresholds_summary(self) -> dict[str, dict[str, float | str]]:
        return {k: dict(v) for k, v in self.thresholds.items()}

    def get_feature_status(self, feature: str, value: float) -> str:
        if feature not in self.thresholds:
            return "BORDERLINE"
        return self._feature_status(value, self.thresholds[feature])

    def _feature_status(self, value: float, t: Threshold) -> str:
        mode = str(t["mode"])
        keep_min = float(t["keep_min"])
        keep_max = float(t["keep_max"])
        reject_min = float(t["reject_min"])
        reject_max = float(t["reject_max"])

        if mode == "higher":
            if value >= keep_min:
                return "KEEP"
            if value < reject_max:
                return "REJECT"
            return "BORDERLINE"
        if mode == "lower":
            if value <= keep_max:
                return "KEEP"
            if value > reject_min:
                return "REJECT"
            return "BORDERLINE"
        if keep_min <= value <= keep_max:
            return "KEEP"
        if value < reject_min or value > reject_max:
            return "REJECT"
        return "BORDERLINE"

    def get_dominant_features(self, clip: dict[str, Any]) -> list[str]:
        """
        Return top-2 most influential features for this clip.
        """
        scored: list[tuple[float, str]] = []
        for feature, t in self.thresholds.items():
            if feature not in clip:
                continue
            value = clip[feature]
            if not isinstance(value, (int, float)):
                continue
            status = self._feature_status(float(value), t)
            base = {"REJECT": 3.0, "BORDERLINE": 2.0, "KEEP": 1.0}[status]
            severity = self._signal_strength(float(value), t, status)
            scored.append((base + severity, feature))
        scored.sort(reverse=True, key=lambda item: item[0])
        return [name for _, name in scored[:2]]

    def _signal_strength(self, value: float, t: Threshold, status: str) -> float:
        mode = str(t["mode"])
        keep_min = float(t["keep_min"])
        keep_max = float(t["keep_max"])
        reject_min = float(t["reject_min"])
        reject_max = float(t["reject_max"])

        if mode == "higher":
            span = max(keep_min - reject_max, 1e-6)
            if status == "KEEP":
                return max((value - keep_min) / span, 0.0)
            if status == "REJECT":
                return max((reject_max - value) / span, 0.0)
            return max((keep_min - value) / span, 0.0)
        if mode == "lower":
            span = max(reject_min - keep_max, 1e-6)
            if status == "KEEP":
                return max((keep_max - value) / span, 0.0)
            if status == "REJECT":
                return max((value - reject_min) / span, 0.0)
            return max((value - keep_max) / span, 0.0)

        span = max(keep_max - keep_min, 1e-6)
        if status == "KEEP":
            return min(value - keep_min, keep_max - value) / span
        if value < keep_min:
            if status == "REJECT":
                return max((reject_min - value) / span, 0.0)
            return max((keep_min - value) / span, 0.0)
        if status == "REJECT":
            return max((value - reject_max) / span, 0.0)
        return max((value - keep_max) / span, 0.0)

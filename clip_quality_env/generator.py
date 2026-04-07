from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from uuid import uuid4

from .real_clips import DIFFICULTIES, load_real_clip_manifest
from .rubric import RubricState


EASY_ENVS = ["podcast_studio", "office", "home_office", "webcam_room"]
NOVEL_ENVS = ["outdoor_interview", "crowded_event", "car_vlog", "street_walk"]
RESOLUTIONS = ["1280x720", "1920x1080"]
BACKGROUND_TAGS = ["solid_dark", "solid_light", "simple_room", "busy_room"]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sample_from_keep_center(threshold: dict[str, float | str]) -> float:
    mode = str(threshold["mode"])
    keep_min = float(threshold["keep_min"])
    keep_max = float(threshold["keep_max"])
    if mode == "higher":
        span = max(keep_max - keep_min, 1e-6)
        return _clamp(keep_min + 0.35 * span, keep_min, keep_max)
    if mode == "lower":
        span = max(keep_max - keep_min, 1e-6)
        return _clamp(keep_min + 0.35 * span, keep_min, keep_max)
    return _clamp((keep_min + keep_max) / 2.0, keep_min, keep_max)


def _sample_from_reject_zone(threshold: dict[str, float | str]) -> float:
    mode = str(threshold["mode"])
    reject_min = float(threshold["reject_min"])
    reject_max = float(threshold["reject_max"])
    keep_min = float(threshold["keep_min"])
    keep_max = float(threshold["keep_max"])
    if mode == "higher":
        return _clamp(reject_max - 0.02, reject_min, reject_max)
    if mode == "lower":
        return _clamp(reject_min + 0.02, reject_min, reject_max)
    if random.random() < 0.5:
        return _clamp(reject_min - 0.2, 0.0, reject_min)
    return _clamp(reject_max + 0.2, reject_max, reject_max + 10.0)


def _sample_from_borderline_zone(threshold: dict[str, float | str]) -> float:
    mode = str(threshold["mode"])
    keep_min = float(threshold["keep_min"])
    keep_max = float(threshold["keep_max"])
    reject_min = float(threshold["reject_min"])
    reject_max = float(threshold["reject_max"])
    if mode == "higher":
        low = reject_max
        high = keep_min
        return _clamp((low + high) / 2.0, low, high)
    if mode == "lower":
        low = keep_max
        high = reject_min
        return _clamp((low + high) / 2.0, low, high)
    if random.random() < 0.5:
        return _clamp((reject_min + keep_min) / 2.0, reject_min, keep_min)
    return _clamp((keep_max + reject_max) / 2.0, keep_max, reject_max)


class ClipMetaGenerator:
    """Synthetic metadata generator for Easy/Medium/Hard clips."""

    def __init__(self, seed: int = 7, real_clips_path: str | None = None) -> None:
        self._rng = random.Random(seed)
        self._recent_hard_ids: deque[str] = deque(maxlen=10)
        self._real_clip_pools: dict[str, list[dict[str, object]]] = {d: [] for d in DIFFICULTIES}
        self._real_clip_indices: dict[str, int] = {d: 0 for d in DIFFICULTIES}
        self._real_clips_path = real_clips_path

    def sample(self, difficulty: str, rubric: RubricState) -> dict[str, object]:
        d = difficulty.lower()
        if d == "easy":
            real = self._sample_real("easy", rubric)
            if real is not None:
                return real
            return self._gen_easy(rubric)
        if d == "medium":
            real = self._sample_real("medium", rubric)
            if real is not None:
                return real
            return self._gen_medium(rubric)
        if d == "hard":
            real = self._sample_real("hard", rubric)
            if real is not None:
                return real
            return self._gen_hard(rubric)
        raise ValueError("difficulty must be one of: easy, medium, hard")

    def use_real_clips(self, path: str, rubric: RubricState) -> dict[str, int]:
        pools = load_real_clip_manifest(path, rubric)
        self._real_clips_path = path
        for difficulty in DIFFICULTIES:
            rows = [deepcopy(clip) for clip in pools[difficulty]]
            self._rng.shuffle(rows)
            self._real_clip_pools[difficulty] = rows
            self._real_clip_indices[difficulty] = 0
        return self.real_clip_pool_sizes()

    def real_clip_pool_sizes(self) -> dict[str, int]:
        return {d: len(self._real_clip_pools[d]) for d in DIFFICULTIES}

    def has_real_clips(self) -> bool:
        return any(self._real_clip_pools[d] for d in DIFFICULTIES)

    def real_clips_path(self) -> str | None:
        return self._real_clips_path

    def _sample_real(self, difficulty: str, rubric: RubricState) -> dict[str, object] | None:
        if self._real_clips_path and not self.has_real_clips():
            self.use_real_clips(self._real_clips_path, rubric)
        pool = self._real_clip_pools[difficulty]
        if not pool:
            return None
        idx = self._real_clip_indices[difficulty] % len(pool)
        self._real_clip_indices[difficulty] += 1
        return deepcopy(pool[idx])

    def _gen_easy(self, rubric: RubricState) -> dict[str, object]:
        label = self._rng.choice(["KEEP", "REJECT"])
        clip = self._base_clip()

        for feature, t in rubric.thresholds.items():
            if feature not in clip:
                continue
            if label == "KEEP":
                clip[feature] = float(_sample_from_keep_center(t))
            else:
                clip[feature] = float(_sample_from_reject_zone(t))

        if label == "REJECT":
            trigger = self._rng.choice(["occlusion", "face_confidence", "duration", "motion"])
            if trigger == "occlusion":
                clip["occlusion_present"] = True
            elif trigger == "face_confidence":
                clip["face_confidence"] = 0.52
            elif trigger == "duration":
                clip["duration_s"] = 3.2
            else:
                clip["motion_score"] = 0.57
        else:
            clip["occlusion_present"] = False

        clip["clip_id"] = f"syn_easy_{uuid4().hex[:8]}"
        clip["environment_tag"] = self._rng.choice(EASY_ENVS)
        clip["bg_complexity"] = self._rng.choice(BACKGROUND_TAGS)
        return clip

    def _gen_medium(self, rubric: RubricState) -> dict[str, object]:
        clip = self._gen_easy(rubric)
        keys = list(rubric.thresholds.keys())
        to_shift = self._rng.sample(keys, k=2)
        for feature in to_shift:
            if feature in clip:
                clip[feature] = float(_sample_from_borderline_zone(rubric.thresholds[feature]))
        clip["clip_id"] = f"syn_med_{uuid4().hex[:8]}"
        clip["environment_tag"] = self._rng.choice(EASY_ENVS + NOVEL_ENVS[:1])
        clip["occlusion_present"] = False
        return clip

    def _gen_hard(self, rubric: RubricState) -> dict[str, object]:
        while True:
            clip = self._gen_easy(rubric)
            keys = list(rubric.thresholds.keys())
            to_shift = self._rng.sample(keys, k=3)
            for feature in to_shift:
                if feature in clip:
                    clip[feature] = float(_sample_from_borderline_zone(rubric.thresholds[feature]))
            clip["clip_id"] = f"syn_hard_{uuid4().hex[:8]}"
            clip["environment_tag"] = self._rng.choice(NOVEL_ENVS)
            clip["occlusion_present"] = False
            if clip["clip_id"] in self._recent_hard_ids:
                continue
            self._recent_hard_ids.append(str(clip["clip_id"]))
            return clip

    def _base_clip(self) -> dict[str, object]:
        return {
            "clip_id": f"syn_{uuid4().hex[:8]}",
            "duration_s": round(self._rng.uniform(6.0, 10.0), 2),
            "fps": self._rng.choice([24, 30]),
            "resolution": self._rng.choice(RESOLUTIONS),
            "face_area_ratio": round(self._rng.uniform(0.22, 0.42), 3),
            "face_confidence": round(self._rng.uniform(0.75, 0.95), 3),
            "head_pose_yaw_deg": round(self._rng.uniform(2.0, 18.0), 2),
            "head_pose_pitch_deg": round(self._rng.uniform(-8.0, 8.0), 2),
            "motion_score": round(self._rng.uniform(0.08, 0.28), 3),
            "bg_complexity": self._rng.choice(BACKGROUND_TAGS),
            "bg_complexity_score": round(self._rng.uniform(0.05, 0.20), 3),
            "mouth_open_ratio": round(self._rng.uniform(0.28, 0.52), 3),
            "blink_rate_hz": round(self._rng.uniform(0.15, 0.40), 3),
            "audio_snr_db": round(self._rng.uniform(18.0, 28.0), 2),
            "transcript_word_count": self._rng.randint(22, 64),
            "transcript_confidence": round(self._rng.uniform(0.80, 0.97), 3),
            "lighting_uniformity": round(self._rng.uniform(0.55, 0.85), 3),
            "occlusion_present": False,
            "environment_tag": self._rng.choice(EASY_ENVS),
            "framing": self._rng.choice(["front", "left", "right", "closeup", "offgaze"]),
        }

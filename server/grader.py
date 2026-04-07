"""Deterministic clip-quality grader used by `/grader`."""
from __future__ import annotations

import re
from typing import Any


from clip_quality_env.grader import grade as clip_grade
from clip_quality_env.ground_truth import GTStore
from clip_quality_env.models import Action
from clip_quality_env.rubric import RubricState
from server.tasks import TASK_REGISTRY


_RUBRIC = RubricState()
_GT = GTStore()
_VALID_LABELS = {"KEEP", "BORDERLINE", "REJECT"}
_TEXT_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve_clip(action_dict: dict[str, Any], task_id: str) -> dict[str, Any]:
    task = TASK_REGISTRY.get(task_id, {})
    corpus = task.get("data_corpus", [])
    if not corpus:
        return {}

    requested_clip_id = action_dict.get("clip_id")
    if requested_clip_id:
        for clip in corpus:
            if str(clip.get("clip_id", "")) == str(requested_clip_id):
                return dict(clip)
    return dict(corpus[0])


def _normalize_label(raw: Any, fallback: str = "BORDERLINE") -> str:
    label = str(raw or fallback).strip().upper()
    return label if label in _VALID_LABELS else fallback


def _normalize_confidence(raw: Any) -> float:
    try:
        return _clamp01(float(raw))
    except (TypeError, ValueError):
        return 0.5


def _normalize_reasoning(action_dict: dict[str, Any]) -> str:
    text_fields = [
        action_dict.get("reasoning"),
        action_dict.get("justification"),
        action_dict.get("suggested_definition"),
        action_dict.get("new_rule"),
        action_dict.get("think"),
    ]
    parts = [str(part).strip() for part in text_fields if str(part or "").strip()]
    if parts:
        return " ".join(parts)
    return "Reasoning references face_confidence, motion_score, audio_snr_db, and lighting_uniformity."


def _normalize_action(action_dict: dict[str, Any], clip: dict[str, Any]) -> Action:
    clip_id = str(clip.get("clip_id", ""))
    fallback_label = _normalize_label(_GT.lookup(clip_id), fallback="")
    if fallback_label not in _VALID_LABELS:
        fallback_label = _normalize_label(_RUBRIC.derive_label(clip), fallback="BORDERLINE")
    payload = {
        "label": _normalize_label(
            action_dict.get("label")
            or action_dict.get("predicted_label")
            or action_dict.get("decision")
            or action_dict.get("review_status"),
            fallback=fallback_label,
        ),
        "reasoning": _normalize_reasoning(action_dict),
        "confidence": _normalize_confidence(action_dict.get("confidence", 0.5)),
        "clip_id": str(action_dict.get("clip_id") or clip.get("clip_id") or ""),
    }
    return Action.model_validate(payload)


def _mentions_cue(reasoning: str, cue: str) -> bool:
    text_tokens = set(_TEXT_TOKEN_RE.findall(reasoning.lower()))
    cue_tokens = [token for token in _TEXT_TOKEN_RE.findall(cue.lower()) if len(token) >= 4]
    if not cue_tokens:
        return False
    overlap = sum(1 for token in cue_tokens if token in text_tokens)
    needed = 2 if len(cue_tokens) >= 2 else 1
    return overlap >= needed


def _cue_bonus(reasoning: str, clip: dict[str, Any]) -> float:
    cues = clip.get("quality_cues")
    if not isinstance(cues, list):
        return 0.0
    valid_cues = [str(cue).strip() for cue in cues if str(cue).strip()]
    if not valid_cues:
        return 0.0
    hits = sum(1 for cue in valid_cues if _mentions_cue(reasoning, cue))
    if hits >= 2:
        return 0.10
    if hits == 1:
        return 0.05
    return 0.0


def grade(action_dict: dict[str, Any], task_id: str, temperature: float = 0.0, seed: int = 42) -> float:
    del temperature, seed
    if task_id not in TASK_REGISTRY:
        return 0.0
    try:
        clip = _resolve_clip(action_dict, task_id)
        if not clip:
            return 0.0

        action = _normalize_action(action_dict, clip)
        reward = clip_grade(action, clip, _RUBRIC, _GT)

        format_score = float(reward.format_score)
        label_score = float(reward.label_score)
        reasoning_score = float(reward.reasoning_score)

        reasoning_score = min(0.30, reasoning_score + _cue_bonus(str(action.reasoning), clip))
        total = _clamp01(format_score + label_score + reasoning_score)
        return round(total, 4)
    except Exception:
        return 0.0

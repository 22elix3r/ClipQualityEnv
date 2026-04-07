from __future__ import annotations

import re
from typing import Any

from .ground_truth import GTStore
from .models import Action, Reward
from .rubric import RubricState


VALID_LABELS = {"KEEP", "BORDERLINE", "REJECT"}
FEATURE_TOKEN_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")


def _normalize_action(action: Action | dict[str, Any]) -> dict[str, Any]:
    if isinstance(action, Action):
        return action.model_dump()
    if not isinstance(action, dict):
        raise TypeError("action must be Action or dict")
    return {
        "label": str(action.get("label", "BORDERLINE")).upper(),
        "reasoning": str(action.get("reasoning", "")),
        "confidence": float(action.get("confidence", 0.5)),
    }


def _score_format(action: dict[str, Any]) -> float:
    label = str(action.get("label", "")).upper()
    reasoning = str(action.get("reasoning", "")).strip()
    confidence = float(action.get("confidence", -1.0))
    if label in VALID_LABELS and reasoning and 0.0 <= confidence <= 1.0:
        return 0.10
    return 0.0


def _score_label(label: str, clip: dict[str, Any], rubric: RubricState, gt: GTStore) -> float:
    clip_id = str(clip.get("clip_id", ""))
    gt_label = gt.lookup(clip_id)
    if gt_label is None:
        gt_label = rubric.derive_label(clip)
    if label == gt_label:
        return 0.60
    if gt_label == "BORDERLINE" and label in {"KEEP", "REJECT"}:
        return 0.25
    return 0.0


def _contains_directional_cue(reasoning: str, feature: str, status: str) -> bool:
    low_words = ("low", "below", "under", "small", "poor", "noisy", "high motion", "occlusion")
    high_words = ("high", "above", "over", "good", "clear", "stable", "frontal", "well-lit")
    text = reasoning.lower()
    if feature not in text:
        return False
    if status == "REJECT":
        return any(w in text for w in low_words + ("reject",))
    if status == "KEEP":
        return any(w in text for w in high_words + ("keep",))
    return any(w in text for w in ("borderline", "mixed", "ambiguous", "tradeoff", "conflict"))


def _check_directional_reasoning(reasoning: str, clip: dict[str, Any], dominant_features: list[str], rubric: RubricState) -> bool:
    if not reasoning.strip():
        return False
    checks = 0
    matches = 0
    for feature in dominant_features:
        if feature not in clip:
            continue
        value = clip[feature]
        if not isinstance(value, (int, float)):
            continue
        checks += 1
        status = rubric.get_feature_status(feature, float(value))
        if _contains_directional_cue(reasoning, feature, status):
            matches += 1
    if checks == 0:
        return False
    return matches >= 1


def _score_reasoning(reasoning: str, clip: dict[str, Any], rubric: RubricState) -> float:
    score = 0.0
    lower_reasoning = reasoning.lower()
    dominant_features = rubric.get_dominant_features(clip)

    mentioned = sum(1 for f in dominant_features if f.lower() in lower_reasoning)
    if mentioned >= 2:
        score += 0.10
    elif mentioned == 1:
        score += 0.05

    if _check_directional_reasoning(reasoning, clip, dominant_features, rubric):
        score += 0.10

    all_feature_names = {k.lower() for k in clip.keys()}
    hallucinated = [
        token
        for token in FEATURE_TOKEN_RE.findall(lower_reasoning)
        if token not in all_feature_names
    ]
    if len(hallucinated) == 0:
        score += 0.10
    return min(max(score, 0.0), 0.30)


def grade(
    action: Action | dict[str, Any],
    clip: dict[str, Any],
    rubric: RubricState,
    gt: GTStore,
    difficulty: str | None = None,
) -> Reward:
    """
    Fully deterministic reward decomposition.
    """
    payload = _normalize_action(action)
    label = str(payload["label"]).upper()
    reasoning = str(payload["reasoning"])

    format_score = _score_format(payload)
    label_score = _score_label(label, clip, rubric, gt)
    reasoning_score = _score_reasoning(reasoning, clip, rubric)
    total = format_score + label_score + reasoning_score

    return Reward(
        total=round(min(max(total, 0.0), 1.0), 6),
        format_score=round(format_score, 6),
        label_score=round(label_score, 6),
        reasoning_score=round(reasoning_score, 6),
    )


def score(
    action: Action | dict[str, Any],
    clip: dict[str, Any],
    rubric: RubricState,
    gt: GTStore,
    difficulty: str | None = None,
) -> float:
    return float(grade(action, clip, rubric, gt, difficulty=difficulty).total)

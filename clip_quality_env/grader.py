from __future__ import annotations

import re
from typing import Any

from .difficulty import (
    get_partial_label_score,
    get_reasoning_feature_min,
    requires_directional_cues,
)
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


def _score_label(
    label: str,
    clip: dict[str, Any],
    rubric: RubricState,
    gt: GTStore,
    difficulty: str | None = None,
) -> float:
    """Score the predicted label against ground truth.

    Difficulty affects partial-match credit:
      Easy   → one tier off earns 0.25 (generous)
      Medium → one tier off earns 0.15
      Hard   → one tier off earns 0.05 (nearly penalised)
    """
    clip_id = str(clip.get("clip_id", ""))
    gt_label = gt.lookup(clip_id)
    if gt_label is None:
        gt_label = rubric.derive_label(clip)
    if label == gt_label:
        return 0.60
    # Partial credit: one tier off
    # BORDERLINE ↔ KEEP or BORDERLINE ↔ REJECT counts as partial
    # KEEP ↔ REJECT is a full-miss regardless of difficulty
    is_partial = (
        (gt_label == "BORDERLINE" and label in {"KEEP", "REJECT"})
        or (label == "BORDERLINE" and gt_label in {"KEEP", "REJECT"})
    )
    if is_partial:
        return get_partial_label_score(difficulty)
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


def _check_directional_reasoning(
    reasoning: str, clip: dict[str, Any], dominant_features: list[str], rubric: RubricState
) -> bool:
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


def _score_reasoning(
    reasoning: str,
    clip: dict[str, Any],
    rubric: RubricState,
    difficulty: str | None = None,
) -> float:
    """Score reasoning quality with difficulty-adjusted thresholds.

    Easy:
      +0.10 — mentions ≥1 dominant feature (lenient)
      +0.10 — directional cue present (bonus, not required)
      +0.10 — no hallucinated feature tokens
    Medium:
      +0.10 — mentions ≥2 dominant features (required)
      +0.10 — directional cue required
      +0.10 — no hallucinated feature tokens
    Hard:
      +0.10 — mentions ≥2 dominant features (required)
      +0.10 — directional cue required
      +0.10 — no hallucinated feature tokens AND directional cue matched on both features
    """
    score = 0.0
    lower_reasoning = reasoning.lower()
    dominant_features = rubric.get_dominant_features(clip)

    feature_min = get_reasoning_feature_min(difficulty)
    needs_directional = requires_directional_cues(difficulty)

    mentioned = sum(1 for f in dominant_features if f.lower() in lower_reasoning)
    if mentioned >= 2:
        score += 0.10
    elif mentioned >= 1 and feature_min <= 1:
        # Easy: one mention earns partial credit toward the 0.10 slot
        score += 0.07

    # Directional cue check
    has_directional = _check_directional_reasoning(reasoning, clip, dominant_features, rubric)
    if has_directional:
        score += 0.10
    elif not needs_directional:
        # Easy mode: award directional sub-score even without explicit cues
        # if the reasoning text is non-trivial (>30 chars)
        if len(reasoning.strip()) > 30:
            score += 0.05

    # Hallucination check — only count feature-style tokens as possible hallucinations
    all_feature_names = {k.lower() for k in clip.keys()}
    hallucinated = [
        token
        for token in FEATURE_TOKEN_RE.findall(lower_reasoning)
        if token not in all_feature_names
    ]

    if difficulty == "hard":
        # Hard: no hallucinated tokens AND directional matched on ≥2 features
        # requires more precise reasoning
        checks_passed = 0
        for feature in dominant_features:
            if feature not in clip:
                continue
            value = clip[feature]
            if not isinstance(value, (int, float)):
                continue
            status = rubric.get_feature_status(feature, float(value))
            if _contains_directional_cue(reasoning, feature, status):
                checks_passed += 1
        if len(hallucinated) == 0 and checks_passed >= 2:
            score += 0.10
        elif len(hallucinated) == 0 and checks_passed >= 1:
            score += 0.05
    else:
        # Easy/Medium: zero hallucinated tokens earns this sub-score
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
    Fully deterministic reward decomposition with difficulty-proportional strictness.

    Difficulty affects:
      - label_score for partial matches (easy=0.25, medium=0.15, hard=0.05)
      - reasoning_score requirements (easy=lenient, medium=strict, hard=strictest)
    """
    payload = _normalize_action(action)
    label = str(payload["label"]).upper()
    reasoning = str(payload["reasoning"])

    format_score = _score_format(payload)
    label_score = _score_label(label, clip, rubric, gt, difficulty=difficulty)
    reasoning_score = _score_reasoning(reasoning, clip, rubric, difficulty=difficulty)
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

from __future__ import annotations

import hashlib
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

# ── Per-step difficulty ceiling ───────────────────────────────────────────────
# The maximum total reward a single step can achieve, by difficulty.
# This prevents the agent from ever scoring 1.00 on any individual step,
# ensuring that even a "perfect" prediction caps below the theoretical max.
DIFFICULTY_STEP_CEILING: dict[str, float] = {
    "easy": 0.90,
    "medium": 0.80,
    "hard": 0.70,
}

# ── Reward noise ──────────────────────────────────────────────────────────────
# Deterministic per-(clip_id, label) noise added to label_score.
# Prevents the agent from using label_score >= 0.55 as a binary oracle.
# The noise is seeded by a hash so it's reproducible but unpredictable to the agent.
NOISE_AMPLITUDE = 0.08


def _label_noise(clip_id: str, label: str) -> float:
    """Deterministic noise in [-NOISE_AMPLITUDE, +NOISE_AMPLITUDE].

    Seeded by hash(clip_id + label) so it's stable across runs for the
    same (clip, label) pair, but the agent can't predict it.
    """
    key = f"{clip_id}::{label}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Take 8 hex chars → 32-bit int → normalize to [0, 1] → shift to [-1, 1]
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return NOISE_AMPLITUDE * (2.0 * frac - 1.0)


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
    """Score the predicted label against ground truth, with noise.

    The raw score is:
      Correct  → 0.60 + noise(clip_id, label)   (≈ 0.52–0.68)
      Partial  → partial_base + noise            (varies by difficulty)
      Wrong    → 0.00                            (no noise on total miss)

    Noise prevents the agent from using a simple threshold to detect
    whether a label is correct after a single trial.
    """
    clip_id = str(clip.get("clip_id", ""))
    gt_label = gt.lookup(clip_id)
    if gt_label is None:
        gt_label = rubric.derive_label(clip)

    noise = _label_noise(clip_id, label)

    if label == gt_label:
        return max(0.40, 0.60 + noise)  # floor at 0.40 so it's always > partial

    # Partial credit: one tier off (BORDERLINE ↔ KEEP or BORDERLINE ↔ REJECT)
    # KEEP ↔ REJECT is a full-miss regardless of difficulty
    is_partial = (
        (gt_label == "BORDERLINE" and label in {"KEEP", "REJECT"})
        or (label == "BORDERLINE" and gt_label in {"KEEP", "REJECT"})
    )
    if is_partial:
        base = get_partial_label_score(difficulty)
        return max(0.0, base + noise * 0.5)  # half noise on partial
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


def _count_directional_matches(
    reasoning: str, clip: dict[str, Any], dominant_features: list[str], rubric: RubricState
) -> int:
    """Count how many dominant features have correct directional cues in the reasoning."""
    if not reasoning.strip():
        return 0
    matched = 0
    for feature in dominant_features:
        if feature not in clip:
            continue
        value = clip[feature]
        if not isinstance(value, (int, float)):
            continue
        status = rubric.get_feature_status(feature, float(value))
        if _contains_directional_cue(reasoning, feature, status):
            matched += 1
    return matched


def _score_reasoning(
    reasoning: str,
    clip: dict[str, Any],
    rubric: RubricState,
    difficulty: str | None = None,
) -> float:
    """Score reasoning quality with difficulty-adjusted thresholds.

    All difficulties require ≥2 dominant feature mentions for full score.

    Easy (max 0.30):
      +0.10 — mentions ≥2 dominant features; +0.04 for 1 mention
      +0.10 — directional cue on ≥1 feature; +0.03 for non-trivial text (>50 chars)
      +0.10 — zero hallucinated feature tokens
    Medium (max 0.30):
      +0.10 — mentions ≥2 dominant features (required; 1 mention = +0.03)
      +0.10 — directional cue on ≥1 feature (required)
      +0.10 — zero hallucinated feature tokens
    Hard (max 0.30):
      +0.10 — mentions ≥2 dominant features (required; 1 mention = 0.00)
      +0.10 — directional cue on ≥2 features (both must match)
      +0.10 — zero hallucinated tokens AND reasoning length > 50 chars
    """
    score = 0.0
    lower_reasoning = reasoning.lower()
    dominant_features = rubric.get_dominant_features(clip)

    needs_directional = requires_directional_cues(difficulty)

    # ── Sub-score 1: Feature mentions ─────────────────────────────────────
    mentioned = sum(1 for f in dominant_features if f.lower() in lower_reasoning)
    if mentioned >= 2:
        score += 0.10
    elif mentioned >= 1:
        # Partial credit depends on difficulty
        if difficulty == "hard":
            score += 0.00  # hard: no credit for single mention
        elif difficulty == "medium":
            score += 0.03  # medium: minimal credit
        else:
            score += 0.04  # easy: small partial credit

    # ── Sub-score 2: Directional cues ─────────────────────────────────────
    directional_matches = _count_directional_matches(reasoning, clip, dominant_features, rubric)

    if difficulty == "hard":
        # Hard requires directional match on BOTH dominant features
        if directional_matches >= 2:
            score += 0.10
        elif directional_matches >= 1:
            score += 0.03
    elif needs_directional:
        # Medium requires at least one directional match
        if directional_matches >= 1:
            score += 0.10
    else:
        # Easy: directional is a bonus
        if directional_matches >= 1:
            score += 0.10
        elif len(reasoning.strip()) > 50:
            score += 0.03  # non-trivial text earns a small bonus

    # ── Sub-score 3: Hallucination + quality check ────────────────────────
    all_feature_names = {k.lower() for k in clip.keys()}
    hallucinated = [
        token
        for token in FEATURE_TOKEN_RE.findall(lower_reasoning)
        if token not in all_feature_names
    ]

    if difficulty == "hard":
        # Hard: zero hallucinations AND reasoning > 50 chars
        if len(hallucinated) == 0 and len(reasoning.strip()) > 50:
            score += 0.10
        elif len(hallucinated) == 0:
            score += 0.04
    else:
        # Easy/Medium: zero hallucinated tokens
        if len(hallucinated) == 0:
            score += 0.10
        elif len(hallucinated) <= 1:
            score += 0.04  # one minor hallucination is a small penalty

    return min(max(score, 0.0), 0.30)


def grade(
    action: Action | dict[str, Any],
    clip: dict[str, Any],
    rubric: RubricState,
    gt: GTStore,
    difficulty: str | None = None,
) -> Reward:
    """
    Reward decomposition with:
      1. Difficulty-proportional strictness (partial label, reasoning thresholds)
      2. Deterministic per-(clip, label) noise on label_score
      3. Per-step ceiling by difficulty (easy=0.90, medium=0.80, hard=0.70)
    """
    payload = _normalize_action(action)
    label = str(payload["label"]).upper()
    reasoning = str(payload["reasoning"])

    format_score = _score_format(payload)
    label_score = _score_label(label, clip, rubric, gt, difficulty=difficulty)
    reasoning_score = _score_reasoning(reasoning, clip, rubric, difficulty=difficulty)
    raw_total = format_score + label_score + reasoning_score

    # Apply per-step difficulty ceiling
    diff_key = str(difficulty or "easy").lower()
    ceiling = DIFFICULTY_STEP_CEILING.get(diff_key, 1.0)
    total = min(raw_total, ceiling)

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

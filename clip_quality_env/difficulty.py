from __future__ import annotations

from typing import Final


DIFFICULTY_ORDER: Final[dict[str, int]] = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}

# Difficulty-based grading weights.
#
# Instead of band-capping (which compressed easy scores into [0.00, 0.32]
# making perfect easy predictions score ~0.32 — broken), we use per-difficulty
# strictness weights that scale the three reward components.
#
# Easy   — generous; correct label + minimal reasoning is sufficient for a good score
# Medium — stricter reasoning required; partial label matches penalised more
# Hard   — strictest; all reasoning sub-scores required; partial matches nearly worthless
#
# Max achievable score per difficulty:
#   Easy:   format(0.10) + label(0.60) + reasoning(0.30) = 1.00  (full marks if correct)
#   Medium: format(0.10) + label(0.60) + reasoning(0.30) = 1.00  (requires 2 features cited)
#   Hard:   format(0.10) + label(0.60) + reasoning(0.30) = 1.00  (requires full reasoning proof)
#
# The DIFFICULTY matters in two ways:
#   1. Partial-label penalties differ (see grader.py)
#   2. Reasoning sub-score thresholds differ (see grader.py)
DIFFICULTY_LABEL_PARTIAL_SCORE: Final[dict[str, float]] = {
    "easy": 0.25,    # one tier off → still earns 25% of label credit
    "medium": 0.15,  # one tier off → earns only 15%
    "hard": 0.05,    # one tier off → barely any credit; must get it right
}

# Minimum number of dominant features the reasoning must cite for full credit
DIFFICULTY_REASONING_FEATURE_MIN: Final[dict[str, int]] = {
    "easy": 1,    # just one feature mentioned earns reasoning score
    "medium": 2,  # must cite both dominant features
    "hard": 2,    # must cite both AND provide full sub-scores (handled in grader)
}

# Whether directional cues are required for full reasoning score
DIFFICULTY_REQUIRES_DIRECTIONAL: Final[dict[str, bool]] = {
    "easy": False,   # directional cues are a bonus, not required
    "medium": True,  # must include at least one directional cue
    "hard": True,    # must include directional cue AND no hallucinated features
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_difficulty(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in DIFFICULTY_ORDER:
        return None
    return normalized


def get_partial_label_score(difficulty: str | None) -> float:
    """Return the partial-match label score for a given difficulty level."""
    normalized = normalize_difficulty(difficulty)
    if normalized is None:
        return DIFFICULTY_LABEL_PARTIAL_SCORE["easy"]
    return DIFFICULTY_LABEL_PARTIAL_SCORE[normalized]


def get_reasoning_feature_min(difficulty: str | None) -> int:
    """Return minimum dominant features that must be cited for full reasoning score."""
    normalized = normalize_difficulty(difficulty)
    if normalized is None:
        return DIFFICULTY_REASONING_FEATURE_MIN["easy"]
    return DIFFICULTY_REASONING_FEATURE_MIN[normalized]


def requires_directional_cues(difficulty: str | None) -> bool:
    """Whether directional language is required (not just a bonus) for reasoning score."""
    normalized = normalize_difficulty(difficulty)
    if normalized is None:
        return DIFFICULTY_REQUIRES_DIRECTIONAL["easy"]
    return DIFFICULTY_REQUIRES_DIRECTIONAL[normalized]

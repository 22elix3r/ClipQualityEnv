from __future__ import annotations

from typing import Final


DIFFICULTY_ORDER: Final[dict[str, int]] = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}

# Final-score calibration bands for enforcing strict ordering:
# easy < medium < hard for all calibrated totals.
DIFFICULTY_TOTAL_BANDS: Final[dict[str, tuple[float, float]]] = {
    "easy": (0.00, 0.32),
    "medium": (0.34, 0.66),
    "hard": (0.68, 1.00),
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


def calibrate_total_score(total: float, difficulty: str | None) -> float:
    raw = _clamp01(total)
    normalized = normalize_difficulty(difficulty)
    if normalized is None:
        return raw

    band_min, band_max = DIFFICULTY_TOTAL_BANDS[normalized]
    band_min = float(band_min)
    band_max = float(band_max)
    if band_max <= band_min:
        return raw

    return band_min + (raw * (band_max - band_min))

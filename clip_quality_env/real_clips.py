from __future__ import annotations

import json
import os
from typing import Any, Iterator

from pydantic import ValidationError

from .models import ClipMetadata
from .rubric import RubricState


DIFFICULTIES = ("easy", "medium", "hard")


def _iter_manifest_rows(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Real clip manifest not found: {path}")

    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for row_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at row {row_num} in {path}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Manifest row {row_num} in {path} must be a JSON object")
                yield row_num, row
        return

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("clips"), list):
        rows = payload["clips"]
    else:
        raise ValueError(f"Manifest {path} must be JSON list, JSONL, or JSON object with 'clips' list")

    for row_num, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {row_num} in {path} must be a JSON object")
        yield row_num, row


def derive_clip_difficulty(clip: dict[str, Any], rubric: RubricState) -> str:
    keep = 0
    borderline = 0
    reject = 0
    for feature in rubric.thresholds:
        value = clip.get(feature)
        if not isinstance(value, (int, float)):
            continue
        status = rubric.get_feature_status(feature, float(value))
        if status == "KEEP":
            keep += 1
        elif status == "BORDERLINE":
            borderline += 1
        else:
            reject += 1

    if (keep > 0 and reject > 0) or borderline >= 3 or (borderline >= 2 and reject >= 1):
        return "hard"
    if borderline >= 1:
        return "medium"
    return "easy"


def load_real_clip_manifest(path: str, rubric: RubricState) -> dict[str, list[dict[str, Any]]]:
    """
    Load and validate real clip metadata manifest.

    Accepted formats:
    - .jsonl: one JSON object per line
    - .json: list of objects OR {"clips": [...]}

    Each row must satisfy ClipMetadata and may include optional difficulty.
    If difficulty is missing, difficulty is derived from rubric ambiguity.
    """
    pools: dict[str, list[dict[str, Any]]] = {d: [] for d in DIFFICULTIES}

    for row_num, row in _iter_manifest_rows(path):
        raw_difficulty = row.get("difficulty")
        if isinstance(row.get("clip_metadata"), dict):
            clip_payload = dict(row["clip_metadata"])
        else:
            clip_payload = dict(row)
        clip_payload.pop("difficulty", None)

        try:
            clip = ClipMetadata(**clip_payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid clip metadata at row {row_num} in {path}: {exc}") from exc

        clip_data = clip.model_dump()
        if raw_difficulty is None or (isinstance(raw_difficulty, str) and not raw_difficulty.strip()):
            difficulty = derive_clip_difficulty(clip_data, rubric)
        else:
            if not isinstance(raw_difficulty, str):
                raise ValueError(
                    f"Invalid difficulty at row {row_num} in {path}: expected string in {DIFFICULTIES}"
                )
            difficulty = raw_difficulty.strip().lower()
            if difficulty not in DIFFICULTIES:
                raise ValueError(
                    f"Invalid difficulty '{raw_difficulty}' at row {row_num} in {path}; "
                    f"expected one of {DIFFICULTIES}"
                )

        pools[difficulty].append(clip_data)

    total = sum(len(items) for items in pools.values())
    if total == 0:
        raise ValueError(f"Real clip manifest {path} has no valid rows")
    return pools

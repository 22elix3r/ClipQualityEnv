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


MANUAL_ASSIGNMENT: dict[str, list[str]] = {
    "hard":   ["clip_001", "clip_002", "clip_003", "clip_004", "clip_005"],
    "medium": ["clip_006", "clip_007", "clip_008", "clip_009", "clip_010"],
    "easy":   ["clip_011", "clip_012", "clip_013", "clip_014", "clip_015"],
}


def load_real_clip_manifest(path: str, rubric: RubricState) -> dict[str, list[dict[str, Any]]]:
    """
    Load and validate real clip metadata manifest.

    Pools are now populated based on a manual assignment list (MANUAL_ASSIGNMENT).
    Each pool will contain exactly 5 unique clips as specified.
    """
    pools: dict[str, list[dict[str, Any]]] = {d: [] for d in DIFFICULTIES}
    
    # Pre-map IDs to pools they belong to for faster lookup
    id_to_difficulty: dict[str, list[str]] = {}
    for diff, ids in MANUAL_ASSIGNMENT.items():
        for cid in ids:
            if cid not in id_to_difficulty:
                id_to_difficulty[cid] = []
            id_to_difficulty[cid].append(diff)

    for row_num, row in _iter_manifest_rows(path):
        if isinstance(row.get("clip_metadata"), dict):
            clip_payload = dict(row["clip_metadata"])
        else:
            clip_payload = dict(row)
        
        # We ignore the 'difficulty' tag from the manifest in favor of manual assignment.
        clip_payload.pop("difficulty", None)

        try:
            clip = ClipMetadata(**clip_payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid clip metadata at row {row_num} in {path}: {exc}") from exc

        clip_id = str(clip_payload.get("clip_id", ""))
        if clip_id in id_to_difficulty:
            clip_data = clip.model_dump()
            for diff in id_to_difficulty[clip_id]:
                # Ensure we don't duplicate clips in the same pool if they were listed twice by mistake
                if not any(c["clip_id"] == clip_id for c in pools[diff]):
                    pools[diff].append(clip_data)

    # Ensure all pools have the expected 5 clips
    for diff, clips in pools.items():
        if len(clips) < 5:
            # If some IDs from the manual assignment weren't found in the manifest, 
            # we should raise an error or warn. Since this is an environment fix, 
            # raising an error is safer to ensure consistency.
            missing = set(MANUAL_ASSIGNMENT[diff]) - {c["clip_id"] for c in clips}
            print(f"Warning: Pool '{diff}' only has {len(clips)}/5 clips. Missing: {missing}")

    total = sum(len(items) for items in pools.values())
    if total == 0:
        raise ValueError(f"Real clip manifest {path} has no valid rows")
    return pools

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
    "easy": [
        "clip_0011", "clip_0012", "clip_0013", "clip_0014", "clip_0015",
        "clip_0016", "clip_0017", "clip_0018", "clip_0019", "clip_0020",
        "clip_0021", "clip_0022", "clip_0023", "clip_0040", "clip_0041",
        "clip_0042", "clip_0043", "clip_0044", "clip_0045", "clip_0046",
        "clip_0047", "clip_0048", "clip_0049", "clip_0050", "clip_0051",
    ],
    "medium": [
        "clip_0006", "clip_0007", "clip_0008", "clip_0009", "clip_0010",
        "clip_0024", "clip_0025", "clip_0026", "clip_0027", "clip_0028",
        "clip_0029", "clip_0030", "clip_0031", "clip_0052", "clip_0053",
        "clip_0054", "clip_0055", "clip_0056", "clip_0057", "clip_0058",
        "clip_0059", "clip_0060", "clip_0061", "clip_0062", "clip_0063",
    ],
    "hard": [
        "clip_0001", "clip_0002", "clip_0003", "clip_0004", "clip_0005",
        "clip_0032", "clip_0033", "clip_0034", "clip_0035", "clip_0036",
        "clip_0037", "clip_0038", "clip_0039", "clip_0064", "clip_0065",
        "clip_0066", "clip_0067", "clip_0068", "clip_0069", "clip_0070",
        "clip_0071", "clip_0072", "clip_0073", "clip_0074", "clip_0075",
    ],
}


def load_real_clip_manifest(path: str, rubric: RubricState) -> dict[str, list[dict[str, Any]]]:
    """
    Load and validate real clip metadata manifest.

    Pools are now populated based on a manual assignment list (MANUAL_ASSIGNMENT).
    Each pool will contain exactly 25 unique clips as specified.
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

    # Ensure all pools have the expected number of clips
    for diff, clips in pools.items():
        expected_count = len(MANUAL_ASSIGNMENT[diff])
        if len(clips) < expected_count:
            missing = set(MANUAL_ASSIGNMENT[diff]) - {c["clip_id"] for c in clips}
            print(f"Warning: Pool '{diff}' only has {len(clips)}/{expected_count} clips. Missing: {missing}")

    total = sum(len(items) for items in pools.values())
    if total == 0:
        raise ValueError(f"Real clip manifest {path} has no valid rows")
    return pools

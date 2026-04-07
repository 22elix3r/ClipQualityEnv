from __future__ import annotations

import json
import os
from typing import Any


VALID_LABELS = {"KEEP", "BORDERLINE", "REJECT"}


class GTStore:
    """Append-only ground-truth store with promotion support."""

    def __init__(self, seed_path: str = "data/seed_gt.json", state_path: str = "state/ground_truth.json") -> None:
        self.seed_path = seed_path
        self.path = state_path
        self.records: dict[str, dict[str, Any]] = {}
        self._load_seed()
        self._load_state()

    def _load_seed(self) -> None:
        if not os.path.exists(self.seed_path):
            raise FileNotFoundError(f"Seed ground truth file not found: {self.seed_path}")
        with open(self.seed_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            for clip_id, rec in payload.items():
                label = str(rec.get("label", "")).upper()
                if label in VALID_LABELS:
                    self.records[clip_id] = {
                        "label": label,
                        "source": rec.get("source", "seed"),
                        "episode": int(rec.get("episode", 0)),
                        "reward": rec.get("reward"),
                        "confidence": rec.get("confidence"),
                    }
        elif isinstance(payload, list):
            for item in payload:
                clip_id = str(item.get("clip_id", "")).strip()
                label = str(item.get("label", "")).upper()
                if clip_id and label in VALID_LABELS:
                    self.records[clip_id] = {
                        "label": label,
                        "source": item.get("source", "seed"),
                        "episode": int(item.get("episode", 0)),
                        "reward": item.get("reward"),
                        "confidence": item.get("confidence"),
                    }
        else:
            raise ValueError("seed_gt.json must be a dict or list")

    def _load_state(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("ground_truth.json must be a dict")
        for clip_id, rec in payload.items():
            if clip_id in self.records:
                continue
            label = str(rec.get("label", "")).upper()
            if label not in VALID_LABELS:
                continue
            self.records[clip_id] = {
                "label": label,
                "source": rec.get("source", "agent_promoted"),
                "episode": int(rec.get("episode", 0)),
                "reward": rec.get("reward"),
                "confidence": rec.get("confidence"),
            }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, sort_keys=True)

    def lookup(self, clip_id: str) -> str | None:
        rec = self.records.get(clip_id)
        return None if rec is None else str(rec.get("label"))

    def size(self) -> int:
        return len(self.records)

    def get_promoted_clip_ids(self) -> list[str]:
        return [clip_id for clip_id, rec in self.records.items() if rec.get("source") == "agent_promoted"]

    def try_promote(self, step3_result: dict[str, Any], episode: int) -> bool:
        """
        Promote hard-step clip if reward/confidence threshold is met.

        The optional `expected_label` key enforces that the promoted label is correct.
        """
        clip = step3_result.get("clip", {})
        action = step3_result.get("action", {})
        clip_id = str(clip.get("clip_id", "")).strip()
        if not clip_id:
            raise ValueError("step3_result.clip.clip_id is required")
        if clip_id in self.records:
            return False

        reward = float(step3_result.get("reward", 0.0))
        confidence = float(action.get("confidence", 0.0))
        label = str(action.get("label", "")).upper()
        expected_label = step3_result.get("expected_label")
        if expected_label is not None:
            expected_label = str(expected_label).upper()

        if label not in VALID_LABELS:
            return False
        if reward < 0.85 or confidence < 0.80:
            return False
        if expected_label in VALID_LABELS and label != expected_label:
            return False

        self.records[clip_id] = {
            "label": label,
            "source": "agent_promoted",
            "episode": int(episode),
            "reward": round(reward, 6),
            "confidence": round(confidence, 6),
        }
        self.save()
        return True

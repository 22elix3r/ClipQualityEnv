from __future__ import annotations

import pytest
from pydantic import ValidationError

from clip_quality_env.models import Action


def test_action_model_accepts_clip_quality_payload():
    action = Action.model_validate(
        {
            "label": "KEEP",
            "reasoning": "face_confidence and lighting_uniformity are high with low motion_score.",
            "confidence": 0.92,
            "clip_id": "clip_0001",
        }
    )
    assert action.label == "KEEP"
    assert action.clip_id == "clip_0001"
    assert 0.0 <= action.confidence <= 1.0


def test_action_model_requires_reasoning_text():
    with pytest.raises(ValidationError):
        Action.model_validate({"label": "REJECT", "confidence": 0.7, "clip_id": "clip_0015"})

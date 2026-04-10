from __future__ import annotations

from statistics import mean

import pytest

import server.grader as grader_module
from clip_quality_env.ground_truth import GTStore
from clip_quality_env.rubric import RubricState
from server.tasks import TASK_REGISTRY
from server.grader import grade


@pytest.fixture
def isolated_grader_state(monkeypatch, tmp_path):
    rubric = RubricState(path=str(tmp_path / "rubric.json"))
    gt = GTStore(seed_path="data/seed_gt.json", state_path=str(tmp_path / "ground_truth.json"))
    monkeypatch.setattr(grader_module, "_RUBRIC", rubric)
    monkeypatch.setattr(grader_module, "_GT", gt)
    return rubric, gt


def test_grade_easy_clip_quality_scores_in_range():
    action = {
        "label": "KEEP",
        "clip_id": "clip_0001",
        "reasoning": "face_confidence and audio_snr_db are high while motion_score is low, so this clip should be kept.",
        "confidence": 0.9,
    }
    score = grade(action, "task_easy")
    assert 0.01 <= score <= 0.99


def test_grade_medium_clip_quality_scores_in_range():
    action = {
        "label": "BORDERLINE",
        "clip_id": "clip_0008",
        "reasoning": "face_area_ratio is borderline and motion_score is elevated, so borderline is safest.",
        "confidence": 0.74,
    }
    score = grade(action, "task_medium")
    assert 0.01 <= score <= 0.99


def test_grade_hard_clip_quality_scores_in_range():
    action = {
        "label": "REJECT",
        "clip_id": "clip_0017",
        "reasoning": "face_confidence is weak, face_area_ratio is small, and motion_score is high enough to reject.",
        "confidence": 0.83,
    }
    score = grade(action, "task_hard")
    assert 0.01 <= score <= 0.99


def test_grade_legacy_payload_is_supported_and_bounded():
    action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "appropriate",
        "suggested_definition": "Keep clips with stable framing and clear speech; reject clips with severe occlusion.",
        "justification": "Makes decisions consistent for borderline metadata combinations.",
    }
    score = grade(action, "task_easy")
    assert 0.01 <= score <= 0.99


def test_grade_task_averages_follow_hard_medium_easy_order(isolated_grader_state):
    del isolated_grader_state

    def task_average(task_id: str) -> float:
        scores: list[float] = []
        for clip in TASK_REGISTRY[task_id]["data_corpus"]:
            action = {
                "label": str(clip.get("expected_label", "BORDERLINE")),
                "clip_id": str(clip.get("clip_id", "")),
                "reasoning": (
                    "face_confidence, motion_score, audio_snr_db, and lighting_uniformity "
                    "support this decision."
                ),
                "confidence": 0.9,
            }
            scores.append(grade(action, task_id))
        return float(mean(scores))

    easy_avg = task_average("task_easy")
    medium_avg = task_average("task_medium")
    hard_avg = task_average("task_hard")

    # Hard task should still have less score by default due to stricter rules
    assert easy_avg >= medium_avg >= hard_avg


def test_grade_all_difficulties_use_standard_clamping(isolated_grader_state):
    del isolated_grader_state

    labels = ("KEEP", "BORDERLINE", "REJECT")
    reasoning_cases = (
        "x",
        "face_confidence, motion_score, audio_snr_db, and lighting_uniformity support this decision.",
    )
    confidence_cases = (0.0, 0.5, 1.0)

    for task_id in ("task_easy", "task_medium", "task_hard"):
        for clip in TASK_REGISTRY[task_id]["data_corpus"]:
            clip_id = str(clip.get("clip_id", ""))
            for label in labels:
                for reasoning in reasoning_cases:
                    for confidence in confidence_cases:
                        score = grade(
                            {
                                "label": label,
                                "clip_id": clip_id,
                                "reasoning": reasoning,
                                "confidence": confidence,
                            },
                            task_id,
                        )
                        # All scores must be in [0.01, 0.99] due to new clamping
                        assert 0.01 <= score <= 0.99

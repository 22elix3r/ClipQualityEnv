from __future__ import annotations

from statistics import mean

import pytest

import server.grader as grader_module
from clip_quality_env.difficulty import DIFFICULTY_TOTAL_BANDS
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


def test_grade_easy_clip_quality_scores_in_easy_band():
    action = {
        "label": "KEEP",
        "clip_id": "clip_0001",
        "reasoning": "face_confidence and audio_snr_db are high while motion_score is low, so this clip should be kept.",
        "confidence": 0.9,
    }
    score = grade(action, "task_easy")
    low, high = DIFFICULTY_TOTAL_BANDS["easy"]
    assert low <= score <= high


def test_grade_medium_clip_quality_scores_in_range():
    action = {
        "label": "BORDERLINE",
        "clip_id": "clip_0008",
        "reasoning": "face_area_ratio is borderline and motion_score is elevated, so borderline is safest.",
        "confidence": 0.74,
    }
    score = grade(action, "task_medium")
    low, high = DIFFICULTY_TOTAL_BANDS["medium"]
    assert low <= score <= high


def test_grade_hard_clip_quality_scores_in_range():
    action = {
        "label": "REJECT",
        "clip_id": "clip_0017",
        "reasoning": "face_confidence is weak, face_area_ratio is small, and motion_score is high enough to reject.",
        "confidence": 0.83,
    }
    score = grade(action, "task_hard")
    low, high = DIFFICULTY_TOTAL_BANDS["hard"]
    assert low <= score <= high


def test_grade_legacy_payload_is_supported_and_bounded():
    action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "appropriate",
        "suggested_definition": "Keep clips with stable framing and clear speech; reject clips with severe occlusion.",
        "justification": "Makes decisions consistent for borderline metadata combinations.",
    }
    score = grade(action, "task_easy")
    assert 0.0 <= score <= 1.0


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

    assert hard_avg > medium_avg > easy_avg


def test_grade_difficulty_bands_do_not_overlap(isolated_grader_state):
    del isolated_grader_state

    labels = ("KEEP", "BORDERLINE", "REJECT")
    reasoning_cases = (
        "x",
        "face_confidence, motion_score, audio_snr_db, and lighting_uniformity support this decision.",
    )
    confidence_cases = (0.0, 0.5, 1.0)

    ranges: dict[str, tuple[float, float]] = {}
    for task_id in ("task_easy", "task_medium", "task_hard"):
        task_scores: list[float] = []
        for clip in TASK_REGISTRY[task_id]["data_corpus"]:
            clip_id = str(clip.get("clip_id", ""))
            for label in labels:
                for reasoning in reasoning_cases:
                    for confidence in confidence_cases:
                        task_scores.append(
                            grade(
                                {
                                    "label": label,
                                    "clip_id": clip_id,
                                    "reasoning": reasoning,
                                    "confidence": confidence,
                                },
                                task_id,
                            )
                        )
        ranges[task_id] = (min(task_scores), max(task_scores))

    easy_min, easy_max = ranges["task_easy"]
    medium_min, medium_max = ranges["task_medium"]
    hard_min, hard_max = ranges["task_hard"]

    assert easy_min >= DIFFICULTY_TOTAL_BANDS["easy"][0]
    assert easy_max <= DIFFICULTY_TOTAL_BANDS["easy"][1]
    assert medium_min >= DIFFICULTY_TOTAL_BANDS["medium"][0]
    assert medium_max <= DIFFICULTY_TOTAL_BANDS["medium"][1]
    assert hard_min >= DIFFICULTY_TOTAL_BANDS["hard"][0]
    assert hard_max <= DIFFICULTY_TOTAL_BANDS["hard"][1]

    assert easy_max < medium_min
    assert medium_max < hard_min

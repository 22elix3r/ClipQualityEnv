from __future__ import annotations

import json
from statistics import mean

from clip_quality_env.env import ClipQualityEnvironment
from clip_quality_env.ground_truth import GTStore
from clip_quality_env.models import Action
from clip_quality_env.rubric import RubricState


def _action_for_clip(clip_id: str, label: str = "BORDERLINE") -> Action:
    return Action.model_validate(
        {
            "label": label,
            "reasoning": f"{label} decision for {clip_id} using clip metadata cues.",
            "confidence": 0.8,
            "clip_id": clip_id,
        }
    )


def test_environment_reset_returns_task_observation():
    env = ClipQualityEnvironment()
    obs = env.reset(task_id="task_easy")
    assert obs.task_id == "task_easy"
    assert obs.step_count == 0
    assert obs.corpus_size == obs.corpus_shown
    assert len(obs.data_corpus) == obs.corpus_size
    assert obs.clip_metadata.clip_id
    assert obs.max_steps == 5
    assert obs.info["steps_remaining"] == 5


def test_environment_step_updates_state_and_reward():
    env = ClipQualityEnvironment()
    env.reset(task_id="task_easy")
    action = Action.model_validate(
        {
            "label": "KEEP",
            "reasoning": "face_confidence and lighting_uniformity are high, motion_score is low, so keep.",
            "confidence": 0.9,
        }
    )
    next_obs = env.step(action)
    assert next_obs.step_count == 1
    assert 0.0 <= next_obs.reward <= 1.0
    assert next_obs.history
    state = env.state
    assert state.step_count == 1
    assert state.actions_taken[-1] in {"KEEP", "BORDERLINE", "REJECT"}


def test_environment_observation_exposes_reward_decomposition():
    env = ClipQualityEnvironment()
    reset_obs = env.reset(task_id="task_easy")

    assert reset_obs.info["format_score"] == 0.0
    assert reset_obs.info["label_score"] == 0.0
    assert reset_obs.info["reasoning_score"] == 0.0
    assert reset_obs.info["raw_total"] == 0.0
    assert reset_obs.info["calibrated_total"] == 0.0
    assert reset_obs.info["reward_total"] == 0.0
    assert reset_obs.info["total_reward"] == 0.0
    assert reset_obs.info["reward_breakdown"] == {
        "format_score": 0.0,
        "label_score": 0.0,
        "reasoning_score": 0.0,
        "raw_total": 0.0,
        "calibrated_total": 0.0,
        "total_reward": 0.0,
    }

    action = Action.model_validate(
        {
            "label": "KEEP",
            "reasoning": "face_confidence and lighting_uniformity are high, motion_score is low, so keep.",
            "confidence": 0.9,
        }
    )
    step_obs = env.step(action)

    assert step_obs.info["format_score"] in {0.0, 0.1}
    assert step_obs.info["label_score"] in {0.0, 0.25, 0.6}
    assert 0.0 <= step_obs.info["reasoning_score"] <= 0.3
    assert abs(
        float(step_obs.info["raw_total"])
        - (
            float(step_obs.info["format_score"])
            + float(step_obs.info["label_score"])
            + float(step_obs.info["reasoning_score"])
        )
    ) < 1e-9
    assert abs(float(step_obs.info["calibrated_total"]) - float(step_obs.reward)) < 1e-9
    assert abs(float(step_obs.info["reward_total"]) - float(step_obs.reward)) < 1e-9
    assert abs(float(step_obs.info["total_reward"]) - float(env.state.total_reward)) < 1e-9
    assert step_obs.info["reward_breakdown"] == {
        "format_score": float(step_obs.info["format_score"]),
        "label_score": float(step_obs.info["label_score"]),
        "reasoning_score": float(step_obs.info["reasoning_score"]),
        "raw_total": float(step_obs.info["raw_total"]),
        "calibrated_total": float(step_obs.info["calibrated_total"]),
        "total_reward": float(step_obs.info["reward_total"]),
    }


def test_environment_derives_expected_label_when_manifest_value_is_null(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest_null_expected.jsonl"
    manifest_row = {
        "difficulty": "easy",
        "clip_id": "manifest_null_expected",
        "expected_label": None,
        "face_confidence": 0.88,
        "motion_score": 0.12,
        "audio_snr_db": 24.0,
        "lighting_uniformity": 0.8,
        "duration_s": 8.0,
    }
    manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

    monkeypatch.setenv("REAL_CLIPS_MANIFEST", str(manifest_path))
    monkeypatch.setenv("CLIP_CORPUS_SOURCE", "manifest")

    env = ClipQualityEnvironment()
    obs = env.reset(task_id="task_easy", seed=2026)

    assert obs.info["corpus_source"] == "manifest:easy"
    assert str(obs.clip_metadata.expected_label).upper() in {"KEEP", "BORDERLINE", "REJECT"}
    assert str(obs.clip_metadata.expected_label).upper() != "NONE"


def test_environment_can_force_task_registry_source(monkeypatch):
    monkeypatch.setenv("CLIP_CORPUS_SOURCE", "task_registry")

    env = ClipQualityEnvironment()
    obs = env.reset(task_id="task_easy", seed=2026)

    assert obs.info["corpus_mode"] == "task_registry"
    assert obs.info["corpus_source"] == "task_registry:task_easy"


def test_environment_observation_includes_full_unsliced_corpus():
    env = ClipQualityEnvironment()
    obs = env.reset(task_id="task_medium")
    synthetic_corpus = []
    template = dict(obs.data_corpus[0]) if obs.data_corpus else {
        "expected_label": "BORDERLINE",
        "review_status": "pending",
    }
    for idx in range(12):
        item = dict(template)
        item["id"] = f"synthetic_{idx:03d}"
        item["clip_id"] = f"clip_synth_{idx:03d}"
        item["expected_label"] = str(item.get("expected_label") or "BORDERLINE")
        item["review_status"] = "pending"
        synthetic_corpus.append(item)
    env._episode_corpus[obs.task_id] = synthetic_corpus

    full_obs = env._state_to_observation(reward=0.0, done=False)

    assert full_obs.corpus_size == len(synthetic_corpus)
    assert full_obs.corpus_shown == len(synthetic_corpus)
    assert len(full_obs.data_corpus) == len(synthetic_corpus)
    assert [item["clip_id"] for item in full_obs.data_corpus] == [
        item["clip_id"] for item in synthetic_corpus
    ]


def test_environment_step_updates_submitted_clip_review_status():
    env = ClipQualityEnvironment()
    obs = env.reset(task_id="task_medium")
    current_clip_id = obs.clip_metadata.clip_id

    current_row = next(item for item in obs.data_corpus if item["clip_id"] == current_clip_id)
    assert str(current_row.get("review_status", "")).lower() == "pending"

    action = Action.model_validate(
        {
            "label": "REJECT",
            "reasoning": "multiple weak cues indicate this clip should be rejected.",
            "confidence": 0.81,
            "clip_id": current_clip_id,
        }
    )
    next_obs = env.step(action)
    updated_row = next(item for item in next_obs.data_corpus if item["clip_id"] == current_clip_id)

    assert updated_row["review_status"] == "REJECT"


def test_environment_instances_do_not_share_runtime_state():
    env_a = ClipQualityEnvironment()
    env_b = ClipQualityEnvironment()

    env_a.reset(task_id="task_easy")
    action = Action.model_validate(
        {
            "label": "KEEP",
            "reasoning": "face_confidence and lighting_uniformity are high, motion_score is low, so keep.",
            "confidence": 0.9,
        }
    )
    env_a.step(action)

    assert env_a is not env_b
    assert env_a.state.step_count == 1
    assert env_b.state.step_count == 0
    assert env_b.state.actions_taken == []


def test_environment_reset_plans_five_clips_from_selected_corpus():
    env = ClipQualityEnvironment()
    obs = env.reset(task_id="task_medium", seed=1234)
    expected_source = obs.info["corpus_source"]
    expected_data = list(obs.data_corpus)
    expected_ids = {str(item["clip_id"]) for item in expected_data}
    first_plan_ids = [str(item.clip.get("clip_id", "")) for item in env._episode_plan]

    assert obs.max_steps == 5
    assert obs.info["steps_remaining"] == 5
    assert len(env._episode_plan) == 5
    assert {item.task_id for item in env._episode_plan} == {"task_medium"}
    assert set(first_plan_ids).issubset(expected_ids)

    repeated = env.reset(task_id="task_medium", seed=1234)
    repeated_plan_ids = [str(item.clip.get("clip_id", "")) for item in env._episode_plan]
    assert repeated.max_steps == 5
    assert repeated.info["steps_remaining"] == 5
    assert repeated.info["corpus_source"] == expected_source
    assert repeated.data_corpus == expected_data
    assert repeated_plan_ids == first_plan_ids


def test_environment_updates_review_status_across_queue_and_summary():
    env = ClipQualityEnvironment()
    obs = env.reset(task_id="task_hard", seed=2026)

    assert obs.info["steps_remaining"] == 5
    assert "episode_summary" not in obs.info

    submitted: list[tuple[str, str]] = []
    final_obs = obs
    for step_index in range(1, 6):
        current_clip_id = final_obs.clip_metadata.clip_id
        label = "KEEP" if step_index % 2 else "BORDERLINE"
        submitted.append((current_clip_id, label))
        final_obs = env.step(_action_for_clip(current_clip_id, label=label))
        queue_row = next(item for item in final_obs.data_corpus if item.get("clip_id") == current_clip_id)
        assert str(queue_row.get("review_status")) == label
        assert final_obs.info["steps_remaining"] == max(0, 5 - step_index)

    assert final_obs.done is True
    assert final_obs.step_count == 5
    assert "episode_summary" in final_obs.info
    summary = final_obs.info["episode_summary"]
    assert summary["steps_completed"] == 5
    assert summary["max_steps"] == 5
    assert abs(float(summary["total_reward"]) - round(float(env.state.total_reward), 4)) < 1e-9
    assert abs(float(final_obs.info["total_reward"]) - float(env.state.total_reward)) < 1e-9
    assert abs(float(summary["average_reward"]) - round(float(env.state.total_reward) / 5.0, 4)) < 1e-9

    final_status_map = {str(item["clip_id"]): str(item["review_status"]) for item in final_obs.data_corpus}
    for clip_id, label in submitted:
        assert final_status_map[clip_id] == label


def test_environment_task_averages_follow_hard_medium_easy_order(monkeypatch, tmp_path):
    monkeypatch.setenv("REAL_CLIPS_MANIFEST", str(tmp_path / "missing_manifest.jsonl"))

    def run_task_average(task_id: str) -> float:
        env = ClipQualityEnvironment()
        env._rubric = RubricState(path=str(tmp_path / f"rubric_{task_id}.json"))
        env._gt_store = GTStore(
            seed_path="data/seed_gt.json",
            state_path=str(tmp_path / f"ground_truth_{task_id}.json"),
        )

        obs = env.reset(task_id=task_id, seed=2026)
        step_rewards: list[float] = []
        while True:
            expected_label = str(obs.clip_metadata.expected_label or "BORDERLINE")
            action = Action.model_validate(
                {
                    "label": expected_label,
                    "reasoning": (
                        "face_confidence, motion_score, audio_snr_db, and lighting_uniformity "
                        "support this decision."
                    ),
                    "confidence": 0.9,
                    "clip_id": obs.clip_metadata.clip_id,
                }
            )
            obs = env.step(action)
            step_rewards.append(float(obs.reward))
            if obs.done:
                break
        return float(mean(step_rewards))

    easy_avg = run_task_average("task_easy")
    medium_avg = run_task_average("task_medium")
    hard_avg = run_task_average("task_hard")

    assert hard_avg > medium_avg > easy_avg


def test_environment_difficulty_score_ranges_are_strictly_ordered(monkeypatch, tmp_path):
    monkeypatch.setenv("REAL_CLIPS_MANIFEST", str(tmp_path / "missing_manifest.jsonl"))

    def run_task_range(task_id: str) -> tuple[float, float]:
        task_scores: list[float] = []
        labels = ("KEEP", "BORDERLINE", "REJECT")
        reasoning_cases = (
            "x",
            "face_confidence, motion_score, audio_snr_db, and lighting_uniformity support this decision.",
        )
        confidence_cases = (0.0, 0.5, 1.0)

        for label in labels:
            for reasoning in reasoning_cases:
                for confidence in confidence_cases:
                    env = ClipQualityEnvironment()
                    env._rubric = RubricState(path=str(tmp_path / f"rubric_range_{task_id}.json"))
                    env._gt_store = GTStore(
                        seed_path="data/seed_gt.json",
                        state_path=str(tmp_path / f"ground_truth_range_{task_id}.json"),
                    )

                    obs = env.reset(task_id=task_id, seed=2026)
                    while True:
                        action = Action.model_validate(
                            {
                                "label": label,
                                "reasoning": reasoning,
                                "confidence": confidence,
                                "clip_id": obs.clip_metadata.clip_id,
                            }
                        )
                        obs = env.step(action)
                        task_scores.append(float(obs.reward))
                        if obs.done:
                            break

        return min(task_scores), max(task_scores)

    easy_min, easy_max = run_task_range("task_easy")
    medium_min, medium_max = run_task_range("task_medium")
    hard_min, hard_max = run_task_range("task_hard")

    assert easy_max < medium_min
    assert medium_max < hard_min

    assert 0.0 <= easy_min <= easy_max <= 1.0
    assert 0.0 <= medium_min <= medium_max <= 1.0
    assert 0.0 <= hard_min <= hard_max <= 1.0

from __future__ import annotations

import inspect
import json
import time

from fastapi.testclient import TestClient
import gradio as gr
import pandas as pd

import server.app as app_module
from server.baseline_runs import BaselineRunTracker
from server.app import app
from server.environment import ClipQualityEnvironment


def _tiered_step_inputs(
    *,
    easy_observation: str = "quick metadata cue.",
    medium_primary_signal: str = "primary signal",
    medium_conflicting_signal: str = "conflicting signal",
    medium_reasoning: str = "balanced assessment rationale.",
    hard_tradeoff_summary: str = "trade-off summary",
    hard_confidence_justification: str = "confidence rationale",
    hard_confidence: float = 0.5,
) -> tuple:
    """Return the 7 tiered input values matching handle_step's signature
    (easy_observation, medium_primary_signal, medium_conflicting_signal,
     medium_reasoning, hard_tradeoff_summary, hard_confidence_justification,
     hard_confidence).  Callers append clip_id and icl_memory_state."""
    return (
        easy_observation,
        medium_primary_signal,
        medium_conflicting_signal,
        medium_reasoning,
        hard_tradeoff_summary,
        hard_confidence_justification,
        hard_confidence,
    )


def test_root_and_health_routes():
    client = TestClient(app)
    root = client.get("/")
    assert root.status_code == 200
    assert root.json().get("status") == "ok"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json().get("status") == "ok"


def test_tasks_route_returns_catalog():
    client = TestClient(app)
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert isinstance(tasks, list)
    assert len(tasks) >= 3
    task_ids = {t["task_id"] for t in tasks}
    assert {"task_easy", "task_medium", "task_hard"}.issubset(task_ids)


def test_dashboard_route_exists():
    client = TestClient(app)
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_dashboard_remaining_steps_defaults_to_five():
    demo = app_module.build_custom_ui()
    step_indicators = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Number) and getattr(block, "label", "") == "Remaining Execution Steps"
    ]
    assert step_indicators
    assert float(step_indicators[0].value) == 5.0


def test_grader_route_scores_action():
    client = TestClient(app)
    resp = client.post(
        "/grader",
        params={"task_id": "task_easy"},
        json={
            "label": "KEEP",
            "reasoning": "face_confidence is high and motion_score is low, so keep this clip.",
            "confidence": 0.91,
            "clip_id": "clip_0001",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "task_easy"
    assert 0.0 <= body["score"] <= 1.0


def _fresh_tracker(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "baseline_run_tracker", BaselineRunTracker())


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.value = float(start)

    def time(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def _wait_for_terminal_status(client: TestClient, run_id: str) -> dict:
    deadline = time.time() + 2.0
    latest: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/baseline/status/{run_id}")
        assert resp.status_code == 200
        latest = resp.json()
        if latest["status"] in {"complete", "failed"}:
            return latest
        time.sleep(0.01)
    return latest


def test_baseline_start_and_status_complete(monkeypatch):
    _fresh_tracker(monkeypatch)

    def _fake_run_baseline(task: str | None = None, icl_memory=None) -> dict:
        assert task == "task_easy"
        return {
            "baseline_scores": {"overall_avg": 0.75},
            "model": "mock-model",
            "runtime_seconds": 0.03,
            "detail": [{"task_id": "task_easy", "reward": 0.75, "steps": 1, "success": True, "mode": "fallback"}],
        }

    monkeypatch.setattr(app_module.inference, "run_baseline", _fake_run_baseline)

    client = TestClient(app)
    start = client.post("/baseline/start", params={"task": "task_easy"})
    assert start.status_code == 200
    start_body = start.json()
    assert start_body["status"] == "running"
    assert "run_id" in start_body
    assert start_body["payload"]["baseline_results"] == []
    assert start_body["payload"]["average_score"] is None
    assert start_body["payload"]["task"] == "task_easy"

    status = _wait_for_terminal_status(client, start_body["run_id"])
    assert status["status"] == "complete"
    assert status["payload"]["average_score"] == 0.75
    assert status["payload"]["baseline_results"][0]["task_id"] == "task_easy"
    assert status["payload"]["model"] == "mock-model"


def test_baseline_status_failed(monkeypatch):
    _fresh_tracker(monkeypatch)

    def _fail_run_baseline(task: str | None = None, icl_memory=None) -> dict:
        raise RuntimeError(f"boom: {task}")

    monkeypatch.setattr(app_module.inference, "run_baseline", _fail_run_baseline)

    client = TestClient(app)
    start = client.post("/baseline/start", params={"task": "task_medium"})
    assert start.status_code == 200
    run_id = start.json()["run_id"]

    status = _wait_for_terminal_status(client, run_id)
    assert status["status"] == "failed"
    assert "error" in status
    assert "boom: task_medium" in status["error"]["message"]


def test_baseline_status_unknown_run_id_returns_404(monkeypatch):
    _fresh_tracker(monkeypatch)

    client = TestClient(app)
    resp = client.get("/baseline/status/missing-run-id")
    assert resp.status_code == 404
    assert "Unknown run_id" in resp.json()["detail"]


def test_baseline_get_route_is_compatibility_wrapper(monkeypatch):
    _fresh_tracker(monkeypatch)

    def _fake_run_baseline(task: str | None = None, icl_memory=None) -> dict:
        return {
            "baseline_scores": {"overall_avg": 0.5},
            "model": "compat-model",
            "runtime_seconds": 0.01,
            "detail": [{"task_id": "task_hard", "reward": 0.5, "steps": 1, "success": False, "mode": "fallback"}],
        }

    monkeypatch.setattr(app_module.inference, "run_baseline", _fake_run_baseline)

    client = TestClient(app)
    resp = client.get("/baseline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert "run_id" in body

    status = _wait_for_terminal_status(client, body["run_id"])
    assert status["status"] == "complete"
    assert status["payload"]["average_score"] == 0.5


def test_baseline_status_cleanup_ttl_removes_expired_runs(monkeypatch):
    clock = _FakeClock(start=10.0)
    tracker = BaselineRunTracker(ttl_seconds=0.1, time_fn=clock.time)
    monkeypatch.setattr(app_module, "baseline_run_tracker", tracker)

    run_id = tracker.create_run()
    tracker.update_partial(run_id, {"baseline_results": [], "average_score": None})

    clock.advance(0.2)
    client = TestClient(app)
    resp = client.get(f"/baseline/status/{run_id}")

    assert resp.status_code == 404
    assert "Unknown run_id" in resp.json()["detail"]


def test_dashboard_wires_environment_state_via_gradio_state():
    demo = app_module.build_custom_ui()
    state_ids = [block_id for block_id, block in demo.blocks.items() if isinstance(block, gr.State)]
    assert state_ids, "Expected a gr.State component for per-session env state"

    state_id = state_ids[0]
    dependencies = demo.config.get("dependencies", [])
    reset_dep = next(dep for dep in dependencies if dep.get("api_name") == "handle_reset")
    step_dep = next(dep for dep in dependencies if dep.get("api_name") == "handle_step")

    assert reset_dep["inputs"][0] == state_id
    assert reset_dep["outputs"][0] == state_id
    assert step_dep["inputs"][0] == state_id
    assert step_dep["outputs"][0] == state_id


def test_dashboard_handlers_accept_env_state_first():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}

    reset_params = list(inspect.signature(handler_map["handle_reset"]).parameters.keys())
    step_params = list(inspect.signature(handler_map["handle_step"]).parameters.keys())

    assert reset_params[0] == "env_state"
    assert step_params[0] == "env_state"


def test_dashboard_state_default_is_none_until_first_event():
    demo = app_module.build_custom_ui()
    state_ids = [block_id for block_id, block in demo.blocks.items() if isinstance(block, gr.State)]
    assert state_ids
    assert demo.blocks[state_ids[0]].value is None


def test_dashboard_handlers_create_isolated_environment_instances():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]
    handle_step = handler_map["handle_step"]

    session_a = handle_reset(None, "task_easy")
    env_a = session_a[0]
    assert isinstance(env_a, ClipQualityEnvironment)
    assert env_a.state.step_count == 0

    # handle_step now runs a full 5-step agent episode
    stepped_a = handle_step(
        env_a,
        "task_easy",
        *_tiered_step_inputs(
            easy_observation="face_confidence is high and motion_score is low.",
        ),
        "",    # clip_id
        None,  # icl_memory_state
    )
    assert stepped_a[0] is env_a
    assert env_a.state.step_count == 5

    session_b = handle_reset(None, "task_easy")
    env_b = session_b[0]
    assert isinstance(env_b, ClipQualityEnvironment)
    assert env_b is not env_a
    assert env_b.state.step_count == 0
    assert env_a.state.step_count == 5

    handle_step(
        env_b,
        "task_easy",
        *_tiered_step_inputs(
            easy_observation="mixed metadata cues suggest borderline quality.",
        ),
        "",    # clip_id
        None,  # icl_memory_state
    )
    assert env_b.state.step_count == 5
    assert env_a.state.step_count == 5


def test_tiered_submission_merge_by_tab():
    easy_label, easy_reasoning, easy_confidence = app_module._resolve_tiered_submission(
        task_id="task_easy",
        selected_input_tab="easy",
        easy_label="KEEP",
        easy_observation="clear frontal face with stable motion.",
        medium_label="REJECT",
        medium_primary_signal="noisy audio",
        medium_conflicting_signal="good framing",
        medium_reasoning="borderline mix",
        hard_label="REJECT",
        hard_tradeoff_summary="trade-offs",
        hard_confidence_justification="hard confidence rationale",
        hard_confidence=0.12,
    )
    assert easy_label == "KEEP"
    assert easy_confidence == 0.5
    assert "Tier: Easy" in easy_reasoning
    assert "Key Observation: clear frontal face with stable motion." in easy_reasoning
    assert "Primary Signal" not in easy_reasoning
    assert "Trade-off Summary" not in easy_reasoning

    medium_label, medium_reasoning, medium_confidence = app_module._resolve_tiered_submission(
        task_id="task_medium",
        selected_input_tab="medium",
        easy_label="KEEP",
        easy_observation="easy cue",
        medium_label="BORDERLINE",
        medium_primary_signal="strong face confidence",
        medium_conflicting_signal="elevated motion score",
        medium_reasoning="mixed indicators warrant caution.",
        hard_label="REJECT",
        hard_tradeoff_summary="hard trade-off",
        hard_confidence_justification="hard confidence rationale",
        hard_confidence=0.91,
    )
    assert medium_label == "BORDERLINE"
    assert medium_confidence == 0.5
    assert "Tier: Medium" in medium_reasoning
    assert "Primary Signal: strong face confidence" in medium_reasoning
    assert "Conflicting Signal: elevated motion score" in medium_reasoning
    assert "Reasoning: mixed indicators warrant caution." in medium_reasoning
    assert "Trade-off Summary" not in medium_reasoning

    hard_label, hard_reasoning, hard_confidence = app_module._resolve_tiered_submission(
        task_id="task_hard",
        selected_input_tab="hard",
        easy_label="KEEP",
        easy_observation="easy cue",
        medium_label="BORDERLINE",
        medium_primary_signal="medium primary",
        medium_conflicting_signal="medium conflicting",
        medium_reasoning="medium reasoning",
        hard_label="REJECT",
        hard_tradeoff_summary="high fidelity but unstable motion.",
        hard_confidence_justification="multiple reject cues outweigh positives.",
        hard_confidence=0.83,
    )
    assert hard_label == "REJECT"
    assert hard_confidence == 0.83
    assert "Tier: Hard" in hard_reasoning
    assert "Trade-off Summary: high fidelity but unstable motion." in hard_reasoning
    assert "Confidence Justification: multiple reject cues outweigh positives." in hard_reasoning
    assert "Confidence: 0.83" in hard_reasoning


def test_task_change_handler_maps_scenario_to_input_tab():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    sync_tabs = handler_map["sync_input_tab_for_task"]

    # sync_tabs returns a tuple (update_dict, tab_id_str)
    easy_update, _ = sync_tabs("task_easy")
    medium_update, _ = sync_tabs("task_medium")
    hard_update, _ = sync_tabs("task_hard")

    assert easy_update["selected"] == "easy"
    assert medium_update["selected"] == "medium"
    assert hard_update["selected"] == "hard"


def test_handle_step_runs_agent_episode_and_returns_valid_outputs():
    """handle_step now runs a full 5-step agent episode internally.
    Verify it produces valid structured outputs."""
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]
    handle_step = handler_map["handle_step"]

    reset_result = handle_reset(None, "task_medium")
    env = reset_result[0]

    step_result = handle_step(
        env,
        "task_medium",
        *_tiered_step_inputs(
            medium_primary_signal="face_confidence is high.",
            medium_conflicting_signal="motion_score is elevated.",
            medium_reasoning="mixed cues imply borderline quality.",
        ),
        "",    # clip_id
        None,  # icl_memory_state
    )

    # handle_step runs a full 5-step episode
    assert step_result[0] is env
    assert env.state.step_count == 5

    # Corpus DataFrame should be populated
    step_df = step_result[1]
    assert not step_df.empty

    # Session history should have 5 rows
    history_df = step_result[9]
    assert len(history_df) == 5

    # All submitted labels should be valid
    for label in history_df["Submitted Label"]:
        assert label in {"KEEP", "BORDERLINE", "REJECT"}


def test_format_obs_sorts_queue_by_clip_id_and_uses_full_corpus_counts():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]

    class _Obs:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def model_dump(self) -> dict:
            return self._payload

    synthetic_corpus = [
        {"clip_id": "clip_010", "expected_label": "BORDERLINE", "review_status": "pending"},
        {"clip_id": "clip_002", "expected_label": "KEEP", "review_status": "pending"},
        {"clip_id": "clip_111", "expected_label": "REJECT", "review_status": "pending"},
        {"clip_id": "clip_021", "expected_label": "BORDERLINE", "review_status": "pending"},
    ]
    fake_obs = {
        "task_id": "task_medium",
        "episode_id": "episode-format",
        "step_count": 0,
        "max_steps": 5,
        "step": 1,
        "rubric_version": 1,
        "rubric_summary": "test rubric",
        "clip_metadata": {"clip_id": "clip_002"},
        "history": [],
        "corpus_size": len(synthetic_corpus),
        "corpus_shown": len(synthetic_corpus),
        "data_corpus": synthetic_corpus,
        "reward": 0.0,
        "done": False,
        "info": {
            "best_score": 0.0,
            "steps_remaining": 5,
            "session_history": [],
            "total_reward": 0.0,
            "format_score": 0.0,
            "label_score": 0.0,
            "reasoning_score": 0.0,
            "reward_total": 0.0,
        },
    }
    env = ClipQualityEnvironment()
    env.reset = lambda task_id: _Obs(fake_obs)  # type: ignore[assignment]

    reformatted_result = handle_reset(env, "task_medium")
    corpus_df = reformatted_result[1]
    corpus_stat = reformatted_result[7]

    expected_clip_ids = sorted(item["clip_id"] for item in synthetic_corpus)
    rendered_clip_ids = corpus_df["Clip ID"].tolist()

    assert rendered_clip_ids == expected_clip_ids
    assert len(rendered_clip_ids) == len(synthetic_corpus)
    assert (
        corpus_stat
        == f"### Clip Queue: **{len(synthetic_corpus)}** of **{len(synthetic_corpus)}** items displayed"
    )


def test_dashboard_queue_row_updates_to_submitted_label_after_step():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]
    handle_step = handler_map["handle_step"]

    reset_result = handle_reset(None, "task_easy")
    env = reset_result[0]

    reset_df = reset_result[1]
    # All clips start as pending
    assert all(str(s).lower() == "pending" for s in reset_df["Current Review Status"])

    # handle_step runs a full 5-step agent episode
    step_result = handle_step(
        env,
        "task_easy",
        *_tiered_step_inputs(
            easy_observation="clear reject cues across motion and confidence.",
        ),
        "",    # clip_id
        None,  # icl_memory_state
    )
    step_df = step_result[1]

    # After a full episode, at least one clip should have its review_status updated
    submitted_clips = step_df.loc[step_df["Current Review Status"].isin(["KEEP", "BORDERLINE", "REJECT"])]
    assert not submitted_clips.empty


def test_dashboard_shows_session_history_tab_and_columns():
    demo = app_module.build_custom_ui()
    history_tables = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.DataFrame) and getattr(block, "label", "") == "Per-step Classification History"
    ]
    assert history_tables
    table_value = history_tables[0].value
    assert isinstance(table_value, dict)
    assert table_value["headers"] == [
        "Step",
        "Difficulty",
        "Clip ID",
        "Submitted Label",
        "Expected Label",
        "Reward",
    ]


def test_dashboard_session_history_updates_and_resets():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]
    handle_step = handler_map["handle_step"]

    reset_result = handle_reset(None, "task_easy")
    env = reset_result[0]
    assert isinstance(env, ClipQualityEnvironment)

    # After reset, session history is empty
    reset_history_df = reset_result[9]
    assert isinstance(reset_history_df, pd.DataFrame)
    assert list(reset_history_df.columns) == ["Step", "Difficulty", "Clip ID", "Submitted Label", "Expected Label", "Reward"]
    assert reset_history_df.empty
    assert "No actions yet" in reset_result[10]

    # handle_step runs a full 5-step episode via the agent
    first_step = handle_step(
        env,
        "task_easy",
        *_tiered_step_inputs(
            easy_observation="face_confidence is high and motion_score is low.",
        ),
        "",    # clip_id
        None,  # icl_memory_state
    )
    first_history_df = first_step[9]
    first_rows = first_history_df.to_dict(orient="records")
    assert len(first_rows) == 5

    # Verify structure of each history row
    for row in first_rows:
        assert isinstance(row["Step"], (int, float))
        assert isinstance(row["Clip ID"], str) and row["Clip ID"]
        assert row["Submitted Label"] in {"KEEP", "BORDERLINE", "REJECT"}
        assert isinstance(row["Reward"], (int, float))
        assert 0.0 <= float(row["Reward"]) <= 1.0

    # Verify match cues markdown is populated
    assert "Match" in first_step[10] or "Mismatch" in first_step[10]

    # After re-reset, history is empty again
    reset_again = handle_reset(env, "task_easy")
    assert reset_again[0] is env
    assert reset_again[9].empty
    assert "No actions yet" in reset_again[10]


def test_dashboard_shows_dominant_features_panel_and_columns():
    demo = app_module.build_custom_ui()

    feature_tables = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.DataFrame) and getattr(block, "label", "") == "Feature Focus Table"
    ]
    assert feature_tables
    table_value = feature_tables[0].value
    assert isinstance(table_value, dict)
    assert table_value["headers"] == [
        "Feature Name",
        "Current Value",
        "Rubric Status",
        "Threshold Range",
    ]


def test_dashboard_handlers_populate_and_refresh_dominant_feature_rows():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]
    handle_step = handler_map["handle_step"]

    reset_result = handle_reset(None, "task_easy")
    env = reset_result[0]
    reset_df = reset_result[3]
    expected_columns = ["Feature Name", "Current Value", "Rubric Status", "Threshold Range"]
    assert list(reset_df.columns) == expected_columns

    expected_reset_df = pd.DataFrame(env.dominant_feature_rows(), columns=expected_columns)
    assert reset_df.to_dict(orient="records") == expected_reset_df.to_dict(orient="records")

    step_result = handle_step(
        env,
        "task_easy",
        *_tiered_step_inputs(
            easy_observation="face_confidence is high and motion_score is low.",
        ),
        "",    # clip_id
        None,  # icl_memory_state
    )
    step_df = step_result[3]
    expected_step_df = pd.DataFrame(env.dominant_feature_rows(), columns=expected_columns)
    assert step_df.to_dict(orient="records") == expected_step_df.to_dict(orient="records")


def test_dashboard_reward_breakdown_panel_initializes_and_updates():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]
    handle_step = handler_map["handle_step"]

    reset_result = handle_reset(None, "task_easy")
    env = reset_result[0]
    reset_reward_md = reset_result[8]
    assert "Format Score" in reset_reward_md
    assert "Label Score" in reset_reward_md
    assert "Reasoning Score" in reset_reward_md
    assert "Total Reward" in reset_reward_md
    assert "0.00" in reset_reward_md

    step_result = handle_step(
        env,
        "task_easy",
        *_tiered_step_inputs(
            easy_observation="face_confidence is high and motion_score is low.",
        ),
        "",    # clip_id
        None,  # icl_memory_state
    )
    step_reward_md = step_result[8]
    step_obs = json.loads(step_result[12])
    info = step_obs["info"]
    assert "Format Score" in step_reward_md
    assert "Label Score" in step_reward_md
    assert "Reasoning Score" in step_reward_md
    assert f"{float(info['format_score']):.2f}" in step_reward_md
    assert f"{float(info['label_score']):.2f}" in step_reward_md
    assert f"{float(info['reasoning_score']):.2f}" in step_reward_md
    assert f"{float(step_obs['reward']):.3f}" in step_reward_md


def test_dashboard_includes_quality_hint_button_and_wiring():
    demo = app_module.build_custom_ui()

    buttons = [block for block in demo.blocks.values() if isinstance(block, gr.Button)]
    assert any(getattr(button, "value", "") == app_module.QUALITY_HINT_BUTTON_LABEL for button in buttons)

    dependencies = demo.config.get("dependencies", [])
    hint_dep = next(dep for dep in dependencies if dep.get("api_name") == "handle_quality_hint")
    assert any(target_event == "click" for _, target_event in hint_dep.get("targets", []))


def test_handle_quality_hint_populates_active_tab_field():
    demo = app_module.build_custom_ui()
    handler_map = {block_fn.fn.__name__: block_fn.fn for block_fn in demo.fns.values()}
    handle_reset = handler_map["handle_reset"]
    handle_quality_hint = handler_map["handle_quality_hint"]

    reset_result = handle_reset(None, "task_easy")
    env = reset_result[0]

    hint_easy = handle_quality_hint(env, "task_easy", "easy", "", "", "", None)
    assert hint_easy[0] is env
    assert isinstance(hint_easy[1], str) and hint_easy[1]
    assert "which is" in hint_easy[1]
    assert hint_easy[2] == ""
    assert hint_easy[3] == ""

    hint_medium = handle_quality_hint(env, "task_medium", "medium", "keep easy", "", "keep hard", None)
    assert hint_medium[0] is env
    assert hint_medium[1] == "keep easy"
    assert isinstance(hint_medium[2], str) and hint_medium[2]
    assert "which is" in hint_medium[2]
    assert hint_medium[3] == "keep hard"


def test_reward_breakdown_markdown_color_coding():
    obs = {
        "reward": 0.85,
        "history": [{"step": 1}],
        "info": {
            "format_score": 0.10,
            "label_score": 0.25,
            "reasoning_score": 0.30,
            "reward_total": 0.85,
            "total_reward": 1.2,
            "best_score": 1.2,
        },
    }
    reward_md = app_module._reward_breakdown_markdown(obs, initialized=False)
    assert "Format Score" in reward_md and "#16a34a" in reward_md
    assert "Label Score" in reward_md and "#f59e0b" in reward_md
    assert "Reasoning Score" in reward_md and "0.30" in reward_md
    assert "Total Reward" in reward_md and "0.850" in reward_md


def test_dashboard_includes_baseline_agent_controls():
    demo = app_module.build_custom_ui()

    buttons = [block for block in demo.blocks.values() if isinstance(block, gr.Button)]
    assert any(getattr(button, "value", "") == app_module.BASELINE_RUN_BUTTON_LABEL for button in buttons)

    accordions = [block for block in demo.blocks.values() if isinstance(block, gr.Accordion)]
    assert any(getattr(accordion, "label", "") == "LLM Baseline Result" for accordion in accordions)

    dependencies = demo.config.get("dependencies", [])
    click_dep = next(dep for dep in dependencies if dep.get("api_name") == "_start_baseline_with_corpus")
    tick_dep = next(dep for dep in dependencies if dep.get("api_name") == "_poll_baseline_with_corpus")

    assert any(target_event == "click" for _, target_event in click_dep.get("targets", []))
    assert any(target_event == "tick" for _, target_event in tick_dep.get("targets", []))


def test_baseline_markdown_warns_when_hf_token_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    payload = {
        "model": "mock-model",
        "baseline_results": [
            {"task_id": "task_easy", "mode": "fallback", "reward": 0.6, "steps": 2, "success": False}
        ],
        "average_score": 0.6,
    }

    md = app_module._format_baseline_result_markdown(payload, status="complete")
    assert "Model" in md
    assert "Deterministic fallback" in md
    assert "score" in md.lower()
    assert "success **No**" in md
    assert app_module.HF_TOKEN_MISSING_WARNING in md


def test_start_and_poll_baseline_ui_run(monkeypatch):
    tracker = BaselineRunTracker()
    monkeypatch.setattr(app_module, "baseline_run_tracker", tracker)

    def _fake_run_baseline(task: str | None = None, icl_memory=None) -> dict:
        assert task == "task_easy"
        return {
            "baseline_scores": {"overall_avg": 0.9},
            "model": "ui-model",
            "runtime_seconds": 0.01,
            "detail": [{"task_id": "task_easy", "reward": 0.9, "steps": 3, "success": True, "mode": "llm"}],
        }

    monkeypatch.setattr(app_module.inference, "run_baseline", _fake_run_baseline)
    monkeypatch.setenv("HF_TOKEN", "token-present")

    run_id, status_text, md_running, button_update, timer_update = app_module._start_baseline_ui_run("task_easy")
    assert "baseline run in progress" in status_text.lower()
    assert "Pending" in md_running
    assert button_update["value"] == app_module.BASELINE_RUN_BUTTON_RUNNING_LABEL
    assert button_update["interactive"] is False
    assert timer_update["active"] is True

    deadline = time.time() + 2.0
    poll = (run_id, "", "", {}, {})
    while time.time() < deadline:
        poll = app_module._poll_baseline_ui_run(run_id)
        if poll[0] is None:
            break
        time.sleep(0.01)

    _, status_done, md_done, button_done, timer_done = poll
    assert "complete" in status_done.lower() or "failed" not in status_done.lower()
    assert "Model" in md_done
    assert button_done["value"] == app_module.BASELINE_RUN_BUTTON_LABEL
    assert button_done["interactive"] is True
    assert timer_done["active"] is False

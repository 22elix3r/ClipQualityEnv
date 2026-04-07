from __future__ import annotations

import argparse
import json
import os
import threading
from typing import Any

import gradio as gr
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, HTTPException
from openenv.core.env_server import create_fastapi_app

import inference
from clip_quality_env.icl_memory import ICLMemory
from clip_quality_env.ground_truth import GTStore as _UIGTStore
from clip_quality_env.rubric import RubricState as _UIRubricState
from models import Action, Observation, TaskInfo
from server.baseline_runs import baseline_run_tracker
from server.environment import ClipQualityEnvironment
from server.grader import grade
from server.tasks import TASK_REGISTRY

PRODUCT_NAME = "ClipQualityEnv"
ENVIRONMENT_ID = "clip_quality_env"
OVERRIDDEN_ROUTES = {"/health", "/state", "/tasks", "/grader", "/baseline"}
TASK_SURFACE_DESCRIPTIONS = {
    "task_easy": "Classify clips with clear quality signals and concise metadata-based reasoning.",
    "task_medium": "Classify borderline clips by balancing mixed quality indicators.",
    "task_hard": "Classify difficult clips with conflicting signals and explicit trade-off reasoning.",
}
TASK_INPUT_TAB_MAP = {
    "task_easy": "easy",
    "task_medium": "medium",
    "task_hard": "hard",
}
CLASS_LABEL_CHOICES = [
    ("KEEP", "KEEP"),
    ("BORDERLINE", "BORDERLINE"),
    ("REJECT", "REJECT"),
]
BASELINE_RUN_BUTTON_LABEL = "Run LLM Baseline Agent"
BASELINE_RUN_BUTTON_RUNNING_LABEL = "Running LLM Baseline Agent..."
QUALITY_HINT_BUTTON_LABEL = "Load Quality Hint"
BASELINE_RESULT_PLACEHOLDER = (
    "Run the baseline agent to view model, mode, reward achieved per step, and success status."
)
HF_TOKEN_MISSING_WARNING = "LLM unavailable — no HF_TOKEN configured. Showing deterministic fallback result."


def _normalized_path(path: str) -> str:
    return path.rstrip("/") or "/"


def _surface_task_description(task_id: str, fallback: str) -> str:
    return TASK_SURFACE_DESCRIPTIONS.get(task_id, fallback)


def _input_tab_for_task(task_id: str) -> str:
    return TASK_INPUT_TAB_MAP.get(task_id, "easy")


def _clean_reasoning_text(value: str | None) -> str:
    return str(value or "").strip()


def _build_reasoning_from_sections(sections: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for title, value in sections:
        cleaned = _clean_reasoning_text(value)
        if cleaned:
            lines.append(f"{title}: {cleaned}")
    return "\n".join(lines) if lines else "No reasoning provided."


def _resolve_tiered_submission(
    task_id: str,
    selected_input_tab: str | None,
    easy_label: str,
    easy_observation: str,
    medium_label: str,
    medium_primary_signal: str,
    medium_conflicting_signal: str,
    medium_reasoning: str,
    hard_label: str,
    hard_tradeoff_summary: str,
    hard_confidence_justification: str,
    hard_confidence: float,
) -> tuple[str, str, float]:
    active_tab = selected_input_tab if selected_input_tab in {"easy", "medium", "hard"} else _input_tab_for_task(task_id)

    if active_tab == "easy":
        label = str(easy_label or "BORDERLINE").strip().upper()
        reasoning = _build_reasoning_from_sections(
            [
                ("Tier", "Easy"),
                ("Predicted Label", label),
                ("Key Observation", easy_observation),
            ]
        )
        confidence = 0.5
    elif active_tab == "medium":
        label = str(medium_label or "BORDERLINE").strip().upper()
        reasoning = _build_reasoning_from_sections(
            [
                ("Tier", "Medium"),
                ("Predicted Label", label),
                ("Primary Signal", medium_primary_signal),
                ("Conflicting Signal", medium_conflicting_signal),
                ("Reasoning", medium_reasoning),
            ]
        )
        confidence = 0.5
    else:
        label = str(hard_label or "BORDERLINE").strip().upper()
        confidence = float(hard_confidence)
        reasoning = _build_reasoning_from_sections(
            [
                ("Tier", "Hard"),
                ("Predicted Label", label),
                ("Trade-off Summary", hard_tradeoff_summary),
                ("Confidence Justification", hard_confidence_justification),
                ("Confidence", f"{confidence:.2f}"),
            ]
        )

    if label not in {"KEEP", "BORDERLINE", "REJECT"}:
        label = "BORDERLINE"
    return label, reasoning, confidence


def _input_tab_update_for_task(task_id: str) -> dict[str, Any]:
    return gr.update(selected=_input_tab_for_task(task_id))


app = create_fastapi_app(
    env=ClipQualityEnvironment,
    action_cls=Action,
    observation_cls=Observation,
)
# Replace selected OpenEnv defaults with reference-style handlers.
overridden_paths = {_normalized_path(path) for path in OVERRIDDEN_ROUTES}
app.router.routes = [
    route
    for route in app.router.routes
    if _normalized_path(getattr(route, "path", "")) not in overridden_paths
]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": ENVIRONMENT_ID,
        "product": PRODUCT_NAME,
        "health": "/health",
        "metadata": "/metadata",
        "dashboard": "/dashboard/",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "clip-quality-api"}


@app.get("/state")
def get_state() -> dict[str, Any]:
    env = ClipQualityEnvironment()
    state = env.state.model_dump()
    state["product"] = PRODUCT_NAME
    state["workflow"] = "clip_quality_analysis"
    return state


@app.get("/tasks")
def list_tasks() -> list[TaskInfo]:
    return [
        TaskInfo(
            task_id=task_id,
            difficulty=task["difficulty"],
            description=_surface_task_description(task_id, task["description"]),
            action_schema=Action.model_json_schema(),
        )
        for task_id, task in TASK_REGISTRY.items()
    ]


@app.post("/grader")
def get_grader_score(task_id: str, action: Action) -> dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    score = grade(action.model_dump(), task_id)
    return {
        "task_id": task_id,
        "score": score,
        "passed": 1 if score > 0.5 else 0,
        "total": 1,
        "metric": "clip_quality_alignment",
    }


def _baseline_payload_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    scores = raw.get("baseline_scores")
    score_payload = scores if isinstance(scores, dict) else {}
    payload: dict[str, Any] = {
        "baseline_results": raw.get("detail", []),
        "average_score": score_payload.get("overall_avg", 0.0),
        "model": raw.get("model", os.environ.get("MODEL_NAME", inference.DEFAULT_MODEL_NAME)),
        "metric": "clip_quality_alignment",
    }
    if "runtime_seconds" in raw:
        payload["runtime_seconds"] = raw.get("runtime_seconds")
    if "warning" in raw:
        payload["warning"] = raw.get("warning")
    return payload


def _initial_baseline_payload(task: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "baseline_results": [],
        "average_score": None,
        "model": os.environ.get("MODEL_NAME", inference.DEFAULT_MODEL_NAME),
        "metric": "clip_quality_alignment",
    }
    if task:
        payload["task"] = task
    return payload


def _public_baseline_status(status: str) -> str:
    return "complete" if status == "completed" else status


def _mode_label(mode: str) -> str:
    return "LLM" if str(mode).strip().lower() == "llm" else "Deterministic fallback"


def _summarize_mode(results: list[dict[str, Any]]) -> str:
    modes = {str(item.get("mode", "fallback")).strip().lower() for item in results}
    if not modes or modes == {"fallback"}:
        return "Deterministic fallback"
    if modes == {"llm"}:
        return "LLM"
    return "Mixed (LLM + deterministic fallback)"


def _baseline_warning_messages(payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not os.environ.get("HF_TOKEN"):
        warnings.append(HF_TOKEN_MISSING_WARNING)
    payload_warning = str(payload.get("warning") or "").strip()
    if payload_warning and payload_warning not in warnings:
        warnings.append(payload_warning)
    return warnings


def _format_score_color(value: float) -> str:
    return "#16a34a" if value >= 0.10 else "#dc2626"


def _label_score_color(value: float) -> str:
    if value >= 0.60:
        return "#16a34a"
    if value >= 0.25:
        return "#f59e0b"
    return "#dc2626"


def _reasoning_score_color(value: float) -> str:
    if value >= 0.30:
        return "#16a34a"
    if value > 0.0:
        return "#f59e0b"
    return "#dc2626"


def _total_reward_color(value: float, has_submission: bool) -> str:
    if not has_submission:
        return "#6b7280"
    if value >= 0.70:
        return "#16a34a"
    if value > 0.0:
        return "#f59e0b"
    return "#dc2626"


def _reward_card_html(title: str, value: float, color: str, max_value: float) -> str:
    return (
        "<div style='background:#ffffff;border:1px solid #fde7cc;border-left:4px solid {color};"
        "border-radius:8px;padding:8px 10px;'>"
        "<div style='font-size:0.8rem;font-weight:600;color:#4b5563;'>{title}</div>"
        "<div style='font-size:1.1rem;font-weight:700;color:{color};'>{value:.2f}</div>"
        "<div style='font-size:0.72rem;color:#6b7280;'>max {max_value:.2f}</div>"
        "</div>"
    ).format(title=title, value=float(value), color=color, max_value=float(max_value))


def _reward_breakdown_markdown(obs: dict[str, Any], initialized: bool = False) -> str:
    info = obs.get("info", {}) if isinstance(obs, dict) else {}
    format_score = float(info.get("format_score", 0.0))
    label_score = float(info.get("label_score", 0.0))
    reasoning_score = float(info.get("reasoning_score", 0.0))
    total_reward = float(obs.get("reward", info.get("reward_total", 0.0)))
    running_total = float(info.get("total_reward", 0.0))
    best_score = float(info.get("best_score", 0.0))
    has_submission = bool(obs.get("history"))

    title = "### Session Initialized" if initialized else "### Latest Submission Reward Breakdown"
    subtitle = (
        "_Submit a classification action to populate live reward decomposition._"
        if initialized
        else "_Live reward decomposition for this submission._"
    )
    cards = (
        "<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;'>"
        f"{_reward_card_html('Format Score', format_score, _format_score_color(format_score), 0.10)}"
        f"{_reward_card_html('Label Score', label_score, _label_score_color(label_score), 0.60)}"
        f"{_reward_card_html('Reasoning Score', reasoning_score, _reasoning_score_color(reasoning_score), 0.30)}"
        "</div>"
    )
    total_color = _total_reward_color(total_reward, has_submission=has_submission)
    return "\n".join(
        [
            title,
            subtitle,
            cards,
            f"### **Total Reward:** <span style='color:{total_color};font-size:1.35rem;'>{total_reward:.3f}</span>",
            f"Current Best Score: **{best_score:.3f}**",
        ]
    )


def _format_baseline_result_markdown(
    payload: dict[str, Any] | None,
    status: str,
    error: dict[str, Any] | None = None,
) -> str:
    payload = payload if isinstance(payload, dict) else {}
    model_name = str(payload.get("model") or os.environ.get("MODEL_NAME", inference.DEFAULT_MODEL_NAME))
    raw_results = payload.get("baseline_results")
    results = [item for item in raw_results if isinstance(item, dict)] if isinstance(raw_results, list) else []
    lines = [f"- **Model:** `{model_name}`"]

    if status == "running":
        lines.append("- **Mode Used:** _Pending..._")
        lines.append("- **Reward achieved per step:** _Pending..._")
        lines.append("- **Success:** _Pending..._")
    else:
        lines.append(f"- **Mode Used:** {_summarize_mode(results)}")
        if "average_score" in payload and payload.get("average_score") is not None:
            lines.append(f"- **Average Score:** `{float(payload.get('average_score', 0.0)):.3f}`")
        if results:
            for result in results:
                task_name = str(result.get("task_id", "unknown_task"))
                steps = int(result.get("steps", 0) or 0)
                average_reward = float(result.get("reward", 0.0))
                total_reward = float(
                    result.get("total_reward", average_reward * steps if steps > 0 else average_reward)
                )
                final_reward = float(result.get("final_reward", average_reward))
                reward_per_step = total_reward / steps if steps > 0 else average_reward
                success = "Yes" if bool(result.get("success")) else "No"
                lines.append(
                    f"- **{task_name}**: {_mode_label(str(result.get('mode', 'fallback')))}; "
                    f"score `{reward_per_step:.3f}`; success **{success}**"
                )
        else:
            lines.append("- **Reward achieved per step:** _No baseline detail available._")
            lines.append("- **Success:** _No baseline detail available._")

    if status == "complete":
        lines.append("- **Status:** Complete")
    elif status == "failed":
        message = str((error or {}).get("message") or "").strip()
        suffix = f" — {message}" if message else ""
        lines.append(f"- **Status:** Failed{suffix}")
    elif status == "running":
        lines.append("- **Status:** Running")

    for warning in _baseline_warning_messages(payload):
        lines.append(f"> {warning}")
    return "\n".join(lines)


def _run_baseline_background(run_id: str, task: str | None = None, icl_memory: ICLMemory | None = None) -> None:
    try:
        raw = inference.run_baseline(task=task, icl_memory=icl_memory)
        baseline_run_tracker.mark_complete(run_id, _baseline_payload_from_raw(raw))
    except Exception as exc:
        baseline_run_tracker.mark_failed(run_id, {"message": str(exc)})


def _start_baseline_ui_run(task: str | None = None, icl_memory: ICLMemory | None = None) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    baseline_run_tracker.cleanup_expired()
    run_id = baseline_run_tracker.create_run()
    initial_payload = _initial_baseline_payload(task=task)
    baseline_run_tracker.update_partial(run_id, initial_payload)
    worker = threading.Thread(
        target=_run_baseline_background,
        args=(run_id, task, icl_memory),
        daemon=True,
        name=f"baseline-ui-{run_id[:8]}",
    )
    worker.start()
    return (
        run_id,
        "### LLM baseline run in progress...",
        _format_baseline_result_markdown(initial_payload, status="running"),
        gr.update(value=BASELINE_RUN_BUTTON_RUNNING_LABEL, interactive=False),
        gr.update(active=True),
    )


def _poll_baseline_ui_run(
    run_id: str | None,
) -> tuple[str | None, str, str, dict[str, Any], dict[str, Any]]:
    if not run_id:
        return (
            None,
            "### Baseline agent idle.",
            BASELINE_RESULT_PLACEHOLDER,
            gr.update(value=BASELINE_RUN_BUTTON_LABEL, interactive=True),
            gr.update(active=False),
        )

    baseline_run_tracker.cleanup_expired()
    run = baseline_run_tracker.get_run(run_id)
    if run is None:
        return (
            None,
            "### Baseline run unavailable (expired or unknown).",
            _format_baseline_result_markdown(_initial_baseline_payload(), status="failed"),
            gr.update(value=BASELINE_RUN_BUTTON_LABEL, interactive=True),
            gr.update(active=False),
        )

    status = _public_baseline_status(run["status"])
    payload = run["result"] if run["status"] == "completed" else run.get("partial")
    payload = payload if isinstance(payload, dict) else {}

    if status == "running":
        return (
            run_id,
            "### LLM baseline run in progress...",
            _format_baseline_result_markdown(payload, status="running"),
            gr.update(value=BASELINE_RUN_BUTTON_RUNNING_LABEL, interactive=False),
            gr.update(active=True),
        )

    if status == "failed":
        return (
            None,
            "### Baseline run failed.",
            _format_baseline_result_markdown(payload, status="failed", error=run.get("error")),
            gr.update(value=BASELINE_RUN_BUTTON_LABEL, interactive=True),
            gr.update(active=False),
        )

    average_score = payload.get("average_score")
    score_display = f"{float(average_score):.3f}" if average_score is not None else "N/A"
    return (
        None,
        f"### Baseline run complete (avg score: {score_display})",
        _format_baseline_result_markdown(payload, status="complete"),
        gr.update(value=BASELINE_RUN_BUTTON_LABEL, interactive=True),
        gr.update(active=False),
    )


def _enqueue_baseline_run(background_tasks: BackgroundTasks, task: str | None = None) -> dict[str, Any]:
    baseline_run_tracker.cleanup_expired()
    run_id = baseline_run_tracker.create_run()
    initial_payload = _initial_baseline_payload(task=task)
    baseline_run_tracker.update_partial(run_id, initial_payload)
    background_tasks.add_task(_run_baseline_background, run_id, task)
    return {
        "run_id": run_id,
        "status": "running",
        "payload": initial_payload,
        "status_url": f"/baseline/status/{run_id}",
    }


@app.post("/baseline/start")
def start_baseline_route(background_tasks: BackgroundTasks, task: str | None = None) -> dict[str, Any]:
    return _enqueue_baseline_run(background_tasks=background_tasks, task=task)


@app.get("/baseline/status/{run_id}")
def baseline_status_route(run_id: str) -> dict[str, Any]:
    baseline_run_tracker.cleanup_expired()
    run = baseline_run_tracker.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

    status = _public_baseline_status(run["status"])
    payload = run["result"] if run["status"] == "completed" else run["partial"]
    response: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "payload": payload if payload is not None else {},
    }
    if status == "failed":
        response["error"] = run["error"]
    return response


@app.get("/baseline")
def run_baseline_route(background_tasks: BackgroundTasks, task: str | None = None) -> dict[str, Any]:
    return _enqueue_baseline_run(background_tasks=background_tasks, task=task)


def build_custom_ui() -> gr.Blocks:
    # ── UI-only expected label lookup ──────────────────────────────────────
    # This is used EXCLUSIVELY for the human-facing dashboard tables.
    # It is completely isolated from the agent — the agent never calls this,
    # the observation pipeline never uses it, and the grader is untouched.
    _ui_gt = _UIGTStore()
    _ui_rubric = _UIRubricState()

    def _ui_expected_label(clip_id: str, clip: dict[str, Any] | None = None) -> str:
        """Resolve the expected label for UI display only.

        Checks GTStore first (human-assigned labels for hard clips),
        then falls back to rubric.derive_label() for easy/medium clips.
        This function is NEVER called by the agent or grader.
        """
        gt_label = _ui_gt.lookup(clip_id)
        if gt_label:
            return gt_label
        if clip:
            return _ui_rubric.derive_label(clip)
        return "N/A"

    dominant_feature_columns = [
        "Feature Name",
        "Current Value",
        "Rubric Status",
        "Threshold Range",
    ]
    session_history_columns = ["Step", "Clip ID", "Submitted Label", "Expected Label", "Reward"]
    corpus_columns = ["Clip ID", "Expected Label", "Predicted Label", "Current Review Status", "Face Confidence", "Motion Score", "Audio SNR (dB)"]

    def format_dominant_features(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=dominant_feature_columns)
        return pd.DataFrame(rows, columns=dominant_feature_columns)

    def format_obs(obs: dict[str, Any], predicted_labels: dict[str, str] | None = None) -> tuple[pd.DataFrame, str, float, int, str, str]:
        if not obs:
            return (
                pd.DataFrame(columns=corpus_columns),
                "### No Clip-Quality Rubric Available",
                0.0,
                5,
                "N/A",
                "### Clip Queue: **0** of **0** items displayed",
            )

        corpus_items = list(obs.get("data_corpus", []))
        corpus_items.sort(key=lambda item: str(item.get("clip_id", item.get("id", ""))))
        _predicted = predicted_labels or {}

        corpus_data = []
        for item in corpus_items:
            cid = item.get("clip_id", item.get("id", "N/A"))
            # Look up expected label from UI-only GT + rubric (isolated from agent)
            expected = _ui_expected_label(str(cid), clip=item)
            corpus_data.append(
                {
                    "Clip ID": cid,
                    "Expected Label": expected,
                    "Predicted Label": _predicted.get(str(cid), "—"),
                    "Current Review Status": item.get("review_status", "pending"),
                    "Face Confidence": item.get("face_confidence", "N/A"),
                    "Motion Score": item.get("motion_score", "N/A"),
                    "Audio SNR (dB)": item.get("audio_snr_db", "N/A"),
                }
            )
        df_corpus = pd.DataFrame(corpus_data) if corpus_data else pd.DataFrame(columns=corpus_columns)

        rubric_summary = obs.get("rubric_summary", "").strip()
        rule_md = "### Active Clip-Quality Rubric\n"
        if rubric_summary:
            rule_md += f"```\n{rubric_summary}\n```"
        else:
            rule_md += "_No rubric summary available._"

        best_score = float(obs.get("info", {}).get("best_score", 0.0))
        steps_left = int(obs.get("info", {}).get("steps_remaining", 0))
        episode_id = str(obs.get("episode_id", "N/A"))[:8]
        shown = int(obs.get("corpus_shown", len(corpus_data)))
        total = int(obs.get("corpus_size", len(corpus_data)))
        corpus_stat = f"### Clip Queue: **{shown}** of **{total}** items displayed"
        return df_corpus, rule_md, best_score, steps_left, episode_id, corpus_stat

    def _extract_baseline_predicted_labels(payload: dict[str, Any]) -> dict[str, str]:
        """Extract per-clip predicted labels from baseline result detail."""
        labels: dict[str, str] = {}
        raw_results = payload.get("baseline_results") or []
        # baseline_results may contain per-task detail including steps
        # Each result item has task_id, and the episode ran EPISODE_STEPS steps.
        # We store predicted labels from action_history where available.
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            # action_history stores the string labels in step order
            action_history = result.get("action_history") or []
            task_label = str(result.get("task_id", ""))
            for idx, lbl in enumerate(action_history):
                labels[f"{task_label}:step{idx+1}"] = str(lbl).upper()
        return labels

    def format_session_history(obs: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
        info = obs.get("info", {}) if isinstance(obs, dict) else {}
        raw_history = info.get("session_history", [])
        session_history = raw_history if isinstance(raw_history, list) else []
        rows: list[dict[str, Any]] = []
        cue_lines: list[str] = []

        for idx, item in enumerate(session_history, start=1):
            step = int(item.get("step", idx))
            clip_id = str(item.get("clip_id", "N/A"))
            submitted_raw = item.get("label", "")
            submitted = str(submitted_raw).upper() if submitted_raw is not None else ""
            # Look up expected label from UI-only GT + rubric (isolated from agent)
            expected = _ui_expected_label(clip_id)
            reward = float(item.get("reward", 0.0))
            rows.append(
                {
                    "Step": step,
                    "Clip ID": clip_id,
                    "Submitted Label": submitted,
                    "Expected Label": expected,
                    "Reward": reward,
                }
            )
            is_match = bool(submitted) and expected not in {"", "N/A"} and submitted == expected
            badge = (
                "<span style='color:#2e7d32; font-weight:700;'>Match</span>"
                if is_match
                else "<span style='color:#c62828; font-weight:700;'>Mismatch</span>"
            )
            cue_lines.append(f"- Step {step} (`{clip_id}`): {badge}")

        history_df = (
            pd.DataFrame(rows, columns=session_history_columns)
            if rows
            else pd.DataFrame(columns=session_history_columns)
        )
        history_cues_md = "### Match Results\n" + "\n".join(cue_lines) if cue_lines else "### Match Results\n_No actions yet._"

        total_reward = float(info.get("total_reward", 0.0))
        total_color = "#2e7d32" if total_reward > 0 else "#c62828" if total_reward < 0 else "#374151"
        history_total_md = ""
        return history_df, history_cues_md, history_total_md

    def _resolve_env(env_state: ClipQualityEnvironment | None) -> ClipQualityEnvironment:
        return env_state if isinstance(env_state, ClipQualityEnvironment) else ClipQualityEnvironment()

    def _resolve_memory(mem_state: Any) -> ICLMemory:
        return mem_state if isinstance(mem_state, ICLMemory) else ICLMemory()

    def _learning_progress_df(icl_mem: ICLMemory) -> pd.DataFrame:
        """Build the Learning Progress DataFrame from session ICL memory.

        NOTE: 'Expected' column intentionally absent — that would expose ground truth.
        Correctness is inferred from label_score (>= 0.55 = exact, 0.10-0.54 = partial).
        """
        learning_columns = [
            "Clip ID", "Runs", "Exact/Partial", "Best Reward",
            "Latest Reward", "Best Label", "Best label_score", "Trend",
        ]
        rows = icl_mem.all_clip_summary()
        if not rows:
            return pd.DataFrame(columns=learning_columns)
        return pd.DataFrame(rows, columns=learning_columns)


    def handle_reset(env_state: ClipQualityEnvironment | None, task_id: str):
        env = _resolve_env(env_state)
        obs = env.reset(task_id=task_id).model_dump()
        df, pol, score, steps, ep, stat = format_obs(obs)
        dominant_df = format_dominant_features(env.dominant_feature_rows())
        history_df, history_cues, history_total = format_session_history(obs)
        reward_msg = _reward_breakdown_markdown(obs, initialized=True)
        return (
            env,
            df,
            pol,
            dominant_df,
            score,
            steps,
            ep,
            stat,
            reward_msg,
            history_df,
            history_cues,
            history_total,
            json.dumps(obs, indent=2),
        )

    def handle_step(
        env_state: ClipQualityEnvironment | None,
        task_id: str,
        easy_observation: str,
        medium_primary_signal: str,
        medium_conflicting_signal: str,
        medium_reasoning: str,
        hard_tradeoff_summary: str,
        hard_confidence_justification: str,
        hard_confidence: float,
        clip_id: str,
        icl_memory_state: Any,
    ):
        """
        Execute Strategic Step with full ICL-RL loop.

        For each of the 5 episode clips:
          1. Auto-generate a memory-aware quality hint
          2. Call agent.act() with ICL context from session memory
          3. Submit the action to the environment
          4. Record reward + label to session memory for next-run improvement
        """
        import copy as _copy
        env = _resolve_env(env_state)
        icl_mem = _resolve_memory(icl_memory_state)

        # Always reset so we run a clean episode
        obs = env.reset(task_id=task_id)
        obs_dict = obs.model_dump()

        _agent = inference.ClipQualityAgent(client=None, model=inference.DEFAULT_MODEL_NAME)

        predicted_labels: dict[str, str] = {}

        for step_idx in range(len(env._episode_plan)):
            current_clip = (
                env._episode_plan[env.state.step_count].clip
                if env.state.step_count < len(env._episode_plan)
                else env._episode_plan[-1].clip
            )
            clip_id_val = str(current_clip.get("clip_id", ""))

            # Memory-aware quality hint (appends prior failure feedback)
            hint_text = env.build_quality_hint(
                clip=_copy.deepcopy(current_clip),
                icl_memory=icl_mem,
            )

            # Full ICL-RL act(): uses memory context + grader-aligned reasoning
            action_dict = _agent.act(
                task_id,
                obs_dict,
                icl_memory=icl_mem,
                quality_hint=hint_text,
            )
            # Override clip_id override only on first step if provided
            if clip_id and clip_id.strip() and step_idx == 0:
                action_dict["clip_id"] = clip_id.strip()
            else:
                action_dict.setdefault("clip_id", clip_id_val)

            obs_obj = env.step(Action.model_validate(action_dict))
            obs_dict = obs_obj.model_dump()
            reward = float(obs_obj.reward)

            # Record to session ICL memory — use raw label_score only.
            # DO NOT pass expected_label: ICLMemory must never store ground-truth.
            # The agent learns from label_score (0.0=wrong, 0.05–0.25=partial, 0.60=correct)
            # and the total reward signal, not from the answer.
            raw_label_score = float(obs_obj.info.get("label_score", 0.0))
            icl_mem.record(
                clip_id=clip_id_val,
                label=action_dict["label"],
                reward=reward,
                reasoning=str(action_dict.get("reasoning", "")),
                episode=icl_mem.episode_count,
                step=step_idx + 1,
                label_score=raw_label_score,
            )
            predicted_labels[clip_id_val] = action_dict["label"]

            if obs_obj.done:
                break

        icl_mem.increment_episode()

        obs = obs_dict
        df, pol, score, steps, ep, stat = format_obs(obs, predicted_labels=predicted_labels)
        dominant_df = format_dominant_features(env.dominant_feature_rows())
        history_df, history_cues, history_total = format_session_history(obs)
        reward_msg = _reward_breakdown_markdown(obs, initialized=False)
        learning_df = _learning_progress_df(icl_mem)
        return (
            env,
            df,
            pol,
            dominant_df,
            score,
            steps,
            ep,
            stat,
            reward_msg,
            history_df,
            history_cues,
            history_total,
            json.dumps(obs, indent=2),
            icl_mem,
            learning_df,
        )

    def handle_quality_hint(
        env_state: ClipQualityEnvironment | None,
        task_id: str,
        selected_input_tab: str,
        easy_observation: str,
        medium_reasoning: str,
        hard_tradeoff_summary: str,
        icl_memory_state: Any,
    ) -> tuple[ClipQualityEnvironment, str, str, str]:
        env = _resolve_env(env_state)
        icl_mem = _resolve_memory(icl_memory_state)
        if env.state.task_id != task_id or not env.state.current_clip_id:
            env.reset(task_id=task_id)

        hint_text = env.build_quality_hint(icl_memory=icl_mem)
        active_tab = selected_input_tab if selected_input_tab in {"easy", "medium", "hard"} else _input_tab_for_task(task_id)

        next_easy_observation = easy_observation
        next_medium_reasoning = medium_reasoning
        next_hard_tradeoff_summary = hard_tradeoff_summary
        if active_tab == "easy":
            next_easy_observation = hint_text
        elif active_tab == "medium":
            next_medium_reasoning = hint_text
        else:
            next_hard_tradeoff_summary = hint_text

        return env, next_easy_observation, next_medium_reasoning, next_hard_tradeoff_summary

    def _baseline_predicted_labels_from_payload(payload: dict[str, Any], env: ClipQualityEnvironment | None) -> dict[str, str]:
        """Build per clip_id predicted-label map from baseline result using clip_ids returned by run_episode."""
        labels: dict[str, str] = {}
        if env is None:
            return labels
        raw_results = (payload or {}).get("baseline_results") or []
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            task_id_r = str(result.get("task_id", ""))
            action_history = result.get("action_history") or []
            clip_ids_list = result.get("clip_ids") or []
            if clip_ids_list and action_history:
                # Preferred: clip IDs returned directly from run_episode
                for cid, lbl in zip(clip_ids_list, action_history):
                    labels[str(cid)] = str(lbl).upper()
            else:
                # Fallback: match by corpus ordering
                corpus = env._episode_corpus.get(task_id_r, [])
                for idx, item in enumerate(corpus):
                    clip_id_val = str(item.get("clip_id", item.get("id", "")))
                    if idx < len(action_history):
                        labels[clip_id_val] = str(action_history[idx]).upper()
                    else:
                        _agent = inference.ClipQualityAgent(client=None, model=inference.DEFAULT_MODEL_NAME)
                        labels[clip_id_val] = _agent._heuristic_label(item)
        # If no labels resolved, fall back to heuristic on all episode plan clips
        if not labels and env._episode_plan:
            _agent = inference.ClipQualityAgent(client=None, model=inference.DEFAULT_MODEL_NAME)
            for ep_clip in env._episode_plan:
                cid = str(ep_clip.clip.get("clip_id", ""))
                labels[cid] = _agent._heuristic_label(ep_clip.clip)
        return labels

    with gr.Blocks(
        title="ClipQualityEnv — Talking-Head Clip Quality Assessment",
        theme=gr.themes.Soft(
            primary_hue="orange",
            font=gr.themes.GoogleFont("Source Sans Pro"),
        ),
        css="""
        .dark {
            background-color: #0c0c0c !important;
        }
        """,
    ) as demo:
        env_state = gr.State(value=None)
        icl_memory_state = gr.State(value=None)   # per-session ICLMemory
        baseline_run_id_state = gr.State(value=None)
        selected_tab_state = gr.State(value="easy")
        baseline_poll_timer = gr.Timer(value=1.0, active=False)

        gr.HTML("<h2 style='text-align: center; color: #10b981;'>ClipQualityEnv — Talking-Head Clip Quality Assessment</h2>")
        gr.Markdown(
            "An RL environment for autonomous video-clip quality classification. "
            "The agent learns to label talking-head clips as **KEEP**, **BORDERLINE**, or **REJECT** "
            "using in-context reinforcement learning — no ground-truth labels are exposed to the agent."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Scenario Metrics")
                with gr.Group():
                    best_score_disp = gr.Number(label="Environment Best Score", value=0.0, interactive=False)
                    steps_left_disp = gr.Number(label="Remaining Execution Steps", value=5, interactive=False)
                    episode_disp = gr.Textbox(label="Active Episode ID", value="N/A", interactive=False)
                
                reward_outcome_disp = gr.Markdown("### Awaiting Scenario...")
                
                with gr.Group():
                    task_id = gr.Dropdown(choices=list(TASK_REGISTRY.keys()), value="task_easy", label="Deployment Scenario")
                    reset_btn = gr.Button("Initialize Scenario", variant="secondary")

            with gr.Column(scale=3):
                corpus_count_disp = gr.Markdown("### Corpus: 0 of 0 incidents displayed")
                with gr.Tabs():
                    with gr.Tab("Data Corpus (Tabular View)"):
                        corpus_table = gr.DataFrame(
                            label="Sampled Posts and System Actions",
                            interactive=False,
                        )
                    with gr.Tab("Active Framework"):
                        policy_display = gr.Markdown("Initialize to view the current rubric summary.")
                    with gr.Tab("Diagnostic JSON"):
                        raw_json_box = gr.Code(label="Environment Raw Response", language="json", interactive=False)
                    with gr.Tab("Session History"):
                        session_history_table = gr.DataFrame(
                            value=pd.DataFrame(columns=session_history_columns),
                            label="Per-step Classification History",
                            headers=session_history_columns,
                            col_count=(len(session_history_columns), "fixed"),
                            datatype=["number", "str", "str", "str", "number"],
                            interactive=False,
                        )
                        session_history_cues = gr.Markdown("### Match Results\n_No actions yet._")
                        session_total_reward = gr.Markdown("", visible=False)
                    with gr.Accordion("LLM Baseline Result", open=False):
                        baseline_run_btn = gr.Button(BASELINE_RUN_BUTTON_LABEL, variant="secondary")
                        baseline_status_disp = gr.Markdown("### Baseline agent idle.")
                        baseline_result_md = gr.Markdown(BASELINE_RESULT_PLACEHOLDER)
                
                dominant_features_table = gr.DataFrame(
                    value=pd.DataFrame(columns=dominant_feature_columns),
                    label="Feature Focus Table",
                    headers=dominant_feature_columns,
                    column_count=len(dominant_feature_columns),
                    column_limits=(len(dominant_feature_columns), len(dominant_feature_columns)),
                    datatype=["str", "number", "str", "str"],
                    interactive=False,
                    visible=False
                )

        gr.Markdown("---")
        with gr.Accordion("📈 ICL Learning Progress", open=False):
            gr.Markdown(
                "_Per-clip reward history accumulated in this session. "
                "Each time you run Execute Strategic Step or the Baseline Agent, "
                "predictions are recorded and the agent learns from prior rewards._"
            )
            learning_progress_table = gr.DataFrame(
                label="Session Learning History",
                headers=["Clip ID", "Runs", "Exact/Partial", "Best Reward", "Latest Reward", "Best Label", "Best label_score", "Trend"],
                interactive=False,
            )

        gr.Markdown("---")
        gr.Markdown("### Propose Strategic Refinement")
        gr.Markdown(
            "_Click **Execute Strategic Step** to auto-generate quality hints for all clips and predict labels automatically. "
            "Use **Load Quality Hint** (optional) to manually preview the hint for the current clip before running._"
        )
        with gr.Group():
            with gr.Tabs(selected="easy") as tiered_input_tabs:
                with gr.Tab("Easy: Definition Refining", id="easy"):
                    easy_observation_input = gr.Textbox(label="Key Observation (auto-filled by quality hint)", lines=2, placeholder="Auto-generated or manually entered hint for the current clip.")
                with gr.Tab("Medium: Gap Detection", id="medium"):
                    medium_primary_signal_input = gr.Textbox(label="Primary Signal", lines=1)
                    medium_conflicting_signal_input = gr.Textbox(label="Conflicting Signal", lines=1)
                    medium_reasoning_input = gr.TextArea(label="Reasoning (auto-filled by quality hint)", lines=4, placeholder="Auto-generated or manually entered reasoning.")
                with gr.Tab("Hard: Full System Evolution", id="hard"):
                    hard_tradeoff_summary_input = gr.TextArea(label="Trade-off Summary (auto-filled by quality hint)", lines=4, placeholder="Auto-generated or manually entered trade-off summary.")
                    with gr.Row():
                        hard_confidence_justification_input = gr.TextArea(label="Confidence Justification", lines=3, scale=2)
                        hard_confidence_input = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.01,
                            label="Confidence",
                            scale=1
                        )

            clip_id_input = gr.Textbox(label="Clip ID override (optional)")
            hint_btn = gr.Button(QUALITY_HINT_BUTTON_LABEL, variant="secondary")

        step_btn = gr.Button("Execute Strategic Step", variant="primary")

        def sync_input_tab_for_task(selected_task_id: str):
            tab_id = _input_tab_for_task(selected_task_id)
            return _input_tab_update_for_task(selected_task_id), tab_id

        def on_tab_select(evt: gr.SelectData) -> str:
            return str(evt.value) if evt and evt.value else "easy"

        task_id.change(
            sync_input_tab_for_task,
            inputs=[task_id],
            outputs=[tiered_input_tabs, selected_tab_state],
        )
        tiered_input_tabs.select(on_tab_select, inputs=None, outputs=[selected_tab_state])

        reset_btn.click(
            handle_reset,
            inputs=[env_state, task_id],
            outputs=[
                env_state,
                corpus_table,
                policy_display,
                dominant_features_table,
                best_score_disp,
                steps_left_disp,
                episode_disp,
                corpus_count_disp,
                reward_outcome_disp,
                session_history_table,
                session_history_cues,
                session_total_reward,
                raw_json_box,
            ],
        )
        step_btn.click(
            handle_step,
            inputs=[
                env_state,
                task_id,
                easy_observation_input,
                medium_primary_signal_input,
                medium_conflicting_signal_input,
                medium_reasoning_input,
                hard_tradeoff_summary_input,
                hard_confidence_justification_input,
                hard_confidence_input,
                clip_id_input,
                icl_memory_state,
            ],
            outputs=[
                env_state,
                corpus_table,
                policy_display,
                dominant_features_table,
                best_score_disp,
                steps_left_disp,
                episode_disp,
                corpus_count_disp,
                reward_outcome_disp,
                session_history_table,
                session_history_cues,
                session_total_reward,
                raw_json_box,
                icl_memory_state,
                learning_progress_table,
            ],
        )
        hint_btn.click(
            handle_quality_hint,
            inputs=[
                env_state,
                task_id,
                selected_tab_state,
                easy_observation_input,
                medium_reasoning_input,
                hard_tradeoff_summary_input,
                icl_memory_state,
            ],
            outputs=[
                env_state,
                easy_observation_input,
                medium_reasoning_input,
                hard_tradeoff_summary_input,
            ],
        )
        def _start_baseline_with_corpus(task_id_val: str, env_st: ClipQualityEnvironment | None, icl_mem_st: Any):
            icl_mem = _resolve_memory(icl_mem_st)
            run_id, status_md, result_md, btn_upd, timer_upd = _start_baseline_ui_run(task_id_val, icl_mem)
            return run_id, status_md, result_md, btn_upd, timer_upd, env_st, gr.update(), icl_mem

        def _poll_baseline_with_corpus(
            run_id: str | None,
            env_st: ClipQualityEnvironment | None,
            current_corpus_df: pd.DataFrame | None,
            icl_mem_st: Any,
        ):
            icl_mem = _resolve_memory(icl_mem_st)
            run_id_out, status_md, result_md, btn_upd, timer_upd = _poll_baseline_ui_run(run_id)
            # When complete, compute predicted labels and update corpus + learning tables
            if run_id_out is None and env_st is not None:
                payload: dict[str, Any] = {}
                if run_id:
                    from server.baseline_runs import baseline_run_tracker as _brt
                    run = _brt.get_run(run_id)
                    if run and run["status"] == "completed":
                        payload = run["result"] or {}
                pred_labels = _baseline_predicted_labels_from_payload(payload, env_st)
                if pred_labels and current_corpus_df is not None and not current_corpus_df.empty:
                    df_upd = current_corpus_df.copy()
                    if "Predicted Label" not in df_upd.columns:
                        df_upd["Predicted Label"] = "—"
                    for idx, row in df_upd.iterrows():
                        cid = str(row.get("Clip ID", ""))
                        if cid in pred_labels:
                            df_upd.at[idx, "Predicted Label"] = pred_labels[cid]
                    learning_df = _learning_progress_df(icl_mem)
                    return run_id_out, status_md, result_md, btn_upd, timer_upd, env_st, gr.update(value=df_upd), icl_mem, learning_df
            learning_df = _learning_progress_df(icl_mem)
            return run_id_out, status_md, result_md, btn_upd, timer_upd, env_st, gr.update(), icl_mem, learning_df

        baseline_run_btn.click(
            _start_baseline_with_corpus,
            inputs=[task_id, env_state, icl_memory_state],
            outputs=[baseline_run_id_state, baseline_status_disp, baseline_result_md, baseline_run_btn, baseline_poll_timer, env_state, corpus_table, icl_memory_state],
        )
        baseline_poll_timer.tick(
            _poll_baseline_with_corpus,
            inputs=[baseline_run_id_state, env_state, corpus_table, icl_memory_state],
            outputs=[baseline_run_id_state, baseline_status_disp, baseline_result_md, baseline_run_btn, baseline_poll_timer, env_state, corpus_table, icl_memory_state, learning_progress_table],
        )

    return demo


custom_demo = build_custom_ui()
app = gr.mount_gradio_app(app, custom_demo, path="/dashboard/")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)

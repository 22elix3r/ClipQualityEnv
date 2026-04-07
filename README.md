---
title: CLIP Quality Analyzer
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 8000
base_path: /dashboard/
tags:
  - openenv
  - reinforcement-learning
  - clip-quality
  - quality-analysis
---

# CLIP Quality Analyzer Environment

ClipQualityEnv is an OpenEnv-compliant RL environment for CLIP quality analysis workflows. It keeps the reference OpenEnv structure while presenting a clip-quality review and classification surface.

## Action Space

Classifier action payload:

- `label`: one of `KEEP`, `BORDERLINE`, `REJECT`
- `reasoning`: concise clip-metadata-grounded explanation
- `confidence`: float in `[0.0, 1.0]`
- `clip_id` (optional): specific clip target

## Observation Space

Each observation includes:
- `task_id`, `episode_id`, `step_count`
- `data_corpus`, `clip_metadata`, `rubric_summary`
- `reward`, `done`, and step diagnostics in `info`

These fields are preserved for OpenEnv/reference compatibility and surfaced as clip-case analysis artifacts in `/dashboard/`.

## Tasks

| Task ID | Difficulty | Description |
|---|---|---|
| `task_easy` | Easy | Classify clips with clear quality signals |
| `task_medium` | Medium | Classify borderline clips with mixed indicators |
| `task_hard` | Hard | Classify hard clips with conflicting quality signals |

Overall total scores are difficulty-calibrated so task-level averages follow: `hard > medium > easy`.

## API + Dashboard Surface

- `GET /` -> JSON status
- `GET /health`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `POST /baseline/start`
- `GET /baseline/status/{run_id}`
- `GET /baseline` (compatibility wrapper for async baseline start)
- OpenEnv endpoints: `/reset`, `/step`, `/ws`, `/metadata`, `/schema`
- Gradio dashboard: `/dashboard/`

The API surface remains reference-compatible; product naming and UI wording are clip-quality focused.

`POST /grader` accepts the classifier action payload above.

## Dashboard Highlights

- 5-step same-scenario review sessions with running totals
- Full corpus queue display (no slicing), sorted by Clip ID, with live review-status updates
- Difficulty-tiered input tabs (Easy / Medium / Hard) with scenario-aware tab switching
- `💡 Load Quality Hint` helper button to scaffold reasoning text from dominant feature boundary cues
- Session history tab with submitted vs expected labels and per-step rewards
- Non-blocking `🤖 Run LLM Baseline Agent` control with async status polling and result summary

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
PYTHONPATH=. python -m pytest -q
```

Run server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Baseline Inference

Set env vars:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_token"
```

Run:

```bash
python inference.py
```

Optional single task:

```bash
python inference.py task_easy
```

## Hugging Face Space Deployment

```bash
openenv push --repo-id elix3r/ClipQualityEnv .
```

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

# ClipQualityEnv

An OpenEnv-compliant reinforcement learning environment for automated talking-head clip quality assessment. The agent learns to classify video clips as KEEP, BORDERLINE, or REJECT by comparing clip metadata against a versioned rubric, with rewards that improve through In-Context Learning (ICL) across episode steps.

## What it does

The environment presents an LLM agent with a 5-step episode. Each step shows one clip's metadata, a rubric summary, and the agent's prior prediction history. The agent classifies the clip and receives a structured reward signal decomposed into three components:

- **Format score** (max 0.10): validates that the label, reasoning, and confidence are all well-formed
- **Label score** (max 0.60): checks label correctness against ground truth or rubric-derived labels
- **Reasoning score** (max 0.30): checks that the reasoning mentions dominant features with directional language and contains no hallucinated feature names

All three difficulty levels (easy, medium, hard) are graded on the same 0 to 1 scale. A perfect submission across all components scores 1.0 regardless of task difficulty.

## Architecture

```
clip_quality_env/
  env.py             # OpenEnv environment, episode management, GT promotion
  grader.py          # Deterministic reward decomposition (format + label + reasoning)
  rubric.py          # Versioned threshold definitions and feature status logic
  ground_truth.py    # Append-only GT store with agent-promotion support
  icl_memory.py      # Per-session ICL memory, context injection, hint feedback
  agent.py           # Lightweight LLM agent (XML tag parser, used in standalone mode)
  difficulty.py      # Difficulty normalization utilities
  models.py          # Pydantic models: Action, Observation, State, Reward, ClipMetadata
  real_clips.py      # Real clip manifest loader

server/
  app.py             # FastAPI application, Gradio dashboard, baseline runner
  grader.py          # /grader endpoint handler
  tasks/             # Task registry: task_easy, task_medium, task_hard

inference.py         # ClipQualityAgent with ICL-RL loop, CLI baseline runner
scripts/
  extract_mp4_metadata.py  # Extracts clip features from MP4 files into manifest
data/
  real_clips_manifest.jsonl  # Per-clip metadata extracted from real video files
  seed_gt.json               # Seed ground truth labels
```

## Clip Metadata Features

Each clip observation exposes the following fields:

| Feature | Description |
|---|---|
| `face_area_ratio` | Fraction of frame occupied by the detected face |
| `face_confidence` | Face detection confidence score |
| `head_pose_yaw_deg` | Head yaw angle in degrees |
| `motion_score` | Frame-to-frame motion intensity |
| `bg_complexity_score` | Background visual complexity |
| `audio_snr_db` | Audio signal-to-noise ratio |
| `duration_s` | Clip length in seconds |
| `mouth_open_ratio` | Ratio of mouth openness |
| `lighting_uniformity` | Consistency of lighting across the frame |
| `sharpness_score` | Frame sharpness estimate |
| `temporal_flicker` | Frame-to-frame brightness flicker |
| `bg_entropy` | Background entropy |
| `eye_contact_ratio` | Fraction of frames with estimated eye contact |
| `speech_rate_wpm` | Estimated words per minute |

## Rubric and Grading

The rubric defines per-feature thresholds with three modes: `higher` (feature should be above threshold to KEEP), `lower` (feature should be below threshold to KEEP), and `band` (feature should fall within a range to KEEP). The rubric is versioned and can tighten automatically over time as the agent's accuracy on easy and medium tasks improves.

Labels are derived from the rubric when no explicit ground truth exists. An agent-predicted label can be promoted into the ground truth store if it achieves reward >= 0.85 and confidence >= 0.80 and matches any existing expected label.

## In-Context Learning

The `ICLMemory` class tracks per-clip prediction history within a session. After each step, the agent's label, raw label score, and reasoning are recorded. On subsequent attempts at the same clip:

- If a prior attempt scored 0.60 (exact match), the agent is directed to keep that label and refine reasoning
- If a prior attempt scored 0.25 (one tier off), the agent is directed to try an adjacent label
- If a prior attempt scored 0.00 (wrong), the agent is told to try a completely different label

The memory never reveals the expected label. All feedback is based on the reward signal alone.

## Tasks

| Task ID | Difficulty | Description |
|---|---|---|
| `task_easy` | Easy | Clear, unambiguous quality signals across most features |
| `task_medium` | Medium | Mixed indicators requiring trade-off reasoning |
| `task_hard` | Hard | Conflicting signals with no dominant clear indicator |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Status and environment metadata |
| `GET` | `/health` | Health check |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | List all tasks with action schema |
| `POST` | `/grader` | Score a single action against a task |
| `POST` | `/baseline/start` | Start an async baseline run |
| `GET` | `/baseline/status/{run_id}` | Poll a running baseline |
| `GET` | `/baseline` | Alias for baseline start (GET-compatible) |
| `POST` | `/reset` | Reset the environment to a new episode |
| `POST` | `/step` | Submit one action and advance the episode |
| `GET` | `/metadata` | OpenEnv environment metadata |
| `GET` | `/schema` | OpenEnv action/observation schema |
| `GET` | `/dashboard/` | Gradio interactive dashboard |

The `/grader` endpoint accepts the same action schema as `/step`:

```json
{
  "label": "KEEP",
  "reasoning": "face_confidence is 0.91, above the KEEP threshold (0.80). motion_score is 0.12, stable below the KEEP ceiling (0.25).",
  "confidence": 0.85,
  "clip_id": "clip_001"
}
```

## Dashboard

The Gradio dashboard at `/dashboard/` provides a full interactive session:

- Difficulty-tiered input tabs (Easy, Medium, Hard) with structured reasoning fields
- Live clip corpus queue sorted by clip ID with predicted and expected labels
- Dominant feature table showing closest-boundary features and their rubric status
- Reward breakdown cards for format, label, and reasoning scores after each submission
- Session history table with submitted vs expected labels and per-step rewards
- Learning progress panel tracking reward trends across ICL runs per clip
- "Load Quality Hint" button that generates a pre-filled hint from rubric thresholds for the current clip
- "Run LLM Baseline Agent" button that runs the full ICL-RL agent in a background thread with async polling

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the server:

```bash
PYTHONPATH=. python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run tests:

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

## Baseline Inference

The baseline agent runs all three tasks in sequence and prints step-level rewards. It uses the LLM if a token is available, otherwise falls back to a deterministic heuristic with grader-aligned reasoning.

Set environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_token_here"
```

Run all tasks:

```bash
PYTHONPATH=. python inference.py
```

Run a single task:

```bash
PYTHONPATH=. python inference.py task_easy
```

Output format (`--output json` for machine-readable):

```
[START] task=task_easy env=ClipQualityEnv model=llama-3.3-70b-versatile mode=llm
[STEP] step=1 label=KEEP reward=0.80 done=false error=null
...
[END] success=true steps=5 score=0.812 total_reward=4.060 final_reward=0.90 rewards=0.80,...
```

## Extracting Real Clip Metadata

To build a real clip manifest from MP4 files:

```bash
pip install -r requirements_extractor.txt
PYTHONPATH=. python scripts/extract_mp4_metadata.py path/to/clips/ --output data/real_clips_manifest.jsonl
```

The manifest is loaded at startup if present at `data/real_clips_manifest.jsonl`. If missing, the environment falls back to the static task corpora.

## Docker

```bash
docker build -f server/Dockerfile -t clip-quality-env .
docker run -p 8000:8000 -e HF_TOKEN=your_token clip-quality-env
```

## Deployment

The project is deployed as a Hugging Face Space using the Docker SDK. The `openenv.yaml` and HuggingFace Space frontmatter in this file configure the deployment.

```bash
git remote add hf-space https://huggingface.co/spaces/your-username/ClipQualityEnv
git push hf-space main
```

## Dependencies

- `openenv-core >= 0.2.3`: OpenEnv environment base classes and FastAPI server factory
- `fastapi >= 0.104.0` + `uvicorn >= 0.24.0`: HTTP server
- `gradio >= 4.0.0`: Interactive dashboard
- `openai >= 1.0.0`: OpenAI-compatible client (used with HuggingFace inference router)
- `pydantic >= 2.0.0`: Data validation and serialization
- `opencv-python-headless >= 4.10.0`: Video processing for metadata extraction
- `numpy >= 1.26.0`: Numerical operations in extraction pipeline
- `pandas >= 2.0.0`: DataFrame rendering in the dashboard

## License

MIT

---

## Academic References

ClipQualityEnv draws from several foundational research areas. The connections below tie each paper directly to specific components of the implementation.

### Curriculum Learning

| Paper | Year | Relevance |
|-------|------|-----------|
| Bengio et al. "Curriculum Learning" | 2009 | Foundation for the Easy → Medium → Hard task progression. Key insight: ordering training samples by difficulty accelerates learning and improves convergence. |
| Graves et al. "Automated Curriculum Learning for Neural Networks" | 2017 | Adaptive curriculum where difficulty self-adjusts based on learner performance. Directly matches the `recalibrate()` logic in `rubric.py`, which tightens thresholds as the agent's accuracy on easier tasks improves. |
| Kumar et al. "Self-Paced Learning with Diversity" | 2010 | Agent chooses its own curriculum pace. The confidence-weighted GT promotion in `try_promote()` is a form of self-pacing — the agent only promotes predictions it is confident in. |

**Application in this environment:** The 3-task difficulty progression implements curriculum learning at the task level. Rubric calibration (`recalibrate()`) implements it across episodes — the environment automatically gets harder as the agent succeeds on simpler clips.

---

### Active Learning & Self-Training

| Paper | Year | Relevance |
|-------|------|-----------|
| Culotta & McCallum "Confidence-Weighted Active Learning" | 2005 | Selectively promote high-confidence predictions to the training set. Direct precedent for `GTStore.try_promote()`, which requires `reward >= 0.85` and `confidence >= 0.80` before accepting a new ground-truth label. |
| Zhu et al. "Semi-Supervised Learning with Graphs" | 2003 | Self-training expands the labeled set iteratively with the model's own confident predictions. The GT expansion flywheel (more promoted clips → richer GT store → better grading signal) follows this pattern. |
| Settles "Active Learning Literature Survey" | 2010 | Comprehensive overview of query strategies including uncertainty sampling. ClipQualityEnv inverts uncertainty sampling: rather than querying uncertain examples for human labeling, it promotes *certain* agent predictions into the GT store. |

**Application in this environment:** GT expansion via `try_promote()` is active learning in reverse. The agent autonomously extends the ground-truth store by promoting high-confidence, high-reward predictions, progressively replacing rubric-derived labels with agent-confirmed ones.

---

### Preference Optimization

| Paper | Year | Relevance |
|-------|------|-----------|
| Rafailov et al. "Direct Preference Optimization (DPO)" | 2023 | Preference-based training without explicit reward models. Partial label credit on BORDERLINE cases mirrors the preference pair structure — a KEEP prediction on a BORDERLINE clip is treated as a *useful* signal, not a hard failure. |
| Christiano et al. "Deep RL from Human Preferences" | 2017 | RLHF foundation. ClipQualityEnv replaces human preference comparisons with a fully verifiable reward function, retaining the reward decomposition insight while eliminating human-in-the-loop overhead. |

**Application in this environment:** Partial label credit (0.25 for KEEP/REJECT when ground truth is BORDERLINE, scaled by difficulty) treats directionally-correct but imprecise decisions as informative signal rather than noise, analogous to weak preferences in RLHF training.

---

### Verifiable Rewards

| Paper | Year | Relevance |
|-------|------|-----------|
| Sutton & Barto "Reinforcement Learning: An Introduction" | 2018 | Core RL principles. The `grade()` function in `grader.py` is a classic deterministic reward function decomposed into format, label, and reasoning components. |
| Ng & Russell "Algorithms for Inverse RL" | 2000 | Reward shaping foundations. The rubric calibration cycle — tightening thresholds based on episode performance — is a form of dynamic reward shaping that keeps the task challenging as the agent improves. |

**Application in this environment:** The grader is fully deterministic and rubric-derived — no LLM judge. This guarantees reproducibility, enables automated validation, and satisfies the OpenEnv spec requirement for programmatic graders that return valid `0.0–1.0` scores.

---

### Self-Play & Co-Evolution

| Paper | Year | Relevance |
|-------|------|-----------|
| Bansal et al. "Emergent Complexity via Multi-Agent Competition" | 2018 | Agents and environments co-evolve, generating emergent difficulty without manual curriculum design. The rubric–GT co-evolution in ClipQualityEnv is a single-agent analogue of this pattern. |
| Leibo et al. "Multi-Agent RL in Sequential Social Dilemmas" | 2017 | Environment complexity scales with agent capability. Matches the calibration logic: as the agent succeeds on BORDERLINE clips, the rubric tightens, creating new BORDERLINE cases. |

**Application in this environment:** The learning flywheel — GT expands as the agent promotes confident predictions → rubric tightens based on accuracy → harder BORDERLINE cases emerge — is co-evolution in a single-agent setting. The environment adapts to the agent's current capability level without external intervention.

---

### In-Context Learning

| Paper | Year | Relevance |
|-------|------|-----------|
| Brown et al. "Language Models are Few-Shot Learners" | 2020 | In-context learning (ICL) enables LLMs to improve on a task purely from examples in the context window, without weight updates. The per-episode ICL loop uses this for within-episode improvement. |
| Xie et al. "An Explanation of In-Context Learning as Implicit Bayesian Inference" | 2022 | Theoretical grounding for why ICL works — the model implicitly updates a prior over task hypotheses from context examples. Validates the step-by-step reward feedback injection in `ICLMemory.get_context_text()`. |

**Application in this environment:** The `ICLMemory` class carries reward feedback from prior attempts at each clip into subsequent steps within the same session. The agent's context window includes `label_score` signals and soft directives across attempts, enabling learning-without-gradient-updates over a 5-step episode and across multiple episode runs within a session.

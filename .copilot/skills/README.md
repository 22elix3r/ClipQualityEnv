# ClipQualityEnv — GitHub Copilot Skills

> Custom skills for developing the ClipQualityEnv OpenEnv RL environment  
> Hackathon: Scaler × Meta/PyTorch India 2026 — OpenEnv Track

---

## Available Skills

| Skill | File | Purpose |
|-------|------|---------|
| **Implement Code** | `implement-code.md` | Build env.py, grader.py, rubric.py, models.py, etc. |
| **Generate Metadata** | `generate-metadata.md` | Create synthetic clip metadata for testing |
| **Run Training** | `run-training.md` | Execute and debug the training loop |
| **Deploy HF Space** | `deploy-hf-space.md` | Deploy to Hugging Face Spaces (NEW) |

---

## Hackathon Requirements

Per `hackathon_requirements.md`, your submission MUST have:

1. ✅ **Typed Pydantic models** — `Observation`, `Action`, `Reward` in `models.py`
2. ✅ **`state()` method** — Returns checkpointable environment state
3. ✅ **`openenv.yaml`** — Environment specification file
4. ✅ **`inference.py`** — Baseline script with OpenAI client
5. ✅ **`Dockerfile`** — For HF Space deployment
6. ✅ **20-minute timeout** — Must complete on 2 vCPU / 8GB RAM

---

## How to Use

Invoke a skill by referencing the file in your prompt:

```
@.copilot/skills/implement-code.md implement the grader.py file
@.copilot/skills/deploy-hf-space.md deploy to HuggingFace
```

Or describe the task and Copilot will select the appropriate skill based on context.

---

## Project Context

**ClipQualityEnv** is a self-evolving **OpenEnv** RL environment where:
- An LLM agent classifies clip metadata as KEEP / BORDERLINE / REJECT
- Each episode has 3 steps: Easy → Medium → Hard
- The environment co-evolves via:
  - **Ground Truth Expansion** — confident hard answers are promoted to GT
  - **Rubric Calibration** — thresholds tighten as accuracy improves

---

## Required File Structure

```
clip_quality_env/
├── clip_quality_env/
│   ├── __init__.py
│   ├── env.py              # ClipQualityEnv class
│   ├── grader.py           # Deterministic reward scorer
│   ├── rubric.py           # RubricState (thresholds)
│   ├── ground_truth.py     # GTStore (expansion + querying)
│   ├── generator.py        # ClipMetaGenerator (synthetic clips)
│   └── models.py           # Pydantic models (NEW)
├── data/
│   └── seed_gt.json        # 20 hand-labeled seed clips
├── state/                  # Persisted state (gitignored)
│   ├── ground_truth.json
│   ├── rubric.json
│   └── history.jsonl
├── tests/
├── app.py                  # FastAPI for HF Space (NEW)
├── inference.py            # Baseline inference (NEW)
├── openenv.yaml            # Environment spec (NEW)
├── Dockerfile              # Container spec (NEW)
├── requirements.txt
└── README.md
```

---

## Core Principles

1. **The grader never calls an LLM** — all scoring is deterministic
2. **Rubric only tightens, never loosens** — calibration is irreversible
3. **Ground truth is append-only** — once promoted, labels don't change
4. **Hard clips are never reused within 10 episodes** — recency buffer
5. **Agent sees human-readable rubric** — not raw threshold numbers

---

## Reward Structure

```
R(step) = R_format (0.10) + R_label (0.60) + R_reasoning (0.30)

Episode Reward = 0.20 × R(step1) + 0.35 × R(step2) + 0.45 × R(step3)
```

---

## Success Metrics

After 500+ episodes:
- `gt.size()` ≥ 60 (started at 20)
- `rubric.version` ≥ 5
- Easy accuracy > 90%
- Medium accuracy > 70%
- Hard accuracy > 50%

---

## Pre-Submission Checklist

- [ ] `openenv.yaml` validates
- [ ] Pydantic models for Observation, Action, Reward
- [ ] `reset()`, `step()`, `state()` methods implemented
- [ ] Dockerfile builds and container responds
- [ ] `inference.py` completes within 20 minutes
- [ ] README.md has all 5 required sections
- [ ] Tests pass: `pytest tests/ -v`

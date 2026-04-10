# Contributing to ClipQualityEnv

Thank you for your interest in improving ClipQualityEnv! This guide covers the task creation process, grading invariants, and development workflow.

---

## Development Setup

```bash
git clone https://github.com/elix3r/ClipQualityEnv.git
cd ClipQualityEnv
pip install -e ".[dev]"
PYTHONPATH=. pytest tests/ -q
```

## Architecture Overview

```
clip_quality_env/
├── env.py          # Core OpenEnv environment (reset/step/state)
├── grader.py       # Reward decomposition: format + label + reasoning + calibration
├── ground_truth.py # GT seed + promoted labels store
├── rubric.py       # Threshold-based clip classification rubric
├── models.py       # Pydantic models (Action, Observation, State, Reward)
├── icl_memory.py   # In-context learning memory for cross-episode hints
└── real_clips.py   # Real clip manifest loader
server/
├── app.py          # FastAPI + Gradio dashboard
├── tasks/          # Task corpora (easy, medium, hard)
└── grader.py       # Server-side grading wrapper
```

## Creating New Tasks

### 1. Define the corpus

Create a new file in `server/tasks/` (e.g., `task_extreme.py`):

```python
EXTREME_TASK = {
    "task_id": "task_extreme",
    "difficulty": "extreme",
    "description": "Classify clips with adversarial signals...",
    "data_corpus": [
        {
            "clip_id": "clip_E001",
            "duration_s": 12.5,
            "fps": 25,
            "resolution": "1280x720",
            "face_area_ratio": 0.22,
            "face_confidence": 0.61,
            # ... all metadata fields ...
            "expected_label": "BORDERLINE",
        },
        # ... 7+ more clips ...
    ],
}
```

### 2. Required metadata fields

Every clip MUST include:

| Field | Type | Description |
|-------|------|-------------|
| `clip_id` | str | Unique identifier (format: `clip_XXXX`) |
| `duration_s` | float | Clip duration in seconds |
| `fps` | int | Frames per second |
| `resolution` | str | Video resolution (e.g., `"1280x720"`) |
| `face_area_ratio` | float | Face-to-frame area ratio [0, 1] |
| `face_confidence` | float | Face detection confidence [0, 1] |
| `head_pose_yaw_deg` | float | Head yaw rotation in degrees |
| `motion_score` | float | Motion magnitude [0, 1] |
| `bg_complexity_score` | float | Background complexity [0, 1] |
| `audio_snr_db` | float | Audio signal-to-noise ratio |
| `lighting_uniformity` | float | Lighting uniformity [0, 1] |
| `mouth_open_ratio` | float | Mouth openness ratio [0, 1] |
| `expected_label` | str | Ground truth: KEEP, BORDERLINE, or REJECT |
| `sharpness_score` | float | Image sharpness [0, 1] |
| `temporal_flicker` | float | Temporal flicker magnitude [0, 1] |
| `bg_entropy` | float | Background entropy [0, 1] |
| `eye_contact_ratio` | float | Eye contact percentage [0, 1] |
| `speech_rate_wpm` | float | Speech rate in words per minute |

### 3. Label distribution guidelines

| Difficulty | KEEP | BORDERLINE | REJECT | Min clips |
|-----------|------|------------|--------|-----------|
| Easy | 3 | 2 | 3 | 8 |
| Medium | 3 | 3 | 2 | 8 |
| Hard | 2 | 3 | 3 | 8 |

### 4. Register the task

Add to `server/tasks/__init__.py`:

```python
from .task_extreme import EXTREME_TASK

TASK_REGISTRY = {
    # ...existing tasks...
    "task_extreme": EXTREME_TASK,
}
```

### 5. Add GT overrides (if needed)

If the rubric derives a different label than `expected_label`, add an override to `data/seed_gt.json`:

```json
{
  "clip_E001": {
    "label": "BORDERLINE",
    "source": "seed",
    "episode": 0,
    "note": "GT override: rubric derives KEEP but expert judgment is BORDERLINE"
  }
}
```

## Grading Invariants

These invariants MUST hold for any valid grader implementation:

### 1. Score bounds
- All reward components ∈ [0.0, 1.0]
- `total = min(format + label + reasoning + calibration, ceiling)`
- Per-step ceiling: easy=0.90, medium=0.80, hard=0.70

### 2. Determinism
- `grade(action, clip, rubric, gt, difficulty)` MUST return identical results for identical inputs
- Label noise is seeded by `hash(clip_id + label)` — stable across runs

### 3. Privacy contract
The agent MUST NEVER see:
- `expected_label` (stripped by `_sanitize_clip_for_agent`)
- `quality_cues` (stripped by sanitizer)
- `rubric_thresholds` (omitted from `obs.info`)

### 4. Reward decomposition
- `format_score` (0.10): Valid label + non-empty reasoning + confidence in [0, 1]
- `label_score` (≤0.68): Correct label + noise; partial credit for adjacent labels
- `reasoning_score` (≤0.30): Feature mentions + directional cues + hallucination check
- `calibration_adj` (±0.05): Confidence calibration bonus/penalty

### 5. Curriculum invariants
- Promotion requires `CURRICULUM_WINDOW` consecutive episodes above `CURRICULUM_PROMOTE_THRESHOLD`
- Demotion requires `CURRICULUM_WINDOW` consecutive episodes below `CURRICULUM_DEMOTE_THRESHOLD`
- Mixed-difficulty episodes follow the plan: 2 easy → 2 medium → 1 hard

## Testing

```bash
# Run full test suite
PYTHONPATH=. pytest tests/ -q

# Run specific test file
PYTHONPATH=. pytest tests/test_grader.py -v

# Verify determinism
PYTHONPATH=. python -c "from inference import run_baseline; r1=run_baseline(); r2=run_baseline(); assert r1['baseline_scores']==r2['baseline_scores'], 'Non-deterministic!'; print('Deterministic ✓')"

# OpenEnv validation
python -m openenv.cli validate

# Docker build & run
docker build -t clip-quality-env .
docker run --rm -p 7860:7860 clip-quality-env
```

## Pull Request Checklist

- [ ] All 57+ existing tests pass
- [ ] `openenv validate` passes
- [ ] `docker build` succeeds
- [ ] Determinism verified (3 identical baseline runs)
- [ ] Privacy contract maintained (no GT leakage to agent)
- [ ] New tasks have balanced label distributions
- [ ] GT overrides added for any rubric/expected_label mismatches

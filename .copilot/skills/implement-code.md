# Skill: Implement ClipQualityEnv Code

> For implementing Python code in the ClipQualityEnv OpenEnv RL environment

---

## Your Role

You are implementing a **self-evolving OpenEnv RL environment** for LLM agents. Your code must follow the architecture in `CLIP_QUALITY_ENV_BUILD_GUIDE.md` and conform to the OpenEnv interface standard for the **Scaler × Meta/PyTorch India 2026 Hackathon**.

---

## Hackathon Requirements

Per `hackathon_requirements.md`, your code MUST have:

1. **Typed Pydantic models** for `Observation`, `Action`, `Reward`
2. **`state()` method** in addition to `reset()` and `step()`
3. **`openenv.yaml`** configuration file
4. **`inference.py`** using OpenAI client with env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
5. **Dockerfile** for containerized deployment
6. Completion within **20-minute timeout** on 2 vCPU / 8GB RAM

---

## Required Pydantic Models (`models.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class ClipMetadata(BaseModel):
    """Single clip's metadata features."""
    clip_id: str
    duration_s: float = Field(..., ge=0.0)
    fps: int = Field(..., ge=1)
    resolution: str
    face_area_ratio: float = Field(..., ge=0.0, le=1.0)
    face_confidence: float = Field(..., ge=0.0, le=1.0)
    head_pose_yaw_deg: float
    head_pose_pitch_deg: float
    motion_score: float = Field(..., ge=0.0, le=1.0)
    bg_complexity: str
    bg_complexity_score: float = Field(..., ge=0.0, le=1.0)
    mouth_open_ratio: float = Field(..., ge=0.0, le=1.0)
    blink_rate_hz: float = Field(..., ge=0.0)
    audio_snr_db: float
    transcript_word_count: int = Field(..., ge=0)
    transcript_confidence: float = Field(..., ge=0.0, le=1.0)
    lighting_uniformity: float = Field(..., ge=0.0, le=1.0)
    occlusion_present: bool
    environment_tag: str


class HistoryItem(BaseModel):
    """Record of a previous step in this episode."""
    step: int
    clip_id: str
    label: str
    reward: float


class Observation(BaseModel):
    """OpenEnv-compliant observation returned by reset() and step()."""
    step: int = Field(..., ge=1, le=3)
    rubric_version: int = Field(..., ge=1)
    rubric_summary: str
    clip_metadata: ClipMetadata
    history: list[HistoryItem] = Field(default_factory=list)


class Action(BaseModel):
    """Agent's structured response."""
    label: Literal["KEEP", "BORDERLINE", "REJECT"]
    reasoning: str = Field(..., min_length=20)
    confidence: float = Field(..., ge=0.0, le=1.0)


class Reward(BaseModel):
    """Decomposed reward breakdown."""
    total: float = Field(..., ge=0.0, le=1.0)
    format_score: float = Field(..., ge=0.0, le=1.0)
    label_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_score: float = Field(..., ge=0.0, le=1.0)
```

---

## OpenEnv Interface (3 Required Methods)

```python
class ClipQualityEnv:
    # Required methods
    def reset(self) -> Observation:        # Returns typed Observation
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
    def state(self) -> dict:               # REQUIRED: Returns checkpointable state
    def render(self) -> str:               # Returns human-readable state
```

---

## File Responsibilities

### `env.py` — ClipQualityEnv

```python
class ClipQualityEnv:
    """
    OpenEnv-compatible environment.
    One episode = 3 steps (EASY, MEDIUM, HARD).
    State persists across episodes via GTStore and RubricState.
    """
    
    def __init__(self, gt_store, rubric, generator): ...
    def reset(self) -> Observation: ...  # Returns step 1 observation
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]: ...
    def state(self) -> dict: ...  # NEW: Returns checkpointable state
    def _post_episode_update(self): ...  # GT expansion + calibration check
    def _build_obs(self, clip, step, history) -> Observation: ...
```

**state() method implementation:**
```python
def state(self) -> dict:
    """Return current environment state for checkpointing."""
    return {
        "episode_count": self.episode_count,
        "current_step": len(self.current_episode_history) + 1,
        "gt_size": self.gt_store.size(),
        "rubric_version": self.rubric.version,
        "current_clip_id": self._current_clips[len(self.current_episode_history)].clip_id if self._current_clips else None,
        "episode_history": self.current_episode_history,
        "rubric_thresholds": self.rubric.get_thresholds_summary(),
    }
```

**Critical invariants:**
- `reset()` always starts a new 3-step episode
- `step()` returns `done=True` only after step 3
- `state()` returns dict suitable for JSON serialization
- `_post_episode_update()` runs after every episode (GT promotion, calibration)

---

### `grader.py` — Deterministic Scorer

```python
def score(action: dict, clip: dict, rubric: RubricState, gt: GTStore) -> float:
    """Fully deterministic. No LLM calls. No randomness."""
    R_format = _score_format(action)      # 0.10 max
    R_label = _score_label(...)           # 0.60 max
    R_reasoning = _score_reasoning(...)   # 0.30 max
    return R_format + R_label + R_reasoning
```

**R_reasoning breakdown:**
- +0.10 if 2 dominant features mentioned by name
- +0.10 if directional reasoning is correct
- +0.10 if no hallucinated features

**NEVER:**
- Call an LLM to judge reasoning quality
- Add randomness to reward calculation
- Change reward based on episode number (that's the rubric's job)

---

### `rubric.py` — RubricState

```python
class RubricState:
    def __init__(self, path="state/rubric.json"): ...
    def derive_label(self, clip: dict) -> str: ...  # Rule-based KEEP/BORDERLINE/REJECT
    def tighten(self, feature: str, direction: str, delta: float): ...
    def recalibrate(self, perf: PerformanceWindow): ...  # Every 50 episodes
    def to_prompt_text(self) -> str: ...  # Human-readable for agent
    def get_dominant_features(self, clip: dict) -> list[str]: ...  # Top 2 features
```

**Threshold structure:**
```python
{
    "face_area_ratio": {"keep_min": 0.25, "keep_max": 1.0, "reject_min": 0.0, "reject_max": 0.18},
    "audio_snr_db": {"keep_min": 20.0, "keep_max": 999, "reject_min": 0.0, "reject_max": 14.0},
    # ... other features
}
```

**Calibration triggers:**
- `easy_accuracy > 0.92` → tighten thresholds, version++
- `medium_accuracy > 0.80` → shift difficulty boundary

**NEVER:**
- Loosen thresholds (only tighten)
- Skip version increment when tightening

---

### `ground_truth.py` — GTStore

```python
class GTStore:
    def __init__(self, seed_path, state_path): ...
    def lookup(self, clip_id: str) -> str | None: ...  # Returns label or None
    def try_promote(self, step3_result: dict, episode: int) -> bool: ...
    def size(self) -> int: ...
    def get_promoted_clip_ids(self) -> list[str]: ...
```

**Promotion criteria (both must pass):**
```python
if reward >= 0.85 and confidence >= 0.80:
    # Promote to GT
```

**NEVER:**
- Modify existing GT labels
- Promote clips that are already in GT
- Promote from Easy or Medium steps (only Hard)

---

### `generator.py` — ClipMetaGenerator

```python
class ClipMetaGenerator:
    def sample(self, difficulty: str, rubric: RubricState) -> dict: ...
    def _gen_easy(self, rubric) -> dict: ...   # All features clearly KEEP or REJECT
    def _gen_medium(self, rubric) -> dict: ... # 1-2 features at threshold boundaries
    def _gen_hard(self, rubric) -> dict: ...   # 2-3 conflicting signals, novel env_tag
```

**Clip metadata schema:**
```python
{
    "clip_id": "syn_xxxx",
    "duration_s": 7.3,
    "fps": 24,
    "resolution": "1280x720",
    "face_area_ratio": 0.31,
    "face_confidence": 0.87,
    "head_pose_yaw_deg": 12.4,
    "head_pose_pitch_deg": -3.1,
    "motion_score": 0.18,
    "bg_complexity": "solid_dark",
    "bg_complexity_score": 0.09,
    "mouth_open_ratio": 0.42,
    "blink_rate_hz": 0.28,
    "audio_snr_db": 24.1,
    "transcript_word_count": 38,
    "transcript_confidence": 0.91,
    "lighting_uniformity": 0.74,
    "occlusion_present": false,
    "environment_tag": "podcast_studio"
}
```

**Difficulty generation strategy:**
- Easy: Sample from center of KEEP or center of REJECT zones
- Medium: 1-2 features in borderline zone
- Hard: 2-3 features conflicting + novel environment_tag

---

### `agent.py` — LLMAgent

```python
class LLMAgent:
    SYSTEM_PROMPT = """You are a dataset quality analyst..."""
    
    def act(self, obs: dict) -> dict: ...
    def _build_prompt(self, obs: dict) -> str: ...
    def _parse_response(self, text: str) -> dict: ...
```

**Output parsing:**
```python
{
    "label": "KEEP" | "BORDERLINE" | "REJECT",
    "reasoning": "...",
    "confidence": 0.0-1.0,
    "raw": "original LLM response"
}
```

**Prompt structure:**
```
[Rubric summary]

[STEP 1 — Previous]
Your label: KEEP | Reward: 0.87

[STEP 2 — Current Clip]
{clip_metadata_json}

Classify this clip.
```

---

### `train.py` — Main Loop

```python
def main():
    rubric = RubricState("state/rubric.json")
    gt = GTStore("data/seed_gt.json", "state/ground_truth.json")
    generator = ClipMetaGenerator()
    env = ClipQualityEnv(gt, rubric, generator)
    agent = LLMAgent(model_name="...")
    
    episode = 0
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        episode += 1
```

---

## Implementation Checklist

When implementing any file:

1. [ ] Read `CLIP_QUALITY_ENV_BUILD_GUIDE.md` for full context
2. [ ] Follow the exact method signatures shown above
3. [ ] Respect the 5 core invariants (no LLM grader, rubric only tightens, etc.)
4. [ ] Add type hints for all function parameters and returns
5. [ ] Include docstrings explaining the purpose
6. [ ] Handle edge cases gracefully (missing files, malformed input)
7. [ ] Write unit tests in `tests/` folder

---

## Error Handling Patterns

```python
# Missing state file
if not os.path.exists(state_path):
    logger.info(f"No existing state at {state_path}, starting fresh")
    self.records = {}

# Malformed action from agent
label = action.get("label", "BORDERLINE")  # Default to BORDERLINE if missing

# Confidence out of range
confidence = max(0.0, min(1.0, float(action.get("confidence", 0.5))))

# Unknown feature in reasoning (hallucination check)
all_features = set(clip.keys())
mentioned = [w for w in reasoning.split() if w in all_features]
```

---

## Testing Strategy

```python
# tests/test_grader.py
def test_perfect_keep():
    """Agent correctly labels KEEP with good reasoning."""
    action = {"label": "KEEP", "reasoning": "face_area_ratio=0.48, audio_snr_db=26", "confidence": 0.9}
    clip = {"clip_id": "test1", "face_area_ratio": 0.48, "audio_snr_db": 26, ...}
    assert grader.score(action, clip, rubric, gt) >= 0.90

def test_partial_credit_borderline():
    """KEEP on BORDERLINE GT gets partial credit."""
    # GT says BORDERLINE, agent says KEEP
    assert grader._score_label("KEEP", clip_borderline, rubric, gt) == 0.25
```

---

## Common Mistakes to Avoid

❌ **Using LLM to score reasoning**
```python
# BAD
score = llm.judge(reasoning)
```
✅ **Use deterministic feature matching**
```python
# GOOD
dominant = rubric.get_dominant_features(clip)
mentioned = sum(1 for f in dominant if f in reasoning)
```

❌ **Hardcoding threshold values**
```python
# BAD
if clip["face_area_ratio"] >= 0.25:
```
✅ **Read from rubric state**
```python
# GOOD
if clip["face_area_ratio"] >= rubric.thresholds["face_area_ratio"]["keep_min"]:
```

❌ **Modifying GT labels**
```python
# BAD
gt.records[clip_id]["label"] = new_label
```
✅ **GT is append-only**
```python
# GOOD
if clip_id not in self.records:
    self.records[clip_id] = {...}
```

---

## inference.py Patterns

Per hackathon requirements, `inference.py` must use OpenAI client with env vars:

```python
import os
import json
from openai import OpenAI

# Required environment variables
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY"))

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_agent_action(observation: dict) -> Action:
    """Call LLM to get classification action."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(observation)},
        ],
        temperature=0.3,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    return parse_action(response.choices[0].message.content)
```

**Timeout management:**
```python
TIMEOUT_SECONDS = 20 * 60  # 20 minutes
BUFFER_SECONDS = 60  # Stop 1 minute early

while True:
    if time.time() - start_time > (TIMEOUT_SECONDS - BUFFER_SECONDS):
        print("Approaching timeout, stopping gracefully")
        break
```

---

## openenv.yaml Template

```yaml
name: clip_quality_env
version: "1.0.0"
description: "Self-evolving RL environment for LoRA dataset curation"
tasks:
  - id: easy_classification
    difficulty: easy
  - id: medium_classification
    difficulty: medium
  - id: hard_classification
    difficulty: hard
observation_space:
  type: object
  required: [step, rubric_version, rubric_summary, clip_metadata, history]
action_space:
  type: object
  required: [label, reasoning, confidence]
reward_range: [0.0, 1.0]
entrypoints:
  environment: "clip_quality_env.env:ClipQualityEnv"
  inference: "inference.py"
```

---

## Required File Structure

```
clip_quality_env/
├── clip_quality_env/
│   ├── __init__.py
│   ├── env.py          # ClipQualityEnv class
│   ├── grader.py       # Deterministic scorer
│   ├── rubric.py       # RubricState
│   ├── ground_truth.py # GTStore
│   ├── generator.py    # ClipMetaGenerator
│   └── models.py       # Pydantic models (NEW)
├── data/
│   └── seed_gt.json    # 20 hand-labeled clips
├── state/              # Persisted state (gitignored)
├── tests/
├── app.py              # FastAPI for HF Space (NEW)
├── inference.py        # Baseline inference (NEW)
├── openenv.yaml        # Environment spec (NEW)
├── Dockerfile          # Container spec (NEW)
├── requirements.txt
└── README.md
```

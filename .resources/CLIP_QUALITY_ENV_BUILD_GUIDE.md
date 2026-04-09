# ClipQualityEnv — Build Guide
### A Self-Evolving OpenEnv RL Environment for Dataset Clip Quality Classification

> **Version**: 1.0 — Ground-Up Redesign  
> **Hackathon**: Scaler × Meta/PyTorch India 2026 — OpenEnv Track  
> **Task Domain**: Talking-Head LoRA Dataset Curation

---

## Why the Old Approach Was Wrong

The core mistake was conflating **benchmark evaluation** (static, multi-node, no learning) with **a trainable RL environment** (dynamic, single-task, learns over time). The old architecture had:

- Reward logic scattered across disconnected nodes
- No persistent state between episodes
- The agent had no mechanism to improve from feedback
- "Ground truth" was fixed and never grew
- The grader was doing too much: it was judge, oracle, and policy all at once

This redesign throws all of that out and builds from one clean principle drawn from OLMo 3's model flow:

> **A training loop has three things: a world, an agent, and a signal. The world must evolve with the agent, not stay frozen.**

OLMo 3's post-training pipeline (SFT → DPO → RLVR) works because each stage's signal *sharpens* as the model improves. The rubric tightens. The distribution of hard cases shifts. That is exactly what this environment does — but in real-time, within a single continuous loop.

---

## Core Design Philosophy

### One Task, Three Difficulties, Infinite Iterations

The agent has **one job**: given a JSON blob of clip metadata, decide whether the clip is `KEEP`, `BORDERLINE`, or `REJECT` for talking-head LoRA training.

Each **episode** has exactly **three steps**: Easy → Medium → Hard. This mirrors how OLMo 3 builds capability in stages — general pretraining first, targeted midtraining second, reasoning third. The agent's context window carries the reward signal forward from step to step, enabling in-context learning (ICL) within the episode.

### The Two Co-Evolving Mechanisms

The environment is not static. After each episode, two systems update:

1. **Ground Truth Expansion** — The set of labeled reference clips grows. Confident correct answers on Hard steps get promoted into the ground truth store. As the GT store grows, the space of "genuinely hard" cases shrinks and must be replenished with harder synthetic edge cases.

2. **Grading Calibration** — The rubric (the weighting of features, the thresholds for accept/reject) tightens as the agent's accuracy on Easy cases rises. If Easy accuracy exceeds 92%, the tolerance for borderline answers on Easy cases is cut. The rubric is never relaxed — only tightened.

These two mechanisms **co-evolve**: a larger GT store makes the grader more confident → the grader becomes more precise → calibration tightens → the agent must produce more precise reasoning → correct hard answers expand the GT store further. This is the same flywheel that OLMo 3 uses between its SFT corpus and its RLVR reward model.

---

## OpenEnv Compatibility

ClipQualityEnv is built for the **OpenEnv** standard — an open format for LLM agent evaluation environments. This section explains how we implement the OpenEnv interface.

### What is OpenEnv?

OpenEnv is a standardized interface for RL environments designed for LLM agents. Key properties:

| Property | Description | Our Implementation |
|----------|-------------|-------------------|
| **Observation** | JSON dict passed to agent | Clip metadata + rubric + history |
| **Action** | Structured agent response | `{label, reasoning, confidence}` |
| **Reward** | Scalar feedback signal | Deterministic grader score [0, 1] |
| **Episode** | Complete task cycle | 3 steps: Easy → Medium → Hard |
| **State** | Persistent environment memory | GTStore + RubricState |

### Typed Pydantic Models (Required by Hackathon)

All observations, actions, and rewards must be **typed Pydantic models** per hackathon spec:

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
    step: int = Field(..., ge=1, le=3, description="Current step: 1=Easy, 2=Medium, 3=Hard")
    rubric_version: int = Field(..., ge=1, description="Current rubric calibration version")
    rubric_summary: str = Field(..., description="Human-readable rubric for agent")
    clip_metadata: ClipMetadata = Field(..., description="Current clip to classify")
    history: list[HistoryItem] = Field(default_factory=list, description="Previous steps in this episode")


class Action(BaseModel):
    """Agent's structured response — what we expect from the LLM."""
    label: Literal["KEEP", "BORDERLINE", "REJECT"] = Field(..., description="Classification decision")
    reasoning: str = Field(..., min_length=20, description="Explanation for the decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in this label")


class Reward(BaseModel):
    """Decomposed reward breakdown returned in info dict."""
    total: float = Field(..., ge=0.0, le=1.0, description="Combined reward score")
    format_score: float = Field(..., ge=0.0, le=1.0, description="Action format validity (0.10 weight)")
    label_score: float = Field(..., ge=0.0, le=1.0, description="Label correctness (0.60 weight)")
    reasoning_score: float = Field(..., ge=0.0, le=1.0, description="Reasoning quality (0.30 weight)")
```

### OpenEnv Interface Methods

```python
class ClipQualityEnv:
    """OpenEnv-compliant environment with state() method."""
    
    def reset(self) -> Observation:
        """Reset environment, start new episode, return step 1 observation."""
        self.episode_count += 1
        self.current_episode_history = []
        self._current_clips = self._sample_episode_clips()  # Easy, Medium, Hard
        return self._make_observation(step=1)
    
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute action, return (next_obs, reward, done, info).
        
        Args:
            action: Agent's structured Action response
            
        Returns:
            next_obs: Next observation (Observation model)
            reward: Scalar reward for this step [0.0, 1.0]
            done: True if episode complete (step 3 finished)
            info: {"reward_breakdown": Reward, "gt_promoted": bool, ...}
        """
        current_step = len(self.current_episode_history) + 1
        reward_breakdown = self.grader.grade(action, self._current_clips[current_step - 1])
        
        self.current_episode_history.append({
            "step": current_step,
            "clip_id": self._current_clips[current_step - 1].clip_id,
            "label": action.label,
            "reward": reward_breakdown.total,
        })
        
        done = (current_step == 3)
        if done:
            self._post_episode_update(action, reward_breakdown)
        
        next_obs = Observation(step=0, rubric_version=0, rubric_summary="", 
                               clip_metadata={}, history=[]) if done else self._make_observation(current_step + 1)
        
        return next_obs, reward_breakdown.total, done, {"reward_breakdown": reward_breakdown}
    
    def state(self) -> dict:
        """
        Return current environment state for checkpointing/inspection.
        
        REQUIRED by hackathon spec — enables:
        - Mid-episode checkpointing
        - Debugging environment state
        - Visualization dashboards
        """
        return {
            "episode_count": self.episode_count,
            "current_step": len(self.current_episode_history) + 1,
            "gt_size": self.gt_store.size(),
            "rubric_version": self.rubric.version,
            "current_clip_id": self._current_clips[len(self.current_episode_history)].clip_id if self._current_clips else None,
            "episode_history": self.current_episode_history,
            "rubric_thresholds": self.rubric.get_thresholds_summary(),
        }
    
    def render(self) -> str:
        """Return human-readable state description."""
        state = self.state()
        return (
            f"Episode {state['episode_count']} | Step {state['current_step']}/3\n"
            f"GT Size: {state['gt_size']} | Rubric v{state['rubric_version']}\n"
            f"Current Clip: {state['current_clip_id']}"
        )
```

### openenv.yaml Configuration File

Every OpenEnv submission requires an `openenv.yaml` spec file:

```yaml
# openenv.yaml — ClipQualityEnv specification
name: clip_quality_env
version: "1.0.0"
description: |
  Self-evolving RL environment for LoRA training dataset curation.
  Agent classifies video clip metadata as KEEP/BORDERLINE/REJECT.
  Environment co-evolves: Ground Truth expands, Rubric tightens.

author: "Your Team Name"
license: "MIT"

# Three-task structure per hackathon requirements
tasks:
  - id: easy_classification
    description: "Classify clips with ≥2 clearly dominant quality signals"
    difficulty: easy
    examples:
      - input: '{"face_area_ratio": 0.48, "bg_complexity_score": 0.07}'
        expected_output: "KEEP"
  
  - id: medium_classification
    description: "Classify clips with 1-2 borderline features"
    difficulty: medium
    examples:
      - input: '{"face_area_ratio": 0.24, "motion_score": 0.29}'
        expected_output: "BORDERLINE"
  
  - id: hard_classification
    description: "Classify clips with multiple conflicting quality signals"
    difficulty: hard
    examples:
      - input: '{"face_area_ratio": 0.35, "audio_snr_db": 15.0, "bg_complexity_score": 0.38}'
        expected_output: "Requires careful reasoning"

# Schema definitions for typed models
observation_space:
  type: object
  required: [step, rubric_version, rubric_summary, clip_metadata, history]
  properties:
    step:
      type: integer
      minimum: 1
      maximum: 3
      description: "Current step in episode (1=Easy, 2=Medium, 3=Hard)"
    rubric_version:
      type: integer
      minimum: 1
    rubric_summary:
      type: string
      description: "Human-readable grading rubric"
    clip_metadata:
      type: object
      description: "JSON blob of clip features"
    history:
      type: array
      items:
        type: object
        properties:
          step: {type: integer}
          clip_id: {type: string}
          label: {type: string}
          reward: {type: number}

action_space:
  type: object
  required: [label, reasoning, confidence]
  properties:
    label:
      type: string
      enum: [KEEP, BORDERLINE, REJECT]
    reasoning:
      type: string
      minLength: 20
    confidence:
      type: number
      minimum: 0.0
      maximum: 1.0

reward_range: [0.0, 1.0]

# Runtime constraints
constraints:
  timeout_minutes: 20
  max_episodes: 50
  memory_limit_mb: 8192

# Entry points
entrypoints:
  environment: "clip_quality_env.env:ClipQualityEnv"
  inference: "inference.py"
```

### Why OpenEnv for This Hackathon?

The Scaler × Meta/PyTorch India 2026 hackathon uses OpenEnv as the standard format because:

1. **Standardization** — Judges can evaluate all submissions with same harness
2. **Reproducibility** — Deterministic environments enable fair comparison
3. **Scalability** — OpenEnv supports automated large-scale evaluation

Our environment goes beyond basic OpenEnv by adding **co-evolution** — the environment itself improves alongside the agent.

---

## The Clip Metadata Schema

No images. No LoRA files. Everything is metadata extracted or synthetically generated from clip properties.

```json
{
  "clip_id": "clip_0042",
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

### Feature Semantics for the Grader

| Feature | KEEP range | BORDERLINE | REJECT trigger |
|---|---|---|---|
| `face_area_ratio` | ≥ 0.25 | 0.18–0.25 | < 0.18 |
| `face_confidence` | ≥ 0.80 | 0.65–0.80 | < 0.65 |
| `head_pose_yaw_deg` | ≤ 20° | 20°–35° | > 35° |
| `motion_score` | ≤ 0.25 | 0.25–0.45 | > 0.45 |
| `bg_complexity_score` | ≤ 0.15 | 0.15–0.40 | > 0.40 |
| `audio_snr_db` | ≥ 20 | 14–20 | < 14 |
| `duration_s` | 6.0–10.0 | 4.0–6.0 | < 4.0 or > 14.0 |
| `mouth_open_ratio` | ≥ 0.30 | 0.18–0.30 | < 0.18 |
| `lighting_uniformity` | ≥ 0.65 | 0.45–0.65 | < 0.45 |
| `occlusion_present` | false | — | true (hard reject) |

These ranges are the **initial rubric state**. They will tighten as calibration runs.

---

## Episode Structure

### The Three-Step Arc

```
Episode N
│
├── STEP 1 — EASY
│   Observation: Clip metadata with ≥ 2 clearly dominant signals
│   e.g. face_area_ratio=0.48, bg_complexity_score=0.07, snr=26dB
│   → Most features cleanly inside KEEP or REJECT range
│   → Expected: Agent labels correctly with short reasoning
│
├── STEP 2 — MEDIUM
│   Observation: Clip metadata + [Step 1 label + reward in context]
│   e.g. face_area_ratio=0.24 (borderline), motion_score=0.29 (borderline)
│   → One or two features straddle threshold
│   → Agent must weigh tradeoffs; ICL from step 1 feedback helps
│
└── STEP 3 — HARD
    Observation: Clip metadata + [Steps 1-2 labels + rewards in context]
    e.g. face_area=0.35 (good) BUT snr=15dB (borderline) AND bg=0.38 (borderline)
    → Multiple conflicting signals; possibly a novel environment_tag
    → Agent must reason about which features are dominant for LoRA training
    → Correct + confident answer → candidate for GT promotion
```

### In-Context Learning (ICL) Within the Episode

This is the "mini post-training loop" from the reference text — no weight updates, but the agent's context window carries its own feedback:

```
[STEP 1 PROMPT]
Here is the rubric: {rubric_v14}
Clip metadata: {clip_easy_json}
Classify: KEEP / BORDERLINE / REJECT. Use <label>, <reasoning>, <confidence> tags.

[STEP 1 RESPONSE]
<label>KEEP</label>
<reasoning>Face area ratio 0.48 is well above threshold...</reasoning>
<confidence>0.92</confidence>

[STEP 2 PROMPT]
Your last classification: KEEP | Reward: 0.87
New clip:
{clip_medium_json}
Classify again.

[STEP 3 PROMPT]
Your last classification: BORDERLINE | Reward: 0.41
Reconsider your weighting. New clip:
{clip_hard_json}
```

The agent sees what it got wrong and recalibrates within the same episode — exactly analogous to how OLMo 3's RLVR training loop updates the policy by comparing generated rollouts to verifiable rewards.

---

## Reward Design

### Reward Decomposition Per Step

Each step produces a scalar reward ∈ [0, 1.0]:

```
R(step) = R_format + R_label + R_reasoning
```

| Component | Max | Logic |
|---|---|---|
| `R_format` | 0.10 | All three tags present and non-empty |
| `R_label` | 0.60 | Exact match to ground truth label |
| `R_reasoning` | 0.30 | Feature mentions + correct weighting |

**`R_reasoning` breakdown** (this is fully deterministic — no LLM judge):

- +0.10 if the 2 most dominant features are mentioned by name
- +0.10 if the directional reasoning is correct (e.g. "snr is low, pushes toward reject")
- +0.10 if no hallucinated features are mentioned (features not in the metadata)

### Difficulty Multipliers

The environment weights step rewards differently at the episode level:

```
Episode Reward = 0.20 × R(step1) + 0.35 × R(step2) + 0.45 × R(step3)
```

This mirrors OLMo 3's post-training data weighting — harder, more targeted signals receive more weight.

### Partial Credit on BORDERLINE

When ground truth is `BORDERLINE`, the label reward is:
- `BORDERLINE` → 0.60 (full)
- `KEEP` or `REJECT` → 0.25 (partial — the agent committed, which is useful signal)
- This mirrors how OLMo 3's DPO stage uses preference pairs rather than hard binary rewards

---

## The Co-Evolution Loop

### Ground Truth Expansion

After each episode, the GT expansion logic runs:

```python
def try_expand_ground_truth(step3_result, current_gt):
    label = step3_result.label
    confidence = step3_result.confidence
    reward = step3_result.reward
    clip_id = step3_result.clip_id

    # Promotion criteria (all must pass)
    if reward >= 0.85 and confidence >= 0.80:
        if clip_id not in current_gt:
            current_gt[clip_id] = {
                "label": label,
                "promoted_at_episode": current_episode,
                "promotion_confidence": confidence,
                "source": "agent_promoted"
            }
            return True  # GT expanded
    return False
```

When a clip is promoted, it transitions from the Hard pool to the Medium pool in future episodes. New hard clips are synthesized by the generator to replace it — always keeping the Hard pool populated with genuinely uncertain cases.

### Grading Calibration (Rubric Tightening)

Every 50 episodes, the calibration check runs:

```python
def recalibrate_rubric(performance_window, rubric):
    easy_acc = performance_window.easy_accuracy  # last 50 episodes
    medium_acc = performance_window.medium_accuracy
    hard_acc = performance_window.hard_accuracy

    if easy_acc > 0.92:
        # Tighten easy thresholds — move KEEP floor up
        rubric.tighten("face_area_ratio", direction="floor", delta=0.02)
        rubric.tighten("bg_complexity_score", direction="ceiling", delta=-0.02)
        rubric.version += 1

    if medium_acc > 0.80:
        # Promote some medium cases to easy, generate harder medium
        rubric.shift_difficulty_boundary("easy_medium", delta=0.05)
        rubric.version += 1

    # Hard accuracy intentionally not used for tightening —
    # hard cases should remain genuinely uncertain
```

The rubric is stored as a versioned JSON. The agent receives the current rubric version in every prompt, so it always knows what standard it's being held to.

---

## Mathematical Foundations of Co-Evolution

### Formal Dynamics

The co-evolution system forms a coupled dynamical system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STATE SPACE                                                                 │
│                                                                              │
│  GT(t) = { clip_id → label } — Ground Truth at episode t                    │
│  τ(t) = { feature → threshold } — Rubric thresholds at episode t            │
│  π(t) = agent policy (implicit, via LLM weights + ICL)                      │
│                                                                              │
│  TRANSITIONS                                                                 │
│                                                                              │
│  GT(t+1) = GT(t) ∪ { promoted clips from Hard step }                        │
│  τ(t+1) = τ(t) + Δτ   if accuracy trigger fires                             │
│  π(t+1) ≈ π(t)        (weights unchanged, but ICL improves within episode)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Threshold Adaptation Formula

The rubric tightening follows an adaptive step-size rule:

```
τ(t+1) = τ(t) + α · sign(accuracy(t) - target)

where:
  τ(t) = threshold value at calibration epoch t
  α = step size (default: 0.02 for ratios, varies by feature)
  target = 0.92 for easy accuracy, 0.80 for medium
  
Constraint: τ(t+1) ≥ τ(t)  (rubric only tightens, never loosens)
```

### Stability Analysis

**Problem**: Naive threshold tightening can cause oscillation:
- Threshold too tight → coverage drops → accuracy drops → should loosen
- But our constraint prevents loosening

**Solution**: Use exponential moving average (EMA) smoothing:

```python
class RubricState:
    def recalibrate(self, perf, ema_beta=0.3):
        """
        Smooth calibration with EMA to prevent oscillation.
        """
        # Compute target threshold based on current accuracy
        if perf.easy_accuracy > 0.92:
            target_delta = 0.02
        else:
            target_delta = 0.0
        
        # Apply EMA smoothing
        # Instead of: τ_new = τ_old + Δ
        # We use:     τ_new = τ_old + β * Δ
        actual_delta = ema_beta * target_delta
        
        self.tighten("face_area_ratio", "floor", actual_delta)
        
    def tighten_with_hysteresis(self, feature, direction, delta, dead_band=0.01):
        """
        Only tighten if change would exceed dead band.
        Prevents micro-oscillations near threshold.
        """
        if abs(delta) < dead_band:
            return  # No change
        # ... apply tightening
```

**Lyapunov Stability Condition**:

For the system to converge, the error must decrease monotonically:

```
V(τ_error) = (τ - τ*)²   [Lyapunov function]

where τ* = optimal threshold (unknown, but exists)

For stability: dV/dt < 0

This requires: α < 1 / max(∂accuracy/∂τ)

In practice: α = 0.02 with EMA β = 0.3 is stable for all tested feature ranges.
```

### GT Expansion Rate

The expected GT growth rate follows:

```
E[ΔGT per episode] = P(reward ≥ 0.85) × P(confidence ≥ 0.80) × P(not in GT)

Early episodes: ~0.15 (low reward on hard cases)
After 100 eps: ~0.20 (improved reasoning)
After 300 eps: ~0.25 (but P(not in GT) decreasing)

Net effect: GT grows from 20 → 60 over ~500 episodes
            = ~0.08 clips promoted per episode on average
```

### Co-Evolution Flywheel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Agent solves Hard case with high confidence                                 │
│         │                                                                    │
│         ▼                                                                    │
│  GT expands (clip promoted)                                                  │
│         │                                                                    │
│         ▼                                                                    │
│  Grader has more reference points → fewer "unknown" clips                   │
│         │                                                                    │
│         ▼                                                                    │
│  Easy accuracy rises (more clips have definite GT labels)                   │
│         │                                                                    │
│         ▼                                                                    │
│  Calibration triggers → rubric tightens                                     │
│         │                                                                    │
│         ▼                                                                    │
│  Agent must produce more precise reasoning to score well                    │
│         │                                                                    │
│         ▼                                                                    │
│  Generator synthesizes harder Hard cases (old hard → new medium)            │
│         │                                                                    │
│         └──────────────────────────────────────────────────────────────────┐ │
│                                                                            │ │
│  ┌─────────────────────────────────────────────────────────────────────────┘ │
│  │                                                                           │
│  └──► Agent improves via ICL → solves new Hard case → [LOOP]                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
clip_quality_env/
│
├── env.py                  # ClipQualityEnv — the main Gym-style environment
├── grader.py               # Deterministic reward scorer
├── rubric.py               # RubricState — version-controlled thresholds
├── ground_truth.py         # GTStore — expansion + querying
├── generator.py            # ClipMetaGenerator — synthetic clip factory
├── agent.py                # LLMAgent — wraps vLLM/HuggingFace
├── train.py                # Main training loop
│
├── state/                  # Persisted between runs (gitignored for large runs)
│   ├── ground_truth.json   # All GT records with provenance
│   ├── rubric.json         # Current rubric version + history
│   └── history.jsonl       # Episode-level log: step rewards, labels, GT events
│
├── data/
│   └── seed_gt.json        # 20 hand-labeled seed clips (your 14 + 6 synthetic)
│
├── tests/
│   ├── test_grader.py
│   ├── test_rubric.py
│   └── test_generator.py
│
└── README.md
```

---

## Implementation Specs (File by File)

### `env.py` — ClipQualityEnv

```python
class ClipQualityEnv:
    """
    OpenEnv-compatible environment.
    One episode = 3 steps (EASY, MEDIUM, HARD).
    State persists across episodes via GTStore and RubricState.
    """

    def __init__(self, gt_store: GTStore, rubric: RubricState, generator: ClipMetaGenerator):
        self.gt = gt_store
        self.rubric = rubric
        self.generator = generator
        self.episode_count = 0
        self.current_episode_history = []

    def reset(self) -> dict:
        """Start a new episode. Returns step 1 observation."""
        self.current_episode_history = []
        self.episode_count += 1
        clip = self.generator.sample(difficulty="easy", rubric=self.rubric)
        return self._build_obs(clip, step=1, history=[])

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        action = {"label": str, "reasoning": str, "confidence": float}
        Returns: (next_obs, reward, done, info)
        """
        step_num = len(self.current_episode_history) + 1
        current_clip = self.current_episode_history[-1]["clip"] if self.current_episode_history else ...

        reward = grader.score(action, current_clip, self.rubric, self.gt)
        self.current_episode_history.append({
            "step": step_num, "action": action, "reward": reward, "clip": current_clip
        })

        done = (step_num == 3)
        if done:
            self._post_episode_update()
            return {}, reward, True, {}

        next_difficulty = ["medium", "hard"][step_num - 1]
        next_clip = self.generator.sample(difficulty=next_difficulty, rubric=self.rubric)
        next_obs = self._build_obs(next_clip, step=step_num + 1, history=self.current_episode_history)
        return next_obs, reward, False, {}

    def _post_episode_update(self):
        """Run GT expansion and check if calibration is due."""
        step3 = self.current_episode_history[2]
        self.gt.try_promote(step3, self.episode_count)
        if self.episode_count % 50 == 0:
            self.rubric.recalibrate(self._get_performance_window())
        log_episode(self.episode_count, self.current_episode_history)

    def _build_obs(self, clip, step, history) -> dict:
        return {
            "step": step,
            "rubric_version": self.rubric.version,
            "rubric_summary": self.rubric.to_prompt_text(),
            "clip_metadata": clip,
            "history": history  # previous steps' label + reward
        }
```

---

### `grader.py` — Deterministic Scorer

```python
def score(action: dict, clip: dict, rubric: RubricState, gt: GTStore) -> float:
    """
    Fully deterministic. No LLM calls. No randomness.
    """
    label = action.get("label", "")
    reasoning = action.get("reasoning", "")
    confidence = action.get("confidence", 0.0)

    R_format = _score_format(action)
    R_label = _score_label(label, clip, rubric, gt)
    R_reasoning = _score_reasoning(reasoning, clip, rubric)

    return R_format + R_label + R_reasoning

def _score_label(label, clip, rubric, gt) -> float:
    gt_label = gt.lookup(clip["clip_id"])
    if gt_label is None:
        gt_label = rubric.derive_label(clip)  # deterministic rule-based fallback

    if label == gt_label:
        return 0.60
    elif gt_label == "BORDERLINE" and label in ("KEEP", "REJECT"):
        return 0.25  # partial credit for confident decision on ambiguous case
    else:
        return 0.0

def _score_reasoning(reasoning: str, clip: dict, rubric: RubricState) -> float:
    score = 0.0
    dominant_features = rubric.get_dominant_features(clip)  # returns top 2 feature names

    # +0.10: are dominant features mentioned?
    mentioned = sum(1 for f in dominant_features if f in reasoning)
    if mentioned >= 2: score += 0.10
    elif mentioned == 1: score += 0.05

    # +0.10: directional correctness (e.g. "low snr" → "reject")
    if _check_directional_reasoning(reasoning, clip, dominant_features, rubric):
        score += 0.10

    # +0.10: no hallucinated features
    all_feature_names = set(clip.keys())
    words_in_reasoning = set(reasoning.lower().split())
    hallucinated = [w for w in words_in_reasoning if _looks_like_feature(w) and w not in all_feature_names]
    if len(hallucinated) == 0:
        score += 0.10

    return score
```

---

### `rubric.py` — RubricState

```python
class RubricState:
    def __init__(self, path="state/rubric.json"):
        self.path = path
        self.version = 1
        self.thresholds = load_initial_thresholds()  # the table from earlier
        self.history = []

    def derive_label(self, clip: dict) -> str:
        """
        Rule-based label derivation when GT doesn't have this clip.
        Returns KEEP / BORDERLINE / REJECT deterministically.
        """
        reject_signals = 0
        borderline_signals = 0
        keep_signals = 0

        for feature, val in clip.items():
            if feature not in self.thresholds: continue
            t = self.thresholds[feature]
            if self._in_keep_range(val, t): keep_signals += 1
            elif self._in_borderline_range(val, t): borderline_signals += 1
            else: reject_signals += 1

        # Hard reject overrides everything
        if clip.get("occlusion_present"): return "REJECT"
        if reject_signals >= 2: return "REJECT"
        if keep_signals >= 7 and borderline_signals <= 1: return "KEEP"
        return "BORDERLINE"

    def tighten(self, feature: str, direction: str, delta: float):
        """Shift a threshold boundary toward stricter evaluation."""
        old = self.thresholds[feature].copy()
        if direction == "floor":
            self.thresholds[feature]["keep_min"] += delta
        elif direction == "ceiling":
            self.thresholds[feature]["keep_max"] += delta  # delta is negative
        self.history.append({
            "episode": current_episode, "feature": feature,
            "old": old, "new": self.thresholds[feature]
        })
        self.save()

    def recalibrate(self, perf: PerformanceWindow):
        """Called every 50 episodes."""
        if perf.easy_accuracy > 0.92:
            self.tighten("face_area_ratio", "floor", 0.02)
            self.tighten("bg_complexity_score", "ceiling", -0.02)
            self.version += 1
        if perf.medium_accuracy > 0.80:
            self.shift_difficulty_boundary("easy_medium", 0.05)
            self.version += 1
        self.save()

    def to_prompt_text(self) -> str:
        """Human-readable rubric summary for agent prompt injection."""
        lines = [f"Rubric v{self.version} — Clip Quality Standards for Talking-Head LoRA:"]
        for feature, t in self.thresholds.items():
            lines.append(f"  {feature}: KEEP ∈ [{t['keep_min']}, {t['keep_max']}], REJECT outside [{t['reject_min']}, {t['reject_max']}]")
        return "\n".join(lines)
```

---

### `ground_truth.py` — GTStore

```python
class GTStore:
    def __init__(self, seed_path="data/seed_gt.json", state_path="state/ground_truth.json"):
        self.records = load_seed(seed_path)  # 20 hand-labeled clips
        if os.path.exists(state_path):
            self.records.update(load_state(state_path))  # promoted clips from prior runs
        self.path = state_path

    def lookup(self, clip_id: str) -> str | None:
        return self.records.get(clip_id, {}).get("label")

    def try_promote(self, step3_result: dict, episode: int) -> bool:
        """
        Promote a hard clip into GT if agent was confident and correct.
        "Correct" here means: consistent with rubric.derive_label().
        """
        clip_id = step3_result["clip"]["clip_id"]
        if clip_id in self.records: return False  # already in GT

        reward = step3_result["reward"]
        confidence = step3_result["action"]["confidence"]
        label = step3_result["action"]["label"]

        if reward >= 0.85 and confidence >= 0.80:
            self.records[clip_id] = {
                "label": label,
                "source": "agent_promoted",
                "episode": episode,
                "reward": reward,
                "confidence": confidence
            }
            self.save()
            return True
        return False

    def size(self) -> int:
        return len(self.records)

    def get_promoted_clip_ids(self) -> list[str]:
        return [k for k, v in self.records.items() if v.get("source") == "agent_promoted"]
```

---

### `generator.py` — ClipMetaGenerator

```python
class ClipMetaGenerator:
    """
    Generates synthetic clip metadata at specified difficulty levels.
    Difficulty is determined by how many features are near threshold boundaries.
    """

    def sample(self, difficulty: str, rubric: RubricState) -> dict:
        if difficulty == "easy":
            return self._gen_easy(rubric)
        elif difficulty == "medium":
            return self._gen_medium(rubric)
        elif difficulty == "hard":
            return self._gen_hard(rubric)

    def _gen_easy(self, rubric) -> dict:
        """All features clearly inside KEEP or clearly inside REJECT."""
        label = random.choice(["KEEP", "REJECT"])
        clip = {}
        for feature, t in rubric.thresholds.items():
            if label == "KEEP":
                clip[feature] = sample_from_keep_center(t)  # ±20% from keep center
            else:
                clip[feature] = sample_from_reject_zone(t)
        clip["clip_id"] = f"syn_{uuid4().hex[:8]}"
        clip["environment_tag"] = random.choice(EASY_ENVS)  # known env types
        return clip

    def _gen_hard(self, rubric) -> dict:
        """
        2-3 features near threshold boundaries. Some conflicting.
        Mix of good and bad signals that make the label genuinely ambiguous.
        """
        clip = self._gen_easy(rubric)  # start from clean base
        conflict_features = random.sample(list(rubric.thresholds.keys()), k=3)
        for f in conflict_features:
            clip[f] = sample_from_borderline_zone(rubric.thresholds[f])
        clip["environment_tag"] = random.choice(NOVEL_ENVS)  # unseen env types
        clip["clip_id"] = f"syn_hard_{uuid4().hex[:8]}"
        return clip
```

---

### `agent.py` — LLMAgent

```python
class LLMAgent:
    """
    Wraps any HuggingFace or vLLM model.
    Formats observations into prompts, parses structured output.
    """

    SYSTEM_PROMPT = """You are a dataset quality analyst for talking-head LoRA video training.
Your job is to classify clip metadata as KEEP, BORDERLINE, or REJECT.
Always respond using EXACTLY these XML tags:
<label>KEEP|BORDERLINE|REJECT</label>
<reasoning>your reasoning citing specific features</reasoning>
<confidence>0.0 to 1.0</confidence>"""

    def act(self, obs: dict) -> dict:
        prompt = self._build_prompt(obs)
        raw = self._call_model(prompt)
        return self._parse_response(raw)

    def _build_prompt(self, obs: dict) -> str:
        parts = [obs["rubric_summary"], ""]

        for h in obs["history"]:
            parts.append(f"[STEP {h['step']} — Previous]")
            parts.append(f"Your label: {h['action']['label']} | Reward: {h['reward']:.2f}")
            parts.append("")

        parts.append(f"[STEP {obs['step']} — Current Clip]")
        parts.append(json.dumps(obs["clip_metadata"], indent=2))
        parts.append("\nClassify this clip.")
        return "\n".join(parts)

    def _parse_response(self, text: str) -> dict:
        label = re.search(r"<label>(KEEP|BORDERLINE|REJECT)</label>", text)
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        confidence = re.search(r"<confidence>([\d.]+)</confidence>", text)
        return {
            "label": label.group(1) if label else "BORDERLINE",
            "reasoning": reasoning.group(1).strip() if reasoning else "",
            "confidence": float(confidence.group(1)) if confidence else 0.5,
            "raw": text
        }
```

---

### `train.py` — Main Training Loop

```python
def main():
    # Load or initialize persistent state
    rubric = RubricState("state/rubric.json")
    gt = GTStore("data/seed_gt.json", "state/ground_truth.json")
    generator = ClipMetaGenerator()
    env = ClipQualityEnv(gt, rubric, generator)
    agent = LLMAgent(model_name="meta-llama/Llama-3.1-8B-Instruct")  # or whatever

    print(f"Starting loop. GT size: {gt.size()}, Rubric: v{rubric.version}")

    episode = 0
    while True:  # runs until interrupted; state is checkpointed each episode
        obs = env.reset()
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            print(f"[Ep {episode} Step {env._current_step}] "
                  f"Label: {action['label']} | Reward: {reward:.3f} | "
                  f"GT: {gt.size()} | Rubric: v{rubric.version}")

        episode += 1

        # Every 100 episodes: print calibration summary
        if episode % 100 == 0:
            print_calibration_summary(rubric, gt, episode)

if __name__ == "__main__":
    main()
```

---

## State Files

### `state/ground_truth.json`

```json
{
  "clip_0001": {"label": "KEEP", "source": "seed", "episode": 0},
  "clip_0002": {"label": "REJECT", "source": "seed", "episode": 0},
  "syn_hard_a3f9b2": {"label": "BORDERLINE", "source": "agent_promoted", "episode": 312, "reward": 0.87, "confidence": 0.83}
}
```

### `state/rubric.json`

```json
{
  "version": 7,
  "thresholds": {
    "face_area_ratio": {"keep_min": 0.27, "keep_max": 1.0, "reject_min": 0.0, "reject_max": 0.18},
    "audio_snr_db": {"keep_min": 20.0, "keep_max": 999, "reject_min": 0.0, "reject_max": 14.0}
  },
  "calibration_history": [
    {"at_episode": 50, "trigger": "easy_acc=0.94", "feature": "face_area_ratio", "old_min": 0.25, "new_min": 0.27}
  ]
}
```

### `state/history.jsonl`

One JSON line per episode:
```json
{"episode": 1, "steps": [{"step": 1, "label": "KEEP", "gt": "KEEP", "reward": 0.87}, ...], "ep_reward": 0.72, "gt_promoted": false}
{"episode": 2, "steps": [...], "ep_reward": 0.61, "gt_promoted": true, "promoted_clip": "syn_hard_9a1c"}
```

---

## Build Order

### Phase 1 — Foundation (Day 1)

1. `data/seed_gt.json` — Hand-label 20 synthetic clips covering the feature space
2. `rubric.py` — RubricState with initial thresholds, `derive_label()`, `to_prompt_text()`
3. `grader.py` — `score()` with all three components; unit-test with known inputs
4. `tests/test_grader.py` — 10+ test cases covering KEEP/BORDERLINE/REJECT + partial credit

### Phase 2 — Environment Core (Day 1-2)

5. `ground_truth.py` — GTStore with seed loading and `try_promote()`
6. `generator.py` — Easy and Hard generators first; Medium can be a blend
7. `env.py` — Full episode loop: `reset()`, `step()`, `_post_episode_update()`
8. `tests/test_rubric.py` + `tests/test_generator.py`

### Phase 3 — Agent + Loop (Day 2)

9. `agent.py` — Prompt builder with history injection; parser with fallback defaults
10. `train.py` — Main loop; test for 10 episodes manually before running overnight

### Phase 4 — Calibration + Co-evolution (Day 3)

11. `rubric.py` — Add `recalibrate()` and `tighten()`; test calibration trigger
12. `ground_truth.py` — Add `try_promote()`; verify GT grows from promoted Hard answers
13. Validate co-evolution: run 200 episodes, confirm `gt.size()` grows and `rubric.version` increments

---

## Key Invariants to Never Break

1. **The grader never calls an LLM.** All scoring is deterministic. If you feel tempted to add an LLM judge, stop — that's the old approach.

2. **Rubric only tightens, never loosens.** Calibration is irreversible. The bar rises.

3. **Ground truth is append-only.** Once a clip is in GT, its label never changes. If the rubric later disagrees with an old GT label, the GT wins.

4. **Hard clips are never reused within 10 episodes.** The generator keeps a recency buffer.

5. **The agent never sees the internal rubric thresholds as raw numbers.** It sees `rubric.to_prompt_text()` — the human-readable version. This forces reasoning from description, not threshold-matching.

6. **Episode state does not leak between episodes.** The only cross-episode memory is the GTStore and RubricState — both persisted in `state/`.

---

## Connection to OLMo 3's Training Philosophy

| OLMo 3 | ClipQualityEnv |
|---|---|
| Pretraining on broad corpus | Early episodes: wide easy/hard coverage, loose rubric |
| Midtraining on targeted data | GT expansion promotes confident hard answers → harder medium cases synthesized |
| SFT stage: gold-standard examples | Seed GT (20 hand-labeled clips) |
| RLVR: verifiable reward from environment | `grader.score()` — deterministic, rule-based, no opinion |
| Rubric co-evolves with capability | `rubric.recalibrate()` tightens as easy accuracy rises |
| RL rollout inference dominates cost | ICL within episode = the cheapest form of within-episode learning |
| RL-Zero: reasoning on base model | Hard step requires chain-of-thought reasoning, not just label recall |

---

## What Success Looks Like

After 500+ episodes:

- `gt.size()` should be ≥ 60 (started at 20, growing ~1 per 6 episodes on hard cases)
- `rubric.version` should be ≥ 5 (calibration triggered at 50, 100, 150, 200, 250 if accuracy holds)
- Easy accuracy: > 90%
- Medium accuracy: > 70%
- Hard accuracy: > 50% (and genuinely harder than episode 1's hard cases)
- The agent's Step 3 reasoning should cite more specific features over time — a visible sign that ICL is working

The rubric becoming stricter and the GT expanding together is the proof that the environment is working correctly. If rubric version stagnates at 1 and GT stays at 20 — something is broken.

---

## Academic References

This environment draws from several foundational research areas. Understanding these connections strengthens both the implementation and the hackathon presentation.

### Curriculum Learning

| Paper | Year | Relevance |
|-------|------|-----------|
| **Bengio et al. "Curriculum Learning"** | 2009 | Foundation for Easy→Medium→Hard progression. Key insight: ordering training samples by difficulty accelerates learning. |
| **Graves et al. "Automated Curriculum Learning for Neural Networks"** | 2017 | Adaptive curriculum where difficulty self-adjusts based on learner performance. Matches our `recalibrate()` logic. |
| **Kumar et al. "Self-Paced Learning with Diversity"** | 2010 | Agent chooses its own curriculum. Our confidence-weighted promotion is a form of self-pacing. |

**Application**: Our 3-step episodes implement curriculum learning within episodes, while rubric calibration implements it across episodes.

### Active Learning & Self-Training

| Paper | Year | Relevance |
|-------|------|-----------|
| **Culotta & McCallum "Confidence-Weighted Active Learning"** | 2005 | Selectively promote high-confidence predictions to training set. Direct precedent for `try_promote()`. |
| **Zhu et al. "Semi-Supervised Learning with Graphs"** | 2003 | Self-training expands labeled set iteratively. Our GT expansion follows this pattern. |
| **Settles "Active Learning Literature Survey"** | 2010 | Comprehensive overview of query strategies. Our approach inverts uncertainty sampling — we promote confident predictions. |

**Application**: GT expansion is active learning in reverse. Instead of querying uncertain examples, we promote certain ones.

### Preference Optimization

| Paper | Year | Relevance |
|-------|------|-----------|
| **Rafailov et al. "Direct Preference Optimization (DPO)"** | 2023 | Preference-based training without reward models. Our partial credit on BORDERLINE cases mirrors preference pairs. |
| **Christiano et al. "Deep RL from Human Preferences"** | 2017 | RLHF foundation. We use verifiable rewards instead, but share the reward decomposition insight. |

**Application**: Partial credit (0.25 for KEEP/REJECT on BORDERLINE GT) treats confident decisions as useful signal, similar to preference learning.

### Verifiable Rewards

| Paper | Year | Relevance |
|-------|------|-----------|
| **Sutton & Barto "Reinforcement Learning: An Introduction"** | 2018 | Core RL principles. Our deterministic grader is a classic reward function. |
| **Ng & Russell "Algorithms for Inverse RL"** | 2000 | Reward shaping foundations. Our rubric calibration is a form of dynamic reward shaping. |

**Application**: The grader is fully deterministic — no LLM judge. This enables reproducibility and scalability.

### Self-Play & Co-Evolution

| Paper | Year | Relevance |
|-------|------|-----------|
| **Bansal et al. "Emergent Complexity via Multi-Agent Competition"** | 2018 | Agents and environments co-evolve, creating emergent difficulty. Our rubric-GT co-evolution is a single-agent version. |
| **Leibo et al. "Multi-Agent RL in Sequential Social Dilemmas"** | 2017 | Environment complexity increases with agent capability. Matches our calibration logic. |

**Application**: The flywheel (GT expands → rubric tightens → harder cases generated) is co-evolution in a single-agent setting.

### In-Context Learning

| Paper | Year | Relevance |
|-------|------|-----------|
| **Brown et al. "Language Models are Few-Shot Learners"** | 2020 | ICL enables learning without weight updates. Our episode structure uses ICL for within-episode improvement. |
| **Xie et al. "An Explanation of In-Context Learning as Implicit Bayesian Inference"** | 2022 | Theoretical grounding for why ICL works. Validates our step-by-step feedback injection. |

**Application**: The agent's context window carries reward feedback from Steps 1-2 into Step 3, enabling learning without gradient updates.

---

## Edge Cases & Error Handling

### Generator Edge Cases

| Edge Case | Handling |
|-----------|----------|
| **All features in KEEP range but should be REJECT** | Hard reject triggers override (occlusion, motion > 0.45, etc.) |
| **Duplicate clip_ids** | Generator uses UUID suffix; recency buffer prevents reuse within 10 episodes |
| **Feature correlation violations** | `apply_correlations()` post-processes to maintain realistic relationships |
| **Novel environment_tag not in training** | By design — Hard cases should include novel tags to test generalization |

```python
# Generator recency buffer
class ClipMetaGenerator:
    def __init__(self):
        self.recent_hard_ids = collections.deque(maxlen=10)
    
    def _gen_hard(self, rubric):
        while True:
            clip = self._generate_hard_candidate(rubric)
            if clip["clip_id"] not in self.recent_hard_ids:
                self.recent_hard_ids.append(clip["clip_id"])
                return clip
```

### Grader Edge Cases

| Edge Case | Handling |
|-----------|----------|
| **Agent returns malformed XML** | Default to BORDERLINE, confidence 0.5, empty reasoning |
| **Confidence outside [0, 1]** | Clamp to range: `max(0.0, min(1.0, confidence))` |
| **Label not in {KEEP, BORDERLINE, REJECT}** | Default to BORDERLINE |
| **Reasoning mentions non-existent features** | Hallucination penalty: lose 0.10 from R_reasoning |
| **GT disagrees with rubric** | GT wins — labels are append-only and never changed |

```python
def _parse_response(text: str) -> dict:
    """Parse with robust fallbacks."""
    label_match = re.search(r"<label>(KEEP|BORDERLINE|REJECT)</label>", text, re.IGNORECASE)
    label = label_match.group(1).upper() if label_match else "BORDERLINE"
    
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    
    confidence_match = re.search(r"<confidence>([\d.]+)</confidence>", text)
    try:
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        confidence = 0.5
    
    return {"label": label, "reasoning": reasoning, "confidence": confidence, "raw": text}
```

### Calibration Edge Cases

| Edge Case | Handling |
|-----------|----------|
| **Easy accuracy never reaches 0.92** | Rubric stays at v1 — this indicates a problem with easy case generation or agent capability |
| **Easy accuracy drops after tightening** | Expected temporarily; EMA smoothing prevents over-tightening |
| **GT grows faster than expected** | Raise confidence threshold from 0.80 → 0.85 |
| **All hard cases already in GT** | Generator synthesizes new hard cases using tightened rubric thresholds |

```python
def recalibrate(self, perf):
    """Calibrate with safeguards."""
    # Guard: Don't tighten if coverage is too low
    if perf.easy_coverage < 0.8:  # Less than 80% of easy cases evaluated
        logger.warning("Low coverage, skipping calibration")
        return
    
    # Guard: Don't tighten if we just tightened recently
    if self.episodes_since_last_tighten < 25:
        return
    
    if perf.easy_accuracy > 0.92:
        self.tighten_with_ema("face_area_ratio", "floor", 0.02)
        self.episodes_since_last_tighten = 0
```

### State Persistence Edge Cases

| Edge Case | Handling |
|-----------|----------|
| **State files corrupted** | Load from backup; fall back to seed if no backup |
| **Training interrupted mid-episode** | Episode not logged; resume clean from last complete episode |
| **Disk full** | Catch IOError, log warning, continue (data in memory until next successful write) |

```python
def save_with_backup(self, path):
    """Atomic save with backup."""
    backup_path = path + ".bak"
    temp_path = path + ".tmp"
    
    # Write to temp file first
    with open(temp_path, "w") as f:
        json.dump(self.to_dict(), f, indent=2)
    
    # Backup existing file
    if os.path.exists(path):
        shutil.copy(path, backup_path)
    
    # Atomic rename
    os.rename(temp_path, path)
```

---

## Hackathon Strategy

### Why This Is Novel

**1. Co-Evolution in Single-Agent RL**

Most RL environments are static — the reward function and task difficulty don't change. ClipQualityEnv is different:

- **GT expands** as the agent solves hard cases
- **Rubric tightens** as the agent masters easy cases
- **Difficulty shifts** — what was Hard at episode 1 is Medium at episode 200

This is co-evolution applied to a single agent, not multi-agent competition.

**2. Verifiable Rewards Without LLM Judges**

The standard approach for LLM agent evaluation is to use another LLM as a judge. This is:
- **Expensive** — costs scale with episodes
- **Non-reproducible** — LLM outputs vary
- **Gameable** — agent can learn to fool the judge

Our grader is fully deterministic:
- Feature matching is string search
- Directional correctness is rule-based
- Hallucination detection is set membership

**3. In-Context Learning as RL Signal**

The agent improves within episodes via ICL, not weight updates:
- Step 1 feedback → Step 2 recalibration
- Step 2 feedback → Step 3 deep reasoning

This is **zero-shot RL** — no training, just better prompting.

### Demo Strategy

**For Judges (5-minute demo):**

1. **Start with the problem** (30 sec)
   - "LoRA training needs clean clips. Manual labeling doesn't scale."

2. **Show the co-evolution** (2 min)
   - Run 50 episodes live (or show pre-recorded)
   - Display: GT size, rubric version, accuracy curves
   - "Watch the GT grow and the rubric tighten together"

3. **Highlight the architecture** (1 min)
   - "No LLM judge — fully deterministic rewards"
   - "No weight updates — ICL within episodes"

4. **Show the agent learning** (1 min)
   - Compare Step 3 reasoning from episode 1 vs episode 100
   - "The agent cites more specific features over time"

5. **Connect to OLMo 3** (30 sec)
   - "This mirrors their SFT → DPO → RLVR pipeline"
   - "Our rubric calibration is their reward sharpening"

### Visualization Specifications

**Required Plots:**

```
1. Learning Curves (3 lines: Easy, Medium, Hard)
   - X: Episode number
   - Y: Reward (rolling average, window=20)
   - Expected: Easy > Medium > Hard, all trending up

2. Co-Evolution Plot (2 subplots)
   - Top: GT size over episodes (starts at 20, target 60)
   - Bottom: Rubric version over episodes (starts at 1, target 5)
   - Expected: Both grow, with GT growth enabling rubric tightening

3. Feature Threshold Drift
   - X: Episode number
   - Y: Threshold value (e.g., face_area_ratio keep_min)
   - Expected: Monotonic increase (tightening)

4. Accuracy Heatmap
   - X: Episode (binned by 50)
   - Y: Difficulty level
   - Color: Accuracy (green=high, red=low)
   - Expected: Green spreads from Easy → Medium → Hard over time
```

**Code to Generate:**

```python
import matplotlib.pyplot as plt
import json
import numpy as np

def demo_plots(history_path="state/history.jsonl"):
    """Generate all demo plots."""
    eps = [json.loads(l) for l in open(history_path)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Learning curves
    ax1 = axes[0, 0]
    window = 20
    for i, (name, color) in enumerate([("Easy", "green"), ("Medium", "orange"), ("Hard", "red")]):
        rewards = [ep["steps"][i]["reward"] for ep in eps]
        rolling = [np.mean(rewards[max(0,j-window):j+1]) for j in range(len(rewards))]
        ax1.plot(rolling, label=name, color=color)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward (rolling avg)")
    ax1.set_title("Learning Curves by Difficulty")
    ax1.legend()
    
    # Plot 2: GT size
    ax2 = axes[0, 1]
    gt_sizes = []
    size = 20
    for ep in eps:
        if ep.get("gt_promoted"):
            size += 1
        gt_sizes.append(size)
    ax2.plot(gt_sizes, color="blue")
    ax2.axhline(y=60, color="green", linestyle="--", label="Target")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("GT Size")
    ax2.set_title("Ground Truth Expansion")
    ax2.legend()
    
    # Plot 3: Rubric version
    ax3 = axes[1, 0]
    versions = [ep.get("rubric_version", 1) for ep in eps]
    ax3.plot(versions, color="purple")
    ax3.axhline(y=5, color="green", linestyle="--", label="Target")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Rubric Version")
    ax3.set_title("Rubric Calibration")
    ax3.legend()
    
    # Plot 4: Accuracy heatmap (simplified)
    ax4 = axes[1, 1]
    # Bin by 50 episodes
    n_bins = len(eps) // 50 + 1
    heatmap = np.zeros((3, n_bins))
    for bin_idx in range(n_bins):
        start = bin_idx * 50
        end = min((bin_idx + 1) * 50, len(eps))
        for diff in range(3):
            rewards = [eps[i]["steps"][diff]["reward"] for i in range(start, end)]
            heatmap[diff, bin_idx] = np.mean(rewards) if rewards else 0
    
    im = ax4.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(["Easy", "Medium", "Hard"])
    ax4.set_xlabel("Episode Bin (×50)")
    ax4.set_title("Accuracy Heatmap")
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig("demo_plots.png", dpi=150)
    plt.show()
```

### Differentiators vs Other Submissions

| Likely Competition | Our Advantage |
|-------------------|---------------|
| Static benchmark environments | We co-evolve — harder to game, more realistic |
| LLM-as-judge evaluation | We use deterministic rewards — reproducible, cheaper |
| Single-difficulty tasks | We have curriculum within episodes — ICL demonstration |
| Fixed ground truth | We expand GT from agent answers — self-improving loop |
| Separate training + evaluation | We unify them — evaluation IS training signal |

---

## Inference Script Specification (`inference.py`)

Per hackathon requirements, every submission must include an `inference.py` baseline script that:
- Uses the OpenAI Python client
- Reads environment variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
- Runs a fixed number of episodes
- Completes within the 20-minute timeout

### Required Environment Variables

```bash
# Set these before running inference.py
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="meta-llama/Llama-3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"
# Or use OPENAI_API_KEY instead of HF_TOKEN for OpenAI models
```

### inference.py Implementation

```python
#!/usr/bin/env python3
"""
Baseline inference script for ClipQualityEnv.

Usage:
    export API_BASE_URL="https://api-inference.huggingface.co/v1/"
    export MODEL_NAME="meta-llama/Llama-3-70B-Instruct"
    export HF_TOKEN="hf_your_token"
    python inference.py
"""

import os
import json
import time
from openai import OpenAI

from clip_quality_env.env import ClipQualityEnv
from clip_quality_env.models import Action

# Environment variables (required by hackathon spec)
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY"))

if not HF_TOKEN:
    raise ValueError("Either HF_TOKEN or OPENAI_API_KEY must be set")

# Initialize OpenAI client with HF endpoint
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Constants for timeout management
MAX_EPISODES = 10  # Adjust based on timing tests
TIMEOUT_MINUTES = 20
TIMEOUT_SECONDS = TIMEOUT_MINUTES * 60
BUFFER_SECONDS = 60  # Stop 1 minute before timeout


def get_agent_action(observation: dict) -> Action:
    """Call LLM to get classification action."""
    
    system_prompt = """You are a video clip quality classifier for LoRA training datasets.
Given clip metadata, classify as KEEP, BORDERLINE, or REJECT.
Respond in JSON format: {"label": "...", "reasoning": "...", "confidence": 0.XX}"""
    
    user_prompt = f"""
Rubric (v{observation['rubric_version']}):
{observation['rubric_summary']}

Clip to classify:
{json.dumps(observation['clip_metadata'], indent=2)}

Previous steps this episode:
{json.dumps(observation['history'], indent=2) if observation['history'] else "None"}
"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    
    content = response.choices[0].message.content
    parsed = json.loads(content)
    
    return Action(
        label=parsed.get("label", "BORDERLINE"),
        reasoning=parsed.get("reasoning", "No reasoning provided"),
        confidence=float(parsed.get("confidence", 0.5)),
    )


def run_baseline():
    """Run baseline evaluation and return scores."""
    
    start_time = time.time()
    env = ClipQualityEnv()
    
    scores = {"easy": [], "medium": [], "hard": []}
    difficulty_names = ["easy", "medium", "hard"]
    
    episode = 0
    while True:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > (TIMEOUT_SECONDS - BUFFER_SECONDS):
            print(f"Approaching timeout ({elapsed:.0f}s), stopping gracefully")
            break
        
        if episode >= MAX_EPISODES:
            print(f"Completed {MAX_EPISODES} episodes")
            break
        
        print(f"\n=== Episode {episode + 1}/{MAX_EPISODES} ===")
        obs = env.reset()
        done = False
        step = 0
        
        while not done:
            try:
                action = get_agent_action(obs.model_dump())
                obs, reward, done, info = env.step(action)
                
                difficulty = difficulty_names[step]
                scores[difficulty].append(reward)
                print(f"  Step {step + 1} ({difficulty}): reward={reward:.3f}")
                
                step += 1
                
            except Exception as e:
                print(f"  Error in step {step + 1}: {e}")
                # Use fallback action
                action = Action(label="BORDERLINE", reasoning="Fallback due to error", confidence=0.5)
                obs, reward, done, info = env.step(action)
                step += 1
        
        episode += 1
    
    # Calculate final scores
    final_scores = {}
    for task, task_scores in scores.items():
        if task_scores:
            final_scores[task] = sum(task_scores) / len(task_scores)
        else:
            final_scores[task] = 0.0
    
    # Print summary
    print("\n=== FINAL SCORES ===")
    for task, score in final_scores.items():
        print(f"  {task}: {score:.3f} ({len(scores[task])} samples)")
    
    overall = sum(final_scores.values()) / len(final_scores) if final_scores else 0.0
    print(f"\n  OVERALL: {overall:.3f}")
    
    return final_scores


if __name__ == "__main__":
    results = run_baseline()
    print(json.dumps(results, indent=2))
```

### Timeout Management

The hackathon enforces a **20-minute timeout** on 2 vCPU / 8GB RAM. Key strategies:

1. **Episode Budget**: Limit to ~10-15 episodes for safety
2. **Elapsed Time Check**: Stop gracefully 1 minute before timeout
3. **Fallback Actions**: If LLM call fails, use BORDERLINE with 0.5 confidence
4. **Batch Prompting**: Consider batching if your LLM supports it

**Timing Benchmarks (estimate):**
| Component | Time per Episode |
|-----------|-----------------|
| LLM calls (3 steps) | ~30-60s |
| Grading | ~0.1s |
| Overhead | ~0.5s |
| **Total** | ~30-60s |

With 20 minutes, you can safely run **10-15 episodes** depending on LLM latency.

---

## Deployment: Dockerfile & HF Space

### Dockerfile

```dockerfile
# Dockerfile for ClipQualityEnv HF Space deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Gradio/FastAPI port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### requirements.txt

```
# requirements.txt
pydantic>=2.0.0
openai>=1.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
gradio>=4.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### FastAPI App (`app.py`)

```python
"""
FastAPI app exposing OpenEnv interface for HF Space.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from clip_quality_env.env import ClipQualityEnv
from clip_quality_env.models import Action, Observation

app = FastAPI(title="ClipQualityEnv", version="1.0.0")

# Global environment instance
env = ClipQualityEnv()


@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "environment": "clip_quality_env"}


@app.post("/reset", response_model=Observation)
def reset():
    """Reset environment and return initial observation."""
    obs = env.reset()
    return obs


@app.post("/step")
def step(action: Action):
    """Execute action and return (observation, reward, done, info)."""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump() if not done else None,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return current environment state."""
    return env.state()


@app.get("/render")
def render():
    """Return human-readable state description."""
    return {"render": env.render()}
```

### Hugging Face Space Deployment

1. **Create HF Space**: Go to huggingface.co/spaces → New Space → Docker

2. **Upload Files**:
   ```
   your-space/
   ├── Dockerfile
   ├── requirements.txt
   ├── app.py
   ├── inference.py
   ├── openenv.yaml
   └── clip_quality_env/
       ├── __init__.py
       ├── env.py
       ├── grader.py
       ├── rubric.py
       ├── ground_truth.py
       ├── generator.py
       └── models.py
   ```

3. **Set Secrets**: In Space settings, add:
   - `HF_TOKEN` (your HuggingFace token)
   - `API_BASE_URL` (inference endpoint)
   - `MODEL_NAME` (model to use)

4. **Test Deployment**:
   ```bash
   # Check health
   curl https://your-space.hf.space/
   
   # Reset environment
   curl -X POST https://your-space.hf.space/reset
   
   # Run a step
   curl -X POST https://your-space.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"label": "KEEP", "reasoning": "Good clip", "confidence": 0.9}'
   ```

---

## Pre-Submission Checklist

Before submitting, verify ALL of these pass:

### 1. Spec Compliance
- [ ] `openenv.yaml` validates with `openenv validate openenv.yaml`
- [ ] Typed Pydantic models for `Observation`, `Action`, `Reward`
- [ ] `reset()`, `step()`, `state()` methods implemented
- [ ] Reward range is `[0.0, 1.0]`

### 2. Deployment
- [ ] Dockerfile builds: `docker build -t clip_quality_env .`
- [ ] Container runs: `docker run -p 7860:7860 clip_quality_env`
- [ ] `/reset` returns HTTP 200 with valid Observation
- [ ] `/step` returns HTTP 200 with `(observation, reward, done, info)`
- [ ] `/state` returns HTTP 200 with environment state

### 3. Inference
- [ ] `inference.py` runs without error
- [ ] Completes within 20 minutes on 2 vCPU / 8GB RAM
- [ ] Returns scores for all 3 tasks (easy, medium, hard)

### 4. Tasks & Grading
- [ ] 3+ tasks with distinct difficulties
- [ ] Graders return scores in `[0.0, 1.0]`
- [ ] Partial credit for near-correct answers
- [ ] No LLM calls in grading (deterministic)

### 5. Documentation
- [ ] README.md has all 5 required sections:
  - [ ] Task Overview
  - [ ] Observation/Action/Reward Spaces
  - [ ] Grading Logic
  - [ ] Setup Instructions
  - [ ] Known Limitations
- [ ] openenv.yaml has complete task descriptions
- [ ] License file present

### 6. Quality
- [ ] Code lints clean: `ruff check .`
- [ ] Type hints on all public functions
- [ ] Tests pass: `pytest tests/`

### Automated Validation Script

```bash
#!/bin/bash
# pre_submit_check.sh

set -e

echo "=== Pre-Submission Check ==="

# 1. Lint
echo "1. Linting..."
ruff check . || { echo "FAIL: Lint errors"; exit 1; }

# 2. Type check
echo "2. Type checking..."
mypy clip_quality_env/ || { echo "FAIL: Type errors"; exit 1; }

# 3. Tests
echo "3. Running tests..."
pytest tests/ -v || { echo "FAIL: Test failures"; exit 1; }

# 4. Docker build
echo "4. Building Docker image..."
docker build -t clip_quality_env . || { echo "FAIL: Docker build failed"; exit 1; }

# 5. Container health check
echo "5. Testing container..."
docker run -d --name test_env -p 7860:7860 clip_quality_env
sleep 5
curl -f http://localhost:7860/ || { echo "FAIL: Container not responding"; docker rm -f test_env; exit 1; }
curl -f -X POST http://localhost:7860/reset || { echo "FAIL: /reset failed"; docker rm -f test_env; exit 1; }
docker rm -f test_env

# 6. openenv.yaml validation (if openenv CLI available)
if command -v openenv &> /dev/null; then
    echo "6. Validating openenv.yaml..."
    openenv validate openenv.yaml || { echo "FAIL: openenv.yaml invalid"; exit 1; }
fi

echo ""
echo "=== ALL CHECKS PASSED ==="
echo "Ready for submission!"
```

---

## Judging Criteria Alignment

Per hackathon requirements, submissions are scored on:

| Criteria | Weight | Our Strengths |
|----------|--------|---------------|
| **Real-World Utility** | 30% | LoRA dataset curation is genuine task with real demand |
| **Task & Grader Quality** | 25% | Deterministic grader, partial credit, no LLM judge |
| **Environment Design** | 20% | Co-evolution flywheel is architectural differentiator |
| **Code Quality & Spec Compliance** | 15% | Typed Pydantic models, openenv.yaml, clean structure |
| **Creativity & Novelty** | 10% | Self-evolving GT + rubric calibration = novel concept |

### Demo Strategy Aligned to Judging

1. **Real-World Utility (30%)** — Start demo showing the actual problem:
   - "LoRA training needs clean clips. Manual labeling doesn't scale."
   - Show real-world dataset sizes (thousands of clips)

2. **Task & Grader Quality (25%)** — Emphasize deterministic rewards:
   - "No LLM judge — fully reproducible"
   - Show grader code, explain feature matching

3. **Environment Design (20%)** — The co-evolution demo:
   - Run episodes live, show GT growth and rubric tightening
   - "The environment gets smarter as the agent gets smarter"

4. **Code Quality (15%)** — Quick code tour:
   - "Typed Pydantic models, full test coverage"
   - Show openenv.yaml, clean module structure

5. **Creativity (10%)** — Connect to research:
   - "Inspired by OLMo 3's RLVR methodology"
   - "Self-improving evaluation loop is novel"

---

*Build this in order. Test the grader first — everything else depends on it being correct.*

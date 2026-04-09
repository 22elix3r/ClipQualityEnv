# Skill: Run & Debug ClipQualityEnv Training

> For executing, monitoring, and debugging the OpenEnv training loop

---

## Your Role

You run the ClipQualityEnv training loop, monitor co-evolution metrics, diagnose issues, and verify the system is working correctly.

---

## Quick Start

### Initial Setup

```bash
# Create project structure
mkdir -p clip_quality_env/{state,data,tests}

# Install dependencies
pip install torch transformers vllm accelerate

# Or use requirements.txt
pip install -r requirements.txt
```

### First Run

```bash
# Start training (runs indefinitely)
python train.py

# Or run with limited episodes for testing
python -c "
from train import main
main(max_episodes=10)  # Stop after 10 episodes
"
```

---

## Monitoring Output

### Expected Console Output

```
Starting loop. GT size: 20, Rubric: v1
[Ep 0 Step 1] Label: KEEP | Reward: 0.870 | GT: 20 | Rubric: v1
[Ep 0 Step 2] Label: BORDERLINE | Reward: 0.650 | GT: 20 | Rubric: v1
[Ep 0 Step 3] Label: KEEP | Reward: 0.780 | GT: 20 | Rubric: v1
[Ep 1 Step 1] Label: REJECT | Reward: 0.920 | GT: 20 | Rubric: v1
...
[Ep 50] Calibration triggered: easy_acc=0.94 → Rubric v2
[Ep 52 Step 3] GT promoted: syn_hard_a3f9 → GT: 21
```

### Key Metrics to Watch

| Metric | Healthy Range | Problem Signal |
|--------|---------------|----------------|
| Step 1 Reward | 0.70–0.95 | <0.60 = agent not learning basics |
| Step 3 Reward | 0.40–0.85 | >0.90 = hard cases too easy |
| GT Size Growth | +1 per 5-10 eps | Stagnant = promotion criteria too strict |
| Rubric Version | +1 per 50 eps | Stagnant = easy_acc not reaching 0.92 |

---

## Common Issues & Fixes

### Issue 1: GT Size Never Grows

**Symptoms:**
- GT stays at 20 after 100+ episodes
- No "GT promoted" messages

**Diagnosis:**
```python
# Check step 3 statistics
import json
with open("state/history.jsonl") as f:
    for line in f:
        ep = json.loads(line)
        step3 = ep["steps"][2]
        if step3["reward"] >= 0.85:
            print(f"Ep {ep['episode']}: reward={step3['reward']}, conf={step3.get('confidence', 'N/A')}")
```

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Reward never reaches 0.85 | Check grader scoring logic |
| Confidence never reaches 0.80 | Agent not outputting high confidence on hard cases |
| Clips already in GT | Generator reusing clip_ids |

```python
# Debug: Print promotion attempts
def try_promote(self, step3_result, episode):
    clip_id = step3_result["clip"]["clip_id"]
    reward = step3_result["reward"]
    confidence = step3_result["action"]["confidence"]
    
    print(f"[DEBUG] Promotion check: {clip_id}")
    print(f"  reward={reward:.3f} (need ≥0.85)")
    print(f"  confidence={confidence:.3f} (need ≥0.80)")
    print(f"  already_in_gt={clip_id in self.records}")
    
    # ... rest of logic
```

---

### Issue 2: Rubric Version Never Increments

**Symptoms:**
- Rubric stays at v1 after 100+ episodes
- No "Calibration triggered" messages

**Diagnosis:**
```python
# Check easy accuracy
import json
history = []
with open("state/history.jsonl") as f:
    for line in f:
        history.append(json.loads(line))

# Calculate easy accuracy over last 50 episodes
recent = history[-50:]
easy_correct = sum(1 for ep in recent if ep["steps"][0]["reward"] >= 0.60)
easy_acc = easy_correct / len(recent)
print(f"Easy accuracy (last 50): {easy_acc:.2%}")
# Needs to be >92% to trigger calibration
```

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Easy cases too hard | Check generator._gen_easy() |
| Agent not learning | Check prompt construction |
| Grader too strict | Review R_label thresholds |

---

### Issue 3: Rewards Are Always 0.10 (Format Only)

**Symptoms:**
- All rewards are exactly 0.10
- Agent getting no label or reasoning credit

**Diagnosis:**
```python
# Check agent output parsing
action = agent.act(obs)
print(f"Parsed action: {json.dumps(action, indent=2)}")
# Check if label, reasoning, confidence are populated
```

**Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| XML tags not in response | LLM not following format |
| Regex not matching | Check _parse_response() patterns |
| Missing tags in response | Add few-shot examples to prompt |

```python
# Debug: Print raw LLM response
def _parse_response(self, text: str) -> dict:
    print(f"[DEBUG] Raw response:\n{text[:500]}...")
    # ... rest of parsing
```

---

### Issue 4: Agent Hallucinating Features

**Symptoms:**
- R_reasoning always losing 0.10 on hallucination check
- Agent mentioning features not in metadata

**Diagnosis:**
```python
# Check what features agent mentions vs what exists
def _score_reasoning(reasoning, clip, rubric):
    all_features = set(clip.keys())
    words = set(reasoning.lower().replace("_", " ").split())
    
    # Find hallucinated feature names
    potential_features = [w for w in words if len(w) > 4]  # Skip short words
    hallucinated = [w for w in potential_features 
                    if _looks_like_feature(w) and w not in all_features]
    
    print(f"[DEBUG] Hallucination check:")
    print(f"  Valid features: {all_features}")
    print(f"  Potential mentions: {potential_features}")
    print(f"  Hallucinated: {hallucinated}")
```

**Fix:**
Add explicit instruction in agent prompt:
```
ONLY cite features that appear in the JSON metadata above.
Do NOT mention "video_quality", "person_count", or other features not shown.
```

---

### Issue 5: Episodes Running Too Slow

**Symptoms:**
- Each episode takes >30 seconds
- vLLM or HuggingFace inference bottleneck

**Diagnosis:**
```python
import time

def act(self, obs):
    start = time.time()
    prompt = self._build_prompt(obs)
    build_time = time.time() - start
    
    start = time.time()
    raw = self._call_model(prompt)
    inference_time = time.time() - start
    
    start = time.time()
    result = self._parse_response(raw)
    parse_time = time.time() - start
    
    print(f"[TIMING] build={build_time:.2f}s, inference={inference_time:.2f}s, parse={parse_time:.2f}s")
    return result
```

**Fixes:**

| Bottleneck | Fix |
|------------|-----|
| Model loading each call | Load model once in __init__ |
| Long prompts | Reduce history or rubric verbosity |
| CPU inference | Use GPU, quantize model |
| Network latency (API) | Use local model with vLLM |

---

## Checkpoint & Resume

### Automatic Checkpointing

The system automatically saves state after every episode:

```
state/
├── ground_truth.json   # GT records (append-only)
├── rubric.json         # Current thresholds + version
└── history.jsonl       # Episode log (one line per episode)
```

### Resume Training

```python
def main(resume=True):
    rubric = RubricState("state/rubric.json")  # Loads existing or creates new
    gt = GTStore("data/seed_gt.json", "state/ground_truth.json")
    
    # Resume episode count from history
    episode = 0
    if os.path.exists("state/history.jsonl"):
        with open("state/history.jsonl") as f:
            episode = sum(1 for _ in f)
    
    print(f"Resuming from episode {episode}, GT: {gt.size()}, Rubric: v{rubric.version}")
    # ... continue training
```

### Manual Checkpoint

```python
def checkpoint(rubric, gt, episode):
    """Force save all state."""
    rubric.save()
    gt.save()
    
    # Also save a timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy("state/rubric.json", f"state/backups/rubric_{timestamp}.json")
    shutil.copy("state/ground_truth.json", f"state/backups/gt_{timestamp}.json")
```

---

## Validation Checks

### Before Long Runs

```python
def validate_system():
    """Run validation checks before starting long training."""
    errors = []
    
    # 1. Check seed GT exists
    if not os.path.exists("data/seed_gt.json"):
        errors.append("Missing data/seed_gt.json")
    else:
        with open("data/seed_gt.json") as f:
            seed = json.load(f)
            if len(seed) < 20:
                errors.append(f"Seed GT has only {len(seed)} clips, need 20")
    
    # 2. Check generator produces valid clips
    gen = ClipMetaGenerator()
    for diff in ["easy", "medium", "hard"]:
        clip = gen.sample(diff, RubricState())
        clip_errors = validate_clip(clip)
        if clip_errors:
            errors.append(f"Generator {diff} error: {clip_errors}")
    
    # 3. Check grader is deterministic
    action = {"label": "KEEP", "reasoning": "test", "confidence": 0.9}
    clip = gen.sample("easy", RubricState())
    r1 = grader.score(action, clip, RubricState(), GTStore())
    r2 = grader.score(action, clip, RubricState(), GTStore())
    if r1 != r2:
        errors.append(f"Grader not deterministic: {r1} != {r2}")
    
    # 4. Check agent produces valid output
    agent = LLMAgent()
    obs = {"step": 1, "rubric_version": 1, "rubric_summary": "test", 
           "clip_metadata": clip, "history": []}
    action = agent.act(obs)
    if action["label"] not in ["KEEP", "BORDERLINE", "REJECT"]:
        errors.append(f"Agent produced invalid label: {action['label']}")
    
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ❌ {e}")
        return False
    
    print("✅ All validation checks passed")
    return True
```

---

## Performance Analysis

### After 100+ Episodes

```python
def analyze_performance(history_path="state/history.jsonl"):
    """Analyze training performance."""
    episodes = []
    with open(history_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    
    # Accuracy by difficulty
    easy_rewards = [ep["steps"][0]["reward"] for ep in episodes]
    medium_rewards = [ep["steps"][1]["reward"] for ep in episodes]
    hard_rewards = [ep["steps"][2]["reward"] for ep in episodes]
    
    print(f"Episodes analyzed: {len(episodes)}")
    print(f"Easy   - Mean: {np.mean(easy_rewards):.3f}, Std: {np.std(easy_rewards):.3f}")
    print(f"Medium - Mean: {np.mean(medium_rewards):.3f}, Std: {np.std(medium_rewards):.3f}")
    print(f"Hard   - Mean: {np.mean(hard_rewards):.3f}, Std: {np.std(hard_rewards):.3f}")
    
    # GT growth
    gt_events = [ep for ep in episodes if ep.get("gt_promoted")]
    print(f"\nGT promotions: {len(gt_events)}")
    
    # Rubric versions
    rubric_changes = set()
    for ep in episodes:
        rubric_changes.add(ep.get("rubric_version", 1))
    print(f"Rubric versions seen: {sorted(rubric_changes)}")
    
    # Learning curve (rolling average)
    window = 20
    rolling_easy = [np.mean(easy_rewards[max(0,i-window):i+1]) 
                    for i in range(len(easy_rewards))]
    
    print(f"\nEasy accuracy trend (last 5 windows):")
    for i in range(-5, 0):
        print(f"  Episode {len(episodes)+i*window}: {rolling_easy[i*window]:.3f}")
```

---

## Success Criteria Verification

After 500+ episodes, verify:

```python
def verify_success():
    """Check if co-evolution is working."""
    with open("state/ground_truth.json") as f:
        gt = json.load(f)
    with open("state/rubric.json") as f:
        rubric = json.load(f)
    
    gt_size = len(gt)
    rubric_version = rubric["version"]
    
    checks = {
        "GT size ≥ 60": gt_size >= 60,
        "Rubric version ≥ 5": rubric_version >= 5,
        "GT grew from promotions": any(v.get("source") == "agent_promoted" for v in gt.values()),
        "Rubric has tightened": rubric_version > 1,
    }
    
    print("Success criteria:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    if gt_size >= 60 and rubric_version >= 5:
        print("\n🎉 CO-EVOLUTION IS WORKING!")
    else:
        print(f"\n⚠️ Not yet converged. GT: {gt_size}/60, Rubric: v{rubric_version}/5")
```

---

## Visualization Commands

### Plot Learning Curves

```python
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_learning_curves():
    episodes = []
    with open("state/history.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    
    easy = [ep["steps"][0]["reward"] for ep in episodes]
    medium = [ep["steps"][1]["reward"] for ep in episodes]
    hard = [ep["steps"][2]["reward"] for ep in episodes]
    
    # Rolling average
    window = 20
    def rolling(data):
        return [np.mean(data[max(0,i-window):i+1]) for i in range(len(data))]
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling(easy), label="Easy", color="green")
    plt.plot(rolling(medium), label="Medium", color="orange")
    plt.plot(rolling(hard), label="Hard", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling avg)")
    plt.title("ClipQualityEnv Learning Curves")
    plt.legend()
    plt.savefig("learning_curves.png")
    plt.show()
```

### Plot Co-Evolution

```python
def plot_coevolution():
    # GT size over time
    gt_sizes = []
    rubric_versions = []
    
    with open("state/history.jsonl") as f:
        gt_size = 20  # seed
        rubric_v = 1
        for line in f:
            ep = json.loads(line)
            if ep.get("gt_promoted"):
                gt_size += 1
            rubric_v = ep.get("rubric_version", rubric_v)
            gt_sizes.append(gt_size)
            rubric_versions.append(rubric_v)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(gt_sizes, color="blue")
    ax1.set_ylabel("Ground Truth Size")
    ax1.axhline(y=60, color="green", linestyle="--", label="Target: 60")
    ax1.legend()
    
    ax2.plot(rubric_versions, color="purple")
    ax2.set_ylabel("Rubric Version")
    ax2.set_xlabel("Episode")
    ax2.axhline(y=5, color="green", linestyle="--", label="Target: v5")
    ax2.legend()
    
    plt.suptitle("Co-Evolution: GT Expansion + Rubric Calibration")
    plt.savefig("coevolution.png")
    plt.show()
```

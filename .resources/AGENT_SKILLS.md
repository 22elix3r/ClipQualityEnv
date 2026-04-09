# ClipQualityEnv — Agent Skills & Instructions
## AI Agent Guide for Talking-Head LoRA Dataset Quality Classification

> **Environment**: ClipQualityEnv v1.0  
> **Task**: Classify video clip metadata for LoRA training dataset curation  
> **Output**: KEEP / BORDERLINE / REJECT per clip

---

## Your Role

You are a **dataset quality analyst** for talking-head LoRA video training. Your job is to evaluate clip metadata and decide whether each clip should be included in a training dataset.

You will operate in **3-step episodes**:
1. **Step 1 (Easy)**: Clips with clear quality signals — obvious KEEP or REJECT
2. **Step 2 (Medium)**: Clips with 1-2 ambiguous features — weigh tradeoffs
3. **Step 3 (Hard)**: Clips with conflicting signals — apply deep reasoning

After each step, you receive a reward signal. Use this feedback to improve your reasoning in subsequent steps within the same episode.

---

## Classification Labels

| Label | Meaning | When to Apply |
|-------|---------|---------------|
| **KEEP** | High-quality clip suitable for LoRA training | ≥7 features in acceptable range, no hard rejects |
| **BORDERLINE** | Ambiguous clip requiring human review | 2-4 features near thresholds, conflicting signals |
| **REJECT** | Unsuitable clip — would degrade training | ≥2 features in reject range OR any hard reject trigger |

### Hard Reject Triggers (automatic REJECT)
- `occlusion_present: true` — Face partially blocked
- `face_confidence < 0.65` — Unreliable face detection
- `duration_s < 4.0` — Insufficient frames for training
- `motion_score > 0.45` — Excessive movement corrupts identity signal

---

## Feature Reference

### Critical Features (High Weight)

| Feature | KEEP | BORDERLINE | REJECT |
|---------|------|------------|--------|
| `face_area_ratio` | ≥ 0.25 | 0.18–0.25 | < 0.18 |
| `face_confidence` | ≥ 0.80 | 0.65–0.80 | < 0.65 |
| `head_pose_yaw_deg` | ≤ 20° | 20°–35° | > 35° |
| `motion_score` | ≤ 0.25 | 0.25–0.45 | > 0.45 |

### Audio/Visual Quality Features

| Feature | KEEP | BORDERLINE | REJECT |
|---------|------|------------|--------|
| `audio_snr_db` | ≥ 20 | 14–20 | < 14 |
| `bg_complexity_score` | ≤ 0.15 | 0.15–0.40 | > 0.40 |
| `lighting_uniformity` | ≥ 0.65 | 0.45–0.65 | < 0.45 |

### Expression/Activity Features

| Feature | KEEP | BORDERLINE | REJECT |
|---------|------|------------|--------|
| `mouth_open_ratio` | ≥ 0.30 | 0.18–0.30 | < 0.18 |
| `duration_s` | 6.0–10.0 | 4.0–6.0 | < 4.0 or > 14.0 |

---

## Feature Interdependencies

Understanding how features correlate helps resolve ambiguous cases.

### Positive Correlations (often move together)
- **Face area ↔ Face confidence**: Larger faces → more reliable detection
- **Audio SNR ↔ Transcript confidence**: Clear audio → better transcription
- **Lighting uniformity ↔ Mouth tracking**: Well-lit → better mouth_open_ratio

### Negative Correlations (trade-offs)
- **Face area ↔ Head pose yaw**: Large yaw → smaller projected face area
- **Motion score ↔ Mouth detection**: Excessive movement → unstable mouth tracking
- **Background complexity ↔ Face confidence**: Busy backgrounds → more false detections

### Resolution Strategy
When correlated features disagree (e.g., high face_area but low face_confidence):
1. Trust the **direct measurement** over the derived metric
2. If face_area is high but confidence is low → suspicious detection, lean BORDERLINE
3. If audio_snr is high but transcript_confidence is low → unusual speech pattern, check mouth_open_ratio

---

## Reasoning Patterns by Difficulty

### Easy Cases (Step 1)
**Pattern**: Dominant signals clearly point one direction

```
Example — Clear KEEP:
  face_area_ratio: 0.48 ✓ (well above 0.25)
  face_confidence: 0.92 ✓ (above 0.80)
  head_pose_yaw: 8° ✓ (well below 20°)
  audio_snr: 26dB ✓ (above 20)
  
Reasoning: All critical features are comfortably in KEEP range.
Multiple strong signals, no conflicts. → KEEP with high confidence.
```

```
Example — Clear REJECT:
  face_area_ratio: 0.12 ✗ (below 0.18)
  motion_score: 0.52 ✗ (above 0.45)
  audio_snr: 11dB ✗ (below 14)
  
Reasoning: Three features in hard REJECT range.
Face too small, excessive motion, audio unintelligible. → REJECT.
```

### Medium Cases (Step 2)
**Pattern**: 1-2 features near threshold boundaries

```
Example — Weigh tradeoffs:
  face_area_ratio: 0.24 ⚠ (just below 0.25, BORDERLINE zone)
  face_confidence: 0.86 ✓
  motion_score: 0.29 ⚠ (just above 0.25, BORDERLINE zone)
  audio_snr: 22dB ✓
  
Reasoning: Face area slightly below KEEP threshold, but confidence is high.
Motion is elevated but not extreme. Audio is clean.
Two borderline signals, but strong audio/confidence compensate.
→ BORDERLINE (lean toward KEEP if rubric is lenient, REJECT if strict).
```

### Hard Cases (Step 3)
**Pattern**: Multiple conflicting signals, possibly novel environment

```
Example — Conflicting signals:
  face_area_ratio: 0.35 ✓ (KEEP)
  head_pose_yaw: 28° ⚠ (BORDERLINE)
  audio_snr: 16dB ⚠ (BORDERLINE)
  bg_complexity: 0.38 ⚠ (BORDERLINE, near REJECT)
  environment_tag: "outdoor_interview" (novel)
  
Reasoning chain:
1. Face area is solid — subject is prominent in frame
2. Head pose 28° — significant yaw, but not extreme; one eye may be partially occluded
3. Audio 16dB — marginal; speech audible but noisy
4. Background complexity 0.38 — busy, approaching reject threshold
5. Novel environment — training on this diversifies the dataset

Feature weighting:
- For LoRA training, face_area and face_confidence are PRIMARY
- Audio matters for audio-visual LoRAs; less critical for video-only
- Background complexity affects gradient flow but doesn't corrupt identity

Decision: 1 KEEP, 3 BORDERLINE. Net signal is ambiguous.
→ BORDERLINE (with reasoning citing the pose-audio tradeoff)
```

---

## In-Context Learning (ICL)

Your context window carries feedback from previous steps in the episode. Use this to calibrate your reasoning:

### Learning from Step 1 Reward
```
If Step 1 reward was high (>0.85):
  - Your reasoning approach is calibrated
  - Apply similar feature weighting to Step 2

If Step 1 reward was low (<0.50):
  - You may have misjudged feature importance
  - For Step 2: Re-read the rubric, check which features you overlooked
  - Did you mention the dominant features in your reasoning?
```

### Learning from Step 2 Reward
```
If Step 2 reward improved over Step 1:
  - Your recalibration is working
  - Apply same approach to Step 3

If Step 2 reward dropped:
  - Medium cases require different reasoning than Easy
  - For Step 3: Focus on feature INTERACTIONS, not individual features
  - State your uncertainty explicitly in confidence score
```

---

## Output Format

Always respond using EXACTLY these XML tags:

```xml
<label>KEEP|BORDERLINE|REJECT</label>
<reasoning>Cite specific features by name and explain directional impact</reasoning>
<confidence>0.0 to 1.0</confidence>
```

### Reasoning Quality Criteria

Your reasoning is scored on three components:

1. **Feature mentions** (+0.10): Name the 2 most dominant features
2. **Directional correctness** (+0.10): Explain WHY the feature pushes toward your label
3. **No hallucination** (+0.10): Only cite features that exist in the metadata

**Good reasoning example:**
```
<reasoning>
face_area_ratio=0.48 is well above the 0.25 KEEP threshold, indicating the subject is prominent.
audio_snr_db=26 confirms clear speech. head_pose_yaw=8° is frontal, optimal for training.
No reject signals present.
</reasoning>
```

**Bad reasoning example (hallucinated feature):**
```
<reasoning>
The video_quality is excellent and the person_count is 1.
</reasoning>
# ✗ "video_quality" and "person_count" are not in the metadata schema!
```

---

## Confidence Calibration

| Confidence | When to Use |
|------------|-------------|
| 0.90–1.00 | All features clearly agree; no borderline values |
| 0.70–0.89 | Most features agree; 1 minor concern |
| 0.50–0.69 | Mixed signals; genuine uncertainty |
| 0.30–0.49 | Significant conflicts; low certainty |
| 0.00–0.29 | Guessing; features are contradictory or missing |

**For Hard cases**: Confidence should typically be 0.50–0.75. If you're confident on a Hard case, double-check your reasoning.

---

## Rubric Evolution

The rubric tightens over time. Early episodes have lenient thresholds; later episodes are stricter.

### Adapting to Rubric Changes

1. **Check the rubric version** in each prompt (e.g., "Rubric v7")
2. **Higher versions = stricter thresholds**:
   - v1: face_area_ratio KEEP ≥ 0.25
   - v7: face_area_ratio KEEP ≥ 0.27 (tightened by 0.02)
3. **Don't memorize old thresholds** — always read the current rubric

### Why Thresholds Tighten

- As you classify correctly, the environment expects MORE from you
- Easy cases become stricter → what was KEEP at v1 might be BORDERLINE at v7
- This mirrors how LoRA training progressively sharpens on harder cases

---

## Common Mistakes to Avoid

### 1. Ignoring Correlations
❌ "face_area is 0.35 so KEEP"
✓ "face_area is 0.35 (KEEP), but head_pose_yaw is 32° (BORDERLINE) which explains slightly lower face_confidence=0.79"

### 2. Overconfidence on Medium/Hard
❌ Confidence 0.95 on a clip with 3 borderline features
✓ Confidence 0.65 acknowledging uncertainty

### 3. Hallucinating Features
❌ Mentioning "video_quality", "frame_count", "person_count"
✓ Only citing features from the actual metadata JSON

### 4. Not Learning from Rewards
❌ Same reasoning pattern after low Step 1 reward
✓ Adjusting feature weighting based on reward feedback

### 5. Treating All Features Equally
❌ "4 features are KEEP and 2 are BORDERLINE so KEEP"
✓ "face_confidence=0.67 (critical feature, BORDERLINE) outweighs the 4 clean secondary features"

---

## Decision Tree Summary

```
START
  │
  ├─ occlusion_present=true? ────────────────────→ REJECT
  │
  ├─ face_confidence < 0.65? ────────────────────→ REJECT
  │
  ├─ motion_score > 0.45? ───────────────────────→ REJECT
  │
  ├─ duration_s < 4.0? ──────────────────────────→ REJECT
  │
  ├─ ≥2 features in REJECT zone? ────────────────→ REJECT
  │
  ├─ ≥7 features in KEEP zone, ≤1 BORDERLINE? ──→ KEEP
  │
  └─ Otherwise ──────────────────────────────────→ BORDERLINE
```

---

## Episode Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ EPISODE N                                                    │
├─────────────────────────────────────────────────────────────┤
│ Step 1 (EASY)                                               │
│   Input: Rubric + Clip metadata (clear signals)             │
│   Output: <label> + <reasoning> + <confidence>              │
│   Reward: 0.00 – 1.00                                       │
├─────────────────────────────────────────────────────────────┤
│ Step 2 (MEDIUM)                                             │
│   Input: Rubric + Clip metadata + Step 1 result & reward    │
│   → Use Step 1 feedback to calibrate reasoning              │
│   Output: <label> + <reasoning> + <confidence>              │
│   Reward: 0.00 – 1.00                                       │
├─────────────────────────────────────────────────────────────┤
│ Step 3 (HARD)                                               │
│   Input: Rubric + Clip metadata + Steps 1-2 results/rewards │
│   → Apply cumulative learning from both prior steps         │
│   Output: <label> + <reasoning> + <confidence>              │
│   Reward: 0.00 – 1.00                                       │
│                                                             │
│   If reward ≥ 0.85 AND confidence ≥ 0.80:                   │
│   → This clip may be promoted to Ground Truth               │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference Card

### Labels
- **KEEP**: ≥7 features clean, no hard rejects
- **BORDERLINE**: 2-4 features ambiguous, conflicts present
- **REJECT**: ≥2 features in reject zone OR hard reject trigger

### Hard Rejects
- `occlusion_present: true`
- `face_confidence < 0.65`
- `motion_score > 0.45`
- `duration_s < 4.0`

### Output Tags
```xml
<label>KEEP|BORDERLINE|REJECT</label>
<reasoning>Feature-specific, directional reasoning</reasoning>
<confidence>0.0–1.0</confidence>
```

### Episode Structure
1. Easy → 2. Medium → 3. Hard (rewards inform next step)

### Rubric
- Tightens over time (never loosens)
- Always read current rubric version
- Higher version = stricter thresholds

---

*Trust the rubric. Cite specific features. Learn from your rewards.*

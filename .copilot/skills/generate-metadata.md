# Skill: Generate Synthetic Clip Metadata

> For creating synthetic clip metadata for ClipQualityEnv OpenEnv testing and training

---

## Your Role

You generate **synthetic clip metadata** that mimics real talking-head video analysis results. This data is used to train and test the RL environment without requiring actual video files.

---

## Metadata Schema

Every clip must have these fields:

```json
{
  "clip_id": "syn_a3f9b2c1",
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

---

## Feature Ranges & Semantics

| Feature | Type | KEEP Range | BORDERLINE | REJECT Range | Notes |
|---------|------|------------|------------|--------------|-------|
| `duration_s` | float | 6.0–10.0 | 4.0–6.0 | <4.0 or >14.0 | Optimal training length |
| `fps` | int | 24, 30, 60 | — | — | Standard values |
| `resolution` | str | "1920x1080", "1280x720" | "854x480" | <480p | Common formats |
| `face_area_ratio` | float | ≥0.25 | 0.18–0.25 | <0.18 | Face % of frame |
| `face_confidence` | float | ≥0.80 | 0.65–0.80 | <0.65 | Detection reliability |
| `head_pose_yaw_deg` | float | ≤20° | 20°–35° | >35° | Horizontal rotation |
| `head_pose_pitch_deg` | float | -15° to 15° | ±15°–25° | >±25° | Vertical rotation |
| `motion_score` | float | ≤0.25 | 0.25–0.45 | >0.45 | Frame-to-frame motion |
| `bg_complexity` | str | "solid_*", "gradient" | "office", "home" | "outdoor", "crowd" | Categorical |
| `bg_complexity_score` | float | ≤0.15 | 0.15–0.40 | >0.40 | Numerical complexity |
| `mouth_open_ratio` | float | ≥0.30 | 0.18–0.30 | <0.18 | Speaking activity |
| `blink_rate_hz` | float | 0.15–0.40 | 0.10–0.15 or 0.40–0.60 | <0.10 or >0.60 | Natural range |
| `audio_snr_db` | float | ≥20 | 14–20 | <14 | Signal-to-noise |
| `transcript_word_count` | int | 20–80 | 10–20 or 80–120 | <10 or >120 | Speech density |
| `transcript_confidence` | float | ≥0.85 | 0.70–0.85 | <0.70 | ASR reliability |
| `lighting_uniformity` | float | ≥0.65 | 0.45–0.65 | <0.45 | Even lighting |
| `occlusion_present` | bool | false | — | true | **Hard reject** |
| `environment_tag` | str | See lists below | Novel tags | — | Context category |

---

## Environment Tags

### Known (Easy) Environments
```python
EASY_ENVS = [
    "podcast_studio",
    "professional_interview",
    "home_office",
    "conference_room",
    "news_desk",
    "classroom",
    "webinar_setup"
]
```

### Novel (Hard) Environments
```python
NOVEL_ENVS = [
    "outdoor_interview",
    "moving_vehicle",
    "crowded_event",
    "industrial_setting",
    "underwater",
    "greenscreen_partial",
    "low_light_artistic",
    "multiple_subjects"
]
```

---

## Generation by Difficulty

### Easy Clips

**Goal**: All features clearly in KEEP or clearly in REJECT zone.

```python
def generate_easy_keep():
    """Generate a clip that should clearly be KEEP."""
    return {
        "clip_id": f"syn_easy_{uuid4().hex[:8]}",
        "duration_s": round(random.uniform(6.5, 9.5), 1),
        "fps": random.choice([24, 30]),
        "resolution": random.choice(["1920x1080", "1280x720"]),
        "face_area_ratio": round(random.uniform(0.30, 0.55), 2),
        "face_confidence": round(random.uniform(0.85, 0.98), 2),
        "head_pose_yaw_deg": round(random.uniform(0, 15), 1),
        "head_pose_pitch_deg": round(random.uniform(-10, 10), 1),
        "motion_score": round(random.uniform(0.05, 0.20), 2),
        "bg_complexity": random.choice(["solid_dark", "solid_light", "gradient"]),
        "bg_complexity_score": round(random.uniform(0.02, 0.12), 2),
        "mouth_open_ratio": round(random.uniform(0.35, 0.55), 2),
        "blink_rate_hz": round(random.uniform(0.20, 0.35), 2),
        "audio_snr_db": round(random.uniform(22, 35), 1),
        "transcript_word_count": random.randint(25, 70),
        "transcript_confidence": round(random.uniform(0.88, 0.98), 2),
        "lighting_uniformity": round(random.uniform(0.70, 0.90), 2),
        "occlusion_present": False,
        "environment_tag": random.choice(EASY_ENVS)
    }

def generate_easy_reject():
    """Generate a clip that should clearly be REJECT."""
    # Start with a base clip
    clip = generate_easy_keep()
    clip["clip_id"] = f"syn_easy_rej_{uuid4().hex[:8]}"
    
    # Add 2-3 hard reject signals
    reject_type = random.choice(["occlusion", "motion", "face", "audio"])
    
    if reject_type == "occlusion":
        clip["occlusion_present"] = True
    elif reject_type == "motion":
        clip["motion_score"] = round(random.uniform(0.50, 0.75), 2)
        clip["face_confidence"] = round(random.uniform(0.55, 0.62), 2)
    elif reject_type == "face":
        clip["face_area_ratio"] = round(random.uniform(0.08, 0.15), 2)
        clip["face_confidence"] = round(random.uniform(0.50, 0.62), 2)
    elif reject_type == "audio":
        clip["audio_snr_db"] = round(random.uniform(5, 12), 1)
        clip["transcript_confidence"] = round(random.uniform(0.45, 0.65), 2)
    
    return clip
```

---

### Medium Clips

**Goal**: 1-2 features at threshold boundaries, creating ambiguity.

```python
def generate_medium():
    """Generate a clip with 1-2 borderline features."""
    clip = generate_easy_keep()
    clip["clip_id"] = f"syn_med_{uuid4().hex[:8]}"
    
    # Pick 1-2 features to put in borderline zone
    borderline_features = random.sample([
        "face_area_ratio",
        "motion_score", 
        "audio_snr_db",
        "head_pose_yaw_deg",
        "bg_complexity_score"
    ], k=random.randint(1, 2))
    
    for feature in borderline_features:
        if feature == "face_area_ratio":
            clip["face_area_ratio"] = round(random.uniform(0.19, 0.24), 2)
        elif feature == "motion_score":
            clip["motion_score"] = round(random.uniform(0.28, 0.42), 2)
        elif feature == "audio_snr_db":
            clip["audio_snr_db"] = round(random.uniform(15, 19), 1)
        elif feature == "head_pose_yaw_deg":
            clip["head_pose_yaw_deg"] = round(random.uniform(22, 32), 1)
        elif feature == "bg_complexity_score":
            clip["bg_complexity_score"] = round(random.uniform(0.18, 0.35), 2)
    
    return clip
```

---

### Hard Clips

**Goal**: 2-3 conflicting signals + novel environment.

```python
def generate_hard():
    """Generate a clip with multiple conflicting signals."""
    clip = generate_easy_keep()
    clip["clip_id"] = f"syn_hard_{uuid4().hex[:8]}"
    clip["environment_tag"] = random.choice(NOVEL_ENVS)
    
    # Create conflict: good in some areas, bad in others
    # This makes the decision genuinely ambiguous
    
    conflict_pattern = random.choice([
        "good_face_bad_audio",
        "good_audio_bad_background",
        "good_overall_novel_env",
        "borderline_everything"
    ])
    
    if conflict_pattern == "good_face_bad_audio":
        # Strong face signals, weak audio
        clip["face_area_ratio"] = round(random.uniform(0.35, 0.50), 2)
        clip["face_confidence"] = round(random.uniform(0.88, 0.95), 2)
        clip["audio_snr_db"] = round(random.uniform(15, 18), 1)
        clip["bg_complexity_score"] = round(random.uniform(0.25, 0.38), 2)
        
    elif conflict_pattern == "good_audio_bad_background":
        # Clean audio but complex/distracting background
        clip["audio_snr_db"] = round(random.uniform(24, 32), 1)
        clip["bg_complexity_score"] = round(random.uniform(0.32, 0.42), 2)
        clip["head_pose_yaw_deg"] = round(random.uniform(25, 33), 1)
        
    elif conflict_pattern == "good_overall_novel_env":
        # All features look good, but environment is unusual
        # Tests whether agent over-weights environment_tag
        clip["environment_tag"] = random.choice(["underwater", "greenscreen_partial"])
        
    elif conflict_pattern == "borderline_everything":
        # Multiple features right at threshold
        clip["face_area_ratio"] = round(random.uniform(0.23, 0.27), 2)
        clip["audio_snr_db"] = round(random.uniform(18, 22), 1)
        clip["motion_score"] = round(random.uniform(0.23, 0.27), 2)
        clip["lighting_uniformity"] = round(random.uniform(0.62, 0.68), 2)
    
    return clip
```

---

## Seed Ground Truth (20 Clips)

When creating `data/seed_gt.json`, include a balanced set:

```json
{
  "seed_keep_01": {"label": "KEEP", "source": "seed", "episode": 0},
  "seed_keep_02": {"label": "KEEP", "source": "seed", "episode": 0},
  "seed_keep_03": {"label": "KEEP", "source": "seed", "episode": 0},
  "seed_keep_04": {"label": "KEEP", "source": "seed", "episode": 0},
  "seed_keep_05": {"label": "KEEP", "source": "seed", "episode": 0},
  "seed_keep_06": {"label": "KEEP", "source": "seed", "episode": 0},
  "seed_keep_07": {"label": "KEEP", "source": "seed", "episode": 0},
  "seed_reject_01": {"label": "REJECT", "source": "seed", "episode": 0},
  "seed_reject_02": {"label": "REJECT", "source": "seed", "episode": 0},
  "seed_reject_03": {"label": "REJECT", "source": "seed", "episode": 0},
  "seed_reject_04": {"label": "REJECT", "source": "seed", "episode": 0},
  "seed_reject_05": {"label": "REJECT", "source": "seed", "episode": 0},
  "seed_reject_06": {"label": "REJECT", "source": "seed", "episode": 0},
  "seed_reject_07": {"label": "REJECT", "source": "seed", "episode": 0},
  "seed_border_01": {"label": "BORDERLINE", "source": "seed", "episode": 0},
  "seed_border_02": {"label": "BORDERLINE", "source": "seed", "episode": 0},
  "seed_border_03": {"label": "BORDERLINE", "source": "seed", "episode": 0},
  "seed_border_04": {"label": "BORDERLINE", "source": "seed", "episode": 0},
  "seed_border_05": {"label": "BORDERLINE", "source": "seed", "episode": 0},
  "seed_border_06": {"label": "BORDERLINE", "source": "seed", "episode": 0}
}
```

Distribution: 7 KEEP, 7 REJECT, 6 BORDERLINE

---

## Feature Correlations to Maintain

When generating clips, maintain realistic correlations:

### Positive Correlations
- `face_area_ratio` ↔ `face_confidence` (larger faces → higher confidence)
- `audio_snr_db` ↔ `transcript_confidence` (cleaner audio → better ASR)
- `lighting_uniformity` ↔ `face_confidence` (better lighting → better detection)

### Negative Correlations
- `head_pose_yaw_deg` ↔ `face_area_ratio` (turned head → smaller apparent face)
- `motion_score` ↔ `face_confidence` (more motion → less reliable detection)
- `bg_complexity_score` ↔ `face_confidence` (busy background → more false positives)

```python
def apply_correlations(clip):
    """Adjust features to maintain realistic correlations."""
    # If head is turned, reduce apparent face area
    if clip["head_pose_yaw_deg"] > 25:
        clip["face_area_ratio"] *= random.uniform(0.85, 0.95)
    
    # High motion degrades face confidence
    if clip["motion_score"] > 0.35:
        clip["face_confidence"] = min(clip["face_confidence"], 
                                       random.uniform(0.70, 0.82))
    
    # Good audio correlates with good transcript
    if clip["audio_snr_db"] >= 22:
        clip["transcript_confidence"] = max(clip["transcript_confidence"],
                                            random.uniform(0.85, 0.95))
    
    return clip
```

---

## Validation

Before using generated clips:

```python
def validate_clip(clip: dict) -> list[str]:
    """Return list of validation errors, empty if valid."""
    errors = []
    
    required_fields = [
        "clip_id", "duration_s", "fps", "resolution", "face_area_ratio",
        "face_confidence", "head_pose_yaw_deg", "head_pose_pitch_deg",
        "motion_score", "bg_complexity", "bg_complexity_score",
        "mouth_open_ratio", "blink_rate_hz", "audio_snr_db",
        "transcript_word_count", "transcript_confidence",
        "lighting_uniformity", "occlusion_present", "environment_tag"
    ]
    
    for field in required_fields:
        if field not in clip:
            errors.append(f"Missing required field: {field}")
    
    # Range checks
    if clip.get("face_area_ratio", 0) < 0 or clip.get("face_area_ratio", 0) > 1:
        errors.append("face_area_ratio must be 0-1")
    if clip.get("face_confidence", 0) < 0 or clip.get("face_confidence", 0) > 1:
        errors.append("face_confidence must be 0-1")
    if clip.get("duration_s", 0) <= 0:
        errors.append("duration_s must be positive")
    
    return errors
```

---

## Batch Generation

```python
def generate_batch(n_easy=100, n_medium=50, n_hard=30):
    """Generate a balanced batch of clips for testing."""
    clips = []
    
    # Easy: half KEEP, half REJECT
    for _ in range(n_easy // 2):
        clips.append(generate_easy_keep())
    for _ in range(n_easy // 2):
        clips.append(generate_easy_reject())
    
    # Medium: all borderline
    for _ in range(n_medium):
        clips.append(generate_medium())
    
    # Hard: conflicting signals
    for _ in range(n_hard):
        clips.append(generate_hard())
    
    # Apply correlations
    clips = [apply_correlations(c) for c in clips]
    
    # Validate
    for clip in clips:
        errors = validate_clip(clip)
        if errors:
            raise ValueError(f"Invalid clip {clip['clip_id']}: {errors}")
    
    return clips
```

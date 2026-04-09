# ClipQualityEnv ICL-RL Batch Execution Plan

## 1) Goal
Implement an in-context RL style workflow for clip classification (no model weight updates) where one click executes classification for all 5 clips in the active scenario, then grades the full batch with penalties for wrong labels.

Target user flow:
1. Select scenario difficulty (easy/medium/hard)
2. Click Initialize Scenario
3. Optionally click Load Quality Hint
4. Click Execute Strategic Step once
5. System predicts labels for all 5 clips and submits all to environment grading

## 2) How ICL-RL Maps To This Project
This repository already has key pieces for RLVR-style behavior:
- Deterministic reward decomposition in clip_quality_env/grader.py
- Episode history in clip_quality_env/env.py via state.episode_history and info.session_history
- 5-step episode plan in clip_quality_env/env.py (EPISODE_STEPS = 5)

To mirror ICL-RL (contextual learning without fine-tuning):
- Keep the model frozen (Transformers inference only)
- Build prompt context from prior predictions + rewards + mistakes from the same episode
- Use deterministic grader output as verifiable reward signal
- Use reward feedback to update the next prediction prompt within the same batch run

This gives contextual adaptation during inference, not gradient descent training.

## 3) Model Recommendation (Hugging Face Transformers)
Primary recommendation:
- Qwen/Qwen2.5-32B-Instruct

Why this is best fit here:
- Strong instruction-following and structured JSON output reliability
- Good trade-off reasoning on borderline cases
- Better cost/latency balance than 70B for repeated 5-clip runs
- Easy to use with transformers AutoTokenizer + AutoModelForCausalLM

Fallbacks:
- meta-llama/Llama-3.3-70B-Instruct (highest quality when multi-GPU budget is available)
- Qwen/Qwen2.5-14B-Instruct (good quality with lower memory/latency requirements)
- mistralai/Mistral-Small-Instruct-2409 (fast fallback when resources are limited)

Transformers integration recommendation:
- Use quantized inference (4-bit) when GPU memory is constrained
- Enforce structured output with JSON schema-like prompt constraints
- Add deterministic fallback heuristic from inference.py when parse fails/timeouts

## 4) Architecture Changes

### 4.1 New batch predictor service
Add a dedicated module, for example:
- server/strategic_batch_predictor.py

Responsibilities:
- Load HF Transformers model once per process
- For each clip, generate:
  - chosen_label: KEEP/BORDERLINE/REJECT
  - confidence: [0,1]
  - reasoning
  - optional candidate_scores for all 3 labels
- Inject contextual history after each graded step:
  - previous clip id
  - submitted label
  - expected label if available in feedback scope
  - received reward and score breakdown
  - short "improvement directive" for next clip

Output contract per clip:
- Compatible with Action model in clip_quality_env/models.py

### 4.2 Remove manual label selection from UI
Update server/app.py:
- Remove easy_label_input, medium_label_input, hard_label_input (Predicted Label radios)
- Keep reasoning helper fields (easy observation, medium reasoning fields, hard tradeoff fields)
- Keep Load Quality Hint optional

Behavior change in handle_step:
- Current: submits one Action built from manually selected label
- New: runs a full episode batch loop (up to remaining clips, typically 5 after reset)

Pseudo-flow in handle_step:
1. Resolve env and ensure scenario is initialized
2. Build optional strategy context from selected tab text fields + hint
3. Loop while not done:
   - read current clip from observation
   - call predictor to get action
   - submit env.step(action)
   - append reward feedback to predictor context
4. Return final observation, updated history, reward summary, and raw JSON

Important implementation note:
- Keep environment step semantics intact (one clip per env.step)
- Implement batch behavior by looping env.step in UI/controller layer, not by bypassing grader logic

### 4.3 Add explicit batch grading penalty
Wrong labels are already penalized per step (label_score in grader), but requirement asks explicit penalty if any clip is wrong.

Implement episode-level batch penalty in clip_quality_env/env.py finalization:
- Compute mismatch_count from state.episode_history (submitted vs expected)
- Add a batch penalty term, for example:
  - batch_penalty = 0.05 * mismatch_count
- Add adjusted episode total in info:
  - episode_raw_total_reward
  - episode_batch_penalty
  - episode_adjusted_total_reward

Suggested formula:
- adjusted_total = max(0.0, raw_total - batch_penalty)

Keep per-step reward unchanged to preserve deterministic grader behavior and compatibility.

### 4.4 Optional grader extension (if stronger penalty needed)
If you want penalty to affect each step instead of only final episode summary:
- Extend clip_quality_env/grader.py with a penalty hook input (error_rate_so_far)
- Or add a post-grade adjustment in env.step based on running mismatch ratio

Start with episode-level penalty first (lower risk, easier rollout).

## 5) File-Level Change List

Primary implementation files:
- server/app.py
  - remove label radios
  - replace single-action handle_step with batch loop
  - add batch prediction result rendering
- clip_quality_env/env.py
  - add episode-level mismatch penalty and new info fields
  - keep step() deterministic and compatible
- clip_quality_env/grader.py
  - no required change for v1 (already deterministic RLVR)
  - optional extension for per-step penalty shaping
- inference.py
  - reuse fallback heuristics and normalization logic
  - optionally share prompt/history utilities with new batch predictor
- clip_quality_env/models.py
  - optional new model(s) for batch diagnostics if UI/API needs typed payloads

Recommended new file:
- server/strategic_batch_predictor.py

## 6) Prompting Strategy For In-Context Improvement
For each clip prompt, include:
- Rubric summary
- Current clip metadata
- Compact recent history from prior clips in same episode:
  - label, reward, mismatch/match, reasoning weakness tags
- Hard constraint to output strict JSON

Add short strategy rules:
- Avoid vague language
- Tie reasoning to concrete clip features
- Explain tradeoffs on borderline/hard clips
- If reward was low previously, explicitly revise feature weighting

## 7) API and UX Output Changes
In response/diagnostic payload (server/app.py UI state):
- Include per-clip batch decisions list with:
  - clip_id
  - chosen_label
  - confidence
  - reward
  - expected_label
  - is_match
  - optional candidate_scores
- Surface final batch summary:
  - clips_processed
  - raw_total_reward
  - batch_penalty
  - adjusted_total_reward

## 8) Test Plan

### 8.1 Update route/UI tests
Modify tests/test_server_routes.py:
- Remove assumptions about manual label radio inputs
- Add test: one Execute Strategic Step click processes 5 clips in one run
- Add test: optional quality hint does not block batch execution

### 8.2 Environment tests
Modify tests/test_environment.py:
- Add test for mismatch_count and batch penalty fields in final episode info
- Add test that adjusted total reward decreases as wrong labels increase
- Preserve existing checks for step reward range and deterministic behavior

### 8.3 Predictor tests
Add tests/test_batch_predictor.py:
- JSON parse robustness
- fallback label path when model output is invalid
- history-context injection after each step
- deterministic behavior under mocked model responses

### 8.4 Regression checks
Keep passing:
- tests/test_grader.py (deterministic score bounds)
- tests/test_inference.py (fallback and logging structure)

## 9) Rollout Plan
1. Build predictor module and unit tests first
2. Switch server/app.py to batch execution loop behind feature flag
3. Add episode-level penalty fields in env.py
4. Update dashboard rendering for per-clip batch results
5. Run full test suite and baseline smoke checks
6. Remove feature flag after validation

## 10) Acceptance Criteria
- Predicted Label radio controls are removed from strategic refinement UI
- One click on Execute Strategic Step runs classification for all 5 clips in active scenario
- Optional quality hint remains usable but not required
- Wrong labels in the 5-clip run reduce final batch score via explicit penalty
- Existing deterministic grading logic remains authoritative and reproducible

## 11) Open Questions / Assumptions
1. Should one-click execution always force a fresh 5-clip episode, or continue from current remaining steps if the user already submitted some clips?
2. Should batch penalty be additive (raw_total - k*mismatches) or multiplicative (raw_total * accuracy_factor)?
3. Do you want candidate scores for KEEP/BORDERLINE/REJECT displayed in UI, or only final chosen label per clip?

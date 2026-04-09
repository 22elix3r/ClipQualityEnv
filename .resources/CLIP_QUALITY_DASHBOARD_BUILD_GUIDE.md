# ClipQualityEnv — Dashboard Upgrade Build Guide
> **Reference**: [PolicyEvolverEnv](https://huggingface.co/spaces/luciferai-devil/devil-policyevolverenv)  
> **Goal**: Match its feature depth and UX polish while keeping the clip-quality classification domain.  
> Each section is written as a **prompt** you can hand directly to an AI coding assistant.

---

## What PolicyEvolverEnv Does That ClipQualityEnv Doesn't

| Feature | PolicyEvolverEnv ✅ | ClipQualityEnv ❌ |
|---|---|---|
| Themed UI (Soft theme + orange accent) | ✅ | ❌ Plain default |
| ≥5 steps per session | ✅ | ❌ 1 step only |
| Mode-specific structured input tabs | ✅ | ❌ Single flat form |
| "Load Expert Suggestion" shortcut | ✅ | ❌ Not present |
| Live reward decomposition display | ✅ | ❌ Single number only |
| Persistent session history tab | ✅ | ❌ Not present |
| LLM baseline callable from UI | ✅ | ❌ API-only |
| Dominant feature hints for current clip | ✅ | ❌ Not surfaced |
| Per-session state isolation | ✅ | ❌ Singleton shared state |
| Full corpus visible in Clip Queue | ✅ | ❌ Capped at 5–10 rows |

---

## Modification 1 — Visual Theme & Branding

> **Prompt:**
>
> Apply the Gradio `Soft` theme with an orange primary hue to the entire dashboard. Update the page title to "CLIP Quality Analyzer: Judge's Console". Replace the current plain `gr.Blocks()` call with one that passes `theme=gr.themes.Soft(primary_hue="orange", font=gr.themes.GoogleFont("Source Sans Pro"))`. Change the main heading HTML to use a dark charcoal color (`#1a1a2e`) and center it. Add a subtitle line: *"Use this console to inspect sampled clip metadata and classify each clip against the active quality rubric."* The left panel variant should be `"panel"` with a light background to visually separate it from the main content area.

---

## Modification 2 — Multi-Step Episode Structure (5 steps minimum)

> **Prompt:**
>
> The environment currently creates a session with only 1 classification step because `_choose_tasks()` returns a single task, resulting in `max_steps=1`. Change this so that when a user initializes a session for `task_easy`, `task_medium`, or `task_hard`, the environment samples **5 clips** from that task's corpus and presents them sequentially — one per step. The `reset()` call should produce an episode plan of 5 clips all from the same selected task rather than 1 clip per task. Update the `Remaining Analysis Steps` display to start at 5. After each submission the tabular view should update to show the submitted clip's `review_status` as the label provided. After all 5 steps are complete the session should show a final episode summary with total accumulated reward.

---

## Modification 11 — Display Full Clip Corpus in the Queue Table

> **Prompt:**
>
> The Clip Queue (Tabular View) currently only renders a slice of the corpus — the environment's `_state_to_observation()` method caps `shown = corpus[:10]`, and `format_obs()` in the Gradio app only iterates over `data_corpus` which inherits that slice. This means the user sees at most 5–10 clips even when the full corpus holds many more.
>
> Change both layers so the full corpus is always displayed: first, update `_state_to_observation()` to pass the complete unsliced corpus list (remove the `[:10]` cap and update `corpus_shown` to reflect the full count). Second, update `format_obs()` in `server/app.py` to iterate over all items in `data_corpus` without any limit. The Clip Queue table should show **every clip in the task corpus** (up to all available entries — currently 20 clips in `real_clips_manifest.jsonl`) ordered by clip ID. The `Current Review Status` column should update from `pending` to the submitted label each time a step is completed, giving the reviewer a real-time overview of the entire queue at a glance.

---

## Modification 3 — Difficulty-Tiered Input Panels

> **Prompt:**
>
> The current classification form is a single flat row (radio + textarea + slider + textbox). Replace it with a mode-switched tab panel that mirrors the difficulty of the selected scenario:
>
> - **Easy mode tab** — "Quick Classification": show the `Predicted Label` radio and a short `Key Observation` single-line text field (e.g. "face_confidence is high"). This is for clips with obviously clear signals.
> - **Medium mode tab** — "Balanced Assessment": show the `Predicted Label` radio, a `Primary Signal` field (the main deciding metric), a `Conflicting Signal` field (the metric that complicates the decision), and a multi-line `Reasoning` textarea.
> - **Hard mode tab** — "Full Trade-off Analysis": show the `Predicted Label` radio, a `Trade-off Summary` textarea pre-labelled "Describe the conflicting quality signals", a `Confidence Justification` field, and the `Confidence` slider.
>
> The selected tab should auto-switch when the user picks a scenario from the dropdown (`task_easy` → Easy tab, `task_medium` → Medium tab, `task_hard` → Hard tab). All tabs feed into the same `handle_step()` function by merging their fields into a single `reasoning` string before submission.

---

## Modification 4 — "Load Quality Hint" Button

> **Prompt:**
>
> Add a "💡 Load Quality Hint" button next to the Submit button. When clicked, it should auto-populate the reasoning field with a template string derived from the current clip's dominant features and their rubric status. The hint should take the form:
> `"[feature_name] is [value] which is [above/below/within] the [KEEP/BORDERLINE/REJECT] threshold. [second_feature_name] is [value] which is [directional cue]."`
>
> Source the dominant features from the rubric by checking which features have values closest to a tier boundary for the current clip. This gives users a scaffold to write better reasoning without giving away the correct answer. The hint button should have a secondary/ghost style so it doesn't compete visually with the primary Submit button.

---

## Modification 5 — Live Reward Decomposition Panel

> **Prompt:**
>
> After each classification submission, instead of showing a single reward number, display a three-row breakdown panel in the left sidebar showing:
> - **Format Score**: `0.10` (green) or `0.00` (red) — did the submission include a valid label, non-empty reasoning, and valid confidence?
> - **Label Score**: `0.60`, `0.25`, or `0.00` — was the label correct, adjacent, or wrong?
> - **Reasoning Score**: `0.00`–`0.30` — how many dominant features were correctly referenced with directional cues?
>
> Show these as three separate `gr.Number` or small Markdown cards with color-coded values (green for full points, orange for partial, red for zero). Below them show the total reward in large bold text. This decomposition is already computed by the grader — it returns a `Reward` object with `format_score`, `label_score`, and `reasoning_score` fields. Surface all three in the observation info dict and pass them through to the UI.

---

## Modification 6 — Dominant Features Hint Panel (Post-Init)

> **Prompt:**
>
> After a session is initialized, add a "🎯 Key Signals for This Clip" section below the Rubric Summary tab. This section should list the 2–3 features that are most determinative for the current clip — i.e. the features that the grader's `get_dominant_features()` method returns for this clip. Display them as a small table with columns: Feature Name, Current Value, Rubric Status (KEEP/BORDERLINE/REJECT), and Threshold Range. This directly tells the reviewer which fields to focus on without giving away the expected label. Populate this table as part of the `handle_reset()` return values.

---

## Modification 7 — Session History Tab

> **Prompt:**
>
> Add a fourth tab to the right-column tab group called "📊 Session History". This tab should show a live-updating table of every classification action taken in the current session with columns: Step, Clip ID, Submitted Label, Expected Label, and Reward. After each `handle_step()` call, append a new row to this table. The table should persist for the entire session and only clear when "Initialize Session" is clicked again. Highlight rows in green where the submitted label matches the expected label, and red where it does not. At the bottom of the tab show the running session total reward.

---

## Modification 8 — LLM Baseline Button in UI

> **Prompt:**
>
> Add a "🤖 Run LLM Baseline Agent" button to the left sidebar below the "Initialize Session" button. When clicked, it should call the existing `inference.run_baseline(task=task_id)` function in a non-blocking way (use Gradio's queue and generator pattern or a background thread). While running, show a spinner / loading indicator replacing the button text. When complete, display the result in a new collapsible `gr.Accordion` panel titled "LLM Baseline Result" showing:
> - Model name used
> - Whether LLM or deterministic fallback was used
> - Reward achieved per step
> - Success: Yes / No
>
> If `HF_TOKEN` is not set, show a warning message inside the accordion: "LLM unavailable — no HF_TOKEN configured. Showing deterministic fallback result." instead of silently running the fallback.

---

## Modification 9 — Per-Session State Isolation

> **Prompt:**
>
> The current `ClipQualityEnvironment` is implemented as a Python module-level singleton, meaning all browser users share the same environment state. When one user calls `reset()`, it corrupts another user's active session. Fix this by removing the singleton `__new__` pattern and instead managing per-session environment instances inside the Gradio `build_custom_ui()` function using Gradio's `gr.State` component. Store the environment instance in a `gr.State` object so each browser tab gets its own isolated environment. Pass the state object as an input/output to `handle_reset()` and `handle_step()`. This requires converting both handler functions to accept and return the environment state as their first argument.

---

## Modification 10 — Improved `/baseline` API Endpoint

> **Prompt:**
>
> The current `GET /baseline` endpoint is synchronous and blocks the server for several seconds while running the LLM agent across all tasks. Refactor it to use FastAPI's `BackgroundTasks` pattern: calling `POST /baseline/start` should enqueue the baseline run and return a `run_id` immediately. A subsequent `GET /baseline/status/{run_id}` endpoint should return the current status (`running`, `complete`, `failed`) and the partial or full results. Store results in an in-memory dict keyed by `run_id` with a TTL of 10 minutes. This prevents the endpoint from timing out under Hugging Face Space's request limits.

---

## Implementation Order (Recommended)

```
Phase 1 — Foundation
  [1] Visual Theme & Branding
  [9] Per-Session State Isolation   ← do this before any UX work

Phase 2 — Core UX
  [2] Multi-Step Episode Structure
  [11] Full Corpus Display in Clip Queue
  [5] Live Reward Decomposition
  [7] Session History Tab

Phase 3 — Intelligence Layer
  [3] Difficulty-Tiered Input Panels
  [4] Load Quality Hint Button
  [6] Dominant Features Hint Panel

Phase 4 — LLM Integration
  [8] LLM Baseline Button in UI
  [10] Non-blocking /baseline API
```

---

## Reference Mapping: PolicyEvolverEnv → ClipQualityEnv

| PolicyEvolverEnv concept | ClipQualityEnv equivalent |
|---|---|
| "Deployment Scenario" | "Clip-Quality Scenario" |
| "Incident Corpus" | "Clip Queue" |
| "Active Framework" (policy text) | "Active Rubric Summary" |
| "Clarification / New Rule / Evolution" tabs | "Quick / Balanced / Trade-off" tabs |
| "Strategic Thought Process" textarea | "Reasoning" textarea |
| "Execute Strategic Step" | "Submit Classification Action" |
| "Load Expert Suggestion" | "Load Quality Hint" |
| Framework score improving over steps | Best Quality Score + per-step rewards |

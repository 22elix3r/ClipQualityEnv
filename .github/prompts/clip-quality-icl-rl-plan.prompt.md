---
name: clip-quality-icl-rl-plan
description: "Create a project plan for ICL-RL style batch clip labeling and grading updates"
argument-hint: "Reference text and constraints for the plan"
agent: agent
---
Create or update a single markdown plan file for this repository.

Use the user-provided argument as the reference concept, but do not copy text 1:1.

Core objective:
Implement an in-context reinforcement-learning style workflow for clip classification where the model remains frozen and improves by using history and reward feedback in context.

Required plan coverage:
1. Explain the ICL-RL adaptation for this repository:
- Frozen policy (no weight updates)
- Contextual feedback loop across steps
- Deterministic RLVR grading

2. Recommend Hugging Face Transformers-compatible models:
- One primary model
- At least two fallback options
- Include why each fits clip-quality reasoning and expected deployment constraints

3. Include required product flow changes:
- Remove manual Predicted Label selectors from the strategic refinement section
- User flow: select task difficulty, initialize scenario, optional quality hint, execute strategic step once
- On one execution, predict and submit labels for all 5 displayed clips
- Ensure wrong labels for any clip are penalized during grading

4. Include file-level implementation plan:
- server/app.py
- clip_quality_env/env.py
- clip_quality_env/grader.py
- inference.py or a new predictor module if needed
- tests/

5. Include testing and acceptance criteria:
- Unit tests for batch execution path
- Tests for penalty behavior on wrong labels
- Regression checks for existing endpoints and reward fields

Output format requirements:
- Keep the plan actionable and implementation-oriented
- Use clear sections: goals, architecture, file changes, scoring changes, model strategy, test plan, rollout, risks
- Include a short list of open questions/assumptions at the end

Before finalizing, inspect current project files so the plan matches existing code paths and naming.

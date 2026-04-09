# Hackathon Compliance Audit — `clip_quality_env`

> Cross-referencing project workspace against [hackathon_requirements.md](file:///home/elix3r/projects/clip_quality_env/hackathon_requirements.md)
> Audit date: 2 April 2026

---

## Legend

| Status | Meaning |
|---|---|
| ✅ PASS | Requirement fully met |
| ❌ FAIL | Missing or non-compliant — **must fix before submission** |
| ⚠️ WARN | Partially met — should improve, not a hard blocker |

---

## 1. OpenEnv SDK Integration

| # | Requirement | Status | Finding |
|---|---|---|---|
| 1.1 | `openenv-core` listed as dependency | ❌ FAIL | `requirements.txt` does **not** include `openenv-core`. Only has `pydantic`, `openai`, `fastapi`, `uvicorn`, etc. |
| 1.2 | Environment inherits from `openenv.core.environment.Environment` | ❌ FAIL | `ClipQualityEnv` is a plain Python class — does **not** inherit from the OpenEnv `Environment` base class. |
| 1.3 | Models inherit from `openenv.core.models.Action` / `Observation` | ❌ FAIL | `Action` and `Observation` inherit from `pydantic.BaseModel` — not from `openenv.core.models`. |
| 1.4 | `step()` returns `StepResult` (OpenEnv type) | ❌ FAIL | `step()` returns a raw `tuple[Observation, float, bool, dict]` instead of an OpenEnv `StepResult` object. |

### Remediation

```python
# 1. Add to requirements.txt:
openenv-core

# 2. In clip_quality_env/env.py — change:
class ClipQualityEnv:
# To:
from openenv.core.environment import Environment
class ClipQualityEnv(Environment):

# 3. In clip_quality_env/models.py — change:
from pydantic import BaseModel
class Action(BaseModel): ...
class Observation(BaseModel): ...
# To:
from openenv.core.models import Action as BaseAction, Observation as BaseObservation
class Action(BaseAction): ...
class Observation(BaseObservation): ...

# 4. step() should return StepResult instead of tuple:
from openenv.core.models import StepResult
# return StepResult(observation=obs, reward=reward, done=done, info=info)
```

> **CAUTION:** This is the **single most critical gap**. Without inheriting from the OpenEnv base classes, the automated programmatic checks will fail and the submission will be **disqualified**.

---

## 2. Directory Structure

| # | Requirement | Status | Finding |
|---|---|---|---|
| 2.1 | `__init__.py` exports Action, Observation, Env | ✅ PASS | `clip_quality_env/__init__.py` exports `Action`, `Observation`, `ClipQualityEnv`. |
| 2.2 | `models.py` with Pydantic Action/Observation | ✅ PASS | Present at `clip_quality_env/models.py`. Well-structured with field validations. |
| 2.3 | `client.py` (EnvClient subclass) | ❌ FAIL | **No `client.py` exists** anywhere in the project. |
| 2.4 | `openenv.yaml` manifest | ✅ PASS | Present in root. Contains tasks, action/observation spaces, constraints, entrypoints. |
| 2.5 | `pyproject.toml` | ❌ FAIL | **Missing entirely.** Only `requirements.txt` exists. OpenEnv requires `pyproject.toml` as the primary dependency spec. |
| 2.6 | `server/` subdirectory | ❌ FAIL | **No `server/` directory exists.** `app.py` and `Dockerfile` are in root instead of `server/`. |
| 2.7 | `outputs/` directory (gitignored) | ⚠️ WARN | `state/` is used instead of the OpenEnv-standard `outputs/`. Not ignored properly (`.gitignore` lists `state/` but not `outputs/`). |
| 2.8 | `inference.py` in root | ✅ PASS | Present at project root with exact name. |
| 2.9 | `README.md` | ✅ PASS | Present and comprehensive. |
| 2.10 | `.dockerignore` | ✅ PASS | Present with reasonable exclusions. |

### Remediation

```bash
# 1. Create client.py:
touch clip_quality_env/client.py
# Implement: class ClipQualityClient(EnvClient): ...

# 2. Create pyproject.toml:
# See Section 7 of hackathon_requirements.md for template

# 3. Create server/ directory and move files:
mkdir -p server
# 4. Add outputs/ directory:
mkdir -p outputs/logs outputs/evals
echo "outputs/" >> .gitignore
```

---

## 3. API Contract (`reset`, `step`, `state`)

| # | Requirement | Status | Finding |
|---|---|---|---|
| 3.1 | `reset()` returns `Observation` | ✅ PASS | Returns `Observation` Pydantic model. |
| 3.2 | `step(action)` accepts `Action`, returns result | ⚠️ WARN | Accepts `Action | dict` (good), but returns raw tuple instead of `StepResult`. |
| 3.3 | `state()` returns `State` with `episode_id`, `step_count` | ⚠️ WARN | Returns a `dict` (not a typed `State` model). Contains `episode_count` and `current_step` but field names differ from OpenEnv convention (`episode_id`, `step_count`). |
| 3.4 | Methods are `async` | ❌ FAIL | All methods (`reset`, `step`, `state`) are **synchronous**. OpenEnv `Environment` base class expects `async` methods. |

### Remediation

```python
# In clip_quality_env/env.py, convert methods to async:
async def reset(self) -> Observation:
    ...
async def step(self, action: Action) -> StepResult:
    ...
async def state(self) -> State:
    ...

# And in server/app.py update to async endpoints:
@app.post("/reset")
async def reset():
    return await env.reset()
```

---

## 4. WebSocket Transport

| # | Requirement | Status | Finding |
|---|---|---|---|
| 4.1 | WebSocket `/ws` endpoint | ❌ FAIL | `app.py` uses only standard **HTTP REST** endpoints (`POST /reset`, `POST /step`, `GET /state`). No WebSocket endpoint exists. |
| 4.2 | `websockets` dependency | ❌ FAIL | Not listed in `requirements.txt`. |

### Remediation

If using `openenv-core`'s `create_env_app()`, WebSocket is handled automatically:

```python
# In server/app.py:
from openenv.core.env_server import create_env_app
from clip_quality_env.env import ClipQualityEnv
from clip_quality_env.models import Action, Observation

env = ClipQualityEnv()
app = create_env_app(env, Action, Observation)
# This auto-registers /ws endpoint + HTTP fallbacks
```

Alternatively, keep current HTTP endpoints **in addition** to adding a WebSocket handler for spec compliance.

---

## 5. `pyproject.toml`

| # | Requirement | Status | Finding |
|---|---|---|---|
| 5.1 | File exists | ❌ FAIL | **Missing.** |
| 5.2 | `requires-python = ">=3.10"` | ❌ FAIL | Can't be set without the file. |
| 5.3 | Lists `openenv-core`, `fastapi`, `uvicorn`, `pydantic` | ❌ FAIL | Can't be set without the file. |

### Remediation

Create `pyproject.toml`:

```toml
[project]
name = "clip-quality-env"
version = "1.0.0"
description = "Self-evolving RL environment for LoRA dataset clip quality classification."
requires-python = ">=3.10"

[project.dependencies]
openenv-core = "*"
pydantic = ">=2.0.0"
openai = ">=1.0.0"
fastapi = ">=0.104.0"
uvicorn = ">=0.24.0"

[project.optional-dependencies]
extractor = [
    "opencv-python-headless>=4.10.0",
    "numpy>=1.26.0",
]
spaces = [
    "gradio>=4.0.0",
]
dev = [
    "pytest>=7.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 6. `client.py` (EnvClient)

| # | Requirement | Status | Finding |
|---|---|---|---|
| 6.1 | `client.py` exists | ❌ FAIL | **File does not exist.** |
| 6.2 | Inherits from `EnvClient` | ❌ FAIL | N/A |
| 6.3 | Supports async and `.sync()` | ❌ FAIL | N/A |

### Remediation

Create `clip_quality_env/client.py`:

```python
from openenv.core.env_client import EnvClient
from .models import Action, Observation

class ClipQualityClient(EnvClient):
    """Client for ClipQualityEnv. Async-first, sync via .sync()."""
    action_type = Action
    observation_type = Observation
```

Update `clip_quality_env/__init__.py` to export:
```python
from .client import ClipQualityClient
```

---

## 7. Containerization (Docker)

| # | Requirement | Status | Finding |
|---|---|---|---|
| 7.1 | `Dockerfile` exists | ✅ PASS | Present in root (should be in `server/` per spec). |
| 7.2 | Builds successfully | ✅ PASS | Standard `python:3.11-slim` base, installs deps and copies code. |
| 7.3 | Port matches OpenEnv default (8000) | ⚠️ WARN | Exposes **7860** (HF Spaces convention) instead of OpenEnv's standard **8000**. |
| 7.4 | Base image is Python 3.10+ | ✅ PASS | Uses `python:3.11-slim`. |
| 7.5 | Runs `uvicorn server.app:app` | ⚠️ WARN | Runs `uvicorn app:app` (root-level) — needs to be `server.app:app` if files move to `server/`. |

### Remediation

After moving `app.py` to `server/app.py` and `Dockerfile` to `server/Dockerfile`:

```dockerfile
# Update CMD line:
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Keep the current `Dockerfile` working for HF Spaces (port 7860) as a separate deployment config if needed.

---

## 8. FastAPI Server (`app.py`)

| # | Requirement | Status | Finding |
|---|---|---|---|
| 8.1 | Uses `create_env_app()` from openenv-core | ❌ FAIL | Manually defines HTTP endpoints instead of using the OpenEnv server factory. |
| 8.2 | Registers `/ws` WebSocket endpoint | ❌ FAIL | Only HTTP endpoints exist. |
| 8.3 | Returns proper `StepResult` from `/step` | ⚠️ WARN | Returns a custom dict with `observation`, `reward`, `done`, `info` keys — close but not the `StepResult` schema. |
| 8.4 | Observation returned even when `done=True` | ❌ FAIL | Returns `None` for observation when `done=True`. OpenEnv expects a terminal observation. |

### Remediation

```python
# Option A: Full migration to OpenEnv server factory
from openenv.core.env_server import create_env_app
app = create_env_app(env, Action, Observation)

# Option B: Keep manual endpoints but fix the step response
@app.post("/step")
async def step(action: Action):
    result = await env.step(action)
    return result.model_dump()  # returns StepResult as dict
```

Also fix the `done=True` case — currently returns `None` for observation:

```python
# In app.py, line 29:
"observation": None if done else obs.model_dump(),
# Should be:
"observation": obs.model_dump(),  # Always return observation
```

---

## 9. Graders & Reward Logic

| # | Requirement | Status | Finding |
|---|---|---|---|
| 9.1 | 3+ tasks with graders | ✅ PASS | 3 tasks defined (easy, medium, hard) in `openenv.yaml` and code. |
| 9.2 | Grader returns `0.0-1.0` | ✅ PASS | `Reward.total` is clamped to `[0.0, 1.0]` in `grader.py`. |
| 9.3 | Deterministic grading | ✅ PASS | `grade()` is fully deterministic — no randomness. |
| 9.4 | Difficulty progression (easy, medium, hard) | ✅ PASS | Episodes follow fixed easy then medium then hard trajectory. |
| 9.5 | Meaningful partial-progress reward | ✅ PASS | Decomposed reward: format (0.10) + label (0.60) + reasoning (0.30). |
| 9.6 | Not hardcoded/static output | ✅ PASS | Reward varies based on action, clip, rubric state, and ground truth. |

**No changes needed. Grader implementation is strong.**

---

## 10. Inference Script (`inference.py`)

| # | Requirement | Status | Finding |
|---|---|---|---|
| 10.1 | Named exactly `inference.py` in root | ✅ PASS | Present. |
| 10.2 | Uses OpenAI client | ✅ PASS | `from openai import OpenAI` — uses `client.chat.completions.create()`. |
| 10.3 | Reads `API_BASE_URL` from env | ✅ PASS | Falls back to `https://router.huggingface.co/v1`. |
| 10.4 | Reads `MODEL_NAME` from env | ✅ PASS | Required — raises ValueError if missing. |
| 10.5 | Reads `HF_TOKEN` / `OPENAI_API_KEY` from env | ✅ PASS | Checks both. |
| 10.6 | 20-minute timeout guard | ✅ PASS | `TIMEOUT_SECONDS = 20 * 60` with 60s buffer. |
| 10.7 | Produces reproducible baseline scores | ✅ PASS | Outputs JSON with per-clip grades. |
| 10.8 | Prints scores for all 3 tasks | ✅ PASS | `per_clip_grades` covers easy/medium/hard. |

**No changes needed. Inference script is fully compliant.**

---

## 11. README Documentation

| # | Required Section | Status | Finding |
|---|---|---|---|
| 11.1 | Environment description and motivation | ✅ PASS | Section 1 covers this well. |
| 11.2 | Action and observation space definitions | ✅ PASS | Section 2 with tables. |
| 11.3 | Task descriptions with difficulty levels | ✅ PASS | Section 3 with table. |
| 11.4 | Setup and usage instructions | ✅ PASS | Section 4 — local, Docker, Spaces. |
| 11.5 | Baseline scores | ⚠️ WARN | Section 5 exists but only shows the CLI command — **no actual baseline score numbers** are listed. Should include real scores from a reference run. |

### Remediation

Run a baseline and add actual scores:

```markdown
## 5. Baseline Scores

| Difficulty | Mean Reward | Label Accuracy |
|---|---|---|
| Easy | 0.XX | XX% |
| Medium | 0.XX | XX% |
| Hard | 0.XX | XX% |

Model: `meta-llama/Llama-3.3-70B-Instruct` - 10 episodes
```

---

## 12. Deployment & Miscellaneous

| # | Requirement | Status | Finding |
|---|---|---|---|
| 12.1 | HF Space tagged `openenv` | ⚠️ WARN | README frontmatter has `emoji` but no explicit `tags: [openenv]` in the HF metadata header. |
| 12.2 | `openenv.yaml` has `name`, `description`, `tags` | ⚠️ WARN | Has `name` and `description` but **no `tags` field** in the manifest. |
| 12.3 | No hardcoded secrets | ✅ PASS | All secrets read from env vars. |
| 12.4 | `.gitignore` excludes `outputs/`, `.env`, `__pycache__` | ⚠️ WARN | Excludes `.env` and `__pycache__`. Excludes `state/` instead of `outputs/`. Missing `outputs/`. |
| 12.5 | Infrastructure: 2 vCPU / 8 GB RAM | ✅ PASS | `openenv.yaml` constraints: `cpu: 2`, `memory_gb: 8`. No GPU deps. |
| 12.6 | `openenv validate` passes | ❌ FAIL | Cannot pass without `openenv-core` installed and base class inheritance. |

### Remediation

```yaml
# In README.md frontmatter, add:
tags:
  - openenv
  - reinforcement-learning

# In openenv.yaml, add:
tags:
  - reinforcement-learning
  - openenv
  - clip-quality
  - hackathon-2026
```

---

## Summary: Priority Fix List

### Critical (Must Fix — Disqualification Risk)

| # | Item | File(s) Affected |
|---|---|---|
| 1 | **Add `openenv-core` dependency** | `requirements.txt`, new `pyproject.toml` |
| 2 | **Inherit from OpenEnv base classes** (`Environment`, `Action`, `Observation`) | `env.py`, `models.py` |
| 3 | **Create `pyproject.toml`** | New file in root |
| 4 | **Create `client.py`** with `EnvClient` subclass | New `clip_quality_env/client.py` |
| 5 | **Add WebSocket `/ws` endpoint** (or use `create_env_app()`) | `app.py` / `server/app.py` |
| 6 | **Convert methods to `async`** | `env.py` |
| 7 | **Return `StepResult` from `step()`** instead of raw tuple | `env.py` |
| 8 | **Fix `step()` to return observation when `done=True`** | `app.py` |

### Important (Should Fix — Score Impact)

| # | Item | File(s) Affected |
|---|---|---|
| 9 | **Restructure into `server/` directory** (`app.py`, `Dockerfile`, `requirements.txt`) | Multiple files move |
| 10 | **Add actual baseline scores to README** | `README.md` |
| 11 | **Add `tags` to `openenv.yaml`** | `openenv.yaml` |
| 12 | **Add `tags: [openenv]` to README HF frontmatter** | `README.md` |
| 13 | **Rename `state()` return fields** to match OpenEnv conventions (`episode_id`, `step_count`) | `env.py` |

### Nice-to-Have (Polish)

| # | Item | File(s) Affected |
|---|---|---|
| 14 | Add `outputs/` directory with `logs/` and `evals/` subdirs | New directories |
| 15 | Update `.gitignore` to include `outputs/` | `.gitignore` |
| 16 | Update `fastapi` version pin from `>=0.100.0` to `>=0.104.0` | `requirements.txt` |
| 17 | Update `uvicorn` version pin from `>=0.23.0` to `>=0.24.0` | `requirements.txt` |

---

## What's Already Strong

- **Grader system** — Fully deterministic, 3-component decomposition, `[0.0, 1.0]` range
- **3 tasks** with graduated difficulty (easy, medium, hard)
- **`inference.py`** — Complete, reads env vars, OpenAI client, timeout protection, per-clip output
- **`openenv.yaml`** — Well-structured with observation/action spaces and constraints
- **README** — Comprehensive (just needs baseline scores)
- **Real-clip support** — Manifest-based with metadata extraction pipeline
- **Ground-truth co-evolution** — Unique self-improving design
- **Rubric recalibration** — Non-stationary environment (interesting for RL)
- **Spaces app** — Full Gradio UI with upload, grading, and state persistence

The core environment logic is solid. The gaps are primarily **framework integration** (inheriting from OpenEnv base classes and using its transport layer) rather than functionality.

# Meta PyTorch OpenEnv AI Hackathon — India 2026

## Technical Requirements & Evaluation Criteria

> **Organizers:** Scaler School of Technology (SST) × Meta × PyTorch × Hugging Face
> **Prize Pool:** $30,000 · **Format:** Two-round national hackathon
> **Sources:** [Scaler Page](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon) · [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) · [HF openenv-course](https://github.com/huggingface/openenv-course)
> **Last synced:** 2026-04-02

---

## 1. Participation Requirements

| Criterion | Requirement |
|---|---|
| **Target Audience** | Developers, ML engineers, CS students in India |
| **Team Size** | Solo (1) or up to **3 members** |
| **RL Experience** | Not required — free prep courses provided |
| **Language** | Python proficiency (basic to intermediate) |
| **ML Familiarity** | Basic ML understanding |
| **Version Control** | Comfort with GitHub |
| **Registration** | Free via Scaler/Unstop portal |
| **Cross-institution** | Teams from different colleges/companies OK |

---

## 2. Round Structure & Timeline

### Round 1 — Online: Build Your Mini-RL Environment
- **Window:** Wed, 25 March – Wed, 8 April 2026
- **Results:** Fri, 10 April 2026
- **Deliverable:** A Mini-RL environment with defined tasks, graders, and reward logic on OpenEnv.
- **Evaluation:** Programmatic checks + LLM-based scoring.

### Advanced Bootcamp (Shortlisted Teams)
- Intensive sessions with Meta engineers before finale.
- Round 2 guidelines available from dashboard on 10 April.

### Round 2 — Grand Finale (In-Person, 48 Hours)
- **Date:** Sat, 25 April – Sun, 26 April 2026
- **Venue:** Scaler School of Technology, Bangalore
- **Judging:** Meta's global engineering team
- **Perks:** All meals, exclusive merch, live sessions from Meta/PyTorch/HF engineers

---

## 3. Core Technical Stack

### Language & Runtime

| Dependency | Version |
|---|---|
| Python | `>= 3.10` |
| pip / uv | Latest (`uv` recommended) |

### Core Framework Dependencies

| Package | Version | Purpose |
|---|---|---|
| `openenv-core` | Latest | OpenEnv SDK & base classes |
| `fastapi` | `>= 0.104.0` | Environment HTTP/WS server |
| `uvicorn` | `>= 0.24.0` | ASGI server |
| `pydantic` | `>= 2.0` | Type-safe models |
| `requests` | `>= 2.25.0` | HTTP client utilities |
| `websockets` | Latest | WebSocket transport |

### Tooling

| Tool | Purpose |
|---|---|
| `openenv` CLI | `openenv init`, `openenv push`, `openenv validate` |
| Docker / Docker Desktop | Container build & local run |
| `pyproject.toml` | Primary dependency spec |
| Hugging Face Spaces | Deployment target |

---

## 4. OpenEnv Framework Compliance

### What is OpenEnv?

OpenEnv is an e2e open-source framework by Meta × Hugging Face for creating **standardized, isolated, and reusable** RL environments. It uses Gymnasium-style APIs, Docker-first packaging, and HTTP-native deployment.

### API Contract (MANDATORY)

All environments **must** implement three core methods:

```python
from openenv.core.environment import Environment
from openenv.core.models import StepResult

class YourEnvironment(Environment):

    async def reset(self) -> Observation:
        """Initialize a new episode. Returns initial Observation."""
        ...

    async def step(self, action: Action) -> StepResult:
        """Execute one action. Returns StepResult(observation, reward, done, info)."""
        ...

    async def state(self) -> State:
        """Return episode metadata (episode_id, step_count, etc.)."""
        ...
```

### Client-Side Wrapper (Required)

```python
from openenv.core.env_client import EnvClient

class YourEnv(EnvClient):
    """Async-first client. Sync via .sync() wrapper."""
    pass
```

### Transport
- Operations use **WebSocket** (`/ws`) — not HTTP.
- ~0.1ms/frame overhead vs ~10-50ms TCP handshake per HTTP call.
- Async: `async with YourEnv(...) as env` — Sync: `YourEnv(...).sync()`

---

## 5. Environment Directory Structure

Scaffold with `openenv init my_env`:

```
my_env/
├── .dockerignore
├── __init__.py             # Exports: YourAction, YourObservation, YourEnv
├── models.py               # Pydantic Action, Observation, State
├── client.py               # YourEnv(EnvClient)
├── README.md               # Tasks, grader logic, usage
├── openenv.yaml            # Environment manifest
├── pyproject.toml          # Dependencies
├── inference.py            # Baseline inference script (MANDATORY — exact name)
├── outputs/                # Runtime outputs (gitignored)
│   ├── logs/
│   └── evals/
└── server/
    ├── your_environment.py # YourEnvironment(Environment)
    ├── app.py              # FastAPI app, WebSocket /ws
    ├── requirements.txt    # Docker deps (auto-gen from pyproject.toml)
    └── Dockerfile          # Container image
```

---

## 6. Functional Requirements (Round 1)

### 6.1 Real-World Task Simulation
- Must simulate a task **humans actually do** — not games, not toys.
- Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

### 6.2 Minimum 3 Tasks with Agent Graders
- Each task defines a concrete objective an agent must accomplish.
- Programmatic **agent grader** scores on `0.0–1.0` scale.
- Tasks must range in difficulty: **easy → medium → hard**.
- Graders must have **clear, deterministic** success/failure criteria.

### 6.3 Meaningful Reward Function
- Signal over the full trajectory (not just binary end-of-episode).
- Rewards **partial progress** toward task completion.
- Penalizes undesirable behavior (e.g., infinite loops, destructive actions).

### 6.4 Baseline Inference Script (`inference.py`)
- Uses the **OpenAI API client** to run a model against the environment.
- Reads API credentials from **environment variables** (see Section 8).
- Produces a **reproducible baseline score** on all 3 tasks.
- **Must be named exactly `inference.py`** in the root directory.

---

## 7. Non-Functional Requirements

### 7.1 Hugging Face Space Deployment
- Environment must run as a **containerized HF Space** tagged with `openenv`.

### 7.2 Containerized Execution
- Working **Dockerfile** included.
- Must start cleanly with `docker build` + `docker run`.

### 7.3 README Documentation
Root `README.md` must include:
1. Environment description and motivation.
2. Action and observation space definitions.
3. Task descriptions with expected difficulty levels.
4. Setup and usage instructions.
5. Baseline scores.

### 7.4 Infrastructure Constraints
- **Timeout:** `inference.py` must complete within **20 minutes**.
- **Resources:** Must run on **2 vCPUs / 8 GB RAM** (no GPU required).

---

## 8. Mandatory Environment Variables & Inference Client

| Variable | Purpose |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier used in the baseline script |
| `HF_TOKEN` | Hugging Face / API Key |

- **Inference Client:** Must use the **OpenAI Client** for all LLM calls.
- Baseline script reads API credentials from environment variables (`OPENAI_API_KEY` / `HF_TOKEN`).

---

## 9. Pre-Submission Validation (Automated Gate)

**Failure on any item = disqualification:**

1. **HF Space deploys:** Ping to Space URL returns HTTP `200`, responds to `reset()`.
2. **OpenEnv spec compliance:** `openenv.yaml` valid, typed models, `step()`/`reset()`/`state()` endpoints.
3. **Dockerfile builds:** Automated `docker build` on submitted repo succeeds.
4. **Baseline reproduces:** `inference.py` completes without error and produces scores.
5. **3+ tasks with graders:** All grader scores are in `[0.0, 1.0]`.

Run `openenv validate` before submitting.

---

## 10. Judging Process (3 Phases)

### Phase 1 — Automated Validation (Pass/Fail Gate)
- HF Space deploys and responds to OpenEnv endpoints.
- OpenEnv spec compliance check.
- Dockerfile builds successfully.
- `inference.py` reproduces without errors.
- 3+ tasks enumerated, graders return valid `0.0–1.0` scores.

### Phase 2 — Agentic Evaluation (Scored)
- Rerun of the baseline agent.
- A standard Open LLM agent (e.g., **Nemotron 3 Super**) runs against all environments.
- Score variance check performed.

### Phase 3 — Human Review (Top Submissions)
- Reviewed by **Meta and Hugging Face engineers**.
- Assessed for real-world utility, creativity, and exploit/hardcoded-grader checks.

---

## 11. Official Scoring Criteria

| Criterion | Weight | Description |
|---|---|---|
| **Real-World Utility** | 30% | Genuine task? Would someone use it to train/evaluate agents? |
| **Task & Grader Quality** | 25% | Well-defined tasks, accurate graders, meaningful difficulty progression |
| **Environment Design** | 20% | Clean state, sensible action/observation spaces, good reward shaping, proper episode boundaries |
| **Code Quality & Spec Compliance** | 15% | Follows OpenEnv spec, clean structure, typed models, documented, tested, Dockerfile works |
| **Creativity & Novelty** | 10% | Novel domain, interesting mechanics, clever reward design |

---

## 12. Disqualification Criteria

- Environment does not deploy or respond.
- Plagiarized or trivially modified existing environments.
- Graders that always return the same score (static/hardcoded).
- No baseline `inference.py` script.
- Environments failing deployment or returning dead pings.

---

## 13. Submission Checklist

### Environment Structure
- [ ] `openenv init` structure followed
- [ ] `__init__.py` exports `YourAction`, `YourObservation`, `YourEnv`
- [ ] `models.py` has valid Pydantic `Action` and `Observation` models

### API Compliance
- [ ] `reset()` → returns `Observation`
- [ ] `step(action)` → returns `StepResult` (observation, reward, done, info)
- [ ] `state()` → returns `State` (episode_id, step_count)
- [ ] WebSocket `/ws` accessible at `http://localhost:8000/ws`

### Tasks & Graders
- [ ] 3+ tasks with clear objectives (easy → medium → hard)
- [ ] Grader for each task returning `0.0–1.0`
- [ ] Reward signal is meaningful (not always 0 or 1)

### Baseline
- [ ] `inference.py` in root directory (exact name)
- [ ] Uses OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] Completes within 20 minutes on 2 vCPU / 8 GB RAM
- [ ] Produces reproducible baseline scores

### Infrastructure
- [ ] `pyproject.toml` with `requires-python = ">=3.10"`
- [ ] `server/Dockerfile` builds successfully
- [ ] Container starts without errors
- [ ] `openenv.yaml` manifest present
- [ ] `openenv validate` passes

### Deployment
- [ ] Pushed to HF Spaces via `openenv push`
- [ ] Space is public, accessible, tagged `openenv`
- [ ] `README.md` documents tasks, graders, usage, baseline scores

### Code Quality
- [ ] No hardcoded secrets or API keys
- [ ] `.gitignore` excludes `outputs/`, `.env`, `__pycache__`
- [ ] `.dockerignore` excludes unnecessary files

---

## 14. Resources & Links

| Resource | URL |
|---|---|
| Official Hackathon Page | https://www.scaler.com/school-of-technology/meta-pytorch-hackathon |
| OpenEnv GitHub (Meta) | https://github.com/meta-pytorch/OpenEnv |
| HuggingFace OpenEnv Course | https://github.com/huggingface/openenv-course |
| OpenEnv Quickstart Docs | https://meta-pytorch.org/OpenEnv/quickstart/ |
| Environment Hub (HF) | https://huggingface.co/collections/openenv/environment-hub |
| TRL + OpenEnv Integration | https://huggingface.co/docs/trl/openenv |
| Discord Community | https://discord.gg/Dedhy5pkWD |
| Round 1 Bootcamp Recording | https://www.youtube.com/live/kkCNMz0Ptd8 |
| GRPO BlackJack Example | https://github.com/meta-pytorch/OpenEnv/tree/main/examples/grpo_blackjack |
| OpenEnv Scaling Guide | https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/03-scaling.md |

### Recommended Learning Path
1. **Module 1** — Why OpenEnv? ([README](https://github.com/huggingface/openenv-course/blob/main/module-1/README.md))
2. **Module 2** — Using Existing Environments
3. **Module 3** — Deploying Environments (`openenv push`)
4. **Module 4** — Building Your Own Environment ← **Critical for Round 1**
5. **Module 5** — Training with OpenEnv + TRL

---

*Last updated: 2 April 2026 · Sourced from Scaler, meta-pytorch/OpenEnv, huggingface/openenv-course, and participant dashboard.*

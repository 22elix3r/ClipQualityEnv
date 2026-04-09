# Skill: Deploy to Hugging Face Space

> For deploying ClipQualityEnv to Hugging Face Spaces as a Docker container

---

## Your Role

You are deploying the ClipQualityEnv OpenEnv environment to a Hugging Face Space. The deployment must:

1. Build and run via Docker
2. Expose the OpenEnv interface via FastAPI
3. Pass all pre-submission checks
4. Complete inference within 20 minutes

---

## Pre-Deployment Checklist

Before deploying, verify locally:

```bash
# 1. Lint
ruff check .

# 2. Type check
mypy clip_quality_env/

# 3. Tests pass
pytest tests/ -v

# 4. Docker builds
docker build -f server/Dockerfile -t clip_quality_env .

# 5. Container responds
docker run -d --name test_env -p 8000:8000 clip_quality_env
sleep 5
curl http://localhost:8000/
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
docker rm -f test_env
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## FastAPI App (`server/app.py`)

```python
from openenv.core import create_app
from clip_quality_env.models import Action, Observation
from server.clip_quality_environment import ClipQualityEnvironment

app = create_app(
    ClipQualityEnvironment,
    Action,
    Observation,
    env_name="clip_quality_env",
    max_concurrent_envs=1,
)

@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": "clip_quality_env",
        "health": "/health",
        "metadata": "/metadata",
    }
```

---

## HF Space Setup

### 1. Create Space

Go to https://huggingface.co/spaces → New Space:
- **Space name**: `clip-quality-env`
- **SDK**: Docker
- **Hardware**: CPU Basic (2 vCPU, 8GB RAM)

### 2. Upload Files

Required files in the Space:

```
clip-quality-env/
├── Dockerfile                 # root-level, generated from server/Dockerfile during push
├── requirements.txt
├── inference.py
├── openenv.yaml
├── README.md
└── clip_quality_env/
    ├── __init__.py
    ├── env.py
    ├── grader.py
    ├── rubric.py
    ├── ground_truth.py
    ├── generator.py
    └── models.py
└── server/
    ├── app.py
    ├── clip_quality_environment.py
    ├── requirements.txt
    └── Dockerfile
```

### 3. Set Secrets

In Space settings → Variables and secrets:

| Name | Value |
|------|-------|
| `HF_TOKEN` | Your HuggingFace token |
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1/` |
| `MODEL_NAME` | `meta-llama/Llama-3-70B-Instruct` |

### 4. Test Endpoints

```bash
SPACE_URL="https://your-username-clip-quality-env.hf.space"

# Readiness / health checks
curl $SPACE_URL/
curl $SPACE_URL/health
curl $SPACE_URL/metadata

# Reset environment
curl -X POST $SPACE_URL/reset

# Run a step
curl -X POST $SPACE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"label": "KEEP", "reasoning": "Good clip with clear face", "confidence": 0.9}'

# Get state
curl $SPACE_URL/state
```

---

## requirements.txt

```
openenv-core>=0.2.3
pydantic>=2.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
requests>=2.25.0
websockets>=12.0
```

---

## Debugging Deployment Issues

### Container won't start

Check logs in HF Space UI. Common issues:

```python
# Missing dependencies
# → Add to requirements.txt

# Port mismatch
# → Ensure Dockerfile exposes 8000 and uvicorn binds to 0.0.0.0:8000

# Import errors
# → Check PYTHONPATH is set to /app
```

### `/reset` returns 500

```python
# Check initialization
# → Ensure data/seed_gt.json exists
# → Verify ClipQualityEnv.__init__ handles missing state files

# Check Pydantic validation
# → Ensure Observation model is correct
```

### Timeout issues

```python
# Reduce episode count in inference.py
MAX_EPISODES = 8  # Lower for safety

# Add timeout buffer
BUFFER_SECONDS = 120  # 2 minutes buffer
```

---

## Pre-Submission Validation Script

```bash
#!/bin/bash
# pre_submit_check.sh
set -e

echo "=== Pre-Submission Check ==="

echo "1. Linting..."
ruff check . || exit 1

echo "2. Type checking..."
mypy clip_quality_env/ || exit 1

echo "3. Tests..."
pytest tests/ -v || exit 1

echo "4. Docker build..."
docker build -f server/Dockerfile -t clip_quality_env . || exit 1

echo "5. Container test..."
docker run -d --name test_env -p 8000:8000 clip_quality_env
sleep 5
curl -f http://localhost:8000/ || { docker rm -f test_env; exit 1; }
curl -f http://localhost:8000/health || { docker rm -f test_env; exit 1; }
curl -f -X POST http://localhost:8000/reset || { docker rm -f test_env; exit 1; }
docker rm -f test_env

echo ""
echo "=== ALL CHECKS PASSED ==="
```

---

## Submission Checklist

- [ ] HF Space deploys and responds to `/` with HTTP 200
- [ ] `GET /health` returns HTTP 200
- [ ] `/reset` returns valid Observation
- [ ] `/step` processes Action and returns reward
- [ ] `/state` returns environment state
- [ ] `inference.py` completes within 20 minutes
- [ ] `openenv.yaml` is valid
- [ ] README.md has all 5 required sections
- [ ] License file present

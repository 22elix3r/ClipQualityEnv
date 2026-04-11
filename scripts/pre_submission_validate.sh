#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_URL="${SPACE_URL:-}"  # Example: https://your-space.hf.space
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-1200}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log() {
  printf "%b\n" "$1"
}

pass() {
  log "${GREEN}[PASS]${NC} $1"
}

fail() {
  log "${RED}[FAIL]${NC} $1"
}

hint() {
  log "${YELLOW}  hint:${NC} $1"
}

run_with_timeout() {
  local timeout_s="$1"
  shift
  timeout "$timeout_s" "$@"
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD} ClipQualityEnv Pre-Submission Check ${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

log "${BOLD}Step 1/4: Checking HF Space /reset${NC} ..."
if [ -z "$SPACE_URL" ]; then
  fail "SPACE_URL is not set"
  hint "Export your Space URL first, for example: export SPACE_URL=https://your-space.hf.space"
  exit 1
fi

PING_URL="${SPACE_URL%/}/reset"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$PING_URL" -H "Content-Type: application/json" -d '{"task_id":"task_easy"}')
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space /reset returned HTTP 200"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening ${SPACE_URL%/} in your browser first."
  exit 1
fi

log "${BOLD}Step 2/4: Running docker build${NC} ..."
if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  exit 1
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  exit 1
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"
if BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1); then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  exit 1
fi

log "${BOLD}Step 3/4: Running openenv validate${NC} ..."
if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  exit 1
fi

if VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1); then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  exit 1
fi

log "${BOLD}Step 4/4: Validating structured inference stdout${NC} ..."
INFER_CMD=(python inference.py --tasks easy --episodes 1 --seed 42 --deterministic-baseline --max-steps 5)
if INFER_OUTPUT=$(cd "$REPO_DIR" && PYTHONPATH=. "${INFER_CMD[@]}" 2>&1); then
  :
else
  fail "Deterministic inference command failed"
  printf "%s\n" "$INFER_OUTPUT" | tail -50
  exit 1
fi

if ! printf "%s\n" "$INFER_OUTPUT" | grep -q '^\[START\] '; then
  fail "Missing [START] block in inference stdout"
  exit 1
fi
if ! printf "%s\n" "$INFER_OUTPUT" | grep -q '^\[STEP\] '; then
  fail "Missing [STEP] block in inference stdout"
  exit 1
fi
if ! printf "%s\n" "$INFER_OUTPUT" | grep -q '^\[END\] '; then
  fail "Missing [END] block in inference stdout"
  exit 1
fi

pass "Structured stdout blocks detected: [START], [STEP], [END]"

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 4/4 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

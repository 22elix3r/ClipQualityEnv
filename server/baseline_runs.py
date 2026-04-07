"""Thread-safe in-memory tracker for async baseline runs."""
from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Callable, Literal, TypedDict

RunStatus = Literal["running", "completed", "failed"]
DEFAULT_RUN_TTL_SECONDS = 10 * 60


class BaselineRun(TypedDict):
    run_id: str
    status: RunStatus
    created_at: float
    updated_at: float
    expires_at: float
    started_at: float
    completed_at: float | None
    failed_at: float | None
    partial: Any
    result: Any
    error: Any


class BaselineRunTracker:
    """Manage async baseline runs in-memory with TTL-based cleanup."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_RUN_TTL_SECONDS,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._ttl_seconds = float(ttl_seconds)
        self._time_fn = time_fn or time.time
        self._lock = threading.RLock()
        self._runs: dict[str, BaselineRun] = {}

    def _now(self) -> float:
        return float(self._time_fn())

    def _is_expired(self, run: BaselineRun, now: float) -> bool:
        return float(run["expires_at"]) <= now

    def _touch(self, run: BaselineRun, now: float) -> None:
        run["updated_at"] = now
        run["expires_at"] = now + self._ttl_seconds

    def _get_for_update(self, run_id: str, now: float) -> BaselineRun:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if self._is_expired(run, now):
            del self._runs[run_id]
            raise KeyError(f"Expired run_id: {run_id}")
        return run

    def _merge_partial(self, run: BaselineRun, payload: Any) -> None:
        current = run.get("partial")
        if isinstance(current, dict) and isinstance(payload, dict):
            merged = deepcopy(current)
            merged.update(deepcopy(payload))
            run["partial"] = merged
            return
        run["partial"] = deepcopy(payload)

    def create_run(self) -> str:
        with self._lock:
            now = self._now()
            run_id = uuid.uuid4().hex
            while run_id in self._runs:
                run_id = uuid.uuid4().hex
            self._runs[run_id] = BaselineRun(
                run_id=run_id,
                status="running",
                created_at=now,
                updated_at=now,
                expires_at=now + self._ttl_seconds,
                started_at=now,
                completed_at=None,
                failed_at=None,
                partial={},
                result=None,
                error=None,
            )
            return run_id

    def mark_running(self, run_id: str, payload: Any | None = None) -> BaselineRun:
        with self._lock:
            now = self._now()
            run = self._get_for_update(run_id, now)
            run["status"] = "running"
            run["result"] = None
            run["error"] = None
            run["completed_at"] = None
            run["failed_at"] = None
            if payload is not None:
                self._merge_partial(run, payload)
            self._touch(run, now)
            return deepcopy(run)

    def update_partial(self, run_id: str, payload: Any) -> BaselineRun:
        return self.mark_running(run_id, payload=payload)

    def mark_complete(self, run_id: str, result: Any) -> BaselineRun:
        with self._lock:
            now = self._now()
            run = self._get_for_update(run_id, now)
            run["status"] = "completed"
            run["result"] = deepcopy(result)
            run["error"] = None
            run["completed_at"] = now
            run["failed_at"] = None
            self._touch(run, now)
            return deepcopy(run)

    def mark_failed(self, run_id: str, error: Any) -> BaselineRun:
        with self._lock:
            now = self._now()
            run = self._get_for_update(run_id, now)
            run["status"] = "failed"
            run["result"] = None
            run["error"] = deepcopy(error)
            run["completed_at"] = None
            run["failed_at"] = now
            self._touch(run, now)
            return deepcopy(run)

    def get_run(self, run_id: str) -> BaselineRun | None:
        with self._lock:
            now = self._now()
            run = self._runs.get(run_id)
            if run is None:
                return None
            if self._is_expired(run, now):
                del self._runs[run_id]
                return None
            return deepcopy(run)

    def cleanup_expired(self) -> int:
        with self._lock:
            now = self._now()
            expired_ids = [run_id for run_id, run in self._runs.items() if self._is_expired(run, now)]
            for run_id in expired_ids:
                del self._runs[run_id]
            return len(expired_ids)


baseline_run_tracker = BaselineRunTracker()

__all__ = [
    "BaselineRun",
    "BaselineRunTracker",
    "DEFAULT_RUN_TTL_SECONDS",
    "baseline_run_tracker",
]

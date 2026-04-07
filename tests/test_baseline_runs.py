from __future__ import annotations

from server.baseline_runs import BaselineRunTracker, DEFAULT_RUN_TTL_SECONDS


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.value = float(start)

    def time(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def test_create_run_defaults_to_running_state():
    clock = _FakeClock(start=100.0)
    tracker = BaselineRunTracker(time_fn=clock.time)

    run_id = tracker.create_run()
    run = tracker.get_run(run_id)

    assert run is not None
    assert run["run_id"] == run_id
    assert run["status"] == "running"
    assert run["partial"] == {}
    assert run["result"] is None
    assert run["error"] is None
    assert run["created_at"] == 100.0
    assert run["updated_at"] == 100.0
    assert run["started_at"] == 100.0
    assert run["completed_at"] is None
    assert run["failed_at"] is None
    assert run["expires_at"] == 100.0 + DEFAULT_RUN_TTL_SECONDS


def test_update_partial_merges_and_refreshes_ttl():
    clock = _FakeClock(start=10.0)
    tracker = BaselineRunTracker(ttl_seconds=10.0, time_fn=clock.time)

    run_id = tracker.create_run()
    clock.advance(3.0)
    tracker.update_partial(run_id, {"processed": 1})

    first = tracker.get_run(run_id)
    assert first is not None
    assert first["status"] == "running"
    assert first["partial"] == {"processed": 1}
    assert first["expires_at"] == 23.0

    clock.advance(2.0)
    tracker.update_partial(run_id, {"total": 5})
    second = tracker.get_run(run_id)

    assert second is not None
    assert second["partial"] == {"processed": 1, "total": 5}
    assert second["expires_at"] == 25.0


def test_status_transitions_failed_to_running_to_complete():
    clock = _FakeClock(start=200.0)
    tracker = BaselineRunTracker(time_fn=clock.time)

    run_id = tracker.create_run()
    clock.advance(1.0)
    tracker.mark_failed(run_id, {"message": "transient error"})

    failed = tracker.get_run(run_id)
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == {"message": "transient error"}
    assert failed["failed_at"] == 201.0

    clock.advance(1.0)
    tracker.mark_running(run_id, {"retry": 1})
    running = tracker.get_run(run_id)
    assert running is not None
    assert running["status"] == "running"
    assert running["error"] is None
    assert running["failed_at"] is None
    assert running["partial"] == {"retry": 1}

    clock.advance(1.0)
    tracker.mark_complete(run_id, {"average_score": 0.87})
    completed = tracker.get_run(run_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == {"average_score": 0.87}
    assert completed["error"] is None
    assert completed["completed_at"] == 203.0


def test_cleanup_expired_respects_ttl_and_touch_updates():
    clock = _FakeClock(start=0.0)
    tracker = BaselineRunTracker(ttl_seconds=10.0, time_fn=clock.time)

    stale_run = tracker.create_run()
    refreshed_run = tracker.create_run()

    clock.advance(6.0)
    tracker.update_partial(refreshed_run, {"progress": 0.5})

    clock.advance(5.0)
    removed = tracker.cleanup_expired()
    assert removed == 1
    assert tracker.get_run(stale_run) is None
    assert tracker.get_run(refreshed_run) is not None

    clock.advance(6.0)
    assert tracker.cleanup_expired() == 1
    assert tracker.get_run(refreshed_run) is None

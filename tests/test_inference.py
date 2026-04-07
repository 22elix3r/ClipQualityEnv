from __future__ import annotations

from typing import Any

import inference


def test_inference_extract_json_handles_fenced_blocks():
    raw = "```json\n{\"x\": 1}\n```"
    parsed = inference._extract_json(raw)
    assert parsed == {"x": 1}


def test_run_baseline_works_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = inference.run_baseline(task="task_easy")
    assert result["detail"][0]["task_id"] == "task_easy"
    assert result["detail"][0]["mode"] == "fallback"
    assert result["detail"][0]["steps"] == 5
    assert 0.0 <= result["detail"][0]["total_reward"] <= 5.0
    assert 0.0 <= result["detail"][0]["final_reward"] <= 1.0
    assert 0.0 <= result["detail"][0]["reward"] <= 1.0
    assert 0.0 <= result["baseline_scores"]["overall_avg"] <= 1.0


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChoice:
    def __init__(self, content: str):
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(content)]


class _DummyCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **_: Any):
        return _DummyResponse(self._content)


class _DummyChat:
    def __init__(self, content: str):
        self.completions = _DummyCompletions(content)


class _DummyClient:
    def __init__(self, content: str):
        self.chat = _DummyChat(content)


def test_run_episode_preserves_logging_and_structure(capsys):
    client = _DummyClient('{"label":"KEEP","reasoning":"face_confidence and motion_score indicate keep","confidence":0.9}')
    result = inference.run_episode("task_easy", client, "dummy-model")
    output = capsys.readouterr().out
    assert "[START]" in output and "[STEP]" in output and "[END]" in output
    assert "total_reward=" in output and "final_reward=" in output
    assert result["task_id"] == "task_easy"
    assert result["mode"] == "llm"
    assert result["steps"] == 5
    assert 0.0 <= result["final_reward"] <= 1.0
    assert 0.0 <= result["total_reward"] <= 5.0
    assert 0.0 <= result["reward"] <= 1.0
    assert abs(result["reward"] * result["steps"] - result["total_reward"]) < 1e-9

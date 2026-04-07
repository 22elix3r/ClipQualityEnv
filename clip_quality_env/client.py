from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import Action, EnvironmentState, Observation


class ClipQualityClient(EnvClient[Action, Observation, EnvironmentState]):
    """Typed async client for ClipQualityEnv with sync wrapper support."""

    def _step_payload(self, action: Action) -> dict[str, Any]:
        payload = action.model_dump()
        if isinstance(payload, dict) and "root" in payload and isinstance(payload["root"], dict):
            return payload["root"]
        return payload if isinstance(payload, dict) else {"action": payload}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Observation]:
        obs_data = payload.get("observation", {})
        if "done" not in obs_data:
            obs_data["done"] = bool(payload.get("done", False))
        if "reward" not in obs_data:
            obs_data["reward"] = payload.get("reward")
        observation = Observation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> EnvironmentState:
        return EnvironmentState.model_validate(payload)

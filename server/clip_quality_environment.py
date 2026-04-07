from __future__ import annotations

from clip_quality_env.env import ClipQualityEnvironment as BaseClipQualityEnvironment


class ClipQualityEnvironment(BaseClipQualityEnvironment):
    """Server runtime wrapper preserving expected import path."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

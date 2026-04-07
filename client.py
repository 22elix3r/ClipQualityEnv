"""Root client module for OpenEnv CLI compatibility."""

from clip_quality_env.client import ClipQualityClient

# Backward-compatibility alias (deprecated).
PolicyEvolverEnv = ClipQualityClient

__all__ = ["ClipQualityClient"]

"""Clip-quality environment package."""

from .client import ClipQualityClient
from .env import ClipQualityEnv, ClipQualityEnvironment
from .models import (
    Action,
    ClipLabel,
    ClipMetadata,
    CorpusIncident,
    EnvironmentState,
    EpisodeHistoryItem,
    HistoryItem,
    Observation,
    Reward,
    State,
    TaskInfo,
)

# Backward-compatibility aliases (deprecated).
PolicyEvolverEnv = ClipQualityClient
PolicyEvolverEnvironment = ClipQualityEnvironment

__all__ = [
    "Action",
    "ClipLabel",
    "ClipMetadata",
    "ClipQualityClient",
    "ClipQualityEnv",
    "ClipQualityEnvironment",
    "CorpusIncident",
    "EnvironmentState",
    "EpisodeHistoryItem",
    "HistoryItem",
    "Observation",
    "Reward",
    "State",
    "TaskInfo",
]

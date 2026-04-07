from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ClipLabel(str, Enum):
    KEEP = "KEEP"
    BORDERLINE = "BORDERLINE"
    REJECT = "REJECT"


class Action(BaseModel):
    """Agent action for clip-quality classification."""

    label: Literal["KEEP", "BORDERLINE", "REJECT"] = Field(description="Predicted clip label")
    reasoning: str = Field(min_length=1, description="Explanation tied to clip metadata")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Model confidence in the label")
    clip_id: Optional[str] = Field(default=None, description="Optional target clip ID")
    model_config = {"extra": "allow"}


class TaskInfo(BaseModel):
    """Returned by /tasks endpoint."""

    task_id: str
    difficulty: str
    description: str
    action_schema: dict


class CorpusIncident(BaseModel):
    """Compatibility record for corpus table views."""

    id: str
    content: str
    review_status: str = "pending"
    model_config = {"extra": "allow"}


class ClipMetadata(BaseModel):
    """Clip metadata payload consumed by the agent.

    NOTE: expected_label is intentionally excluded from this model.
    It lives only on the raw clip dict inside EpisodeClip.clip and is
    accessed by the grader directly — never serialised to the agent.
    """

    clip_id: str
    duration_s: Optional[float] = Field(default=None, ge=0.0)
    fps: Optional[int] = Field(default=None, ge=1)
    resolution: Optional[str] = None
    face_area_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    face_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    head_pose_yaw_deg: Optional[float] = None
    head_pose_pitch_deg: Optional[float] = None
    motion_score: Optional[float] = Field(default=None, ge=0.0)
    bg_complexity: Optional[str] = None
    bg_complexity_score: Optional[float] = Field(default=None, ge=0.0)
    mouth_open_ratio: Optional[float] = Field(default=None, ge=0.0)
    blink_rate_hz: Optional[float] = Field(default=None, ge=0.0)
    audio_snr_db: Optional[float] = None
    transcript_word_count: Optional[int] = Field(default=None, ge=0)
    transcript_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lighting_uniformity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    occlusion_present: bool = False
    environment_tag: Optional[str] = None
    framing: Optional[str] = None
    # Option-A enriched features
    sharpness_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temporal_flicker: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bg_entropy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    eye_contact_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speech_rate_wpm: Optional[float] = Field(default=None, ge=0.0)
    model_config = {"extra": "ignore"}


class HistoryItem(BaseModel):
    """Agent-facing step history — contains NO ground-truth labels."""

    step: int = 0
    clip_id: str = ""
    label: str = ""       # the agent's own submitted label
    reward: float = 0.0   # reward signal — the only feedback the agent gets
    model_config = {"extra": "ignore"}


class EpisodeHistoryItem(BaseModel):
    step: int = 0
    difficulty: str = ""
    clip_id: str = ""
    label: str = ""
    expected_label: str = ""
    reward: float = 0.0
    model_config = {"extra": "allow"}


class Observation(BaseModel):
    """What the agent sees after reset() or step()."""

    task_id: str
    episode_id: str
    step_count: int
    max_steps: int = 5
    step: int = Field(default=1, ge=1, description="1-indexed episode step")
    rubric_version: int = Field(default=1, ge=1)
    rubric_summary: str = ""
    clip_metadata: ClipMetadata
    history: List[HistoryItem] = Field(default_factory=list)
    corpus_size: int = 0
    corpus_shown: int = 0
    data_corpus: List[Dict[str, Any]] = Field(default_factory=list, description="Task clip samples")
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class State(BaseModel):
    """Episode metadata — returned by state() endpoint."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    episode_count: int = 0
    step_count: int = 0
    max_steps: int = 5
    current_score: float = 0.0
    total_reward: float = 0.0
    best_score: float = 0.0
    current_clip_id: str = ""
    gt_size: int = 0
    rubric_version: int = 1
    actions_taken: List[str] = Field(default_factory=list)
    episode_history: List[EpisodeHistoryItem] = Field(default_factory=list)
    rubric_thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


EnvironmentState = State


class Reward(BaseModel):
    total: float = Field(default=0.0, ge=0.0, le=1.0)
    format_score: float = Field(default=0.0, ge=0.0, le=1.0)
    label_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_score: float = Field(default=0.0, ge=0.0, le=1.0)

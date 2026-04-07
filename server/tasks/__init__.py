from .task_easy import EASY_TASK
from .task_medium import MEDIUM_TASK
from .task_hard import HARD_TASK

TASK_REGISTRY = {
    "task_easy": EASY_TASK,
    "task_medium": MEDIUM_TASK,
    "task_hard": HARD_TASK,
}

TASK_IDS = tuple(TASK_REGISTRY.keys())

__all__ = ["EASY_TASK", "MEDIUM_TASK", "HARD_TASK", "TASK_REGISTRY", "TASK_IDS"]

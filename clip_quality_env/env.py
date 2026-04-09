from __future__ import annotations

import copy
import os
import random
import uuid
from dataclasses import dataclass
from typing import Any

from openenv.core import Environment

from .grader import grade
from .ground_truth import GTStore
from .icl_memory import ICLMemory
from .models import Action, ClipMetadata, EpisodeHistoryItem, HistoryItem, Observation, State
from .real_clips import load_real_clip_manifest
from .rubric import RubricState
from server.tasks import TASK_REGISTRY

EPISODE_STEPS = 25
DEFAULT_REAL_CLIPS_MANIFEST = "data/real_clips_manifest.jsonl"

# Mixed-difficulty episode layout: how many clips from each difficulty
MIXED_DIFFICULTY_PLAN: list[str] = (
    ["easy"] * 10 + ["medium"] * 8 + ["hard"] * 7
)

# Curriculum auto-promotion/demotion thresholds
# (min_avg_reward_over_N_episodes, N_episodes_required)
CURRICULUM_PROMOTE_THRESHOLD = 0.75   # average reward to promote
CURRICULUM_DEMOTE_THRESHOLD = 0.35    # average reward to demote
CURRICULUM_WINDOW = 2                 # number of episodes to consider
DIFFICULTY_ORDER = ["easy", "medium", "hard", "mixed"]

# Mapping from difficulty label to task_id
DIFFICULTY_TO_TASK: dict[str, str] = {
    "easy": "task_easy",
    "medium": "task_medium",
    "hard": "task_hard",
}


@dataclass
class EpisodeClip:
    task_id: str
    difficulty: str
    clip: dict[str, Any]


class ClipQualityEnvironment(Environment[Action, Observation, State]):
    """OpenEnv environment for clip-quality classification tasks."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._rubric = RubricState()
        self._gt_store = GTStore()
        self._state = State(
            max_steps=EPISODE_STEPS,
            total_reward=0.0,
            rubric_version=self._rubric.version,
            gt_size=self._gt_store.size(),
            rubric_thresholds=self._rubric.get_thresholds_summary(),
        )
        self._episode_plan: list[EpisodeClip] = []
        self._episode_corpus: dict[str, list[dict[str, Any]]] = {}
        self._persistent_best_score = 0.0
        self._last_reward_breakdown = {
            "format_score": 0.0,
            "label_score": 0.0,
            "reasoning_score": 0.0,
            "total_reward": 0.0,
        }
        self._corpus_source = "task_registry"
        self._manifest_warning = ""
        self._manifest_path = os.environ.get("REAL_CLIPS_MANIFEST", DEFAULT_REAL_CLIPS_MANIFEST)
        self._real_clip_pools = self._load_real_clip_pools()
        # Cross-episode curriculum state (persists across resets)
        self._curriculum_level = "easy"
        self._curriculum_history: list[dict[str, Any]] = []

    def _load_real_clip_pools(self) -> dict[str, list[dict[str, Any]]]:
        if not os.path.exists(self._manifest_path):
            self._manifest_warning = (
                f"Real clip manifest not found at {self._manifest_path}; using static task corpora."
            )
            return {}
        try:
            pools = load_real_clip_manifest(self._manifest_path, self._rubric)
        except Exception as exc:
            self._manifest_warning = (
                f"Failed loading real clip manifest from {self._manifest_path}: {exc}; using static task corpora."
            )
            return {}
        self._manifest_warning = ""
        return pools

    def _choose_tasks(self, seed: int | None = None, task_id: str | None = None) -> list[str]:
        if task_id is not None:
            # Allow "task_mixed" even though it's not in TASK_REGISTRY
            if task_id != "task_mixed" and task_id not in TASK_REGISTRY:
                raise KeyError(f"Unknown task_id: {task_id}")
            return [task_id]
        rng = random.Random(seed)
        return [rng.choice(list(TASK_REGISTRY.keys()))]

    def _load_task_corpus(self, task_id: str) -> list[dict[str, Any]]:
        task = TASK_REGISTRY[task_id]
        difficulty = str(task.get("difficulty", "")).lower()
        corpus: list[dict[str, Any]]

        if difficulty and self._real_clip_pools.get(difficulty):
            corpus = copy.deepcopy(self._real_clip_pools[difficulty])
            self._corpus_source = f"manifest:{difficulty}"
        else:
            corpus = copy.deepcopy(task.get("data_corpus", []))
            self._corpus_source = f"task_registry:{task_id}"

        if not corpus:
            raise ValueError(f"No clip corpus configured for task_id={task_id}")

        for clip in corpus:
            clip.setdefault("clip_id", clip.get("id", str(uuid.uuid4())))
            if not clip.get("expected_label"):
                clip["expected_label"] = self._rubric.derive_label(clip)
            clip["review_status"] = str(clip.get("review_status", "pending")).lower()

        corpus.sort(key=lambda item: str(item.get("clip_id", item.get("id", ""))))
        return corpus

    def _sample_episode_clips(
        self,
        corpus: list[dict[str, Any]],
        count: int = EPISODE_STEPS,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sample `count` clips from corpus using a deterministic seed.

        The corpus is sorted by clip_id before sampling to guarantee that
        the same seed always produces the same episode regardless of
        insertion order.
        """
        if not corpus:
            raise ValueError("Cannot sample episode clips from an empty corpus")
        # Sort for deterministic ordering before RNG sampling
        stable_corpus = sorted(corpus, key=lambda c: str(c.get("clip_id", "")))
        rng = random.Random(seed)
        if len(stable_corpus) >= count:
            return [copy.deepcopy(clip) for clip in rng.sample(stable_corpus, k=count)]

        pool = [copy.deepcopy(clip) for clip in stable_corpus]
        rng.shuffle(pool)
        sampled: list[dict[str, Any]] = []
        while len(sampled) < count:
            sampled.append(copy.deepcopy(pool[len(sampled) % len(pool)]))
        return sampled

    def _build_mixed_episode(
        self,
        seed: int | None = None,
    ) -> tuple[list[EpisodeClip], dict[str, list[dict[str, Any]]]]:
        """Build a mixed-difficulty episode: 2 easy + 2 medium + 1 hard.

        Creates progressive difficulty within a single episode.
        """
        rng = random.Random(seed)
        plan: list[EpisodeClip] = []
        corpus_map: dict[str, list[dict[str, Any]]] = {}

        # Count how many clips needed per difficulty
        from collections import Counter
        diff_counts = Counter(MIXED_DIFFICULTY_PLAN)

        for diff_level, count in diff_counts.items():
            task_id = DIFFICULTY_TO_TASK.get(diff_level, f"task_{diff_level}")
            if task_id not in TASK_REGISTRY:
                continue
            corpus = self._load_task_corpus(task_id)
            corpus_map[task_id] = corpus
            # Use a sub-seed derived from the main seed for per-difficulty sampling
            sub_seed = rng.randint(0, 2**31) if seed is not None else None
            sampled = self._sample_episode_clips(corpus, count=count, seed=sub_seed)
            for clip in sampled:
                plan.append(EpisodeClip(task_id=task_id, difficulty=diff_level, clip=clip))

        # Sort plan by difficulty order for progressive escalation
        diff_order = {d: i for i, d in enumerate(DIFFICULTY_ORDER)}
        plan.sort(key=lambda ep: diff_order.get(ep.difficulty, 99))
        return plan, corpus_map

    def _sample_episode_plan(
        self,
        task_ids: list[str],
        seed: int | None = None,
    ) -> tuple[list[EpisodeClip], dict[str, list[dict[str, Any]]]]:
        """Build episode plan for the given task.

        If task_id is 'task_mixed', builds a mixed-difficulty episode.
        Otherwise samples from a single task corpus.
        """
        if len(task_ids) != 1:
            raise ValueError("Episode planning expects exactly one task id")
        task_id = task_ids[0]

        # Mixed-difficulty episode
        if task_id == "task_mixed":
            return self._build_mixed_episode(seed=seed)

        task = TASK_REGISTRY[task_id]
        difficulty = str(task.get("difficulty", ""))
        corpus = self._load_task_corpus(task_id)
        sampled_clips = self._sample_episode_clips(corpus, seed=seed)
        plan = [
            EpisodeClip(task_id=task_id, difficulty=difficulty, clip=clip)
            for clip in sampled_clips
        ]
        corpus_map: dict[str, list[dict[str, Any]]] = {task_id: corpus}
        return plan, corpus_map

    def _threshold_range_text(self, feature: str) -> str:
        threshold = self._rubric.thresholds.get(feature)
        if not threshold:
            return "N/A"
        mode = str(threshold["mode"])
        keep_min = float(threshold["keep_min"])
        keep_max = float(threshold["keep_max"])
        reject_min = float(threshold["reject_min"])
        reject_max = float(threshold["reject_max"])
        if mode == "higher":
            return f"KEEP >= {keep_min:.3g}; BORDERLINE [{reject_max:.3g}, {keep_min:.3g}); REJECT < {reject_max:.3g}"
        if mode == "lower":
            return f"KEEP <= {keep_max:.3g}; BORDERLINE ({keep_max:.3g}, {reject_min:.3g}]; REJECT > {reject_min:.3g}"
        return (
            f"KEEP [{keep_min:.3g}, {keep_max:.3g}]; BORDERLINE [{reject_min:.3g}, {keep_min:.3g})"
            f" U ({keep_max:.3g}, {reject_max:.3g}]; REJECT outside [{reject_min:.3g}, {reject_max:.3g}]"
        )

    def _closest_boundary_features(self, clip: dict[str, Any], top_n: int = 2) -> list[str]:
        distances: list[tuple[float, str]] = []
        for feature, threshold in self._rubric.thresholds.items():
            value = clip.get(feature)
            if not isinstance(value, (int, float)):
                continue
            mode = str(threshold["mode"])
            keep_min = float(threshold["keep_min"])
            keep_max = float(threshold["keep_max"])
            reject_min = float(threshold["reject_min"])
            reject_max = float(threshold["reject_max"])
            if mode == "higher":
                boundaries = [keep_min, reject_max]
            elif mode == "lower":
                boundaries = [keep_max, reject_min]
            else:
                boundaries = [keep_min, keep_max, reject_min, reject_max]
            min_distance = min(abs(float(value) - boundary) for boundary in boundaries)
            distances.append((min_distance, feature))
        distances.sort(key=lambda item: item[0])
        return [feature for _, feature in distances[:top_n]]

    def dominant_feature_rows(self, clip: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if clip is None:
            current_index = min(self._state.step_count, max(len(self._episode_plan) - 1, 0))
            if not self._episode_plan:
                return []
            clip = self._episode_plan[current_index].clip
        dominant_features = self._rubric.get_dominant_features(clip)
        rows: list[dict[str, Any]] = []
        for feature in dominant_features:
            value = clip.get(feature)
            if not isinstance(value, (int, float)):
                continue
            status = self._rubric.get_feature_status(feature, float(value))
            rows.append(
                {
                    "Feature Name": feature,
                    "Current Value": float(value),
                    "Rubric Status": status,
                    "Threshold Range": self._threshold_range_text(feature),
                }
            )
        return rows

    def build_quality_hint(
        self,
        clip: dict[str, Any] | None = None,
        icl_memory: "ICLMemory | None" = None,
    ) -> str:
        """
        Build a rubric-anchored quality hint for the current (or given) clip.

        When icl_memory is provided and the clip has prior attempt history,
        a feedback suffix is appended to tell the agent what went wrong and
        what to correct — implementing the cross-episode ICL signal.
        """
        if clip is None:
            current_index = min(self._state.step_count, max(len(self._episode_plan) - 1, 0))
            if not self._episode_plan:
                return ""
            clip = self._episode_plan[current_index].clip
        focus_features = self._closest_boundary_features(clip, top_n=2)
        if not focus_features:
            base_hint = "Use dominant clip metadata cues and compare each value against rubric thresholds."
        else:
            segments: list[str] = []
            for idx, feature in enumerate(focus_features):
                value = clip.get(feature)
                if not isinstance(value, (int, float)):
                    continue
                status = self._rubric.get_feature_status(feature, float(value))
                threshold = self._rubric.thresholds.get(feature, {})
                mode = str(threshold.get("mode", ""))
                keep_min = float(threshold.get("keep_min", 0.0))
                keep_max = float(threshold.get("keep_max", 0.0))
                reject_min = float(threshold.get("reject_min", 0.0))
                reject_max = float(threshold.get("reject_max", 0.0))

                if mode == "higher":
                    if status == "KEEP":
                        direction = f"above the KEEP threshold ({keep_min:.3g})"
                    elif status == "REJECT":
                        direction = f"below the REJECT threshold ({reject_max:.3g})"
                    else:
                        direction = f"within the BORDERLINE range [{reject_max:.3g}, {keep_min:.3g})"
                elif mode == "lower":
                    if status == "KEEP":
                        direction = f"below the KEEP ceiling ({keep_max:.3g})"
                    elif status == "REJECT":
                        direction = f"above the REJECT threshold ({reject_min:.3g})"
                    else:
                        direction = f"within the BORDERLINE range ({keep_max:.3g}, {reject_min:.3g}]"
                else:
                    if status == "KEEP":
                        direction = f"within the KEEP band [{keep_min:.3g}, {keep_max:.3g}]"
                    elif status == "REJECT":
                        direction = f"outside the acceptable range [{reject_min:.3g}, {reject_max:.3g}]"
                    else:
                        direction = (
                            f"within a BORDERLINE edge zone around [{reject_min:.3g}, {keep_min:.3g})"
                            f" or ({keep_max:.3g}, {reject_max:.3g}]"
                        )

                prefix = "" if idx == 0 else " "
                segments.append(f"{prefix}{feature} is {float(value):.3g}, which is {direction}.")

            base_hint = "".join(segments).strip()

        # Append ICL feedback when session memory has prior history for this clip
        if icl_memory is not None:
            clip_id = str(clip.get("clip_id", ""))
            feedback = icl_memory.get_hint_feedback(clip_id)
            if feedback:
                base_hint = f"{base_hint} {feedback}".strip()

        return base_hint

    # ── Helpers to strip ground-truth from agent-facing payloads ─────────────

    @staticmethod
    def _sanitize_clip_for_agent(clip: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the clip dict with all answer-revealing fields removed."""
        to_strip = {"expected_label", "quality_cues"}
        return {k: v for k, v in clip.items() if k not in to_strip}

    @staticmethod
    def _sanitize_corpus_for_agent(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip expected_label and quality_cues from every clip in the corpus."""
        to_strip = {"expected_label", "quality_cues"}
        return [{k: v for k, v in item.items() if k not in to_strip} for item in corpus]

    @staticmethod
    def _sanitize_session_history(history: list[Any]) -> list[dict[str, Any]]:
        """Strip expected_label from episode history entries sent to the agent."""
        result = []
        for item in history:
            d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            d.pop("expected_label", None)
            result.append(d)
        return result

    def _state_to_observation(self, reward: float, done: bool) -> Observation:
        current_index = min(self._state.step_count, max(len(self._episode_plan) - 1, 0))
        current = self._episode_plan[current_index]
        corpus = self._episode_corpus.get(current.task_id, [])
        # Sanitize corpus — agent must NOT see expected_label or quality_cues
        agent_corpus = self._sanitize_corpus_for_agent(list(corpus))
        # Build history items with NO expected_label
        history_items = [
            HistoryItem(
                step=h.step,
                clip_id=h.clip_id,
                label=h.label,
                reward=h.reward,
            )
            for h in self._state.episode_history
        ]
        steps_remaining = max(0, self._state.max_steps - self._state.step_count)
        # Session history for UI — also strip expected_label
        session_history = self._sanitize_session_history(self._state.episode_history)
        # Sanitize clip metadata — ClipMetadata has extra="ignore" so extra keys
        # (including expected_label) are silently dropped on model_validate.
        agent_clip = self._sanitize_clip_for_agent(current.clip)
        return Observation(
            task_id=current.task_id,
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            step=max(1, self._state.step_count + (0 if done else 1)),
            rubric_version=self._rubric.version,
            rubric_summary=self._rubric.to_prompt_text(),
            clip_metadata=ClipMetadata.model_validate(agent_clip),
            history=history_items,
            corpus_size=len(corpus),
            corpus_shown=len(agent_corpus),
            data_corpus=agent_corpus,
            reward=float(reward),
            done=done,
            info={
                "difficulty": current.difficulty,
                "task_description": TASK_REGISTRY.get(current.task_id, {}).get("description", ""),
                "best_score": self._persistent_best_score,
                "last_reward": float(reward),
                "action_history": self._state.actions_taken,
                "steps_remaining": steps_remaining,
                "total_reward": float(self._state.total_reward),
                "reward_breakdown": dict(self._last_reward_breakdown),
                "format_score": float(self._last_reward_breakdown["format_score"]),
                "label_score": float(self._last_reward_breakdown["label_score"]),
                "reasoning_score": float(self._last_reward_breakdown["reasoning_score"]),
                "reward_total": float(self._last_reward_breakdown["total_reward"]),
                "session_history": session_history,
                "corpus_source": self._corpus_source,
                # ── Step-level progression info ──────────────────────────
                "difficulty_trend": [ep.difficulty for ep in self._episode_plan],
                "cumulative_accuracy": self._compute_cumulative_accuracy(),
                "curriculum_level": self._curriculum_level,
                "curriculum_history": self._curriculum_history[-5:],  # last 5 episodes
                # rubric_thresholds intentionally OMITTED — exposing the grader's
                # internal threshold values lets the agent reconstruct the exact
                # decision function and achieve perfect scores trivially.
            },
        )

    def _compute_cumulative_accuracy(self) -> float:
        """Fraction of correct predictions so far in this episode."""
        if not self._state.episode_history:
            return 0.0
        correct = sum(1 for h in self._state.episode_history if h.label == h.expected_label)
        return round(correct / len(self._state.episode_history), 4)

    def _update_curriculum(self, episode_avg_reward: float) -> None:
        """Auto-promote or demote difficulty based on rolling episode performance.

        Promotion:  avg reward > CURRICULUM_PROMOTE_THRESHOLD for CURRICULUM_WINDOW episodes → next difficulty
        Demotion:   avg reward < CURRICULUM_DEMOTE_THRESHOLD for CURRICULUM_WINDOW episodes → previous difficulty
        """
        self._curriculum_history.append({
            "episode": self._state.episode_count,
            "task_id": self._state.task_id,
            "difficulty": self._curriculum_level,
            "avg_reward": round(episode_avg_reward, 4),
            "accuracy": self._compute_cumulative_accuracy(),
        })

        # Only consider recent episodes at the current difficulty level
        current_level = self._curriculum_level
        recent = [
            h for h in self._curriculum_history[-CURRICULUM_WINDOW * 2:]
            if h["difficulty"] == current_level
        ][-CURRICULUM_WINDOW:]

        if len(recent) < CURRICULUM_WINDOW:
            return

        avg = sum(h["avg_reward"] for h in recent) / len(recent)
        idx = DIFFICULTY_ORDER.index(current_level) if current_level in DIFFICULTY_ORDER else 0

        if avg >= CURRICULUM_PROMOTE_THRESHOLD and idx < len(DIFFICULTY_ORDER) - 1:
            self._curriculum_level = DIFFICULTY_ORDER[idx + 1]
        elif avg <= CURRICULUM_DEMOTE_THRESHOLD and idx > 0:
            self._curriculum_level = DIFFICULTY_ORDER[idx - 1]

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        task_id = kwargs.get("task_id")

        # Auto-curriculum: if no task_id specified and curriculum mode,
        # use the current curriculum level to pick the task
        if task_id is None and kwargs.get("curriculum", False):
            task_id = DIFFICULTY_TO_TASK.get(self._curriculum_level, "task_easy")
            if self._curriculum_level == "mixed":
                task_id = "task_mixed"

        task_ids = self._choose_tasks(seed=seed, task_id=task_id)
        self._episode_plan, self._episode_corpus = self._sample_episode_plan(task_ids, seed=seed)
        self._last_reward_breakdown = {
            "format_score": 0.0,
            "label_score": 0.0,
            "reasoning_score": 0.0,
            "total_reward": 0.0,
        }
        # Preserve cross-episode state
        prev_episode_count = self._state.episode_count
        prev_curriculum_history = self._state.curriculum_history
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=self._episode_plan[0].task_id,
            episode_count=prev_episode_count + 1,
            step_count=0,
            max_steps=len(self._episode_plan),
            current_score=0.0,
            total_reward=0.0,
            best_score=self._persistent_best_score,
            current_clip_id=str(self._episode_plan[0].clip.get("clip_id", "")),
            gt_size=self._gt_store.size(),
            rubric_version=self._rubric.version,
            actions_taken=[],
            episode_history=[],
            rubric_thresholds=self._rubric.get_thresholds_summary(),
            curriculum_level=self._curriculum_level,
            curriculum_history=prev_curriculum_history,
        )
        obs = self._state_to_observation(reward=0.0, done=False)
        if self._manifest_warning:
            obs.info["warning"] = self._manifest_warning
        return obs

    def step(
        self,
        action: Action | dict[str, Any],
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        del timeout_s, kwargs
        if not self._episode_plan:
            self.reset()

        current_index = min(self._state.step_count, len(self._episode_plan) - 1)
        current = self._episode_plan[current_index]
        clip = current.clip
        payload = action.model_dump() if isinstance(action, Action) else dict(action)
        action_obj = Action.model_validate(payload)

        reward_obj = grade(action_obj, clip, self._rubric, self._gt_store, difficulty=current.difficulty)
        reward = float(reward_obj.total)
        self._last_reward_breakdown = {
            "format_score": float(reward_obj.format_score),
            "label_score": float(reward_obj.label_score),
            "reasoning_score": float(reward_obj.reasoning_score),
            "total_reward": float(reward),
        }
        self._state.total_reward += reward
        self._state.current_score = float(self._state.total_reward)
        self._state.best_score = max(self._state.best_score, reward)
        self._persistent_best_score = max(self._persistent_best_score, self._state.best_score)

        action_label = str(action_obj.label).upper()
        submitted_clip_id = str(action_obj.clip_id or clip.get("clip_id", ""))
        current_clip_id = str(clip.get("clip_id", ""))
        corpus = self._episode_corpus.get(current.task_id, [])
        status_updated = False
        for item in corpus:
            if str(item.get("clip_id", "")) == submitted_clip_id:
                item["review_status"] = action_label
                status_updated = True
                break

        if not status_updated and submitted_clip_id != current_clip_id:
            for item in corpus:
                if str(item.get("clip_id", "")) == current_clip_id:
                    item["review_status"] = action_label
                    break

        self._state.actions_taken.append(action_label)
        self._state.episode_history.append(
            EpisodeHistoryItem(
                step=current_index + 1,
                difficulty=current.difficulty,
                clip_id=str(clip.get("clip_id", "")),
                label=action_label,
                expected_label=str(clip.get("expected_label", "")),
                reward=reward,
            )
        )
        self._state.step_count += 1

        done = self._state.step_count >= self._state.max_steps
        if done:
            hard_entry = self._state.episode_history[-1]
            step3_result = {
                "clip": self._episode_plan[-1].clip,
                "action": action_obj.model_dump(),
                "reward": reward,
                "expected_label": self._episode_plan[-1].clip.get("expected_label"),
            }
            try:
                current_difficulty = self._state.task_id.replace("task_", "") if self._state.task_id else "easy"
                gt_promoted = self._gt_store.try_promote(
                    step3_result,
                    episode=self._state.episode_count,
                    difficulty=current_difficulty,
                )
            except ValueError:
                gt_promoted = False
            self._state.gt_size = self._gt_store.size()

            # Compute actual episode accuracy for recalibration
            correct_count = sum(
                1 for h in self._state.episode_history
                if h.label == h.expected_label
            )
            episode_accuracy = correct_count / max(len(self._state.episode_history), 1)

            # Build real performance object for recalibration —
            # place accuracy on the dimension matching current difficulty
            perf_kwargs = {"easy_accuracy": 0.0, "medium_accuracy": 0.0, "hard_accuracy": 0.0}
            if current_difficulty in perf_kwargs:
                perf_kwargs[f"{current_difficulty}_accuracy"] = episode_accuracy
            self._rubric.recalibrate(
                perf=type("Perf", (), perf_kwargs)(),
                current_episode=self._state.episode_count,
            )
            self._state.rubric_version = self._rubric.version
            self._state.rubric_thresholds = self._rubric.get_thresholds_summary()
            # Update multi-episode curriculum tracking
            episode_avg_reward = float(self._state.total_reward) / max(1, self._state.max_steps)
            self._update_curriculum(episode_avg_reward)
            self._state.curriculum_level = self._curriculum_level
            self._state.curriculum_history = self._curriculum_history

            obs = self._state_to_observation(reward=reward, done=True)
            obs.info["gt_promoted"] = bool(gt_promoted)
            obs.info["episode_summary"] = {
                "steps_completed": self._state.step_count,
                "max_steps": self._state.max_steps,
                "total_reward": round(float(self._state.total_reward), 4),
                "average_reward": round(episode_avg_reward, 4),
                "accuracy": self._compute_cumulative_accuracy(),
                "curriculum_level": self._curriculum_level,
                "curriculum_promoted": self._curriculum_level != current_difficulty,
            }
            if self._manifest_warning:
                obs.info["warning"] = self._manifest_warning
            return obs

        next_idx = self._state.step_count
        self._state.task_id = self._episode_plan[next_idx].task_id
        self._state.current_clip_id = str(self._episode_plan[next_idx].clip.get("clip_id", ""))
        obs = self._state_to_observation(reward=reward, done=False)
        if self._manifest_warning:
            obs.info["warning"] = self._manifest_warning
        return obs

    @property
    def state(self) -> State:
        return self._state


ClipQualityEnv = ClipQualityEnvironment
# Backward-compatibility alias (deprecated).
PolicyEvolverEnvironment = ClipQualityEnvironment

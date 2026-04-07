from __future__ import annotations

import argparse
from statistics import mean

from .env import ClipQualityEnv
from .rubric import RubricState


def _heuristic_action(task_id: str, clip: dict) -> dict:
    rubric = RubricState()
    predicted = rubric.derive_label(clip)
    if task_id == "task_easy":
        reasoning = "Clip has dominant, consistent quality signals with clear threshold alignment."
    elif task_id == "task_medium":
        reasoning = "Clip shows mixed cues and one or two borderline metrics, requiring cautious acceptance."
    else:
        reasoning = "Clip contains conflicting hard-case signals requiring conservative quality judgment."
    return {
        "label": predicted,
        "reasoning": reasoning,
        "confidence": 0.75 if predicted != "BORDERLINE" else 0.65,
        "clip_id": clip.get("clip_id"),
    }


def run_training(episodes: int, model_name: str | None = None) -> dict[str, float]:
    del model_name
    env = ClipQualityEnv()

    easy_rewards: list[float] = []
    medium_rewards: list[float] = []
    hard_rewards: list[float] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = bool(obs.done)
        while not done:
            task_id = obs.task_id
            action = _heuristic_action(task_id, obs.clip_metadata.model_dump())
            obs = env.step(action)
            reward = float(obs.reward or 0.0)
            done = bool(obs.done)
            if task_id == "task_easy":
                easy_rewards.append(reward)
            elif task_id == "task_medium":
                medium_rewards.append(reward)
            else:
                hard_rewards.append(reward)

        if ep % 10 == 0:
            print(f"[Episode {ep}] Best={env.state.best_score:.3f} Last={reward:.3f}")

    return {
        "easy": mean(easy_rewards) if easy_rewards else 0.0,
        "medium": mean(medium_rewards) if medium_rewards else 0.0,
        "hard": mean(hard_rewards) if hard_rewards else 0.0,
        "best_score": float(env.state.best_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClipQualityEnv training loop.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes.")
    parser.add_argument("--model-name", type=str, default=None, help="Override MODEL_NAME.")
    args = parser.parse_args()

    summary = run_training(episodes=args.episodes, model_name=args.model_name)
    print("Training summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

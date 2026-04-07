from __future__ import annotations

import os

import gradio as gr

from models import Action
from server.environment import ClipQualityEnvironment
from server.tasks import TASK_REGISTRY


def _run_step(task_id: str, label: str, reasoning: str, confidence: float, clip_id: str) -> str:
    env = ClipQualityEnvironment()
    obs = env.reset(task_id=task_id)
    payload = {
        "label": label,
        "reasoning": (reasoning or "No reasoning provided.").strip(),
        "confidence": float(confidence),
    }
    if clip_id and clip_id.strip():
        payload["clip_id"] = clip_id.strip()

    next_obs = env.step(Action.model_validate(payload))
    current_clip = next_obs.clip_metadata.clip_id
    expected = next_obs.clip_metadata.expected_label
    return (
        f"Clip-quality task: {obs.task_id}\n"
        f"Classified clip: {current_clip}\n"
        f"Expected label: {expected}\n"
        f"Step: {next_obs.step_count}\n"
        f"Reward: {next_obs.reward:.4f}\n"
        f"Done: {next_obs.done}\n"
        f"Best quality score: {next_obs.info.get('best_score', 0.0):.4f}"
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CLIP Quality Analyzer Dashboard") as demo:
        gr.Markdown("# CLIP Quality Analyzer Dashboard")
        gr.Markdown("Reference-style dashboard for clip-quality classification.")

        task_id = gr.Dropdown(choices=list(TASK_REGISTRY.keys()), value="task_easy", label="Clip-Quality Scenario")
        label = gr.Radio(
            choices=["KEEP", "BORDERLINE", "REJECT"],
            value="BORDERLINE",
            label="Predicted Label",
        )
        reasoning = gr.Textbox(label="Reasoning (reference clip metadata)")
        confidence = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Confidence")
        clip_id = gr.Textbox(label="Clip ID override (optional)")
        run_btn = gr.Button("Submit Classification Action")
        output = gr.Textbox(label="Classification Result", lines=8)
        run_btn.click(_run_step, inputs=[task_id, label, reasoning, confidence, clip_id], outputs=[output])
    return demo


def _run_demo() -> None:
    demo = build_demo()
    demo.queue()
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    _run_demo()

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from .models import Action, Observation


LABEL_RE = re.compile(r"<label>\s*(KEEP|BORDERLINE|REJECT)\s*</label>", re.IGNORECASE)
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
CONFIDENCE_RE = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", re.IGNORECASE)


class LLMAgent:
    """OpenAI-client-backed agent for ClipQualityEnv."""

    SYSTEM_PROMPT = (
        "You are a dataset quality analyst for talking-head LoRA training clips.\n"
        "Classify each clip as KEEP, BORDERLINE, or REJECT.\n"
        "Always respond with XML tags exactly:\n"
        "<label>...</label>\n<reasoning>...</reasoning>\n<confidence>0.0-1.0</confidence>"
    )

    def __init__(
        self,
        model_name: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> None:
        self.model_name = model_name or os.environ.get("MODEL_NAME", "meta-llama/Llama-3-70B-Instruct")
        self.api_base_url = api_base_url or os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
        self.api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Either HF_TOKEN or OPENAI_API_KEY must be set")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)

    def act(self, obs: Observation | dict[str, Any]) -> Action:
        observation = obs if isinstance(obs, Observation) else Observation(**obs)
        prompt = self._build_prompt(observation)
        raw = self._call_model(prompt)
        return self._parse_response(raw)

    def _build_prompt(self, obs: Observation) -> str:
        parts = [obs.rubric_summary, ""]
        for h in obs.history:
            parts.append(f"[STEP {h.step} - Previous]")
            parts.append(f"Your label: {h.label} | Reward: {h.reward:.2f}")
            parts.append("")
        parts.append(f"[STEP {obs.step} - Current Clip]")
        parts.append(json.dumps(obs.clip_metadata.model_dump(), indent=2))
        parts.append("")
        parts.append("Classify this clip.")
        return "\n".join(parts)

    def _call_model(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            return "<label>BORDERLINE</label><reasoning>No content.</reasoning><confidence>0.5</confidence>"
        return content

    def _parse_response(self, text: str) -> Action:
        label_match = LABEL_RE.search(text)
        reasoning_match = REASONING_RE.search(text)
        confidence_match = CONFIDENCE_RE.search(text)

        label = label_match.group(1).upper() if label_match else "BORDERLINE"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
        confidence = 0.5
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return Action(label=label, reasoning=reasoning, confidence=confidence)

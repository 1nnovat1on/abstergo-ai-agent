from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

import requests

from agent.actions import ACTION_SCHEMA, DEFAULT_REFLECTION
from agent.state import AgentState

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """
You are an autonomous desktop agent. You control mouse and keyboard. Respond ONLY with JSON following the provided schema.
- Use normalized coordinates (0..1) relative to the latest screenshot.
- Keep actions concise and deterministic.
- Avoid requesting new screenshots unless necessary.
- Summarize inner monologue in rationale fields.

Scene understanding (from vision model):
{scene}

Agent mode: {mode}
Current goal: {goal}
Current task: {task}
Last action: {last_action}
Last action time (UTC): {last_action_time}
Inner monologue summary: {monologue}
Emotion vector (10 floats 0..1): {emotions}
Active window: {active_start} -> {active_stop}
Agent status: {status}
Time since last action (s): {tsla}

Action schema (JSON Schema): {schema}
"""


class HybridPlanner:
    def __init__(
        self,
        florence_base_url_env: str = "FLORENCE_BASE_URL",
        florence_model_env: str = "FLORENCE_MODEL",
        text_base_url_env: str = "TEXT_BASE_URL",
        text_model_env: str = "TEXT_MODEL",
        text_api_key_env: str = "TEXT_API_KEY",
        default_florence_url: str = "http://127.0.0.1:8000/v1",
        default_text_url: str = "http://127.0.0.1:11434/v1",
        default_text_model: str = "deepseek-r1:14b",
    ) -> None:
        self.florence_base_url = os.getenv(florence_base_url_env, default_florence_url).rstrip("/")
        self.florence_model = os.getenv(florence_model_env, "")
        self.text_base_url = os.getenv(text_base_url_env, default_text_url).rstrip("/")
        self.text_model = os.getenv(text_model_env, default_text_model)
        self.text_api_key = os.getenv(text_api_key_env, "")

    def plan(self, state: AgentState, screenshot_b64: Optional[str]) -> Dict[str, Any]:
        vision_scene = self._run_florence(screenshot_b64)
        prompt = PROMPT_TEMPLATE.format(
            scene=json.dumps(vision_scene, ensure_ascii=False, separators=(",", ":")),
            mode=state.current_mode,
            goal=state.current_goal or "<none>",
            task=state.current_task or "<none>",
            last_action=state.last_action or "<none>",
            last_action_time=state.last_action_time or "<never>",
            monologue=state.inner_monologue_summary or "",
            emotions=state.emotion_vector,
            active_start=state.active_window_start or "<unset>",
            active_stop=state.active_window_stop or "<unset>",
            status=state.agent_status,
            tsla=state.time_since_last_action(),
            schema=ACTION_SCHEMA,
        )

        payload = {
            "model": self.text_model,
            "messages": [
                {"role": "system", "content": prompt},
            ],
            "temperature": 0,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if self.text_api_key:
            headers["Authorization"] = f"Bearer {self.text_api_key}"

        try:
            response = requests.post(f"{self.text_base_url}/chat/completions", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            content = response.json()
            text = self._extract_text(content)
            return self._safe_json(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Hybrid planner text LLM failed; returning fallback WAIT action.")
            return self._fallback(f"Hybrid text planning failed: {exc}")

    def _run_florence(self, screenshot_b64: Optional[str]) -> Dict[str, Any]:
        if not screenshot_b64:
            return {"scene": "No screenshot provided"}

        payload: Dict[str, Any] = {
            "inputs": [],
            "parameters": {
                "task": "detailed_scene",
            },
        }

        image_bytes = base64.b64decode(screenshot_b64)
        files = {"image": ("screenshot.png", image_bytes, "image/png")}

        if self.florence_model:
            payload["model"] = self.florence_model

        try:
            response = requests.post(
                f"{self.florence_base_url}/vision",
                data={"payload": json.dumps(payload)},
                files=files,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Florence vision extraction failed; returning minimal scene.")
            return {"scene": f"Vision extraction failed: {exc}"}

    def _extract_text(self, content: Dict[str, Any]) -> str:
        choices = content.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        return text

    def _safe_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            logger.warning("Hybrid text response was empty; returning WAIT fallback.")
            return self._fallback("Planner returned empty response.")

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Hybrid text response was not valid JSON: %s", exc)
            return self._fallback("Planner returned non-JSON response.")

    def _fallback(self, reason: str) -> Dict[str, Any]:
        action = dict(DEFAULT_REFLECTION["actions"][0])
        if reason:
            action["rationale"] = reason
        return {"actions": [action]}

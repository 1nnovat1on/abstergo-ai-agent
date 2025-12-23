from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from agent.actions import ACTION_SCHEMA, DEFAULT_REFLECTION
from agent.state import AgentState

import logging
logger = logging.getLogger(__name__)
logger.warning("LOADED PLANNER FILE: %s", __file__)


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """
You are an autonomous desktop agent. You control mouse and keyboard. Respond ONLY with JSON following the provided schema.
- Use normalized coordinates (0..1) relative to the latest screenshot.
- Keep actions concise and deterministic.
- Avoid requesting new screenshots unless necessary.
- Summarize inner monologue in rationale fields.
- Execute provided actions in order before replanning, and verify the outcome on the next observation before changing course.

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
Screenshot meta (width x height @ dpi): {screenshot_meta}

Action schema (JSON Schema): {schema}
"""


class LocalVLMPlanner:
    def __init__(
        self,
        base_url_env: str = "VLM_BASE_URL",
        model_env: str = "VLM_MODEL",
        api_key_env: str = "VLM_API_KEY",
        default_base_url: str = "http://127.0.0.1:11434",
        default_model: str = "llava:7b",
    ) -> None:
        # Ollama runs a local HTTP server; we only need the host/port, not an OpenAI-style path.
        self.base_url = os.getenv(base_url_env, default_base_url).rstrip("/")
        self.model = os.getenv(model_env, default_model)
        self.api_key = os.getenv(api_key_env, "")
        self.next_allowed_time: float = 0.0

    def plan(
        self,
        state: AgentState,
        screenshot_b64: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        if self.next_allowed_time and now < self.next_allowed_time:
            wait_for = max(0.0, self.next_allowed_time - now)
            return self._fallback(f"Rate limited; retry after {wait_for:.1f}s.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(
            state=state,
            screenshot_b64=screenshot_b64,
            metadata=metadata,
        )

        try:
            response = requests.post(
                self._chat_url(),
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()
            text = self._extract_text(content)
            self.next_allowed_time = 0.0
            return self._safe_json(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Local VLM planning failed; returning fallback WAIT action.")
            self.next_allowed_time = time.time() + self._backoff_seconds(exc)
            return self._fallback(f"Local VLM planning failed: {exc}")

    def _chat_url(self) -> str:
        """Return the Ollama chat endpoint, normalizing away OpenAI-style suffixes."""

        # Strip any trailing path segments that mimic OpenAI (e.g., /v1 or /v1/chat/completions)
        trimmed = self.base_url
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if trimmed.endswith(suffix):
                trimmed = trimmed[: -len(suffix)]
                break

        trimmed = trimmed.rstrip("/")

        # Respect users who already provide /api or /api/chat
        if trimmed.endswith("/api/chat"):
            return trimmed
        if trimmed.endswith("/api"):
            return f"{trimmed}/chat"

        return f"{trimmed}/api/chat"

    def _build_payload(
        self,
        state: AgentState,
        screenshot_b64: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        screenshot_meta = None
        if metadata:
            screenshot_meta = metadata.get("screenshot_meta")

        prompt = PROMPT_TEMPLATE.format(
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
            screenshot_meta=screenshot_meta or "<unknown>",
            schema=ACTION_SCHEMA,
        )

        clean_b64 = None
        if screenshot_b64:
            clean_b64 = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": prompt,
            }
        ]

        if clean_b64:
            messages.append(
                {
                    "role": "user",
                    "content": "Analyze the screenshot and propose actions following the schema.",
                    "images": [clean_b64],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "No screenshot available. Provide actions using the schema.",
                }
            )

        # `format: json` asks Ollama to adhere to JSON output; combined with the system prompt this
        # helps ensure the planner returns a parsable action list.
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
        }

    def _extract_text(self, content: Dict[str, Any]) -> str:
        choices = content.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return message.get("content") or ""

        message = content.get("message") or {}
        return message.get("content") or ""

    def _safe_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            logger.warning("Local VLM response was empty; returning WAIT fallback.")
            return self._fallback("Planner returned empty response.")

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Local VLM response was not valid JSON: %s", exc)
            return self._fallback("Planner returned non-JSON response.")

    def _fallback(self, reason: str) -> Dict[str, Any]:
        action = dict(DEFAULT_REFLECTION["actions"][0])
        if reason:
            action["rationale"] = reason
        return {"actions": [action]}

    def _backoff_seconds(self, exc: Exception) -> float:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                return 10.0
        message = str(exc).lower()
        if "429" in message or "rate" in message or "limit" in message:
            return 10.0
        return 5.0

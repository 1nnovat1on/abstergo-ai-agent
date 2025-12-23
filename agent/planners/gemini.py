# gemini_client.py
from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from agent.actions import ACTION_SCHEMA, DEFAULT_REFLECTION
from agent.state import AgentState


logger = logging.getLogger(__name__)

genai_spec = importlib.util.find_spec("google.generativeai")
genai_available = genai_spec and genai_spec.loader is not None
if genai_available:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
else:
    genai = None
    GenerationConfig = None


PROMPT_TEMPLATE = """
You are an autonomous desktop agent. You control mouse and keyboard.
Respond ONLY with a single JSON object that matches the provided schema. No markdown. No code fences.

Grounding rules:
- All coordinates MUST be normalized (0..1) relative to the latest screenshot provided in this request.
- If no screenshot is provided, do NOT guess coordinates; prefer WAIT or non-visual actions (hotkeys).
- When clicking UI elements, choose a safe click point: near the center of the intended element, not near edges.
- If you are not confident you can target the correct element (confidence < 0.55), return WAIT and explain what you need to see next.
- If a sequence of actions is provided, execute them in order before replanning. Verify the outcome using the next observation before changing course.

Speed & reliability:
- Prefer keyboard shortcuts/hotkeys when they are likely to work (e.g., Ctrl+L, Ctrl+F, Ctrl+S, Ctrl+A/C/V, Alt+Tab, Enter, Esc, Tab).
- Use mouse actions when hotkeys are not available or require visual targeting.

Keep actions concise and deterministic. Return actions in execution order.

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
Screenshot provided this request: {has_screenshot}
Screenshot meta (width x height @ dpi): {screenshot_meta}

Action schema (JSON Schema): {schema}
"""


class GeminiPlanner:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        model_env: str = "GEMINI_MODEL",
    ) -> None:
        self.model_name = os.getenv(model_env, model)
        self.api_key_env = api_key_env
        self.client: Optional[Any] = None
        self.unavailable_reason: Optional[str] = None
        self.next_allowed_time: float = 0.0

        if not genai:
            self.unavailable_reason = "google.generativeai is not installed."
            return

        api_key = os.getenv(api_key_env)
        if not api_key:
            self.unavailable_reason = f"{api_key_env} is not set."
            return

        try:
            genai.configure(api_key=api_key)
            generation_config = self._build_generation_config()
            model_kwargs: Dict[str, Any] = {}
            if generation_config is not None:
                model_kwargs["generation_config"] = generation_config

            self.client = genai.GenerativeModel(self.model_name, **model_kwargs)
        except Exception as exc:  # noqa: BLE001
            self.unavailable_reason = f"Gemini client init failed: {exc}"
            logger.exception(self.unavailable_reason)

    def plan(self, state: AgentState, screenshot_b64: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.client:
            logger.warning("Falling back to default reflection: %s", self.unavailable_reason)
            return self._fallback(self.unavailable_reason or "Gemini client unavailable.")

        now = time.time()
        if self.next_allowed_time and now < self.next_allowed_time:
            wait_for = max(0.0, self.next_allowed_time - now)
            return self._fallback(f"Rate limited; retry after {wait_for:.1f}s.")

        screenshot_meta = metadata.get("screenshot_meta") if metadata else None
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
            has_screenshot=bool(screenshot_b64),
            screenshot_meta=screenshot_meta or "<unknown>",
            schema=ACTION_SCHEMA,
        )

        parts: list[Any] = [prompt]
        if screenshot_b64:
            parts.append(
                {
                    "mime_type": "image/png",
                    "data": screenshot_b64,
                }
            )

        try:
            response = self.client.generate_content(parts, request_options={"timeout": 120})
            text = self._extract_text(response)
            self.next_allowed_time = 0.0
            return self._safe_json(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini planning failed; returning fallback WAIT action.")
            self.next_allowed_time = time.time() + self._backoff_seconds(exc)
            return self._fallback(f"Gemini planning failed: {exc}")

    def _extract_text(self, response: Any) -> str:
        """Extract the model text response from a GenerateContentResponse.

        We prefer the `.text` helper, but fall back to the first candidate part
        to be resilient to SDK changes or empty helper fields.
        """

        if not response:
            return ""

        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text

        return ""

    def _safe_json(self, text: str) -> Dict[str, Any]:
        import json

        cleaned = (text or "").strip()
        if not cleaned:
            logger.warning("Gemini response was empty; returning WAIT fallback.")
            return self._fallback("Planner returned empty response.")

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned)

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Minimal salvage: extract the first JSON object if the model added preamble text
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(0).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                logger.warning("Extracted JSON still invalid: %s", exc)

        logger.warning("Gemini response was not valid JSON; returning WAIT fallback.")
        return self._fallback("Planner returned non-JSON response.")

    def _fallback(self, reason: str) -> Dict[str, Any]:
        action = dict(DEFAULT_REFLECTION["actions"][0])
        if reason:
            action["rationale"] = reason
        return {"actions": [action]}

    def _build_generation_config(self) -> Optional[Any]:
        """Build a GenerationConfig that is compatible with installed SDK.

        Older google-generativeai versions do not support structured output
        fields like ``response_mime_type``/``response_schema``. To keep the
        planner working across versions, we only pass arguments that are
        present in the detected signature.
        """

        if not GenerationConfig:
            return None

        desired = {
            "response_mime_type": "application/json",
            "response_schema": ACTION_SCHEMA,
        }

        try:
            signature = inspect.signature(GenerationConfig)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.debug("Could not inspect GenerationConfig signature; using defaults.")
            signature = None

        supported_kwargs: Dict[str, Any] = {}
        for name, value in desired.items():
            if signature and name not in signature.parameters:
                logger.debug("GenerationConfig missing %s; skipping.", name)
                continue
            supported_kwargs[name] = value

        if not supported_kwargs:
            return None

        try:
            return GenerationConfig(**supported_kwargs)
        except TypeError:
            logger.warning("GenerationConfig rejected structured output args; continuing without.")
            return None

    def _backoff_seconds(self, exc: Exception) -> float:
        message = str(exc).lower()
        if "429" in message or "rate" in message or "exhausted" in message:
            return 10.0
        return 5.0

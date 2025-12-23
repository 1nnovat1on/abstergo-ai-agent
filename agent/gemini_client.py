from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional

from .actions import ACTION_SCHEMA, DEFAULT_REFLECTION
from .state import AgentState


genai_spec = importlib.util.find_spec("google.generativeai")
genai_available = genai_spec and genai_spec.loader is not None
if genai_available:
    import google.generativeai as genai
else:
    genai = None


PROMPT_TEMPLATE = """
You are an autonomous desktop agent. You control mouse and keyboard. Respond ONLY with JSON following the provided schema.
- Use normalized coordinates (0..1) relative to the latest screenshot.
- Keep actions concise and deterministic.
- Avoid requesting new screenshots unless necessary.
- Summarize inner monologue in rationale fields.

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


class GeminiPlanner:
    def __init__(self, model: str = "gemini-1.5-flash", api_key_env: str = "GEMINI_API_KEY") -> None:
        self.model_name = model
        self.api_key_env = api_key_env
        self.client = None
        if genai:
            api_key = os.getenv(api_key_env)
            if api_key:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)

    def plan(self, state: AgentState, screenshot_b64: Optional[str]) -> Dict[str, Any]:
        if not self.client:
            return DEFAULT_REFLECTION

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
            schema=ACTION_SCHEMA,
        )

        parts: list[Any] = [prompt]
        if screenshot_b64:
            parts.append({
                "mime_type": "image/png",
                "data": screenshot_b64,
            })

        response = self.client.generate_content(parts, request_options={"timeout": 120})
        text = response.text or ""
        return self._safe_json(text)

    def _safe_json(self, text: str) -> Dict[str, Any]:
        import json

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return DEFAULT_REFLECTION

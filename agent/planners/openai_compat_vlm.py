from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from agent.actions import DEFAULT_REFLECTION, SUPPORTED_ACTIONS
from agent.state import AgentState

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """
You are an autonomous desktop agent. Control mouse and keyboard to accomplish the goal. Respond ONLY with raw JSON.

Allowed actions: {actions}
Required keys per action: action, confidence, rationale. Optional: target (x,y,width,height normalized 0..1), text, keys, scroll, wait_seconds, expected_outcome.
Rules:
- Output strictly a JSON object with an "actions" array. No code fences or prose.
- Keep plans short (1-3 steps), deterministic, and aligned to the latest screenshot (image may be downscaled; coordinates remain normalized 0..1).
- Prefer WAIT when uncertain and avoid requesting extra screenshots.
- Execute listed actions in order, then observe before changing course.

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

Example (do not describe it, just follow the structure):
{{"actions":[{{"action":"WAIT","confidence":0.55,"rationale":"Review the scene","expected_outcome":"Next step chosen","wait_seconds":2.0}}]}}
"""


class LocalVLMPlanner:
    def __init__(
        self,
        base_url_env: str = "VLM_BASE_URL",
        model_env: str = "VLM_MODEL",
        api_key_env: str = "VLM_API_KEY",
        default_base_url: str = "http://127.0.0.1:11434",
        default_model: str = "qwen3-vl",
    ) -> None:
        # Ollama runs a local HTTP server; we only need the host/port, not an OpenAI-style path.
        self.base_url = os.getenv(base_url_env, default_base_url).rstrip("/")
        self.model = os.getenv(model_env, default_model)
        self.api_key = os.getenv(api_key_env, "")
        self.next_allowed_time: float = 0.0
        self.failure_count: int = 0
        self._request_lock = threading.Lock()

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

        if not self._request_lock.acquire(blocking=False):
            self.next_allowed_time = max(self.next_allowed_time, time.time() + 1.0)
            return self._fallback("Planner busy; waiting for previous request to finish.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(
            state=state,
            screenshot_b64=screenshot_b64,
            metadata=metadata,
        )

        last_error: Optional[Exception] = None

        try:
            for url in self._chat_urls():
                try:
                    logger.warning("POSTING TO VLM: %s", url)
                    logger.warning(
                        "MODEL=%s has_image=%s b64_len=%s",
                        self.model,
                        bool(screenshot_b64),
                        len(screenshot_b64 or ""),
                    )
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=45,
                    )
                    if response.status_code == 404:
                        last_error = requests.HTTPError("404 Not Found", response=response)
                        logger.warning("VLM endpoint 404; trying next fallback if available.")
                        continue

                    response.raise_for_status()
                    content = response.json()
                    text = self._extract_text(content)
                    self.next_allowed_time = 0.0
                    self.failure_count = 0
                    logger.warning("VLM RESP status=%s", response.status_code)

                    return self._safe_json(text)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 404:
                        continue
                    raise

            raise last_error or RuntimeError("Local VLM call failed without error detail")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Local VLM planning failed; returning fallback WAIT action.")
            self.failure_count += 1
            self.next_allowed_time = time.time() + self._backoff_seconds(exc)
            return self._fallback(f"Local VLM planning failed: {exc}")
        finally:
            self._request_lock.release()

    def _chat_urls(self) -> list[str]:
        """Return a list of chat endpoints to try (OpenAI-style then Ollama)."""

        trimmed = self.base_url.rstrip("/")

        # Respect users who provide explicit OpenAI-style paths.
        openai_url = None
        if trimmed.endswith("/v1/chat/completions"):
            openai_url = trimmed
            trimmed = trimmed[: -len("/v1/chat/completions")]
        elif trimmed.endswith("/chat/completions"):
            openai_url = trimmed
            trimmed = trimmed[: -len("/chat/completions")]
        elif trimmed.endswith("/v1"):
            openai_url = f"{trimmed}/chat/completions"
            trimmed = trimmed[: -len("/v1")]
        else:
            openai_url = f"{trimmed}/v1/chat/completions"

        # Respect users who already provide /api or /api/chat for Ollama.
        if trimmed.endswith("/api/chat"):
            ollama_url = trimmed
        elif trimmed.endswith("/api"):
            ollama_url = f"{trimmed}/chat"
        else:
            ollama_url = f"{trimmed}/api/chat"

        if openai_url == ollama_url:
            return [openai_url]
        return [openai_url, ollama_url]

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
            actions=", ".join(SUPPORTED_ACTIONS),
        )

        clean_b64 = None
        if screenshot_b64:
            clean_b64 = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64

        messages: List[Dict[str, Any]] = []

        if clean_b64:
            messages.append({
                "role": "user",
                "content": prompt + "\n\nAnalyze the screenshot and respond with JSON only.",
                "images": [clean_b64],
            })
        else:
            messages.append({
                "role": "user",
                "content": prompt + "\n\nNo screenshot available. Provide actions using the same JSON format.",
            })

        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": 220,
                "temperature": 0.0,
                "top_p": 0.9,
                "seed": 1,
            },
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
        scale = min(self.failure_count, 4)
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                return min(5.0 * (2**scale), 60.0)
        message = str(exc).lower()
        if "429" in message or "rate" in message or "limit" in message:
            return min(5.0 * (2**scale), 60.0)
        return min(3.0 * (2**scale), 45.0)

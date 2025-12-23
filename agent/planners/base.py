from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from agent.state import AgentState


class Planner(Protocol):
    def plan(self, state: AgentState, screenshot_b64: Optional[str]) -> Dict[str, Any]:
        ...

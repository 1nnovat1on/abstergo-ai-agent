from __future__ import annotations

import os

from agent.planners.base import Planner
from agent.planners.gemini import GeminiPlanner
from agent.planners.hybrid_florence_text import HybridPlanner
from agent.planners.openai_compat_vlm import LocalVLMPlanner


def create_planner() -> Planner:
    backend = os.getenv("PLANNER_BACKEND", "gemini").lower()
    if backend == "vlm":
        return LocalVLMPlanner()
    if backend == "hybrid":
        return HybridPlanner()
    return GeminiPlanner()

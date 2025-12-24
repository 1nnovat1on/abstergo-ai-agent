from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .actions import ActionStep, ActionTarget
from .platform_agent import PlatformAdapter, Screenshot


class ActionExecutor:
    def __init__(self, adapter: PlatformAdapter) -> None:
        self.adapter = adapter
        self.last_capture_size: Optional[Tuple[int, int]] = None

    def _resolve_coords(self, target: ActionTarget, screenshot_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        width, height = screenshot_size or self.adapter.screen_size()
        x = int(target.x * width)
        y = int(target.y * height)
        return x, y

    def execute(self, actions: List[ActionStep], screenshot: Optional[Screenshot]) -> List[str]:
        executed: List[str] = []
        if screenshot:
            self.last_capture_size = (screenshot.width, screenshot.height)
        size = self.last_capture_size

        for step in actions:
            if step.action != "WAIT" and step.confidence < 0.55:
                continue
            summary = self._execute_step(step, size)
            executed.append(summary)
        return executed

    def _execute_step(self, step: ActionStep, screenshot_size: Optional[Tuple[int, int]]) -> str:
        target_coords = None
        if step.target:
            target_coords = self._resolve_coords(step.target, screenshot_size)

        if step.action == "MOVE" and target_coords:
            self.adapter.move(*target_coords)
        elif step.action == "CLICK" and target_coords:
            self.adapter.click(*target_coords, clicks=1)
        elif step.action == "DOUBLE_CLICK" and target_coords:
            self.adapter.click(*target_coords, clicks=2)
        elif step.action == "DRAG" and step.target:
            end = target_coords or (0, 0)
            screen_w, screen_h = self.adapter.screen_size()
            self.adapter.drag((screen_w // 2, screen_h // 2), end)
        elif step.action == "SCROLL":
            dx = int(step.scroll.get("dx", 0)) if step.scroll else 0
            dy = int(step.scroll.get("dy", 0)) if step.scroll else 0
            self.adapter.scroll(dx, dy)
        elif step.action == "TYPE" and step.text:
            self.adapter.type_text(step.text)
        elif step.action == "KEYPRESS" and step.keys:
            self.adapter.keypress(step.keys, repeat=step.repeat, hold_ms=step.hold_ms)
        elif step.action == "WAIT":
            time.sleep(step.wait_seconds)
        elif step.action == "FOCUS_WINDOW":
            # Placeholder: focus logic depends on OS-specific integration.
            time.sleep(0.05)

        return step.summary()

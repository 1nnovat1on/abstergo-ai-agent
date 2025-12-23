from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import hashlib

from .actions import ActionStep, parse_actions
from .controller import ActionExecutor
from .planners.factory import create_planner
from .platform import PlatformAdapter, Screenshot, default_adapter
from .state import StateManager, utc_now
from .storage import log_action, snapshot_state


class AgentOrchestrator:
    def __init__(self, state_path: Path, data_dir: Path, adapter: Optional[PlatformAdapter] = None) -> None:
        self.state_manager = StateManager(state_path)
        self.planner = create_planner()
        self.adapter = adapter or default_adapter()
        self.executor = ActionExecutor(self.adapter)
        self.data_dir = data_dir
        self.log_path = data_dir / "actions.log"
        self.snapshot_dir = data_dir / "snapshots"
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.last_screenshot: Optional[Screenshot] = None
        self.last_vision_time: Optional[float] = None
        self.pending_actions: list[ActionStep] = []
        self.cached_plan_key: Optional[str] = None
        self.cached_plan_response: Optional[dict[str, Any]] = None
        self.cached_plan_consumed: bool = False
        self.last_logged: Optional[tuple[str, str]] = None

    @property
    def state(self):
        return self.state_manager.state

    def start(self, mode: str, goal: str, active_start: Optional[str], active_stop: Optional[str]) -> None:
        with self.lock:
            if self.running:
                return
            self.state_manager.set_goal(goal, mode)
            self.state_manager.set_active_window(active_start, active_stop)
            self.state_manager.mark_active()
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.state_manager.mark_stopped()
        if self.thread:
            self.thread.join(timeout=1.0)

    def _in_active_window(self) -> bool:
        if not self.state.active_window_start or not self.state.active_window_stop:
            return True
        now = datetime.now()
        start = datetime.strptime(self.state.active_window_start, "%H:%M").time()
        stop = datetime.strptime(self.state.active_window_stop, "%H:%M").time()
        return start <= now.time() <= stop

    def _reflect(self) -> None:
        summary = "Sleeping and reflecting on recent actions."
        self.state.decay_emotions()
        self.state_manager.update_monologue(summary)
        snapshot_state(self.snapshot_dir, self.state)

    def _maybe_capture(self) -> Optional[Screenshot]:
        now = time.time()
        if self.last_vision_time and now - self.last_vision_time < 5:
            return None
        screenshot = self.adapter.capture()
        self.last_screenshot = screenshot
        self.last_vision_time = now
        return screenshot

    def _screenshot_key(self, screenshot: Optional[Screenshot]) -> Optional[str]:
        if not screenshot:
            return None
        try:
            # Hash raw pixel bytes to avoid recompressing base64 for cache keys.
            digest = hashlib.sha256(screenshot.image.tobytes()).hexdigest()
            dpi = getattr(screenshot, "dpi", (96.0, 96.0))
            return f"{digest}:{screenshot.width}x{screenshot.height}:dpi={dpi}"
        except Exception:
            return None

    def _run_loop(self) -> None:
        while True:
            with self.lock:
                if not self.running:
                    break

            if not self._in_active_window():
                self.state_manager.mark_sleeping()
                self._reflect()
                time.sleep(5)
                continue

            self.state_manager.mark_active()
            screenshot = self._maybe_capture()
            screenshot_to_use = screenshot or self.last_screenshot
            screenshot_b64 = screenshot_to_use.to_base64() if screenshot_to_use else None
            screenshot_key = self._screenshot_key(screenshot_to_use)
            screenshot_meta = None
            if screenshot_to_use:
                dpi = getattr(screenshot_to_use, "dpi", (96.0, 96.0))
                screenshot_meta = f"{screenshot_to_use.width}x{screenshot_to_use.height} @ {dpi} dpi"

            sleep_after_loop = 0.5

            if not self.pending_actions:
                perception_marker = self.last_vision_time or 0
                context_key = (
                    f"{screenshot_key}:{perception_marker}:{self.state.current_goal}:"
                    f"{self.state.current_task}:{self.state.current_mode}"
                )
                plan_response: Optional[Dict[str, Any]] = None

                if self.cached_plan_key == context_key:
                    if self.cached_plan_response is not None and not self.cached_plan_consumed:
                        plan_response = self.cached_plan_response
                    else:
                        sleep_after_loop = 0.2
                else:
                    plan_response = self.planner.plan(
                        self.state,
                        screenshot_b64,
                        metadata={"screenshot_key": screenshot_key, "screenshot_meta": screenshot_meta},
                    )
                    self.cached_plan_key = context_key
                    self.cached_plan_response = plan_response
                    self.cached_plan_consumed = False

                if plan_response is not None:
                    actions = parse_actions(plan_response, min_confidence=0.55)
                    # Allow WAIT even if low confidence to avoid churn.
                    filtered_actions = [a for a in actions if a.action == "WAIT" or a.confidence >= 0.55]
                    self.pending_actions.extend(filtered_actions)
                else:
                    sleep_after_loop = 0.2

            if self.pending_actions:
                actions_to_run = list(self.pending_actions)
                self.pending_actions.clear()

                summaries = self.executor.execute(actions_to_run, screenshot_to_use)

                for summary in summaries:
                    self.state_manager.update_after_action(summary)
                    plan_payload = {"plan_key": self.cached_plan_key}
                    if not self.last_logged or self.last_logged != (summary, self.cached_plan_key or ""):
                        log_action(self.log_path, summary, plan_payload)
                        self.last_logged = (summary, self.cached_plan_key or "")

                # If the last action was a WAIT, reduce extra sleep; otherwise keep small cushion.
                if actions_to_run and actions_to_run[-1].action == "WAIT":
                    sleep_after_loop = 0.1
                self.cached_plan_consumed = True
                self.last_vision_time = None

            self.state.stimulate_emotions(0.02)
            self.state_manager.save()
            snapshot_state(self.snapshot_dir, self.state)
            time.sleep(sleep_after_loop)

    def get_state(self) -> Dict[str, Any]:
        return self.state.to_dict()

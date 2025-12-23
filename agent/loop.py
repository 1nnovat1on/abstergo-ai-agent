from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .actions import parse_actions
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
            screenshot_b64 = screenshot.to_base64() if screenshot else None

            plan_response = self.planner.plan(self.state, screenshot_b64)
            actions = parse_actions(plan_response)
            summaries = self.executor.execute(actions, screenshot)

            for summary in summaries:
                self.state_manager.update_after_action(summary)
                log_action(self.log_path, summary, {"plan": plan_response})

            self.state.stimulate_emotions(0.02)
            self.state_manager.save()
            snapshot_state(self.snapshot_dir, self.state)
            time.sleep(2)

    def get_state(self) -> Dict[str, Any]:
        return self.state.to_dict()

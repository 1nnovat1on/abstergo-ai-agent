from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AgentState:
    current_mode: str = "GOAL"  # GOAL | FREEROAM
    current_goal: str = ""
    current_task: str = ""
    last_action: str = ""
    last_action_time: Optional[str] = None
    agent_status: str = "STOPPED"  # ACTIVE | SLEEPING | STOPPED
    inner_monologue_summary: str = ""
    emotion_vector: List[float] = field(default_factory=lambda: [0.2] * 10)
    session_start_time: str = field(default_factory=lambda: utc_now().strftime(ISO_FORMAT))
    time_active_today: float = 0.0
    active_window_start: Optional[str] = None  # HH:MM local time
    active_window_stop: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_mode": self.current_mode,
            "current_goal": self.current_goal,
            "current_task": self.current_task,
            "last_action": self.last_action,
            "last_action_time": self.last_action_time,
            "agent_status": self.agent_status,
            "inner_monologue_summary": self.inner_monologue_summary,
            "emotion_vector": self.emotion_vector,
            "session_start_time": self.session_start_time,
            "time_active_today": self.time_active_today,
            "active_window_start": self.active_window_start,
            "active_window_stop": self.active_window_stop,
            "time_since_last_action": self.time_since_last_action(),
        }

    def time_since_last_action(self) -> Optional[float]:
        if not self.last_action_time:
            return None
        try:
            last = datetime.strptime(self.last_action_time, ISO_FORMAT)
        except ValueError:
            return None
        delta = utc_now() - last.replace(tzinfo=timezone.utc)
        return max(delta.total_seconds(), 0.0)

    def update_after_action(self, summary: str) -> None:
        now = utc_now().strftime(ISO_FORMAT)
        self.last_action = summary
        self.last_action_time = now
        self.agent_status = "ACTIVE"
        self.inner_monologue_summary = self.inner_monologue_summary or summary

    def update_monologue(self, reflection: str) -> None:
        self.inner_monologue_summary = reflection

    def decay_emotions(self, decay: float = 0.03) -> None:
        self.emotion_vector = [max(0.0, min(1.0, value * (1 - decay))) for value in self.emotion_vector]

    def stimulate_emotions(self, delta: float = 0.05) -> None:
        self.emotion_vector = [max(0.0, min(1.0, value + delta)) for value in self.emotion_vector]


class StateManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> AgentState:
        if not self.path.exists():
            return AgentState()
        try:
            data = json.loads(self.path.read_text())
            return AgentState(
                current_mode=data.get("current_mode", "GOAL"),
                current_goal=data.get("current_goal", ""),
                current_task=data.get("current_task", ""),
                last_action=data.get("last_action", ""),
                last_action_time=data.get("last_action_time"),
                agent_status=data.get("agent_status", "STOPPED"),
                inner_monologue_summary=data.get("inner_monologue_summary", ""),
                emotion_vector=data.get("emotion_vector", [0.2] * 10),
                session_start_time=data.get("session_start_time", utc_now().strftime(ISO_FORMAT)),
                time_active_today=data.get("time_active_today", 0.0),
                active_window_start=data.get("active_window_start"),
                active_window_stop=data.get("active_window_stop"),
            )
        except json.JSONDecodeError:
            return AgentState()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state.to_dict(), indent=2))

    def set_goal(self, goal: str, mode: str) -> None:
        self.state.current_goal = goal
        self.state.current_mode = mode
        self.save()

    def set_active_window(self, start: Optional[str], stop: Optional[str]) -> None:
        self.state.active_window_start = start
        self.state.active_window_stop = stop
        self.save()

    def mark_sleeping(self) -> None:
        self.state.agent_status = "SLEEPING"
        self.save()

    def mark_active(self) -> None:
        self.state.agent_status = "ACTIVE"
        self.save()

    def mark_stopped(self) -> None:
        self.state.agent_status = "STOPPED"
        self.save()

    def update_monologue(self, summary: str) -> None:
        self.state.update_monologue(summary)
        self.save()

    def update_after_action(self, summary: str) -> None:
        self.state.update_after_action(summary)
        self.save()

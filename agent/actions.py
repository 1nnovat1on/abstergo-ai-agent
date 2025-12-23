# actions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


SUPPORTED_ACTIONS = [
    "MOVE",
    "CLICK",
    "DOUBLE_CLICK",
    "DRAG",
    "SCROLL",
    "TYPE",
    "KEYPRESS",
    "WAIT",
    "FOCUS_WINDOW",
]


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action", "confidence"],
                "properties": {
                    "action": {"enum": SUPPORTED_ACTIONS},
                    "target": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "y": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "width": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "height": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                    },
                    "text": {"type": "string"},
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    # Optional: allows things like Alt+Tab twice, or holding modifier keys briefly
                    "repeat": {"type": "integer", "minimum": 1, "maximum": 10},
                    "hold_ms": {"type": "integer", "minimum": 0, "maximum": 2000},
                    "scroll": {
                        "type": "object",
                        "properties": {
                            "dx": {"type": "number"},
                            "dy": {"type": "number"},
                        },
                    },
                    "wait_seconds": {"type": "number", "minimum": 0.0, "maximum": 30.0},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                    "expected_outcome": {"type": "string"},
                },
            },
        }
    },
}


@dataclass
class ActionTarget:
    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass
class ActionStep:
    action: str
    confidence: float
    rationale: str = ""
    expected_outcome: str = ""
    target: Optional[ActionTarget] = None
    text: Optional[str] = None
    keys: Optional[List[str]] = None
    scroll: Optional[Dict[str, float]] = None
    wait_seconds: float = 0.0
    # Hotkey helpers
    repeat: int = 1
    hold_ms: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionStep":
        target = None
        if target_data := data.get("target"):
            target = ActionTarget(
                x=float(target_data.get("x", 0.0)),
                y=float(target_data.get("y", 0.0)),
                width=target_data.get("width"),
                height=target_data.get("height"),
            )

        repeat = data.get("repeat", 1)
        hold_ms = data.get("hold_ms", 0)
        try:
            repeat = int(repeat)
        except Exception:
            repeat = 1
        try:
            hold_ms = int(hold_ms)
        except Exception:
            hold_ms = 0

        if repeat < 1:
            repeat = 1
        if repeat > 10:
            repeat = 10
        if hold_ms < 0:
            hold_ms = 0
        if hold_ms > 2000:
            hold_ms = 2000

        return cls(
            action=data.get("action", "WAIT"),
            confidence=float(data.get("confidence", 0.0)),
            rationale=data.get("rationale", ""),
            expected_outcome=data.get("expected_outcome", ""),
            target=target,
            text=data.get("text"),
            keys=data.get("keys"),
            scroll=data.get("scroll"),
            wait_seconds=float(data.get("wait_seconds", 0.0)),
            repeat=repeat,
            hold_ms=hold_ms,
        )

    def summary(self) -> str:
        target_desc = f" @ ({self.target.x:.2f},{self.target.y:.2f})" if self.target else ""
        if self.action == "KEYPRESS" and self.keys:
            keys = "+".join(self.keys)
            rep = f"x{self.repeat}" if self.repeat and self.repeat > 1 else ""
            hold = f" hold={self.hold_ms}ms" if self.hold_ms else ""
            return f"{self.action}({keys}{rep}{hold}) ({self.confidence:.2f})"
        return f"{self.action}{target_desc} ({self.confidence:.2f})"


def parse_actions(response: Dict[str, Any], min_confidence: float = 0.0) -> List[ActionStep]:
    actions: List[ActionStep] = []
    for raw in response.get("actions", []):
        step = ActionStep.from_dict(raw)
        if step.action != "WAIT" and step.confidence < min_confidence:
            continue
        actions.append(step)
    return actions


DEFAULT_REFLECTION = {
    "actions": [
        {
            "action": "WAIT",
            "confidence": 0.5,
            "rationale": "Holding until vision is needed.",
            "expected_outcome": "Agent remains idle.",
            "wait_seconds": 2.0,
        }
    ]
}

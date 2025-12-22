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
        )

    def summary(self) -> str:
        target_desc = f" @ ({self.target.x:.2f},{self.target.y:.2f})" if self.target else ""
        return f"{self.action}{target_desc} ({self.confidence:.2f})"


def parse_actions(response: Dict[str, Any]) -> List[ActionStep]:
    actions: List[ActionStep] = []
    for raw in response.get("actions", []):
        actions.append(ActionStep.from_dict(raw))
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

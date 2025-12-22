from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .state import AgentState


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_action(log_path: Path, action_summary: str, metadata: Dict[str, Any] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": utc_timestamp(),
        "summary": action_summary,
        "meta": metadata or {},
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def snapshot_state(snapshot_dir: Path, state: AgentState) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"state-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(state.to_dict(), indent=2))

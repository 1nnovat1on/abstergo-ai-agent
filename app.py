from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from agent.loop import AgentOrchestrator


DATA_DIR = Path("data")
STATE_PATH = DATA_DIR / "state.json"

app = Flask(__name__, static_folder="static")
CORS(app)

orchestrator = AgentOrchestrator(STATE_PATH, DATA_DIR)


@app.route("/")
def index() -> Any:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state", methods=["GET"])
def api_state() -> Any:
    return jsonify(orchestrator.get_state())


@app.route("/api/start", methods=["POST"])
def api_start() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True)
    mode = payload.get("mode", "GOAL")
    goal = payload.get("goal", "")
    active_start = payload.get("active_start")
    active_stop = payload.get("active_stop")
    orchestrator.start(mode, goal, active_start, active_stop)
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop() -> Any:
    orchestrator.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/config", methods=["POST"])
def api_config() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True)
    goal = payload.get("goal", orchestrator.state.current_goal)
    mode = payload.get("mode", orchestrator.state.current_mode)
    active_start = payload.get("active_start", orchestrator.state.active_window_start)
    active_stop = payload.get("active_stop", orchestrator.state.active_window_stop)
    orchestrator.state_manager.set_goal(goal, mode)
    orchestrator.state_manager.set_active_window(active_start, active_stop)
    return jsonify({"status": "updated", "state": orchestrator.get_state()})


@app.route("/api/logs", methods=["GET"])
def api_logs() -> Any:
    log_path = DATA_DIR / "actions.log"
    if not log_path.exists():
        return jsonify([])
    lines = log_path.read_text().splitlines()[-50:]
    return jsonify([json.loads(line) for line in lines])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

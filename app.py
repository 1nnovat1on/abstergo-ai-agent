from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import atexit
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from agent.loop import AgentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def ensure_servers_running() -> List[subprocess.Popen]:
    """Ensures that the vision and text servers are running if hybrid mode is enabled."""
    backend = os.getenv("PLANNER_BACKEND", "gemini").lower()
    if backend != "hybrid":
        return []

    logger.info("Hybrid backend detected. Ensuring model servers are running...")
    processes = []

    # Check/Start Vision Server
    vision_port = 8001
    # Very basic check: just see if we can connect or if we assume they aren't running
    # For this implementation, we will try to spawn them.
    # Ideally, we should check if the port is in use.

    env = os.environ.copy()

    # Start Vision Server
    logger.info(f"Starting Vision Server on port {vision_port}...")
    vision_proc = subprocess.Popen(
        [sys.executable, "agent/vision_server.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes.append(vision_proc)

    # Start Text Server
    text_port = 11434
    logger.info(f"Starting Text Server on port {text_port}...")
    text_proc = subprocess.Popen(
        [sys.executable, "agent/text_server.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes.append(text_proc)

    # Give them a moment to start
    time.sleep(5)

    # check if they died immediately
    if vision_proc.poll() is not None:
        out, err = vision_proc.communicate()
        logger.error(f"Vision server failed to start: {err.decode()}")

    if text_proc.poll() is not None:
        out, err = text_proc.communicate()
        logger.error(f"Text server failed to start: {err.decode()}")

    return processes


if __name__ == "__main__":
    server_procs = []
    # Only start servers if not in reloader (Flask debug mode spawns two processes)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        server_procs = ensure_servers_running()

    def cleanup():
        for proc in server_procs:
            if proc.poll() is None:
                logger.info(f"Terminating subprocess {proc.pid}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    atexit.register(cleanup)

    try:
        app.run(host="0.0.0.0", port=8000, debug=True)
    finally:
        cleanup()

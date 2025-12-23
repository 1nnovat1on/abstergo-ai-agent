# abstergo-ai-agent

Local autonomous UI agent powered by pluggable planners (Gemini by default). The agent runs inside a simple Flask app with a single-page UI to start/stop, choose Goal vs Free Roam modes, and view live status.

## Features
- Event-driven perception → decision → action loop with on-demand screenshots
- Action grammar (move, click, drag, scroll, type, keypress, wait, focus placeholder) using normalized coordinates
- Persistent agent state (goal, task, mode, status, inner monologue summary, emotion vector)
- Sleeping/reflection loop during idle windows with snapshot logging
- Pluggable platform adapter (pyautogui/mss when available, null adapter fallback)
- Pluggable planners that all return the same JSON action schema:
  - **Gemini** (default) — `PLANNER_BACKEND=gemini`
  - **Local VLM** (e.g., LLaVA via OpenAI-compatible API) — `PLANNER_BACKEND=vlm`
  - **Hybrid** (Florence-2 vision → text LLM for reasoning) — `PLANNER_BACKEND=hybrid`
- Local HTML UI for control + telemetry

## Installation
1. Install dependencies (Python 3.10+ recommended):
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Choose your planner backend (defaults to Gemini if unset):
   - **Gemini** (structured output supported):
     - Set `GEMINI_API_KEY`
     - Optional: `GEMINI_MODEL` (default `gemini-2.5-flash`)
   - **Local VLM** (OpenAI-compatible, e.g., LLaVA):
     - Set `PLANNER_BACKEND=vlm`
     - Optional: `VLM_BASE_URL` (default `http://127.0.0.1:11434`)
     - Optional: `VLM_MODEL` (e.g., `llava:7b`)
     - Optional: `VLM_API_KEY` if your endpoint requires it
     - The planner first tries the OpenAI-compatible path (`/v1/chat/completions`) and falls back to Ollama's native `/api/chat` endpoint if the first call returns 404. Keep `VLM_BASE_URL` set to the root (e.g., `http://127.0.0.1:11434`) so either path works.
   - **Hybrid** (Florence-2 for vision, text LLM for reasoning):
     - Set `PLANNER_BACKEND=hybrid`
     - Vision: `FLORENCE_BASE_URL` (default `http://127.0.0.1:8000/v1`), optional `FLORENCE_MODEL`
     - Text LLM: `TEXT_BASE_URL` (default `http://127.0.0.1:11434/v1`), `TEXT_MODEL` (default `deepseek-r1:14b`), optional `TEXT_API_KEY`
3. Run the server:
   ```bash
   python app.py
   ```
4. Open http://localhost:8000 to start/stop the agent and configure mode, goal, and active window.

The agent writes state + logs to the `data/` folder. By default, if `pyautogui` is unavailable it falls back to a no-op adapter so you can exercise the UI without controlling your machine.

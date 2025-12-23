# abstergo-ai-agent

Local autonomous UI agent powered by Google Gemini. The agent runs inside a simple Flask app with a single-page UI to start/stop, choose Goal vs Free Roam modes, and view live status.

## Features
- Event-driven perception → decision → action loop with on-demand screenshots
- Action grammar (move, click, drag, scroll, type, keypress, wait, focus placeholder) using normalized coordinates
- Persistent agent state (goal, task, mode, status, inner monologue summary, emotion vector)
- Sleeping/reflection loop during idle windows with snapshot logging
- Pluggable platform adapter (pyautogui/mss when available, null adapter fallback)
- Gemini integration template that returns structured JSON action plans
- Local HTML UI for control + telemetry

## Getting started
1. Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Export your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-key"
   ```
3. Run the server:
   ```bash
   python app.py
   ```
4. Open http://localhost:8000 to start/stop the agent and configure mode, goal, and active window.

The agent writes state + logs to the `data/` folder. By default, if `pyautogui` is unavailable it falls back to a no-op adapter so you can exercise the UI without controlling your machine.

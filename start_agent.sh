#!/bin/bash
echo "Starting Vision Server on port 8001..."
python agent/vision_server.py > vision.log 2>&1 &
VISION_PID=$!

echo "Starting Text Server on port 11434..."
python agent/text_server.py > text.log 2>&1 &
TEXT_PID=$!

echo "Waiting for servers to initialize..."
sleep 10

echo "Starting Agent App on port 8000..."
export PLANNER_BACKEND=hybrid
export FLORENCE_BASE_URL=http://127.0.0.1:8001/v1
export TEXT_BASE_URL=http://127.0.0.1:11434/v1
python app.py

# Cleanup
kill $VISION_PID
kill $TEXT_PID

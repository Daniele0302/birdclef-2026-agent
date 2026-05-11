#!/bin/bash
cd /Users/danielemalerba/Downloads/birdclef-agent
PID=$(pgrep -f train_strong_v40b.py | head -1)
echo "[wait_then_eval_3way] watching PID=$PID" >&2
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
echo "[wait_then_eval_3way] training exited, running eval" >&2
.venv/bin/python eval_v40_3way.py

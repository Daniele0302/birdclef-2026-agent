#!/bin/bash
cd /Users/danielemalerba/Downloads/birdclef-agent
# Wait until v40 training process exits (PID 95718)
while kill -0 95718 2>/dev/null; do sleep 10; done
echo "[wait_then_eval] training pid 95718 has exited" >&2
echo "[wait_then_eval] running eval_v40_local.py" >&2
.venv/bin/python eval_v40_local.py

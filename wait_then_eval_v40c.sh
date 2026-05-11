#!/bin/bash
cd /Users/danielemalerba/Downloads/birdclef-agent
PID=$(pgrep -f train_strong_v40c.py | head -1)
echo "[wait_then_eval_v40c] watching PID=$PID" >&2
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
echo "[wait_then_eval_v40c] training exited, running eval" >&2

# Append v40c to candidates and re-run greedy
.venv/bin/python - <<'PYEOF'
import os, sys
sys.path.insert(0, '/Users/danielemalerba/Downloads/birdclef-agent')
from eval_v40_full_ensemble import CANDIDATES, compute_val_predictions, greedy_forward, macro_auc
import json

CANDIDATES['v40c'] = '/Users/danielemalerba/Downloads/birdclef-agent/model_v40c_strong.keras'

Yv, preds = compute_val_predictions()
LEAKY = {"exp58_sc", "exp78_chkp_sc", "exp78_sc", "exp97_sc"}
clean_preds = {n: p for n, p in preds.items() if n not in LEAKY}
print('clean candidates:', list(clean_preds.keys()))
best_auc, final_w, history = greedy_forward(Yv, clean_preds)
print(f'BEST: {best_auc:.4f}')
for n, w in final_w.items():
    if w > 0:
        print(f'  {n} weight = {w:.3f}')
out = {'best_auc': best_auc, 'weights': {k: float(v) for k, v in final_w.items()}, 'history': history}
with open('/Users/danielemalerba/Downloads/birdclef-agent/eval_v40c_4way.json', 'w') as f:
    json.dump(out, f, indent=2, default=float)
PYEOF

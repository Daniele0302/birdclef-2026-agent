#!/bin/bash
cd /Users/danielemalerba/Downloads/birdclef-agent
PID=$(pgrep -f train_focal_member.py | head -1)
echo "[wait_eval_focal] watching PID=$PID"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "[wait_eval_focal] focal trained, running 7-way greedy"

.venv/bin/python - <<'PYEOF'
import os, sys, json
sys.path.insert(0, '/Users/danielemalerba/Downloads/birdclef-agent')
from eval_v40_full_ensemble import CANDIDATES, compute_val_predictions, greedy_forward, macro_auc

CANDIDATES['v40_s2'] = '/Users/danielemalerba/Downloads/birdclef-agent/model_v40_s2.keras'
CANDIDATES['v40_s3'] = '/Users/danielemalerba/Downloads/birdclef-agent/model_v40_s3.keras'
CANDIDATES['v40_s4'] = '/Users/danielemalerba/Downloads/birdclef-agent/model_v40_s4.keras'
CANDIDATES['v40_focal'] = '/Users/danielemalerba/Downloads/birdclef-agent/model_v40_focal.keras'

Yv, preds = compute_val_predictions()
LEAKY = {"exp58_sc", "exp78_chkp_sc", "exp78_sc", "exp97_sc"}
clean = {n: p for n, p in preds.items() if n not in LEAKY}
print('clean candidates:', list(clean.keys()))
best_auc, final_w, history = greedy_forward(Yv, clean)
print(f'BEST: {best_auc:.4f}')
for n, w in final_w.items():
    if w > 0:
        print(f'  {n} weight = {w:.3f}')
out = {'best_auc': float(best_auc), 'weights': {k: float(v) for k, v in final_w.items()}, 'history': history}
with open('/Users/danielemalerba/Downloads/birdclef-agent/eval_focal_7way.json', 'w') as f:
    json.dump(out, f, indent=2, default=float)
print('saved eval_focal_7way.json')
PYEOF

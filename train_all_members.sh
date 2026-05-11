#!/bin/bash
set -e
cd /Users/danielemalerba/Downloads/birdclef-agent

# Member A: same as v40 hyperparams, different seed
.venv/bin/python train_seed_member.py --seed 2 --out model_v40_s2.keras --phase1 3 --phase2 4 --mixup 0.2 --unfreeze 30 --boost 3.0

# Member B: lighter mixup, different unfreeze, different boost
.venv/bin/python train_seed_member.py --seed 3 --out model_v40_s3.keras --phase1 3 --phase2 4 --mixup 0.4 --unfreeze 40 --boost 5.0

# Member C: heavier mixup, lighter fine-tune
.venv/bin/python train_seed_member.py --seed 4 --out model_v40_s4.keras --phase1 3 --phase2 4 --mixup 0.5 --unfreeze 20 --boost 8.0

echo "ALL MEMBERS DONE"

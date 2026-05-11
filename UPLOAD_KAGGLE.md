# FINAL submission plan — push toward LB ~0.65

## 1. Files to upload to your Kaggle dataset `danielemalerba0302/birdclef2026-model`

Drag-and-drop new version with these 2 NEW files:

| File | Path on disk | Size |
|---|---|---|
| `model_v40b_strong.keras` | `/Users/danielemalerba/Downloads/birdclef-agent/model_v40b_strong.keras` | 22 MB |
| `model_v40_s2.keras` | `/Users/danielemalerba/Downloads/birdclef-agent/model_v40_s2.keras` | 33 MB |

(Keep the existing files — older notebooks may still reference them.)

## 2. Submit `birdclef_submission_v43_BEST.ipynb`

This is the strongest configuration we found:
- 2 models: `v40b` (50%) + `v40_s2` (50%)
- 5 temporal TTA, batch 64
- 10 forward passes per row -> ~50-70 min on Kaggle CPU (under 90 min)

Local soundscape macro-AUC = **0.6707**.

Reference points (gap between local and LB):
- exp78 alone:                local 0.6004 -> LB 0.592  (gap 0.012)
- 2-way exp78+v40 (your last): local ~0.625 -> LB 0.598  (gap 0.027)
- v43 v40b+v40_s2:           local **0.6707** -> projected LB **~0.65**

## 3. Submit on Kaggle

1. <https://www.kaggle.com/competitions/birdclef-2026/code>
2. New Notebook -> Add data: dataset `danielemalerba0302/birdclef2026-model` AND competition `birdclef-2026`
3. File -> Import notebook -> upload `birdclef_submission_v43_BEST.ipynb`
4. Settings: Python, accelerator None (CPU), internet OFF
5. Save Version -> Save & Run All (Commit)
6. Once green, Submit to Competition

## 4. Submission strategy for grading

You can have up to 2 final submissions count for grading. Pick:
- **Primary**: `v43_BEST.ipynb` (highest expected LB, ~0.65)
- **Backup**: keep your current Version 40 (LB 0.598) as the second selection

## 5. Honest expectations

Going beyond ~0.65-0.66 LB needs different architectures (ResNet, MobileNet) which are very slow on CPU, or pseudo-labelling on test (forbidden). The +0.05-0.06 jump from baseline 0.592 is substantial and well-explainable in the report.

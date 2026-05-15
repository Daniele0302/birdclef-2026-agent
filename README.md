# BirdCLEF+ 2026 — Autonomous Research Agent

An autonomous research agent that designs, runs, evaluates and iterates on deep-learning experiments for bird-species recognition — driven by a locally-hosted LLM under a CPU compute budget.

|  |  |
|---|---|
| **Competition** | [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) |
| **Task** | Probability of presence for each of 234 wildlife species in 5-second windows of Pantanal soundscape audio |
| **Metric** | Macro-averaged ROC-AUC |
| **Course** | Advanced Predictive Analytics 2025/2026 — Universidade Católica Portuguesa |
| **Reasoning core** | Google Gemma 4 E4B via Ollama (local, no network egress) |
| **Autonomous experiments logged** | 108 |
| **Best CPU-feasible focal val. macro-AUC** | 0.8533 |
| **Architecture verified at scale (Kaggle public LB)** | 0.725 |

The full write-up is in [`report.pdf`](report.pdf).

## What this repository is

An autonomous research agent, not a model-tuning project. The deliverable is the agent's *discovery process*: across 108 logged experiments it ruled out training a CNN from scratch after two trials, settled on a frozen-backbone EfficientNetB0, swept the mel-spectrogram parameters, and converged on a configuration reaching `val_AUC = 0.8533` on the focal validation set. The 0.725 Kaggle leaderboard score is reported as confirmation that the agent's architectural choice scales — not as the goal.

## How the agent works

A closed **propose → generate → execute → evaluate → analyse → iterate** loop, all under a CPU compute budget:

```
   ┌────────────────────────┐
   │  Local LLM             │
   │  (Gemma 4 E4B, Ollama) │
   └─────────┬──────────────┘
             │ propose JSON hyperparameters
             ▼
   ┌────────────────────────┐
   │  Write params_NNN.json │
   └─────────┬──────────────┘
             │
             ▼
   ┌────────────────────────┐
   │  Sandboxed subprocess  │   experiment_template.py
   │  (30-minute cap)       │
   └─────────┬──────────────┘
             │ metrics / stdout / stderr
             ▼
   ┌────────────────────────┐
   │  ExperimentMemory      │   appends to experiment_log.json
   └─────────┬──────────────┘
             │ summary of last 10 experiments
             ▼
   ┌────────────────────────┐
   │  LLM analysis          │   "what worked, what didn't, what next?"
   └─────────┬──────────────┘
             │
             └──── loop ──────────────────────────►
```

The audio pipeline (mel-spectrogram parameters, normalisation, label encoding) is **locked** inside `experiment_template.py` — the LLM only fills in a strict JSON parameter file. This keeps the data path human-written, human-reviewable, and resilient to small bugs the LLM would otherwise silently introduce.

## Pipeline reliability

Every iteration is scored across five stages. The agent reports both cumulative and conditional success rates, persisted to `agent_reliability_stats.json`:

```
================================================================
AGENT RELIABILITY REPORT (over 108 logged experiments)
================================================================
Stage                                Count      Rate     Cond.
----------------------------------------------------------------
S1 LLM produced text                   108     1.000     1.000
S2 valid JSON parsed                   106     0.981     0.981
S3 subprocess started                  106     0.981     1.000
S4 training completed                   81     0.750     0.764
S5 metrics recovered                    81     0.750     1.000
================================================================
```

The weak link is S4. The 25 S4 failures break down into 14 validation-split edge cases (a template robustness bug the agent's exploration surfaced), 10 environment failures, and just **1 genuine wall-clock timeout** — the agent self-regulates well within the per-experiment budget (median run time ≈ 4.5 minutes).

## Setup

Prerequisites: Python 3.10+, [Ollama](https://ollama.com), 16 GB RAM.

```bash
git clone https://github.com/Daniele0302/birdclef-2026-agent.git
cd birdclef-2026-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull gemma4:e4b
```

The BirdCLEF+ 2026 dataset must be downloaded from Kaggle and placed under `data/`:

```
data/train.csv
data/taxonomy.csv
data/train_audio/
data/train_soundscapes/
data/train_soundscapes_labels.csv
data/sample_submission.csv
```

## Running the agent

```bash
# Terminal 1
ollama serve

# Terminal 2
.venv/bin/python agent.py
```

`MAX_ITERATIONS = 10` is set in `config.py`. At the end of the run the agent prints the reliability report above and appends new entries to `experiments/experiment_log.json`.

## Project structure

```
birdclef-agent/
├── agent.py                          # main loop
├── llm_provider.py                   # local Ollama client
├── code_executor.py                  # sandboxed subprocess runner
├── memory.py                         # JSON memory + reliability metrics
├── experiment_template.py            # fixed Keras training script
├── config.py                         # central configuration
├── baseline_model.py                 # manual CNN-from-scratch baseline
│
├── utils/
│   ├── audio_pipeline.py             # audio → mel-spectrogram
│   └── data_loader.py                # dataset + label encoding
│
├── train_strong_v40{,b,c}.py         # scaling-up ensemble members
├── train_seed_member.py              # multi-seed training
├── train_focal_member.py             # focal-loss variant
├── train_kfold_cv.py                 # 5-fold CV ablation
├── train_gpu_v50.ipynb               # final GPU training (Kaggle T4)
│
├── eval_v40_local.py                 # local soundscape macro-AUC eval
├── eval_v40_3way.py                  # 3-way ensemble eval
├── eval_v40_full_ensemble.py         # greedy forward selection over models
├── analyze_agent.py                  # log analysis
├── analyze_cv.py
│
├── v34_exp78_3view_tta.ipynb                # submission 2 — agent baseline (LB 0.592)
├── birdclef_submission_v41_ensemble.ipynb   # submission 3 (LB 0.598)
├── birdclef_submission_v44_uniform3.ipynb   # submission 5 — CPU best (LB 0.633)
├── birdclef_submission_v55_v50only.ipynb    # submission 6 — at-scale check (LB 0.725)
│
├── experiments/
│   ├── experiment_log.json           # 108 logged experiments
│   ├── params_001…060.json           # per-iteration LLM-generated params
│   └── retest_*.json, final_*.json   # scaling-up params
│
├── agent_reliability_stats.json      # per-stage reliability output
├── eval_*.json                       # per-configuration eval results
├── *_log.json                        # scaling-up training logs
│
├── report.pdf                        # final write-up
├── requirements.txt
└── README.md
```

Pretrained ImageNet weights and trained `*.keras` models are not committed — they are produced by the training scripts and shipped via the Kaggle Dataset that the submission notebooks read from.

## Reproducing the result

1. Reproduce the autonomous exploration phase: `.venv/bin/python agent.py`. The agent identifies EfficientNetB0 + tuned mel-spectrogram as the best CPU-feasible configuration.
2. Train the scaling-up ensemble: `.venv/bin/python train_strong_v40.py`, then `train_seed_member.py --seed 2 …` for additional seeds.
3. Open `train_gpu_v50.ipynb` on Kaggle with a T4 GPU and run all cells (~1 h). The output `model_v50.keras` is the model that scored 0.725 on the public LB.
4. Upload `model_v50.keras` to a Kaggle Dataset and run `birdclef_submission_v55_v50only.ipynb` as a CPU notebook (~30 min) to produce `submission.csv`.

## Team

- **Daniele Malerba** — agent architecture, reliability instrumentation, Kaggle pipeline, scaling-up training
- **Irene Perdomo Bolaños** — audio preprocessing, augmentation, report writing
- **Miguel Afonso Magalhães** — LLM prompt engineering, log analysis

## License

This repository is shared with the course staff for grading. Competition data is governed by the BirdCLEF+ 2026 Kaggle terms; do not redistribute.

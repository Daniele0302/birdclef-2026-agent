# BirdCLEF+ 2026 — Autonomous Research Agent

An autonomous research agent that designs, trains, evaluates and iterates on deep-learning
models for bird-species recognition from audio recordings — driven by a locally-hosted
Large Language Model.

| | |
|---|---|
| **Competition** | [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) |
| **Task** | Identify the probability of presence for each of 234 wildlife species in 5-second audio windows from the Pantanal wetlands |
| **Metric** | Macro-averaged ROC-AUC (classes with no positives are skipped) |
| **Course** | Advanced Predictive Analytics 2025/2026 — Universidade Católica Portuguesa |
| **Best Kaggle public score** | **0.725** |
| **Total agent experiments** | 108 logged |
| **LLM** | Google Gemma 4 E4B via Ollama |

## What this repository contains

- A **fully autonomous research agent** (`agent.py`) that loops on Propose → Generate →
  Execute → Evaluate → Analyse → Iterate, driven by a local LLM.
- The **108 experiments** the agent ran during the cheap-exploration phase
  (`experiments/experiment_log.json`).
- The **scaling-up training scripts** that took the agent's best CPU configuration and
  scaled it to a deep ensemble (`train_strong_v40*.py`, `train_seed_member.py`, ...).
- The **GPU training notebook** (`train_gpu_v50.ipynb`) used for the final
  EfficientNetB0 trained on the full 35k recordings (Kaggle T4 GPU, ~1 h).
- The **CPU submission notebooks** used to submit on Kaggle within the 90-minute
  inference constraint (`birdclef_submission_v55_v50only.ipynb` is the best at LB 0.725).
- The **course report** (`report.tex`, compiles in Overleaf or with `pdflatex`).
- A **manual baseline** (`baseline_model.py`) for comparison.

## Prerequisites

- Python 3.10+ (developed with 3.13)
- [Ollama](https://ollama.com) installed and running locally
- 16 GB RAM recommended
- For optional GPU training: a Kaggle account with Notebooks enabled (free T4 GPU)

## Setup

```bash
# Clone
git clone https://github.com/Daniele0302/birdclef-2026-agent.git
cd birdclef-2026-agent

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Local LLM
ollama pull gemma4:e4b      # the model we standardised on

# Competition data
# Download the BirdCLEF+ 2026 dataset from Kaggle and place it in data/
# After unzipping you should have:
#   data/train.csv
#   data/taxonomy.csv
#   data/train_audio/
#   data/train_soundscapes/
#   data/train_soundscapes_labels.csv
#   data/sample_submission.csv
```

## Running the agent

In one terminal, start the local LLM server:

```bash
ollama serve
```

In a second terminal, run the agent:

```bash
.venv/bin/python agent.py
```

The agent runs for `MAX_ITERATIONS = 10` (see `config.py`) without human input. At each
iteration it asks the LLM for a JSON parameter set, fills `experiment_template.py` with
it, runs that template as a sandboxed subprocess with a 30-minute wall-clock cap,
captures the validation macro-AUC, and feeds the results back to the LLM for analysis.
At the end of the run it prints a per-stage **reliability report** built from the
cumulative log of all experiments seen so far:

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

The full per-experiment log lives in `experiments/experiment_log.json` and is
appended-to across runs.

## Project structure

```
birdclef-agent/
├── agent.py                        # main agent loop
├── llm_provider.py                 # Ollama / OpenAI-compatible LLM client
├── code_executor.py                # sandboxed subprocess runner
├── memory.py                       # JSON memory + reliability metrics
├── experiment_template.py          # parameterised Keras training script
├── config.py                       # central configuration (paths, LLM, hyperparams)
├── baseline_model.py               # manual CNN-from-scratch baseline
├── utils/
│   ├── audio_pipeline.py           # audio → mel-spectrogram pipeline
│   └── data_loader.py              # label encoding + dataset assembly
│
├── train_strong_v40.py             # scaling-up: 8k focal + soundscape + mixup
├── train_strong_v40b.py            # variant: stronger soundscape boost
├── train_strong_v40c.py            # variant: multi-window focal sampling
├── train_seed_member.py            # multi-seed deep-ensemble members
├── train_focal_member.py           # focal-loss variant
├── train_kfold_cv.py               # 5-fold CV training (used as ablation)
├── train_gpu_v50.ipynb             # final GPU training (Kaggle Notebook)
│
├── eval_v40_local.py               # local soundscape macro-AUC eval
├── eval_v40_3way.py                # 3-way ensemble grid search
├── eval_v40_full_ensemble.py       # greedy forward selection over all models
│
├── birdclef_submission_v55_v50only.ipynb     # best submission notebook (LB 0.725)
├── birdclef_submission_v44_uniform3.ipynb    # CPU-only fallback (LB 0.633)
├── birdclef_submission_v41_ensemble.ipynb    # CPU ensemble (LB 0.615)
├── v34_exp78_3view_tta.ipynb                 # initial agent submission (LB 0.592)
│
├── experiments/
│   ├── experiment_log.json         # 108 logged experiments (cheap exploration)
│   └── params_NNN.json             # per-iteration LLM-generated parameter files
│
├── report.tex                      # 10-page course report (LaTeX)
├── requirements.txt
└── README.md
```

Model files (`*.keras`, ~20 MB each) and the precomputed `cache/` directory are
**not** committed to git because of file-size limits; they are produced by running the
training scripts above and uploaded separately to the Kaggle Dataset that the
submission notebooks read from.

## How the agent works

The agent follows a closed **propose–generate–execute–evaluate–analyse–iterate** loop:

```
   ┌────────────────────────┐
   │  Local LLM             │
   │  (Gemma 4 E4B)         │
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
   │  (timeout 30 min)      │
   └─────────┬──────────────┘
             │ stdout / stderr / metrics
             ▼
   ┌────────────────────────┐
   │  ExperimentMemory      │   appends to experiment_log.json
   └─────────┬──────────────┘
             │ summary of last 10 experiments
             ▼
   ┌────────────────────────┐
   │  LLM analysis prompt   │   "what worked, what didn't, what next?"
   └─────────┬──────────────┘
             │
             └────── loop ────────────────────────►
```

The audio pipeline (mel-spectrogram parameters, normalisation, label encoding) is
locked inside `experiment_template.py` — the LLM only fills in a strict JSON parameter
file. This prevents the LLM from breaking the data path through small bugs that would
silently destroy performance.

## Reproducing the leaderboard score

1. Run the agent locally to reproduce the 108-experiment exploration phase
   (`.venv/bin/python agent.py`) — this identifies EfficientNetB0 with our mel
   parameters as the best configuration.
2. Train one or more of the scaling-up scripts on CPU:
   `.venv/bin/python train_strong_v40.py`,
   `train_seed_member.py --seed 2 --out model_v40_s2.keras`, etc.
3. Open `train_gpu_v50.ipynb` on Kaggle with a T4 GPU and run all cells (~1 h). The
   output `model_v50.keras` is the model that scored 0.725 on the public LB.
4. Upload `model_v50.keras` to a Kaggle Dataset and run
   `birdclef_submission_v55_v50only.ipynb` as a CPU notebook (~30 min) to produce
   `submission.csv`. Submit it.

## Team members

- Daniele Malerba — agent architecture, scaling-up training, Kaggle pipeline
- Irene Perdomo Bolaños — audio preprocessing, augmentation, report
- Miguel Afonso Magalhães — LLM prompt engineering, log analysis

## License

This repository is shared with the course staff for grading purposes. Competition data
is governed by the BirdCLEF+ 2026 Kaggle terms; do not redistribute.

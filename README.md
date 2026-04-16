# BirdCLEF+ 2026 — Autonomous Research Agent

An AI-powered autonomous research agent that designs, trains, evaluates, and iterates on deep learning models for bird species recognition from audio recordings.

**Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)  
**Task:** Identify 234 wildlife species from audio recordings in the Pantanal wetlands  
**Metric:** Macro-averaged ROC-AUC  
**Course:** Advanced Predictive Analytics 2025/2026

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- 16+ GB RAM recommended

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_TEAM/birdclef-2026-agent.git
cd birdclef-2026-agent

# Install dependencies
pip install -r requirements.txt

# Download and install a local LLM
ollama pull gemma2:9b

# Download the dataset from Kaggle
# Place it in the data/ directory:
#   data/train.csv
#   data/taxonomy.csv
#   data/train_audio/
#   data/train_soundscapes/
#   data/sample_submission.csv
```

### 3. Run the Agent

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, start the agent
python agent.py
```

The agent runs autonomously — no human input needed after launch.

## Project Structure

```
birdclef-agent/
├── agent.py                 # Main agent loop
├── llm_provider.py          # LLM communication (Ollama)
├── code_executor.py         # Sandboxed code execution
├── memory.py                # Experiment logging & memory
├── prompt_templates.py      # Prompt engineering for the LLM
├── config.py                # All configurable parameters
├── baseline_model.py        # Manual baseline for comparison
├── requirements.txt         # Python dependencies
├── utils/
│   ├── audio_pipeline.py    # Audio → mel-spectrogram conversion
│   └── data_loader.py       # Dataset loading and preparation
├── experiments/             # Auto-generated experiment logs
└── data/                    # Dataset (not in repo)
```

## How the Agent Works

1. **Propose**: The LLM proposes a model architecture and hyperparameters
2. **Generate**: The LLM writes executable Python training code
3. **Execute**: The code runs in a sandboxed subprocess with timeout
4. **Evaluate**: Training metrics (AUC, loss) are captured
5. **Analyze**: The LLM reviews results and decides what to try next
6. **Iterate**: The loop repeats with accumulated knowledge

## Team Members

- Member 1 — Role
- Member 2 — Role
- Member 3 — Role
- Member 4 — Role

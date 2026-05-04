"""
agent.py — Autonomous BirdCLEF 2026 agent (v3 - multi-architecture)

What's new v3:
- The LLM can choose between a custom CNN and EfficientNet (transfer learning)
- Augmentation types: noise, time_shift, freq_mask, all
- Fine-tuning: unfreeze_layers for EfficientNet
"""

import json
import os
import subprocess
import time
from llm_provider import call_llm
from memory import ExperimentMemory

EXPERIMENTS_DIR = "experiments_fast"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an autonomous ML research agent for BirdCLEF 2026 (Track B).
FAST SEARCH MODE: use small datasets to quickly find best preprocessing.

You must respond with ONLY a valid JSON object. No text, no explanation, no markdown.

The JSON must have ALL of these keys:
{
    "experiment_name": "short_descriptive_name",
    "model_type": "efficientnet",
    "learning_rate": 0.0003,
    "batch_size": 16,
    "epochs": 8,
    "n_filters_1": 0,
    "n_filters_2": 0,
    "n_filters_3": 0,
    "dropout_rate": 0.4,
    "dense_units": 256,
    "n_mels": 64,
    "n_fft": 2048,
    "hop_length": 256,
    "fmin": 20,
    "fmax": 14000,
    "top_db": 60.0,
    "mel_norm": "slaney",
    "mel_scale": "htk",
    "max_samples": 2000,
    "use_augmentation": true,
    "augmentation_type": "time_shift",
    "augmentation_noise": 0.01,
    "unfreeze_layers": 0
}

FIXED PARAMETERS — NEVER CHANGE THESE:
- model_type: ALWAYS "efficientnet"
- max_samples: ALWAYS 2000
- epochs: ALWAYS 8
- batch_size: ALWAYS 16
- learning_rate: ALWAYS 0.0003
- dropout_rate: ALWAYS 0.4
- dense_units: ALWAYS 256
- unfreeze_layers: ALWAYS 0
- use_augmentation: ALWAYS true
- augmentation_type: ALWAYS "time_shift"

BEST CONFIG SO FAR: val_auc=0.8372
n_mels=64, hop=256, fmin=20, fmax=14000, top_db=60, mel_norm=slaney, mel_scale=htk

WHAT TO EXPLORE — vary ONLY 1-2 parameters at a time:
- n_mels: try 32, 48, 64, 96, 128
- hop_length: try 128, 160, 256, 320, 512
- n_fft: try 512, 1024, 2048, 4096
- fmin: try 0, 20, 50, 100
- fmax: try 8000, 10000, 12000, 14000, 16000
- top_db: try 40, 50, 60, 70, 80, 100
- mel_norm: try null, "slaney"
- mel_scale: try "htk", "slaney"
- augmentation_noise: try 0.001, 0.005, 0.01, 0.02

STRATEGY:
- Change ONLY 1-2 parameters per experiment
- Never repeat the same combination
- Goal: find preprocessing that generalizes to real soundscapes

Respond with ONLY the JSON.""" 


EXPERIMENTS_DIR = "experiments"

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

def build_prompt(memory_summary):
    return f"""{SYSTEM_PROMPT}

Previous experiments and results:
{memory_summary}

Based on the results, propose the next experiment. Respond with ONLY a JSON object."""


def parse_json_from_llm(response):
    text = response.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def run_experiment_from_params(params, config_path):
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=2)

    try:
        result = subprocess.run(
            ['python3', 'experiment_template.py', '--config', config_path],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min per EfficientNet con fine-tuning a due fasi
        )

        metrics = None
        for line in reversed(result.stdout.strip().split('\n')):
            try:
                metrics = json.loads(line.strip())
                if 'val_auc' in metrics:
                    break
            except:
                continue

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "metrics": metrics
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "TIMEOUT: experiment exceeded 1800 seconds",
            "metrics": None
        }


def main():
    memory = ExperimentMemory()

    N_ITERATIONS = 60

    print("=" * 60)
    print("AUTONOMOUS AGENT — BirdCLEF 2026 (v3 multi-architecture)")
    print(f"Planned iterations: {N_ITERATIONS}")
    print(f"Available architectures: custom CNN, EfficientNetB0")
    print("=" * 60)

    for iteration in range(1, N_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{N_ITERATIONS}")
        print(f"{'='*60}")

        # Step 1: Request parameters
        print("\n[1/4] Asking the LLM to propose parameters...")
        memory_summary = memory.summarize_recent(n=15)
        prompt = build_prompt(memory_summary)
        raw_response = call_llm(prompt)
        print(f"LLM response: {raw_response[:500]}")

        # Step 2: Parse JSON
        print("\n[2/4] Parsing parameters...")
        params = parse_json_from_llm(raw_response)

        if params is None:
            print("ERROR: Invalid JSON. Skipping iteration.")
            memory.add_experiment(
                prompt=prompt,
                code=raw_response,
                result={"success": False, "stdout": "", "stderr": "Invalid JSON", "metrics": None},
                analysis="JSON parsing failed."
            )
            continue

        print(f"Model: {params.get('model_type', 'cnn')}")
        print(f"Parameters: {json.dumps(params, indent=2)}")

        # Step 3: Run
        config_path = os.path.join(EXPERIMENTS_DIR, f"params_{iteration:03d}.json")
        print(f"\n[3/4] Running '{params.get('experiment_name', 'unknown')}'...")
        start = time.time()
        result = run_experiment_from_params(params, config_path)
        elapsed = time.time() - start

        print(f"Completed in {elapsed:.0f}s | Success: {result['success']}")
        if result['metrics']:
            print(f"Metrics: {json.dumps(result['metrics'])}")
        elif result['stderr']:
            print(f"Error: {result['stderr'][:500]}")

        # Step 4: Analysis
        print("\n[4/4] LLM analysis...")
        metrics_str = json.dumps(result['metrics']) if result['metrics'] else "No metrics"
        analysis_prompt = f"""Analyze this BirdCLEF experiment concisely.

Parameters: {json.dumps(params)}
Success: {result['success']}
Metrics: {metrics_str}
Best AUC so far: {memory.best_auc}

In 3-5 sentences: what worked, what didn't, what to try next.
Focus on whether to use CNN or EfficientNet and why."""

        analysis = call_llm(analysis_prompt)
        print(f"Analysis: {analysis[:500]}")

        memory.add_experiment(
            prompt=prompt,
            code=json.dumps(params),
            result=result,
            analysis=analysis
        )

        print(f"\nBest AUC so far: {memory.best_auc}")

    print(f"\n{'='*60}")
    print(f"AGENT COMPLETED — {N_ITERATIONS} experiments")
    print(f"Best AUC: {memory.best_auc}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

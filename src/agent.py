"""
agent.py — Autonomous BirdCLEF 2026 agent (v3 - multi-architecture)

Features:
- The LLM can choose between a custom CNN and EfficientNet (transfer learning)
- Augmentation types: noise, time_shift, freq_mask, all
- Fine-tuning: unfreeze_layers for EfficientNet
"""

import json
import os
import subprocess
import sys
import time
from llm_provider import call_llm
from memory import ExperimentMemory

# Resolve key paths relative to the repo root, so this script works
# regardless of the user's cwd. agent.py lives in src/.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
TEMPLATE_PATH = os.path.join(SRC_DIR, "experiment_template.py")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an autonomous ML research agent for BirdCLEF 2026 (Track B).

Your job is to propose experiment configurations for classifying mel-spectrograms
of wildlife audio (234 species, multi-label, sigmoid output, binary crossentropy loss).

You must respond with ONLY a valid JSON object. No text, no explanation, no markdown.

The JSON must have ALL of these keys:
{
    "experiment_name": "short_descriptive_name",
    "model_type": "efficientnet",
    "learning_rate": 0.0003,
    "batch_size": 16,
    "epochs": 10,
    "n_filters_1": 0,
    "n_filters_2": 0,
    "n_filters_3": 0,
    "dropout_rate": 0.4,
    "dense_units": 256,
    "n_mels": 64,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 20,
    "fmax": 14000,
    "top_db": 80.0,
    "mel_norm": null,
    "mel_scale": "htk",
    "max_samples": 3000,
    "use_augmentation": true,
    "augmentation_type": "time_shift",
    "augmentation_noise": 0.01,
    "unfreeze_layers": 0
}

WHAT WE ALREADY KNOW — DO NOT CHANGE THESE:
- model_type: ALWAYS "efficientnet"
- max_samples: ALWAYS 3000
- batch_size: ALWAYS 16
- learning_rate: ALWAYS 0.0003
- dropout_rate: ALWAYS 0.4
- dense_units: ALWAYS 256
- unfreeze_layers: ALWAYS 0
- use_augmentation: ALWAYS true
- augmentation_type: ALWAYS "time_shift"
- n_mels: ALWAYS 64 — best found so far
- fmin: ALWAYS 20 — best found so far

BEST CONFIG SO FAR: val_auc=0.826
n_mels=64, fmin=20, fmax=14000, n_fft=2048, hop_length=512, top_db=80, noise=0.01

WHAT TO EXPLORE — vary ONLY these parameters one or two at a time:
- hop_length: try 160, 256, 320, 512
- n_fft: try 512, 1024, 2048, 4096
- fmax: try 10000, 12000, 14000, 16000
- top_db: try 60.0, 80.0, 100.0
- augmentation_noise: try 0.002, 0.005, 0.01, 0.02
- epochs: try 12, 15 if models converge too fast
- mel_norm: try null, "slaney"
- mel_scale: try "htk", "slaney"


STRATEGY:
- Change only 1-2 parameters per experiment
- Never repeat the same combination
- Focus especially on hop_length and top_db — not yet explored
- Goal: beat val_auc=0.826

Respond with ONLY the JSON."""

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
            [sys.executable, TEMPLATE_PATH, '--config', config_path],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min per EfficientNet con fine-tuning a due fasi
            cwd=PROJECT_ROOT  # so the template resolves data/ and cache/ paths against the root
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

    N_ITERATIONS = 10

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
    print(f"{'='*60}\n")

   
    memory.print_reliability_report()


if __name__ == "__main__":
    main()

"""
memory.py — Gestione memoria esperimenti per l'agente BirdCLEF 2026

Salva ogni esperimento in un log JSON strutturato.
Fornisce un riassunto testuale da inserire nei prompt per l'LLM.
Tiene traccia del miglior AUC raggiunto.
"""

import json
import os
from datetime import datetime

EXPERIMENTS_DIR = "experiments"
LOG_FILE = os.path.join(EXPERIMENTS_DIR, "experiment_log.json")


class ExperimentMemory:
    def __init__(self):
        """
        Carica gli esperimenti precedenti dal file di log, se esiste.
        Altrimenti parte da una lista vuota.
        """
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
        self.experiments = []
        self.best_auc = 0.0
        self.best_experiment_id = None

        # Carica log esistente
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r') as f:
                    self.experiments = json.load(f)
                # Ricalcola best AUC
                for exp in self.experiments:
                    auc = exp.get("metrics", {}).get("val_auc", 0) if exp.get("metrics") else 0
                    if auc > self.best_auc:
                        self.best_auc = auc
                        self.best_experiment_id = exp.get("id")
                print(f"Memoria caricata: {len(self.experiments)} esperimenti, best AUC={self.best_auc}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"Errore nel caricamento della memoria: {e}")
                self.experiments = []

    def add_experiment(self, prompt, code, result, analysis):
        """
        Aggiunge un nuovo esperimento alla memoria e salva su disco.

        Args:
            prompt: il prompt inviato all'LLM
            code: il codice/parametri generati dall'LLM
            result: dict con success, stdout, stderr, metrics
            analysis: l'analisi dell'LLM sui risultati
        """
        exp_id = len(self.experiments) + 1

        # Estrai metriche
        metrics = result.get("metrics", None)
        val_auc = metrics.get("val_auc", 0) if metrics else 0

        # Aggiorna best
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_experiment_id = exp_id
            print(f"*** NUOVO RECORD! AUC={val_auc} (esperimento #{exp_id}) ***")

        entry = {
            "id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "params_or_code": code[:2000] if isinstance(code, str) else str(code)[:2000],
            "success": result.get("success", False),
            "metrics": metrics,
            "analysis": analysis[:1000] if isinstance(analysis, str) else str(analysis)[:1000],
            "stderr_snippet": result.get("stderr", "")[:500]
        }

        self.experiments.append(entry)
        self._save()

    def _save(self):
        """Salva il log su disco in formato JSON."""
        try:
            with open(LOG_FILE, 'w') as f:
                json.dump(self.experiments, f, indent=2)
        except Exception as e:
            print(f"Errore nel salvataggio della memoria: {e}")

    def summarize_recent(self, n=10):
        """
        Produce un riassunto testuale degli ultimi N esperimenti.
        Questo testo viene inserito nel prompt per l'LLM.

        Esempio di output:
            Experiment #1: lr=0.001, filters=32/64/128, dropout=0.3
              → val_auc=0.6326, val_loss=0.0344 (SUCCESS)
            Experiment #2: lr=0.0005, filters=64/128/256, dropout=0.4
              → val_auc=0.6812, val_loss=0.0298 (SUCCESS) ← BEST
        """
        if not self.experiments:
            return "No previous experiments. This is the first run. Start with a reasonable baseline variation."

        recent = self.experiments[-n:]
        lines = []

        for exp in recent:
            exp_id = exp.get("id", "?")
            success = exp.get("success", False)
            metrics = exp.get("metrics", {})

            # Prova a parsare i parametri se sono JSON
            params_str = ""
            try:
                params = json.loads(exp.get("params_or_code", "{}"))
                if isinstance(params, dict) and "learning_rate" in params:
                    params_str = (
                        f"lr={params.get('learning_rate')}, "
                        f"filters={params.get('n_filters_1')}/{params.get('n_filters_2')}/{params.get('n_filters_3')}, "
                        f"dropout={params.get('dropout_rate')}, "
                        f"dense={params.get('dense_units')}, "
                        f"batch={params.get('batch_size')}, "
                        f"aug={'ON' if params.get('use_augmentation') else 'OFF'}"
                    )
            except:
                params_str = "custom experiment"

            if success and metrics:
                val_auc = metrics.get('val_auc', 0)
                val_loss = metrics.get('val_loss', 0)
                is_best = " ← BEST" if val_auc == self.best_auc else ""
                lines.append(
                    f"Experiment #{exp_id}: {params_str}\n"
                    f"  → val_auc={val_auc}, val_loss={val_loss} (SUCCESS){is_best}"
                )
            else:
                error = exp.get("stderr_snippet", "unknown error")[:100]
                lines.append(
                    f"Experiment #{exp_id}: {params_str}\n"
                    f"  → FAILED: {error}"
                )

        summary = "\n".join(lines)
        summary += f"\n\nCurrent best: val_auc={self.best_auc} (Experiment #{self.best_experiment_id})"

        return summary

"""
llm_provider.py — Comunicazione con l'LLM locale via Ollama

Usa la libreria ollama per comunicare con il modello.
Il modello è configurabile cambiando MODEL_NAME.
"""

import ollama

# Cambia questo se vuoi usare un modello diverso
# Modelli testati: gemma4:e4b, qwen3.5:9b, gemma2:9b
MODEL_NAME = "gemma4:e4b"


def clean_code(text):
    """
    Pulisce la risposta dell'LLM rimuovendo markdown fences e testo extra.

    L'LLM a volte risponde con:
        ```python
        codice qui
        ```
    Questa funzione estrae solo il codice/JSON pulito.
    """
    text = text.strip()

    # Se contiene blocchi markdown, estrai il contenuto
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Rimuovi il tag del linguaggio (python, json, ecc.)
            if part.startswith("python"):
                return part[len("python"):].strip()
            if part.startswith("json"):
                return part[len("json"):].strip()
            # Se contiene import o parentesi graffe, probabilmente è codice/JSON
            if "import " in part or part.startswith("{"):
                return part.strip()

    return text.strip()


def call_llm(prompt):
    """
    Manda un prompt all'LLM locale e restituisce la risposta pulita.

    Args:
        prompt: stringa con il prompt completo

    Returns:
        Risposta dell'LLM come stringa (pulita da markdown)
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise ML research assistant. "
                        "When asked for JSON, return ONLY valid JSON. "
                        "When asked for analysis, be concise and practical. "
                        "Never use markdown code fences."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        raw = response["message"]["content"]
        return clean_code(raw)

    except Exception as e:
        return f"ERRORE LLM: {str(e)}"

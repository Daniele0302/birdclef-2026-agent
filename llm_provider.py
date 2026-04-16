"""
llm_provider.py — Communication with local LLM via Ollama

Uses the `ollama` library to communicate with the model.
The model can be changed by updating MODEL_NAME.
"""

import ollama

# Change this if you want to use a different model
# Tested models: gemma4:e4b, qwen3.5:9b, gemma2:9b
MODEL_NAME = "gemma4:e4b"


def clean_code(text):
    """
    Cleans LLM responses by removing markdown fences and extra text.

    The LLM sometimes replies with:
        ```python
        code here
        ```
    This function extracts only the clean code/JSON.
    """
    text = text.strip()

    # If it contains markdown fences, extract the inner content
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Remove the language tag (python, json, etc.)
            if part.startswith("python"):
                return part[len("python"):].strip()
            if part.startswith("json"):
                return part[len("json"):].strip()
            # If it contains 'import' or starts with '{', it's likely code/JSON
            if "import " in part or part.startswith("{"):
                return part.strip()

    return text.strip()


def call_llm(prompt):
    """
    Sends a prompt to the local LLM and returns the cleaned response.

    Args:
        prompt: the full prompt string

    Returns:
        The LLM response as a string (cleaned from markdown)
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
        return f"LLM ERROR: {str(e)}"

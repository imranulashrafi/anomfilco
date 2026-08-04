import json

from filco.logging.logger import log

SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."


def load_problems(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of objects with a 'prompt' field")
    rows = [
        x for x in data
        if x.get("prompt")
    ]
    if not rows:
        raise ValueError(f"No valid rows in {path}")
    log(f"Loaded {len(rows)} problems from {path}")
    return rows


def _format_prompt_instruct(problem: dict, tokenizer, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem["prompt"]},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            s = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            bos = getattr(tokenizer, "bos_token", None)
            if bos and s.startswith(bos):
                s = s[len(bos):]
            return s
        except Exception:
            pass
    return f"System: {system_prompt}\n\nUser: {problem['prompt']}\n\nAssistant:"


def _format_prompt_base(problem: dict, tokenizer, system_prompt: str) -> str:
    return f"{system_prompt}\n\n{problem['prompt']}\n\n"


def format_prompt(problem: dict, tokenizer, model_type: str = "instruct") -> str:
    if model_type == "instruct":
        return _format_prompt_instruct(problem, tokenizer, SYSTEM_PROMPT)
    return _format_prompt_base(problem, tokenizer, SYSTEM_PROMPT)

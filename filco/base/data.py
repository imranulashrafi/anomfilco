import json

from filco.logging.logger import log

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
OPEN_ENDED_SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."


def load_problems(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list with keys: prompt, answer")
    rows = [
        x for x in data
        if x.get("prompt") and (x.get("answer") is not None or x.get("chosen"))
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


def format_prompt(
    problem: dict, tokenizer, model_type: str = "instruct", task_type: str = "math"
) -> str:
    system_prompt = OPEN_ENDED_SYSTEM_PROMPT if task_type == "open" else SYSTEM_PROMPT
    if model_type == "instruct":
        return _format_prompt_instruct(problem, tokenizer, system_prompt)
    return _format_prompt_base(problem, tokenizer, system_prompt)

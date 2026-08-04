import datetime
import json
import os
import pickle
import shutil

from filco.utils.distributed import IS_RANK0
from filco.logging.logger import log


def save_completions(episode: int, problems: list, data: list, output_dir: str):
    if not IS_RANK0:
        return
    path = os.path.join(output_dir, f"episode_{episode}_completions.pkl")
    with open(path, "wb") as f:
        pickle.dump({"episode": episode, "problems": problems, "data": data}, f)
    log(f"Completions saved → {path}")


def load_completions(episode: int, output_dir: str) -> dict:
    path = os.path.join(output_dir, f"episode_{episode}_completions.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _cleanup_old_checkpoints(output_dir: str, current_episode: int, keep_last_n: int):
    if keep_last_n <= 0:
        return
    for ep in range(1, current_episode - keep_last_n + 1):
        old = os.path.join(output_dir, f"checkpoint_ep{ep}")
        if os.path.isdir(old):
            shutil.rmtree(old)
            log(f"Deleted old checkpoint → {old}")


def save_run_state(output_dir: str, state: dict):
    if not IS_RANK0:
        return
    state["saved_at"] = datetime.datetime.now().isoformat()
    path = os.path.join(output_dir, "run_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_run_state(output_dir: str) -> dict:
    path = os.path.join(output_dir, "run_state.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

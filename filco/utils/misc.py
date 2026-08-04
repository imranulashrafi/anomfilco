import datetime
import os
import random

import numpy as np
import torch

from filco.utils.distributed import LOCAL_RANK


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_mem(device: int = None) -> str:
    if device is None:
        device = LOCAL_RANK
    cur = torch.cuda.memory_allocated(device) / 1e9
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    return f"{cur:.1f}GB/{peak:.1f}GB"


def _make_output_dir(model_name: str, data_path: str, method: str) -> str:
    model_short = model_name.split("/")[-1]
    dataset_name = os.path.basename(os.path.dirname(os.path.abspath(data_path)))
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    _filco_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(_filco_root, "runs", f"{model_short}_{dataset_name}_{method}_{date_str}")


def detect_model_type(model_name: str) -> str:
    name_lower = model_name.lower()
    if any(tag in name_lower for tag in ("instruct", "chat", "rlhf", "dpo", "it")):
        return "instruct"
    return "base"


def _eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    remaining = (elapsed / done) * (total - done)
    h, m = divmod(int(remaining), 3600)
    m //= 60
    return f"{h}h{m:02d}m"

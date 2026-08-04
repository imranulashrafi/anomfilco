import datetime
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import bitsandbytes  # noqa: F401
    _BNB_AVAILABLE = True
except (ImportError, OSError):
    _BNB_AVAILABLE = False

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_RANK0 = LOCAL_RANK == 0


def init_distributed():
    if WORLD_SIZE > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
        torch.cuda.set_device(LOCAL_RANK)


def cleanup_distributed():
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.destroy_process_group()


def get_raw_model(model):
    if isinstance(model, DDP):
        return model.module
    return model

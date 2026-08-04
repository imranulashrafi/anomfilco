import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from filco.utils.distributed import (
    IS_RANK0, LOCAL_RANK, WORLD_SIZE,
    _BNB_AVAILABLE,
    get_raw_model,
)
from filco.logging.logger import log

try:
    from bitsandbytes.optim import AdamW8bit
except (ImportError, OSError):
    pass

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    _PEFT_AVAILABLE = True
    try:
        import peft.import_utils as _piu
        _piu.is_torchao_available = lambda *a, **k: False
        import peft.tuners.lora.torchao as _plt
        _plt.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
except ImportError:
    _PEFT_AVAILABLE = False

from torch.nn.parallel import DistributedDataParallel as DDP
from filco.utils.misc import gpu_mem


def load_model_and_tokenizer(cfg, model_path: str = None):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    load_from = model_path or cfg.model_name

    log(f"[Rank {LOCAL_RANK}] Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=getattr(cfg, "trust_remote_code", True), padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"[Rank {LOCAL_RANK}] Loading model ({dtype}) from {load_from}")
    model = AutoModelForCausalLM.from_pretrained(
        load_from, trust_remote_code=getattr(cfg, "trust_remote_code", True), torch_dtype=dtype
    ).to(device)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if getattr(cfg, "use_lora", False):
        if not _PEFT_AVAILABLE:
            raise ImportError("use_lora=True but peft is not installed (pip install peft)")
        targets = [t.strip() for t in getattr(cfg, "lora_target_modules", "").split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=targets or None, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()
        if IS_RANK0:
            model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[Rank {LOCAL_RANK}] Trainable: {trainable/1e6:.1f}M  GPU mem: {gpu_mem()}")
    return model, tokenizer


def wrap_model_for_training(model, cfg):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    if WORLD_SIZE <= 1:
        return model.to(device)
    model = model.to(device)
    model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)
    log(f"[Rank {LOCAL_RANK}] Model wrapped in DDP")
    return model


def create_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.use_8bit_adam:
        if not _BNB_AVAILABLE:
            log("WARNING: bitsandbytes not installed; falling back to AdamW")
            return torch.optim.AdamW(params, lr=cfg.learning_rate)
        log("Using 8-bit AdamW (bitsandbytes)")
        return AdamW8bit(params, lr=cfg.learning_rate)
    return torch.optim.AdamW(params, lr=cfg.learning_rate)


def save_model_checkpoint(model, tokenizer, path: str, cfg):
    os.makedirs(path, exist_ok=True)
    raw = get_raw_model(model)
    if _PEFT_AVAILABLE and isinstance(raw, PeftModel):
        raw.merge_adapter()
        try:
            if IS_RANK0:
                base = raw.get_base_model()
                clean = {}
                for k, v in base.state_dict().items():
                    if ".lora_" in k or "lora_embedding" in k:
                        continue
                    clean[k.replace(".base_layer.", ".")] = v
                base.save_pretrained(path, state_dict=clean, safe_serialization=True)
                if tokenizer is not None:
                    tokenizer.save_pretrained(path)
                log(f"LoRA merged checkpoint saved → {path}")
        finally:
            raw.unmerge_adapter()
    else:
        if IS_RANK0:
            raw.save_pretrained(path, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            log(f"Checkpoint saved → {path}")

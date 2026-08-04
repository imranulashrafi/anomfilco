import os
from dataclasses import dataclass, field
from typing import Optional

from filco.utils.misc import detect_model_type


@dataclass
class BaseConfig:
    method: str = ""

    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    model_type: str = ""

    data_path: str = "data/MATH-TTT/train.json"

    n_vote: int = 32
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 3072

    num_episodes: int = 20
    learning_rate: float = 5e-7
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8

    train_subset_size: Optional[int] = None
    eval_subset_size: Optional[int] = 500
    keep_last_n_checkpoints: int = 1
    eval_max_new_tokens: int = 3072
    eval_start_episode: int = 0

    dry_run_episodes: int = 0
    resume_dir: str = ""

    output_dir: str = ""
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    trust_remote_code: bool = True

    verbose_samples: bool = False

    sglang_devices: str = "0,1"
    sglang_tp: int = 0
    sglang_num_servers: int = 1
    sglang_host: str = "127.0.0.1"
    sglang_port: int = 30000
    sglang_timeout_s: int = 300
    sglang_context_length: int = 0
    sglang_mem_fraction: float = 0.80

    seed: int = 42


    task_type: str = "math"

    sglang_ports: list = field(default_factory=list)
    sglang_device_groups: list = field(default_factory=list)

    def __post_init__(self):
        if not self.model_type:
            self.model_type = detect_model_type(self.model_name)
        devices = self.sglang_devices.split(",")
        n_servers = self.sglang_num_servers
        if len(devices) % n_servers != 0:
            raise ValueError(
                f"sglang_devices has {len(devices)} GPUs, not divisible by "
                f"sglang_num_servers={n_servers}"
            )
        tp_per_server = len(devices) // n_servers
        if self.sglang_tp == 0:
            self.sglang_tp = tp_per_server
        self.sglang_device_groups = [
            ",".join(devices[i * tp_per_server: (i + 1) * tp_per_server])
            for i in range(n_servers)
        ]
        self.sglang_ports = [self.sglang_port + i for i in range(n_servers)]


@dataclass
class FilcoConfig(BaseConfig):
    method: str = "filco"
    task_type: str = "open"

    n_vote: int = 32
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 4096
    eval_max_new_tokens: int = 4096

    num_episodes: int = 15
    learning_rate: float = 8e-7
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 8

    simpo_beta: float = 2.0
    simpo_gamma: float = 0.5
    max_train_completion_tokens: int = 3072

    gate_pct: float = 95.0
    suffix_words: int = 30
    prefix_words: int = 30
    rejected_strategy: str = "lowest_centrality_all"
    min_target_word_length: int = 20

    encoder_name: str = "Alibaba-NLP/gte-large-en-v1.5"
    encoder_max_tokens: int = 4096
    encoder_batch_size: int = 4
    encoder_pooling: str = "mean"

    reward_model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    reward_model_type: str = "skywork"
    reward_model_cpu: bool = True
    reward_device: str = ""
    reward_eval_enabled: bool = True
    eval_subset_size: Optional[int] = 200
    save_eval_completions: bool = True

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True


def build_config(kwargs: dict):
    import dataclasses
    import typing
    cls = FilcoConfig
    fields_map = {f.name: f for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in kwargs.items():
        if k not in fields_map or v is None:
            continue
        if isinstance(v, str):
            ann = fields_map[k].type
            origin = getattr(ann, "__origin__", None)
            args = getattr(ann, "__args__", ())
            if origin is typing.Union and type(None) in args:
                inner = next(a for a in args if a is not type(None))
            else:
                inner = ann
            if inner is float:
                v = float(v)
            elif inner is int:
                v = int(v)
            elif inner is bool:
                v = v.lower() not in ("false", "0", "no")
        filtered[k] = v
    return cls(**filtered)

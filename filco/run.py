import argparse
import os

import yaml

from filco.base.config import build_config
from filco.training.filco_trainer import FilcoTrainer

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_defaults(config_override: str = None) -> dict:
    defaults = _load_yaml(os.path.join(_CONFIGS_DIR, "filco.yaml"))
    if config_override:
        defaults.update(_load_yaml(config_override))
    return defaults


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FilCo: test-time training for open-ended generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Optional YAML file overlaid on top of the defaults")
    p.add_argument("--model_name", default=None)
    p.add_argument("--model_type", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--n_vote", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--train_subset_size", type=int, default=None)
    p.add_argument("--eval_subset_size", type=int, default=None)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=None)
    p.add_argument("--eval_max_new_tokens", type=int, default=None)
    p.add_argument("--eval_start_episode", type=int, default=None)
    p.add_argument("--dry_run_episodes", type=int, default=None)
    p.add_argument("--resume_dir", default=None,
                   help="Output dir of a previous run to resume from")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_bf16", action="store_true", default=None)
    p.add_argument("--no_gradient_checkpointing", action="store_true", default=None)
    p.add_argument("--use_8bit_adam", action="store_true", default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--sglang_devices", default=None)
    p.add_argument("--sglang_tp", type=int, default=None)
    p.add_argument("--sglang_num_servers", type=int, default=None)
    p.add_argument("--sglang_host", default=None)
    p.add_argument("--sglang_port", type=int, default=None)
    p.add_argument("--sglang_timeout_s", type=int, default=None)
    p.add_argument("--sglang_context_length", type=int, default=None)
    p.add_argument("--sglang_mem_fraction", type=float, default=None)
    p.add_argument("--min_target_word_length", type=int, default=None)

    p.add_argument("--simpo_beta", type=float, default=None)
    p.add_argument("--simpo_gamma", type=float, default=None)
    p.add_argument("--max_train_completion_tokens", type=int, default=None,
                   help="cap completion tokens in the SimPO loss (bounds full-FT activation OOM)")
    p.add_argument("--gate_pct", type=float, default=None)
    p.add_argument("--suffix_words", type=int, default=None)
    p.add_argument("--prefix_words", type=int, default=None)
    p.add_argument("--rejected_strategy", default=None,
                   choices=["lowest_centrality_all", "lowest_centrality_survivor", "highest_predinfo"])
    p.add_argument("--encoder_name", default=None)
    p.add_argument("--encoder_max_tokens", type=int, default=None)
    p.add_argument("--encoder_batch_size", type=int, default=None)
    p.add_argument("--reward_model_cpu_off", action="store_true", default=None,
                   help="score reward model on GPU instead of CPU")
    p.add_argument("--reward_device", default=None,
                   help="explicit device for the reward model, e.g. cuda:1 (the sglang card) "
                        "for live reward during full FT without OOM on the training GPU")
    p.add_argument("--save_eval_completions_off", action="store_true", default=None)

    p.add_argument("--use_lora", action="store_true", default=None)
    p.add_argument("--no_lora", action="store_true", default=None,
                   help="force full fine-tuning (overrides a method's use_lora default)")
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument("--lora_target_modules", default=None)

    p.add_argument("--reward_model_name", default=None)
    p.add_argument("--reward_model_type", default=None)
    p.add_argument("--reward_model_cpu", action="store_true", default=None)
    p.add_argument("--no_reward_eval", action="store_true", default=None)
    p.add_argument("--no_trust_remote_code", action="store_true", default=None)
    p.add_argument("--verbose_samples", action="store_true", default=None)

    return p


def _args_to_dict(args: argparse.Namespace) -> dict:
    d = {}
    negated = {
        "no_bf16": ("bf16", False),
        "no_gradient_checkpointing": ("gradient_checkpointing", False),
        "no_reward_eval": ("reward_eval_enabled", False),
        "no_trust_remote_code": ("trust_remote_code", False),
        "reward_model_cpu_off": ("reward_model_cpu", False),
        "save_eval_completions_off": ("save_eval_completions", False),
        "no_lora": ("use_lora", False),
        "use_lora": ("use_lora", True),
    }
    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        if k in negated:
            if v:
                key, val = negated[k]
                d[key] = val
        else:
            d[k] = v
    return d


def main():
    args = _build_parser().parse_args()
    merged = {**_load_defaults(args.config), **_args_to_dict(args)}
    cfg = build_config(merged)

    print(f"[filco] model={cfg.model_name}  episodes={cfg.num_episodes}  "
          f"n_vote={cfg.n_vote}  lr={cfg.learning_rate}")

    FilcoTrainer(cfg).train()


if __name__ == "__main__":
    main()

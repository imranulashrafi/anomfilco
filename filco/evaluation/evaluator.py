import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from filco.base.data import format_prompt
from filco.logging.logger import log, log_verbose_block
from filco.utils.distributed import IS_RANK0, WORLD_SIZE
from filco.utils.distributed import get_raw_model


def _eval_mode(model, cfg):
    raw = get_raw_model(model)
    raw.eval()
    if cfg.gradient_checkpointing:
        raw.gradient_checkpointing_disable()
    return raw


def _train_mode(raw_model, cfg):
    raw_model.train()
    if cfg.gradient_checkpointing:
        raw_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def evaluate_reward_at_1(
    model, tokenizer, problems: list, cfg,
    reward_model, reward_tokenizer,
    label: str = "Eval",
    backend=None,
) -> dict:
    result = {"mean_reward": 0.0, "n_problems": 0, "elapsed_s": 0.0, "completions": [], "instructions": []}
    if not IS_RANK0:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()
        return result

    raw_model = _eval_mode(model, cfg)
    log(f"  [{label}]  {len(problems)} problems (greedy completions)")
    t0 = time.perf_counter()

    prompts = [format_prompt(p, tokenizer, cfg.model_type) for p in problems]
    instructions = [p["prompt"] for p in problems]

    completions = backend.generate_greedy(prompts, cfg.eval_max_new_tokens)

    if reward_model is None or reward_tokenizer is None:
        log(f"  [{label}]  reward model not loaded, skipping reward scoring")
        elapsed = time.perf_counter() - t0
        result = {"mean_reward": 0.0, "n_problems": len(completions), "elapsed_s": round(elapsed, 1), "completions": completions, "instructions": instructions}
        _train_mode(raw_model, cfg)
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()
        return result

    reward_device = next(reward_model.parameters()).device
    reward_model_type = getattr(cfg, "reward_model_type", "skywork")
    scores = []
    score_bar = tqdm(
        zip(instructions, completions),
        total=len(instructions),
        desc="  Scoring rewards",
        unit="prob",
        dynamic_ncols=True,
        leave=True,
        colour="yellow",
        bar_format="{l_bar}{bar:25}{r_bar}",
    )
    for instruction, response in score_bar:
        if not response.strip():
            scores.append(0.0)
            continue
        conversation = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        try:
            input_text = reward_tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            inputs = reward_tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=4096
            ).to(reward_device)
            with torch.no_grad():
                outputs = reward_model(**inputs)
            if reward_model_type == "armorm":
                score = float(outputs.score.item())
            else:
                score = float(outputs.logits[0][0].item())
            scores.append(score)
        except Exception as e:
            log(f"  reward scoring failed: {e}")
            scores.append(0.0)

    elapsed = time.perf_counter() - t0
    mean_reward = float(sum(scores) / len(scores)) if scores else 0.0
    result = {"mean_reward": mean_reward, "n_problems": len(scores), "elapsed_s": round(elapsed, 1), "completions": completions, "instructions": instructions}
    log_verbose_block(
        f"\n{'='*60}\n  RESULTS — {label}\n{'─'*60}\n"
        f"  mean reward : {mean_reward:.4f}\n"
        f"  problems    : {len(scores)}   elapsed: {elapsed:.0f}s\n{'='*60}\n"
    )
    _train_mode(raw_model, cfg)

    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()
    return result

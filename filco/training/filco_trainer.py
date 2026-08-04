import os
import pickle
from typing import Optional

import numpy as np

from filco.base.selection import _gate_survivors, compute_sample_predinfo, select_pair
from filco.base.semantic import Encoder, centrality_from_embeddings
from filco.logging.logger import FilcoEpisodeDiagnostics, log
from filco.training.base_trainer import BaseTrainer
from filco.training.filco_losses import simpo_loss
from filco.utils.distributed import IS_RANK0, LOCAL_RANK
from filco.utils.distributed import get_raw_model


class FilcoTrainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._encoder = None

    def get_method_label(self) -> str:
        return "filco"

    def _make_banner(self, cfg, n_problems, n_eval, total_steps, warmup_steps):
        base = super()._make_banner(cfg, n_problems, n_eval, total_steps, warmup_steps)
        lora = (f"on (r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout})"
                if getattr(cfg, "use_lora", False) else "off (full fine-tuning)")
        extra = (
            f"  FilCo selection + SimPO\n"
            f"  {'─'*56}\n"
            f"  SimPO beta / gamma : {cfg.simpo_beta} / {cfg.simpo_gamma}\n"
            f"  gate_pct           : {cfg.gate_pct}   "
            f"suffix/prefix words: {cfg.suffix_words}/{cfg.prefix_words}\n"
            f"  rejected strategy  : {cfg.rejected_strategy}\n"
            f"  encoder            : {cfg.encoder_name}  (max_tok={cfg.encoder_max_tokens})\n"
            f"  LoRA               : {lora}\n"
            f"  reward eval        : {cfg.reward_model_name}  "
            f"on {getattr(cfg, 'reward_device', '') or ('cpu' if cfg.reward_model_cpu else f'cuda:{LOCAL_RANK}')}\n"
            f"{'='*60}\n"
        )
        return base + extra

    def create_diagnostics(self) -> FilcoEpisodeDiagnostics:
        return FilcoEpisodeDiagnostics()


    def load_auxiliary_models(self, cfg):
        if not IS_RANK0:
            return
        log(f"Loading encoder {cfg.encoder_name} on cuda:{LOCAL_RANK} "
            f"(max_tokens={cfg.encoder_max_tokens}, pooling={cfg.encoder_pooling})")
        self._encoder = Encoder(
            cfg.encoder_name, device=f"cuda:{LOCAL_RANK}",
            max_seq_length=cfg.encoder_max_tokens, pooling=cfg.encoder_pooling,
        )
        if not cfg.eval_subset_size or not getattr(cfg, "reward_eval_enabled", True):
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        rdev = getattr(cfg, "reward_device", "") or ""
        if rdev:
            device, dtype, devstr = rdev, torch.float16, rdev
        elif cfg.reward_model_cpu:
            device, dtype, devstr = "cpu", torch.float32, "cpu"
        else:
            device, dtype, devstr = LOCAL_RANK, torch.float16, f"cuda:{LOCAL_RANK}"
        log(f"Loading reward model {cfg.reward_model_name} on {devstr} (eval only)")
        self._reward_tokenizer = AutoTokenizer.from_pretrained(
            cfg.reward_model_name, trust_remote_code=True
        )
        self._reward_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.reward_model_name,
            torch_dtype=dtype,
            device_map={"": device}, trust_remote_code=True,
        ).eval()


    def _augment_episode_data(self, ep_data: list, cfg):
        if not IS_RANK0 or self._encoder is None:
            return
        import torch

        raw = get_raw_model(self._model)
        was_training = raw.training
        raw.eval()
        if cfg.gradient_checkpointing:
            raw.gradient_checkpointing_disable()

        log(f"  FilCo augment: encoding + predinfo for {len(ep_data)} prompts ...")
        n_ok = 0
        try:
            for row in ep_data:
                samples = row["sample_completions"]
                lps = row["sample_logprobs"]
                valid_idx = [
                    k for k, (s, lp) in enumerate(zip(samples, lps))
                    if s and s.strip() and len(s.split()) >= cfg.min_target_word_length and lp
                ]
                row["filco_valid_idx"] = valid_idx
                if len(valid_idx) < 4:
                    row["filco_embeddings"] = None
                    row["filco_predinfo"] = None
                    continue
                texts = [samples[k] for k in valid_idx]
                emb = self._encoder.encode(texts, batch_size=cfg.encoder_batch_size)
                with torch.no_grad():
                    predinfo = compute_sample_predinfo(
                        self._model, self._train_tok, row["prompt_text"], texts,
                        cfg.suffix_words, cfg.prefix_words, cfg,
                    )
                row["filco_embeddings"] = np.asarray(emb, dtype=np.float16)
                row["filco_predinfo"] = predinfo
                n_ok += 1
        finally:
            if was_training:
                raw.train()
                if cfg.gradient_checkpointing:
                    raw.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
        log(f"  FilCo augment: {n_ok}/{len(ep_data)} prompts have a usable pool (>=4 valid)")


    def train_on_problem(self, row, model, tokenizer, diag) -> Optional[float]:
        cfg = self.cfg
        valid_idx = row.get("filco_valid_idx")
        emb = row.get("filco_embeddings")
        predinfo = row.get("filco_predinfo")

        if not valid_idx or emb is None or predinfo is None or len(valid_idx) < 4:
            diag.record_skip("no_pool")
            return None

        emb = np.asarray(emb, dtype=np.float64)
        pair = select_pair(
            emb, predinfo, gate_pct=cfg.gate_pct,
            rejected_strategy=cfg.rejected_strategy,
        )
        if pair is None:
            diag.record_skip("same_pair")
            return None
        ci, ri = pair

        samples = row["sample_completions"]
        chosen = samples[valid_idx[ci]]
        rejected = samples[valid_idx[ri]]

        cen = centrality_from_embeddings(emb)
        keep = _gate_survivors(np.asarray(predinfo, dtype=np.float64), cfg.gate_pct)
        n_gated = int((~keep).sum())

        if cfg.dry_run_episodes > 0:
            diag.record_trained(
                float(cen[ci] - cen[ri]), 0.0, 0.0, float(cen[ci]), float(cen[ri]),
                n_gated, len(chosen.split()), len(rejected.split()), None,
            )
            return None

        result = simpo_loss(
            model, tokenizer, row["prompt_text"], chosen, rejected, cfg,
            backward_scale=cfg.gradient_accumulation_steps,
        )
        if result is None:
            diag.record_skip("loss")
            return None

        diag.record_trained(
            result["margin"], result["mean_lp_chosen"], result["mean_lp_rejected"],
            float(cen[ci]), float(cen[ri]), n_gated,
            len(chosen.split()), len(rejected.split()), result["loss"],
        )
        return result["loss"]


    def _on_eval_complete(self, episode, eval_result, eval_problems, cfg):
        if not IS_RANK0 or not getattr(cfg, "save_eval_completions", True):
            return
        comps = eval_result.get("completions")
        if not comps:
            return
        n = min(len(comps), len(eval_problems))
        payload = {
            "episode": episode,
            "instructions": [eval_problems[i]["prompt"] for i in range(n)],
            "completions": comps[:n],
            "gold_chosen": [eval_problems[i].get("chosen", "") for i in range(n)],
            "gold_rejected": [eval_problems[i].get("rejected", "") for i in range(n)],
            "mean_reward": eval_result.get("mean_reward", 0.0),
        }
        out = os.path.join(cfg.output_dir, f"episode_{episode}_eval_completions.pkl")
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        log(f"  Saved eval completions for offline judge: {out}")

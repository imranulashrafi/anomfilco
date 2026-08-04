import atexit
import datetime
import os
import random
import signal
import time
from typing import Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from filco.base.data import format_prompt, load_problems
from filco.base.model import (
    create_optimizer,
    load_model_and_tokenizer,
    save_model_checkpoint,
    wrap_model_for_training,
)
from filco.evaluation.evaluator import evaluate_reward_at_1
from filco.logging.logger import (
    FilcoEpisodeDiagnostics,
    MetricsLogger,
    _close_log_file,
    _setup_log_file,
    log,
    log_verbose_block,
)
from filco.generation import get_backend
from filco.utils.checkpointing import (
    _cleanup_old_checkpoints,
    load_completions,
    load_run_state,
    save_completions,
    save_run_state,
)
from filco.utils.distributed import (
    IS_RANK0,
    LOCAL_RANK,
    WORLD_SIZE,
    cleanup_distributed,
    init_distributed,
)
from filco.utils.misc import _eta, _make_output_dir, gpu_mem, set_seed


class BaseTrainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self._backend = None
        self._metrics_logger: Optional[MetricsLogger] = None
        self._reward_model = None
        self._reward_tokenizer = None


    def get_method_label(self) -> str:
        return self.cfg.method

    def create_diagnostics(self):
        return FilcoEpisodeDiagnostics()

    def load_auxiliary_models(self, cfg):
        pass

    def _augment_episode_data(self, ep_data: list, cfg):
        pass

    def _on_eval_complete(self, episode: int, eval_result: dict, eval_problems: list, cfg):
        pass

    def train_on_problem(
        self, row: dict, model, tokenizer, diag: FilcoEpisodeDiagnostics
    ) -> Optional[float]:
        raise NotImplementedError


    def _register_signal_handlers(self):
        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            if IS_RANK0:
                from filco.logging.logger import logger
                logger.info(f"Signal {sig_name} received - shutting down")
            _close_log_file(reason=f"terminated by {sig_name}")
            if IS_RANK0 and self._backend:
                self._backend.shutdown()
            cleanup_distributed()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, _handler)
            except (OSError, ValueError):
                pass


    def _run_phase1(self, episode: int, problems: list, tok_gen, cfg):
        log(f"\n{'─'*60}\n  PHASE 1: GENERATION (greedy + samples w/ logprobs)\n{'─'*60}")
        all_prompts = [format_prompt(p, tok_gen, cfg.model_type) for p in problems]

        greedy_sampling = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": cfg.eval_max_new_tokens}
        sample_sampling = {"temperature": cfg.temperature, "top_p": cfg.top_p,
                           "max_new_tokens": cfg.max_new_tokens}

        greedy_out, sample_out, sample_lps = self._backend.generate_phase1(
            all_prompts, cfg.n_candidates, greedy_sampling, sample_sampling
        )

        ep_data = [
            {
                "problem_idx": i + 1,
                "prompt": problems[i]["prompt"],
                "prompt_text": all_prompts[i],
                "greedy_completion": greedy_out[i],
                "sample_completions": sample_out[i],
                "sample_logprobs": sample_lps[i],
            }
            for i in range(len(problems))
        ]
        self._augment_episode_data(ep_data, cfg)
        save_completions(episode, problems, ep_data, cfg.output_dir)
        log(f"Generation complete: {len(ep_data)} problems")


    def _make_banner(self, cfg, n_problems: int, n_eval: int,
                     total_steps: int, warmup_steps: int) -> str:
        return (
            f"\n{'='*60}\n"
            f"  FilCo  |  {cfg.model_name}\n"
            f"  {'─'*56}\n"
            f"  Backend      : SGLANG"
            f"  devices=[{cfg.sglang_devices}]\n"
            f"  Problems     : {n_problems}  |  eval: {n_eval}\n"
            f"  Episodes     : {cfg.num_episodes}  |  n_candidates: {cfg.n_candidates}"
            f"  |  temp: {cfg.temperature}\n"
            f"  LR           : {cfg.learning_rate}  |  warmup: {warmup_steps} steps\n"
            f"  Grad accum   : {cfg.gradient_accumulation_steps}"
            f"  |  total steps: {total_steps}\n"
            f"  8-bit Adam   : {cfg.use_8bit_adam}\n"
            f"  Output dir   : {cfg.output_dir}\n"
            f"  Dry-run eps  : {cfg.dry_run_episodes}\n"
            f"{'='*60}\n"
        )


    def train(self):
        cfg = self.cfg

        if IS_RANK0:
            if not cfg.output_dir:
                cfg.output_dir = _make_output_dir(cfg.model_name, cfg.data_path, cfg.method)
            os.makedirs(cfg.output_dir, exist_ok=True)

        if WORLD_SIZE > 1:
            sync_file = os.path.join(cfg.output_dir, f".output_dir_sync_{cfg.method}")
            if IS_RANK0:
                os.makedirs(cfg.output_dir, exist_ok=True)
                with open(sync_file, "w") as f:
                    f.write(cfg.output_dir)
            while not os.path.exists(sync_file):
                time.sleep(0.1)
            time.sleep(0.5)
            if not IS_RANK0:
                with open(sync_file, "r") as f:
                    cfg.output_dir = f.read().strip()

        if IS_RANK0:
            run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(cfg.output_dir, f"run_{run_ts}.log")
            _setup_log_file(log_path)
            log(f"Logging to: {log_path}")
            atexit.register(_close_log_file, reason="process exited normally")
            self._register_signal_handlers()
            self._metrics_logger = MetricsLogger(
                os.path.join(cfg.output_dir, "metrics.jsonl")
            )

        init_distributed()
        if WORLD_SIZE > 1:
            dist.barrier()

        set_seed(cfg.seed + LOCAL_RANK)

        problems = load_problems(cfg.data_path)
        if cfg.train_subset_size:
            problems = problems[: cfg.train_subset_size]
        if cfg.eval_subset_size is None:
            eval_problems = problems
        elif cfg.eval_subset_size <= 0:
            eval_problems = []
        else:
            eval_problems = problems[: cfg.eval_subset_size]

        tok_gen = None
        if IS_RANK0:
            tok_gen = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
            if tok_gen.pad_token is None:
                tok_gen.pad_token = tok_gen.eos_token

        resume_episode = 0
        resume_ckpt = None
        if cfg.resume_dir:
            state = load_run_state(cfg.resume_dir)
            last_ep = state.get("last_episode", 0)
            if last_ep > 0:
                ckpt_candidate = os.path.join(cfg.resume_dir, f"checkpoint_ep{last_ep}")
                if os.path.isdir(ckpt_candidate):
                    resume_episode = last_ep
                    resume_ckpt = ckpt_candidate
                    log(f"Resuming from episode {resume_episode} "
                        f"(checkpoint: {resume_ckpt})")

        total_steps = cfg.num_episodes * len(problems) // max(cfg.gradient_accumulation_steps, 1)
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

        model, tok = load_model_and_tokenizer(
            cfg, model_path=resume_ckpt
        )
        model = wrap_model_for_training(model)
        optimizer = create_optimizer(model, cfg)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        self._model = model
        self._train_tok = tok

        if IS_RANK0:
            self.load_auxiliary_models(cfg)

        if IS_RANK0:
            log(self._make_banner(cfg, len(problems), len(eval_problems),
                                  total_steps, warmup_steps))

        if IS_RANK0:
            backend_init = resume_ckpt or cfg.model_name
            log(f"Starting sglang backend from {backend_init}...")
            self._backend = get_backend(cfg, backend_init)

        if WORLD_SIZE > 1:
            dist.barrier()

        train_start = time.perf_counter()
        max_episodes = cfg.dry_run_episodes if cfg.dry_run_episodes > 0 else cfg.num_episodes

        if cfg.eval_start_episode == 0 and resume_episode == 0 and eval_problems and cfg.dry_run_episodes == 0:
            log(f"\n{'─'*60}\n  BASELINE EVALUATION (pre-training)\n{'─'*60}")
            r_res = evaluate_reward_at_1(
                model, tok, eval_problems, cfg,
                self._reward_model, self._reward_tokenizer,
                label="Ep 0 baseline (reward@1)",
                backend=self._backend,
            )
            if IS_RANK0:
                log(f"  Baseline mean reward: {r_res['mean_reward']:.4f}")
                if self._metrics_logger:
                    self._metrics_logger.log_episode(0, {
                        "mean_reward": r_res["mean_reward"],
                        "trained": 0, "skipped": 0, "ep_time_min": 0.0,
                    })
                self._on_eval_complete(0, r_res, eval_problems, cfg)

        for episode in range(resume_episode + 1, max_episodes + 1):
            ep_start = time.perf_counter()
            log(f"\n{'='*60}")
            log(f"  Episode {episode}/{max_episodes}  |  {gpu_mem()}"
                f"  {'[DRY RUN]' if cfg.dry_run_episodes > 0 else ''}")
            log(f"{'='*60}")

            resumed_gen = False
            if IS_RANK0:
                pkl_path = os.path.join(cfg.output_dir, f"episode_{episode}_completions.pkl")
                if os.path.exists(pkl_path):
                    log(f"  Resumed: episode {episode} completions found, skipping Phase 1")
                    resumed_gen = True

            if IS_RANK0 and not resumed_gen:
                self._run_phase1(episode, problems, tok_gen, cfg)

            if WORLD_SIZE > 1:
                dist.barrier()

            log(f"\n{'─'*60}\n  PHASE 2: TRAINING ({self.get_method_label()})\n{'─'*60}")

            comp_data = load_completions(episode, cfg.output_dir)
            ep_data = comp_data["data"]

            rng = random.Random(cfg.seed + episode * 1000 + LOCAL_RANK)
            indices = list(range(len(ep_data)))
            rng.shuffle(indices)
            shuffled = [ep_data[i] for i in indices]

            diag = self.create_diagnostics()
            global_step = (episode - 1) * len(problems) // cfg.gradient_accumulation_steps
            accum_count = 0
            accum_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            prob_bar = tqdm(
                enumerate(shuffled, 1),
                total=len(shuffled),
                desc=f"  Train Ep {episode}/{max_episodes} [R{LOCAL_RANK}]",
                unit="prob",
                dynamic_ncols=True,
                leave=True,
                colour="green",
                bar_format="{l_bar}{bar:25}{r_bar}",
                disable=not IS_RANK0,
            )

            for prob_idx, row in prob_bar:
                loss_val = self.train_on_problem(row, model, tok, diag)

                if loss_val is None:
                    continue

                accum_loss += loss_val
                accum_count += 1

                force_step = prob_idx == len(shuffled)
                if accum_count >= cfg.gradient_accumulation_steps or force_step:
                    if accum_count > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.max_grad_norm
                        ).item()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if IS_RANK0:
                            elapsed = time.perf_counter() - train_start
                            eta = _eta(
                                elapsed,
                                prob_idx + (episode - 1) * len(problems),
                                max_episodes * len(problems),
                            )
                            log(
                                f"  STEP #{global_step}  ep{episode} p{prob_idx}  "
                                f"loss_avg={accum_loss/accum_count:.3e}  "
                                f"gnorm={grad_norm:.3f}  "
                                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                                f"accum={accum_count}/{cfg.gradient_accumulation_steps}  "
                                f"ETA={eta}"
                            )
                        accum_loss = 0.0
                        accum_count = 0

                if IS_RANK0:
                    prob_bar.set_postfix({
                        "loss": f"{loss_val:.2e}",
                        "step": global_step,
                        "tr": diag.n_trained,
                        "sk": diag.n_skipped,
                    })

            if accum_count > 0 and cfg.dry_run_episodes == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            ep_elapsed = time.perf_counter() - ep_start
            log(
                f"\n{'='*60}\n  EPISODE {episode}/{max_episodes} COMPLETE\n"
                f"  time: {ep_elapsed/60:.1f} min  |  GPU: {gpu_mem()}\n{'='*60}"
            )
            log_verbose_block(diag.summary())

            if cfg.dry_run_episodes > 0:
                log(f"  Dry-run episode {episode} done.")
                continue

            log(f"\n{'─'*60}\n  PHASE 3: CHECKPOINT + WEIGHT UPDATE\n{'─'*60}")
            ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_ep{episode}")
            if IS_RANK0:
                _cleanup_old_checkpoints(cfg.output_dir, episode, cfg.keep_last_n_checkpoints)
            save_model_checkpoint(model, tok, ckpt_path)

            if WORLD_SIZE > 1:
                dist.barrier()

            if IS_RANK0:
                self._backend.sync_weights(model, ckpt_path)
                save_run_state(cfg.output_dir, {
                    "method": cfg.method,
                    "last_episode": episode,
                    "output_dir": cfg.output_dir,
                })

            if WORLD_SIZE > 1:
                dist.barrier()

            if episode >= cfg.eval_start_episode and eval_problems:
                log(f"\n{'─'*60}\n  PHASE 4: EVALUATION\n{'─'*60}")
                ep_label = f"Ep {episode}/{max_episodes}"
                r_res = evaluate_reward_at_1(
                    model, tok, eval_problems, cfg,
                    self._reward_model, self._reward_tokenizer,
                    label=f"{ep_label} (reward@1)",
                    backend=self._backend,
                )
                if IS_RANK0:
                    log(f"  Ep {episode} mean reward: {r_res['mean_reward']:.4f}")
                    if self._metrics_logger:
                        self._metrics_logger.log_episode(episode, {
                            "mean_reward": r_res["mean_reward"],
                            "trained": diag.n_trained,
                            "skipped": diag.n_skipped,
                            "ep_time_min": round(ep_elapsed / 60, 2),
                        })
                    self._on_eval_complete(episode, r_res, eval_problems, cfg)

        total_elapsed = time.perf_counter() - train_start
        log(f"\n{'='*60}\n  TRAINING COMPLETE!  Total time: {total_elapsed/3600:.2f}h\n{'='*60}\n")

        if cfg.dry_run_episodes == 0:
            final_path = os.path.join(cfg.output_dir, "final_model")
            if IS_RANK0:
                _cleanup_old_checkpoints(cfg.output_dir, cfg.num_episodes + 1, 1)
            save_model_checkpoint(model, tok, final_path)

        if IS_RANK0 and self._backend:
            self._backend.shutdown()

        _close_log_file(reason="training complete")
        cleanup_distributed()

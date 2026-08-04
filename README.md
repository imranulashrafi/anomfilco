# FilCo

Test-time training for open-ended generation. For each prompt the policy samples
`N` candidates, scores them for embedding consensus and for degeneracy, discards
the most degenerate ones, forms a preference pair from the survivors, and updates
itself with a reference-free SimPO objective. No references, reward models, or
LLM judges are used in the training loop.

## Requirements

- 2 GPUs: one for generation (SGLang), one for training.
- Python 3.12, CUDA 12.8.

## Install

```bash
bash setup.sh
```

This installs torch, SGLang 0.5.2 and the training dependencies, then applies a
small SGLang patch that skips an unused `torchao` import. Point the Hugging Face
cache at a volume with room for the model weights:

```bash
export HF_HOME=$PWD/hf_cache
```

## Run

```bash
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=1 -m filco.run \
  --model_name Qwen/Qwen2.5-7B \
  --data_path data/ultrafeedback_binarized_500.json \
  --sglang_devices 1 \
  --output_dir runs/filco_ultrafeedback
```

Training runs on GPU 0 and generation on GPU 1. Every episode samples candidates,
trains on the selected pairs, and evaluates the updated policy's greedy output
under the held-out reward model, printing the mean reward. Per-episode
completions and a checkpoint are written to `--output_dir`.

Defaults live in `filco/configs/filco.yaml`; any of them can be overridden on the
command line or with `--config path/to/your.yaml`.

## Key options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n_candidates` | 32 | candidates sampled per prompt |
| `--gate_pct` | 95.0 | degeneracy percentile above which candidates are discarded |
| `--suffix_words` / `--prefix_words` | 30 / 30 | window used to score degeneracy |
| `--rejected_strategy` | `lowest_centrality_all` | how the negative is chosen |
| `--simpo_beta` / `--simpo_gamma` | 2.0 / 0.5 | SimPO loss parameters |
| `--encoder_name` | `Alibaba-NLP/gte-large-en-v1.5` | sentence encoder for consensus |
| `--num_episodes` | 15 | training episodes |
| `--learning_rate` | 8e-7 | |
| `--use_lora` / `--no_lora` | full fine-tuning | LoRA instead of full fine-tuning |
| `--reward_model_name` | Skywork-Reward-Llama-3.1-8B-v0.2 | held-out model for per-episode reward |
| `--no_reward_eval` | off | skip the per-episode reward evaluation |

Setting `--gate_pct 100` disables the degeneracy filter and selects on consensus
alone, which is the ablation baseline.

## Data

`data/` holds the three instruction-following prompt sets used in the
experiments, each 500 prompts in the format the loader expects:

```json
{"prompt": "..."}
```

Any JSON list of objects with a `prompt` field works. Other keys are ignored.

## Layout

```
filco/
  run.py                  entry point
  base/       config, data loading, model setup, encoder, pair selection
  training/   SimPO loss and the episode training loop
  generation/ SGLang backend
  evaluation/ per-episode reward scoring
```

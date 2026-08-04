#!/usr/bin/env bash
set -euo pipefail

python - <<'PY' || pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY

pip install --upgrade pip
pip install "sglang[all]==0.5.2"
pip install bitsandbytes sentence-transformers peft
pip install "torchao>=0.17" --no-deps
python patch_sglang.py

python - <<'PY'
import sglang, bitsandbytes, torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
print("setup complete")
PY

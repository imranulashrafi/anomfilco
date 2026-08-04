import importlib.util
import sys

spec = importlib.util.find_spec("sglang.srt.layers.torchao_utils")
if not spec or not spec.origin:
    sys.exit("ERROR: sglang not found. Install sglang before running this patch.")

path = spec.origin
with open(path) as f:
    src = f.read()

SENTINEL = "# FILCO-PATCH-torchao-skip"
if SENTINEL in src:
    print(f"Already patched: {path}")
    sys.exit(0)

target = "    from torchao.quantization import ("
if target not in src:
    sys.exit(f"ERROR: could not find the torchao import in {path}. "
             "sglang version differs; patch manually by adding an early "
             "`if torchao_config in ('', None): return model` at the top of "
             "apply_torchao_config_to_model.")

guard = (
    f"    {SENTINEL}\n"
    '    if torchao_config == "" or torchao_config is None:\n'
    "        return model\n"
    + target
)
src = src.replace(target, guard, 1)
with open(path, "w") as f:
    f.write(src)
print(f"Patched: {path}")

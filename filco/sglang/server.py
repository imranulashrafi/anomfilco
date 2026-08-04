import atexit
import os
import socket
import subprocess
import time

import requests

from filco.logging.logger import log

_TORCHRUN_ENV_VARS = [
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE", "OMP_NUM_THREADS",
]


def _is_tcp_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _wait_for_server(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_tcp_open(host, port, timeout_s=1.0):
            try:
                r = requests.get(f"http://{host}:{port}/health", timeout=2)
                if r.status_code < 500:
                    return True
            except Exception:
                pass
        time.sleep(1)
    return False


def _launch_one_sglang(cfg, model_path: str, gpu_devices: str, port: int, server_idx: int):
    host = cfg.sglang_host
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, f"sglang_server_{server_idx}.log")
    context_len = cfg.sglang_context_length or max(
        cfg.max_new_tokens * 2, cfg.eval_max_new_tokens * 2, 4096
    )
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host, "--port", str(port),
        "--tp", str(cfg.sglang_tp),
        "--context-length", str(context_len),
        "--cuda-graph-max-bs", "16",
        "--mem-fraction-static", str(cfg.sglang_mem_fraction)
    ]
    env = os.environ.copy()
    for key in _TORCHRUN_ENV_VARS:
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    log(f"Launching SGLang #{server_idx}: GPU={gpu_devices} port={port} ctx={context_len}")
    log(f"  SGLang cmd: {' '.join(cmd)}")
    logf = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)

    def _cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        logf.close()

    atexit.register(_cleanup)
    if not _wait_for_server(host, port, cfg.sglang_timeout_s):
        if proc.poll() is None:
            proc.terminate()
        raise RuntimeError(
            f"SGLang server #{server_idx} did not become ready within "
            f"{cfg.sglang_timeout_s}s. Check log: {log_path}"
        )
    log(f"SGLang server #{server_idx} ready at http://{host}:{port}")
    return proc


def start_sglang_servers(cfg, model_path: str) -> list:
    procs = []
    for i, (gpu_devs, port) in enumerate(zip(cfg.sglang_device_groups, cfg.sglang_ports)):
        if _is_tcp_open(cfg.sglang_host, port):
            log(f"Reusing existing SGLang server at http://{cfg.sglang_host}:{port}")
            procs.append(None)
        else:
            procs.append(_launch_one_sglang(cfg, model_path, gpu_devs, port, i))
    log(f"All {len(procs)} SGLang server(s) ready")
    return procs


def stop_sglang_servers(procs: list):
    for i, proc in enumerate(procs):
        if proc is not None and proc.poll() is None:
            log(f"Stopping SGLang server #{i}...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
    log("All SGLang servers stopped")


def update_sglang_weights_all(cfg, ckpt_path: str, server_procs: list) -> list:
    log(f"Updating SGLang weights on {len(server_procs)} server(s) <- {ckpt_path}")
    new_procs = []
    for i, (proc, gpu_devs, port) in enumerate(
        zip(server_procs, cfg.sglang_device_groups, cfg.sglang_ports)
    ):
        url = f"http://{cfg.sglang_host}:{port}/update_weights_from_disk"
        try:
            r = requests.post(
                url, json={"model_path": ckpt_path, "load_format": "auto"}, timeout=300
            )
            r.raise_for_status()
            log(f"  Server #{i} weights updated (hot-reload)")
            new_procs.append(proc)
        except Exception as e:
            log(f"  Server #{i} hot-reload failed ({e}); restarting...")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except Exception:
                    proc.kill()
            new_procs.append(_launch_one_sglang(cfg, ckpt_path, gpu_devs, port, i))
    return new_procs


def _get_server_urls(cfg) -> list:
    return [f"http://{cfg.sglang_host}:{port}/generate" for port in cfg.sglang_ports]

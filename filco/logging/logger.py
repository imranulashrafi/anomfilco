import datetime
import json
import logging
from typing import Optional

from filco.utils.distributed import IS_RANK0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("filco")
_log_file_handler: Optional[logging.FileHandler] = None


def log(msg: str):
    if IS_RANK0:
        logger.info(msg)
        if _log_file_handler is not None:
            _log_file_handler.stream.write(
                datetime.datetime.now().strftime("%H:%M:%S") + "  " + msg + "\n"
            )
            _log_file_handler.stream.flush()


def log_verbose_block(text: str):
    if not IS_RANK0:
        return
    if _log_file_handler is not None:
        _log_file_handler.stream.write(text + "\n")
        _log_file_handler.stream.flush()
    print(text, flush=True)


def _preview(text: str, head: int = 100, tail: int = 120) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= head + tail + 5:
        return text
    return f"{text[:head]} ⋯ {text[-tail:]}"


def sample_log_gen(idx: int, total: int, output: str):
    if not IS_RANK0:
        return
    from tqdm import tqdm
    words = len((output or "").split())
    tqdm.write(f"gen {idx}/{total}  ~{words}w")
    tqdm.write(f"   {_preview(output)}")


def _setup_log_file(log_path: str):
    global _log_file_handler
    _log_file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    _log_file_handler.setLevel(logging.INFO)
    _log_file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_log_file_handler)


def _close_log_file(reason: str = "run ended"):
    global _log_file_handler
    if _log_file_handler is None:
        return
    if IS_RANK0:
        logger.info(f"\n{'='*60}\n  Log closed - {reason}\n{'='*60}\n")
    _log_file_handler.flush()
    _log_file_handler.close()
    logger.removeHandler(_log_file_handler)
    _log_file_handler = None


def _safe_mean(xs: list) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


class FilcoEpisodeDiagnostics:

    def __init__(self):
        self.total = 0
        self._trained = 0
        self.skip_no_pool = 0
        self.skip_same_pair = 0
        self.skip_loss = 0
        self.margins = []
        self.mean_lp_chosen = []
        self.mean_lp_rejected = []
        self.cen_chosen = []
        self.cen_rejected = []
        self.n_gated = []
        self.len_chosen = []
        self.len_rejected = []
        self.loss_values = []

    @property
    def n_trained(self) -> int:
        return self._trained

    @property
    def n_skipped(self) -> int:
        return self.total - self._trained

    def record_skip(self, reason: str):
        self.total += 1
        if reason == "no_pool":
            self.skip_no_pool += 1
        elif reason == "same_pair":
            self.skip_same_pair += 1
        elif reason == "loss":
            self.skip_loss += 1

    def record_trained(self, margin, mlp_c, mlp_r, cen_c, cen_r,
                        n_gated, len_c, len_r, loss):
        self.total += 1
        self._trained += 1
        self.margins.append(margin)
        self.mean_lp_chosen.append(mlp_c)
        self.mean_lp_rejected.append(mlp_r)
        self.cen_chosen.append(cen_c)
        self.cen_rejected.append(cen_r)
        self.n_gated.append(n_gated)
        self.len_chosen.append(len_c)
        self.len_rejected.append(len_r)
        if loss is not None:
            self.loss_values.append(loss)

    def summary(self) -> str:
        def pct(x, tot): return f"{100*x/max(tot,1):.1f}%"
        return (
            f"\n{'─'*60}\n  EPISODE DIAGNOSTICS  (FilCo consensus + predinfo gate)\n{'─'*60}\n"
            f"  total problems seen         : {self.total}\n"
            f"  ├─ skipped: pool too small  : {self.skip_no_pool}\n"
            f"  ├─ skipped: chosen==rejected: {self.skip_same_pair}\n"
            f"  ├─ skipped: loss None/naN   : {self.skip_loss}\n"
            f"  └─ TRAINED                  : {self._trained}  ({pct(self._trained, self.total)})\n"
            f"  mean SimPO margin (c-r)     : {_safe_mean(self.margins):+.4f}\n"
            f"  mean logprob chosen         : {_safe_mean(self.mean_lp_chosen):+.4f}   <- collapse monitor\n"
            f"  mean logprob rejected       : {_safe_mean(self.mean_lp_rejected):+.4f}   <- collapse monitor\n"
            f"  mean centrality chosen      : {_safe_mean(self.cen_chosen):+.4f}\n"
            f"  mean centrality rejected    : {_safe_mean(self.cen_rejected):+.4f}\n"
            f"  mean gated-out per prompt   : {_safe_mean(self.n_gated):.2f}\n"
            f"  mean len chosen / rejected  : {_safe_mean(self.len_chosen):.0f} / {_safe_mean(self.len_rejected):.0f} words\n"
            f"  mean loss                   : {_safe_mean(self.loss_values):.4e}\n"
            f"{'─'*60}"
        )


class MetricsLogger:
    def __init__(self, path: str):
        self.path = path

    def log_episode(self, episode: int, metrics: dict):
        if not IS_RANK0:
            return
        entry = {
            "episode": episode,
            **metrics,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

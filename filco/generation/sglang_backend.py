from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from filco.generation.backend import GenerationBackend
from filco.logging.logger import log, sample_log_gen
from filco.sglang.generation import _generate_from_one_server
from filco.sglang.server import (
    _get_server_urls,
    start_sglang_servers,
    stop_sglang_servers,
    update_sglang_weights_all,
)


class SGLangBackend(GenerationBackend):

    def __init__(self, cfg, model_path: str):
        self.cfg = cfg
        self._procs = start_sglang_servers(cfg, model_path)

    def generate_phase1(
        self,
        prompts: list,
        n_candidates: int,
        greedy_sampling: dict,
        sample_sampling: dict,
    ) -> tuple:
        urls = _get_server_urls(self.cfg)
        n = len(prompts)

        greedy_outs = [""] * n
        sample_outs = [[""] * n_candidates for _ in range(n)]
        sample_lps = [[[] for _ in range(n_candidates)] for _ in range(n)]

        with ThreadPoolExecutor(max_workers=max(len(urls), 4)) as pool:
            gfutures = {
                pool.submit(
                    _generate_from_one_server,
                    urls[i % len(urls)], pt, 1, greedy_sampling, False,
                ): ("g", i)
                for i, pt in enumerate(prompts)
            }
            sfutures = {
                pool.submit(
                    _generate_from_one_server,
                    urls[i % len(urls)], pt, n_candidates, sample_sampling, True,
                ): ("s", i)
                for i, pt in enumerate(prompts)
            }
            pbar = tqdm(
                as_completed(list(gfutures) + list(sfutures)),
                total=n * 2,
                desc="  Generating",
                unit="req",
                dynamic_ncols=True,
                leave=True,
                colour="cyan",
                bar_format="{l_bar}{bar:25}{r_bar}",
            )
            verbose = getattr(self.cfg, "verbose_samples", False)
            every = max(1, getattr(self.cfg, "sample_log_every", 50))
            seen = 0
            for fut in pbar:
                tag, idx = gfutures.get(fut) or sfutures.get(fut)
                try:
                    texts, lps = fut.result()
                except Exception as e:
                    log(f"  SGLang gen failed ({tag}) problem {idx + 1}: {e}")
                    texts = [""] * (1 if tag == "g" else n_candidates)
                    lps = [[]] * (1 if tag == "g" else n_candidates)
                if tag == "g":
                    greedy_outs[idx] = texts[0] if texts else ""
                    seen += 1
                    if verbose and seen % every == 0:
                        sample_log_gen(seen, n, greedy_outs[idx])
                else:
                    sample_outs[idx] = (texts + [""] * n_candidates)[:n_candidates]
                    sample_lps[idx] = (lps + [[]] * n_candidates)[:n_candidates]

        return greedy_outs, sample_outs, sample_lps

    def generate_greedy(self, prompts: list, max_new_tokens: int) -> list:
        urls = _get_server_urls(self.cfg)
        sampling = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": max_new_tokens}
        outs = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max(len(urls), 4)) as pool:
            futures = {
                pool.submit(
                    _generate_from_one_server,
                    urls[i % len(urls)], pt, 1, sampling, False,
                ): i
                for i, pt in enumerate(prompts)
            }
            pbar = tqdm(
                as_completed(futures),
                total=len(prompts),
                desc="  Eval greedy",
                unit="req",
                dynamic_ncols=True,
                leave=True,
                colour="magenta",
                bar_format="{l_bar}{bar:25}{r_bar}",
            )
            verbose = getattr(self.cfg, "verbose_samples", False)
            every = max(1, getattr(self.cfg, "sample_log_every", 50))
            seen = 0
            for fut in pbar:
                idx = futures[fut]
                try:
                    texts, _ = fut.result()
                    outs[idx] = texts[0] if texts else ""
                except Exception as e:
                    log(f"  SGLang eval failed problem {idx + 1}: {e}")
                seen += 1
                if verbose and seen % every == 0:
                    sample_log_gen(seen, len(prompts), outs[idx])
        return outs

    def sync_weights(self, model, ckpt_path: str) -> None:
        self._procs = update_sglang_weights_all(self.cfg, ckpt_path, self._procs)

    def shutdown(self) -> None:
        stop_sglang_servers(self._procs)
        self._procs = []

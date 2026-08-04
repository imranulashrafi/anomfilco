from typing import List, Optional, Tuple

import numpy as np

from filco.base.semantic import centrality_from_embeddings


def _split_window(text: str, suffix_words: int, prefix_words: int):
    words = (text or "").split()
    if len(words) < suffix_words + prefix_words:
        return None, None
    suffix = " ".join(words[-suffix_words:])
    prefix = " ".join(words[-(suffix_words + prefix_words):-suffix_words])
    return prefix, suffix


def compute_sample_predinfo(
    model, tokenizer, prompt_text: str, samples: List[str],
    suffix_words: int, prefix_words: int, cfg,
) -> List[float]:
    from filco.training.losses import compute_mean_logprob

    out: List[float] = []
    for s in samples:
        prefix, suffix = _split_window(s, suffix_words, prefix_words)
        if prefix is None:
            out.append(float("nan"))
            continue
        a = compute_mean_logprob(model, tokenizer, prompt_text + " " + prefix, suffix, cfg)
        b = compute_mean_logprob(model, tokenizer, prompt_text, suffix, cfg)
        out.append(float(a - b) if (a is not None and b is not None) else float("nan"))
    return out


def _gate_survivors(predinfo: np.ndarray, gate_pct: float) -> np.ndarray:
    e = np.asarray(predinfo, dtype=np.float64)
    finite = e[~np.isnan(e)]
    if finite.size < 2:
        return np.ones(e.shape[0], dtype=bool)
    thresh = np.percentile(finite, gate_pct)
    keep = np.isnan(e) | (e <= thresh)
    if keep.sum() < 2:
        keep = np.ones(e.shape[0], dtype=bool)
    return keep


def select_pair(
    embeddings: np.ndarray,
    predinfo,
    gate_pct: float = 95.0,
    rejected_strategy: str = "lowest_centrality_all",
    min_pool: int = 4,
) -> Optional[Tuple[int, int]]:
    emb = np.asarray(embeddings, dtype=np.float64)
    n = emb.shape[0]
    if n < min_pool:
        return None
    e = np.asarray(predinfo, dtype=np.float64)

    keep = _gate_survivors(e, gate_pct)
    keep_idx = np.where(keep)[0]

    cen_surv = centrality_from_embeddings(emb[keep])
    chosen = int(keep_idx[int(np.argmax(cen_surv))])

    if rejected_strategy == "highest_predinfo":
        finite = np.where(~np.isnan(e))[0]
        if finite.size == 0:
            return None
        rejected = int(finite[int(np.argmax(e[finite]))])
    elif rejected_strategy == "lowest_centrality_survivor":
        rejected = int(keep_idx[int(np.argmin(cen_surv))])
    else:
        cen_all = centrality_from_embeddings(emb)
        rejected = int(np.argmin(cen_all))

    if chosen == rejected:
        return None
    return chosen, rejected

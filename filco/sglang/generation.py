import requests



def _parse_sglang_output(out, want_logprobs: bool) -> tuple:
    items = out if isinstance(out, list) else [out]
    texts, all_lps = [], []
    for it in items:
        texts.append(str(it.get("text", "")))
        if want_logprobs:
            meta = it.get("meta_info", {}) or {}
            lps_raw = meta.get("output_token_logprobs") or []
            lps = []
            for entry in lps_raw:
                if isinstance(entry, (list, tuple)) and len(entry) >= 1 and entry[0] is not None:
                    try:
                        lps.append(float(entry[0]))
                    except (TypeError, ValueError):
                        pass
            all_lps.append(lps)
        else:
            all_lps.append([])
    return texts, all_lps


def _generate_from_one_server(
    url: str, prompt: str, n: int, sampling: dict, want_logprobs: bool = False
) -> tuple:
    payload = {"text": prompt, "sampling_params": {**sampling, "n": n}}
    if want_logprobs:
        payload["return_logprob"] = True
        payload["logprob_start_len"] = 0

    for attempt_payload in [payload, {**payload, "text": [prompt] * n,
                                      "sampling_params": sampling}]:
        try:
            r = requests.post(url, json=attempt_payload, timeout=600)
            r.raise_for_status()
            texts, lps = _parse_sglang_output(r.json(), want_logprobs)
            if len(texts) == n:
                return texts, lps
        except Exception:
            pass

    texts: list = []
    all_lps: list = []
    single = {"text": prompt, "sampling_params": sampling}
    if want_logprobs:
        single["return_logprob"] = True
        single["logprob_start_len"] = 0
    for _ in range(n):
        try:
            r = requests.post(url, json=single, timeout=600)
            r.raise_for_status()
            t, lp = _parse_sglang_output(r.json(), want_logprobs)
            texts.append(t[0] if t else "")
            all_lps.append(lp[0] if lp else [])
        except Exception:
            texts.append("")
            all_lps.append([])
    return texts, all_lps



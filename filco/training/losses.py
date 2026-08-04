from typing import Optional

import torch
import torch.nn.functional as F

from filco.utils.distributed import get_raw_model


def compute_mean_logprob(
    model,
    tokenizer,
    prompt_text: str,
    completion: str,
    cfg,
) -> Optional[float]:
    if not completion.strip():
        return None

    raw_model = get_raw_model(model)
    device = next(raw_model.parameters()).device

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=4096
    ).input_ids[0]
    comp_ids = tokenizer(
        completion, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]

    prompt_len = prompt_ids.shape[0]
    max_total = prompt_len + cfg.max_new_tokens
    full_ids = torch.cat([prompt_ids, comp_ids])[:max_total]
    if full_ids.shape[0] <= prompt_len:
        return None

    labels = full_ids.clone()
    labels[:prompt_len] = -100

    input_ids = full_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    labels = labels.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = raw_model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_tok = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    n_comp_tokens = int((shift_labels != -100).sum().item())
    if n_comp_tokens == 0:
        return None

    return float(per_tok.sum().item() / n_comp_tokens)

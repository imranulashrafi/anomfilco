from typing import Optional

import torch
import torch.nn.functional as F

from filco.utils.distributed import get_raw_model


def _mean_logprob(raw_model, tokenizer, prompt_text: str, completion: str, cfg,
                  with_grad: bool):
    if not completion or not completion.strip():
        return None

    device = next(raw_model.parameters()).device
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=4096
    ).input_ids[0]
    comp_ids = tokenizer(
        completion, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]

    cap = getattr(cfg, "max_train_completion_tokens", 3072)
    if cap and comp_ids.shape[0] > cap:
        comp_ids = comp_ids[:cap]

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

    grad_ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with grad_ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = raw_model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_tok = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    n_comp = int((shift_labels != -100).sum().item())
    if n_comp == 0:
        return None
    return per_tok.sum() / n_comp


def simpo_loss(
    model, tokenizer, prompt_text: str, chosen: str, rejected: str, cfg,
    backward_scale: float = 1.0,
) -> Optional[dict]:
    if not chosen.strip() or not rejected.strip():
        return None

    raw_model = get_raw_model(model)
    beta = float(cfg.simpo_beta)
    gamma = float(cfg.simpo_gamma)
    scale = max(backward_scale, 1.0)

    sc0 = _mean_logprob(raw_model, tokenizer, prompt_text, chosen, cfg, with_grad=False)
    if sc0 is None:
        return None
    sr0 = _mean_logprob(raw_model, tokenizer, prompt_text, rejected, cfg, with_grad=False)
    if sr0 is None:
        return None

    z = beta * (sc0.float() - sr0.float()) - gamma
    loss = -F.logsigmoid(z)
    if not torch.isfinite(loss):
        return None
    coef = (torch.sigmoid(z) - 1.0).detach()

    sc_g = _mean_logprob(raw_model, tokenizer, prompt_text, chosen, cfg, with_grad=True)
    if sc_g is not None:
        (sc_g * (coef * beta / scale)).backward()

    sr_g = _mean_logprob(raw_model, tokenizer, prompt_text, rejected, cfg, with_grad=True)
    if sr_g is not None:
        (sr_g * (-coef * beta / scale)).backward()

    return {
        "loss": float(loss.detach().item()),
        "mean_lp_chosen": float(sc0.item()),
        "mean_lp_rejected": float(sr0.item()),
        "margin": float((sc0 - sr0).item()),
    }

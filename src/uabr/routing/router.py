"""Learned token routers (notebook 06). Loss ONLY at the decision position (all other labels ignored).

format "action"      : input  <BOS><USER> q <ASSISTANT>          -> 2-way CE over {<ANSWER>, <CALL>}
                       score = z_CALL - z_ANSWER ; UABR margin m = |z_CALL - z_ANSWER|        (PLAN §6)
format "answer_slot" : input  <BOS><USER> q <ASSISTANT><ANSWER>   -> full-vocab CE; target <CALL> for CALL
                       labels, and the model's OWN greedy first answer token for ANSWER labels (no new answers
                       are taught); score = logit P(<CALL>) ; m = |z_CALL - max_{v != CALL} z_v|
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from uabr.data import qa_format as Q
from uabr.data.tokenizer import ANSWER_ID, CALL_ID, EOS_ID, PAD_ID
from uabr.routing.metrics import auroc

FORMATS = ("action", "answer_slot")


def router_prompt(tok, question: str, fmt: str) -> List[int]:
    p = Q.closed_prompt_ids(tok, question)               # ... <ASSISTANT><ANSWER>
    return p[:-1] if fmt == "action" else p


def _batch(seqs: Sequence[List[int]], device):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    last = torch.tensor([len(s) - 1 for s in seqs])
    return x.to(device), last.to(device)


def _decision_logits(model, x, last, device):
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        logits, _ = model(x)
    return logits[torch.arange(len(last), device=x.device), last].float()


@torch.no_grad()
def router_scores(model, prompts: Sequence[List[int]], fmt: str, device: str, bs: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    model.eval(); S, M = [], []
    for i in range(0, len(prompts), bs):
        x, last = _batch(prompts[i:i + bs], device)
        z = _decision_logits(model, x, last, device)
        if fmt == "action":
            d = z[:, CALL_ID] - z[:, ANSWER_ID]; S.append(d.cpu().numpy()); M.append(d.abs().cpu().numpy())
        else:
            lp = F.log_softmax(z, -1)[:, CALL_ID]
            S.append((lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))).cpu().numpy())
            other = z.clone(); other[:, CALL_ID] = -1e9
            M.append((z[:, CALL_ID] - other.max(-1).values).abs().cpu().numpy())
    return np.concatenate(S), np.concatenate(M)


def _replay_batch(examples, idx, device):
    """Right-padded Q&A examples with targets only on answer tokens (flags)."""
    seqs = [examples[i] for i in idx]
    L = max(len(t) for t, _ in seqs)
    x = torch.full((len(seqs), L), PAD_ID, dtype=torch.long); y = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
    for i, (t, f) in enumerate(seqs):
        x[i, :len(t)] = torch.tensor(t)
        for j in range(1, len(t)):
            if f[j]:
                y[i, j - 1] = t[j]
    return x.to(device), y.to(device)


def train_router(model, train_prompts, train_targets, val_prompts, val_y, fmt: str, device: str,
                 lr: float = 3e-5, epochs: int = 3, bs: int = 32, eval_every: int = 50, seed: int = 0, log=print,
                 freeze_embed: bool = True, replay_examples=None, replay_bs: int = 16, replay_weight: float = 1.0,
                 qualify=None):
    """Returns (best, history). Loss = router CE at the decision position (+ replay_weight x answer-token LM loss on
    replay_examples, e.g. half-A Q&A theta0 was already trained on). freeze_embed keeps the tied embedding /
    output matrix fixed (moving <ANSWER>/<CALL> output vectors also moves them as INPUT tokens).
    qualify(model) -> (ok: bool, info: dict): only qualifying checkpoints can become `best` (e.g. answer EM kept)."""
    g = torch.Generator().manual_seed(seed)
    if freeze_embed:
        model.embed.weight.requires_grad_(False)          # tied with lm_head.weight
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    n = len(train_prompts); step = 0; hist = []
    best = {"auroc": -1.0, "step": -1, "state": None, "info": None}

    def evaluate():
        s, _ = router_scores(model, val_prompts, fmt, device)
        a = auroc(val_y, s)
        ok, info = qualify(model) if qualify else (True, {})
        hist.append({"step": step, "val_auroc": a, "qualifies": bool(ok), **info})
        if ok and a > best["auroc"]:
            best.update(auroc=a, step=step, state=copy.deepcopy(model.state_dict()), info=info)
        model.train(); return a, ok, info
    evaluate()
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(n, generator=g).split(bs):
            x, last = _batch([train_prompts[i] for i in b], device)
            tgt = torch.tensor([train_targets[i] for i in b], device=device)
            z = _decision_logits(model, x, last, device)
            loss = F.cross_entropy(z[:, [ANSWER_ID, CALL_ID]], tgt) if fmt == "action" else F.cross_entropy(z, tgt)
            if replay_examples:
                ridx = torch.randint(0, len(replay_examples), (replay_bs,), generator=g).tolist()
                rx, ry = _replay_batch(replay_examples, ridx, device)
                with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                    _, rl = model(rx[:, :-1], ry)
                loss = loss + replay_weight * rl
            opt.zero_grad(set_to_none=True)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            step += 1
            if step % eval_every == 0:
                a, ok, info = evaluate()
                log(f"  [{fmt}] epoch {ep+1} step {step} loss {loss.item():.4f} val AUROC {a:.4f} qualifies={ok} {info}")
    evaluate()
    if best["state"] is None:                               # nothing qualified: keep the best AUROC, flagged
        i = int(np.nanargmax([h["val_auroc"] for h in hist]))
        log(f"  [{fmt}] WARNING: no checkpoint met the answer-preservation rule; best-AUROC step {hist[i]['step']} kept (flagged)")
        best.update(auroc=hist[i]["val_auroc"], step=hist[i]["step"], state=copy.deepcopy(model.state_dict()), info={"unqualified": True})
    return best, hist


def answer_slot_targets(labels: Sequence[int], greedy_first_ids: Sequence) -> List[int]:
    return [CALL_ID if y else (int(t) if t is not None and t >= 0 else EOS_ID) for y, t in zip(labels, greedy_first_ids)]

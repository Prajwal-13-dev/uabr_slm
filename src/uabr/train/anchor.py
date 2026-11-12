"""Notebook 12 methods: decision-slot anchoring during SFT (A1 / A2) and post-SFT readout repair (B).

Motivation (notebook 11): after plain SFT a re-fitted linear probe reads the routing decision as well as before
(0.788 -> 0.79) but the pre-SFT probe falls to chance and the token readout drops (0.838 -> 0.70): the information
moves to a new direction and the <ANSWER>/<CALL> readout does not follow.

A1  SFT-prompt anchoring (no labels, no extra data): at the <ASSISTANT> position of every SFT example, keep the
    student's 2-way decision sigmoid(z_CALL - z_ANSWER) close to the frozen theta_route teacher's (soft BCE = KL + const).
A2  Question-pool anchoring (no labels): the same loss on random half-B question prompts (teacher scores precomputed);
    for the critic also on half-B (question, window) prompts at the critique slot (<IRR>, <REL>).
B   Readout repair after SFT: only the <ANSWER>/<CALL> embedding rows and the final RMSNorm gain train (2-logit CE
    on half-B labels) + SFT LM replay; the norm gain re-weights every hidden dimension, so the decision direction can
    rotate (a diagonal re-fit of the readout) while the body stays exactly the SFT body.
"""
from __future__ import annotations

import copy
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from uabr.data.tokenizer import ANSWER_ID, ASSISTANT_ID, CALL_ID
from uabr.routing.router import _batch, _replay_batch


def _diff(z, pair):
    return z[:, pair[1]].float() - z[:, pair[0]].float()


class SFTSlotAnchor:
    """A1: soft-BCE between student and teacher 2-way decision at the first slot token of each SFT row."""

    def __init__(self, teacher, pair=(ANSWER_ID, CALL_ID), slot_id: int = ASSISTANT_ID):
        self.t = teacher.eval(); self.pair = pair; self.slot = slot_id
        for p in self.t.parameters():
            p.requires_grad_(False)

    def __call__(self, step, model, x, logits):
        has = (x == self.slot).any(1)
        if not bool(has.any()):
            return None
        rows = has.nonzero(as_tuple=True)[0]
        pos = (x[rows] == self.slot).int().argmax(1)
        L = int(pos.max()) + 1
        with torch.no_grad():
            tz = self.t(x[rows, :L])[0]
        dt = _diff(tz[torch.arange(len(rows), device=x.device), pos], self.pair)
        ds = _diff(logits[rows, pos], self.pair)
        return F.binary_cross_entropy_with_logits(ds, torch.sigmoid(dt))


class PoolAnchor:
    """A2: soft-BCE on random prompts from fixed pools with precomputed teacher scores d_t = z[pair1] - z[pair0]."""

    def __init__(self, pools: Sequence[Tuple[List[List[int]], np.ndarray, Tuple[int, int]]], bs: int = 16, seed: int = 0):
        self.pools = [(p, torch.tensor(np.asarray(d), dtype=torch.float32), pair) for p, d, pair in pools]
        self.bs = bs; self.g = torch.Generator().manual_seed(seed)

    def __call__(self, step, model, x, logits):
        losses = []
        for prompts, dt, pair in self.pools:
            idx = torch.randint(0, len(prompts), (self.bs,), generator=self.g)
            xb, last = _batch([prompts[i] for i in idx.tolist()], x.device)
            z = model(xb)[0][torch.arange(len(idx), device=x.device), last]
            losses.append(F.binary_cross_entropy_with_logits(_diff(z, pair), torch.sigmoid(dt[idx].to(x.device))))
        return sum(losses) / len(losses)


def repair_readout(model, prompts, targets, device: str, sft_replay=None, lr: float = 1e-3, epochs: int = 3, bs: int = 32,
                   replay_bs: int = 8, eval_every: int = 25, seed: int = 0, evaluate: Optional[Callable] = None, log=print):
    """B: train only embedding rows <ANSWER>, <CALL> and the final norm gain. evaluate(model) -> (score, qualifies, info);
    the best qualifying score is kept (fallback: best score, flagged)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.embed.weight.requires_grad_(True); model.norm.weight.requires_grad_(True)
    keep = torch.zeros(model.embed.weight.shape[0], 1, device=model.embed.weight.device)
    keep[[ANSWER_ID, CALL_ID]] = 1.0
    h = model.embed.weight.register_hook(lambda g: g * keep)
    params = [model.embed.weight, model.norm.weight]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    n = len(prompts); step = 0; hist = []
    best = {"score": -1.0, "step": -1, "state": None, "info": None}; best_any = dict(best)

    def do_eval():
        s, ok, info = evaluate(model) if evaluate else (float("nan"), True, {})
        hist.append({"step": step, "score": s, "qualifies": bool(ok), **info})
        if ok and s > best["score"]:
            best.update(score=s, step=step, state=copy.deepcopy(model.state_dict()), info=info)
        if not ok and s > best_any["score"]:
            best_any.update(score=s, step=step, state=copy.deepcopy(model.state_dict()), info={**info, "unqualified": True})
        model.train()
    do_eval()
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(n, generator=g).split(bs):
            x, last = _batch([prompts[i] for i in b], device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                z = model(x)[0][torch.arange(len(b), device=x.device), last]
                loss = F.cross_entropy(z[:, [ANSWER_ID, CALL_ID]].float(), torch.tensor([targets[i] for i in b], device=x.device))
                if sft_replay:
                    ridx = torch.randint(0, len(sft_replay), (replay_bs,), generator=g).tolist()
                    rx, ry = _replay_batch(sft_replay, ridx, device)
                    _, rl = model(rx[:, :-1], ry)
                    loss = loss + rl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
            if step % eval_every == 0:
                do_eval(); log(f"  repair step {step} loss {loss.item():.4f} | {hist[-1]}")
    do_eval(); h.remove()
    for p in model.parameters():
        p.requires_grad_(True)
    if best["state"] is None:
        log(f"  WARNING: no repair checkpoint qualified; best-score step {best_any['step']} kept (flagged)"); best = best_any
    return best, hist

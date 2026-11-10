"""Instruction SFT shared by the Phase 5 pilot (notebook 07) and every Phase 6 arm.

Protocol (uabr-slm-main configs/sft_methods/standard.yaml, kept unchanged): ~6M instruction tokens, 1 epoch,
batch 32 sequences, AdamW (0.9, 0.95), weight decay 0.01 (matrices only), LR 5e-5 constant after 30 warm-up
steps, grad clip 1.0, fp16. The constant LR makes the 25 % / 50 % checkpoints of one run equal to separate
shorter runs. No checkpoint selection: every branch is compared at the same update count.

Every branch with the same seed sees exactly the same instruction batches in the same order; replay examples
(Phase 6) are inserted INSIDE those batches (add_replay), so all branches make the same number of updates.
Examples are (tokens, is-target flags) as in uabr.data.qa_format.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from uabr.data.tokenizer import PAD_ID

Example = Tuple[List[int], List[bool]]

STANDARD = {"batch_size": 32, "lr": 5e-5, "warmup_steps": 30, "weight_decay": 0.01, "betas": (0.9, 0.95),
            "grad_clip": 1.0, "precision": "fp16", "log_every_steps": 50}


def instruction_batches(examples: Sequence[Example], batch_size: int, seed: int) -> List[List[Example]]:
    """Seeded shuffle, then consecutive batches (the last one may be smaller)."""
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    ex = [examples[i] for i in order]
    return [ex[k:k + batch_size] for k in range(0, len(ex), batch_size)]


def add_replay(batches, replay_examples, seed: int):
    """Insert replay examples evenly: example j joins batch floor(j * n_batches / n_replay) (replay order shuffled
    with seed + 7919). Returns new batches; the input is not modified."""
    out = [list(b) for b in batches]
    rep = list(replay_examples)
    random.Random(seed + 7919).shuffle(rep)
    n_b, n_r = len(out), len(rep)
    for j, e in enumerate(rep):
        out[min(n_b - 1, j * n_b // n_r)].append(e)
    return out


def collate(batch: Sequence[Example], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """x = tokens[:-1] right-padded with <PAD>; y[t] = tokens[t+1] where that token is a target, else -100."""
    L = max(len(t) for t, _ in batch) - 1
    x = torch.full((len(batch), L), PAD_ID, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (t, f) in enumerate(batch):
        n = len(t) - 1
        x[i, :n] = torch.tensor(t[:-1])
        tt = torch.tensor(t[1:]); ff = torch.tensor(f[1:], dtype=torch.bool)
        y[i, :n] = torch.where(ff, tt, torch.full_like(tt, -100))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step: int, cfg: Dict) -> float:
    w = cfg["warmup_steps"]
    return cfg["lr"] * (step + 1) / w if step < w else cfg["lr"]


def count_tokens(examples: Sequence[Example]) -> Dict:
    return {"examples": len(examples), "tokens": int(sum(len(t) for t, _ in examples)),
            "supervised_tokens": int(sum(sum(f[1:]) for _, f in examples))}


@torch.no_grad()
def sft_val_loss(model, examples: Sequence[Example], device, batch_size: int = 32) -> float:
    """Mean per-token loss on held-out instruction responses (is SFT learning?)."""
    model.eval()
    tot, n = 0.0, 0
    for k in range(0, len(examples), batch_size):
        x, y = collate(examples[k:k + batch_size], device)
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            _, loss = model(x, y)
        m = int((y != -100).sum())
        tot += loss.item() * m; n += m
    return tot / max(n, 1)


def save_weights(model, path, model_cfg: Dict, extra: Dict):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    st = {"model": model.state_dict(), "model_cfg": model_cfg, **extra}
    tmp = path.with_suffix(".tmp"); torch.save(st, tmp); tmp.replace(path)
    return path


def train_sft(model, batches: Sequence[Sequence[Example]], cfg: Dict, seed: int, out_dir, save_at_steps=(),
              log=print) -> Dict:
    """One pass over pre-built batches. Writes step_{N}.pt at save_at_steps, final.pt, train_log.csv, summary.json."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    use_amp = device.type == "cuda" and cfg.get("precision", "fp16") == "fp16"
    for p in model.parameters():
        p.requires_grad_(True)                              # plain SFT: every weight trains (incl. embeddings)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=tuple(cfg["betas"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = len(batches); save_at = set(save_at_steps)
    mcfg = model.cfg.to_dict()
    tokens = sup = 0; t0 = time.time(); rows = []
    model.train()
    for step, batch in enumerate(batches):
        x, y = collate(batch, device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
        scaler.step(opt); scaler.update()
        done = step + 1
        tokens += int((x != PAD_ID).sum()) + len(batch); sup += int((y != -100).sum())
        if done % cfg.get("log_every_steps", 50) == 0 or done == total:
            rows.append({"step": done, "loss": round(loss.item(), 4), "lr": opt.param_groups[0]["lr"],
                         "grad_norm": round(gn, 3), "tokens": tokens, "supervised_tokens": sup,
                         "minutes": round((time.time() - t0) / 60, 2)})
            log(f"  step {done:>5}/{total} | loss {rows[-1]['loss']:.4f} | lr {rows[-1]['lr']:.1e} | grad {gn:.2f}")
        if done in save_at:
            save_weights(model, out / f"step_{done}.pt", mcfg, {"step": done, "total_steps": total, "seed": seed})
    save_weights(model, out / "final.pt", mcfg, {"step": total, "total_steps": total, "seed": seed})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "train_log.csv", index=False)
    summ = {"seed": seed, "updates": total, "save_at_steps": sorted(save_at), "tokens": tokens, "supervised_tokens": sup,
            "minutes": round((time.time() - t0) / 60, 2), "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    return summ

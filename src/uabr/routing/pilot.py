"""Phase 5 SFT-only pilot (notebook 07): decision-slot diagnostics, flips by margin (A4), the paired clustered
bootstrap and the GO / RECALIBRATION_ONLY / NO_CHANGE rule (uabr-slm-main docs/pilot_decision.md, fixed before
any result was seen; only the bootstrap is subject-clustered here, amendment A9).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uabr.data.tokenizer import ANSWER_ID, CALL_ID
from uabr.routing.metrics import auroc
from uabr.routing.router import _batch, _decision_logits


@torch.no_grad()
def decision_slot(model, prompts: Sequence[List[int]], device: str, bs: int = 64) -> pd.DataFrame:
    """At <ASSISTANT>: score = z_CALL - z_ANSWER (the policy), margin |score| (UABR m_i), and A5 collapse
    diagnostics over the FULL vocabulary: P(<ANSWER>) + P(<CALL>), whether argmax is an action token, and the
    best rank of an action token."""
    model.eval(); rec = []
    for i in range(0, len(prompts), bs):
        x, last = _batch(prompts[i:i + bs], device)
        z = _decision_logits(model, x, last, device)
        p = F.softmax(z, -1)
        d = z[:, CALL_ID] - z[:, ANSWER_ID]
        best_action = torch.maximum(z[:, CALL_ID], z[:, ANSWER_ID])
        rank = (z > best_action[:, None]).sum(-1)                  # 0 = an action token is the argmax
        for j in range(len(d)):
            rec.append({"score": float(d[j]), "margin": float(d[j].abs()), "p_action": float(p[j, ANSWER_ID] + p[j, CALL_ID]),
                        "p_call_2way": float(torch.sigmoid(d[j])), "argmax_is_action": bool(rank[j] == 0),
                        "action_rank": int(rank[j])})
    return pd.DataFrame(rec)


def top_fraction(score, frac: float, seed: int = 0) -> np.ndarray:
    """Boolean CALL vector for the top `frac` of scores (ties broken by a fixed tiny jitter)."""
    s = np.asarray(score, float)
    k = int(round(frac * len(s)))
    call = np.zeros(len(s), bool)
    if k:
        j = np.random.default_rng(seed).random(len(s)) * 1e-9
        call[np.argsort(-(s + j))[:k]] = True
    return call


def outcome(call, r_answer, r_call, y_benefit=None) -> Dict:
    call = np.asarray(call, bool); ra = np.asarray(r_answer, float); rc = np.asarray(r_call, float)
    r = np.where(call, rc, ra)
    harmful = (~call) & (ra == 0) & (rc == 1)                  # answered wrongly where evidence would have been right
    return {"accuracy": float(r.mean()), "call_rate": float(call.mean()), "harmful_answer_rate": float(harmful.mean())}


def paired_cluster_bootstrap(fn: Callable[[np.ndarray, str], float], clusters, n_boot: int = 1000, seed: int = 0) -> Dict:
    """fn(idx, which) -> metric on rows idx of checkpoint `which` ("a" = later checkpoint, "b" = reference).
    Resamples whole subject clusters; both checkpoints use the SAME resampled rows. Returns diff = a - b."""
    c = np.asarray(clusters)
    uniq, inv = np.unique(c, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    full = np.arange(len(c))
    diff = fn(full, "a") - fn(full, "b")
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        v = fn(idx, "a") - fn(idx, "b")
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)
    return {"diff": float(diff), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot_valid": len(vals)}


def decide(per_seed: Dict[str, Dict]) -> str:
    """per_seed[seed]['significant_drop'] has keys auroc, acc_budget_10, acc_frozen_thr_10 (bool: CI upper < 0).
    GO: in every seed AUROC or re-picked acc@10% drops. RECALIBRATION_ONLY: not GO, but in every seed GO-level
    damage or at least the frozen-threshold acc@10% drops. NO_CHANGE: otherwise."""
    if not per_seed:
        return "NO_CHANGE"
    go = [s["significant_drop"]["auroc"] or s["significant_drop"]["acc_budget_10"] for s in per_seed.values()]
    rec = [g or s["significant_drop"]["acc_frozen_thr_10"] for g, s in zip(go, per_seed.values())]
    if all(go):
        return "GO"
    if all(rec):
        return "RECALIBRATION_ONLY"
    return "NO_CHANGE"


def flips_by_margin(score0, score1, margin0, clusters, boundary0: float = 0.0, boundary1: float = 0.0,
                    n_bins: int = 5) -> Dict:
    """A4: does a small pre-SFT margin predict a decision flip after SFT?
    Decision = CALL iff score > boundary. Margin bins are quantiles of the pre-SFT margin (bin 1 = smallest).
    Returns the per-bin table, flip directions and AUROC(-margin -> flipped)."""
    s0, s1, m = (np.asarray(a, float) for a in (score0, score1, margin0))
    d0, d1 = s0 > boundary0, s1 > boundary1
    flip = d0 != d1
    edges = np.unique(np.quantile(m, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, len(edges) - 2)
    tab = pd.DataFrame({"bin": b + 1, "margin": m, "flip": flip, "a2c": (~d0) & d1, "c2a": d0 & (~d1)}) \
        .groupby("bin").agg(n=("flip", "size"), margin_lo=("margin", "min"), margin_hi=("margin", "max"),
                            flip_rate=("flip", "mean"), answer_to_call=("a2c", "sum"), call_to_answer=("c2a", "sum")).reset_index()
    from uabr.routing.metrics import clustered_bootstrap_auroc
    a = clustered_bootstrap_auroc(flip.astype(int), -m, clusters, n_boot=1000) if 0 < flip.sum() < len(flip) else \
        {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {"table": tab, "flip_rate": float(flip.mean()), "answer_to_call": int(((~d0) & d1).sum()),
            "call_to_answer": int((d0 & (~d1)).sum()), "auroc_small_margin_predicts_flip": a}

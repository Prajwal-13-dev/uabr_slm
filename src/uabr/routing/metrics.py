"""Routing metrics: AUROC / AUPRC (CALL = positive), utility at a fixed call budget, clustered bootstrap."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(np.unique(y[ok])) == 2 else float("nan")


def auprc(y, s) -> float:
    from sklearn.metrics import average_precision_score
    y, s = np.asarray(y), np.asarray(s, dtype=float); ok = np.isfinite(s)
    return float(average_precision_score(y[ok], s[ok])) if len(np.unique(y[ok])) == 2 else float("nan")


def utility_at_budget(score, r_answer, r_call, lam: float, budget: float, seed: int = 0) -> Dict:
    """CALL the top `budget` fraction by score (ties broken at random, fixed seed); R = realised correctness."""
    s = np.asarray(score, dtype=float).copy()
    s[~np.isfinite(s)] = -np.inf
    n = len(s); k = int(round(budget * n))
    jitter = np.random.default_rng(seed).random(n) * 1e-9
    call = np.zeros(n, bool)
    if k > 0:
        call[np.argsort(-(s + jitter))[:k]] = True
    r = np.where(call, r_call, r_answer)
    return {"utility": float(r.mean() - lam * call.mean()), "accuracy": float(r.mean()), "call_rate": float(call.mean())}


def oracle_utility_at_budget(r_answer, r_call, lam: float, budget: float) -> float:
    gain = np.asarray(r_call, float) - np.asarray(r_answer, float)
    return utility_at_budget(gain, r_answer, r_call, lam, budget)["utility"]


def clustered_bootstrap_auroc(y, s, clusters, n_boot: int = 1000, seed: int = 0) -> Dict:
    y, s, c = np.asarray(y), np.asarray(s, float), np.asarray(clusters)
    uniq, inv = np.unique(c, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        v = auroc(y[idx], s[idx])
        if np.isfinite(v): vals.append(v)
    lo, hi = (np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan))
    return {"auroc": auroc(y, s), "lo": float(lo), "hi": float(hi)}

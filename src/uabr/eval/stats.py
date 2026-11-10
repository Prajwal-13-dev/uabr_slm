"""Subject-clustered paired bootstrap (amendment A9)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def clustered_bootstrap_mean(values: Sequence[float], clusters: Sequence, n_boot: int = 2000,
                             seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Mean of per-item values (e.g. paired differences) with a CI that resamples whole clusters."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    uniq, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(v)), "clusters": int(len(uniq))}


def clustered_bootstrap_diff(values: Sequence[float], clusters: Sequence, in_a: Sequence[bool], in_b: Sequence[bool],
                             n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """mean(values | A) - mean(values | B), CI from resampling whole clusters (groups may share clusters)."""
    v = np.asarray(values, dtype=float); a = np.asarray(in_a, bool); b = np.asarray(in_b, bool)
    if a.sum() == 0 or b.sum() == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_a": int(a.sum()), "n_b": int(b.sum())}
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = len(uniq)
    sa, na = np.bincount(inv, v * a, k), np.bincount(inv, a.astype(float), k)
    sb, nb = np.bincount(inv, v * b, k), np.bincount(inv, b.astype(float), k)
    idx = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sa[idx].sum(1) / na[idx].sum(1) - sb[idx].sum(1) / nb[idx].sum(1)
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return {"diff": float(v[a].mean() - v[b].mean()), "lo": float("nan"), "hi": float("nan"), "n_a": int(a.sum()), "n_b": int(b.sum())}
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(v[a].mean() - v[b].mean()), "lo": float(lo), "hi": float(hi),
            "n_a": int(a.sum()), "n_b": int(b.sum())}

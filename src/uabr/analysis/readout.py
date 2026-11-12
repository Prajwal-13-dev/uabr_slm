"""Post-hoc analyses for notebook 11 (no training): readout vs representation, weight interpolation /
task-vector re-addition, threshold recalibration.

Readout framing: the hidden state h at the decision position is post-RMSNorm and the embeddings are tied, so the
token score z_CALL - z_ANSWER = h . (e_CALL - e_ANSWER) is itself a FIXED linear readout of h. A re-fitted linear
probe on the same h asks whether the information is still there after SFT and only the readout direction broke.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uabr.routing.metrics import auroc


def fit_probe(X, y, C: float = 0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, np.asarray(y).astype(int))


def probe_scores(pipe, X) -> np.ndarray:
    return pipe.decision_function(X)


def learning_curve(Xtr, ytr, Xte, yte, ns: Iterable[int], reps: int = 5, seed: int = 0, C: float = 0.1) -> pd.DataFrame:
    """AUROC on the test set of probes fitted on n random training examples (both classes required)."""
    ytr = np.asarray(ytr).astype(int); rng = np.random.default_rng(seed); rows = []
    for n in ns:
        n = min(int(n), len(ytr))
        for r in range(reps if n < len(ytr) else 1):
            for _ in range(100):
                idx = rng.choice(len(ytr), n, replace=False)
                if 0 < ytr[idx].sum() < n:
                    break
            p = fit_probe(Xtr[idx], ytr[idx], C)
            rows.append({"n_labels": n, "rep": r, "auroc": auroc(yte, probe_scores(p, Xte))})
    return pd.DataFrame(rows)


def interpolate(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    """(1 - alpha) * a + alpha * b for floating tensors (alpha = 0 -> a, 1 -> b); others copied from a."""
    assert sd_a.keys() == sd_b.keys()
    out = {}
    for k, va in sd_a.items():
        vb = sd_b[k]
        assert va.shape == vb.shape, k
        out[k] = (va + alpha * (vb - va)) if va.is_floating_point() else va.clone()
    return out


def task_vector_add(sd_base: Dict[str, torch.Tensor], sd_plus: Dict[str, torch.Tensor], sd_minus: Dict[str, torch.Tensor],
                    beta: float) -> Dict[str, torch.Tensor]:
    """base + beta * (plus - minus) for floating tensors with matching shapes; mismatched tensors keep base."""
    out = {}
    for k, v in sd_base.items():
        if v.is_floating_point() and k in sd_plus and k in sd_minus and sd_plus[k].shape == v.shape == sd_minus[k].shape:
            out[k] = v + beta * (sd_plus[k] - sd_minus[k])
        else:
            out[k] = v.clone()
    return out


def fit_threshold(score, y) -> float:
    """Threshold t (decision = score > t) maximising balanced accuracy against y on the calibration set."""
    s = np.asarray(score, float); y = np.asarray(y).astype(int)
    cand = np.unique(np.concatenate([[-np.inf], s]))
    best, bt = -1.0, 0.0
    P, N = max(y.sum(), 1), max((1 - y).sum(), 1)
    order = np.argsort(s); ss, yy = s[order], y[order]
    # decision = s > t: for t = ss[i], positives predicted are those with index > last index of value ss[i]
    tp_above = np.concatenate([np.cumsum(yy[::-1])[::-1], [0]])          # positives with index >= i
    for t in cand:
        i = np.searchsorted(ss, t, side="right")
        tp = tp_above[i]; fp = (len(ss) - i) - tp
        ba = 0.5 * (tp / P + (N - fp) / N)
        if ba > best:
            best, bt = ba, t
    return float(bt)


def balanced_accuracy(decision, y) -> float:
    d = np.asarray(decision, bool); y = np.asarray(y).astype(bool)
    return 0.5 * (d[y].mean() + (~d[~y]).mean())

import numpy as np
import torch

from uabr.analysis.readout import (balanced_accuracy, fit_probe, fit_threshold, interpolate, learning_curve,
                                   probe_scores, task_vector_add)
from uabr.routing.metrics import auroc


def test_probe_and_learning_curve():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8)); y = (X[:, 0] + 0.3 * rng.normal(size=600) > 0.8).astype(int)
    p = fit_probe(X[:400], y[:400])
    assert auroc(y[400:], probe_scores(p, X[400:])) > 0.9
    lc = learning_curve(X[:400], y[:400], X[400:], y[400:], [20, 100, 400], reps=3)
    assert set(lc.n_labels) == {20, 100, 400} and (lc.n_labels == 400).sum() == 1
    assert lc.groupby("n_labels").auroc.mean().loc[400] > 0.9


def test_interpolate_and_task_vector():
    a = {"w": torch.zeros(2, 2), "n": torch.tensor([1])}; b = {"w": torch.ones(2, 2), "n": torch.tensor([5])}
    m = interpolate(a, b, 0.25)
    assert torch.allclose(m["w"], torch.full((2, 2), 0.25)) and m["n"].item() == 1
    assert torch.equal(interpolate(a, b, 0.0)["w"], a["w"]) and torch.equal(interpolate(a, b, 1.0)["w"], b["w"])
    base = {"w": torch.ones(2), "e": torch.ones(3)}; plus = {"w": torch.full((2,), 3.0), "e": torch.ones(4)}; minus = {"w": torch.ones(2), "e": torch.ones(4)}
    t = task_vector_add(base, plus, minus, 0.5)
    assert torch.allclose(t["w"], torch.full((2,), 2.0)) and torch.equal(t["e"], base["e"])


def test_fit_threshold_balanced_accuracy():
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9]); y = np.array([0, 0, 0, 1, 1, 1])
    t = fit_threshold(s, y)
    assert 0.3 <= t < 0.6 and balanced_accuracy(s > t, y) == 1.0
    s2 = s + 5
    assert balanced_accuracy(s2 > fit_threshold(s2, y), y) == 1.0
    rng = np.random.default_rng(1); s3 = rng.normal(size=300); y3 = (s3 + rng.normal(size=300) > 1).astype(int)
    t3 = fit_threshold(s3, y3)
    grid = max(balanced_accuracy(s3 > u, y3) for u in np.unique(s3))
    assert abs(balanced_accuracy(s3 > t3, y3) - grid) < 1e-12

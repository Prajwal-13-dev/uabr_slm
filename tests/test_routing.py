import numpy as np
import torch
from uabr.data import tokenizer as T
from uabr.model.llama import LlamaLM, ModelConfig
from uabr.routing.benefit import choose_lambda, greedy_features
from uabr.routing.metrics import auroc, clustered_bootstrap_auroc, oracle_utility_at_budget, utility_at_budget
from uabr.routing.router import answer_slot_targets, router_scores, train_router


def test_choose_lambda():
    b = np.array([0.5, 0.3, 0.1, 0.0, -0.2, 0.05])
    d = np.array([1, 1, 0, 0, -1, 0])
    r = choose_lambda(b, d, [0, 0.0625, 0.125, 0.25])
    assert r["chosen"]["lambda"] in (0.125, 0.25) and r["chosen"]["balanced_acc"] == 1.0


def test_utility_and_oracle():
    ra, rc = np.array([1, 0, 0, 1]), np.array([1, 1, 0, 0])
    s = np.array([0.0, 5.0, 1.0, -1.0])
    u = utility_at_budget(s, ra, rc, lam=0.1, budget=0.25)
    assert u["call_rate"] == 0.25 and abs(u["accuracy"] - 0.75) < 1e-9 and abs(u["utility"] - 0.725) < 1e-9
    assert oracle_utility_at_budget(ra, rc, 0.1, 0.25) == u["utility"]


def test_auroc_bootstrap():
    y = np.r_[np.ones(50), np.zeros(50)]; s = np.r_[np.ones(50), np.zeros(50)] + np.random.default_rng(0).normal(0, .1, 100)
    r = clustered_bootstrap_auroc(y, s, np.arange(100), n_boot=200)
    assert r["auroc"] > 0.99 and r["lo"] > 0.95


def _toy():
    torch.manual_seed(0)
    return LlamaLM(ModelConfig(vocab_size=60, n_layers=1, d_model=32, n_heads=4, d_ff=64, max_seq_len=32))


def test_router_learns_both_formats():
    # label = CALL iff token 20 appears in the prompt (learnable signal)
    rng = np.random.default_rng(0)
    prompts, y = [], []
    for i in range(240):
        body = list(rng.integers(10, 40, 6)); lab = int(rng.random() < 0.5)
        if lab: body[2] = 50
        prompts.append([T.BOS_ID, T.USER_ID] + [int(t) for t in body] + [T.ASSISTANT_ID]); y.append(lab)
    tr, va = slice(0, 180), slice(180, 240)
    m = _toy(); best, hist = train_router(m, prompts[tr], y[tr], prompts[va], y[va], "action", "cpu", lr=3e-3, epochs=6, bs=16, eval_every=1000, log=lambda *a: None)
    m.load_state_dict(best["state"]); s, mg = router_scores(m, prompts[va], "action", "cpu")
    assert auroc(y[va], s) > 0.9 and (mg >= 0).all()
    p2 = [p + [T.ANSWER_ID] for p in prompts]
    tg = answer_slot_targets(y, [11] * len(y))
    m = _toy(); best, _ = train_router(m, p2[tr], tg[tr], p2[va], y[va], "answer_slot", "cpu", lr=3e-3, epochs=6, bs=16, eval_every=1000, log=lambda *a: None)
    m.load_state_dict(best["state"]); s, _ = router_scores(m, p2[va], "answer_slot", "cpu")
    assert auroc(y[va], s) > 0.9


def test_greedy_features_shapes():
    m = _toy().eval()
    f = greedy_features(m, [1, 3, 12, 13, 4, 5], [20, 21], "cpu")
    assert f["conf_mean_lp"] < 0 and 0 <= f["conf_first_p"] <= 1 and f["h_assistant"].shape == (32,)
    f2 = greedy_features(m, [1, 3, 12, 13, 4, 5], [], "cpu")
    assert np.isnan(f2["conf_mean_lp"])

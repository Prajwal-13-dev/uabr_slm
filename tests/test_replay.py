import numpy as np
import pandas as pd
import torch

from uabr.data import tokenizer as T
from uabr.model.llama import LlamaLM, ModelConfig
from uabr.replay.selectors import Cycler, ReplaySchedule, class_budgets, jaccard, priority, select_by_class
from uabr.routing.router import router_scores
from uabr.train.sft import instruction_batches, replay_loss, train_sft


def _pool(n=40, seed=0):
    rng = np.random.default_rng(seed)
    toks = rng.integers(5, 12, n)
    return pd.DataFrame({"id": np.arange(n), "y": (np.arange(n) % 4 == 0).astype(int), "tokens": toks,
                         "prompt": [[T.BOS_ID, T.USER_ID] + [20 + (i % 10)] * (t - 3) + [T.ASSISTANT_ID] for i, t in enumerate(toks)],
                         "margin": rng.random(n) * 3, "b_net": rng.normal(0, 0.3, n), "unc": rng.random(n)})


def test_priorities():
    p = _pool()
    assert np.array_equal(np.argsort(-priority(p, "bpr")), np.argsort(p.margin.values))
    assert np.allclose(priority(p, "benefit"), p.b_net.abs())
    tau = float(np.median(p.margin))
    assert np.allclose(priority(p, "uabr"), np.exp(-p.margin / tau) * p.b_net.abs())
    assert np.allclose(priority(p, "casr"), p.unc)
    assert np.allclose(priority(p, "random", 3), priority(p, "random", 3)) and not np.allclose(priority(p, "random", 3), priority(p, "random", 4))


def test_select_by_class_respects_ratio_and_budget():
    p = _pool(200)
    sel_tok = int(0.25 * p.tokens.sum())
    b = class_budgets(p, sel_tok)
    share = p.loc[p.y == 1, "tokens"].sum() / p.tokens.sum()
    assert abs(b[1] - sel_tok * share) <= 1 and abs(sum(b.values()) - sel_tok) <= 1
    s = select_by_class(p, priority(p, "bpr"), b)
    for c in (0, 1):
        g = s[s.y == c]
        assert g.tokens.sum() <= b[c]
        rest = p[(p.y == c) & ~p.id.isin(g.id)]
        assert g.margin.max() <= rest.margin.min()            # BPR took the smallest margins of that class
    assert s.id.is_unique


def test_schedule_equal_tokens():
    p = _pool(100)
    sel = select_by_class(p, priority(p, "random", 1), class_budgets(p, 200))
    for rt in (1000, 2345):
        sch = ReplaySchedule(rt, 37); cyc = Cycler(sel, 11)
        for t in range(37):
            sch.take(t, cyc)
        r = sch.report()
        assert rt <= r["replay_tokens"] < rt + p.tokens.max() and r["distinct_prompts"] == len(sel)
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5


def _toy():
    torch.manual_seed(0)
    return LlamaLM(ModelConfig(vocab_size=60, n_layers=1, d_model=32, n_heads=4, d_ff=64, max_seq_len=32))


def test_train_sft_with_replay_moves_decision(tmp_path):
    m = _toy()
    ex = [([1, 3, 20, 21, 4, 40, 41, 2], [False] * 5 + [True] * 3) for _ in range(64)]
    prompts = [[T.BOS_ID, T.USER_ID, 30, T.ASSISTANT_ID]] * 4
    s0, _ = router_scores(m, prompts[:1], "action", "cpu")
    cfg = {"batch_size": 8, "lr": 3e-3, "warmup_steps": 1, "weight_decay": 0.0, "betas": (0.9, 0.95), "grad_clip": 1.0,
           "precision": "fp32", "log_every_steps": 4}
    calls = []
    def replay(step, model):
        calls.append(step); return prompts, [1, 1, 1, 1]          # always CALL
    s = train_sft(m, instruction_batches(ex, 8, 11), cfg, 11, tmp_path, replay=replay, beta=1.0, log=lambda *a: None)
    s1, _ = router_scores(m, prompts[:1], "action", "cpu")
    assert calls == list(range(8)) and s["beta"] == 1.0 and s1[0] > s0[0] + 1.0
    l = replay_loss(m, prompts, [1, 1, 1, 1], "cpu")
    assert float(l) < 0.5
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert log.replay_loss.notna().all()

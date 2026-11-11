import numpy as np
import torch

from uabr.critic.selfrag import add_critique_tokens, critique_ids, critique_scores, freeze_old_rows, last_hidden, train_critic
from uabr.data import tokenizer as T
from uabr.model.llama import LlamaLM, ModelConfig
from uabr.routing.metrics import auroc


def _toy():
    torch.manual_seed(0)
    return LlamaLM(ModelConfig(vocab_size=60, n_layers=1, d_model=32, n_heads=4, d_ff=64, max_seq_len=64))


def test_add_tokens_keeps_old_model():
    m = _toy().eval(); m2 = add_critique_tokens(m)
    assert m2.cfg.vocab_size == 62 and critique_ids(m2) == (60, 61)
    x = torch.tensor([[1, 3, 20, 21, 4, 6, 7, 30, 8]])
    with torch.no_grad():
        a = m(x)[0][0, :, :60]; b = m2(x)[0][0, :, :60]
    assert torch.allclose(a, b, atol=1e-5)                   # old logits unchanged
    assert m2.lm_head.weight.data_ptr() == m2.embed.weight.data_ptr()


def test_freeze_old_rows_and_train_critic_learns():
    m = add_critique_tokens(_toy())
    old = m.embed.weight[:60].detach().clone()
    # relevant iff token 40 appears in the "window"
    P, Y = [], []
    rng = np.random.default_rng(0)
    for i in range(128):
        rel = i % 2
        w = list(rng.integers(20, 39, 6)) + ([40] if rel else [39])
        rng.shuffle(w)
        P.append([1, 3, 20, 4, 6, 7] + [int(t) for t in w] + [8]); Y.append(rel)
    s0 = critique_scores(m, P, "cpu")
    qa = [([1, 3, 25, 4, 5, 30, 2], [False] * 5 + [True] * 2)]
    rr = ([[1, 3, 22, 4]], [1])
    ev = lambda mm: (auroc(Y, critique_scores(mm, P, "cpu")), True, {})
    best, hist = train_critic(m, P, Y, "cpu", n_old=60, lr=3e-3, epochs=6, bs=16, eval_every=8, qa_replay=qa, router_replay=rr, evaluate=ev, log=lambda *a: None)
    m.load_state_dict(best["state"])
    assert torch.equal(m.embed.weight[:60], old)            # old rows frozen
    assert best["score"] > 0.9 > auroc(Y, s0) and len(hist) > 2
    h = last_hidden(m, P[:5], "cpu")
    assert h.shape == (5, 32)


def test_train_critic_fallback_when_nothing_qualifies():
    m = add_critique_tokens(_toy())
    P = [[1, 3, 20, 4, 6, 7, 30 + i % 3, 8] for i in range(32)]; Y = [i % 2 for i in range(32)]
    ev = lambda mm: (auroc(Y, critique_scores(mm, P, "cpu")), False, {})
    best, _ = train_critic(m, P, Y, "cpu", n_old=60, lr=1e-3, epochs=1, bs=8, eval_every=2, evaluate=ev, log=lambda *a: None)
    assert best["info"]["unqualified"] and best["state"] is not None

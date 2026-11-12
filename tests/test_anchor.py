import numpy as np
import torch

from uabr.data import tokenizer as T
from uabr.model.llama import LlamaLM, ModelConfig
from uabr.routing.metrics import auroc
from uabr.routing.router import router_scores
from uabr.train.anchor import PoolAnchor, SFTSlotAnchor, repair_readout
from uabr.train.sft import instruction_batches, train_sft


def _toy(seed=0):
    torch.manual_seed(seed)
    return LlamaLM(ModelConfig(vocab_size=60, n_layers=1, d_model=32, n_heads=4, d_ff=64, max_seq_len=64))


CFG = {"batch_size": 8, "lr": 3e-3, "warmup_steps": 1, "weight_decay": 0.0, "betas": (0.9, 0.95), "grad_clip": 1.0,
       "precision": "fp32", "log_every_steps": 4}
# SFT data that pushes the token after <ASSISTANT> toward <CALL>-like token 6 being UNlikely: responses start with token 40
EX = [([T.BOS_ID, T.USER_ID, 20 + i % 5, T.ASSISTANT_ID, 40, 41, T.EOS_ID], [False] * 4 + [True] * 3) for i in range(64)]
PROMPTS = [[T.BOS_ID, T.USER_ID, 20 + i % 5, T.ASSISTANT_ID] for i in range(10)]


def _drift(model, teacher):
    a, _ = router_scores(model, PROMPTS, "action", "cpu"); b, _ = router_scores(teacher, PROMPTS, "action", "cpu")
    return float(np.abs(a - b).mean())


def test_sft_slot_anchor_reduces_drift(tmp_path):
    t = _toy(); plain = _toy(); anch = _toy()
    train_sft(plain, instruction_batches(EX, 8, 11), CFG, 11, tmp_path / "p", log=lambda *a: None)
    s = train_sft(anch, instruction_batches(EX, 8, 11), CFG, 11, tmp_path / "a", log=lambda *a: None, aux=SFTSlotAnchor(t), gamma=5.0)
    assert s["gamma"] == 5.0 and _drift(anch, t) < _drift(plain, t)


def test_pool_anchor_reduces_drift(tmp_path):
    t = _toy(); plain = _toy(); anch = _toy()
    d, _ = router_scores(t, PROMPTS, "action", "cpu")
    train_sft(plain, instruction_batches(EX, 8, 11), CFG, 11, tmp_path / "p", log=lambda *a: None)
    train_sft(anch, instruction_batches(EX, 8, 11), CFG, 11, tmp_path / "a", log=lambda *a: None,
              aux=PoolAnchor([(PROMPTS, d, (T.ANSWER_ID, T.CALL_ID))], bs=4), gamma=5.0)
    assert _drift(anch, t) < _drift(plain, t)


def test_repair_readout_only_touches_two_rows_and_norm():
    m = _toy()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    P = [[T.BOS_ID, T.USER_ID, 20 + i % 10, T.ASSISTANT_ID] for i in range(64)]; Y = [int(i % 10 >= 5) for i in range(64)]
    ev = lambda mm: (auroc(Y, router_scores(mm, P, "action", "cpu")[0]), True, {})
    best, hist = repair_readout(m, P, Y, "cpu", sft_replay=EX, lr=3e-2, epochs=8, bs=16, eval_every=4, evaluate=ev, log=lambda *a: None)
    m.load_state_dict(best["state"])
    after = m.state_dict()
    for k in before:
        if k in ("embed.weight", "lm_head.weight"):
            changed = (before[k] != after[k]).any(1).nonzero().flatten().tolist()
            assert set(changed) <= {T.ANSWER_ID, T.CALL_ID}
        elif k == "norm.weight":
            continue
        else:
            assert torch.equal(before[k], after[k]), k
    assert best["score"] > hist[0]["score"] and all(p.requires_grad for p in m.parameters())

import numpy as np
import torch

from uabr.data import tokenizer as T
from uabr.data.sft_data import audit_overlap, build_sft, prompt_body, rows_from_hf, sft_example
from uabr.eval.batched import greedy_batch
from uabr.model.llama import LlamaLM, ModelConfig
from uabr.routing.pilot import decide, decision_slot, flips_by_margin, outcome, paired_cluster_bootstrap, top_fraction
from uabr.train.sft import add_replay, collate, count_tokens, instruction_batches, lr_at, sft_val_loss, train_sft


class Tok:
    """Minimal stand-in: one id per character (ids 20+)."""
    def encode(self, s, add_special_tokens=False):
        class E: pass
        e = E(); e.ids = [20 + (ord(c) % 30) for c in s]; return e


def _toy(vocab=60, seq=64):
    torch.manual_seed(0)
    return LlamaLM(ModelConfig(vocab_size=vocab, n_layers=1, d_model=32, n_heads=4, d_ff=64, max_seq_len=seq))


def test_rows_and_format():
    dolly = [{"instruction": "Say hi", "context": "", "response": "hi"}, {"instruction": " ", "context": "", "response": "x"}]
    alp = [{"instruction": "Add", "input": "1 and 2", "output": "3"}]
    rows = rows_from_hf(dolly, alp)
    assert [r["id"] for r in rows] == ["dolly-0", "alpaca-0"] and rows[1]["context"] == "1 and 2"
    assert prompt_body("Add", "1 and 2") == "Add\n\n1 and 2"
    t, f = sft_example(Tok(), "ab", "", "cd")
    assert t[:2] == [T.BOS_ID, T.USER_ID] and t[-1] == T.EOS_ID and T.ANSWER_ID not in t and T.CALL_ID not in t
    a = t.index(T.ASSISTANT_ID)
    assert not any(f[:a + 1]) and all(f[a + 1:])          # loss only on " cd<EOS>"


def test_build_sft_budget_and_val():
    rows = [{"id": f"r{i}", "source": "dolly15k" if i % 2 else "alpaca-cleaned", "instruction": "q" * (i % 7 + 1),
             "context": "", "response": "a" * (i % 5 + 1)} for i in range(200)]
    rows.append({"id": "long", "source": "dolly15k", "instruction": "x" * 600, "context": "", "response": "y"})
    tr, va, tr_ids, va_ids, st = build_sft(Tok(), rows, max_len=64, budget_tokens=500, n_val=20, seed=1)
    assert st["dropped_too_long"] == 1 and len(va) == 20 and not set(tr_ids) & set(va_ids)
    assert st["train_tokens"] >= 500 and st["train_tokens"] - len(tr[-1][0]) < 500
    tr2, *_ = build_sft(Tok(), rows, max_len=64, budget_tokens=500, n_val=20, seed=1)
    assert tr2 == tr                                          # deterministic


def test_audit_overlap():
    texts = ["Barack Obama was born in Honolulu, Hawaii.", "Obama likes Honolulu", "The capital of France is Paris."]
    facts = [(1, "Barack Obama", ["Honolulu"]), (2, "France", ["Paris"]), (3, "Germany", ["Berlin"]), (4, "France", ["Lyon"])]
    hits = audit_overlap(texts, facts)
    assert hits == {0: [1], 2: [2]}


def test_batches_replay_collate():
    ex = [([1, 3, 20 + i, 4, 30, 2], [False] * 4 + [True] * 2) for i in range(10)]
    b1, b2 = instruction_batches(ex, 4, 11), instruction_batches(ex, 4, 11)
    assert b1 == b2 and [len(b) for b in b1] == [4, 4, 2] and instruction_batches(ex, 4, 29) != b1
    rep = [([1, 3, 50, 4, 6], [False] * 4 + [True]) for _ in range(3)]
    b3 = add_replay(b1, rep, 11)
    assert len(b3) == len(b1) and sum(len(b) for b in b3) == 13 and [len(b) for b in b1] == [4, 4, 2]
    x, y = collate([ex[0], rep[0]], "cpu")
    assert x.shape == (2, 5) and x[1, 4] == T.PAD_ID
    assert y[0].tolist() == [-100, -100, -100, 30, 2] and y[1].tolist() == [-100, -100, -100, 6, -100]
    assert count_tokens(ex)["supervised_tokens"] == 20
    cfg = {"lr": 1e-3, "warmup_steps": 4}
    assert lr_at(0, cfg) == 2.5e-4 and lr_at(3, cfg) == 1e-3 and lr_at(100, cfg) == 1e-3


def test_train_sft_learns_and_checkpoints(tmp_path):
    ex = [([1, 3, 20, 21, 4, 40, 41, 2], [False] * 5 + [True] * 3) for _ in range(64)]
    m = _toy()
    v0 = sft_val_loss(m, ex[:8], "cpu")
    cfg = {"batch_size": 8, "lr": 3e-3, "warmup_steps": 2, "weight_decay": 0.01, "betas": (0.9, 0.95), "grad_clip": 1.0,
           "precision": "fp32", "log_every_steps": 4}
    batches = instruction_batches(ex, 8, 11)
    s = train_sft(m, batches, cfg, 11, tmp_path, save_at_steps=[2, 4], log=lambda *a: None)
    assert (tmp_path / "step_2.pt").exists() and (tmp_path / "step_4.pt").exists() and (tmp_path / "final.pt").exists()
    assert s["updates"] == 8 and sft_val_loss(m, ex[:8], "cpu") < v0
    from uabr.eval.generate import load_model
    m2 = load_model(str(tmp_path / "final.pt"), "cpu")
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))


def test_greedy_batch_matches_single():
    m = _toy().eval()
    prompts = [[1, 3] + list(range(20, 20 + n)) + [4, 5] for n in (1, 5, 9, 3)]
    stop = [7]
    def single(p):
        x = list(p); out = []
        for _ in range(6):
            with torch.no_grad():
                z = m(torch.tensor([x]))[0][0, -1].float()
            z[T.CALL_ID] = -1e9; t = int(z.argmax())
            if t in (T.EOS_ID, 7): break
            out.append(t); x.append(t)
        return out
    got = greedy_batch(m, prompts, "cpu", max_new=6, stop_ids=stop, bs=3)
    assert got == [single(p) for p in prompts]


def test_decision_slot_and_metrics():
    m = _toy().eval()
    prompts = [[1, 3, 20 + i, 4] for i in range(5)]
    d = decision_slot(m, prompts, "cpu")
    assert len(d) == 5 and (d.margin == d.score.abs()).all() and d.p_action.between(0, 1).all()
    from uabr.routing.router import router_scores
    s, mg = router_scores(m, prompts, "action", "cpu")
    assert np.allclose(s, d.score, atol=1e-5)
    call = top_fraction(np.array([0.1, 0.9, 0.5, 0.2]), 0.5)
    assert call.tolist() == [False, True, True, False]
    o = outcome(call, [1, 0, 0, 0], [1, 1, 0, 1])
    assert o["accuracy"] == 0.5 and o["call_rate"] == 0.5 and o["harmful_answer_rate"] == 0.25


def test_paired_bootstrap_and_decide():
    rng = np.random.default_rng(0)
    a, b = rng.random(200) - 0.2, rng.random(200)
    r = paired_cluster_bootstrap(lambda idx, w: (a if w == "a" else b)[idx].mean(), np.arange(200) // 2, n_boot=300)
    assert r["ci_hi"] < 0 and abs(r["diff"] - (a.mean() - b.mean())) < 1e-12
    sd = lambda au, acc, fr: {"significant_drop": {"auroc": au, "acc_budget_10": acc, "acc_frozen_thr_10": fr}}
    assert decide({"11": sd(True, False, False), "29": sd(False, True, False)}) == "GO"
    assert decide({"11": sd(True, False, False), "29": sd(False, False, True)}) == "RECALIBRATION_ONLY"
    assert decide({"11": sd(True, False, False), "29": sd(False, False, False)}) == "NO_CHANGE"


def test_flips_by_margin():
    s0 = np.array([-3, -2, -0.1, 0.1, 0.2, 2, 3, -0.2, 1.5, -1.5])
    s1 = np.array([-3, -2, 0.5, -0.3, -0.1, 2, 3, 0.3, 1.5, -1.5])
    r = flips_by_margin(s0, s1, np.abs(s0), np.arange(10), n_bins=2)
    assert r["flip_rate"] == 0.4 and r["answer_to_call"] == 2 and r["call_to_answer"] == 2
    t = r["table"]
    assert t.n.tolist() == [4, 6] and t.flip_rate.iloc[0] == 1.0 and t.flip_rate.iloc[1] == 0.0
    assert r["auroc_small_margin_predicts_flip"]["auroc"] == 1.0

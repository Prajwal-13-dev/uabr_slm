import json, os, subprocess, sys
import numpy as np
import pandas as pd
import torch
from uabr.data import qa_format as Q
from uabr.data import tokenizer as T
from uabr.data.popqa import split_train_halves


def _tok():
    texts = ["What is George Rankin's occupation? politician. Hamlet was directed by Laurence Olivier."] * 30
    return T.train_bpe(texts, vocab_size=300, min_frequency=1)


def test_closed_and_evidence_examples_mask_only_answer():
    tok = _tok()
    t, f = Q.closed_example(tok, "What is George Rankin's occupation?", "politician")
    assert t[:2] == [T.BOS_ID, T.USER_ID] and T.ANSWER_ID in t and t[-1] == T.EOS_ID
    a = t.index(T.ANSWER_ID)
    assert not any(f[: a + 1]) and all(f[a + 1:])                  # action token itself is not a target
    (t2, f2), k = Q.evidence_example(tok, "Who directed Hamlet?", ["Hamlet: by Laurence Olivier", "x"], "Laurence Olivier")
    assert k == 2 and T.CALL_ID in t2 and T.EVIDENCE_END_ID in t2
    e = t2.index(T.EVIDENCE_END_ID)
    assert not any(f2[: e + 1]) and all(f2[e + 1:])


def test_evidence_overflow_drops_windows():
    tok = _tok()
    ws = ["Hamlet was directed by Laurence Olivier. " * 20] * 3
    (t, f), k = Q.evidence_example(tok, "Who directed Hamlet?", ws, "Laurence Olivier", seq_len=200)
    assert len(t) <= 201 and k < 3


def test_pack_targets_align():
    tok = _tok()
    ex = [Q.closed_example(tok, "What is George Rankin's occupation?", "politician") for _ in range(20)]
    X, M = Q.pack(ex, seq_len=64)
    assert X.shape[1] == 65 and M.shape[1] == 64 and M.any(axis=1).all()
    # every masked target is an answer token or EOS of some example
    flat_t = [tok_ for t, _ in ex for tok_ in t]
    ans = set(t[i] for t, fl in ex for i, x in enumerate(fl) if x)
    for i in range(len(X)):
        assert set(X[i, 1:][M[i]].tolist()) <= ans


def test_lm_blocks():
    s = np.arange(1000, dtype=np.uint16)
    X, M = Q.lm_blocks(s, seq_len=100)
    assert X.shape == (9, 101) and M.all() and (X[1, 0] == 100)
    X2, _ = Q.lm_blocks(s, seq_len=100, n=5, rng=np.random.default_rng(0))
    assert X2.shape == (5, 101)


def test_split_train_halves_grouped():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"split": rng.choice(["train", "val"], n, p=[.8, .2]), "in_main": rng.random(n) > .1,
                       "group_id": rng.integers(0, 900, n), "prop": rng.choice(list("abc"), n),
                       "dose_group": rng.choice(["d0", "d16", "undosed"], n)})
    h = split_train_halves(df)
    tr = df[(df.split == "train") & df.in_main]
    assert set(h[tr.index]) == {"A", "B"} and (h[~df.index.isin(tr.index)] == "").all()
    g = pd.DataFrame({"g": tr.group_id, "h": h[tr.index]}).groupby("g").h.nunique()
    assert (g == 1).all()
    share = (h[tr.index] == "A").mean()
    assert 0.45 < share < 0.55


def test_trainer_with_items(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.integers(9, 200, size=(64, 33)).astype(np.uint16)
    M = np.zeros((64, 32), bool); M[:, 20:] = True
    np.save(tmp_path / "x.npy", X); np.save(tmp_path / "m.npy", M)
    rng.integers(0, 200, 32 * 20 + 1).astype(np.uint16).tofile(tmp_path / "val.bin")
    cfg = dict(train_bin="", val_bin=str(tmp_path / "val.bin"), out_dir=str(tmp_path / "out"),
               items_x=str(tmp_path / "x.npy"), items_m=str(tmp_path / "m.npy"),
               model=dict(vocab_size=200, n_layers=1, d_model=32, n_heads=4, d_ff=64), seq_len=32,
               global_batch_seqs=8, micro_batch_seqs=4, warmup_steps=1, eval_every=4, eval_tokens=320, log_every=2)
    json.dump(cfg, open(tmp_path / "c.json", "w"))
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")}
    r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                        "-m", "uabr.train.pretrain", "--config", str(tmp_path / "c.json")], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.load(open(tmp_path / "out" / "summary.json"))
    assert s["completed"] and s["total_steps"] == 8

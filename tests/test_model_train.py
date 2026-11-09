import json, os, subprocess, sys
import numpy as np
import torch
from uabr.model.llama import LlamaLM, ModelConfig, expected_param_count
from uabr.train.pretrain import BlockStream, lr_at, TrainConfig


def test_param_count_40m():
    c = ModelConfig()
    assert expected_param_count(c) == 39_707_136          # 8x512, d_ff 2048, vocab 12k, tied
    assert LlamaLM(c).num_params() == expected_param_count(c)


def test_causal_and_tied():
    torch.manual_seed(0)
    c = ModelConfig(vocab_size=100, n_layers=2, d_model=64, n_heads=4, d_ff=128, max_seq_len=32)
    m = LlamaLM(c).eval()
    assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr()
    x = torch.randint(0, 100, (1, 16)); y = x.clone(); y[0, 10:] = (y[0, 10:] + 1) % 100
    a, _ = m(x); b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5) and not torch.allclose(a[0, 10:], b[0, 10:])
    _, loss = m(x[:, :-1], x[:, 1:]); loss.backward()
    assert torch.isfinite(loss)
    out = m.generate(x[:, :4], 5)
    assert out.shape == (1, 9)


def test_block_stream_exact_once(tmp_path):
    p = tmp_path / "t.bin"
    np.arange(10 * 8 + 1, dtype=np.uint16).tofile(p)
    seen = []
    for r in range(2):
        s = BlockStream(str(p), 8, r, 2, seed=1)
        assert s.per_rank == 5
        b = s.batch(0, 5)
        assert b.shape == (5, 9)
        assert (b[:, 1:] - b[:, :-1] == 1).all()          # contiguous with +1 target
        seen += list(s.blocks)
    assert sorted(seen) == list(range(10))


def test_lr_schedule():
    c = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10)
    assert abs(lr_at(9, 100, c) - 1e-3) < 1e-12 and lr_at(0, 100, c) < lr_at(5, 100, c)
    assert abs(lr_at(100, 100, c) - 1e-4) < 1e-9


def test_torchrun_two_ranks_and_resume(tmp_path):
    rng = np.random.default_rng(0)
    rng.integers(0, 200, 64 * 33 * 4 + 1).astype(np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, 200, 64 * 20 + 1).astype(np.uint16).tofile(tmp_path / "val.bin")
    cfg = dict(train_bin=str(tmp_path / "train.bin"), val_bin=str(tmp_path / "val.bin"), out_dir=str(tmp_path / "out"),
               model=dict(vocab_size=200, n_layers=2, d_model=32, n_heads=4, d_ff=64), seq_len=32,
               global_batch_seqs=8, micro_batch_seqs=2, warmup_steps=2, eval_every=5, eval_tokens=320, log_every=2,
               max_steps=6)
    json.dump(cfg, open(tmp_path / "c.json", "w"))
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
           "-m", "uabr.train.pretrain", "--config", str(tmp_path / "c.json")]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.load(open(tmp_path / "out" / "summary.json"))
    assert s["completed"] and s["steps_done"] == 6 and s["world"] == 2
    # resume: raise max_steps, rerun -> continues from 6
    cfg["max_steps"] = 10; json.dump(cfg, open(tmp_path / "c.json", "w"))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "resumed from step 6" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    assert json.load(open(tmp_path / "out" / "summary.json"))["steps_done"] == 10

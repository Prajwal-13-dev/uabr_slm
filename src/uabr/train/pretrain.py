"""Pretraining (notebook 03). Launch:  torchrun --standalone --nproc_per_node=N -m uabr.train.pretrain --config run.json

Data order: the notebook-01 stream is cut into non-overlapping 512-token blocks (+1 target
token), the block order is permuted once (seed), and ranks take blocks round-robin. Training
is ONE exact pass, so every exposure copy is seen exactly once (the incomplete last global
batch is dropped and reported).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.distributed as dist

from uabr.model.llama import LlamaLM, ModelConfig, expected_param_count


@dataclass
class TrainConfig:
    train_bin: str = ""
    val_bin: str = ""
    out_dir: str = "out"
    model: dict = field(default_factory=dict)        # ModelConfig overrides
    seq_len: int = 512
    global_batch_seqs: int = 256                      # 131k tokens / optimizer step
    micro_batch_seqs: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "auto"                           # auto -> bf16 on sm80+, fp16 on T4/P100, fp32 on CPU
    compile: bool = False
    seed: int = 1234
    data_seed: int = 4321
    eval_every: int = 500
    eval_tokens: int = 2_000_000
    log_every: int = 20
    ckpt_minutes: float = 30.0
    max_hours: float = 11.0                           # stop + checkpoint before Kaggle's 12h limit
    max_steps: int = 0                                # 0 = full single pass (testing only otherwise)
    init_from: str = ""                               # continued pretraining: load model weights only (fresh optimizer)


# --------------------------------------------------------------------------- dist helpers
def setup_dist():
    if "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world, device


def is_dist():
    return dist.is_available() and dist.is_initialized()


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


# --------------------------------------------------------------------------- data
class BlockStream:
    """Exact single pass over fixed blocks of a uint16 token file, sharded across ranks."""

    def __init__(self, path: str, seq_len: int, rank: int, world: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.T = seq_len
        self.n_blocks = (len(self.data) - 1) // seq_len
        perm = np.random.default_rng(seed).permutation(self.n_blocks)
        self.per_rank = self.n_blocks // world                       # equal share per rank
        self.blocks = perm[rank::world][: self.per_rank]

    def batch(self, start: int, n: int) -> torch.Tensor:
        rows = [self.data[b * self.T: b * self.T + self.T + 1] for b in self.blocks[start:start + n]]
        return torch.from_numpy(np.stack(rows).astype(np.int64))


def lr_at(step: int, total: int, c: TrainConfig) -> float:
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    p = min(1.0, (step - c.warmup_steps) / max(1, total - c.warmup_steps))
    return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1 + math.cos(math.pi * p))


def pick_dtype(c: TrainConfig, device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if c.precision == "auto":
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[c.precision]


@torch.no_grad()
def evaluate(model, val: np.ndarray, c: TrainConfig, rank, world, device, ctx) -> float:
    model.eval()
    T = c.seq_len
    n_blocks = min((len(val) - 1) // T, max(1, c.eval_tokens // T))
    mine = list(range(rank, n_blocks, world))
    tot, cnt = torch.zeros(1, device=device), torch.zeros(1, device=device)
    for i in range(0, len(mine), c.micro_batch_seqs):
        rows = [val[b * T: b * T + T + 1] for b in mine[i:i + c.micro_batch_seqs]]
        xy = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
        with ctx:
            _, loss = model(xy[:, :-1], xy[:, 1:])
        tot += loss.float() * xy.shape[0]; cnt += xy.shape[0]
    if is_dist():
        dist.all_reduce(tot); dist.all_reduce(cnt)
    model.train()
    return float(tot / cnt.clamp(min=1))


def save_ckpt(path, raw_model, opt, scaler, step, c, extra):
    tmp = path + ".tmp"
    torch.save({"model": raw_model.state_dict(), "optim": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None, "step": step,
                "train_cfg": asdict(c), "model_cfg": raw_model.cfg.to_dict(), **extra}, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- main
def main(cfg_path: str):
    c = TrainConfig(**json.load(open(cfg_path)))
    rank, world, device = setup_dist()
    master = rank == 0
    torch.manual_seed(c.seed + rank)
    os.makedirs(c.out_dir, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if master else (lambda *a: None)

    mcfg = ModelConfig(**{**ModelConfig().to_dict(), **c.model, "max_seq_len": c.seq_len})
    model = LlamaLM(mcfg).to(device)
    n_params = model.num_params()
    assert n_params == expected_param_count(mcfg), (n_params, expected_param_count(mcfg))

    accum = c.global_batch_seqs // (c.micro_batch_seqs * world)
    assert accum >= 1 and accum * c.micro_batch_seqs * world == c.global_batch_seqs, \
        "global_batch_seqs must be divisible by micro_batch_seqs * world_size"
    stream = BlockStream(c.train_bin, c.seq_len, rank, world, c.data_seed)
    seqs_per_rank_step = c.micro_batch_seqs * accum
    total_steps = stream.per_rank // seqs_per_rank_step
    if c.max_steps:
        total_steps = min(total_steps, c.max_steps)
    tokens_per_step = c.global_batch_seqs * c.seq_len
    dropped = (stream.n_blocks - total_steps * c.global_batch_seqs) * c.seq_len
    val = np.memmap(c.val_bin, dtype=np.uint16, mode="r")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=c.lr, betas=(c.beta1, c.beta2), eps=1e-8,
                            fused=(device.type == "cuda"))
    dtype = pick_dtype(c, device)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    ctx = torch.autocast(device.type, dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()

    start_step, ckpt_path = 0, os.path.join(c.out_dir, "ckpt_latest.pt")
    if c.init_from and not os.path.exists(ckpt_path):
        init = torch.load(c.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model"])
        log(f"initialised weights from {c.init_from}")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optim"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_step = int(ck["step"])
        log(f"resumed from step {start_step}")

    raw = model
    if c.compile:
        model = torch.compile(model)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    log(f"params {n_params:,} | world {world} | precision {dtype} | accum {accum} | "
        f"steps {total_steps:,} x {tokens_per_step:,} tok = {total_steps*tokens_per_step/1e9:.3f}B | "
        f"dropped tail {dropped:,} tok | start {start_step}")
    log_path = os.path.join(c.out_dir, "train_log.csv")
    if master and start_step == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens", "lr", "train_loss", "val_loss", "tok_per_s", "elapsed_min"])

    def write_log(row):
        if master:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    t0 = time.time(); last_ckpt = t0; tput = []; run_loss = None
    if start_step == 0:
        v0 = evaluate(model, val, c, rank, world, device, ctx)
        log(f"step 0 val_loss {v0:.4f}"); write_log([0, 0, 0, "", f"{v0:.5f}", "", 0])

    step = start_step
    for step in range(start_step, total_steps):
        ts = time.time()
        lr = lr_at(step, total_steps, c)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros(1, device=device)
        base = step * seqs_per_rank_step
        for m in range(accum):
            xy = stream.batch(base + m * c.micro_batch_seqs, c.micro_batch_seqs).to(device, non_blocking=True)
            sync = (m == accum - 1) or world == 1
            with (nullcontext() if sync else model.no_sync()), ctx:
                _, loss = model(xy[:, :-1], xy[:, 1:])
                loss = loss / accum
            (scaler.scale(loss) if scaler else loss).backward()
            loss_acc += loss.detach()
        if scaler:
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), c.grad_clip)
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % c.log_every == 0 or step == start_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            l = float(all_reduce_mean(loss_acc.clone()))
            if not math.isfinite(l):
                raise FloatingPointError(f"non-finite loss at step {step}")
            run_loss = l if run_loss is None else 0.9 * run_loss + 0.1 * l
            tps = tokens_per_step / (time.time() - ts)
            if step - start_step >= 10:
                tput.append(tps)
            el = (time.time() - t0) / 60
            eta = el / (step + 1 - start_step) * (total_steps - step - 1)
            log(f"step {step+1:6d}/{total_steps} loss {l:.4f} (ema {run_loss:.4f}) lr {lr:.2e} "
                f"gnorm {float(gnorm):.2f} {tps/1e3:.1f}k tok/s  {el:.1f} min  eta {eta:.0f} min")
            write_log([step + 1, (step + 1) * tokens_per_step, f"{lr:.3e}", f"{l:.5f}", "", f"{tps:.0f}", f"{el:.2f}"])

        if (step + 1) % c.eval_every == 0 or step + 1 == total_steps:
            v = evaluate(model, val, c, rank, world, device, ctx)
            log(f"step {step+1} val_loss {v:.4f}")
            write_log([step + 1, (step + 1) * tokens_per_step, f"{lr:.3e}", "", f"{v:.5f}", "", f"{(time.time()-t0)/60:.2f}"])

        flag = torch.tensor([float((time.time() - t0) / 3600 > c.max_hours)], device=device)
        if is_dist():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)        # all ranks stop together
        out_of_time = bool(flag.item())
        if master and (time.time() - last_ckpt > c.ckpt_minutes * 60 or out_of_time):
            save_ckpt(ckpt_path, raw, opt, scaler, step + 1, c, {"rank_world": world})
            last_ckpt = time.time(); log(f"checkpoint saved at step {step+1}")
        if out_of_time:
            log("time budget reached - stopping; resume by re-running with this checkpoint")
            break
    else:
        step = total_steps - 1

    done = step + 1 >= total_steps
    if master:
        save_ckpt(ckpt_path, raw, opt, scaler, step + 1, c, {"rank_world": world})
        if done:
            torch.save({"model": raw.state_dict(), "model_cfg": mcfg.to_dict()},
                       os.path.join(c.out_dir, "model_final.pt"))
        json.dump({"params": n_params, "expected_params": expected_param_count(mcfg), "world": world,
                   "precision": str(dtype), "total_steps": total_steps, "steps_done": step + 1,
                   "completed": done, "tokens_trained": (step + 1) * tokens_per_step,
                   "tokens_per_step": tokens_per_step, "dropped_tail_tokens": int(dropped),
                   "throughput_tok_per_s_median": float(np.median(tput)) if tput else None,
                   "wall_minutes_this_session": (time.time() - t0) / 60,
                   "model_cfg": mcfg.to_dict(), "train_cfg": asdict(c)},
                  open(os.path.join(c.out_dir, "summary.json"), "w"), indent=2)
        log(f"finished: steps {step+1}/{total_steps} completed={done}")
    if is_dist():
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)

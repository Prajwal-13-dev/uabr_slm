"""Batched-by-sample generation and answer log-likelihood for the base model (no KV cache needed
at this scale: prompts <= 512 tokens, answers <= 12 tokens)."""
from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from uabr.data.tokenizer import BOS_ID, EOS_ID


def load_model(path: str, device: str):
    from uabr.model.llama import LlamaLM, ModelConfig

    st = torch.load(path, map_location=device, weights_only=False)
    m = LlamaLM(ModelConfig(**st["model_cfg"])).to(device).eval()
    m.load_state_dict(st["model"])
    return m


def encode(tok, text: str) -> List[int]:
    return [BOS_ID] + tok.encode(text, add_special_tokens=False).ids


def fit_evidence(tok, build, windows: Sequence[str], max_len: int) -> Tuple[List[int], int]:
    """Use as many top windows as fit (3 -> 2 -> 1 -> 0). build(windows) -> prompt text."""
    for k in range(len(windows), -1, -1):
        ids = encode(tok, build(list(windows[:k])))
        if len(ids) <= max_len:
            return ids, k
    return ids[-max_len:], 0


def seed_for(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "big") % (2**63)


@torch.no_grad()
def generate(model, ids: List[int], n: int, max_new: int, temperature: float, seed: int, device: str,
             stop_ids: Sequence[int] = ()) -> List[List[int]]:
    """n continuations of one prompt (greedy if temperature == 0). Returns new token ids per sample."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.tensor([ids] * n, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    stop = torch.tensor(list(stop_ids) + [EOS_ID], device=device)
    for _ in range(max_new):
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits, _ = model(x[:, -model.cfg.max_seq_len:])
        logits = logits[:, -1].float()
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=g).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, EOS_ID), nxt)
        x = torch.cat([x, nxt[:, None]], 1)
        done |= torch.isin(nxt, stop)
        if bool(done.all()):
            break
    out = []
    for row in x[:, len(ids):].tolist():
        cut = next((i for i, t in enumerate(row) if t == EOS_ID), len(row))
        out.append(row[:cut])
    return out


@torch.no_grad()
def answer_mean_logprob(model, prompt_ids: List[int], answer_ids: List[int], device: str) -> float:
    """mean_t log P(a_t | prompt, a_<t)  (PLAN §6 likelihood proxy)."""
    ids = (prompt_ids + answer_ids)[-model.cfg.max_seq_len:]
    n_a = len(answer_ids)
    x = torch.tensor([ids], device=device)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        logits, _ = model(x)
    lp = F.log_softmax(logits[0, -n_a - 1:-1].float(), -1)
    return float(lp.gather(1, torch.tensor(ids[-n_a:], device=device)[:, None]).mean())

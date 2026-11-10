"""Decoder-only LLaMA-style model (PLAN §4; amendment A6).

8 layers, d_model 512, 8 heads (head dim 64), SwiGLU FFN (d_ff 2048), RoPE, pre-norm RMSNorm,
no biases, tied input/output embeddings, vocab 12k, context 512  ->  ~39.7M parameters.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 12000
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02

    def to_dict(self):
        return asdict(self)


def expected_param_count(c: ModelConfig) -> int:
    """Analytic count with tied embeddings (embedding counted once)."""
    per_layer = 4 * c.d_model * c.d_model + 3 * c.d_model * c.d_ff + 2 * c.d_model
    return c.vocab_size * c.d_model + c.n_layers * per_layer + c.d_model


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); rotate pairs (even, odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.h, self.hd = c.n_heads, c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.w_up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.w_down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model, c.norm_eps), Attention(c)
        self.n2, self.mlp = RMSNorm(c.d_model, c.norm_eps), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class LlamaLM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight                      # tied
        cos, sin = rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():                         # scaled residual projections
            if n.endswith("proj.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, 0.0, c.init_std / math.sqrt(2 * c.n_layers))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())             # tied weight counted once

    def forward(self, idx, targets=None, ignore_index: int = -100, return_hidden: bool = False):
        T = idx.shape[1]
        assert T <= self.cfg.max_seq_len, f"sequence {T} > max_seq_len {self.cfg.max_seq_len}"
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        h = self.norm(x)
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=ignore_index)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.0, eos_id=None, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len:])
            logits = logits[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator)
            idx = torch.cat([idx, nxt], 1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx

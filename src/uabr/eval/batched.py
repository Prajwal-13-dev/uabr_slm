"""Batched greedy decoding with right padding (no KV cache, no attention mask needed).

Rows are right-padded; each row's next token is read at its OWN last position. With causal attention the pad
tokens after a position never influence it, so each row decodes exactly as it would alone (up to fp16
rounding). Same semantics as notebook 06 `greedy_ban`: `banned` tokens get -inf, decoding stops at <EOS> or any
stop token (tokens whose text contains a newline), and the stop token is not part of the answer.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

from uabr.data.tokenizer import CALL_ID, EOS_ID, PAD_ID


@torch.no_grad()
def greedy_batch(model, prompts: Sequence[List[int]], device: str, max_new: int = 16, banned=(CALL_ID,),
                 stop_ids=(), bs: int = 64, max_len: int = 512) -> List[List[int]]:
    model.eval()
    stop = set(int(s) for s in stop_ids) | {EOS_ID}
    out: List[List[int]] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))      # similar lengths -> less padding
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        P = [list(prompts[i])[-max_len:] for i in idx]
        lens = torch.tensor([len(p) for p in P])
        L = min(max_len, int(lens.max()) + max_new)
        x = torch.full((len(P), L), PAD_ID, dtype=torch.long)
        for r, p in enumerate(P):
            x[r, :len(p)] = torch.tensor(p)
        x = x.to(device); cur = lens.clone().to(device)
        gen = [[] for _ in P]; alive = torch.ones(len(P), dtype=torch.bool, device=device)
        for _ in range(max_new):
            if not bool(alive.any()):
                break
            T = int(cur.max())
            if T >= L:
                break
            with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
                logits = model(x[:, :T])[0]
            z = logits[torch.arange(len(P), device=x.device), cur - 1].float()
            if banned:
                z[:, list(banned)] = -1e9
            nxt = z.argmax(-1)
            for r in range(len(P)):
                if not alive[r]:
                    continue
                t = int(nxt[r])
                if t in stop:
                    alive[r] = False; continue
                gen[r].append(t)
                x[r, cur[r]] = t; cur[r] += 1
                if cur[r] >= L:
                    alive[r] = False
        for r, i in enumerate(idx):
            out[i] = gen[r]
    return out

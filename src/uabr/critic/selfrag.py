"""Self-RAG-adapted critique token (notebook 10). Team decisions (Nov 11 2026):

* Only [Retrieve] (= the existing <ANSWER>/<CALL> router) and [IsRel] are kept; [IsSup] is merged into IsRel
  (for a one-entity answer "the window states the fact" is the same question) and [IsUse] is dropped.
* Two new tokens <REL> / <IRR> are APPENDED to the vocabulary (ids = old vocab size, +1); the tokenizer file is not
  changed because the ids are inserted directly.
* Critique slot = the last position of the existing single-window evidence prompt
      <BOS><USER> q<ASSISTANT><CALL><EVIDENCE> w</EVIDENCE>
  scored as z_REL - z_IRR (2-logit CE, never generated), so the answer format is unchanged.
* Training: transformer body + the 2 new embedding rows train; the old (tied) embedding rows stay frozen (R5);
  half-A Q&A replay (answer tokens) and half-B router replay (2-logit decision) keep answering and routing intact.
* Training labels = "subject AND gold alias in the window" (automatic proxy; G2: precision 79 %, recall 81 %).
"""
from __future__ import annotations

import copy
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from uabr.data.tokenizer import ANSWER_ID, CALL_ID
from uabr.routing.metrics import auroc
from uabr.routing.router import _batch, _decision_logits, _replay_batch


def critique_ids(model):
    """(REL_ID, IRR_ID) of a model that already has the critique tokens."""
    V = model.cfg.vocab_size
    return V - 2, V - 1


def add_critique_tokens(model, seed: int = 0):
    """New model with vocab + 2; old weights copied; the 2 new rows = mean embedding + small noise."""
    from uabr.model.llama import LlamaLM, ModelConfig
    c = model.cfg.to_dict(); n_old = c["vocab_size"]; c["vocab_size"] = n_old + 2
    dev = next(model.parameters()).device
    new = LlamaLM(ModelConfig(**c)).to(dev)
    sd = model.state_dict()
    with torch.no_grad():
        W = sd["embed.weight"]
        g = torch.Generator(device="cpu").manual_seed(seed)
        extra = W.mean(0, keepdim=True).cpu() + 0.02 * W.std().cpu() * torch.randn(2, W.shape[1], generator=g)
        sd = {k: v for k, v in sd.items() if k not in ("embed.weight", "lm_head.weight")}
        new.load_state_dict(sd, strict=False)
        new.embed.weight.copy_(torch.cat([W, extra.to(W.device, W.dtype)], 0))
    assert new.lm_head.weight.data_ptr() == new.embed.weight.data_ptr()
    return new.eval()


def freeze_old_rows(model, n_old: int):
    """Gradient hook: only embedding rows >= n_old (the new tokens) receive gradient. Returns the hook handle."""
    def hook(g):
        g = g.clone(); g[:n_old] = 0; return g
    model.embed.weight.requires_grad_(True)
    return model.embed.weight.register_hook(hook)


@torch.no_grad()
def critique_scores(model, prompts: Sequence[List[int]], device: str, bs: int = 64) -> np.ndarray:
    rel, irr = critique_ids(model)
    model.eval(); out = []
    for i in range(0, len(prompts), bs):
        x, last = _batch(prompts[i:i + bs], device)
        z = _decision_logits(model, x, last, device)
        out.append((z[:, rel] - z[:, irr]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def last_hidden(model, prompts: Sequence[List[int]], device: str, bs: int = 64) -> np.ndarray:
    """Final-layer hidden state at the last prompt position (for the hidden-state probe baseline)."""
    model.eval(); out = []
    for i in range(0, len(prompts), bs):
        x, last = _batch(prompts[i:i + bs], device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            _, h = model(x, return_hidden=True)
        out.append(h[torch.arange(len(last), device=x.device), last].float().cpu().numpy())
    return np.concatenate(out)


def train_critic(model, prompts, targets, device: str, n_old: int, lr: float = 3e-5, epochs: int = 3, bs: int = 32,
                 eval_every: int = 100, seed: int = 0, log=print, qa_replay=None, qa_bs: int = 16,
                 router_replay=None, router_bs: int = 16, evaluate: Optional[Callable] = None):
    """Loss = CE over {<IRR>, <REL>} at the critique slot (target 1 = relevant) + Q&A replay (answer tokens)
    + router replay (2-logit {<ANSWER>, <CALL>} at <ASSISTANT>). Old embedding rows frozen.
    evaluate(model) -> (score, qualifies, info); the best QUALIFYING score is kept (else best score, flagged)."""
    rel, irr = critique_ids(model)
    h = freeze_old_rows(model, n_old)
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    n = len(prompts); step = 0; hist = []
    best = {"score": -1.0, "step": -1, "state": None, "info": None}
    best_any = {"score": -1.0, "step": -1, "state": None, "info": None}

    def do_eval():
        s, ok, info = evaluate(model) if evaluate else (float("nan"), True, {})
        hist.append({"step": step, "score": s, "qualifies": bool(ok), **info})
        if ok and s > best["score"]:
            best.update(score=s, step=step, state=copy.deepcopy(model.state_dict()), info=info)
        if not ok and s > best_any["score"] and best["state"] is None:     # fallback only while nothing qualifies
            best_any.update(score=s, step=step, state=copy.deepcopy(model.state_dict()), info={**info, "unqualified": True})
        model.train()
        return s, ok, info
    do_eval()
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(n, generator=g).split(bs):
            x, last = _batch([prompts[i] for i in b], device)
            tgt = torch.tensor([targets[i] for i in b], device=device)
            z = _decision_logits(model, x, last, device)
            loss = F.cross_entropy(z[:, [irr, rel]], tgt)
            if qa_replay:
                ridx = torch.randint(0, len(qa_replay), (qa_bs,), generator=g).tolist()
                rx, ry = _replay_batch(qa_replay, ridx, device)
                with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                    _, rl = model(rx[:, :-1], ry)
                loss = loss + rl
            if router_replay:
                rp, rt = router_replay
                ridx = torch.randint(0, len(rp), (router_bs,), generator=g).tolist()
                rx, rlast = _batch([rp[i] for i in ridx], device)
                rz = _decision_logits(model, rx, rlast, device)
                loss = loss + F.cross_entropy(rz[:, [ANSWER_ID, CALL_ID]], torch.tensor([rt[i] for i in ridx], device=device))
            opt.zero_grad(set_to_none=True)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            step += 1
            if step % eval_every == 0:
                s, ok, info = do_eval()
                log(f"  epoch {ep+1} step {step} loss {loss.item():.4f} | selection {s:.4f} qualifies={ok} {info}")
    do_eval()
    h.remove()
    if best["state"] is None:
        log(f"  WARNING: no checkpoint met the preservation rule; best-score step {best_any['step']} kept (flagged)")
        best = best_any
    return best, hist

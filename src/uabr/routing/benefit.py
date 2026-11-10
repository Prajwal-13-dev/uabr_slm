"""Counterfactual benefit labels with the frozen theta0 (PLAN §6, amendment A1; notebook 06).

Per question: greedy + k samples (T=0.7, fixed seeds) closed-book (A) and with evidence (C);
b_quality = mean alias-F1(C samples) - mean alias-F1(A samples) in [-1, 1] (1/k steps).
b_net = b_quality - lambda * c (c = 1 for CALL). Label CALL iff b_net > 0.
Also records pre-retrieval routing features from the closed-book greedy pass: answer mean token
log-prob, first-token prob, first-token entropy, and the hidden state at the action position.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from uabr.data import qa_format as Q
from uabr.eval.generate import answer_mean_logprob, generate, seed_for
from uabr.eval.metrics import alias_em, alias_f1
from uabr.eval.prompts import clean_generation


@torch.no_grad()
def greedy_features(model, prompt_ids: List[int], gen_ids: List[int], device: str) -> Dict:
    """Confidence features of the model's own greedy answer + hidden state at the action position
    (the last prompt token, i.e. <ANSWER>; the hidden state at <ASSISTANT> is the one before it)."""
    ids = (prompt_ids + (gen_ids or [])) [-model.cfg.max_seq_len:]
    x = torch.tensor([ids], device=device)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        logits, h = model(x, return_hidden=True)
    lp = F.log_softmax(logits[0].float(), -1)
    p0 = len(prompt_ids) - 1                               # position predicting the first answer token
    ent = float(-(lp[p0].exp() * lp[p0]).sum())
    if gen_ids:
        tok_lp = lp[torch.arange(p0, p0 + len(gen_ids), device=device), torch.tensor(gen_ids, device=device)]
        mean_lp, first_p = float(tok_lp.mean()), float(tok_lp[0].exp())
    else:
        mean_lp, first_p = float("nan"), float("nan")
    return {"conf_mean_lp": mean_lp, "conf_first_p": first_p, "first_entropy": ent,
            "h_assistant": h[0, p0 - 1].float().cpu().numpy().astype(np.float16)}


def label_question(model, tok, q: str, gold: Sequence[str], windows: Sequence[str], qid, seed: int, k: int,
                   temp: float, max_new: int, seq: int, device: str, stop_ids, answer: str = None) -> Dict:
    pa = Q.closed_prompt_ids(tok, q)
    pc, nw = Q.evidence_prompt_ids(tok, q, windows, seq - max_new)
    rec = {"n_windows_used": nw}
    for cond, p in [("A", pa), ("C", pc)]:
        g = generate(model, p, 1, max_new, 0.0, 0, device, stop_ids)[0]
        rec[f"greedy_{cond}"] = clean_generation(tok.decode(g))
        rec[f"em_{cond}"] = alias_em(rec[f"greedy_{cond}"], gold)
        ss = [clean_generation(tok.decode(s)) for s in generate(model, p, k, max_new, temp, seed_for(seed, qid, cond), device, stop_ids)]
        rec[f"samples_{cond}"] = ss
        rec[f"sample_f1_{cond}"] = float(np.mean([alias_f1(s, gold) for s in ss]))
        rec[f"sample_em_{cond}"] = float(np.mean([alias_em(s, gold) for s in ss]))
        if cond == "A":
            rec.update(greedy_features(model, p, g, device)); rec["greedy_ids_A"] = g
        if answer:   # likelihood proxy (A1 secondary): mean log-prob of the canonical answer (+<EOS>)
            rec[f"lp_{cond}"] = answer_mean_logprob(model, p, Q._enc(tok, " " + answer) + [2], device)
    rec["b_quality"] = rec["sample_f1_C"] - rec["sample_f1_A"]
    rec["delta_greedy_em"] = rec["em_C"] - rec["em_A"]
    if answer:
        rec["b_likelihood"] = float(np.exp(rec["lp_C"]) - np.exp(rec["lp_A"]))
    return rec


def choose_lambda(b: np.ndarray, delta_em: np.ndarray, grid: Sequence[float]) -> Dict:
    """Pick lambda on validation: labels CALL = (b - lambda > 0) should agree best (balanced accuracy)
    with the realised greedy outcome 'retrieval strictly helps' (delta_em > 0)."""
    y = delta_em > 0
    best, rows = None, []
    for lam in grid:
        pred = (b - lam) > 0
        tpr = (pred & y).sum() / max(1, y.sum()); tnr = (~pred & ~y).sum() / max(1, (~y).sum())
        bal = 0.5 * (tpr + tnr)
        rows.append({"lambda": float(lam), "balanced_acc": float(bal), "call_rate": float(pred.mean())})
        if best is None or bal > best["balanced_acc"] + 1e-12:
            best = rows[-1]
    return {"chosen": best, "grid": rows}

"""Knowledge probes that do not depend on free generation (notebook 04 v2).

1. Ranking probe: score the gold answer against N distractor answers of the same relation by the
   log-probability of " <answer>" after a zero-shot cloze prompt. Subject-free baseline: the same
   ranking with the subject replaced by "the subject" (the model's own relation prior). Knowledge =
   accuracy with the subject minus accuracy without it.
2. Verbatim probe: for injected fact sentences, the sentence prefix up to the answer -> does the model
   continue with the answer (memorisation in the original wording)?
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from uabr.data.tokenizer import PAD_ID
from uabr.eval.metrics import normalize_answer


@torch.no_grad()
def score_continuations(model, prompt_ids: List[int], conts: List[List[int]], device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Sum and mean log P(cont | prompt) for each continuation (right-padded batch; causal LM, so padding
    after the continuation cannot affect it)."""
    L = max(len(c) for c in conts)
    x = torch.full((len(conts), len(prompt_ids) + L), PAD_ID, dtype=torch.long)
    for i, c in enumerate(conts):
        x[i, :len(prompt_ids)] = torch.tensor(prompt_ids); x[i, len(prompt_ids):len(prompt_ids) + len(c)] = torch.tensor(c)
    x = x[:, -model.cfg.max_seq_len:].to(device)
    off = x.shape[1] - (len(prompt_ids) + L)          # >0 only if truncated on the left
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        logits, _ = model(x)
    lp = F.log_softmax(logits.float(), -1)
    sums, means = np.zeros(len(conts)), np.zeros(len(conts))
    p0 = len(prompt_ids) + off
    for i, c in enumerate(conts):
        pos = torch.arange(p0 - 1, p0 - 1 + len(c), device=device)
        tok = torch.tensor(c, device=device)
        v = lp[i, pos, tok]
        sums[i], means[i] = float(v.sum()), float(v.mean())
    return sums, means


def candidate_sets(answers_by_rel: Dict[str, List[str]], gold_by_q: Dict, rel_by_q: Dict, n_distract: int, seed: int):
    """Per question: [gold canonical] + n distractors from the same relation (no alias of gold)."""
    rng = np.random.default_rng(seed)
    out = {}
    for q, (canon, aliases) in gold_by_q.items():
        bad = {normalize_answer(a) for a in aliases}
        pool = [a for a in answers_by_rel[rel_by_q[q]] if normalize_answer(a) not in bad]
        k = min(n_distract, len(pool))
        out[q] = [canon] + list(rng.choice(pool, size=k, replace=False))
    return out


def answer_span(sentence: str, aliases: Sequence[str], min_chars: int = 3) -> Optional[Tuple[int, str]]:
    """Earliest whole-word, case-insensitive occurrence of a gold alias in the raw sentence."""
    best = None
    for a in sorted({a for a in aliases if len(a) >= min_chars}, key=len, reverse=True):
        m = re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", sentence, flags=re.IGNORECASE)
        if m and m.start() > 0 and (best is None or m.start() < best[0]):
            best = (m.start(), sentence[m.start():m.end()])
    return best

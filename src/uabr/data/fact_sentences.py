"""Option A re-injection (G1 fallback; team decisions 2026-11-10, PLAN §16).

- Fact sentence = an AUTHENTIC sentence from the notebook-01 corpus that names the subject
  (subject name or alias, >= 4 chars, whole word) AND contains a gold answer alias (>= 3 chars).
  Sources per question: the subject's article, the object's article, and the articles of the
  question's top-20 BM25 windows (notebook 02). This is a string-match proxy for "states the fact";
  its precision should be audited like G2.
- 4,000 eligible subjects (all splits, in_main questions only) stratified by primary relation x
  popularity bin, assigned 1,000 each to doses 0 / 16 / 64 / 256 injected sentences PER FACT.
  Copies cycle through all of a fact's sentences (wording variety where it exists).
- The old pretraining exposure groups (+0/+1/+4/+16 article copies) are REPLACED for analysis.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from uabr.eval.metrics import alias_contains

DOSES = {"d0": 0, "d16": 16, "d64": 64, "d256": 256}


@dataclass
class ReinjectConfig:
    n_subjects: int = 4000
    doses: Dict[str, int] = field(default_factory=lambda: dict(DOSES))
    fact_share: float = 0.25           # fact-sentence tokens / all continued-pretraining tokens
    min_words: int = 4
    max_words: int = 80
    subj_min_chars: int = 4
    gold_min_chars: int = 3
    max_sentences_per_fact: int = 16
    seed: int = 77

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- sentences
_ABBR = {"st", "dr", "mr", "mrs", "ms", "jr", "sr", "mt", "no", "vs", "co", "inc", "ltd", "u.s", "u.k", "e.g", "i.e", "ca", "gen", "col", "lt", "rev", "prof"}
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        pieces = _SPLIT.split(para)
        buf = ""
        for p in pieces:
            buf = f"{buf} {p}".strip() if buf else p.strip()
            if not buf:
                continue
            toks = buf.rstrip(".!?").split()          # may be empty for pieces like "..." or "?!"
            last = toks[-1].lower() if toks else ""
            # keep merging when the period belongs to an initial ("H.") or an abbreviation ("St.")
            if buf.endswith(".") and last and ((len(last) == 1 and last.isalpha()) or last in _ABBR):
                continue
            out.append(buf); buf = ""
        if buf:
            out.append(buf)
    return out


def fact_sentences_for(texts: Iterable, subj_names: Sequence[str], gold: Sequence[str],
                       cfg: ReinjectConfig) -> List[str]:
    """texts: raw article texts, or already-split sentence lists (list of str per article)."""
    seen, out = set(), []
    for t in texts:
        for s in (split_sentences(t) if isinstance(t, str) else t):
            n = len(s.split())
            if n < cfg.min_words or n > cfg.max_words or s in seen:
                continue
            if alias_contains(s, gold, min_chars=cfg.gold_min_chars) and alias_contains(s, subj_names, min_chars=cfg.subj_min_chars):
                seen.add(s); out.append(s)
                if len(out) >= cfg.max_sentences_per_fact:
                    return out
    return out


# --------------------------------------------------------------------------- doses
def assign_doses(subjects: pd.DataFrame, cfg: ReinjectConfig) -> pd.DataFrame:
    """subjects: one row per eligible subject with columns title_n, primary_prop, pop_bin.
    Returns the n_subjects selected rows with dose_group / dose (balanced within strata)."""
    rng = np.random.default_rng(cfg.seed)
    s = subjects.sort_values("title_n").reset_index(drop=True)
    n = min(cfg.n_subjects, len(s))
    strata = s.groupby(["primary_prop", "pop_bin"]).indices
    # proportional allocation with largest remainder
    quota = {k: len(v) * n / len(s) for k, v in strata.items()}
    base = {k: int(np.floor(q)) for k, q in quota.items()}
    rem = n - sum(base.values())
    for k in sorted(quota, key=lambda k: -(quota[k] - base[k]))[:rem]:
        base[k] += 1
    names = list(cfg.doses)
    chosen, groups = [], []
    j = int(rng.integers(0, len(names)))          # one rotation carried across strata -> exact overall balance
    for k in sorted(strata):
        for i in rng.permutation(strata[k])[: base[k]]:
            chosen.append(i); groups.append(names[j % len(names)]); j += 1
    out = s.loc[chosen].copy()
    out["dose_group"] = groups
    out["dose"] = out["dose_group"].map(cfg.doses).astype(int)
    return out.reset_index(drop=True)


def injection_docs(facts: pd.DataFrame) -> List[str]:
    """facts: rows with `dose` and `sentences` (list). Returns the list of injected sentence docs,
    cycling through each fact's sentences until `dose` copies."""
    docs = []
    for dose, sents in zip(facts["dose"], facts["sentences"]):
        if dose <= 0 or len(sents) == 0:
            continue
        docs.extend(sents[i % len(sents)] for i in range(int(dose)))
    return docs

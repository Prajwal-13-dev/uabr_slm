"""Cloze prompts for probing the plain-Wikipedia base model (notebook 04, G1 pilot).

ASSUMPTION (team did not answer the prompt-format question, 2026-11-09): relation-specific cloze
templates with 3 same-relation TRAIN facts as demonstrations (LAMA-style probing). The evidence
condition puts the retrieved windows before the same prompt. Evidence windows: top-3 if they fit
the context, else top-2, else top-1 (team decision 2026-11-09); the count used is logged.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# PopQA relation -> cloze template ({s} = subject name). Ends right before the answer.
CLOZE_TEMPLATES: Dict[str, str] = {
    "occupation": "{s}'s occupation is",
    "place of birth": "{s} was born in",
    "genre": "The genre of {s} is",
    "father": "The father of {s} is",
    "mother": "The mother of {s} is",
    "country": "{s} is located in the country of",
    "producer": "{s} was produced by",
    "director": "{s} was directed by",
    "screenwriter": "The screenplay of {s} was written by",
    "composer": "The music of {s} was composed by",
    "author": "{s} was written by",
    "capital of": "{s} is the capital of",
    "capital": "The capital of {s} is",
    "color": "The color of {s} is",
    "religion": "The religion of {s} is",
    "sport": "{s} plays the sport of",
}


def cloze(relation: str, subj: str) -> str:
    return CLOZE_TEMPLATES[relation].format(s=subj)


def pick_demos(train: pd.DataFrame, n: int = 3, seed: int = 0) -> Dict[str, List[Tuple[str, str]]]:
    """Fixed demonstrations per relation: n TRAIN facts with n distinct answers (in_main rows only)."""
    rng = np.random.default_rng(seed)
    out = {}
    for rel, g in train[train["in_main"]].groupby("prop"):
        g = g.iloc[rng.permutation(len(g))]
        demos, seen = [], set()
        for r in g.itertuples(index=False):
            if r.answer.lower() not in seen:
                demos.append((r.subj, r.answer)); seen.add(r.answer.lower())
            if len(demos) == n:
                break
        out[rel] = demos
    return out


def demo_block(relation: str, demos: Dict[str, List[Tuple[str, str]]]) -> str:
    return "".join(f"{cloze(relation, s)} {a}.\n" for s, a in demos.get(relation, []))


def closed_prompt(relation: str, subj: str, demos) -> str:
    return demo_block(relation, demos) + cloze(relation, subj)


def evidence_prompt(windows: Sequence[str], relation: str, subj: str, demos) -> str:
    return "".join(w + "\n" for w in windows) + "\n" + closed_prompt(relation, subj, demos)


_CUT = re.compile(r"\n|\.(?:\s|$)|,|;|\s\(")


def clean_generation(text: str) -> str:
    """Answer span = continuation up to the first newline / sentence end / comma / ' ('."""
    return _CUT.split(text.strip(), maxsplit=1)[0].strip()

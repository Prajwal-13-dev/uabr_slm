"""Alias-aware answer scoring used everywhere in the project.

PLAN §6: canonical answer string for likelihood scoring, aliases for correctness.
EM / F1 follow SQuAD-style normalisation, plus accent folding so that
"Beyoncé" == "Beyonce". `alias_contains` mirrors the PopQA paper's
"any gold alias appears in the generation" metric, but ignores very short
aliases (PopQA has aliases such as "pol" for politician) to avoid
substring false positives.
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _remove_punct(s: str) -> str:
    # ASCII punctuation plus any Unicode punctuation category (P*)
    return "".join(
        " " if (ch in string.punctuation or unicodedata.category(ch).startswith("P")) else ch
        for ch in s
    )


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = _strip_accents(str(s)).lower()
    s = _remove_punct(s)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def dedupe_aliases(aliases: Iterable[str]) -> List[str]:
    """Keep first occurrence of each normalised alias; drop empties."""
    seen, out = set(), []
    for a in aliases:
        if a is None:
            continue
        a = str(a).strip()
        n = normalize_answer(a)
        if n and n not in seen:
            seen.add(n)
            out.append(a)
    return out


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def alias_em(pred: str, aliases: Sequence[str]) -> float:
    np_ = normalize_answer(pred)
    return float(any(np_ == normalize_answer(a) for a in aliases)) if np_ else 0.0


def alias_f1(pred: str, aliases: Sequence[str]) -> float:
    return max((token_f1(pred, a) for a in aliases), default=0.0)


def alias_contains(pred: str, aliases: Sequence[str], min_chars: int = 4) -> float:
    """1 if any (sufficiently long) gold alias occurs as a whole-word span in pred."""
    np_ = f" {normalize_answer(pred)} "
    for a in aliases:
        na = normalize_answer(a)
        if len(na) >= min_chars and f" {na} " in np_:
            return 1.0
    return 0.0

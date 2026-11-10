"""Instruction-tuning data for the Phase 5 pilot and the Phase 6 arms (notebook 07).

Sources (same as the uabr-slm-main protocol, docs/pilot_decision.md):
  databricks/databricks-dolly-15k (CC BY-SA 3.0) and yahma/alpaca-cleaned (CC BY 4.0).

Plain SFT format (amendment A5, PRIMARY: no action tokens):
    <BOS><USER> instruction[\n\ncontext]<ASSISTANT> response<EOS>
Loss only on the response tokens and <EOS>. An example is (tokens, is-target flags), the same convention as
uabr.data.qa_format, so instruction and replay examples can share one batch.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from uabr.data.tokenizer import ASSISTANT_ID, BOS_ID, EOS_ID, USER_ID
from uabr.eval.metrics import normalize_answer

Example = Tuple[List[int], List[bool]]

SOURCES = {"dolly15k": "databricks/databricks-dolly-15k", "alpaca-cleaned": "yahma/alpaca-cleaned"}


def rows_from_hf(dolly, alpaca) -> List[Dict]:
    """Unify the two datasets into {id, source, instruction, context, response}; drop empty rows."""
    rows = []
    for i, ex in enumerate(dolly):
        rows.append({"id": f"dolly-{i}", "source": "dolly15k", "instruction": ex["instruction"],
                     "context": ex.get("context") or "", "response": ex["response"]})
    for i, ex in enumerate(alpaca):
        rows.append({"id": f"alpaca-{i}", "source": "alpaca-cleaned", "instruction": ex["instruction"],
                     "context": ex.get("input") or "", "response": ex["output"]})
    return [r for r in rows if str(r["instruction"]).strip() and str(r["response"]).strip()]


def prompt_body(instruction: str, context: str = "") -> str:
    body = instruction.strip()
    if context and context.strip():
        body += "\n\n" + context.strip()
    return body


def sft_example(tok, instruction: str, context: str, response: str) -> Example:
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    p = [BOS_ID, USER_ID] + enc(" " + prompt_body(instruction, context)) + [ASSISTANT_ID]
    r = enc(" " + response.strip()) + [EOS_ID]
    return p + r, [False] * len(p) + [True] * len(r)


def build_sft(tok, rows: Sequence[Dict], max_len: int = 512, budget_tokens: int = 6_000_000, n_val: int = 500,
              seed: int = 2026):
    """Shuffle (seed), drop examples longer than max_len tokens, first n_val -> validation, then training examples
    until budget_tokens (prompt + response tokens) is reached. Returns (train, val, train_ids, val_ids, stats)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    kept, too_long = [], 0
    for r in rows:
        ex = sft_example(tok, r["instruction"], r["context"], r["response"])
        if len(ex[0]) > max_len:
            too_long += 1
            continue
        kept.append((r, ex))
    val = kept[:n_val]
    train, used = [], 0
    for r, ex in kept[n_val:]:
        if used >= budget_tokens:
            break
        train.append((r, ex)); used += len(ex[0])
    src = defaultdict(int)
    for r, _ in train:
        src[r["source"]] += 1
    stats = {"train_examples": len(train), "train_tokens": used,
             "train_supervised_tokens": sum(sum(f) for _, (t, f) in train),
             "val_examples": len(val), "dropped_too_long": too_long, "max_len": max_len,
             "budget_tokens": budget_tokens, "seed": seed, "sources": dict(src)}
    return ([e for _, e in train], [e for _, e in val], [r["id"] for r, _ in train], [r["id"] for r, _ in val], stats)


_WORD = re.compile(r"[a-z0-9]+")


def audit_overlap(texts: Sequence[str], facts: Iterable[Tuple[int, str, Sequence[str]]], min_alias_chars: int = 4) -> Dict[int, List[int]]:
    """Find instruction examples that state a PopQA fact.

    texts: one string per SFT row (instruction + context + response).
    facts: (fact_id, subject, gold_aliases).
    A row is flagged for a fact when the normalised subject AND a normalised gold alias (>= min_alias_chars
    characters) both occur in it as whole-word spans. Returns {row index: [fact ids]}.
    """
    norm = [f" {normalize_answer(t)} " for t in texts]
    post = defaultdict(set)
    for i, t in enumerate(norm):
        for w in set(_WORD.findall(t)):
            post[w].add(i)
    hits: Dict[int, List[int]] = defaultdict(list)
    for fid, subj, aliases in facts:
        s = normalize_answer(subj)
        words = _WORD.findall(s)
        if not words:
            continue
        lists = sorted((post.get(w, set()) for w in set(words)), key=len)
        cand = set(lists[0])
        for l in lists[1:]:
            cand &= l
            if not cand:
                break
        if not cand:
            continue
        al = [f" {normalize_answer(a)} " for a in aliases if len(normalize_answer(a)) >= min_alias_chars]
        for i in cand:
            if f" {s} " in norm[i] and any(a in norm[i] for a in al):
                hits[i].append(fid)
    return dict(hits)

"""Q&A formats for mixed training (notebook 05) and their packing into fixed-length items.

Closed book (ANSWER path):  <BOS><USER> question<ASSISTANT><ANSWER> answer<EOS>
With evidence (CALL path):  <BOS><USER> question<ASSISTANT><CALL><EVIDENCE> w1\nw2\nw3</EVIDENCE> answer<EOS>

Loss only on the answer tokens and <EOS> (never on the question, evidence or the action token, so the
model gets no routing preference here). Items are 512+1 tokens with a per-target mask; LM text items
(injection stream / replay) have every target on.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from uabr.data.tokenizer import (ASSISTANT_ID, ANSWER_ID, BOS_ID, CALL_ID, EOS_ID, EVIDENCE_ID,
                                 EVIDENCE_END_ID, PAD_ID, USER_ID)

Example = Tuple[List[int], List[bool]]     # tokens, is-target flags (same length)


def _enc(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False).ids


def closed_prompt_ids(tok, question: str) -> List[int]:
    return [BOS_ID, USER_ID] + _enc(tok, " " + question) + [ASSISTANT_ID, ANSWER_ID]


def evidence_prompt_ids(tok, question: str, windows: Sequence[str], max_len: int) -> Tuple[List[int], int]:
    """Top-3 windows if they fit in max_len, else top-2, else top-1, else none (team rule D10)."""
    head = [BOS_ID, USER_ID] + _enc(tok, " " + question) + [ASSISTANT_ID, CALL_ID, EVIDENCE_ID]
    for k in range(len(windows), -1, -1):
        body = _enc(tok, " " + "\n".join(windows[:k])) if k else []
        ids = head + body + [EVIDENCE_END_ID]
        if len(ids) <= max_len:
            return ids, k
    return head + [EVIDENCE_END_ID], 0


def with_answer(prompt: List[int], tok, answer: str) -> Example:
    a = _enc(tok, " " + answer) + [EOS_ID]
    return prompt + a, [False] * len(prompt) + [True] * len(a)


def closed_example(tok, question: str, answer: str) -> Example:
    return with_answer(closed_prompt_ids(tok, question), tok, answer)


def evidence_example(tok, question: str, windows: Sequence[str], answer: str, seq_len: int = 512) -> Tuple[Example, int]:
    a_len = len(_enc(tok, " " + answer)) + 1
    prompt, k = evidence_prompt_ids(tok, question, windows, seq_len + 1 - a_len)
    return with_answer(prompt, tok, answer), k


def pack(examples: Sequence[Example], seq_len: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy packing into items of seq_len+1 tokens. Returns X (N, seq_len+1) uint16 and
    M (N, seq_len) bool where M[i, k] = token X[i, k+1] is a training target."""
    T = seq_len + 1
    xs, ms, cur_t, cur_f = [], [], [], []

    def flush():
        if not cur_t:
            return
        t = cur_t + [PAD_ID] * (T - len(cur_t)); f = cur_f + [False] * (T - len(cur_f))
        xs.append(t); ms.append(f[1:])
    for t, f in examples:
        if len(t) > T:
            raise ValueError(f"example of {len(t)} tokens exceeds {T}")
        if len(cur_t) + len(t) > T:
            flush(); cur_t, cur_f = [], []
        cur_t += list(t); cur_f += list(f)
    flush()
    X = np.asarray(xs, dtype=np.uint16); M = np.asarray(ms, dtype=bool)
    assert M.any(axis=1).all(), "every packed item must contain at least one target"
    return X, M


def lm_blocks(stream: np.ndarray, seq_len: int = 512, n: int = 0, rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """LM items from a flat token stream: all non-overlapping blocks (n=0) or n random blocks."""
    T = seq_len + 1
    if n == 0:
        nb = (len(stream) - 1) // seq_len
        X = np.stack([np.asarray(stream[i * seq_len: i * seq_len + T]) for i in range(nb)]).astype(np.uint16)
    else:
        starts = rng.integers(0, len(stream) - T, size=n)
        X = np.stack([np.asarray(stream[s: s + T]) for s in starts]).astype(np.uint16)
    return X, np.ones((len(X), seq_len), dtype=bool)

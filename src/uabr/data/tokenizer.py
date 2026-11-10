"""Byte-level BPE tokenizer (PLAN §4): 12k vocab, 9 special tokens with fixed ids 0..8.

<EOS> and <PAD> were added to the plan's 7 special tokens (team decision 2026-11-09).
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>", "<ANSWER>", "<CALL>",
                  "<EVIDENCE>", "</EVIDENCE>"]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
USER_ID, ASSISTANT_ID, ANSWER_ID, CALL_ID, EVIDENCE_ID, EVIDENCE_END_ID = 3, 4, 5, 6, 7, 8


def train_bpe(texts: Iterable[str], vocab_size: int = 12000, min_frequency: int = 2):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    check_special_ids(tok)
    return tok


def load(path: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    check_special_ids(tok)
    return tok


def check_special_ids(tok) -> None:
    for i, t in enumerate(SPECIAL_TOKENS):
        got = tok.token_to_id(t)
        assert got == i, f"special token {t} has id {got}, expected {i}"


def encode_batch(tok, texts: List[str]) -> List[np.ndarray]:
    """Encode raw texts without special tokens -> list of uint16 arrays."""
    assert tok.get_vocab_size() < 65535, "uint16 storage needs vocab < 65535"
    return [np.asarray(e.ids, dtype=np.uint16) for e in tok.encode_batch(texts, add_special_tokens=False)]


def wrap_doc(ids: np.ndarray) -> np.ndarray:
    """<BOS> ids <EOS> as one pretraining document."""
    return np.concatenate([[BOS_ID], ids, [EOS_ID]]).astype(np.uint16)

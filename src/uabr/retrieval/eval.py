"""Automatic retrieval diagnostics (proxies only; G2 needs the manual support audit)."""
from __future__ import annotations

from typing import Sequence

from uabr.eval.metrics import alias_contains, normalize_answer


def answer_in_window(window: str, gold_aliases: Sequence[str]) -> bool:
    return bool(alias_contains(window, gold_aliases, min_chars=4))


def subject_in_window(window: str, window_title: str, subj_title: str, subj_names: Sequence[str]) -> bool:
    """Window comes from the subject's own article, or names the subject (>=4-char alias)."""
    if normalize_answer(window_title) == normalize_answer(subj_title) and subj_title:
        return True
    return bool(alias_contains(window, subj_names, min_chars=4))

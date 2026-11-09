"""Evidence selection from the cached top-20 (team decision 2026-11-09).

Walk the ranked windows; skip any window that overlaps (same article, start offsets closer than
the window length) a window already chosen; stop at k. Then the overflow rule (top-3 -> 2 -> 1)
is applied by the prompt builder.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def select_non_overlapping(doc_ids: Sequence[int], article: Dict[int, int], start: Dict[int, int],
                           k: int = 3, window_words: int = 60) -> Tuple[List[int], int]:
    chosen: List[int] = []
    skipped = 0
    for d in doc_ids:
        d = int(d)
        if d < 0:
            continue
        if any(article[d] == article[c] and abs(start[d] - start[c]) < window_words for c in chosen):
            skipped += 1
            continue
        chosen.append(d)
        if len(chosen) == k:
            break
    return chosen, skipped

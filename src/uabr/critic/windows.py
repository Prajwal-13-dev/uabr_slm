"""(question, window) items with automatic relevance labels (shared by notebooks 10 and 11)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from uabr.data import qa_format as Q
from uabr.retrieval.select import select_non_overlapping


def window_items(qdf, top, tok, wtext, wart, wstart, k: int, max_len: int, raw: bool = False) -> pd.DataFrame:
    """qdf: id, question, group_id, prop. top: popqa_top20 indexed by id. raw=True -> first k raw BM25 positions
    (as in the G2 audit); else the non-overlapping top-k. Windows that do not fit in max_len are dropped
    (n_dropped in df.attrs)."""
    rows = []
    for r in qdf.itertuples(index=False):
        t = top.loc[r.id]; docs = list(t.doc_ids); pos = {int(d): j for j, d in enumerate(docs)}
        chosen = [int(d) for d in docs[:k] if d >= 0] if raw else select_non_overlapping(docs, wart, wstart, k=k, window_words=60)[0]
        for rank, d in enumerate(chosen):
            j = pos[int(d)]
            rows.append({"id": int(r.id), "group_id": r.group_id, "prop": r.prop, "doc_id": int(d), "rank": rank,
                         "bm25_rank": j, "bm25_score": float(t.scores[j]), "answer_in": bool(t.answer_in[j]),
                         "subject_in": bool(t.subject_in[j]), "own_article": bool(t.own_article[j]),
                         "auto_rel": int(bool(t.answer_in[j]) and bool(t.subject_in[j])), "question": r.question})
    df = pd.DataFrame(rows)
    built = [Q.evidence_prompt_ids(tok, q, [wtext[d]], max_len) for q, d in zip(df.question, df.doc_id)]
    df["prompt"] = [p for p, _ in built]
    fits = np.array([kk == 1 for _, kk in built])
    out = df[fits].reset_index(drop=True)
    out.attrs["n_dropped"] = int((~fits).sum())
    return out

"""Window-level BM25 over the notebook-01 corpus (PLAN §6, §7.2; team decisions 2026-11-09).

Windows: 60 whitespace words, stride 30 (50 % overlap), each prefixed with its article
title: "<title>: <60 words>". Every article yields at least one window.

Index: hashed bag-of-words (sklearn HashingVectorizer: lowercase, >=2-char word tokens,
English stop words removed, 2**24 buckets, no stemming) -> BM25 weights (Lucene idf,
k1=0.9, b=0.4 = Pyserini defaults) stored as a scipy CSC matrix (terms x windows), built
in chunks with a forked process pool so ~25M windows fit in Kaggle RAM.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass
class BM25Config:
    window_words: int = 60
    stride_words: int = 30
    prepend_title: bool = True
    n_features: int = 2 ** 24
    k1: float = 0.9
    b: float = 0.4
    articles_per_task: int = 5000

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- windows
def window_starts(n_words: int, size: int, stride: int) -> List[int]:
    if n_words <= size:
        return [0]
    starts = list(range(0, n_words - size + 1, stride))
    if starts[-1] + size < n_words:
        starts.append(n_words - size)  # final window flush with the end
    return starts


def window_text(title: str, words: Sequence[str], start: int, cfg: BM25Config) -> str:
    body = " ".join(words[start:start + cfg.window_words])
    return f"{title}: {body}" if cfg.prepend_title else body


def make_vectorizer(cfg: BM25Config):
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(n_features=cfg.n_features, lowercase=True, stop_words="english",
                             token_pattern=r"(?u)\b\w\w+\b", alternate_sign=False, norm=None,
                             dtype=np.float32)


# --------------------------------------------------------------------------- parallel build
_G = {}  # globals shared with forked workers (copy-on-write)


def _encode_range(lo_hi: Tuple[int, int]):
    lo, hi = lo_hi
    cfg, titles, texts = _G["cfg"], _G["titles"], _G["texts"]
    vec = make_vectorizer(cfg)
    art, starts, docs = [], [], []
    for a in range(lo, hi):
        words = texts[a].split()
        for s in window_starts(len(words), cfg.window_words, cfg.stride_words):
            art.append(a); starts.append(s); docs.append(window_text(titles[a], words, s, cfg))
    X = vec.transform(docs).tocsr()
    X.sum_duplicates()
    return X, np.asarray(art, np.int32), np.asarray(starts, np.int32)


class BM25Index:
    def __init__(self, cfg: BM25Config):
        self.cfg = cfg
        self.W: Optional[sp.csc_matrix] = None   # windows x terms, BM25 weights (CSC: per-term postings)
        self.doc_article = self.doc_start = None
        self.idf = None
        self._buf = None

    # ---------------------------------------------------------------- build
    def build(self, titles: Sequence[str], texts: Sequence[str], n_jobs: Optional[int] = None, log=print):
        cfg = self.cfg
        _G.update(cfg=cfg, titles=list(titles), texts=list(texts))
        n = len(_G["texts"])
        ranges = [(i, min(n, i + cfg.articles_per_task)) for i in range(0, n, cfg.articles_per_task)]
        n_jobs = n_jobs or os.cpu_count() or 1
        mats, arts, starts = [], [], []
        if n_jobs > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(n_jobs) as pool:
                for i, (X, a, s) in enumerate(pool.imap(_encode_range, ranges)):
                    mats.append(X); arts.append(a); starts.append(s)
                    if i % 50 == 0:
                        log(f"  encoded {ranges[i][1]:,}/{n:,} articles")
        else:
            for r in ranges:
                X, a, s = _encode_range(r); mats.append(X); arts.append(a); starts.append(s)
        _G.clear()
        X = sp.vstack(mats, format="csr", dtype=np.float32); del mats
        self.doc_article = np.concatenate(arts); self.doc_start = np.concatenate(starts)
        N = X.shape[0]
        dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(dl.mean()) if N else 1.0
        df = np.bincount(X.indices, minlength=cfg.n_features).astype(np.float64)
        self.idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
        # BM25 weight per nonzero, computed in row blocks (no nnz-sized row array)
        k1, b = cfg.k1, cfg.b
        for r0 in range(0, N, 1_000_000):
            r1 = min(N, r0 + 1_000_000)
            p0, p1 = X.indptr[r0], X.indptr[r1]
            rows = np.repeat(np.arange(r0, r1), np.diff(X.indptr[r0:r1 + 1]))
            tf = X.data[p0:p1]
            norm = k1 * (1 - b + b * dl[rows] / avgdl)
            X.data[p0:p1] = self.idf[X.indices[p0:p1]] * tf * (k1 + 1) / (tf + norm)
        self.W = X.tocsc(); del X
        self.stats = {"windows": int(N), "articles": n, "avg_terms_per_window": avgdl,
                      "nnz": int(self.W.nnz)}
        log(f"  index: {N:,} windows, nnz {self.W.nnz:,}, avg terms/window {avgdl:.1f}")
        return self

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        q = make_vectorizer(self.cfg).transform([query])
        N = self.W.shape[0]
        if self._buf is None or len(self._buf) != N:
            self._buf = np.zeros(N, dtype=np.float32)
        buf, touched = self._buf, []
        for t, c in zip(q.indices, q.data):
            s, e = self.W.indptr[t], self.W.indptr[t + 1]
            if s == e:
                continue
            idx = self.W.indices[s:e]
            buf[idx] += c * self.W.data[s:e]
            touched.append(idx)
        if not touched:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        allt = np.concatenate(touched)          # may contain duplicates (multi-term matches)
        sc = buf[allt]
        m = min(len(allt), k * (len(touched) + 1))   # a window repeats at most once per term
        pre = np.argpartition(-sc, m - 1)[:m] if m < len(allt) else np.arange(len(allt))
        cand = np.unique(allt[pre])
        csc_ = buf[cand]
        buf[allt] = 0.0
        order = np.lexsort((cand, -csc_))[:k]   # score desc, ties -> lower window id
        return cand[order].astype(np.int64), csc_[order]

    def window(self, doc_id: int, titles: Sequence[str], texts: Sequence[str]) -> str:
        a, s = int(self.doc_article[doc_id]), int(self.doc_start[doc_id])
        return window_text(titles[a], texts[a].split(), s, self.cfg)

    # ---------------------------------------------------------------- io
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        sp.save_npz(os.path.join(path, "weights_csc.npz"), self.W, compressed=False)
        np.savez(os.path.join(path, "meta.npz"), doc_article=self.doc_article, doc_start=self.doc_start, idf=self.idf)

    @classmethod
    def load(cls, path: str, cfg: BM25Config):
        ix = cls(cfg)
        ix.W = sp.load_npz(os.path.join(path, "weights_csc.npz")).tocsc()
        m = np.load(os.path.join(path, "meta.npz"))
        ix.doc_article, ix.doc_start, ix.idf = m["doc_article"], m["doc_start"], m["idf"]
        return ix

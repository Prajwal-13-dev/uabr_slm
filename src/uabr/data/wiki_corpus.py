"""Wikipedia pretraining / retrieval corpus with controlled PopQA-subject exposure.

Team decisions (2026-11-09), see docs/PLAN.md §14:
- Snapshot: HF `wikimedia/wikipedia`, config 20231101.en, pinned revision.
- Corpus: ALL PopQA subject + object articles (forced) + random articles to ~1B natural tokens.
  The same natural articles form the BM25 retrieval corpus (notebook 02).
- Exposure: every PopQA subject article found in the snapshot is assigned, stratified by
  (primary relation x popularity bin), to Control / Low / Medium / High =
  0 / 1 / 4 / 16 ADDITIONAL copies of its first `cap_tokens` tokens. Copies come ON TOP of
  the 1B natural tokens and are shuffled uniformly through the token stream.
- Ledger v0: per PopQA fact, does a gold alias occur in the subject's own article / in the
  repeated prefix. Observational, own-article only (full corpus ledger comes later, PLAN §12.8).
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from uabr.eval.metrics import alias_contains

WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"

EXPOSURE_GROUPS = {"control": 0, "low": 1, "medium": 4, "high": 16}  # additional copies


@dataclass
class CorpusConfig:
    target_natural_tokens: int = 1_000_000_000
    vocab_size: int = 12_000
    cap_tokens: int = 1024                 # length of each repeated copy
    candidate_frac: float = 0.30           # share of non-PopQA articles kept as fill candidates
    min_chars: int = 300                   # drop very short non-PopQA articles
    lm_val_frac: float = 0.005             # held-out fill articles for LM validation loss
    tokenizer_sample_chars: int = 300_000_000
    seed_select: int = 11
    seed_exposure: int = 22
    seed_shuffle: int = 33
    exposure_groups: Dict[str, int] = field(default_factory=lambda: dict(EXPOSURE_GROUPS))
    wiki_repo: str = WIKI_REPO
    wiki_config: str = WIKI_CONFIG
    wiki_revision: str = WIKI_REVISION

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- text utils
_TRAILING_SECTIONS = {"references", "external links", "see also", "further reading", "notes",
                      "bibliography", "sources", "citations", "footnotes", "notes and references"}
_MULTI_NL = re.compile(r"\n{3,}")


def normalize_title(t) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    t = unicodedata.normalize("NFC", str(t)).replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t[:1].upper() + t[1:] if t else t


def clean_article(text: str) -> str:
    """Cut trailing reference-type sections; collapse blank lines."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if i > 0 and ln.strip().lower() in _TRAILING_SECTIONS:
            lines = lines[:i]
            break
    return _MULTI_NL.sub("\n\n", "\n".join(lines)).strip()


def is_disambiguation(title: str, text: str) -> bool:
    head = text[:400].lower()
    return title.endswith("(disambiguation)") or "may refer to:" in head or "may also refer to:" in head


def stable_unit(key: str, salt: int) -> float:
    """Deterministic uniform [0,1) from a string key (independent of Python hash seed)."""
    h = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


# --------------------------------------------------------------------------- PopQA titles
def popqa_titles(popqa: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    subj = {normalize_title(t) for t in popqa["s_wiki_title"]} - {""}
    obj = {normalize_title(t) for t in popqa["o_wiki_title"]} - {""}
    return subj, obj


# --------------------------------------------------------------------------- snapshot access
def list_shards(cfg: CorpusConfig, local_dir: Optional[str] = None) -> List[str]:
    if local_dir:
        return sorted(os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.endswith(".parquet"))
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(cfg.wiki_repo, repo_type="dataset", revision=cfg.wiki_revision)
    return sorted(f for f in files if f.startswith(cfg.wiki_config + "/") and f.endswith(".parquet"))


def fetch_shard(name: str, cfg: CorpusConfig, tmp_dir: str, local: bool) -> str:
    if local:
        return name
    from huggingface_hub import hf_hub_download

    return hf_hub_download(cfg.wiki_repo, name, repo_type="dataset", revision=cfg.wiki_revision,
                           local_dir=tmp_dir)


def scan_shard(path: str, subj_titles: Set[str], obj_titles: Set[str], cfg: CorpusConfig
               ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split one shard into forced PopQA articles and random fill candidates."""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["id", "title", "text"]).to_pandas()
    stats = {"articles": len(t), "chars": int(t["text"].str.len().sum())}
    t["title_n"] = t["title"].map(normalize_title)
    t["is_subj"] = t["title_n"].isin(subj_titles)
    t["is_obj"] = t["title_n"].isin(obj_titles)
    forced_m = t["is_subj"] | t["is_obj"]

    forced = t[forced_m].copy()
    forced["text"] = forced["text"].map(clean_article)

    rest = t[~forced_m]
    u = rest["id"].astype(str).map(lambda k: stable_unit(k, cfg.seed_select))
    cand = rest[u < cfg.candidate_frac].copy()
    cand["u"] = u[u < cfg.candidate_frac]
    keep = ~np.array([is_disambiguation(a, b) for a, b in zip(cand["title"], cand["text"])], dtype=bool)
    cand = cand[keep]
    cand["text"] = cand["text"].map(clean_article)
    cand = cand[cand["text"].str.len() >= cfg.min_chars]
    cand["lm_val"] = cand["id"].astype(str).map(lambda k: stable_unit(k, cfg.seed_select + 1) < cfg.lm_val_frac)
    for d in (forced, cand):
        d["n_chars"] = d["text"].str.len()
    stats.update(forced=len(forced), candidates=len(cand))
    cols = ["id", "title", "title_n", "text", "n_chars"]
    return forced[cols + ["is_subj", "is_obj"]], cand[cols + ["u", "lm_val"]], stats


# --------------------------------------------------------------------------- selection
def select_fill(cand_meta: pd.DataFrame, char_budget: float) -> pd.DataFrame:
    """Lowest-u candidates (excluding LM-val) until the char budget is met; LM-val kept separately."""
    train = cand_meta[~cand_meta["lm_val"]].sort_values("u")
    cum = train["n_chars"].cumsum()
    n = int(np.searchsorted(cum.to_numpy(), char_budget, side="left")) + 1
    if n > len(train):
        raise ValueError(f"candidate pool too small: need {char_budget:,.0f} chars, have {cum.iloc[-1]:,.0f}."
                         " Increase candidate_frac.")
    return train.iloc[:n]


# --------------------------------------------------------------------------- exposure
def assign_exposure(popqa: pd.DataFrame, found_titles: Set[str], cfg: CorpusConfig) -> pd.DataFrame:
    """One row per PopQA subject article found in the corpus, with its exposure group."""
    p = popqa.copy()
    p["title_n"] = p["s_wiki_title"].map(normalize_title)
    p = p[p["title_n"].isin(found_titles)]
    subj = (p.groupby("title_n")
              .agg(primary_prop=("prop", lambda s: s.value_counts().index[0]),
                   pop_bin=("pop_bin", "first"), n_questions=("id", "size"),
                   subj_ids=("subj_id", lambda s: sorted(set(int(v) for v in s))))
              .reset_index().sort_values("title_n").reset_index(drop=True))
    names = list(cfg.exposure_groups)
    rng = np.random.default_rng(cfg.seed_exposure)
    group = pd.Series(index=subj.index, dtype=object)
    for _, idx in subj.groupby(["primary_prop", "pop_bin"]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        offset = int(rng.integers(0, len(names)))
        for j, i in enumerate(idx):
            group[i] = names[(j + offset) % len(names)]
    subj["exposure_group"] = group
    subj["extra_copies"] = subj["exposure_group"].map(cfg.exposure_groups).astype(int)
    return subj


# --------------------------------------------------------------------------- ledger v0
def ledger_v0(popqa: pd.DataFrame, articles: Dict[str, str], capped_text: Dict[str, str],
              exposure: pd.DataFrame) -> pd.DataFrame:
    """Per PopQA question: is a gold alias present in the subject's own article / repeated prefix."""
    grp = exposure.set_index("title_n")
    rows = []
    for r in popqa.itertuples(index=False):
        t = normalize_title(r.s_wiki_title)
        art = articles.get(t)
        found = art is not None
        in_art = alias_contains(art, r.gold_aliases) if found else 0.0
        in_cap = alias_contains(capped_text.get(t, ""), r.gold_aliases) if found else 0.0
        g = grp.loc[t] if t in grp.index else None
        extra = int(g["extra_copies"]) if g is not None else 0
        rows.append({"id": r.id, "subj_title": t, "subj_found": found,
                     "exposure_group": g["exposure_group"] if g is not None else "missing",
                     "extra_copies": extra, "obj_in_article": in_art, "obj_in_capped": in_cap,
                     "est_fact_exposures": in_art + extra * in_cap})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- token writer
class TokenWriter:
    """Append uint16 documents to a flat .bin file and keep a document index."""

    def __init__(self, path: str):
        self.path, self.f, self.n = path, open(path, "wb"), 0
        self.index: List[Tuple[int, int, str, str]] = []  # (offset, length, kind, ref)

    def write(self, arr: np.ndarray, kind: str, ref: str):
        self.f.write(arr.astype(np.uint16).tobytes())
        self.index.append((self.n, len(arr), kind, ref))
        self.n += len(arr)

    def close(self) -> pd.DataFrame:
        self.f.close()
        return pd.DataFrame(self.index, columns=["offset", "length", "kind", "ref"])

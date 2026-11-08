"""PopQA loading, cleaning, grouping, splitting, pilot selection and the
relation-prior baseline (PLAN §3, §7.1, §8; amendment A9).

Everything here is deterministic given (data, config, seed).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from uabr.eval.metrics import alias_em, dedupe_aliases, normalize_answer

HF_DATASET = "akariasai/PopQA"
LIST_FIELDS = ("s_aliases", "o_aliases", "possible_answers")


# --------------------------------------------------------------------------- config
@dataclass
class SplitConfig:
    seed: int = 1234
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    n_folds: int = 20                       # 14/3/3 folds -> 70/15/15
    n_pop_bins: int = 5                     # quantile bins of log10(s_pop)
    ood_relations: Tuple[str, ...] = ("religion", "mother", "capital of")
    pilot_size: int = 500
    pilot_min_per_relation: int = 15
    majority_candidates: int = 30           # top-k frequent objects tried as the relation prior

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ood_relations"] = list(self.ood_relations)
        return d


# --------------------------------------------------------------------------- loading
def load_popqa(source: Optional[str] = None, revision: Optional[str] = None) -> pd.DataFrame:
    """Load PopQA from a local file (csv/tsv/parquet/jsonl) or the HF Hub."""
    if source:
        if source.endswith(".parquet"):
            return pd.read_parquet(source)
        if source.endswith(".jsonl"):
            return pd.read_json(source, lines=True)
        sep = "\t" if source.endswith(".tsv") else ","
        return pd.read_csv(source, sep=sep)
    from datasets import load_dataset  # imported lazily so tests don't need it

    ds = load_dataset(HF_DATASET, split="test", revision=revision)
    return ds.to_pandas()


def hub_revision_sha() -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(HF_DATASET).sha
    except Exception:
        return None


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of (id, question, possible_answers) - detects silent data changes."""
    h = hashlib.sha256()
    for r in df.sort_values("id")[["id", "question", "possible_answers"]].itertuples(index=False):
        h.update(f"{r[0]}\x1f{r[1]}\x1f{json.dumps(r[2], ensure_ascii=False)}\n".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- cleaning
def _parse_list(x) -> List[str]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v) for v in x]
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return [str(t) for t in v] if isinstance(v, list) else [str(v)]
    except json.JSONDecodeError:
        # fall back for python-repr lists
        import ast

        try:
            v = ast.literal_eval(s)
            return [str(t) for t in v] if isinstance(v, (list, tuple)) else [str(v)]
        except Exception:
            return [s]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Parse list fields, build gold alias lists, add log-popularity."""
    df = df.copy()
    for c in LIST_FIELDS:
        df[c] = df[c].map(_parse_list)
    # gold = canonical obj first, then PopQA's possible_answers, de-duplicated
    df["gold_aliases"] = [dedupe_aliases([o] + list(pa)) for o, pa in zip(df["obj"], df["possible_answers"])]
    df["answer"] = df["obj"].astype(str)  # canonical string for likelihood scoring
    df["n_aliases"] = df["gold_aliases"].map(len)
    df["log_pop"] = np.log10(df["s_pop"].clip(lower=1).astype(float))
    df["q_norm"] = df["question"].map(normalize_answer)
    # Same question text for different subjects (homonyms) -> unanswerable closed-book
    n_subj = df.groupby("q_norm")["subj_id"].transform("nunique")
    df["ambiguous_question"] = n_subj > 1
    return df


# --------------------------------------------------------------------------- grouping
class _UnionFind:
    def __init__(self):
        self.p: Dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """group_id = connected component over (subj_id) and (normalised question text).

    Grouping by subject alone is not enough: homonymous subjects with identical
    question text could otherwise straddle train/test.
    """
    uf = _UnionFind()
    for sid, q in zip(df["subj_id"], df["q_norm"]):
        uf.union(("s", sid), ("q", q))
    roots = [uf.find(("s", sid)) for sid in df["subj_id"]]
    codes = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    df = df.copy()
    df["group_id"] = [codes[r] for r in roots]
    return df


def add_pop_bins(df: pd.DataFrame, n_bins: int) -> Tuple[pd.DataFrame, List[float]]:
    edges = np.quantile(df["log_pop"], np.linspace(0, 1, n_bins + 1)).tolist()
    inner = edges[1:-1]
    df = df.copy()
    df["pop_bin"] = np.searchsorted(inner, df["log_pop"], side="right").astype(int)
    return df, edges


# --------------------------------------------------------------------------- splitting
def assign_splits(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    is_ood = df["prop"].isin(cfg.ood_relations)
    ind = df[~is_ood]

    n_train = round(cfg.n_folds * cfg.train_frac)
    n_val = round(cfg.n_folds * cfg.val_frac)
    assert n_train + n_val < cfg.n_folds, "fractions leave no test folds"

    strata = ind["prop"].astype(str) + "|" + ind["pop_bin"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold = pd.Series(-1, index=ind.index)
    for k, (_, test_idx) in enumerate(sgkf.split(ind, strata, groups=ind["group_id"])):
        fold.iloc[test_idx] = k

    # randomise which folds become train/val/test (folds are ordered by size otherwise)
    order = np.random.default_rng(cfg.seed).permutation(cfg.n_folds)
    role = {}
    for pos, k in enumerate(order):
        role[k] = "train" if pos < n_train else ("val" if pos < n_train + n_val else "test")

    df["fold"] = -1
    df.loc[ind.index, "fold"] = fold
    df["split"] = "ood"
    df.loc[ind.index, "split"] = fold.map(role)

    # OOD rows whose subject-group also appears in training (name-recognition risk)
    train_groups = set(df.loc[df["split"] == "train", "group_id"])
    df["group_in_train"] = df["group_id"].isin(train_groups)
    return df


def select_pilot(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """500 validation-only questions, stratified by relation (with a floor) and popularity."""
    rng = np.random.default_rng(cfg.seed + 1)
    val = df[df["split"] == "val"]
    counts = val["prop"].value_counts()
    # proportional allocation with a per-relation floor, then rescale to pilot_size
    alloc = {r: max(cfg.pilot_min_per_relation, c * cfg.pilot_size / len(val)) for r, c in counts.items()}
    scale = cfg.pilot_size / sum(alloc.values())
    alloc = {r: min(int(counts[r]), max(1, int(round(a * scale)))) for r, a in alloc.items()}
    # fix rounding so total == pilot_size
    diff = cfg.pilot_size - sum(alloc.values())
    for r in sorted(counts.index, key=lambda r: -counts[r]):
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        if 1 <= alloc[r] + step <= counts[r]:
            alloc[r] += step
            diff -= step

    chosen = []
    for r, n in alloc.items():
        sub = val[val["prop"] == r]
        # within relation: proportional over popularity bins (largest remainder)
        bins = sub["pop_bin"].value_counts()
        quota = (bins / bins.sum() * n)
        base = np.floor(quota).astype(int)
        rem = n - base.sum()
        for b in (quota - base).sort_values(ascending=False).index[:rem]:
            base[b] += 1
        for b, q in base.items():
            pool = sub[sub["pop_bin"] == b].index.to_numpy()
            if q > 0:
                chosen.extend(rng.choice(pool, size=min(q, len(pool)), replace=False).tolist())
    df = df.copy()
    df["is_pilot"] = df.index.isin(chosen)
    return df


# --------------------------------------------------------------------------- relation prior
def relation_majority(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Best constant answer per relation, chosen on TRAIN with alias-aware EM (PLAN §7.1).

    OOD relations have no training rows; their prior is chosen on the OOD rows
    themselves (an optimistic, oracle prior - flagged in the table).
    Returns (per-relation table, per-row `prior_correct` series).
    """
    rows, prior_correct = [], pd.Series(0.0, index=df.index)
    for rel, sub in df.groupby("prop"):
        fit = sub[sub["split"] == "train"]
        oracle = fit.empty
        if oracle:
            fit = sub
        cands = fit["answer"].map(str).value_counts().index[: cfg.majority_candidates]
        best, best_acc = None, -1.0
        for c in cands:
            acc = float(np.mean([alias_em(c, g) for g in fit["gold_aliases"]]))
            if acc > best_acc:
                best, best_acc = c, acc
        hit = sub["gold_aliases"].map(lambda g, b=best: alias_em(b, g))
        prior_correct.loc[sub.index] = hit
        rec = {"relation": rel, "prior_answer": best, "prior_fit_on": "ood(oracle)" if oracle else "train",
               "n_distinct_answers": sub["answer"].nunique(), "n": len(sub)}
        for s in ("train", "val", "test", "ood"):
            m = sub["split"] == s
            rec[f"acc_{s}"] = float(hit[m].mean()) if m.any() else np.nan
            rec[f"n_{s}"] = int(m.sum())
        rec["acc_pilot"] = float(hit[sub["is_pilot"]].mean()) if sub["is_pilot"].any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("relation").reset_index(drop=True), prior_correct


# --------------------------------------------------------------------------- checks
def split_checks(df: pd.DataFrame, cfg: SplitConfig, expected_rows: int = 14267) -> List[dict]:
    """Return a list of {name, passed, detail, severity} checks for the gate cell."""
    out = []

    def add(name, ok, detail, severity="fail"):
        out.append({"name": name, "passed": bool(ok), "detail": detail, "severity": severity})

    add("row_count", len(df) == expected_rows, f"{len(df)} rows (expected {expected_rows})", "warn")
    ind = df[df["split"] != "ood"]
    g_split = ind.groupby("group_id")["split"].nunique()
    add("no_group_leak", (g_split == 1).all(), f"{int((g_split > 1).sum())} groups span >1 split")
    s_split = ind.groupby("subj_id")["split"].nunique()
    add("no_subject_leak", (s_split == 1).all(), f"{int((s_split > 1).sum())} subjects span >1 split")
    add("ood_relations_isolated",
        set(df.loc[df["split"] == "ood", "prop"]) == set(cfg.ood_relations) & set(df["prop"])
        and not ind["prop"].isin(cfg.ood_relations).any(),
        f"ood relations: {sorted(set(df.loc[df['split']=='ood','prop']))}")
    pilot = df[df["is_pilot"]]
    add("pilot_size", len(pilot) == cfg.pilot_size, f"{len(pilot)} pilot questions")
    add("pilot_in_val_only", (pilot["split"] == "val").all(), f"splits: {pilot['split'].value_counts().to_dict()}")
    add("pilot_not_in_test", not (pilot["split"] == "test").any(), "A9")
    add("gold_aliases_nonempty", (df["n_aliases"] > 0).all(), f"{int((df['n_aliases']==0).sum())} empty")

    frac = ind["split"].value_counts(normalize=True)
    target = {"train": cfg.train_frac, "val": cfg.val_frac, "test": cfg.test_frac}
    dev = {s: round(float(frac.get(s, 0) - t), 4) for s, t in target.items()}
    add("overall_fractions_within_2pp", all(abs(v) <= 0.02 for v in dev.values()), f"deviation {dev}")

    worst = 0.0
    for rel, sub in ind.groupby("prop"):
        if len(sub) < 100:
            continue
        f = sub["split"].value_counts(normalize=True)
        worst = max(worst, max(abs(f.get(s, 0) - t) for s, t in target.items()))
    add("per_relation_fractions_within_5pp", worst <= 0.05, f"worst deviation {worst:.3f}", "warn")

    pb = pd.crosstab(ind["split"], ind["pop_bin"], normalize="index")
    pb_dev = float((pb - pb.loc["train"]).abs().max().max())
    add("popularity_balance_within_3pp", pb_dev <= 0.03, f"max pop-bin share gap vs train {pb_dev:.3f}", "warn")
    return out

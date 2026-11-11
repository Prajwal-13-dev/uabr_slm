"""Replay selection for Phase 6 (notebook 08). Team decisions (Nov 11 2026):

* Replay examples are ACTION-TOKEN ONLY: prompt <BOS><USER> q<ASSISTANT>, loss = 2-logit CE over {<ANSWER>, <CALL>}
  at the last position (same loss as the router), target = the frozen theta0 label y = 1[b_quality - lambda > 0].
* Pool = the half-B questions the router was trained on, scored once with theta_route,preSFT.
* Budget (as uabr-slm-main configs/replay.yaml): each method selects distinct examples worth 25 % of the pool's
  tokens, then cycles them (fixed shuffled order) up to 5 % of the instruction tokens.
* Class ratio: identical for every arm = the pool's ANSWER:CALL token share; each method ranks WITHIN each class.

Priorities (higher = replayed first):
  E2 random   seeded uniform
  E3 casr     theta0's own answer uncertainty (first-token entropy of its closed-book generation)  [Hybrid-CASR-adapted]
  E5 bpr      -m                         (m = |z_CALL - z_ANSWER| of theta_route,preSFT; boundary only)
  E6 benefit  |b_net|                    (b_net = b_quality - lambda; benefit only)
  E7 uabr     exp(-m / tau) * |b_net|    (tau = median m of the pool)
  E4 opr      on-policy: every N updates the CURRENT model scores the pool; prompts whose current decision equals the
              frozen label pass the frozen-utility filter; a random selection worth 25 % of the pool tokens (same class
              ratio) is cycled until the next refresh.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

STATIC = ("random", "casr", "bpr", "benefit", "uabr")
ARMS = {"E2": "random", "E3": "casr", "E4": "opr", "E5": "bpr", "E6": "benefit", "E7": "uabr"}


def priority(pool: pd.DataFrame, method: str, seed: int = 0, tau: Optional[float] = None) -> np.ndarray:
    """pool needs columns margin, b_net, unc (and nothing else)."""
    if method == "random":
        return np.random.default_rng(seed).random(len(pool))
    if method == "casr":
        u = pool["unc"].astype(float)
        return u.fillna(u.max()).to_numpy()
    if method == "bpr":
        return -pool["margin"].to_numpy(float)
    if method == "benefit":
        return pool["b_net"].abs().to_numpy(float)
    if method == "uabr":
        t = float(np.median(pool["margin"])) if tau is None else float(tau)
        return np.exp(-pool["margin"].to_numpy(float) / max(t, 1e-6)) * pool["b_net"].abs().to_numpy(float)
    raise ValueError(f"unknown method {method!r}")


def class_budgets(pool: pd.DataFrame, select_tokens: int) -> Dict[int, int]:
    """Token budget per target class, proportional to the class's token share in the pool."""
    tot = pool["tokens"].sum()
    return {c: int(round(select_tokens * pool.loc[pool.y == c, "tokens"].sum() / tot)) for c in (0, 1)}


def select_by_class(pool: pd.DataFrame, score: np.ndarray, budgets: Dict[int, int]) -> pd.DataFrame:
    """Within each class, take candidates in descending score (ties: ascending id) until the class budget would be
    exceeded (at least one per non-empty class). Returns the distinct selection with a `score` column."""
    p = pool.assign(score=score)
    out = []
    for c, b in budgets.items():
        g = p[p.y == c].sort_values(["score", "id"], ascending=[False, True])
        tot, keep = 0, []
        for r in g.itertuples(index=False):
            if keep and tot + r.tokens > b:
                break
            keep.append(r); tot += r.tokens
        if keep:
            out.append(pd.DataFrame(keep))
    return pd.concat(out, ignore_index=True) if out else p.iloc[:0]


class Cycler:
    """Infinite cycle over a fixed selection in a seeded shuffled order (placement independent of rank)."""

    def __init__(self, sel: pd.DataFrame, seed: int):
        self.rows = sel.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        self.k = 0

    def next(self):
        r = self.rows.iloc[self.k % len(self.rows)]
        self.k += 1
        return r


class ReplaySchedule:
    """Equal replay tokens per arm: at update t (0-based) rows are drawn until the cumulative replay tokens reach
    (t + 1) * replay_tokens / total_steps. `source()` returns the current Cycler (static arms: always the same;
    OPR: replaced at every refresh). Returns (prompts, targets) per step and keeps a log."""

    def __init__(self, replay_tokens: int, total_steps: int):
        self.q = replay_tokens / total_steps
        self.total_steps = total_steps
        self.done_tokens = 0
        self.rows = 0
        self.targets = {0: 0, 1: 0}
        self.ids = set()

    def take(self, step: int, cycler: Cycler):
        prompts, targets = [], []
        goal = (step + 1) * self.q
        while self.done_tokens < goal:
            r = cycler.next()
            prompts.append(list(r.prompt)); targets.append(int(r.y))
            self.done_tokens += int(r.tokens); self.rows += 1; self.targets[int(r.y)] += 1; self.ids.add(int(r.id))
        return prompts, targets

    def report(self) -> Dict:
        return {"replay_tokens": int(self.done_tokens), "replay_rows": self.rows,
                "targets_answer": self.targets[0], "targets_call": self.targets[1], "distinct_prompts": len(self.ids)}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)

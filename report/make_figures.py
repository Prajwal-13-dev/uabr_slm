"""Make every figure of the UABR-SLM report.

    python report/make_figures.py --root .            # run from the repo root
Reads the result files in data/runs/{07,10,11,12} (copied from the Kaggle outputs) and report/data/*.csv
(numbers transcribed from the Kaggle logs of notebooks 08 and 09). Writes report/figures/*.png (300 dpi, print-ready).
If data/runs/09/test_routing.csv exists (download it from the 09 output), it replaces report/data/test_routing.csv,
which fills in the confidence intervals that the log did not show.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reference palette (validated categorical slots 1-3) + neutral greys for reference models
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY_DARK, GREY_LIGHT = "#6f6e69", "#b9b8b1"
FAMILY = {"E0": GREY_DARK, "E1": GREY_LIGHT, "A1": BLUE, "A2": BLUE, "B": BLUE,
          "E2": ORANGE, "E3": ORANGE, "E4": ORANGE, "E5": ORANGE, "E6": ORANGE, "E7": ORANGE,
          "interp0.5": AQUA, "taskvec1": AQUA}
NAMES = {"E0": "E0 pre-SFT router", "E1": "E1 plain SFT", "A1": "A1 SFT-prompt anchoring", "A2": "A2 question anchoring",
         "B": "B readout repair", "E2": "E2 random replay", "E3": "E3 Hybrid-CASR-adapted", "E4": "E4 OPR-adapted",
         "E5": "E5 boundary-only (BPR)", "E6": "E6 benefit-only", "E7": "E7 UABR", "interp0.5": "Interpolation α=0.5",
         "taskvec1": "Task vector β=1"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, out, name):
    p = os.path.join(out, name); fig.savefig(p); plt.close(fig); print("wrote", p)


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(8.5, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)
    boxes = [
        (1, 30, "00 PopQA splits\ngrouped 70/15/15"), (17, 30, "01 Wikipedia 1.0B tok\nBPE 12k"), (33, 30, "02 BM25\n60-word windows"),
        (49, 30, "03 pretrain 40M\n(03b fact re-inject)"), (65, 30, "05 mixed training\nθ0 = arm M"), (81, 30, "06 benefit labels\n+ router → θ_route"),
        (81, 12, "07 plain SFT pilot\nGO decision"), (65, 12, "08 replay arms\nE2–E7 (UABR)"), (49, 12, "10 critic token\n<REL>/<IRR>"),
        (33, 12, "11 post-hoc: readout,\ninterpolation, recal."), (17, 12, "12 anchoring A1/A2\nreadout repair B"), (1, 12, "09 sealed TEST\n(run once)")]
    for x, y, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), 15, 11, facecolor="#f4f3f0", edgecolor=INK2, linewidth=0.8))
        ax.text(x + 7.5, y + 5.5, t, ha="center", va="center", fontsize=5.4, color=INK)
    arrow = dict(arrowstyle="->", color=INK2, lw=0.8)
    for i in range(5):
        ax.annotate("", xy=(boxes[i + 1][0], 35.5), xytext=(boxes[i][0] + 15, 35.5), arrowprops=arrow)
    ax.annotate("", xy=(88.5, 23), xytext=(88.5, 30), arrowprops=arrow)
    for i in range(6, 11):
        ax.annotate("", xy=(boxes[i + 1][0] + 15, 17.5), xytext=(boxes[i][0], 17.5), arrowprops=arrow)
    ax.text(50, 3.5, "Validation decides every design choice; the test split is opened once at the end (notebook 09).",
            ha="center", fontsize=7, color=INK2)
    save(fig, out, "fig01_pipeline.png")


def fig_sft_trajectory(root, out):
    t = pd.read_csv(os.path.join(root, "data/runs/07/pilot_table.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, col, lab in zip(axs, ["auroc", "default_call_rate", "closed_em"], ["Routing AUROC", "Default call rate", "Closed-book EM"]):
        base = t[t.seed.astype(str) == "all"][col].item()
        for seed, mk in [("11", "o"), ("29", "s")]:
            d = t[t.seed.astype(str) == seed].sort_values("frac")
            x = [0] + list(d.frac * 100); y = [base] + list(d[col])
            ax.plot(x, y, color=BLUE if seed == "11" else ORANGE, marker=mk, ms=4, lw=1.6, label=f"seed {seed}")
        ax.set_title(lab, fontsize=9, color=INK); ax.set_xlabel("% of SFT updates")
        if col != "auroc": ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axs[0].legend(fontsize=7)
    fig.suptitle("Plain SFT damages the pre-SFT routing policy (validation, notebook 07)", fontsize=9.5, y=1.04)
    save(fig, out, "fig02_sft_damage.png")


def fig_flips(root, out):
    a = pd.read_csv(os.path.join(root, "data/runs/07/a4_flips_by_margin.csv"))
    a = a[(a.boundary.astype(str) == "0") & (a.frac == 1.0)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    w = 0.38
    for i, (seed, c) in enumerate([(11, BLUE), (29, ORANGE)]):
        d = a[a.seed == seed].sort_values("bin")
        ax.bar(d.bin + (i - 0.5) * w, d.flip_rate, width=w - 0.04, color=c, label=f"seed {seed}")
    ax.set_xlabel("Pre-SFT margin quintile (1 = closest to boundary)"); ax.set_ylabel("Decisions flipped by SFT")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.legend(fontsize=7)
    ax.set_title("Small margins flip most (A4, validation)", fontsize=9)
    save(fig, out, "fig03_flips_by_margin.png")


def fig_replay_val(out):
    r = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/replay_val_08.csv"))
    order = ["E1", "E5", "E7", "E3", "E6", "E2", "E4"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, arm in enumerate(order):
        d = r[r.arm == arm]
        ax.scatter([i] * len(d), d.auroc, color=FAMILY[arm], s=26, zorder=3, edgecolor="white", linewidth=0.8)
        ax.plot([i - 0.25, i + 0.25], [d.auroc.mean()] * 2, color=INK, lw=1.2)
    ax.axhline(0.838, color=GREY_DARK, ls="--", lw=1); ax.text(6.4, 0.842, "E0 pre-SFT", ha="right", fontsize=7, color=INK2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(["E1\nSFT", "E5\nBPR", "E7\nUABR", "E3\nCASR", "E6\nbenefit", "E2\nrandom", "E4\nOPR"], fontsize=7)
    ax.set_ylabel("Routing AUROC (validation)"); ax.set_title("Replay arms, equal budget (notebook 08)", fontsize=9)
    save(fig, out, "fig04_replay_val.png")


def load_test(root):
    p = os.path.join(root, "data/runs/09/test_routing.csv")
    if os.path.exists(p):
        t = pd.read_csv(p); t["source"] = "09 output"; return t
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "data/test_routing.csv"))


def fig_test_auroc(root, out):
    t = load_test(root); t["seed"] = t.seed.astype(str)
    order = ["E1", "E5", "E7", "E3", "taskvec1", "E6", "E2", "interp0.5", "E4", "A1", "A2"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, m in enumerate(order):
        for j, s in enumerate(["11", "29"]):
            d = t[(t.model == m) & (t.seed == s)]
            if d.empty: continue
            r = d.iloc[0]; y = i + (j - 0.5) * 0.32
            if pd.notna(r.get("lo")):
                ax.plot([r.lo, r.hi], [y, y], color=FAMILY[m], lw=1.6)
            ax.scatter(r.auroc, y, color=FAMILY[m], s=24, marker="o" if s == "11" else "s", zorder=3, edgecolor="white", linewidth=0.6)
    e0 = t[t.model == "E0"].iloc[0]
    ax.axvspan(e0.lo, e0.hi, color=GREY_LIGHT, alpha=0.35, lw=0); ax.axvline(e0.auroc, color=GREY_DARK, ls="--", lw=1)
    ax.text(e0.auroc + 0.002, len(order) - 0.4, "E0 pre-SFT (95% CI)", fontsize=7, color=INK2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[m] for m in order], fontsize=7.5)
    ax.set_xlabel("Routing AUROC on TEST vs frozen θ0 labels (● seed 11, ■ seed 29; bars = 95% clustered CI)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="new: anchoring"), Patch(color=ORANGE, label="replay (08)"),
                       Patch(color=AQUA, label="no-training references"), Patch(color=GREY_LIGHT, label="plain SFT")],
              loc="lower right", fontsize=7)
    ax.set_title("Final test result (notebook 09)", fontsize=9.5)
    save(fig, out, "fig05_test_auroc.png")


def fig_forest(root, out):
    p = os.path.join(root, "data/runs/09/test_comparisons.csv")
    c = pd.read_csv(p).rename(columns=lambda k: k.replace("_ci_lo", "_lo").replace("_ci_hi", "_hi")) if os.path.exists(p) \
        else pd.read_csv(os.path.join(os.path.dirname(__file__), "data/test_comparisons.csv"))
    c = c[c.family.isin(["primary", "original_hypothesis", "critic"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    n = len(c)
    for i, r in c.iterrows():
        y = n - 1 - i
        col = BLUE if r.family == "primary" else (ORANGE if r.family == "original_hypothesis" else AQUA)
        for j, s in enumerate(["s11", "s29"]):
            yy = y + (j - 0.5) * 0.3
            ax.plot([r[f"{s}_lo"], r[f"{s}_hi"]], [yy, yy], color=col, lw=1.6)
            ax.scatter(r[f"{s}_diff"], yy, color=col, s=18, marker="o" if s == "s11" else "s", zorder=3, edgecolor="white", linewidth=0.5)
        ax.text(1.01, y, r.verdict, fontsize=6.6, va="center", color=INK2, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(n)); ax.set_yticklabels(list(c.comparison[::-1]), fontsize=7.5)
    ax.set_xlim(-0.13, 0.175); ax.set_xlabel("Difference in AUROC (paired clustered bootstrap, 2,000 resamples; ● seed 11, ■ seed 29)")
    ax.set_title("Pre-registered test comparisons (blue: primary family, orange: original UABR hypothesis, aqua: critic IsRel)", fontsize=8.5)
    save(fig, out, "fig06_test_forest.png")


def fig_interpolation(root, out):
    t = pd.read_csv(os.path.join(root, "data/runs/11/interpolation.csv")); t = t[t.kind == "interp"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for seed, mk in [(11, "o"), (29, "s")]:
        d = t[t.seed == seed].sort_values("alpha")
        axs[0].plot(d.alpha, d.route_auroc, color=BLUE, marker=mk, ms=3.5, lw=1.4, label=f"routing AUROC, seed {seed}")
        axs[0].plot(d.alpha, d.sft_gain_kept, color=ORANGE, marker=mk, ms=3.5, lw=1.4, label=f"SFT gain kept, seed {seed}")
        axs[1].plot(d.alpha, d.closed_em, color=BLUE, marker=mk, ms=3.5, lw=1.4, label=f"closed-book EM, seed {seed}")
        axs[1].plot(d.alpha, d.evidence_em, color=AQUA, marker=mk, ms=3.5, lw=1.4, label=f"evidence EM, seed {seed}")
    axs[0].set_xlabel("α  (θ = θ_route + α(θ_SFT − θ_route))"); axs[0].set_title("Ranking vs SFT gain", fontsize=9)
    axs[1].set_xlabel("α"); axs[1].set_title("Answer paths", fontsize=9); axs[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in axs: ax.legend(fontsize=6); ax.axvline(0.5, color=GREY_LIGHT, ls=":", lw=1)
    fig.suptitle("Weight interpolation: SFT gain is front-loaded, damage back-loaded (validation, notebook 11)", fontsize=9.5, y=1.04)
    save(fig, out, "fig07_interpolation.png")


def fig_readout(root, out):
    r = pd.read_csv(os.path.join(root, "data/runs/11/readout_routing.csv")); r["seed"] = r.seed.astype(str)
    lc = pd.read_csv(os.path.join(root, "data/runs/11/probe_learning_curve.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    e0 = r[r.model == "E0"].iloc[0]
    for seed, mk in [("11", "o"), ("29", "s")]:
        d = r[r.seed == seed].sort_values("frac")
        x = [0] + list(d.frac * 100)
        for col, c, lab in [("token", GREY_DARK, "token readout z_CALL−z_ANSWER"), ("refit_probe", BLUE, "re-fitted linear probe"),
                            ("frozen_pre_probe", ORANGE, "pre-SFT probe, unchanged")]:
            axs[0].plot(x, [e0[col]] + list(d[col]), color=c, marker=mk, ms=3.5, lw=1.4, label=lab if seed == "11" else None)
    axs[0].axhline(0.5, color=GREY_LIGHT, lw=0.8); axs[0].set_xlabel("% of SFT updates"); axs[0].set_ylabel("Routing AUROC (validation)")
    axs[0].set_title("Information stays, readout breaks", fontsize=9); axs[0].legend(fontsize=6.3)
    g = lc.groupby(["model", "n_labels"]).auroc.mean().reset_index()
    for m, c in [("E0_sall", GREY_DARK), ("E1_s11", BLUE), ("E1_s29", ORANGE)]:
        d = g[g.model == m]; axs[1].plot(d.n_labels, d.auroc, color=c, marker="o", ms=3.5, lw=1.4, label={"E0_sall": "pre-SFT (E0)", "E1_s11": "after SFT, seed 11", "E1_s29": "after SFT, seed 29"}[m])
    axs[1].set_xscale("log"); axs[1].set_xlabel("Labels used to fit the probe"); axs[1].set_title("Re-learning the readout needs labels", fontsize=9); axs[1].legend(fontsize=6.5)
    fig.suptitle("Readout vs representation (notebook 11)", fontsize=9.5, y=1.04)
    save(fig, out, "fig08_readout.png")


def fig_isrel(root, out):
    g2 = pd.read_csv(os.path.join(root, "data/runs/10/rq_s1_g2_auroc.csv"))
    ret = pd.read_csv(os.path.join(root, "data/runs/10/rq_s3_retention.csv"))
    cr = pd.read_csv(os.path.join(root, "data/runs/12/results_critic.csv")); cr["seed"] = cr.seed.astype(str)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    lab = {"critic": "critic token", "b_lexical_lr": "lexical LR", "b_own_article": "own article", "b_subject_in": "subject mention",
           "b_hidden_probe": "hidden probe", "b_bm25_score": "BM25 score", "b_bm25_rank": "BM25 rank"}
    g2 = g2.sort_values("auroc")
    y = np.arange(len(g2))
    axs[0].barh(y, g2.auroc - 0.5, left=0.5, color=[BLUE if s == "critic" else GREY_LIGHT for s in g2.scorer], height=0.6)
    axs[0].errorbar(g2.auroc, y, xerr=[g2.auroc - g2.lo, g2.hi - g2.auroc], fmt="none", ecolor=INK2, lw=0.8)
    axs[0].set_yticks(y); axs[0].set_yticklabels([lab[s] for s in g2.scorer], fontsize=7.5); axs[0].set_xlim(0.5, 0.95)
    axs[0].set_xlabel("IsRel AUROC on G2 (597 windows)"); axs[0].set_title("RQ-S1: critic vs answer-free cues", fontsize=9)
    rows = [("critic\npre-SFT", cr[cr.model == "critic"].isrel_g2.item(), GREY_DARK)]
    for s in ["11", "29"]:
        rows.append((f"SFT\ns{s}", cr[(cr.model == "critic_SFT") & (cr.seed == s)].isrel_g2.item(), GREY_LIGHT))
        rows.append((f"SFT+A2\ns{s}", cr[(cr.model == "A2c") & (cr.seed == s)].isrel_g2.item(), BLUE))
    axs[1].bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], width=0.6)
    axs[1].set_xticks(range(len(rows))); axs[1].set_xticklabels([r[0] for r in rows], fontsize=6.5); axs[1].set_ylim(0.5, 0.92)
    axs[1].axhline(g2[g2.scorer == "b_lexical_lr"].auroc.item(), color=ORANGE, ls="--", lw=1); axs[1].text(3, 0.858, "lexical LR baseline", fontsize=6.5, color=INK2, ha="center")
    axs[1].set_ylabel("IsRel AUROC (G2)"); axs[1].set_title("RQ-S3 + anchoring: IsRel after SFT", fontsize=9)
    save(fig, out, "fig09_isrel.png")


def fig_tradeoff(root, out):
    r = pd.read_csv(os.path.join(root, "data/runs/12/results_routing.csv")); r["seed"] = r.seed.astype(str)
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    e0 = r[r.method == "E0"].auroc_eval.item()
    ax.axhline(e0, color=GREY_DARK, ls="--", lw=1); ax.text(0.605, e0 + 0.003, "E0 pre-SFT (0% gain)", fontsize=6.8, color=INK2)
    style = {"E1": ("o", GREY_LIGHT, (6, -3), "E1 plain SFT"), "interp0.5": ("o", AQUA, (6, 2), "interp. α=0.5"),
             "taskvec1": ("D", AQUA, (-58, -12), "task vector"), "B": ("^", BLUE, (6, -10), "B repair"),
             "A1": ("s", BLUE, (6, -4), "A1"), "A2": ("o", BLUE, (6, 2), "A2")}
    for m, (mk, c, off, lab) in style.items():
        d = r[r.method == m]
        ax.scatter(d.sft_gain_kept, d.auroc_eval, color=c, marker=mk, s=32, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(lab, (d.sft_gain_kept.max(), d.auroc_eval.mean()), textcoords="offset points", xytext=off, fontsize=7, color=INK)
    ax.set_xlim(0.6, 1.08); ax.set_xlabel("Share of SFT instruction-loss gain kept"); ax.set_ylabel("Routing AUROC (1,454 val questions)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.set_title("Retention vs SFT gain (notebook 12)", fontsize=9)
    save(fig, out, "fig10_tradeoff.png")


def fig_answering(root, out):
    t = load_test(root); t["seed"] = t.seed.astype(str)
    order = ["E0", "E1", "E2", "E4", "E7", "A1", "A2", "taskvec1", "interp0.5"]
    rows = []
    for m in order:
        d = t[(t.model == m) & t.closed_em.notna()]
        rows.append((m, d.closed_em.mean(), d.evidence_em.mean()))
    fig, ax = plt.subplots(figsize=(7.2, 2.4)); x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, [r[1] for r in rows], width=w - 0.04, color=BLUE, label="closed-book EM")
    ax.bar(x + w / 2, [r[2] for r in rows], width=w - 0.04, color=AQUA, label="evidence EM (top-3 windows)")
    ax.set_xticks(x); ax.set_xticklabels([NAMES[r[0]] for r in rows], fontsize=6.6, rotation=15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.legend(fontsize=7)
    ax.set_title("Answer paths on TEST (mean of two seeds): only interpolation keeps answering", fontsize=9)
    save(fig, out, "fig11_answering_test.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(out, exist_ok=True)
    fig_pipeline(out); fig_sft_trajectory(a.root, out); fig_flips(a.root, out); fig_replay_val(out)
    fig_test_auroc(a.root, out); fig_forest(a.root, out); fig_interpolation(a.root, out); fig_readout(a.root, out)
    fig_isrel(a.root, out); fig_tradeoff(a.root, out); fig_answering(a.root, out)

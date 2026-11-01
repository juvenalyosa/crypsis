#!/usr/bin/env python3
"""Figures of the manuscript, from the result files written by the other scripts in this directory.

    python scripts/paper_figures.py data   # with the benchmark environment -> paper/figures/data.json
    python scripts/paper_figures.py plot   # with any environment that has matplotlib -> paper/figures/fig*.pdf

Inputs (all produced by this repository's scripts):
  ~/crypsis_data/work/control_paired.json, control_columns.json        controls.py paired / columns
  ~/crypsis_data/work/heldout_cryptobench.json                          heldout_eval.py --set cryptobench
  ~/crypsis_data/work_temporal/heldout_temporal.json, interaction.json  heldout_eval.py --set temporal, interaction.py
  ~/crypsis_data/work_seeds/seed_summary.json                           seed_ensembles.py summary
  ~/crypsis_data/work*/perturbation_controls.json                       perturbation_controls.py summary
  ~/crypsis_data/work/site_types.json                                   site_types.py
  ~/crypsis_data/cases/v0.2.1/4LDJ_hybrid.json and cases/pdb            literature_cases.py
Per-protein AUCs are recomputed from the saved scores (heldout_eval.all_scores).  Main figures fig1-fig5;
supplementary figures figS1 (single descriptors) and figS2 (ensemble seeds).
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

if sys.argv[1:] == ["plot"]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

D = os.path.expanduser("~/crypsis_data")
MM = 1 / 25.4
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL = {"hybrid_650M": "#2a78d6", "hybrid_150M": "#2a78d6", "hybrid_static": "#86b3ea", "physics": "#eb6834",
       "physics_static": "#f2a27f", "esm_only_650M": "#1baf7a", "esm_only_150M": "#1baf7a",
       "esm_pooled_650M": "#5cc99e", "esm_mlp_650M": "#0d7a53", "fpocket": "#8a8985", "p2rank": "#5f5d59",
       "pocketminer": "#b4b2ac"}
NAME = {"hybrid_650M": "Hybrid", "hybrid_150M": "Hybrid (ESM-2 150M)", "hybrid_static": "Hybrid, no ensembles",
        "physics": "Physics", "physics_static": "Physics, no ensembles", "esm_only_650M": "ESM-2 only (PCs)",
        "esm_only_150M": "ESM-2 150M only", "esm_pooled_650M": "ESM-2 only (PCs + neighbours)",
        "esm_mlp_650M": "ESM-2 only (MLP, full)", "fpocket": "fpocket", "p2rank": "P2Rank 2.5.1",
        "pocketminer": "PocketMiner"}
SETS = {"cryptobench": "CryptoBench test (222)", "temporal": "Temporal set (193)"}

RC = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
      "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 7.5, "axes.labelweight": "bold",
      "axes.titleweight": "bold", "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
      "axes.edgecolor": INK, "axes.linewidth": 0.8, "xtick.color": INK, "ytick.color": INK,
      "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
      "xtick.major.size": 3, "ytick.major.size": 3, "xtick.direction": "out", "ytick.direction": "out",
      "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
HELVETICA = "/System/Library/Fonts/Helvetica.ttc"   # macOS; elsewhere the figures fall back to Arial


def use_helvetica():
    """Register Helvetica Regular and Bold with matplotlib.  macOS ships them in one collection (.ttc), and
    matplotlib reads only its first face, so the two faces are written out as single fonts in a cache folder."""
    if not os.path.exists(HELVETICA):
        return
    from fontTools.ttLib import TTCollection
    from matplotlib import font_manager

    cache = os.path.join(os.path.expanduser("~"), ".cache", "crypsis-fonts")
    os.makedirs(cache, exist_ok=True)
    for index, name in ((0, "Helvetica.ttf"), (1, "Helvetica-Bold.ttf")):
        path = os.path.join(cache, name)
        if not os.path.exists(path):
            TTCollection(HELVETICA).fonts[index].save(path)
        font_manager.fontManager.addfont(path)
DATA = os.path.join(OUT, "data.json")


def load(path):
    with open(os.path.join(D, path)) as fh:
        return json.load(fh)


def panel(ax, letter):
    y = 1.02 if not ax.get_title() else 1.02 + 0.06 * (ax.get_title().count("\n") + 2)   # above a title
    ax.text(-0.02, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def vgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    for ax in fig.axes:                                        # Helvetica Bold for the axis labels and ticks
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("->", name)


# ------------------------------------------------------------------------------------------------ figure 1

def kras_data():
    from crypsis import core

    import literature_cases as lc

    pdb = os.path.join(D, "cases", "pdb")
    true = lc.true_site(pdb, "4LDJ", ["A"], "6OIM", "A", "MOV")
    with open(os.path.join(D, "cases", "v0.2.1", "4LDJ_hybrid.json")) as fh:
        res = json.load(fh)
    ch = core.read_chain(os.path.join(pdb, "4LDJ.cif.gz"), chains=["A"], keep_hetero={"GDP", "MG"})
    feats, _extra = core.features(ch, dict(core.load_model().get("params") or {}))
    assert ch.res_keys == res["residues"]
    return {"num": [int(k.split("_")[1]) for k in ch.res_keys], "true": [k in true for k in ch.res_keys],
            "static": feats[:, core.FEATURES.index("score")].tolist(),
            "ens": feats[:, core.FEATURES.index("ens_score_mean")].tolist(), "prob": res["probability"],
            "site1": [k in set(res["sites"][0]) for k in ch.res_keys]}


def fig1(kr):
    fig = plt.figure(figsize=(180 * MM, 70 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1], wspace=0.22, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_axis_off()
    steps = ["Apo structure (one chain or complex)",
             "Elastic network: 30 slowest ANM modes,\nsign fixed",
             "Ensembles: 24 random combinations of the\nmodes at 3 Å and 5 Å, and with side chains\ncut back to Cβ",
             "Pocket grid in every conformer:\nburied, probe-accessible points",
             "49 per-residue descriptors: 32 of the apo\nstructure, 17 of the ensembles\n(+ PCA of frozen ESM-2 embeddings)",
             "Gradient-boosted trees: residue\nprobability; sites with pocket volumes"]
    ys = np.linspace(0.97, 0.03, len(steps))
    boxes = []
    for k, (y, s) in enumerate(zip(ys, steps)):
        boxes.append(ax.text(0.5, y, s, ha="center", va="center", fontsize=6.3, transform=ax.transAxes,
                             bbox=dict(boxstyle="round,pad=0.35", fc="#f3f2ef", ec=INK2, lw=0.5)))
        if k:
            ax.annotate("", xy=(0.5, 1.0), xycoords=boxes[k], xytext=(0.5, 0.0), textcoords=boxes[k - 1],
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=6, shrinkA=4,
                                        shrinkB=4))
    panel(ax, "a")

    num, is_true, s1 = np.array(kr["num"]), np.array(kr["true"]), np.array(kr["site1"])
    static, ens, prob = np.array(kr["static"]), np.array(kr["ens"]), np.array(kr["prob"])

    ax1 = fig.add_subplot(gs[0, 1])
    top = max(static.max(), ens.max()) * 1.1
    ax1.vlines(num[is_true], 0, top, color="#e9e8e4", lw=1.6, zorder=0)
    ax1.plot(num, static, color=INK2, lw=0.9, label="Pocket score, apo structure")
    ax1.plot(num, ens, color=COL["physics"], lw=1.0, label="Mean pocket score, ensemble")
    ax1.set_ylim(0, top)
    ax1.set_ylabel("Pocket score")
    ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.03, 1.0), ncol=2, handlelength=1.2,
               borderaxespad=0.1)
    ax1.set_xlim(num.min() - 1, num.max() + 1)
    ax1.tick_params(labelbottom=False)
    panel(ax1, "b")

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.vlines(num[is_true], 0, prob.max() * 1.15, color="#e9e8e4", lw=1.6, zorder=0)
    ax2.plot(num, prob, color=COL["hybrid_650M"], lw=1.0)
    ax2.plot(num[s1], prob[s1], "o", ms=3, mfc=COL["hybrid_650M"], mec="white", mew=0.5, label="First predicted site")
    ax2.set_ylim(0, prob.max() * 1.15)
    ax2.set_ylabel("Probability (hybrid)")
    ax2.set_xlabel("K-Ras residue (4LDJ); grey bands: sotorasib site (6OIM)")
    ax2.legend(frameon=False, loc="upper right", handlelength=1.0)
    panel(ax2, "c")
    save(fig, "fig1_method_kras.pdf")


# ---------------------------------------------------------------------------------- supplementary figure 1

def figS1():
    cols = load("work/control_columns.json")
    paired = load("work/control_paired.json")["columns"]
    pairs = [("ens_score_mean", "score", "Pocket score"), ("ens_volume_mean", "volume", "Pocket volume"),
             ("ens_open_fraction", "score", "Opening frequency (3 Å)\nvs static score"),
             ("amp50_score", "score", "Pocket score, 5 Å ensemble\nvs static score")]
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 50 * MM), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    y = np.arange(len(pairs))[::-1]
    for ax, met, lab in ((axes[0], "auc", "ROC AUC (no learning)"), (axes[1], "auprc", "AUPRC (no learning)")):
        for yi, (new, old, _t) in zip(y, pairs):
            a, b = cols["test"][old][met], cols["test"][new][met]
            ax.plot([a, b], [yi, yi], color=GRID, lw=1.2, zorder=1)
            ax.plot(a, yi, "o", ms=4.5, mfc="white", mec=INK2, mew=0.9, zorder=2)
            ax.plot(b, yi, "o", ms=4.5, mfc=COL["physics"], mec="white", mew=0.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([t for _n, _o, t in pairs] if ax is axes[0] else [])
        ax.set_xlabel(lab)
        vgrid(ax)
    axes[1].axvline(cols["positive_rate"]["test"], color=INK2, lw=0.5, ls=":")
    axes[1].text(cols["positive_rate"]["test"], y[0] + 0.45, " prevalence", fontsize=6, color=INK2, va="center")
    axes[0].plot([], [], "o", ms=4.5, mfc="white", mec=INK2, label="Apo structure")
    axes[0].plot([], [], "o", ms=4.5, mfc=COL["physics"], mec="white", label="Ensemble")
    axes[0].legend(frameon=False, loc="lower right", handletextpad=0.3)
    ax = axes[2]
    for k, met in enumerate(("auc", "auprc")):
        for yi, (new, old, _t) in zip(y, pairs):
            lo, hi = paired[f"{new} - {old}"][f"{met}_diff_ci"]
            yy = yi + (0.14 if k == 0 else -0.14)
            ax.plot([lo, hi], [yy, yy], color=INK if k == 0 else INK2, lw=1.4, solid_capstyle="round")
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Ensemble − apo structure, 95% CI")
    ax.plot([], [], color=INK, lw=1.4, label="ΔAUC")
    ax.plot([], [], color=INK2, lw=1.4, label="ΔAUPRC")
    ax.legend(frameon=False, loc="upper left")
    vgrid(ax)
    for a, l in zip(axes, "abc"):
        panel(a, l)
    save(fig, "figS1_single_descriptors.pdf")


# ------------------------------------------------------------------------------------------------ figure 2

MODELS = ["hybrid_650M", "hybrid_static", "physics", "physics_static", "esm_pooled_650M", "esm_only_650M",
          "esm_mlp_650M", "pocketminer", "p2rank", "fpocket"]


def present(h):
    """The models of MODELS scored on both sets (P2Rank and PocketMiner were added in the second round of review)."""
    return [m for m in MODELS if all(m in h[s]["all"]["models"] for s in ("cryptobench", "temporal"))]


def fig2(h):
    models = present(h)
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, (62 + 5 * (len(models) - 8)) * MM),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    y = np.arange(len(models))[::-1]
    for ax, met, lab in ((axes[0], "auc", "ROC AUC"), (axes[1], "auprc", "AUPRC ÷ prevalence")):
        for k, (s, mark) in enumerate((("cryptobench", "o"), ("temporal", "s"))):
            for yi, m in zip(y, models):
                r = h[s]["all"]["models"][m]
                norm = 1.0 if met == "auc" else r["positive_rate"]
                lo, hi = (np.array(r[f"{met}_ci"]) / norm)
                v = r[met] / norm
                yy = yi + (0.17 if k == 0 else -0.17)
                ax.plot([lo, hi], [yy, yy], color=COL[m], lw=1.0, alpha=0.9)
                ax.plot(v, yy, mark, ms=4, mfc=COL[m] if k == 0 else "white", mec=COL[m], mew=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([NAME[m] for m in models] if ax is axes[0] else [])
        ax.set_xlabel(lab)
        vgrid(ax)
    axes[0].plot([], [], "o", ms=4, mfc=INK2, mec=INK2, label=SETS["cryptobench"])
    axes[0].plot([], [], "s", ms=4, mfc="white", mec=INK2, label=SETS["temporal"])
    axes[0].legend(frameon=False, loc="upper left")
    # c: AUC change from CryptoBench to the temporal set
    ax = axes[2]
    for yi, m in zip(y, models):
        a, b = h["cryptobench"]["all"]["models"][m]["auc"], h["temporal"]["all"]["models"][m]["auc"]
        ax.barh(yi, b - a, height=0.55, color=COL[m], edgecolor="white", linewidth=0.5)
        ax.text(min(b - a, 0) - 0.002, yi, f"{b - a:+.3f}", ha="right", va="center", fontsize=6, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-0.098, 0.005)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("AUC, temporal − CryptoBench")
    vgrid(ax)
    for a, l in zip(axes, "abc"):
        panel(a, l)
    save(fig, "fig2_models_two_sets.pdf")


# ------------------------------------------------------------------------------------------------ figure 3

DIFFS = [("hybrid_650M - physics", "Hybrid − physics"), ("hybrid_650M - esm_only_650M", "Hybrid − ESM-2 only (PCs)"),
         ("physics - esm_only_650M", "Physics − ESM-2 only (PCs)"),
         ("physics - esm_pooled_650M", "Physics − ESM-2 only (PCs + neighbours)"),
         ("physics - esm_mlp_650M", "Physics − ESM-2 only (MLP)"),
         ("physics_static - esm_only_650M", "Physics, no ensembles − ESM-2 only (PCs)"),
         ("physics - physics_static", "Physics − physics, no ensembles"),
         ("hybrid_650M - hybrid_static", "Hybrid − hybrid, no ensembles"),
         ("hybrid_650M - hybrid_150M", "Hybrid 650M − hybrid 150M")]


def fig3(h):
    fig, axes = plt.subplots(1, 2, figsize=(165 * MM, 70 * MM), sharey=True)
    y = np.arange(len(DIFFS))[::-1]
    for ax, met, lab in ((axes[0], "auc", "ΔROC AUC, 95% CI"), (axes[1], "auprc", "ΔAUPRC, 95% CI")):
        for k, (s, colr) in enumerate((("cryptobench", INK), ("temporal", "#2a78d6"))):
            for yi, (key, _t) in zip(y, DIFFS):
                r = h[s]["all"]["differences"][key]
                lo, hi = r[f"{met}_diff_ci"]
                yy = yi + (0.15 if k == 0 else -0.15)
                ax.plot([lo, hi], [yy, yy], color=colr, lw=1.6, solid_capstyle="round")
                p = r[f"{met}_diff_p_holm"]                  # paired bootstrap of the pooled difference
                w = r["per_protein"][f"{met}_wilcoxon_p_holm"]   # per-protein Wilcoxon
                mark = ("*" if p >= 1e-3 else "**") if p < 0.05 else ""
                mark += "†" if w < 0.05 else ""
                if mark:
                    ax.text(hi + 0.003, yy, mark, color=colr, fontsize=6.5, va="center")
        ax.axvline(0, color=INK2, lw=0.6)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.012)                    # room for the markers right of the intervals
        ax.set_xlabel(lab)
        vgrid(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([t for _k, t in DIFFS])
    axes[1].plot([], [], color=INK, lw=1.6, label=SETS["cryptobench"])
    axes[1].plot([], [], color="#2a78d6", lw=1.6, label=SETS["temporal"])
    axes[1].legend(frameon=False, loc="lower right")
    for a, l in zip(axes, "ab"):
        panel(a, l)
    save(fig, "fig3_paired_differences.pdf")


# ------------------------------------------------------------------------------------------------ figure 4

CONTROLS = [("static", "Apo structure only"), ("jitter", "Apo, rigid moves (grid jitter)"),
            ("dilated", "Apo, dilated pockets"), ("random", "Uncorrelated noise, 3 Å"),
            ("smooth_3", "Smooth random field, 3 Å"), ("fast", "ANM modes 31–60, 3 Å"),
            ("slow", "ANM modes 1–30, 3 Å (shipped)")]


def fig4():
    """Second round of review: per-protein evaluation (scripts/perturbation_controls.py per-protein), with the
    pooled raw AUC of the first version shown for comparison."""
    pp = {"cryptobench": load("work/perturbation_per_protein.json"),
          "temporal": load("work_temporal/perturbation_per_protein.json")}
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 56 * MM), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    y = np.arange(len(CONTROLS))[::-1]
    hue = {c: INK2 for c, _t in CONTROLS}
    hue.update(fast=COL["physics"], slow=COL["physics"], smooth_3="#7048e8")
    for k, (s, mark) in enumerate((("cryptobench", "o"), ("temporal", "s"))):
        for yi, (c, _t) in zip(y, CONTROLS):
            r = pp[s]["controls"][c]
            yy = yi + (0.17 if k == 0 else -0.17)
            lo, hi = r["mean_auc_per_protein_ci"]
            axes[0].plot([lo, hi], [yy, yy], color=hue[c], lw=1.0)
            axes[0].plot(r["mean_auc_per_protein"], yy, mark, ms=4, mfc=hue[c] if k == 0 else "white", mec=hue[c],
                         mew=0.9)
            axes[1].plot(r["pooled_auc"], yy, mark, ms=4, mfc="none", mec=hue[c], mew=0.6, alpha=0.6)
            axes[1].plot(r["pooled_auc_rank_normalized"], yy, mark, ms=4, mfc=hue[c] if k == 0 else "white",
                         mec=hue[c], mew=0.9)
            axes[1].plot([r["pooled_auc"], r["pooled_auc_rank_normalized"]], [yy, yy], color=hue[c], lw=0.5,
                         alpha=0.6)
            if c != "slow":
                d = pp[s]["slow_vs"][c]
                lo, hi = d["mean_diff_ci"]
                colr = INK if k == 0 else "#2a78d6"
                axes[2].plot([lo, hi], [yy, yy], color=colr, lw=1.6, solid_capstyle="round")
                p = d["wilcoxon_p_holm"]
                if p < 0.05:
                    axes[2].text(hi + 0.002, yy, "*" if p >= 1e-3 else "**", color=colr, fontsize=6.5, va="center")
    for ax, lab in ((axes[0], "Mean per-protein ROC AUC"), (axes[1], "Pooled ROC AUC")):
        ax.set_yticks(y)
        ax.set_yticklabels([t for _c, t in CONTROLS] if ax is axes[0] else [])
        ax.set_xlabel(lab)
        vgrid(ax)
    axes[0].plot([], [], "o", ms=4, mfc=INK2, mec=INK2, label=SETS["cryptobench"])
    axes[0].plot([], [], "s", ms=4, mfc="white", mec=INK2, label=SETS["temporal"])
    axes[0].set_xlim(0.56, None)
    axes[0].legend(frameon=False, loc="lower left", borderaxespad=0.2, fontsize=6)
    axes[1].text(0.02, 0.01, "open: raw scores\nfilled: ranked within protein", transform=axes[1].transAxes,
                 fontsize=6, color=INK2, va="bottom")
    axes[1].set_xlim(0.56, None)
    ax = axes[2]
    ax.axvline(0, color=INK2, lw=0.6)
    ax.text(0.002, y[-1], "reference", va="center", ha="left", fontsize=6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(axes[0].get_ylim())
    ax.set_xlabel("ΔAUC per protein, modes 1–30 − row")
    ax.plot([], [], color=INK, lw=1.6, label="CryptoBench")
    ax.plot([], [], color="#2a78d6", lw=1.6, label="Temporal")
    ax.legend(frameon=False, loc="lower right")
    vgrid(ax)
    for a, l in zip(axes, "abc"):
        panel(a, l)
    save(fig, "fig4_perturbation_controls.pdf")


def figS7():
    """Dose response of the raw ensemble score (mean per-protein AUC) to the amplitude of each perturbation."""
    pp = {"cryptobench": load("work/perturbation_per_protein.json"),
          "temporal": load("work_temporal/perturbation_per_protein.json")}
    series = [("slow", "ANM modes 1–30", COL["physics"], {1.0: "slow_1", 1.5: "slow_1.5", 3.0: "slow", 5.0: "slow_5"}),
              ("fast", "ANM modes 31–60", "#f2a27f", {1.0: "fast_1", 3.0: "fast", 5.0: "fast_5"}),
              ("smooth", "Smooth random field", "#7048e8", {1.0: "smooth_1", 3.0: "smooth_3", 5.0: "smooth_5"}),
              ("random", "Uncorrelated noise", INK2, {1.0: "random_1", 3.0: "random"})]
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 55 * MM), sharey=True)
    for ax, s in zip(axes, ("cryptobench", "temporal")):
        c = pp[s]["controls"]
        for _k, lab, colr, pts in series:
            xs = sorted(pts)
            ax.plot(xs, [c[pts[x]]["mean_auc_per_protein"] for x in xs], "o-", color=colr, ms=3.5, lw=1.0, label=lab)
        for key, ls, lab in (("static", ":", "apo structure"), ("jitter", "--", "rigid moves")):
            ax.axhline(c[key]["mean_auc_per_protein"], color=INK2, lw=0.8, ls=ls, label=lab)
        ax.set_xlabel("Cα RMSD of the copies (Å)")
        ax.set_xticks([1, 1.5, 3, 5])
        ax.set_title(SETS[s], color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean per-protein AUC (raw score)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "figS7_dose_response.pdf")


def figS6():
    """True-site volume per copy (K-Ras and the Fig. 6 proteins) and the overlap of the modes with apo -> holo."""
    es = load("work_temporal/ensemble_sites.json")
    mo = {s: load(f"{w}/mode_overlap.json")["entities"] for s, w in (("cryptobench", "work"), ("temporal", "work_temporal"))}
    fig = plt.figure(figsize=(180 * MM, 110 * MM))
    kinds = [("modes_1_30", "Modes 1–30", COL["physics"]), ("modes_31_60", "Modes 31–60", "#f2a27f"),
             ("rigid", "Rigid moves", INK2), ("smooth", "Smooth field", "#7048e8")]
    for i, case in enumerate(es):
        ax = fig.add_axes([0.07 + i * 0.235, 0.58, 0.19, 0.34])
        for j, (k, lab, colr) in enumerate(kinds):
            v = np.array(case[k])
            xx = j + np.random.default_rng(j).uniform(-0.18, 0.18, len(v))
            ax.scatter(xx, v, s=6, color=colr, alpha=0.8, edgecolors="none")
            ax.plot([j - 0.28, j + 0.28], [v.mean()] * 2, color=INK, lw=1.0)
        ax.axhline(case["apo"], color=INK2, lw=0.8, ls=":")
        ax.axhline(case["holo"], color="#0ca678", lw=1.0, ls="--")
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([lab for _k, lab, _c in kinds], rotation=45, ha="right", fontsize=6)
        ax.set_title(case["name"].replace(" (", "\n("), fontsize=6.5, color=INK)
        if i == 0:
            ax.set_ylabel("True-site pocket volume (Å³)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        if i == 0:
            panel(ax, "a")
    ax = fig.add_axes([0.07, 0.1, 0.27, 0.3])
    for s, mark, colr in (("cryptobench", "o", INK), ("temporal", "s", "#2a78d6")):
        rows = [r for r in mo[s].values() if "overlap_1_30" in r]
        ax.scatter([r["overlap_31_60"] for r in rows], [r["overlap_1_30"] for r in rows], s=6, marker=mark,
                   facecolors=colr if s == "cryptobench" else "none", edgecolors=colr, linewidths=0.5, alpha=0.7,
                   label=SETS[s])
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Overlap with apo-to-holo change, modes 31–60")
    ax.set_ylabel("Overlap, modes 1–30")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    panel(ax, "b")
    ax = fig.add_axes([0.44, 0.1, 0.24, 0.3])
    for s, colr in (("cryptobench", INK), ("temporal", "#2a78d6")):
        v = [r["conf_cos_median"] for r in mo[s].values() if "conf_cos_median" in r]
        ax.hist(v, bins=np.linspace(0, 0.6, 25), histtype="step", color=colr, lw=1.0, label=SETS[s])
    ax.set_xlabel("Median |cos|, conformer vs apo-to-holo change")
    ax.set_ylabel("Proteins")
    panel(ax, "c")
    ax = fig.add_axes([0.76, 0.1, 0.22, 0.3])
    for s, colr in (("cryptobench", INK), ("temporal", "#2a78d6")):
        v = [r["rmsd_apo_holo"] for r in mo[s].values() if "rmsd_apo_holo" in r]
        ax.hist(v, bins=np.linspace(0, 8, 33), histtype="step", color=colr, lw=1.0)
    ax.axvline(3.0, color=COL["physics"], lw=1.0, ls="--")
    ax.text(3.1, ax.get_ylim()[1] * 0.9, "ensemble RMSD", fontsize=6, color=COL["physics"])
    ax.set_xlabel("Apo–holo Cα RMSD (Å)")
    panel(ax, "d")
    save(fig, "figS6_ensembles.pdf")


# ------------------------------------------------------------------------------------------------ figure 5

def per_protein_auc():
    import cryptobench as cb
    import heldout_eval as he

    out = {}
    for s in ("cryptobench", "temporal"):
        cb._DATA.clear()
        a = SimpleNamespace(data=he.SETS[s][0], work=he.SETS[s][1], physics="v3")
        recs = cb.load_split(a, "test")
        sc = he.all_scores(recs, a)
        ys = [r["y"] for r in recs]
        out[s] = {m: [None if np.isnan(v) else float(v) for v in he.per_protein(sc[m], ys)[0]] for m in MODELS
                  if m in sc}
        # site state from the ligand fit (scripts/holo_geometry.py; second round of review)
        geo = load(os.path.join(os.path.basename(a.work), "holo_geometry.json"))["entities"]
        out[s]["site_state"] = [geo[r["id"]]["state"] for r in recs]
        out[s]["ids"] = [r["id"] for r in recs]
    return out


def fig5(pp):
    pp = {s: {m: (np.array([np.nan if v is None else v for v in x]) if m not in ("site_state", "ids") else x)
              for m, x in d.items()} for s, d in pp.items()}
    st = load("work/site_state.json")
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), gridspec_kw={"width_ratios": [1, 1, 1.5], "wspace": 0.38})
    for ax, s, l in ((axes[0], "cryptobench", "a"), (axes[1], "temporal", "b")):
        x, yv = pp[s]["esm_only_650M"], pp[s]["physics"]
        ok = ~np.isnan(x)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
        ax.scatter(x[ok], yv[ok], s=7, color=COL["physics"], edgecolors="white", linewidths=0.3, alpha=0.9)
        wins = int((yv[ok] > x[ok]).sum())
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("AUC per protein, ESM-2 only")
        ax.set_ylabel("AUC per protein, physics" if ax is axes[0] else "")
        ax.set_title(f"{SETS[s]}\nphysics higher in {wins} of {int(ok.sum())}", color=INK)
        panel(ax, l)
    # c: mean per-protein AUC by the share of the holo ligand that fits in the apo structure, both sets; 95% CI
    # over 2,000 protein resamples; stars: physics - ESM-2 within the stratum, Holm over nine tests (site_state.py)
    ax = axes[2]
    groups = [("closed", "Closed"), ("partly_open", "Partly\nopen"), ("open", "Open")]
    order = ["hybrid_650M", "physics", "esm_only_650M"]
    xt, xl = [], []
    for si, s in enumerate(("cryptobench", "temporal")):
        state = np.array(pp[s]["site_state"])
        for g, (key, lab) in enumerate(groups):
            base = si * 3.6 + g
            for j, m in enumerate(order):
                v = pp[s][m][state == key]
                v = v[~np.isnan(v)]
                boot = v[np.random.default_rng(0).integers(0, len(v), (2000, len(v)))].mean(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                xx = base + (j - 1) * 0.22
                ax.plot([xx, xx], [lo, hi], color=COL[m], lw=1.0)
                ax.plot(xx, v.mean(), "o" if s == "cryptobench" else "s", ms=3.8,
                        mfc=COL[m] if s == "cryptobench" else "white", mec=COL[m], mew=0.8,
                        label=NAME[m] if (g == 0 and si == 0) else None)
            t = st[s]["definitions"]["ligand"]["strata"][key]["tests"]["physics - esm_only_650M"]["wilcoxon_p_holm"]
            if t < 0.05:
                ax.text(base, 1.0, "*" if t >= 1e-3 else "**", ha="center", fontsize=7)
            xt.append(base)
            xl.append(f"{lab}\n{int((state == key).sum())}")
        ax.text(si * 3.6 + 1, 0.47, SETS[s].split(" (")[0], ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("Mean per-protein AUC")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2, fontsize=6)
    ax.set_title("By the share of the holo ligand that fits\nin the apo structure", color=INK)
    panel(ax, "c")
    save(fig, "fig5_per_protein.pdf")


# ---------------------------------------------------------------------------------- supplementary figure 2

def figS2(h):
    seeds = load("work_seeds/seed_summary.json")
    keys = [("auc", "ROC AUC"), ("auprc", "AUPRC"), ("site_top1", "First site"), ("site_top3", "Top-3 site")]
    models = [("hybrid 650M", "hybrid_650M"), ("hybrid 150M", "hybrid_150M"), ("physics", "physics")]
    fig, ax = plt.subplots(figsize=(110 * MM, 50 * MM))
    x = np.arange(len(keys))
    for j, (sname, m) in enumerate(models):
        r = h["cryptobench"]["all"]["models"][m]
        rng = [100 * (seeds[sname][k]["max"] - seeds[sname][k]["min"]) for k, _l in keys]
        ax.bar(x + (j - 1) * 0.26, rng, width=0.24, color=COL[m] if m != "hybrid_150M" else "#9cc0ec",
               edgecolor="white", linewidth=0.5, label=NAME[m])
        ci = [100 * (r["auc_ci"][1] - r["auc_ci"][0]), 100 * (r["auprc_ci"][1] - r["auprc_ci"][0])]
        for i in range(2):
            ax.plot([x[i] + (j - 1) * 0.26 - 0.11, x[i] + (j - 1) * 0.26 + 0.11], [ci[i]] * 2, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _k, l in keys])
    ax.set_ylabel("Range over 5 seeds (points)")
    ax.plot([], [], color=INK, lw=0.8, label="Width of the 95% bootstrap CI")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "figS2_seed_variability.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:] == ["data"]:
        with open(DATA, "w") as fh:
            json.dump({"kras": kras_data(), "per_protein_auc": per_protein_auc()}, fh)
        print("->", DATA)
        return
    use_helvetica()
    plt.rcParams.update(RC)
    with open(DATA) as fh:
        data = json.load(fh)
    h = {"cryptobench": load("work/heldout_cryptobench.json"), "temporal": load("work_temporal/heldout_temporal.json")}
    fig1(data["kras"])
    fig2(h)
    fig3(h)
    fig4()
    fig5(data["per_protein_auc"])
    figS1()
    figS2(h)
    figS6()
    figS7()


if __name__ == "__main__":
    main()

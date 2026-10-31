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
       "esm_pooled_650M": "#5cc99e", "esm_mlp_650M": "#0d7a53", "fpocket": "#8a8985"}
NAME = {"hybrid_650M": "Hybrid", "hybrid_150M": "Hybrid (ESM-2 150M)", "hybrid_static": "Hybrid, no ensembles",
        "physics": "Physics", "physics_static": "Physics, no ensembles", "esm_only_650M": "ESM-2 only (PCs)",
        "esm_only_150M": "ESM-2 150M only", "esm_pooled_650M": "ESM-2 only (PCs + neighbours)",
        "esm_mlp_650M": "ESM-2 only (MLP, full)", "fpocket": "fpocket"}
SETS = {"cryptobench": "CryptoBench test (222)", "temporal": "Temporal set (193)"}

RC = ({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
DATA = os.path.join(OUT, "data.json")


def load(path):
    with open(os.path.join(D, path)) as fh:
        return json.load(fh)


def panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def vgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
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
             "Gradient-boosted trees → residue\nprobability; sites with pocket volumes"]
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
          "esm_mlp_650M", "fpocket"]


def fig2(h):
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), gridspec_kw={"width_ratios": [1, 1, 1]})
    y = np.arange(len(MODELS))[::-1]
    for ax, met, lab in ((axes[0], "auc", "ROC AUC"), (axes[1], "auprc", "AUPRC ÷ prevalence")):
        for k, (s, mark) in enumerate((("cryptobench", "o"), ("temporal", "s"))):
            for yi, m in zip(y, MODELS):
                r = h[s]["all"]["models"][m]
                norm = 1.0 if met == "auc" else r["positive_rate"]
                lo, hi = (np.array(r[f"{met}_ci"]) / norm)
                v = r[met] / norm
                yy = yi + (0.17 if k == 0 else -0.17)
                ax.plot([lo, hi], [yy, yy], color=COL[m], lw=1.0, alpha=0.9)
                ax.plot(v, yy, mark, ms=4, mfc=COL[m] if k == 0 else "white", mec=COL[m], mew=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([NAME[m] for m in MODELS] if ax is axes[0] else [])
        ax.set_xlabel(lab)
        vgrid(ax)
    axes[0].plot([], [], "o", ms=4, mfc=INK2, mec=INK2, label=SETS["cryptobench"])
    axes[0].plot([], [], "s", ms=4, mfc="white", mec=INK2, label=SETS["temporal"])
    axes[0].legend(frameon=False, loc="upper left")
    # c: AUC change from CryptoBench to the temporal set
    ax = axes[2]
    for yi, m in zip(y, MODELS):
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
                p = r["per_protein"][f"{met}_wilcoxon_p_holm"]
                if p < 0.05:
                    ax.text(hi + 0.003, yy, "*" if p >= 1e-3 else "**", color=colr, fontsize=6.5, va="center")
        ax.axvline(0, color=INK2, lw=0.6)
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
            ("dilated", "Apo, dilated pockets"), ("random", "Isotropic random displacement"),
            ("fast", "ANM modes 31–60"), ("slow", "ANM modes 1–30 (shipped)")]


def fig4():
    pc = {"cryptobench": load("work/perturbation_controls.json"),
          "temporal": load("work_temporal/perturbation_controls.json")}
    get = lambda s, c: pc[s]["static"] if c == "static" else pc[s]["controls"][c]
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 50 * MM), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    y = np.arange(len(CONTROLS))[::-1]
    hue = {"static": INK2, "jitter": INK2, "dilated": INK2, "random": INK2, "fast": COL["physics"],
           "slow": COL["physics"]}
    for ax, met, lab in ((axes[0], "auc", "ROC AUC (no learning)"), (axes[1], "auprc", "AUPRC ÷ prevalence")):
        for k, (s, mark) in enumerate((("cryptobench", "o"), ("temporal", "s"))):
            for yi, (c, _t) in zip(y, CONTROLS):
                r = get(s, c)
                norm = 1.0 if met == "auc" else r["positive_rate"]
                lo, hi = np.array(r[f"{met}_ci"]) / norm
                yy = yi + (0.17 if k == 0 else -0.17)
                ax.plot([lo, hi], [yy, yy], color=hue[c], lw=1.0)
                ax.plot(r[met] / norm, yy, mark, ms=4, mfc=hue[c] if k == 0 else "white", mec=hue[c], mew=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([t for _c, t in CONTROLS] if ax is axes[0] else [])
        ax.set_xlabel(lab)
        vgrid(ax)
    axes[0].plot([], [], "o", ms=4, mfc=INK2, mec=INK2, label=SETS["cryptobench"])
    axes[0].plot([], [], "s", ms=4, mfc="white", mec=INK2, label=SETS["temporal"])
    axes[0].legend(frameon=False, loc="lower left")
    ax = axes[2]
    for k, (s, colr) in enumerate((("cryptobench", INK), ("temporal", "#2a78d6"))):
        for yi, (c, _t) in zip(y, CONTROLS):
            if c == "slow":
                continue
            d = get(s, "slow")["this_minus_static"] if c == "static" else get(s, c)["slow_minus_this"]
            lo, hi = d["auc_diff_ci"]
            yy = yi + (0.15 if k == 0 else -0.15)
            ax.plot([lo, hi], [yy, yy], color=colr, lw=1.6, solid_capstyle="round")
    ax.axvline(0, color=INK2, lw=0.6)
    ax.text(0.003, y[-1], "reference", va="center", ha="left", fontsize=6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(axes[0].get_ylim())             # the reference row has no bar; keep the rows aligned with a and b
    ax.set_xlabel("ΔAUC, modes 1–30 − row, 95% CI")
    ax.plot([], [], color=INK, lw=1.6, label="CryptoBench")
    ax.plot([], [], color="#2a78d6", lw=1.6, label="Temporal")
    ax.legend(frameon=False, loc="lower right")
    vgrid(ax)
    for a, l in zip(axes, "abc"):
        panel(a, l)
    save(fig, "fig4_perturbation_controls.pdf")


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
        out[s] = {m: [None if np.isnan(v) else float(v) for v in he.per_protein(sc[m], ys)[0]] for m in MODELS}
        if s == "cryptobench":          # site state as in scripts/site_types.py
            from crypsis import core

            share = [float(((r["feats"][:, core.FEATURES.index("score")] > 0)[r["y"] == 1]).mean()) for r in recs]
            out[s]["site_state"] = ["closed" if x < 1 / 3 else ("partly_open" if x <= 2 / 3 else "open") for x in share]
    return out


def fig5(pp):
    pp = {s: {m: (np.array([np.nan if v is None else v for v in x]) if m != "site_state" else x) for m, x in d.items()}
          for s, d in pp.items()}
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 60 * MM), gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.42})
    for ax, s, l in ((axes[0], "cryptobench", "a"), (axes[1], "temporal", "b")):
        x, yv = pp[s]["esm_only_650M"], pp[s]["physics"]
        ok = ~np.isnan(x)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
        ax.scatter(x[ok], yv[ok], s=7, color=COL["physics"], edgecolors="white", linewidths=0.3, alpha=0.9)
        wins = int((yv[ok] > x[ok]).sum())
        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0.2, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("Per-protein AUC, ESM-2 650M only")
        ax.set_ylabel("Per-protein AUC, physics" if ax is axes[0] else "")
        ax.set_title(f"{SETS[s]}\nphysics higher in {wins} of {int(ok.sum())}", color=INK)
        panel(ax, l)
    # c: mean per-protein AUC by how much of the site lines a pocket of the apo structure (CryptoBench); 95% CI
    # over 2,000 protein resamples, as heldout_eval.per_protein_means
    ax = axes[2]
    state = np.array(pp["cryptobench"]["site_state"])
    groups = [("closed", "Closed\n(< 1/3)"), ("partly_open", "Partly open\n(1/3–2/3)"), ("open", "Open\n(> 2/3)")]
    order = ["hybrid_650M", "physics", "physics_static", "esm_only_650M"]
    for j, m in enumerate(order):
        for g, (key, _l) in enumerate(groups):
            v = pp["cryptobench"][m][state == key]
            v = v[~np.isnan(v)]
            boot = v[np.random.default_rng(0).integers(0, len(v), (2000, len(v)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            xx = g + (j - 1.5) * 0.17
            ax.plot([xx, xx], [lo, hi], color=COL[m], lw=1.0)
            ax.plot(xx, v.mean(), "o", ms=3.8, mfc=COL[m], mec="white", mew=0.5,
                    label=NAME[m] if g == 0 else None)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{l}\nn = {int((state == k).sum())}" for k, l in groups])
    ax.set_ylabel("Mean per-protein AUC")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2)
    ax.set_title("CryptoBench test: share of site residues\nlining a pocket of the apo structure", color=INK)
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


if __name__ == "__main__":
    main()

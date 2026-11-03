#!/usr/bin/env python3
"""Does Crypsis miss the secondary cryptic sites of a protein?

Prompted by the ABL1 myristoyl pocket, which Crypsis 0.3 did not call in any of 91 PDB chains although its own
pocket descriptors saw it.  For every apo structure, the pockets of its holo records (residues on the scored chains)
are merged into sites (single linkage, overlap coefficient >= 0.5).  The primary site is the one with most holo
records (ties: more residues); a secondary site shares less than 20% of its residues with the primary
(overlap coefficient).  Proteins with at least one secondary site are compared within protein:

  pct      mean within-protein percentile of the site's residue probabilities
  top3/5   the site is found by one of the first 3 / 5 predicted sites (cryptobench.site_hits rule applied to that
           site: >= 3 shared residues and >= half of the predicted site or >= a third of the true site)

Sets: CryptoBench out-of-fold predictions of the 4 training folds (models that did not see the protein),
CryptoBench test (shipped model), temporal set (shipped model); hybrid 650M, Crypsis 0.3.
Paired tests: Wilcoxon signed-rank on per-protein (primary - mean of secondaries) percentile; McNemar-style exact
binomial test on discordant top-5 pairs (primary found / secondary not vs the reverse, first secondary per protein).
Size control: the same comparison restricted to pairs whose sizes differ by at most 30%.  by_size_all_sites: every
site of every protein of the set (not only proteins with a secondary site), share found among the first 3 and 5
predicted sites in three size bands (3-8, 9-15, >= 16 residues).

    python secondary_sites.py [0.2.1|0.3]      # default 0.2.1, the version the paper evaluates
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from crypsis import core

D = Path.home() / "crypsis_data"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "0.2.1"     # 0.2.1: the paper's work dirs; 0.3: work*_v03
W, WT = ("work", "work_temporal") if VERSION == "0.2.1" else ("work_v03", "work_temporal_v03")
SETS = {
    "cryptobench_oof": (D / W / "oof_v3_hybrid_650M_256.npz", D / W / "features",
                        D / "cryptobench/cryptobench-dataset/dataset.json"),
    "cryptobench_test": (D / W / "test_shipped_v3_650M_scores.npz", D / W / "features",
                         D / "cryptobench/cryptobench-dataset/dataset.json"),
    "temporal": (D / WT / "test_shipped_v3_650M_scores.npz", D / WT / "features",
                 D / "temporal/dataset.json"),
}


def sites_of(holos):
    chains = set(holos[0]["apo_chain"].split("-"))
    pockets = []
    for h in holos:
        p = frozenset(k for k in h["apo_pocket_selection"] if k.split("_")[0] in chains)
        if p:
            pockets.append(p)
    groups = []                                           # [residue set, records]
    for p in pockets:
        hit = [g for g in groups if len(g[0] & p) >= 0.5 * min(len(g[0]), len(p))]
        merged = [set(p), 1]
        for g in hit:
            merged[0] |= g[0]
            merged[1] += g[1]
            groups.remove(g)
        groups.append(merged)
    groups.sort(key=lambda g: (-g[1], -len(g[0])))
    return groups


def found(site, predicted, k):
    for st in predicted[:k]:
        m = set(st["residues"])
        n = len(m & site)
        if n >= 3 and (n >= 0.5 * len(m) or n >= len(site) / 3.0):
            return True
    return False


def load_scores(npz_path):
    """{id: hybrid 650M probabilities}.  Our own score files: the OOF file stores one array per id; the test files
    store object arrays ('ids', 'hybrid'), written by scripts/cryptobench.py, so allow_pickle is safe here."""
    z = np.load(npz_path, allow_pickle=True)
    if "hybrid" in z.files:
        return {str(i): np.asarray(a, dtype=float) for i, a in zip(z["ids"], z["hybrid"])}
    return {str(i): np.asarray(z[str(i)], dtype=float) for i in z["ids"]}


def run(name, npz_path, feat_dir, ds_path):
    ds = json.loads(Path(ds_path).read_text())
    scores = load_scores(npz_path)
    rows = []
    for sid, prob in scores.items():
        f = Path(feat_dir) / f"{sid}.npz"
        if not f.exists():
            continue
        apo = sid[:4]
        if apo not in ds:
            continue
        fz = np.load(f)
        keys = [str(k) for k in fz["keys"]]
        if len(prob) != len(keys):
            continue
        groups = sites_of(ds[apo])
        if not groups:
            continue
        prim = groups[0][0]
        secs = [g[0] for g in groups[1:] if len(g[0] & prim) < 0.2 * min(len(g[0]), len(prim))]
        n = len(keys)
        ch = core.Chain(fz["ca"], np.arange(n), keys, [str(x) for x in fz["names"]], np.arange(n), np.zeros(n))
        zeros = np.zeros(n)
        pred = core.sites_from_scores(ch, prob, zeros, zeros, zeros, threshold=None)
        pct = prob.argsort().argsort() / max(n - 1, 1)
        kidx = {k: i for i, k in enumerate(keys)}

        def m(site):
            idx = [kidx[k] for k in site if k in kidx]
            return {"n": len(idx), "pct": float(pct[idx].mean()) if idx else np.nan,
                    "top3": found(site, pred, 3), "top5": found(site, pred, 5)}

        rows.append({"id": sid, "primary": m(prim), "secondary": [m(s) for s in secs],
                     "all_sites": [m(g[0]) for g in groups]})
    return rows


BANDS = ((3, 8), (9, 15), (16, 10 ** 6))


def by_size(rows):
    """Every site of every protein (>= 3 residues): share found among the first 3 and 5 predicted sites, by size."""
    sites = [s for r in rows for s in r["all_sites"] if s["n"] >= 3]
    out = {}
    for lo, hi in BANDS:
        sel = [s for s in sites if lo <= s["n"] <= hi]
        out[f"{lo}-{hi if hi < 10 ** 6 else ''}"] = {
            "sites": len(sel), "top3": round(float(np.mean([s["top3"] for s in sel])), 3) if sel else None,
            "top5": round(float(np.mean([s["top5"] for s in sel])), 3) if sel else None,
            "pct_median": round(float(np.median([s["pct"] for s in sel])), 3) if sel else None}
    return {"proteins": len(rows), "sites": len(sites), "bands": out}


def summarize(rows):
    size = by_size(rows)
    rows = [r for r in rows if r["primary"]["n"] >= 3 and any(s["n"] >= 3 for s in r["secondary"])]
    for r in rows:
        r["secondary"] = [s for s in r["secondary"] if s["n"] >= 3]
    P = [r["primary"] for r in rows]
    S = [s for r in rows for s in r["secondary"]]
    d = np.array([r["primary"]["pct"] - np.mean([s["pct"] for s in r["secondary"]]) for r in rows])
    first = [(r["primary"], r["secondary"][0]) for r in rows]
    b = sum(p["top5"] and not s["top5"] for p, s in first)
    c = sum(s["top5"] and not p["top5"] for p, s in first)
    sized = [(p, s) for p, s in first if abs(p["n"] - s["n"]) <= 0.3 * max(p["n"], s["n"])]
    bs = sum(p["top5"] and not s["top5"] for p, s in sized)
    cs = sum(s["top5"] and not p["top5"] for p, s in sized)
    out = {
        "by_size_all_sites": size,
        "proteins": len(rows), "secondary_sites": len(S),
        "size_median": [float(np.median([p["n"] for p in P])), float(np.median([s["n"] for s in S]))],
        "pct_median": [round(float(np.median([p["pct"] for p in P])), 3), round(float(np.median([s["pct"] for s in S])), 3)],
        "top3": [round(float(np.mean([p["top3"] for p in P])), 3), round(float(np.mean([s["top3"] for s in S])), 3)],
        "top5": [round(float(np.mean([p["top5"] for p in P])), 3), round(float(np.mean([s["top5"] for s in S])), 3)],
        "pct_diff_median": round(float(np.median(d)), 3),
        "pct_wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) >= 5 else None,
        "top5_discordant": [b, c], "top5_binom_p": float(binomtest(b, b + c).pvalue) if b + c else None,
        "size_matched": {"pairs": len(sized),
                         "top5": [round(float(np.mean([p["top5"] for p, _ in sized])), 3) if sized else None,
                                  round(float(np.mean([s["top5"] for _, s in sized])), 3) if sized else None],
                         "pct_median": [round(float(np.median([p["pct"] for p, _ in sized])), 3) if sized else None,
                                        round(float(np.median([s["pct"] for _, s in sized])), 3) if sized else None],
                         "discordant": [bs, cs], "binom_p": float(binomtest(bs, bs + cs).pvalue) if bs + cs else None},
    }
    return out


def main():
    res = {}
    for name, args in SETS.items():
        rows = run(name, *args)
        res[name] = summarize(rows)
        print(name, json.dumps(res[name]), flush=True)
    out = D / W / "secondary_sites.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())

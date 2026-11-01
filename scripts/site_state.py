#!/usr/bin/env python3
"""Performance by the state of the site in the apo structure, with definitions that do not use Crypsis, on both
held-out sets.  Second round of review (the earlier analysis, scripts/site_types.py, used Crypsis' own pocket
detector and CryptoBench only).

    python scripts/site_state.py            # after scripts/holo_geometry.py --set {cryptobench,temporal}

Three definitions of a protein's site state, each with the cut-offs of the earlier analysis (closed below 1/3,
open above 2/3, partly open between):

  ligand    share of the main holo ligand's heavy atoms that fit in the apo structure after local superposition
            (no apo heavy atom within 2.5 A; scripts/holo_geometry.py); independent of every predictor
  fpocket   share of the site's residues that line an fpocket 4.0 pocket of the apo structure
  crypsis   share of the site's residues that line a pocket of Crypsis' own detector (static score > 0), the
            earlier definition, kept for comparison

Per set, definition and stratum: the number of proteins, the mean per-protein AUC of each model, paired
per-protein Wilcoxon tests of hybrid - ESM-2 control, physics - ESM-2 control and hybrid - physics (Holm over the
3 strata x 3 comparisons of a definition and set), and the share of proteins whose first site, or one of the
first three, finds a true site (rule of cryptobench.site_hits).  Also, per model and set, two stricter site
criteria: the ligand centroid of a holo record within 4 A or 8 A of the site centre (the Calpha centroid of the
predicted site), and a predicted site holding at least half of a true site's residues.  Results:
~/crypsis_data/work/site_state.json.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402
import heldout_eval as he  # noqa: E402
import holo_geometry as hg  # noqa: E402

MODELS = ["hybrid_650M", "physics", "esm_only_650M", "fpocket", "p2rank", "pocketminer"]
TESTS = [("hybrid_650M", "esm_only_650M"), ("physics", "esm_only_650M"), ("hybrid_650M", "physics")]
STATES = ("closed", "partly_open", "open")


def state(share):
    """Closed below one third, open above two thirds (shares computed from counts, compared with a tolerance)."""
    eps = 1e-12
    return np.where(np.isnan(share), "none", np.where(3 * share < 1 - eps, "closed",
                                                      np.where(3 * share <= 2 + eps, "partly_open", "open")))


def site_detail(scores, recs, data, geo):
    """Per protein: whether site 1 / sites 1-3 find a true site (three rules), under cryptobench.site_hits' calling."""
    from crypsis import core

    out = []
    for sc, r in zip(scores, recs):
        n = len(r["ca"])
        ch = core.Chain(r["ca"], np.arange(n), r["keys"], r["names"], np.arange(n), np.zeros(n))
        z = np.zeros(n)
        sites = core.sites_from_scores(ch, np.asarray(sc), z, z, z, threshold=None)[:3]
        true = cb.pockets_of(data, r["apo"])
        cents = [np.array(x["centroid"]) for x in geo[r["id"]]["records"] if "centroid" in x]
        row = {"default": [], "dcc4": [], "dcc8": [], "cover50": []}
        for st in sites:
            m = set(st["residues"])
            row["default"].append(any(len(m & t) >= 3 and (len(m & t) >= 0.5 * len(m) or len(m & t) >= len(t) / 3.0)
                                      for t in true))
            d = min((np.linalg.norm(np.array(st["centre"]) - c) for c in cents), default=np.inf)
            row["dcc4"].append(d <= 4.0)
            row["dcc8"].append(d <= 8.0)
            row["cover50"].append(any(len(m & t) >= 0.5 * len(t) for t in true))
        out.append({k: (bool(v[:1] and v[0]), any(v)) for k, v in row.items()})
    return out


def rate(rows, rule, k, idx=None):
    idx = range(len(rows)) if idx is None else idx
    v = [rows[i][rule][k] for i in idx]
    return {"rate": round(float(np.mean(v)), 4) if v else None, "ci": he.wilson(np.mean(v), len(v)) if v else None,
            "n": len(v)}


def main():
    from scipy.stats import spearmanr, wilcoxon
    from crypsis import core

    out = {}
    for name, (data, work) in he.SETS.items():
        a = SimpleNamespace(data=data, work=work, physics="v3")
        recs = cb.load_split(a, "test")
        sc = he.all_scores(recs, a)
        models = [m for m in MODELS if m in sc]
        with open(os.path.join(work, "holo_geometry.json")) as fh:
            geo = json.load(fh)["entities"]
        ys = [r["y"] for r in recs]
        fit = []
        for r in recs:
            g = geo[r["id"]]["records"]
            main_ = [x for x in g if x["main"] and "error" not in x] or \
                sorted((x for x in g if "error" not in x), key=lambda x: -x["pRMSD"])
            fit.append(hg.fit_share(main_[0]) if main_ else np.nan)
        fit = np.array(fit)
        fp = np.array([(s[y == 1] > 0).mean() for s, y in zip(sc["fpocket"], ys)])
        crypsis = np.array([(r["feats"][:, core.FEATURES.index("score")][r["y"] == 1] > 0).mean() for r in recs])
        defs = {"ligand": state(fit), "fpocket": state(fp), "crypsis": state(crypsis)}
        per = {m: he.per_protein(sc[m], ys)[0] for m in models}
        sites = {m: site_detail(sc[m], recs, data, geo) for m in models}
        res = {"proteins": len(recs), "models": models, "definitions": {},
               "crosstab_ligand_vs_crypsis": {f"{s}|{t}": int(((defs["ligand"] == s) & (defs["crypsis"] == t)).sum())
                                              for s in STATES for t in STATES},
               "crosstab_ligand_vs_fpocket": {f"{s}|{t}": int(((defs["ligand"] == s) & (defs["fpocket"] == t)).sum())
                                              for s in STATES for t in STATES},
               "fit_share_quantiles": [round(float(q), 3) for q in np.nanquantile(fit, [0.1, 0.25, 0.5, 0.75, 0.9])],
               "state_counts_other_cutoffs": {}, "sites": {}}
        for c in ("2.0", "3.0"):
            alt = []
            for r in recs:
                g = [x for x in geo[r["id"]]["records"] if "error" not in x]
                main_ = [x for x in g if x["main"]] or sorted(g, key=lambda x: -x["pRMSD"])
                alt.append(hg.fit_share(main_[0], c) if main_ else np.nan)
            st = state(np.array(alt))
            res["state_counts_other_cutoffs"][c] = {s: int((st == s).sum()) for s in STATES}
        d = per["physics"] - per["esm_only_650M"]
        ok = ~np.isnan(d) & ~np.isnan(fit)
        rho = spearmanr(fit[ok], d[ok])
        res["spearman_fit_share_vs_physics_minus_esm"] = {"rho": round(float(rho.correlation), 3),
                                                          "p": float(rho.pvalue), "n": int(ok.sum())}
        for dname, st in defs.items():
            block = {"counts": {s: int((st == s).sum()) for s in STATES}, "strata": {}}
            ps, where = [], []
            for s in STATES:
                idx = np.flatnonzero(st == s)
                row = {"n": int(len(idx)), "mean_auc": {}, "site_top1": {}, "site_top3": {}, "tests": {}}
                for m in models:
                    v = per[m][idx]
                    row["mean_auc"][m] = round(float(np.nanmean(v)), 4) if len(idx) else None
                    row["site_top1"][m] = rate(sites[m], "default", 0, idx)
                    row["site_top3"][m] = rate(sites[m], "default", 1, idx)
                for x, y in TESTS:
                    dd = (per[x] - per[y])[idx]
                    dd = dd[~np.isnan(dd)]
                    p = float(wilcoxon(dd).pvalue) if len(dd) > 5 else np.nan
                    row["tests"][f"{x} - {y}"] = {"n": int(len(dd)), "mean_diff": round(float(dd.mean()), 4),
                                                  "mean_diff_ci": he.per_protein_diff_ci(dd),
                                                  "median_diff": round(float(np.median(dd)), 4),
                                                  "wins": int((dd > 0).sum()), "wilcoxon_p": p}
                    ps.append(p)
                    where.append((s, f"{x} - {y}"))
                block["strata"][s] = row
            for (s, k), q in zip(where, he.holm(ps)):
                block["strata"][s]["tests"][k]["wilcoxon_p_holm"] = q
            res["definitions"][dname] = block
        for m in models:
            res["sites"][m] = {f"{rule}_top{k + 1 if k == 0 else 3}": rate(sites[m], rule, k)
                               for rule in ("default", "dcc4", "dcc8", "cover50") for k in (0, 1)}
        out[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k not in ("definitions", "sites")}))
        for dname, block in res["definitions"].items():
            print(" ", dname, block["counts"])
            for s, row in block["strata"].items():
                print("    ", s, row["n"], {m: row["mean_auc"][m] for m in models},
                      {k: (t["mean_diff"], round(t["wilcoxon_p_holm"], 4)) for k, t in row["tests"].items()})
        for m in models:
            print("  sites", m, {k: v["rate"] for k, v in res["sites"][m].items()})
    with open(os.path.join(he.SETS["cryptobench"][1], "site_state.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

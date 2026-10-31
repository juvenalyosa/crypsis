#!/usr/bin/env python3
"""One table for a held-out set: the shipped Crypsis models, the ESM-2-only control and fpocket, scored with the
same metrics, with paired protein-bootstrap differences and per-protein paired tests.

    python scripts/heldout_eval.py --set cryptobench                      # CryptoBench test split (222)
    python scripts/heldout_eval.py --set temporal                         # temporal set (scripts/temporal_set.py)
    python scripts/heldout_eval.py --set temporal --holm-only             # add Holm-adjusted P to an existing file

The temporal set is first run through the CryptoBench pipeline with its own data and work directories (a
CryptoBench-style folds.json whose "test" holds every apo structure):

    python scripts/cryptobench.py {features,descriptors,ensemble} --split test --data <tdata> --work <twork>
    python scripts/cryptobench.py esm --esm {650M,150M} --work <twork>
    python scripts/cryptobench.py test --shipped --esm 650M --data <tdata> --work <twork>   # and 150M
    python scripts/baselines.py fpocket --data <tdata> --work <twork>

Nothing is refitted on the held-out set.  The ESM-2-only control is fitted on the four CryptoBench training folds,
as in scripts/controls.py; for the CryptoBench test split its saved scores are reused.  Subsets of the temporal set
(fixed before scoring): apo structures released after CryptoBench (apo_is_new), and proteins whose highest global
identity to any CryptoBench apo chain is below 0.40 or 0.30 (max_identity_to_cryptobench).
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controls  # noqa: E402
import cryptobench as cb  # noqa: E402

HOME = os.path.expanduser("~/crypsis_data")
SETS = {"cryptobench": (f"{HOME}/cryptobench", f"{HOME}/work"),
        "temporal": (f"{HOME}/temporal_bench", f"{HOME}/work_temporal")}
TRAIN = SETS["cryptobench"]
PCS = {"650M": 256, "150M": 128}
DIFFS = [("hybrid_650M", "physics"), ("hybrid_650M", "esm_only_650M"), ("physics", "esm_only_650M"),
         ("hybrid_150M", "esm_only_150M"), ("physics", "esm_only_150M"), ("physics", "fpocket"),
         ("hybrid_650M", "fpocket"), ("hybrid_650M", "hybrid_150M"),
         # added after review: static-only ablation (scripts/ablation_static.py) and stronger embedding-only
         # controls (scripts/controls.py esm-strong); used only when their score files exist
         ("physics", "physics_static"), ("hybrid_650M", "hybrid_static"), ("physics_static", "esm_only_650M"),
         ("hybrid_static", "esm_only_650M"), ("physics", "esm_pooled_650M"), ("physics", "esm_mlp_650M"),
         ("hybrid_650M", "esm_pooled_650M"), ("hybrid_650M", "esm_mlp_650M")]


def per_protein(scores, ys):
    from sklearn.metrics import average_precision_score, roc_auc_score

    ok = [0 < y.sum() < len(y) for y in ys]
    auc = np.array([roc_auc_score(y, s) if k else np.nan for s, y, k in zip(scores, ys, ok)])
    ap = np.array([average_precision_score(y, s) if k else np.nan for s, y, k in zip(scores, ys, ok)])
    return auc, ap


def per_protein_means(auc, ap, n=2000, seed=0):
    """Mean per-protein AUC and AUPRC with 95% intervals over protein resamples."""
    ok = ~np.isnan(auc)
    auc, ap = auc[ok], ap[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(auc), (n, len(auc)))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"mean_auc_per_protein": round(float(auc.mean()), 4), "mean_auc_per_protein_ci": ci(auc[idx].mean(1)),
            "mean_auprc_per_protein": round(float(ap.mean()), 4), "mean_auprc_per_protein_ci": ci(ap[idx].mean(1))}


def wilson(p, n, z=1.96):
    """95% Wilson interval of a proportion."""
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 4), round(float(c + h), 4)]


def esm_only_scores(recs, esm, work):
    path = os.path.join(work, f"control_esm_only_{esm}_{PCS[esm]}_scores.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)                      # our own file
        pos = {i: k for k, i in enumerate(z["ids"])}
        if all(r["id"] in pos for r in recs):
            return [np.asarray(z["esm_only"][pos[r["id"]]], dtype=float) for r in recs]
    t = SimpleNamespace(data=TRAIN[0], work=TRAIN[1], physics="v3")
    model = controls.fit_esm_only(cb.load_split(t, "train", esm), PCS[esm])
    s = controls.predict_esm_only(model, recs)
    np.savez_compressed(path, ids=np.array([r["id"] for r in recs]), esm_only=np.array(s, dtype=object))
    return s


def all_scores(recs, a):
    ids = [r["id"] for r in recs]
    out = {}
    for esm in PCS:
        z = np.load(os.path.join(a.work, f"test_shipped_v3_{esm}_scores.npz"), allow_pickle=True)  # our own file
        pos = {i: k for k, i in enumerate(z["ids"])}
        if esm == "650M":
            out["physics"] = [np.asarray(z["physics"][pos[i]], dtype=float) for i in ids]
        out[f"hybrid_{esm}"] = [np.asarray(z["hybrid"][pos[i]], dtype=float) for i in ids]
    for esm in PCS:
        emb = cb.load_split(a, "test", esm)
        assert [r["id"] for r in emb] == ids
        out[f"esm_only_{esm}"] = esm_only_scores(emb, esm, a.work)
    fp = []
    for r in recs:
        z = np.load(os.path.join(a.work, "fpocket", f"{r['id']}.npz"))
        assert [str(k) for k in z["keys"]] == r["keys"], r["id"]
        fp.append(z["score"])
    out["fpocket"] = fp
    path = os.path.join(a.work, "ablation_static_scores.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)                      # our own file
        pos = {i: k for k, i in enumerate(z["ids"])}
        for k in ("physics_static", "hybrid_static"):
            out[k] = [np.asarray(z[k][pos[i]], dtype=float) for i in ids]
    for v in ("pooled", "mlp"):
        path = os.path.join(a.work, f"control_esm_{v}_650M_scores.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)                  # our own file
            pos = {i: k for k, i in enumerate(z["ids"])}
            out[f"esm_{v}_650M"] = [np.asarray(z["esm_only"][pos[i]], dtype=float) for i in ids]
    return out


def subsets(recs, data):
    d = json.load(open(os.path.join(data, "cryptobench-dataset", "dataset.json")))
    out = {"all": list(range(len(recs))), "single_chain": [k for k, r in enumerate(recs) if "-" not in r["id"][4:]]}
    meta = [d[r["apo"]][0] for r in recs]
    if "max_identity_to_cryptobench" in meta[0]:
        out["apo_is_new"] = [k for k, m in enumerate(meta) if m["apo_is_new"]]
        out["identity<0.40"] = [k for k, m in enumerate(meta) if m["max_identity_to_cryptobench"] < 0.40]
        out["identity<0.30"] = [k for k, m in enumerate(meta) if m["max_identity_to_cryptobench"] < 0.30]
        out["xray"] = [k for k, m in enumerate(meta) if "X-RAY" in m["method_apo"].upper()]   # added after review
        nov = os.path.join(os.path.dirname(data.rstrip("/")), "temporal", "novelty.json")
        if os.path.exists(nov):                                  # scripts/temporal_novelty.py, added after review
            with open(nov) as fh:
                nv = json.load(fh)
            out["no_homolog"] = [k for k, r in enumerate(recs) if r["apo"] in set(nv["apo_no_homolog"])]
            out["no_homolog_no_pfam"] = [k for k, r in enumerate(recs) if r["apo"] in set(nv["apo_novel"])]
    return out


def holm(ps):
    """Holm step-down adjusted P values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def add_holm(out):
    """Holm adjustment within each subset and metric, over the comparisons in DIFFS (per-protein Wilcoxon tests)."""
    for res in out.values():
        diffs = res["differences"]
        for met in ("auc", "auprc"):
            keys = list(diffs)
            for k, q in zip(keys, holm([diffs[k]["per_protein"][f"{met}_wilcoxon_p"] for k in keys])):
                diffs[k]["per_protein"][f"{met}_wilcoxon_p_holm"] = q
    return out


def evaluate(scores, recs, idx, a):
    from scipy.stats import wilcoxon

    sub = [recs[k] for k in idx]
    ys = [r["y"] for r in sub]
    res = {"models": {}, "differences": {}}
    pp = {}
    for name, sc in scores.items():
        s = [sc[k] for k in idx]
        auc, ap = per_protein(s, ys)
        pp[name] = (auc, ap)
        sites = cb.site_hits(s, sub, a.data)
        res["models"][name] = dict(cb.metrics(s, ys), **sites, **cb.bootstrap(s, ys, n=a.boot),
                                   median_auprc_per_protein=round(float(np.nanmedian(ap)), 4),
                                   **per_protein_means(auc, ap, a.boot),
                                   **{f"{k}_ci": wilson(v, len(idx)) for k, v in sites.items()})
    for x, y in [d for d in DIFFS if d[0] in scores and d[1] in scores]:
        d = cb.bootstrap([scores[x][k] for k in idx], ys, n=a.boot, other=[scores[y][k] for k in idx])
        ok = ~np.isnan(pp[x][0])
        da, dp = pp[x][0][ok] - pp[y][0][ok], pp[x][1][ok] - pp[y][1][ok]
        res["differences"][f"{x} - {y}"] = {
            "auc_diff_ci": d["auc_diff_ci"], "auprc_diff_ci": d["auprc_diff_ci"],
            "per_protein": {"n": int(ok.sum()), "auc_median_diff": round(float(np.median(da)), 4),
                            "auc_wins": int((da > 0).sum()), "auc_wilcoxon_p": float(wilcoxon(da).pvalue),
                            "auprc_median_diff": round(float(np.median(dp)), 4),
                            "auprc_wins": int((dp > 0).sum()), "auprc_wilcoxon_p": float(wilcoxon(dp).pvalue)}}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--holm-only", action="store_true")
    a = ap.parse_args()
    a.data, a.work = SETS[a.set]
    path = os.path.join(a.work, f"heldout_{a.set}.json")
    if a.holm_only:
        with open(path) as fh:
            out = add_holm(json.load(fh))
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print("->", path)
        return
    a.physics = "v3"
    recs = cb.load_split(a, "test")
    scores = all_scores(recs, a)
    out = {}
    for name, idx in subsets(recs, a.data).items():
        out[name] = evaluate(scores, recs, idx, a)
        print(f"== {a.set} {name}: {len(idx)} proteins", flush=True)
        for m, r in out[name]["models"].items():
            print(f"  {m:15s} AUC {r['auc']:.3f} {r['auc_ci']} AUPRC {r['auprc']:.3f} {r['auprc_ci']} "
                  f"site {r['site_top1']:.2f}/{r['site_top3']:.2f}", flush=True)
        for k, r in out[name]["differences"].items():
            p = r["per_protein"]
            print(f"  {k:32s} dAUC {r['auc_diff_ci']} dAUPRC {r['auprc_diff_ci']}  per-protein AUC wins "
                  f"{p['auc_wins']}/{p['n']} p={p['auc_wilcoxon_p']:.2g}", flush=True)
    with open(path, "w") as fh:
        json.dump(add_holm(out), fh, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Performance against protein size, added after review: the normal-mode conformers are deformed to a fixed Calpha
RMSD, which distorts small proteins more (scripts/ensemble_geometry.py).  Does accuracy, or the gain from the
ensembles, depend on size?

    python scripts/size_strata.py

For each set (CryptoBench test, temporal) and model (scores as in scripts/heldout_eval.py): Spearman correlation of
the per-protein AUC with the number of residues scored, and the mean per-protein AUC in size tertiles (with the
2,000-resample interval of each mean, heldout_eval.per_protein_means).  Also the per-protein ensemble gain
(physics - physics_static, hybrid_650M - hybrid_static) against size.  Results: work/size_strata.json.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402
import heldout_eval as he  # noqa: E402

MODELS = ["hybrid_650M", "physics", "physics_static", "hybrid_static", "esm_only_650M", "fpocket"]


def _mean(v):
    m = he.per_protein_means(v, v)
    return [m["mean_auc_per_protein"], *m["mean_auc_per_protein_ci"]]


def main():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    res = {}
    for name, (data, work) in he.SETS.items():
        a = SimpleNamespace(data=data, work=work, physics="v3")
        recs = cb.load_split(a, "test")
        sc = he.all_scores(recs, a)
        ys = [r["y"] for r in recs]
        size = np.array([len(y) for y in ys])
        tert = np.quantile(size, [1 / 3, 2 / 3])
        grp = np.where(size <= tert[0], "small", np.where(size <= tert[1], "medium", "large"))
        out = {"proteins": len(recs), "size_tertiles": [int(t) for t in tert], "models": {}}
        per = {}
        for m in MODELS:
            if m not in sc:
                continue
            per[m] = np.array([roc_auc_score(y, s) for s, y in zip(sc[m], ys)])
            rho, p = spearmanr(size, per[m])
            out["models"][m] = {"spearman_auc_size": round(float(rho), 3), "p": float(p),
                                "tertile_mean_auc": {g: _mean(per[m][grp == g]) for g in ("small", "medium", "large")}}
        for full, static in (("physics", "physics_static"), ("hybrid_650M", "hybrid_static")):
            if full in per and static in per:
                d = per[full] - per[static]
                rho, p = spearmanr(size, d)
                out["models"][f"{full} - {static}"] = {
                    "spearman_gain_size": round(float(rho), 3), "p": float(p),
                    "tertile_mean_gain": {g: round(float(d[grp == g].mean()), 4) for g in ("small", "medium", "large")}}
        res[name] = out
        print(name, json.dumps(out, indent=1), flush=True)
    with open(os.path.join(he.SETS["cryptobench"][1], "size_strata.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

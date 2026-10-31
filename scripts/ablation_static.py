#!/usr/bin/env python3
"""Static-only models: the physics and hybrid recipes without any column that depends on the normal modes or the
ensembles, added after review to isolate what the ensembles contribute.

    python scripts/ablation_static.py --esm 650M --pcs 256

The 17 dropped columns (ENSEMBLE_COLUMNS) are the ensemble pocket statistics of every ensemble (3 A, 5 A, trimmed,
shallow pass), their neighbourhood means and smoothed versions, and the ANM mobility; the 32 kept columns describe
the apo structure alone (static pockets, trimmed-side-chain pockets of the structure, exposure, contacts, Gaussian
network fluctuations, geometry, B-factor, propensity).  Same learner, settings, PCA and training labels as the
shipped models; 4-fold cross-validation on the training folds, then one fit on all of them, scored once on the
CryptoBench test split and the temporal set.  Scores go to <work>/ablation_static_scores.npz for
scripts/heldout_eval.py.
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

ENSEMBLE_COLUMNS = ["ens_score_mean", "ens_volume_mean", "ens_open_fraction", "ens_gain", "nma_mobility",
                    "patch_ens_score", "patch_ens_gain", "smooth_ens_volume_mean", "smooth_ens_open_fraction",
                    "shallow_ens_volume", "shallow_ens_open", "amp50_score", "amp50_open", "amp50_gain",
                    "trim_ens_score", "trim_ens_open", "trim_ens_gain"]
KEEP = [j for j, c in enumerate(controls.COLUMNS) if c not in ENSEMBLE_COLUMNS]


def fit(recs, n_pcs):
    from sklearn.decomposition import PCA

    prop = cb.fit_propensity(recs)
    X = np.vstack([x[:, KEEP] for x in cb.physics(recs, prop)])
    y = np.concatenate([r["y"] for r in recs])
    pca = PCA(n_pcs, random_state=0).fit(np.vstack([r["emb"] for r in recs]))
    E = pca.transform(np.vstack([r["emb"] for r in recs]))
    return {"prop": prop, "phys": cb.hgb().fit(X, y), "pca": pca, "hyb": cb.hgb().fit(np.column_stack([X, E]), y)}


def predict(m, recs):
    Xs = [x[:, KEEP] for x in cb.physics(recs, m["prop"])]
    return ([m["phys"].predict_proba(X)[:, 1] for X in Xs],
            [m["hyb"].predict_proba(np.column_stack([X, m["pca"].transform(r["emb"])]))[:, 1] for X, r in zip(Xs, recs)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esm", default="650M")
    ap.add_argument("--pcs", type=int, default=256)
    a = ap.parse_args()
    assert len(KEEP) == 32, len(KEEP)
    home = os.path.expanduser("~/crypsis_data")
    t = SimpleNamespace(data=f"{home}/cryptobench", work=f"{home}/work", physics="v3")
    folds = {i: cb.load_split(t, f"train-{i}", a.esm) for i in range(4)}
    res = {"dropped": ENSEMBLE_COLUMNS, "cv": {}}
    oof = {"physics_static": [], "hybrid_static": []}
    ys = []
    for i in range(4):
        m = fit([r for j in range(4) if j != i for r in folds[j]], a.pcs)
        p, h = predict(m, folds[i])
        oof["physics_static"] += p
        oof["hybrid_static"] += h
        ys += [r["y"] for r in folds[i]]
        print(f"fold {i} done", flush=True)
    for k, v in oof.items():
        res["cv"][k] = cb.metrics(v, ys)
        print("cv", k, json.dumps(res["cv"][k]), flush=True)
    m = fit([r for i in range(4) for r in folds[i]], a.pcs)
    for name, data, work in (("cryptobench", f"{home}/cryptobench", f"{home}/work"),
                             ("temporal", f"{home}/temporal_bench", f"{home}/work_temporal")):
        cb._DATA.clear()
        recs = cb.load_split(SimpleNamespace(data=data, work=work, physics="v3"), "test", a.esm)
        p, h = predict(m, recs)
        np.savez_compressed(os.path.join(work, "ablation_static_scores.npz"), ids=np.array([r["id"] for r in recs]),
                            physics_static=np.array(p, dtype=object), hybrid_static=np.array(h, dtype=object))
        print(name, "scored", len(recs), flush=True)
    with open(os.path.join(home, "work", "ablation_static.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

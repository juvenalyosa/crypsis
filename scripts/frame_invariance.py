#!/usr/bin/env python3
"""Does a prediction depend on the coordinate frame of the input file?  Second round of review.

    python scripts/frame_invariance.py         # CryptoBench test split, 30 proteins chosen at random (seed 0)

The pocket grid is aligned with the axes of the input file, so a rigid rotation and translation of the same
structure changes which grid points are free and buried.  For each chosen protein the shipped physics model
(pipeline.predict) is run on the structure as distributed and after three random rigid moves (scipy Rotation.random
with seeds 1-3, and a translation drawn uniformly within +-5 A).  Reported per protein: per-protein ROC AUC in each
frame, the largest change of a residue's probability, the Spearman correlation of the probabilities with the
original frame, and whether site 1 changes (Jaccard index of its residues with the original's below 0.5).  Results:
~/crypsis_data/work/frame_invariance.json.
"""

import concurrent.futures as cf
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

DATA = os.path.expanduser("~/crypsis_data/cryptobench")
N, MOVES = 30, 3


def _one(job):
    from scipy.spatial.transform import Rotation
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from crypsis import core, pipeline

    apo, chain, pos = job
    ch = core.read_chain(os.path.join(DATA, "cif", f"{apo}.cif.gz"), chains=chain.split("-"))
    y = np.array([k in set(pos) for k in ch.res_keys], dtype=int)
    centre = ch.xyz.mean(0)
    probs, sites = [], []
    for k in range(MOVES + 1):
        c = copy.deepcopy(ch)
        if k:
            rng = np.random.default_rng(k)
            c.xyz = (ch.xyz - centre) @ Rotation.random(random_state=k).as_matrix().T + centre + rng.uniform(-5, 5, 3)
        r = pipeline.predict(c, model="physics")
        probs.append(np.asarray(r["probability"], float))
        sites.append(set(r["sites"][0]["residues"]) if r["sites"] else set())
    jac = [len(sites[0] & s) / max(len(sites[0] | s), 1) for s in sites[1:]]
    return {"id": f"{apo}{chain}", "n_res": int(ch.n_res),
            "auc": [round(float(roc_auc_score(y, p)), 4) for p in probs],
            "max_abs_change": round(float(max(np.abs(p - probs[0]).max() for p in probs[1:])), 4),
            "spearman_min": round(float(min(spearmanr(p, probs[0]).correlation for p in probs[1:])), 4),
            "site1_jaccard": [round(j, 3) for j in jac], "site1_changes": bool(min(jac) < 0.5)}


def main():
    ents = cb.entries(DATA, "test")
    pick = sorted(np.random.default_rng(0).choice(len(ents), N, replace=False).tolist())
    with cf.ProcessPoolExecutor(3, max_tasks_per_child=1) as ex:
        rows = list(ex.map(_one, [ents[i] for i in pick]))
    rng_auc = [max(r["auc"]) - min(r["auc"]) for r in rows]
    summary = {"proteins": len(rows), "moves": MOVES,
               "auc_range_median": round(float(np.median(rng_auc)), 4), "auc_range_max": round(float(max(rng_auc)), 4),
               "max_abs_change_median": round(float(np.median([r["max_abs_change"] for r in rows])), 4),
               "max_abs_change_max": round(float(max(r["max_abs_change"] for r in rows)), 4),
               "spearman_min_median": round(float(np.median([r["spearman_min"] for r in rows])), 4),
               "site1_changes": int(sum(r["site1_changes"] for r in rows))}
    print(json.dumps(summary, indent=1))
    with open(os.path.expanduser("~/crypsis_data/work/frame_invariance.json"), "w") as fh:
        json.dump({"summary": summary, "proteins": rows}, fh, indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Geometry of the normal-mode conformers, added after review: how much do the linear, rigid-residue conformers
distort the chain and how many clashes do they create?

    python scripts/ensemble_geometry.py --n 60

For a random sample of CryptoBench test structures (seed 0), every conformer of the shipped 3 A and 5 A ensembles
(core.conformers, shipped seed) is compared with the apo structure:
  * virtual-bond deviation: |d(CA_i, CA_i+1) - d0| over consecutive residues whose apo distance d0 is below 4.2 A;
  * clashes: pairs of heavy atoms of residues at least two apart in the chain, closer than 2.2 A, per 100 residues
    (the apo structure's own count is reported for reference).
Results: work/ensemble_geometry.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402

HOME = os.path.expanduser("~/crypsis_data")


def clashes(xyz, res, cutoff=2.2):
    from scipy.spatial import cKDTree

    pairs = cKDTree(xyz).query_pairs(cutoff, output_type="ndarray")
    if not len(pairs):
        return 0
    return int((np.abs(res[pairs[:, 0]] - res[pairs[:, 1]]) >= 2).sum())


def main():
    from crypsis import core

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    data = f"{HOME}/cryptobench"
    ent = cb.entries(data, "test")
    rng = np.random.default_rng(0)
    pick = [ent[i] for i in sorted(rng.choice(len(ent), a.n, replace=False))]
    rows = []
    for apo, ch_id, _pos in pick:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=ch_id.split("-"))
        p = dict(core.DEFAULTS, **(core.load_model().get("params") or {}))
        vals, modes = core.anm_modes(ch.xyz[ch.ca_index], p["anm_cutoff_A"], p["n_modes"])
        prot = ch.atom_res >= 0
        ca0 = ch.xyz[ch.ca_index]
        d0 = np.linalg.norm(np.diff(ca0, axis=0), axis=1)
        bonded = d0 < 4.2
        row = {"id": f"{apo}{ch_id}", "n_res": int(ch.n_res),
               "apo_clashes_per100": 100 * clashes(ch.xyz[prot], ch.atom_res[prot]) / ch.n_res}
        for rmsd in (3.0, 5.0):
            dev, cl = [], []
            for x in core.conformers(ch, vals, modes, p["n_conformers"], rmsd, p["seed"]):
                ca = x[ch.ca_index]
                d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
                dev.append(np.abs(d - d0)[bonded])
                cl.append(100 * clashes(x[prot], ch.atom_res[prot]) / ch.n_res)
            dev = np.concatenate(dev)
            row[f"{rmsd:.0f}A"] = {"bond_dev_median": round(float(np.median(dev)), 3),
                                    "bond_dev_p95": round(float(np.percentile(dev, 95)), 3),
                                    "bond_dev_frac_gt_0.5": round(float((dev > 0.5).mean()), 4),
                                    "clashes_per100_mean": round(float(np.mean(cl)), 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summ = {"structures": len(rows), "apo_clashes_per100_median": float(np.median([r["apo_clashes_per100"] for r in rows]))}
    for k in ("3A", "5A"):
        for m in ("bond_dev_median", "bond_dev_p95", "bond_dev_frac_gt_0.5", "clashes_per100_mean"):
            v = [r[k][m] for r in rows]
            summ[f"{k}_{m}"] = {"median": round(float(np.median(v)), 4), "max": round(float(np.max(v)), 4)}
        sizes = np.array([r["n_res"] for r in rows])
        summ[f"{k}_bond_dev_p95_vs_size_spearman"] = round(float(
            __import__("scipy.stats").stats.spearmanr(sizes, [r[k]["bond_dev_p95"] for r in rows])[0]), 3)
    print(json.dumps(summ, indent=1))
    with open(os.path.join(HOME, "work", "ensemble_geometry.json"), "w") as fh:
        json.dump({"summary": summ, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()

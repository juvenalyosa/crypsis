#!/usr/bin/env python3
"""Do the normal modes point towards the holo structure, and do the shipped conformers go there?  Second round of
review, both held-out sets.

    python scripts/mode_overlap.py --set cryptobench     # after scripts/holo_geometry.py (same holo files)

For each scored apo entity and its main holo record, the holo Calpha atoms are paired with the apo ones by the
residue-number offset of each chain's pocket residues (holo_pocket_selection against apo_pocket_selection) and
superposed on all paired atoms (Kabsch).  With dr the apo -> holo displacement of the paired atoms:

  overlap_1_30, overlap_31_60   cumulative overlap sqrt(sum_k (v_k . dr / |dr|)^2) of ANM modes 1-30 and 31-60
                                (each mode restricted to the paired atoms and renormalized)
  rmsd_apo_holo                 Calpha RMSD of the paired atoms after superposition
  conf_cos_median               median |cos| between the displacement of each conformer of the shipped 3 A ensemble
                                and dr
  conf_closer                   share of conformers closer to the holo structure than the apo structure is
  components                    connected components of the 15 A Calpha network (more than one: more than six
                                zero modes, which core.anm_modes does not handle)

The summary also relates overlap_1_30 to the per-protein gain of the raw ensemble score over the static score
(Spearman; scripts/perturbation_controls.py run must have been done).

Entities with fewer than 80% of their residues paired are skipped.  Results: <work>/mode_overlap.json.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402
import heldout_eval as he  # noqa: E402
import holo_geometry as hg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from crypsis import core  # noqa: E402


def components(ca, cutoff):
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(ca).query_pairs(cutoff, output_type="ndarray")
    from scipy.sparse import coo_matrix

    n = len(ca)
    m = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        coo_matrix((n, n))
    return int(connected_components(m, directed=False)[0])


def _one(job):
    apo, chain_id, data, holo_dir, records = job
    out = {"id": f"{apo}{chain_id}"}
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
        p = dict(core.DEFAULTS, **(core.load_model().get("params") or {}))
        ca = ch.xyz[ch.ca_index]
        out["components"] = components(ca, p["anm_cutoff_A"])
        out["n_res"] = int(ch.n_res)
        chains = set(chain_id.split("-"))
        main = [h for h in records if h["is_main_holo_structure"] and cb._pocket(h, chains)] or \
            sorted((h for h in records if cb._pocket(h, chains)), key=lambda h: -h["pRMSD"])
        h = main[0]
        hca = hg.read_holo(os.path.join(holo_dir, f"{h['holo_pdb_id']}.cif.gz"))[0]
        offs = {}
        for a, b in zip(h["apo_pocket_selection"], h["holo_pocket_selection"]):
            (ca_, na), (cb_, nb) = a.split("_"), b.split("_")
            if ca_ in chains and ca_ not in offs and na.lstrip("-").isdigit() and nb.lstrip("-").isdigit():
                offs[ca_] = (cb_, int(nb) - int(na))
        idx, hx = [], []
        for i, k in enumerate(ch.res_keys):
            c, num = k.split("_")
            if c in offs and num.lstrip("-").isdigit():
                key = f"{offs[c][0]}_{int(num) + offs[c][1]}"
                if key in hca:
                    idx.append(i)
                    hx.append(hca[key])
        out["paired"] = len(idx)
        if len(idx) < 0.8 * ch.n_res:
            out["skipped"] = "fewer than 80% of residues paired"
            return out
        idx, hx = np.array(idx), np.array(hx)
        r, t = hg.kabsch(hx, ca[idx])
        hx = hx @ r.T + t
        dr = (hx - ca[idx]).ravel()
        out["rmsd_apo_holo"] = round(float(np.sqrt((dr.reshape(-1, 3) ** 2).sum(1).mean())), 3)
        drn = dr / np.linalg.norm(dr)
        k = int(p["n_modes"])
        vals, modes = core.anm_modes(ca, p["anm_cutoff_A"], 2 * k)
        sub = modes[:, idx, :].reshape(len(vals), -1)
        norm = np.linalg.norm(sub, axis=1)
        ov = np.where(norm > 1e-9, (sub @ drn) / np.maximum(norm, 1e-12), 0.0)   # a mode on unpaired atoms only: 0
        out["overlap_1_30"] = round(float(np.sqrt((ov[:k] ** 2).sum())), 4)
        out["overlap_31_60"] = round(float(np.sqrt((ov[k:2 * k] ** 2).sum())), 4)
        out["overlap_best_single"] = round(float(np.abs(ov[:k]).max()), 4)
        v, m = core.anm_modes(ca, p["anm_cutoff_A"], k)
        coss, closer = [], []
        for x in core.conformers(ch, v, m, int(p["n_conformers"]), p["rmsd_A"], p["seed"]):
            d = (x[ch.ca_index][idx] - ca[idx]).ravel()
            coss.append(abs(float(d @ drn / np.linalg.norm(d))))
            rx = np.sqrt(((x[ch.ca_index][idx] - hx) ** 2).sum(1).mean())
            closer.append(rx < out["rmsd_apo_holo"])
        out["conf_cos_median"] = round(float(np.median(coss)), 4)
        out["conf_closer"] = round(float(np.mean(closer)), 4)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def gain_correlation(res, data, work):
    """Spearman correlation of overlap_1_30 with the per-protein AUC gain of the raw ensemble score (perturbation
    controls' `slow`) over the static pocket score."""
    from types import SimpleNamespace

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    x, g = [], []
    for r in cb.load_split(SimpleNamespace(data=data, work=work, physics="v3"), "test"):
        o, f = res.get(r["id"], {}), os.path.join(work, "perturbation", f"{r['id']}.npz")
        if "overlap_1_30" not in o or not os.path.exists(f) or not 0 < r["y"].sum() < len(r["y"]):
            continue
        slow = np.load(f)["slow"]
        x.append(o["overlap_1_30"])
        g.append(roc_auc_score(r["y"], slow) - roc_auc_score(r["y"], r["feats"][:, core.FEATURES.index("score")]))
    rho = spearmanr(x, g)
    return {"rho": round(float(rho.correlation), 3), "p": float(rho.pvalue), "n": len(x)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(he.SETS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    data, work = he.SETS[a.set]
    ds = cb._dataset(data)
    jobs = [(apo, ch, data, hg.HOLO[a.set], ds[apo]) for apo, ch, _pos in cb.entries(data, "test")]
    with cf.ProcessPoolExecutor(a.workers) as ex:
        res = {r["id"]: r for r in ex.map(_one, jobs)}
    ok = [r for r in res.values() if "overlap_1_30" in r]
    q = lambda key: [round(float(v), 3) for v in np.nanquantile([r[key] for r in ok], [0.25, 0.5, 0.75])]
    summary = {"entities": len(res), "analysed": len(ok),
               "disconnected": sorted(k for k, r in res.items() if r.get("components", 1) > 1),
               "errors": {k: r["error"] for k, r in res.items() if "error" in r},
               **{f"{k}_quartiles": q(k) for k in ("overlap_1_30", "overlap_31_60", "rmsd_apo_holo",
                                                   "conf_cos_median", "conf_closer")},
               "share_with_any_conformer_closer": round(float(np.mean([r["conf_closer"] > 0 for r in ok])), 3),
               "share_1_30_above_31_60": round(float(np.mean([r["overlap_1_30"] > r["overlap_31_60"] for r in ok])), 3)}
    summary["spearman_overlap_vs_raw_ensemble_gain"] = gain_correlation(res, data, work)
    print(json.dumps(summary, indent=1))
    with open(os.path.join(work, "mode_overlap.json"), "w") as fh:
        json.dump({"summary": summary, "entities": res}, fh, indent=1)


if __name__ == "__main__":
    main()

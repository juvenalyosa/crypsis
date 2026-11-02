#!/usr/bin/env python3
"""Does the ensemble open the true site, or only resample the grid?  Per-conformer pocket volume of the true
cryptic site under four kinds of perturbation, against its volume in the apo and the holo structure.  Second round
of review (Supplementary Fig. S6).

    python scripts/ensemble_sites.py        # after scripts/holo_geometry.py; Fig. 6 proteins as paper_structures.py

Cases: K-Ras G12C with GDP and Mg2+ kept (4LDJ; switch-II site of sotorasib, 6OIM) and the three temporal-set
proteins of Fig. 6 (main holo record).  The site is the set of apo residues of the true pocket; its volume in a
structure is core.site_volume, the pocket grid points (clusters of at least the minimum volume) within 4 A of the
site residues' atoms, as Crypsis reports for its sites.  24 copies each, shipped seed:

  modes 1-30   the shipped 3 A ensemble          modes 31-60   the same weighting and RMSD on the next 30 modes
  rigid        24 random rigid moves of the apo structure (grid resampling only)
  smooth       a smooth random deformation (10 A Gaussian correlation length), 3 A Calpha RMSD

The holo value is the same site volume in the holo structure (ligand and other hetero groups removed; site residues
in holo numbering).  Results: ~/crypsis_data/work_temporal/ensemble_sites.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import literature_cases as lc  # noqa: E402
import paper_structures as ps  # noqa: E402
import perturbation_controls as pc  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from crypsis import core  # noqa: E402

D = os.path.expanduser("~/crypsis_data")


def volumes(ch, idx, xs, p):
    out = []
    for x in xs:
        g = core.pocket_grid(x, ch.atom_res, ch.n_res, p)
        big = np.isin(g["label"], np.flatnonzero(g["cluster_volume"] >= p["min_volume_A3"]))
        out.append(core.site_volume(x, ch.atom_res, idx, g["points"][big], p["grid_A"]))
    return out


def case(name, apo_path, chains, site_keys, holo_path, holo_chains, holo_keys, hetero=None):
    p = dict(core.DEFAULTS, **(core.load_model().get("params") or {}))
    ch = core.read_chain(apo_path, chains=chains, keep_hetero=hetero)
    idx = np.array([i for i, k in enumerate(ch.res_keys) if k in set(site_keys)])
    n, k, nconf, rho = ch.n_res, int(p["n_modes"]), int(p["n_conformers"]), p["rmsd_A"]
    ca = ch.xyz[ch.ca_index]
    vals, modes = core.anm_modes(ca, p["anm_cutoff_A"], 2 * k)
    v, m = core.anm_modes(ca, p["anm_cutoff_A"], k)
    rng = np.random.default_rng(p["seed"])
    centre = ch.xyz.mean(axis=0)
    rigid = [(ch.xyz - centre) @ pc._rotation(rng).T + centre + rng.uniform(-0.5, 0.5, size=3) for _ in range(nconf)]
    d2 = ((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1)
    kern = np.exp(-d2 / (2 * pc.L_SMOOTH ** 2))
    rng = np.random.default_rng(p["seed"])
    smooth = [pc._scaled(ch, kern @ rng.normal(size=(n, 3)), rho) for _ in range(nconf)]
    res = {"name": name, "site_residues": len(idx),
           "apo": volumes(ch, idx, [ch.xyz], p)[0],
           "modes_1_30": volumes(ch, idx, core.conformers(ch, v, m, nconf, rho, p["seed"]), p),
           "modes_31_60": volumes(ch, idx, core.conformers(ch, vals[k:], modes[k:], nconf, rho, p["seed"]), p),
           "rigid": volumes(ch, idx, rigid, p), "smooth": volumes(ch, idx, smooth, p)}
    hch = core.read_chain(holo_path, chains=holo_chains)
    hidx = np.array([i for i, kk in enumerate(hch.res_keys) if kk in set(holo_keys)])
    res["holo"] = volumes(hch, hidx, [hch.xyz], p)[0]
    print(name, {kk: (round(float(np.mean(vv)), 1) if isinstance(vv, list) else vv) for kk, vv in res.items()},
          flush=True)
    return res


def main():
    out = []
    pdb = os.path.join(D, "cases", "pdb")
    true = lc.true_site(pdb, "4LDJ", ["A"], "6OIM", "A", "MOV")
    out.append(case("K-Ras G12C + GDP (4LDJ; 6OIM)", os.path.join(pdb, "4LDJ.cif.gz"), ["A"], true,
                    os.path.join(pdb, "6OIM.cif.gz"), ["A"], true, hetero={"GDP", "MG"}))
    for s, p in ps.pick_main(ps.load("temporal")).items():
        rec = p["record"]
        chains = p["chain"].split("-")
        apo_keys = [a for a in rec["apo_pocket_selection"] if a.split("_")[0] in chains]
        holo_keys = [b for a, b in zip(rec["apo_pocket_selection"], rec["holo_pocket_selection"])
                     if a.split("_")[0] in chains]
        hchains = sorted({b.split("_")[0] for b in holo_keys})
        out.append(case(f"{ps.STATE_NAME[s]}: {p['apo'].upper()} {p['chain']} ({rec['holo_pdb_id'].upper()})",
                        os.path.join(p["data"], "cif", f"{p['apo']}.cif.gz"), chains, apo_keys,
                        os.path.join(p["holo_dir"], f"{rec['holo_pdb_id']}.cif.gz"), hchains, holo_keys))
    with open(os.path.join(D, "work_temporal", "ensemble_sites.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

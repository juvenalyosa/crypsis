#!/usr/bin/env python3
"""How the temporal-set result depends on how the set was built.  Second round of review.

    python scripts/temporal_sensitivity.py       # after scripts/holo_geometry.py --set temporal

Each variant changes which holo records label an apo entity (a residue is positive when it lines the pocket of a
remaining record, the rule of cryptobench.entries) or which entities are kept, and is scored with the stored
scores of scripts/heldout_eval.py (nothing is refitted).  The variants were fixed before any of them was scored:

  all                 the set as built (193 proteins)
  records<=10         without proteins labelled by more than 10 holo records (fragment campaigns and symmetry
                      copies)
  positives<=15%      without proteins of which more than 15% of residues are positive
  no_glycan_lipid     without records whose ligand is a residue of a branched (oligosaccharide) entity, is
                      covalently attached to the protein (any heavy atom within 2 A of a polymer atom), or is a
                      lipid or detergent (LIPIDS); proteins left without records are dropped
  no_altloc           without records whose ligand or a holo pocket residue has alternate locations (the set
                      keeps the first conformer); proteins left without records are dropped
  main_record_only    positives from the main (largest-pRMSD) holo record only
  curated             records<=10, positives<=15% and no_glycan_lipid together
  holo_<year>         proteins by the release year of their main holo record

Per variant: pooled AUC / AUPRC with protein-bootstrap intervals for the physics and hybrid models and the
ESM-2-only controls, their paired differences (pooled, with paired bootstrap P; per protein, with Wilcoxon P),
Holm over the comparisons of a variant.  Results: ~/crypsis_data/work_temporal/temporal_sensitivity.json.
"""

import copy
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402
import heldout_eval as he  # noqa: E402

LIPIDS = {"Y01", "CLR", "CHD", "LMT", "LMN", "OLC", "OLA", "OLB", "PLM", "MYR", "STE", "BOG", "PCW", "POV", "PEE",
          "PEF", "PEV", "LHG", "3PE", "PGW", "PGV", "CDL", "DGA", "LPP", "D10", "D12", "HTG", "LDA", "SDS", "UMQ",
          "LMU", "DDQ", "C8E", "CPS", "2CV", "PTY", "P5S", "PC1", "6PL", "UNL"}
MODELS = ["physics", "hybrid_650M", "esm_only_650M"]  # the models of the main comparison


def variants(ds, geo, recs):
    """{name: {apo: [records kept]}} (a protein is dropped when it has no records)."""
    flags = {}
    for r in recs:
        g = {x["record"]: x for x in geo[r["id"]]["records"]}
        flags[r["apo"]] = g
    base = {r["apo"]: list(enumerate(ds[r["apo"]])) for r in recs}
    pos_share = {r["apo"]: r["y"].mean() for r in recs}

    def keep(pred):
        out = {}
        for apo, hs in base.items():
            k = [(i, h) for i, h in hs if pred(apo, i, h)]
            if k:
                out[apo] = k
        return out

    def bad_ligand(apo, i, h):
        f = flags[apo].get(i, {})
        return bool(f.get("branched") or f.get("covalent")) or h["ligand"] in LIPIDS

    v = {"all": base,
         "records<=10": {a: hs for a, hs in base.items() if len(hs) <= 10},
         "positives<=15%": {a: hs for a, hs in base.items() if pos_share[a] <= 0.15},
         "no_glycan_lipid": keep(lambda apo, i, h: not bad_ligand(apo, i, h)),
         "no_altloc": keep(lambda apo, i, h: not (flags[apo].get(i, {}).get("ligand_altloc")
                                                  or flags[apo].get(i, {}).get("pocket_altloc"))),
         "main_record_only": keep(lambda apo, i, h: h["is_main_holo_structure"])}
    gl = v["no_glycan_lipid"]
    v["curated"] = {a: hs for a, hs in gl.items() if len(base[a]) <= 10 and pos_share[a] <= 0.15}
    years = {}
    for apo, hs in base.items():
        m = [h for _i, h in hs if h["is_main_holo_structure"]][0]
        years.setdefault(m["holo_release_date"][:4], []).append(apo)
    for y, apos in sorted(years.items()):
        v[f"holo_{y}"] = {a: base[a] for a in apos}
    return v


def main():
    data, work = he.SETS["temporal"]
    a = SimpleNamespace(data=data, work=work, physics="v3", boot=2000)
    recs = cb.load_split(a, "test")
    scores = he.all_scores(recs, a)
    scores = {m: scores[m] for m in MODELS if m in scores}
    ds = copy.deepcopy(cb._dataset(data))
    with open(os.path.join(work, "holo_geometry.json")) as fh:
        geo = json.load(fh)["entities"]
    import concurrent.futures as cf

    jobs = {}
    with cf.ProcessPoolExecutor(4) as ex:                      # variants in parallel; same results
        for name, kept in variants(ds, geo, recs).items():
            jobs[name] = ex.submit(score_variant, name, kept, recs, scores, ds, data, a)
    out = {name: f.result() for name, f in jobs.items()}
    he.add_holm(out)
    with open(os.path.join(work, "temporal_sensitivity.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def score_variant(name, kept, recs, scores, ds, data, a):
    """Relabel the kept records, score the stored predictions and return heldout_eval.evaluate's result."""
    idx = [k for k, r in enumerate(recs) if r["apo"] in kept]
    sub = []
    filtered = {}
    for k in idx:
        r = dict(recs[k])
        hs = [h for _i, h in kept[r["apo"]]]
        filtered[r["apo"]] = hs
        chains = set(ds[r["apo"]][0]["apo_chain"].split("-"))
        pos = set()
        for h in hs:
            pos |= cb._pocket(h, chains) or set()
        r["y"] = np.array([1 if key in pos else 0 for key in r["keys"]], dtype=int)
        sub.append(r)
    cb._DATA.update(path=data, dataset=filtered)              # site hits use the kept records
    sc = {m: [v[k] for k in idx] for m, v in scores.items()}
    res = he.evaluate(sc, sub, list(range(len(sub))), a)
    res["proteins"], res["records"] = len(sub), sum(len(kept[r["apo"]]) for r in sub)
    res["positives"] = int(sum(r["y"].sum() for r in sub))
    p = res["models"]
    d = res["differences"].get("physics - esm_only_650M", {})
    print(f"{name:18s} n={len(sub):3d} rec={res['records']:4d} pos={res['positives']:5d} physics "
          f"{p['physics']['auc']:.3f} hybrid {p['hybrid_650M']['auc']:.3f} esm {p['esm_only_650M']['auc']:.3f} "
          f"phys-esm {d.get('auc_diff_ci')} p_boot={d.get('auc_diff_p')} "
          f"p_wil={d.get('per_protein', {}).get('auc_wilcoxon_p'):.2g}", flush=True)
    return res


if __name__ == "__main__":
    main()

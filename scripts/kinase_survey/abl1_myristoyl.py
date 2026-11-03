"""Crypsis and the untrained physics scores (scripts/untrained_physics.py) at the ABL1 myristoyl pocket
(A340, F341, C462, A463, V502; residues lining myristate in 2FO0), on the ABL1 chains of the kinase survey.

  python abl1_untrained.py one PDB CHAIN    # features of one chain (own process: gemmi 0.7.5 segfaults otherwise)
  python abl1_untrained.py summary          # ranks of the pocket and the sites called on each score

Version: DATA=<kinase run directory> CRYPSIS_SRC=<source tree>; defaults are the v0.2.1 run and the installed package.
The summary goes to DATA/results/abl1_myristoyl.json."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # run_chains
sys.path.insert(0, str(HERE.parent))                # untrained_physics
if os.environ.get("CRYPSIS_SRC"):           # e.g. the v0.2.1 worktree
    sys.path.insert(0, os.environ["CRYPSIS_SRC"])
from crypsis import core  # noqa: E402

# the kinase run of this version (DATA, default the paper's v0.2.1 run); 0.2.1 computes in the frame of the file
K = Path(os.environ.get("DATA", Path.home() / "crypsis_data/kinases_v021"))
MYR = (340, 341, 462, 463, 502)
HOLO = {"2fo0", "3pyy", "6npe", "6npu", "6npv", "7n9g", "8ssn"}


def one(pdb, ch):
    import run_chains as R

    out = K / f"feats_abl1/{pdb}_{ch}.npz"
    if out.exists():
        return
    info = json.loads((K / "meta/kinases.json").read_text())["ABL1"]
    chain, nums = R.domain_chain(K / f"raw_cif/{pdb}_updated.cif", ch, info)
    c = core.in_canonical_frame(chain) if hasattr(core, "in_canonical_frame") else chain
    feats, _ = core.features(c)
    np.savez_compressed(out, feats=feats, nums=np.array([n if n is not None else -1 for n in nums]),
                        keys=np.array(c.res_keys), names=np.array(c.res_names), ca=c.xyz[c.ca_index])


def summary():
    from untrained_physics import untrained

    rows = []
    for f in sorted((K / "feats_abl1").glob("*.npz")):
        z = np.load(f)
        nums = list(z["nums"])
        idx = [nums.index(n) for n in MYR if n in nums]
        n = len(nums)
        ch = core.Chain(z["ca"], np.arange(n), [str(k) for k in z["keys"]], [str(x) for x in z["names"]],
                        np.arange(n), np.zeros(n))
        zeros = np.zeros(n)
        r = {"id": f.stem, "holo": f.stem[:4] in HOLO}
        pred = json.loads((K / f"pred/ABL1/{f.stem}.json").read_text())
        scores = dict(untrained(z["feats"]), crypsis=np.asarray(pred["probability"]))
        for name, s in scores.items():
            pct = (rankdata(s, method="average") - 1) / (n - 1)     # ties share their average rank
            sites = core.sites_from_scores(ch, s, zeros, zeros, zeros, threshold=None)
            rank = next((st["rank"] for st in sites if len(set(idx) & {ch.res_keys.index(k) for k in st["residues"]}) >= 3), None)
            r[name] = (round(float(pct[idx].mean()), 2), rank)
        rows.append(r)
    out = {}
    for grp in (True, False):
        sel = [r for r in rows if r["holo"] == grp]
        key = "ligand_in_pocket" if grp else "no_ligand_in_pocket"
        out[key] = {"chains": len(sel)}
        print(key, len(sel), "chains")
        for name in ("crypsis", "static", "U1", "U2", "U3"):
            p = float(np.median([r[name][0] for r in sel]))
            ranks = [r[name][1] for r in sel]
            t3, anyr = sum(1 for x in ranks if x and x <= 3), sum(1 for x in ranks if x)
            out[key][name] = {"pocket_percentile_median": round(p, 3), "site_top3": t3, "site_any": anyr}
            print(f"  {name:8} median percentile {p:.2f} | site in top 3: {t3}/{len(sel)} | in any site: {anyr}/{len(sel)}")
    (K / "results").mkdir(exist_ok=True)
    (K / "results/abl1_myristoyl.json").write_text(json.dumps({"rows": rows, "summary": out}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3])
    else:
        summary()

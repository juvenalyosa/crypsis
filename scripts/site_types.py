#!/usr/bin/env python3
"""What the CryptoBench labels reward: cryptic sites against any ligand site, and where the ensembles help.
Added after review.  CryptoBench test split only (it ships the non-cryptic pockets of its apo structures).

    python scripts/site_types.py

Uses CryptoBench's auxiliary-data/non-cryptic-pockets/noncryptic-pockets.json: ligand pockets of the same apo
structures whose apo and holo forms differ by less than the crypticity criterion.  Residues that line only such
pockets are labelled negative in the benchmark.  For each model (scores as in scripts/heldout_eval.py) and for the
static and ensemble pocket columns:

  excl_noncryptic   pooled AUC / AUPRC with non-cryptic-site residues removed from the negatives
  cryptic_vs_noncryptic  AUC of cryptic-site residues against non-cryptic-site residues only
  noncryptic_vs_rest     AUC of non-cryptic-site residues against the other negatives (do models find any site?)
  by_apo_pocket     positives split by whether the residue lines a pocket of the apo structure (static score > 0):
                    AUC of each group against all negatives
  by_site_state     proteins split by the share of their site residues that line an apo pocket (closed < 1/3,
                    partly open 1/3-2/3, open > 2/3): mean per-protein AUC per group
  by_prmsd          proteins split into tertiles of the largest pocket RMSD of their records: mean per-protein AUC

Results: work/site_types.json.  Intervals: 2,000 protein resamples (cryptobench.bootstrap) where stated.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402
import heldout_eval as he  # noqa: E402

MODELS = ["hybrid_650M", "physics", "esm_only_650M", "physics_static", "hybrid_static", "fpocket", "static_score",
          "ens_score_mean"]


def auc(pos, neg):
    from sklearn.metrics import roc_auc_score

    if len(pos) == 0 or len(neg) == 0:
        return None
    return round(float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])), 4)


def main():
    from crypsis import core
    from sklearn.metrics import roc_auc_score

    data, work = he.SETS["cryptobench"]
    a = SimpleNamespace(data=data, work=work, physics="v3")
    recs = cb.load_split(a, "test")
    sc = he.all_scores(recs, a)
    sc["static_score"] = [r["feats"][:, core.FEATURES.index("score")] for r in recs]
    sc["ens_score_mean"] = [r["feats"][:, core.FEATURES.index("ens_score_mean")] for r in recs]
    with open(os.path.join(data, "cryptobench-dataset", "auxiliary-data", "non-cryptic-pockets",
                           "noncryptic-pockets.json")) as fh:
        nc = json.load(fh)
    ds = cb._dataset(data)
    noncryp, lines_apo = [], []
    for r in recs:
        chains = set(r["id"][4:].split("-"))
        keys = set()
        for h in nc.get(r["apo"], []):
            keys |= {k for k in h["apo_pocket_selection"] if k.split("_")[0] in chains}
        noncryp.append(np.array([k in keys for k in r["keys"]]) & (r["y"] == 0))
        lines_apo.append(r["feats"][:, core.FEATURES.index("score")] > 0)
    ys = [r["y"] for r in recs]
    res = {"proteins_with_noncryptic": int(sum(m.any() for m in noncryp)),
           "noncryptic_residues": int(sum(m.sum() for m in noncryp)),
           "positives": int(sum(y.sum() for y in ys)),
           "positives_lining_apo_pocket": int(sum((y.astype(bool) & l).sum() for y, l in zip(ys, lines_apo))),
           "models": {}}
    share = np.array([(l[y == 1]).mean() if y.sum() else np.nan for y, l in zip(ys, lines_apo)])
    state = np.where(share < 1 / 3, "closed", np.where(share <= 2 / 3, "partly_open", "open"))
    prmsd = np.array([max(h["pRMSD"] for h in ds[r["apo"]]) for r in recs])
    tert = np.quantile(prmsd, [1 / 3, 2 / 3])
    pbin = np.where(prmsd <= tert[0], "low", np.where(prmsd <= tert[1], "mid", "high"))
    res["site_state_counts"] = {s: int((state == s).sum()) for s in ("closed", "partly_open", "open")}
    res["prmsd_tertiles"] = [round(float(t), 2) for t in tert]
    for m in MODELS:
        if m not in sc:
            continue
        s = sc[m]
        keep = [~nc_ for nc_ in noncryp]
        ex_s, ex_y = [x[k] for x, k in zip(s, keep)], [y[k] for y, k in zip(ys, keep)]
        allneg = np.concatenate([x[(y == 0) & ~n] for x, y, n in zip(s, ys, noncryp)])
        pos = np.concatenate([x[y == 1] for x, y in zip(s, ys)])
        ncr = np.concatenate([x[n] for x, n in zip(s, noncryp)])
        pos_open = np.concatenate([x[(y == 1) & l] for x, y, l in zip(s, ys, lines_apo)])
        pos_closed = np.concatenate([x[(y == 1) & ~l] for x, y, l in zip(s, ys, lines_apo)])
        neg_all = np.concatenate([x[y == 0] for x, y in zip(s, ys)])
        per = np.array([roc_auc_score(y, x) if 0 < y.sum() < len(y) else np.nan for x, y in zip(s, ys)])
        out = {"excl_noncryptic": dict(cb.metrics(ex_s, ex_y), **cb.bootstrap(ex_s, ex_y, n=2000)),
               "cryptic_vs_noncryptic": auc(pos, ncr), "noncryptic_vs_rest": auc(ncr, allneg),
               "by_apo_pocket": {"positives_lining_apo_pocket": auc(pos_open, neg_all),
                                 "positives_not_lining_apo_pocket": auc(pos_closed, neg_all)},
               "by_site_state": {g: round(float(np.nanmean(per[state == g])), 4) for g in ("closed", "partly_open", "open")},
               "by_prmsd": {g: round(float(np.nanmean(per[pbin == g])), 4) for g in ("low", "mid", "high")}}
        res["models"][m] = out
        print(m, json.dumps({k: v for k, v in out.items() if k != "excl_noncryptic"}),
              "excl", out["excl_noncryptic"]["auc"], out["excl_noncryptic"]["auprc"], flush=True)
    with open(os.path.join(work, "site_types.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print({k: v for k, v in res.items() if k != "models"})


if __name__ == "__main__":
    main()

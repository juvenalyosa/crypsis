#!/usr/bin/env python3
"""Does a paired difference between two models change from the CryptoBench test split to the temporal set?

    python scripts/interaction.py

For each pair of models, the paired difference (pooled AUC and AUPRC) is bootstrapped over proteins within each set
independently (2000 draws, the same resampling as cryptobench.bootstrap); the interval of (temporal - CryptoBench)
is the 2.5-97.5 percentiles of the difference of the two bootstrap distributions.  The two sets share no protein
cluster, so independent resampling is appropriate.  Scores are those saved by scripts/heldout_eval.py's inputs.
A two-sided bootstrap P value (twice the smaller tail of the change distribution at zero) is given for each
comparison and Holm-adjusted over the comparisons, separately for AUC and AUPRC.  AUPRC depends on prevalence, which
differs between the sets (5.4% and 4.0%), so its changes are reported but should not be read as transfer.
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402
import heldout_eval as he  # noqa: E402

PAIRS = [("physics", "esm_only_650M"), ("hybrid_650M", "physics"), ("hybrid_650M", "esm_only_650M"),
         ("physics", "fpocket"), ("physics", "esm_only_150M"),
         ("physics_static", "esm_only_650M"), ("physics", "physics_static"), ("physics", "esm_mlp_650M"),
         ("physics", "esm_pooled_650M")]


def boot_diffs(sc, ys, x, y, n=2000, seed=0):
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(ys), len(ys))
        yy = np.concatenate([ys[i] for i in idx])
        a = np.concatenate([sc[x][i] for i in idx])
        b = np.concatenate([sc[y][i] for i in idx])
        out.append((roc_auc_score(yy, a) - roc_auc_score(yy, b),
                    average_precision_score(yy, a) - average_precision_score(yy, b)))
    return np.array(out)


def main():
    res = {}
    draws = {}
    for k, name in enumerate(("cryptobench", "temporal")):
        cb._DATA.clear()
        a = SimpleNamespace(data=he.SETS[name][0], work=he.SETS[name][1], physics="v3")
        recs = cb.load_split(a, "test")
        sc = he.all_scores(recs, a)
        ys = [r["y"] for r in recs]
        draws[name] = {p: boot_diffs(sc, ys, *p, seed=k) for p in PAIRS if p[0] in sc and p[1] in sc}
    from heldout_eval import holm

    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    pval = lambda v: float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
    for p in [p for p in PAIRS if p in draws["temporal"] and p in draws["cryptobench"]]:
        d = draws["temporal"][p] - draws["cryptobench"][p]
        res[f"{p[0]} - {p[1]}"] = {"auc_change_ci": ci(d[:, 0]), "auprc_change_ci": ci(d[:, 1]),
                                   "auc_p": pval(d[:, 0]), "auprc_p": pval(d[:, 1])}
    for met in ("auc", "auprc"):
        keys = list(res)
        for k, q in zip(keys, holm([res[k][f"{met}_p"] for k in keys])):
            res[k][f"{met}_p_holm"] = q
    for k, r in res.items():
        print(f"{k}: temporal minus CryptoBench  dAUC {r['auc_change_ci']} (Holm P {r['auc_p_holm']:.3g})  "
              f"dAUPRC {r['auprc_change_ci']} (Holm P {r['auprc_p_holm']:.3g})", flush=True)
    with open(os.path.join(he.SETS["temporal"][1], "interaction.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

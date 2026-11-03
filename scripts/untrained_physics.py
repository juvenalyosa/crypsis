#!/usr/bin/env python3
"""Untrained physics scores against the trained Crypsis 0.3 models.

Rules fixed before looking at any result; no labels, no ESM-2, no classifier.  Columns of core.FEATURES, as stored
by scripts/cryptobench.py in work*_v03/features:

  static     score               pocket score of the structure as deposited (what is already open)
  U1         patch_ens_score     mean pocket score over the ANM ensemble, averaged over an 8 A patch
  U2         patch_ens_gain      gain of the ensemble pocket score over the structure's, 8 A patch
  U3         mean within-protein rank of patch_ens_score, burial and patch_hydrophobic

Untrained scores never saw a label, so every CryptoBench protein (1107, all folds) and the temporal set (193) are
held out for them.  Trained models are compared where they are held out: hybrid 650M out-of-fold on the
CryptoBench training folds, the shipped physics and hybrid models on the CryptoBench test split and on the temporal
set.  Labels for both sets: the evaluation rule of cryptobench.entries('test'), the union of all holo pockets on the
scored chains (not the stored training labels, which differ for a few temporal proteins).

static, U1 and U2 are scored raw and ranked within the protein (U3 is already a rank); the paper uses the ranked
form, as in Supplementary Fig. S1.  Metrics: pooled AUC and AUPRC with 95% protein-bootstrap intervals (cryptobench.bootstrap, 2,000 resamples), mean
per-protein AUC, and the site rule of cryptobench.site_hits (top-10% calling) for the first and first-three sites.

    python untrained_physics.py [0.2.1|0.3]    # default 0.2.1, the version the paper evaluates
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cryptobench as cb  # noqa: E402
from crypsis import core  # noqa: E402

D = Path.home() / "crypsis_data"
CB = D / "cryptobench"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "0.2.1"     # 0.2.1: the paper's work dirs; 0.3: work*_v03
W, WT = ("work", "work_temporal") if VERSION == "0.2.1" else ("work_v03", "work_temporal_v03")
F = core.FEATURES


def rank(v):
    """Within-protein rank in [0, 1], ties sharing their average rank (the apo pocket score is zero for most
    residues; breaking those ties by residue order would add noise)."""
    from scipy.stats import rankdata

    return (rankdata(v, method="average") - 1) / max(len(v) - 1, 1)


def untrained(feats, ranked=False):
    """The fixed rules; ranked=True ranks static, U1 and U2 within the protein (as the models and Supplementary
    Fig. S1 see these columns: the raw scale differs between proteins, which affects pooled metrics only)."""
    c = {n: feats[:, F.index(n)] for n in F}
    out = {"static": c["score"], "U1": c["patch_ens_score"], "U2": c["patch_ens_gain"],
           "U3": (rank(c["patch_ens_score"]) + rank(c["burial"]) + rank(c["patch_hydrophobic"])) / 3.0}
    if ranked:
        out = {k: (rank(v) if k != "U3" else v) for k, v in out.items()}
    return out


def load_scores(path):
    z = np.load(path, allow_pickle=True)            # our own score files (object arrays)
    if "hybrid" in z.files:
        return {m: {str(i): np.asarray(a, float) for i, a in zip(z["ids"], z[m])} for m in ("physics", "hybrid")}
    return {"hybrid": {str(i): np.asarray(z[str(i)], float) for i in z["ids"]}}


def records(feat_dir, ids, label_fn):
    out = []
    for sid in ids:
        z = np.load(feat_dir / f"{sid}.npz")
        keys = [str(k) for k in z["keys"]]
        y = label_fn(sid, keys, z)
        if y is None or y.sum() == 0 or y.sum() == len(y):
            continue
        out.append({"id": sid, "apo": sid[:4], "keys": keys, "names": [str(n) for n in z["names"]],
                    "ca": z["ca"], "y": y, "u": untrained(z["feats"]), "ur": untrained(z["feats"], ranked=True)})
    return out


def site_hits(scores, recs, pockets):
    top1 = top3 = 0
    for sc, r in zip(scores, recs):
        n = len(r["ca"])
        ch = core.Chain(r["ca"], np.arange(n), r["keys"], r["names"], np.arange(n), np.zeros(n))
        zeros = np.zeros(n)
        sites = core.sites_from_scores(ch, np.asarray(sc), zeros, zeros, zeros, threshold=None)
        true = pockets(r["apo"])
        found = []
        for st in sites[:3]:
            m = set(st["residues"])
            found.append(any(len(m & t) >= 3 and (len(m & t) >= 0.5 * len(m) or len(m & t) >= len(t) / 3.0)
                             for t in true))
        top1 += bool(found[:1] and found[0])
        top3 += any(found)
    return round(top1 / len(recs), 3), round(top3 / len(recs), 3)


def evaluate(name, recs, scores, pockets):
    ys = [r["y"] for r in recs]
    from sklearn.metrics import average_precision_score
    allS, allY = np.concatenate(scores), np.concatenate(ys)
    bs = cb.bootstrap(scores, ys)
    per = [roc_auc_score(y, s) for s, y in zip(scores, ys)]
    t1, t3 = site_hits(scores, recs, pockets)
    return {"score": name, "proteins": len(recs), "auc": round(roc_auc_score(allY, allS), 3), "auc_ci": bs["auc_ci"],
            "auprc": round(average_precision_score(allY, allS), 3), "auprc_ci": bs["auprc_ci"],
            "auc_per_protein": round(float(np.mean(per)), 3), "site_top1": t1, "site_top3": t3}


def main():
    res = {}
    cb_ds = json.loads((CB / "cryptobench-dataset/dataset.json").read_text())
    folds = json.loads((CB / "cryptobench-dataset/folds.json").read_text())

    def cb_pockets(apo):
        return cb.pockets_of(str(CB), apo)

    def cb_label(sid, keys, _z):
        pos = set().union(*cb_pockets(sid[:4])) if cb_pockets(sid[:4]) else set()
        return np.array([k in pos for k in keys], dtype=np.int8)

    def cb_ids(split):
        out = []
        for apo in (sum((folds[f"train-{i}"] for i in range(4)), []) if split == "train" else folds["test"]):
            sid = apo + cb_ds[apo][0]["apo_chain"]
            if (D / W / "features" / f"{sid}.npz").exists():
                out.append(sid)
        return out

    tdir = D / WT / "features"

    t_ds = json.loads((D / "temporal/dataset.json").read_text())

    def t_pockets(apo):
        holos = t_ds[apo]
        chains = set(holos[0]["apo_chain"].split("-"))
        out = []
        for h in holos:
            q = {k for k in h["apo_pocket_selection"] if k.split("_")[0] in chains}
            if q and q not in out:
                out.append(q)
        return out

    def t_label(sid, keys, _z):
        # the evaluation rule of cryptobench.entries("test"), not the stored training labels
        pos = set().union(*t_pockets(sid[:4])) if t_pockets(sid[:4]) else set()
        return np.array([k in pos for k in keys], dtype=np.int8)

    oof = load_scores(D / W / "oof_v3_hybrid_650M_256.npz")
    cbt = load_scores(D / W / "test_shipped_v3_650M_scores.npz")
    tmp = load_scores(D / WT / "test_shipped_v3_650M_scores.npz")
    sets = {
        "cryptobench_train_folds": (records(D / W / "features", cb_ids("train"), cb_label), cb_pockets,
                                    {"hybrid_oof": oof["hybrid"]}),
        "cryptobench_test": (records(D / W / "features", cb_ids("test"), cb_label), cb_pockets,
                             {"physics_trained": cbt["physics"], "hybrid_trained": cbt["hybrid"]}),
        "temporal": (records(tdir, sorted(p.stem for p in tdir.glob("*.npz")), t_label), t_pockets,
                     {"physics_trained": tmp["physics"], "hybrid_trained": tmp["hybrid"]}),
    }
    for sname, (recs, pockets, trained) in sets.items():
        rows = []
        for u in ("static", "U1", "U2"):
            rows.append(evaluate(u + " (raw)", recs, [r["u"][u] for r in recs], pockets))
        for u in ("static", "U1", "U2", "U3"):
            rows.append(evaluate(u + " (ranked)", recs, [r["ur"][u] for r in recs], pockets))
        for tname, sc in trained.items():
            sub = [r for r in recs if r["id"] in sc and len(sc[r["id"]]) == len(r["y"])]
            rows.append(dict(evaluate(tname, sub, [sc[r["id"]] for r in sub], pockets), note=f"{len(sub)} proteins"))
        res[sname] = rows
        print(f"== {sname}: {len(recs)} proteins", flush=True)
        for r in rows:
            print(f"  {r['score']:18} AUC {r['auc']:.3f} {r['auc_ci']}  AUPRC {r['auprc']:.3f} {r['auprc_ci']}  "
                  f"per-prot {r['auc_per_protein']:.3f}  top1 {r['site_top1']:.0%} top3 {r['site_top3']:.0%}"
                  f"  n={r['proteins']}", flush=True)
    (D / W / "untrained_physics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())

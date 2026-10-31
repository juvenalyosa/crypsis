#!/usr/bin/env python3
"""Controls added for the manuscript, after the shipped models had been scored on the test split.

    python scripts/controls.py esm-only --esm 650M --pcs 256 --work ~/crypsis_data/work
    python scripts/controls.py esm-only --esm 150M --pcs 128 --work ~/crypsis_data/work
    python scripts/controls.py columns --work ~/crypsis_data/work
    python scripts/controls.py esm-strong --esm 650M --variant pooled   # and --variant mlp

esm-only: the hybrid recipe without the physics columns (the same PCA of the ESM-2 embeddings, the same boosted
trees and settings), cross-validated on the four training folds, trained on all of them and scored once on the
test split, with protein-bootstrap intervals and paired differences to the shipped 0.2.1 hybrid and physics
models (work/test_shipped_v3_<esm>_scores.npz).

paired: paired protein-bootstrap differences between single ensemble and static pocket columns on the test
split, and the shipped models' intervals on the single-chain test subset (184 structures) that Seq2Pocket uses.

columns: each of the 49 physics columns used alone as a score, without any learning: pooled ROC AUC and AUPRC
on the training folds and on the test split. Nothing is fitted, except the amino-acid propensity column, which
is fitted on the training folds.

esm-strong: two stronger embedding-only controls, added after review. "pooled": the 256 principal components plus
their Gaussian neighbourhood mean over Calpha neighbours (pipeline.smoothing_matrix, the operator the physics
columns use), with the same boosted trees; "mlp": a two-layer perceptron (256-256, ReLU, L2 1e-4, Adam, 10 epochs,
batch 2048) on the full standardized embeddings, like the CryptoBench paper's pLM-NN.  Each is cross-validated on the
four training folds, trained on all of them and scored once on the CryptoBench test split and on the temporal set,
with paired differences to the shipped physics and hybrid models.

Nothing here changes the shipped models or the configuration chosen by cross-validation.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from crypsis import core, descriptors  # noqa: E402

SHALLOW = ["shallow_volume", "shallow_score", "shallow_ens_volume", "shallow_ens_open"]
COLUMNS = (core.FEATURES + ["smooth_" + n for n in ("ens_volume_mean", "ens_open_fraction", "score")]
           + ["propensity"] + SHALLOW + list(descriptors.DESCRIPTORS) + list(descriptors.ENSEMBLE))


def fit_esm_only(recs, n_pcs):
    from sklearn.decomposition import PCA

    E = np.vstack([r["emb"] for r in recs])
    y = np.concatenate([r["y"] for r in recs])
    pca = PCA(n_pcs, random_state=0).fit(E)
    return pca, cb.hgb().fit(pca.transform(E), y)


def predict_esm_only(model, recs):
    pca, clf = model
    return [clf.predict_proba(pca.transform(r["emb"]))[:, 1] for r in recs]


def cmd_esm_only(a):
    folds = {i: cb.load_split(a, f"train-{i}", a.esm) for i in range(4)}
    scores, ys = [], []
    for i in range(4):
        train = sum((folds[j] for j in range(4) if j != i), [])
        s = predict_esm_only(fit_esm_only(train, a.pcs), folds[i])
        scores += s
        ys += [r["y"] for r in folds[i]]
        print(f"fold {i}: {cb.metrics(s, [r['y'] for r in folds[i]])}", flush=True)
    cv = dict(cb.metrics(scores, ys), model="esm-only", esm=a.esm, pcs=a.pcs)
    print("cv", json.dumps(cv), flush=True)

    model = fit_esm_only(sum(folds.values(), []), a.pcs)
    recs = cb.load_split(a, "test", a.esm)
    ys = [r["y"] for r in recs]
    ids = [r["id"] for r in recs]
    s = predict_esm_only(model, recs)
    ref = np.load(os.path.join(a.work, f"test_shipped_v3_{a.esm}_scores.npz"), allow_pickle=True)  # our own file
    pos = {i: k for k, i in enumerate(ref["ids"])}
    test = dict(cb.metrics(s, ys), **cb.site_hits(s, recs, a.data), **cb.bootstrap(s, ys, n=a.boot))
    for name in ("hybrid", "physics"):
        other = [np.asarray(ref[name][pos[i]], dtype=float) for i in ids]
        d = cb.bootstrap(other, ys, n=a.boot, other=s)          # shipped model minus esm-only
        test[f"{name}_minus_esm_only"] = {"auc_diff_ci": d["auc_diff_ci"], "auprc_diff_ci": d["auprc_diff_ci"]}
    print("test", json.dumps(test), flush=True)
    out = os.path.join(a.work, f"control_esm_only_{a.esm}_{a.pcs}.json")
    json.dump({"cv": cv, "test": test}, open(out, "w"), indent=1)
    np.savez_compressed(out.replace(".json", "_scores.npz"), ids=np.array(ids), esm_only=np.array(s, dtype=object))
    print("->", out)


def _pooled(recs, pca):
    from crypsis import pipeline

    out = []
    for r in recs:
        z = pca.transform(r["emb"])
        out.append(np.column_stack([z, pipeline.smoothing_matrix(r["ca"]) @ z]))
    return out


def fit_strong(recs, variant, n_pcs=256):
    y = np.concatenate([r["y"] for r in recs])
    if variant == "pooled":
        from sklearn.decomposition import PCA

        pca = PCA(n_pcs, random_state=0).fit(np.vstack([r["emb"] for r in recs]))
        return pca, cb.hgb().fit(np.vstack(_pooled(recs, pca)), y)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    E = np.vstack([r["emb"] for r in recs])
    sc = StandardScaler().fit(E)
    clf = MLPClassifier((256, 256), alpha=1e-4, batch_size=2048, learning_rate_init=1e-3, max_iter=10,
                        shuffle=True, random_state=0)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(E), y)
    return sc, clf


def predict_strong(model, recs, variant):
    a, clf = model
    if variant == "pooled":
        return [clf.predict_proba(x)[:, 1] for x in _pooled(recs, a)]
    return [clf.predict_proba(a.transform(r["emb"]))[:, 1] for r in recs]


def cmd_esm_strong(a):
    from types import SimpleNamespace

    folds = {i: cb.load_split(a, f"train-{i}", a.esm) for i in range(4)}
    scores, ys = [], []
    for i in range(4):
        train = [r for j in range(4) if j != i for r in folds[j]]
        scores += predict_strong(fit_strong(train, a.variant, a.pcs), folds[i], a.variant)
        ys += [r["y"] for r in folds[i]]
        print(f"fold {i} done", flush=True)
    res = {"cv": cb.metrics(scores, ys)}
    print("cv", json.dumps(res["cv"]), flush=True)
    model = fit_strong([r for i in range(4) for r in folds[i]], a.variant, a.pcs)
    home = os.path.expanduser("~/crypsis_data")
    for name, data, work in (("cryptobench", a.data, a.work),
                             ("temporal", f"{home}/temporal_bench", f"{home}/work_temporal")):
        cb._DATA.clear()
        t = SimpleNamespace(data=data, work=work, physics="v3")
        recs = cb.load_split(t, "test", a.esm)
        ids, ys = [r["id"] for r in recs], [r["y"] for r in recs]
        s = predict_strong(model, recs, a.variant)
        out = dict(cb.metrics(s, ys), **cb.site_hits(s, recs, data), **cb.bootstrap(s, ys, n=a.boot))
        ref = np.load(os.path.join(work, f"test_shipped_v3_{a.esm}_scores.npz"), allow_pickle=True)  # our own file
        pos = {i: k for k, i in enumerate(ref["ids"])}
        for m in ("hybrid", "physics"):
            other = [np.asarray(ref[m][pos[i]], dtype=float) for i in ids]
            d = cb.bootstrap(other, ys, n=a.boot, other=s)
            out[f"{m}_minus_control"] = {"auc_diff_ci": d["auc_diff_ci"], "auprc_diff_ci": d["auprc_diff_ci"]}
        res[name] = out
        print(name, json.dumps(out), flush=True)
        np.savez_compressed(os.path.join(work, f"control_esm_{a.variant}_{a.esm}_scores.npz"), ids=np.array(ids),
                            esm_only=np.array(s, dtype=object))
    with open(os.path.join(a.work, f"control_esm_{a.variant}_{a.esm}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def column_metrics(recs, prop):
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = np.vstack(cb.physics(recs, prop))
    y = np.concatenate([r["y"] for r in recs])
    assert X.shape[1] == len(COLUMNS), (X.shape, len(COLUMNS))
    return {c: {"auc": round(float(roc_auc_score(y, X[:, j])), 4),
                "auprc": round(float(average_precision_score(y, X[:, j])), 4)} for j, c in enumerate(COLUMNS)}


def cmd_columns(a):
    train = cb.load_split(a, "train")
    prop = cb.fit_propensity(train)
    test = cb.load_split(a, "test")
    res = {"columns": COLUMNS, "train": column_metrics(train, prop), "test": column_metrics(test, prop),
           "positive_rate": {"train": float(np.concatenate([r["y"] for r in train]).mean()),
                             "test": float(np.concatenate([r["y"] for r in test]).mean())}}
    out = os.path.join(a.work, "control_columns.json")
    json.dump(res, open(out, "w"), indent=1)
    for c in COLUMNS:
        print(f"{c:28s} train AUC {res['train'][c]['auc']:.3f} AUPRC {res['train'][c]['auprc']:.3f}   "
              f"test AUC {res['test'][c]['auc']:.3f} AUPRC {res['test'][c]['auprc']:.3f}")
    print("->", out)


PAIRS = [("ens_score_mean", "score"), ("ens_volume_mean", "volume"), ("ens_open_fraction", "score"),
         ("amp50_score", "score")]


def cmd_paired(a):
    """Test split: paired protein-bootstrap differences between single columns (ensemble minus static pocket
    score, no learning), and the shipped models on the single-chain subset used by Seq2Pocket (184 structures)."""
    recs = cb.load_split(a, "test")
    prop = cb.fit_propensity(cb.load_split(a, "train"))
    X = cb.physics(recs, prop)
    ys = [r["y"] for r in recs]
    col = {c: [x[:, j] for x in X] for j, c in enumerate(COLUMNS)}
    out = {"columns": {}}
    for new, old in PAIRS:
        d = cb.bootstrap(col[new], ys, n=a.boot, other=col[old])
        out["columns"][f"{new} - {old}"] = d
        print(f"{new} - {old}: {json.dumps(d)}", flush=True)
    ref = np.load(os.path.join(a.work, "test_shipped_v3_650M_scores.npz"), allow_pickle=True)  # our own file
    pos = {i: k for k, i in enumerate(ref["ids"])}
    sub = [k for k, r in enumerate(recs) if "-" not in r["id"][4:]]
    ys_sub = [ys[k] for k in sub]
    out["single_chain_subset"] = {"proteins": len(sub), "residues": int(sum(len(y) for y in ys_sub)),
                                  "positives": int(sum(y.sum() for y in ys_sub))}
    for name in ("hybrid", "physics"):
        s = [np.asarray(ref[name][pos[recs[k]["id"]]], dtype=float) for k in sub]
        out["single_chain_subset"][name] = dict(cb.metrics(s, ys_sub), **cb.bootstrap(s, ys_sub, n=a.boot))
        print(name, "single-chain subset", json.dumps(out["single_chain_subset"][name]), flush=True)
    json.dump(out, open(os.path.join(a.work, "control_paired.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["esm-only", "esm-strong", "columns", "paired"])
    ap.add_argument("--variant", default="pooled", choices=["pooled", "mlp"])
    ap.add_argument("--data", default=os.path.expanduser("~/crypsis_data/cryptobench"))
    ap.add_argument("--work", default=os.path.expanduser("~/crypsis_data/work"))
    ap.add_argument("--esm", default="650M", choices=["150M", "650M"])
    ap.add_argument("--pcs", type=int, default=256)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--physics", default="v3")
    a = ap.parse_args()
    {"esm-only": cmd_esm_only, "esm-strong": cmd_esm_strong, "columns": cmd_columns, "paired": cmd_paired}[a.command](a)


if __name__ == "__main__":
    main()

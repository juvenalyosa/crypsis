"""Export the trained scikit-learn models to plain arrays (no pickle in the package).

    python scripts/export_models.py final_models.pkl src/crypsis/models

The pickle holds the residue propensity table, the PCA of the ESM-2 embeddings and the two
HistGradientBoostingClassifier models (physics, hybrid) fitted on the CryptoBench training folds.
Writes physics_v2.npz, hybrid_v2.npz (trees + PCA) and propensity.json, then checks that crypsis'
own tree evaluator reproduces scikit-learn's probabilities.
"""
import json
import os
import pickle
import sys

import numpy as np


def trees(model):
    """Concatenated nodes of all trees: feature, threshold, left, right, is_leaf, value, missing_left."""
    if model.is_categorical_ is not None and np.any(model.is_categorical_):
        raise ValueError("categorical splits are not supported")
    cols = {k: [] for k in ("feature", "threshold", "left", "right", "leaf", "value", "missing_left")}
    offsets = [0]
    for (pred,) in model._predictors:
        n = pred.nodes
        cols["feature"].append(n["feature_idx"].astype(np.int32))
        cols["threshold"].append(n["num_threshold"].astype(np.float64))
        cols["left"].append(n["left"].astype(np.int32))
        cols["right"].append(n["right"].astype(np.int32))
        cols["leaf"].append(n["is_leaf"].astype(np.bool_))
        cols["value"].append(n["value"].astype(np.float64))
        cols["missing_left"].append(n["missing_go_to_left"].astype(np.bool_))
        offsets.append(offsets[-1] + len(n))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["offsets"] = np.array(offsets, dtype=np.int64)
    out["baseline"] = np.array(float(np.ravel(model._baseline_prediction)[0]))
    out["n_features"] = np.array(model.n_features_in_)
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = pickle.load(open(src, "rb"))
    import sklearn

    meta = {"sklearn": sklearn.__version__, "trained_on": "CryptoBench training folds 0-3 (Skrhak et al., 2025)"}
    np.savez_compressed(os.path.join(dst, "physics_v2.npz"), **trees(m["phys"]), meta=json.dumps(meta))
    np.savez_compressed(os.path.join(dst, "hybrid_v2.npz"), **trees(m["hyb"]), pca_mean=m["pca"].mean_,
                        pca_components=m["pca"].components_, meta=json.dumps(dict(meta, esm="esm2_t30_150M_UR50D")))
    with open(os.path.join(dst, "propensity.json"), "w") as fh:
        json.dump(m["prop"], fh, indent=1, sort_keys=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from crypsis.gbm import TreeEnsemble

    rng = np.random.default_rng(0)
    for key, name in (("phys", "physics_v2.npz"), ("hyb", "hybrid_v2.npz")):
        X = rng.random((5000, m[key].n_features_in_))
        if key == "hyb":
            X[:, 22:] = rng.normal(scale=3.0, size=(5000, 128))
        ref = m[key].predict_proba(X)[:, 1]
        got = TreeEnsemble.load(os.path.join(dst, name)).predict_proba(X)
        print(name, "max |difference| vs scikit-learn:", float(np.abs(ref - got).max()))
        assert np.abs(ref - got).max() < 1e-9


if __name__ == "__main__":
    main()

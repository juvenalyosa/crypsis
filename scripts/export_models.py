"""Export the trained scikit-learn models to plain arrays (no pickle in the package).

    python scripts/export_models.py work/final_models_v3_650M_256.pkl src/crypsis/models 650M
    python scripts/export_models.py work/final_models_v3_150M_128.pkl src/crypsis/models 150M --hybrid-only

The pickle (scripts/cryptobench.py train) holds the residue propensity table, the PCA of the ESM-2 embeddings and
the two HistGradientBoostingClassifier models (physics, hybrid) fitted on the CryptoBench training folds.  Writes
the physics model, the hybrid of that ESM-2 size (trees + PCA) and propensity.json under the names of
pipeline.MODEL_FILES, then checks that crypsis' own tree evaluator reproduces scikit-learn's probabilities.  The
150M pickle has the same physics model and propensities as the 650M one (same data and seed), hence --hybrid-only.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


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
    src, dst, esm = sys.argv[1], sys.argv[2], sys.argv[3]
    hybrid_only = "--hybrid-only" in sys.argv[4:]
    import sklearn

    from crypsis.gbm import TreeEnsemble
    from crypsis.pipeline import ESM_MODELS, MODEL_FILES

    m = pickle.load(open(src, "rb"))
    n_phys = m["phys"].n_features_in_
    meta = {"sklearn": sklearn.__version__, "trained_on": "CryptoBench training folds 0-3 (Skrhak et al., 2025)",
            "physics_columns": int(n_phys)}
    written = []
    if not hybrid_only:
        np.savez_compressed(os.path.join(dst, MODEL_FILES["physics"]), **trees(m["phys"]), meta=json.dumps(meta))
        with open(os.path.join(dst, "propensity.json"), "w") as fh:
            json.dump(m["prop"], fh, indent=1, sort_keys=True)
        written.append(("phys", MODEL_FILES["physics"]))
    name, layer, dim = ESM_MODELS[esm]
    if m["pca"].components_.shape[1] != dim:
        raise SystemExit(f"the pickle's PCA is for {m['pca'].components_.shape[1]}-dim embeddings, not ESM-2 {esm}")
    hyb_file = MODEL_FILES[f"hybrid_{esm}"]
    np.savez_compressed(os.path.join(dst, hyb_file), **trees(m["hyb"]), pca_mean=m["pca"].mean_,
                        pca_components=m["pca"].components_,
                        meta=json.dumps(dict(meta, esm=name, esm_layer=layer, pcs=int(m["pca"].n_components_))))
    written.append(("hyb", hyb_file))

    rng = np.random.default_rng(0)
    for key, fname in written:
        X = rng.random((5000, m[key].n_features_in_))
        if key == "hyb":
            X[:, n_phys:] = rng.normal(scale=3.0, size=(5000, m[key].n_features_in_ - n_phys))
        ref = m[key].predict_proba(X)[:, 1]
        got = TreeEnsemble.load(os.path.join(dst, fname)).predict_proba(X)
        print(fname, "max |difference| vs scikit-learn:", float(np.abs(ref - got).max()))
        assert np.abs(ref - got).max() < 1e-9


if __name__ == "__main__":
    main()

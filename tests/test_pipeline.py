import json
import os

import numpy as np
import pytest

import crypsis
from crypsis import cli, descriptors, pipeline
from crypsis.gbm import TreeEnsemble

DATA = os.path.join(os.path.dirname(__file__), "data")
CLEFT = ["A_35", "A_52", "A_62", "A_63", "A_101", "A_108"]   # lysozyme substrate cleft


def test_tree_evaluator_on_a_hand_built_ensemble():
    # one stump (x0 <= 0.5 -> -1, else +2; NaN goes right) and one leaf (+0.5), baseline 0.25
    t = TreeEnsemble({"feature": np.array([0, 0, 0, 0]), "threshold": np.array([0.5, 0, 0, 0]),
                      "left": np.array([1, 0, 0, 0]), "right": np.array([2, 0, 0, 0]),
                      "leaf": np.array([False, True, True, True]), "value": np.array([0, -1.0, 2.0, 0.5]),
                      "missing_left": np.array([False, False, False, False]), "offsets": np.array([0, 3, 4]),
                      "baseline": np.array(0.25), "n_features": np.array(1)})
    raw = t.decision_function(np.array([[0.1], [0.9], [np.nan]]))
    assert np.allclose(raw, [0.25 - 1 + 0.5, 0.25 + 2 + 0.5, 0.25 + 2 + 0.5])


def test_shipped_models_load():
    n = 22 + len(descriptors.DESCRIPTORS) + len(descriptors.ENSEMBLE)
    assert pipeline._load("physics").n_features == n
    for esm, pcs in (("650M", 256), ("150M", 128)):
        hyb = pipeline._load(f"hybrid_{esm}")
        assert hyb.n_features == n + pcs
        assert hyb.extra["pca_components"].shape == (pcs, pipeline.ESM_MODELS[esm][2])
        assert hyb.meta["esm"] == pipeline.ESM_MODELS[esm][0]
    prop = pipeline._load("propensity")
    assert len(prop) == 20 and prop["TRP"] > prop["GLU"]


def test_have_esm_does_not_import_torch():
    """Importing torch before the normal modes can flip LAPACK eigenvector signs, so the check must only look."""
    import subprocess
    import sys

    code = "import sys; from crypsis import pipeline; pipeline.have_esm(); print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_smoothing_matrix_rows_sum_to_one():
    ca = np.random.default_rng(0).normal(scale=6.0, size=(50, 3))
    W = pipeline.smoothing_matrix(ca).toarray()
    has = (W > 0).any(axis=1)
    assert np.allclose(W[has].sum(axis=1), 1.0) and np.all(np.diag(W) == 0)


@pytest.mark.parametrize("model", ["physics", "v1"])
def test_lysozyme_cleft_is_found(model):
    res = crypsis.predict(os.path.join(DATA, "2LZT.pdb"), model=model)
    prob = dict(zip(res["residues"], res["probability"]))
    assert res["model"] == model and all(0.0 <= p <= 1.0 for p in prob.values())
    others = [k for k in prob if k not in CLEFT]
    assert np.mean([prob[k] for k in CLEFT]) > np.percentile([prob[k] for k in others], 75)
    assert set(CLEFT) & set(sum((s["residues"] for s in res["sites"][:3]), []))
    for s in res["sites"]:
        assert s["kind"] in ("already open", "grows with motion", "forms with motion")
        assert 16.0 <= min(s["docking_box"]["size"]) and max(s["docking_box"]["size"]) <= 26.0


def test_hybrid_accepts_precomputed_embeddings():
    path = os.path.join(DATA, "2LZT.pdb")
    ch = crypsis.read_chain(path)
    assert pipeline.sequences(ch)[0].startswith("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTP")
    for esm, dim in (("650M", 1280), ("150M", 640)):
        emb = np.zeros((ch.n_res, dim), dtype=np.float32)
        res = crypsis.predict(ch, model="hybrid", embeddings=emb, esm=esm)
        assert res["model"] == "hybrid" and res["esm"] == esm and len(res["probability"]) == ch.n_res
        with pytest.raises(ValueError):
            crypsis.predict(ch, model="hybrid", embeddings=emb[:10], esm=esm)
    with pytest.raises(ValueError):                       # 150M-sized embeddings for the 650M model
        crypsis.predict(ch, model="hybrid", embeddings=np.zeros((ch.n_res, 640), dtype=np.float32))


def test_cli_writes_outputs(tmp_path, capsys):
    assert cli.main(["predict", os.path.join(DATA, "2LZT.pdb"), "--model", "physics", "-o", str(tmp_path), "-q"]) == 0
    for suffix in ("_residues.csv", "_sites.json", "_crypsis.pdb", "_crypsis.pml"):
        assert (tmp_path / f"2LZT{suffix}").exists()
    sites = json.loads((tmp_path / "2LZT_sites.json").read_text())
    assert sites["model"] == "physics" and sites["sites"][0]["rank"] == 1
    assert "physics model" in capsys.readouterr().out


def test_descriptors_of_lysozyme():
    from crypsis import core

    ch = crypsis.read_chain(os.path.join(DATA, "2LZT.pdb"))
    d = descriptors.descriptors(ch)
    assert d.shape == (ch.n_res, len(descriptors.DESCRIPTORS)) and np.isfinite(d).all()
    col = {n: d[:, j] for j, n in enumerate(descriptors.DESCRIPTORS)}
    assert 0.0 <= col["rsasa"].min() and np.median(col["rsasa"]) < 0.6 and col["rsasa"].max() < 1.6
    assert abs(col["gnm_slow"].mean() - 1.0) < 1e-6 and abs(col["gnm_fast"].mean() - 1.0) < 1e-6
    assert col["end_distance"][0] == 0 and col["end_distance"][-1] == 0 and col["end_distance"].max() == 20
    helix = np.median(col["ca_angle"][4:14])                  # lysozyme helix A: residues 5-14
    assert 85.0 < helix < 100.0
    # cutting side chains back can only open pockets: the gains are never negative
    for nm in ("trim_cb_gain", "trim_cg_gain"):
        assert col[nm].min() >= 0.0
    p = dict(core.load_model()["params"])
    s0 = core.pocket_grid(ch.xyz, ch.atom_res, ch.n_res, p)["residue_score"]
    e = descriptors.ensemble_descriptors(ch, s0, params=p)
    assert e.shape == (ch.n_res, len(descriptors.ENSEMBLE)) and np.isfinite(e).all()
    anm = core.anm_modes(ch.xyz[ch.ca_index], p["anm_cutoff_A"], p["n_modes"])
    assert np.allclose(descriptors.ensemble_descriptors(ch, s0, params=p, anm=anm), e)


def test_atom_sasa_of_an_isolated_atom():
    a = descriptors.atom_sasa(np.zeros((1, 3)), ["C"])
    assert abs(a[0] - 4 * np.pi * (1.7 + 1.4) ** 2) < 1e-6

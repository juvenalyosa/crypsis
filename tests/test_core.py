import os

import numpy as np
import pytest

from crypsis import core as cp

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def lysozyme():
    return cp.read_chain(os.path.join(DATA, "2LZT.pdb"))


def test_read_chain_keeps_protein_heavy_atoms(lysozyme):
    assert lysozyme.n_res == 129 and len(lysozyme.ca_index) == 129
    assert lysozyme.res_keys[0] == "A_1" and lysozyme.res_names[0] == "LYS"


def test_sparse_hessian_matches_dense(lysozyme):
    ca = lysozyme.xyz[lysozyme.ca_index][:60]
    n = len(ca)
    d = ca[:, None, :] - ca[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    contact = (r2 < 15.0 ** 2) & ~np.eye(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        blk = -d[:, :, :, None] * d[:, :, None, :] / r2[:, :, None, None]
    blk[~contact] = 0.0
    dense = np.zeros((n, 3, n, 3))
    dense[:] = blk.transpose(0, 2, 1, 3)
    for i in range(n):
        dense[i, :, i, :] = -blk[i].sum(axis=0)
    assert np.abs(cp.anm_hessian(ca).toarray() - dense.reshape(3 * n, 3 * n)).max() < 1e-10
    vals, modes = cp.anm_modes(ca, n_modes=10)
    assert vals.shape == (10,) and modes.shape == (10, n, 3) and vals[0] > 1e-6


def test_mode_signs_do_not_depend_on_lapack(lysozyme):
    vals, modes = cp.anm_modes(lysozyme.xyz[lysozyme.ca_index])
    flat = modes.reshape(len(vals), -1)
    assert ((flat ** 3).sum(axis=1) > 0).all()
    flipped = flat * np.where(np.arange(len(vals)) % 3 == 0, -1.0, 1.0)[:, None]
    assert np.array_equal(cp.canonical_signs(flipped), flat)


def test_conformers_have_the_requested_rmsd(lysozyme):
    vals, modes = cp.anm_modes(lysozyme.xyz[lysozyme.ca_index])
    for x in cp.conformers(lysozyme, vals, modes, n_conf=3, rmsd=2.0):
        d = x[lysozyme.ca_index] - lysozyme.xyz[lysozyme.ca_index]
        assert abs(np.sqrt((d ** 2).sum(axis=1).mean()) - 2.0) < 1e-6


@pytest.mark.skipif(not cp.have_extension(), reason="C++ extension not built")
def test_cpp_and_numpy_agree(lysozyme):
    a = cp.pocket_grid(lysozyme.xyz, lysozyme.atom_res, lysozyme.n_res)
    b = cp.pocket_grid(lysozyme.xyz, lysozyme.atom_res, lysozyme.n_res, force_numpy=True)
    assert len(a["points"]) == len(b["points"]) > 50
    assert np.allclose(np.sort(a["cluster_volume"]), np.sort(b["cluster_volume"]))
    assert np.abs(a["residue_volume"] - b["residue_volume"]).max() == 0.0
    assert np.abs(a["residue_score"] - b["residue_score"]).max() < 1e-3


def test_kept_cofactor_fills_its_pocket_and_is_not_scored():
    """keep_hetero: a bound cofactor counts as occupied space (no artificial empty pocket) but is not a residue."""
    path = os.path.join(DATA, "1ATP.pdb")
    a = cp.read_chain(path, chains=["E"])
    b = cp.read_chain(path, chains=["E"], keep_hetero={"ATP"})
    assert a.n_res == b.n_res and (b.atom_res < 0).sum() > 20 and (b.motion_res >= 0).all()
    ga = cp.pocket_grid(a.xyz, a.atom_res, a.n_res)
    gb = cp.pocket_grid(b.xyz, b.atom_res, b.n_res)
    assert gb["cluster_volume"].sum() < ga["cluster_volume"].sum()        # the ATP pocket is no longer empty
    gn = cp.pocket_grid(b.xyz, b.atom_res, b.n_res, force_numpy=True)
    assert np.abs(gn["residue_volume"] - gb["residue_volume"]).max() == 0.0
    auto = cp.read_chain(path, chains=["E"], keep_hetero="auto")
    assert (auto.atom_res < 0).sum() >= (b.atom_res < 0).sum()


def test_v1_model_file_is_consistent():
    model = cp.load_model()
    assert set(model["features"]) <= set(cp.FEATURES)
    assert len(model["coef"]) == len(model["features"]) == len(model["mean"])
    assert "CryptoBench" in model["trained_on"]

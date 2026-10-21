"""Residue probabilities and ranked cryptic sites for one structure.

Three models, all fitted on the CryptoBench training folds only:

  hybrid    physics features + ESM-2 (150M) embeddings, gradient-boosted trees   (test AUC 0.881, AUPRC 0.372)
  physics   physics features only, gradient-boosted trees                         (test AUC 0.846, AUPRC 0.280)
  v1        logistic model on three ensemble features                             (test AUC 0.807, AUPRC 0.179)

The physics features of the boosted models are the 14 columns of core.FEATURES, rank-normalized within the
protein, three of them smoothed over the residue graph (Calpha within 8 A, Gaussian sigma 4 A), the residue's
amino-acid log-odds propensity, and a second, shallower pocket pass (burial >= 0.55) in the structure and the
ensemble.  model="auto" uses the hybrid when PyTorch and fair-esm are installed and the physics model otherwise.
"""

import json
import os

import numpy as np

from . import core

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LOG_FEATURES = {"score", "volume", "ens_score_mean", "ens_volume_mean", "ens_gain", "patch_score", "patch_ens_score",
                "patch_ens_gain"}
SMOOTHED = ["ens_volume_mean", "ens_open_fraction", "score"]
SHALLOW_MIN_FRAC = 0.55
ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
              "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
              "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y"}
_CACHE = {}


# ------------------------------------------------------------------------------------------------ models

def _load(name):
    if name not in _CACHE:
        from .gbm import TreeEnsemble

        if name == "propensity":
            with open(os.path.join(MODELS_DIR, "propensity.json")) as fh:
                _CACHE[name] = json.load(fh)
        else:
            _CACHE[name] = TreeEnsemble.load(os.path.join(MODELS_DIR, f"{name}_v2.npz"))
    return _CACHE[name]


def have_esm():
    """Whether PyTorch and fair-esm are installed (found, not imported: importing PyTorch before the normal modes are
    computed can flip the arbitrary sign of some LAPACK eigenvectors, which changes the sampled ensemble)."""
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("esm") is not None


# ------------------------------------------------------------------------------------------------ features

def _rank(v):
    from scipy.stats import rankdata

    return rankdata(v) / len(v)


def smoothing_matrix(ca, radius=8.0, sigma=4.0):
    """Row-normalized Gaussian weights between residues whose Calpha atoms are within ``radius`` (sparse)."""
    from scipy.sparse import csr_matrix, diags
    from scipy.spatial import cKDTree

    d = cKDTree(ca).sparse_distance_matrix(cKDTree(ca), radius, output_type="coo_matrix")
    keep = d.row != d.col
    w = csr_matrix((np.exp(-d.data[keep] ** 2 / (2 * sigma ** 2)), (d.row[keep], d.col[keep])), shape=d.shape)
    return diags(1.0 / (np.asarray(w.sum(axis=1)).ravel() + 1e-9)) @ w


def shallow_pass(chain, extra):
    """Pocket volume per residue with a shallower burial cut, in the structure and across the same ensemble."""
    p = dict(extra["params"], min_frac=SHALLOW_MIN_FRAC)
    vals, modes = extra["anm"]
    s0 = core.pocket_grid(chain.xyz, chain.atom_res, chain.n_res, p)
    ens = np.array([core.pocket_grid(x, chain.atom_res, chain.n_res, p)["residue_volume"]
                    for x in core.conformers(chain, vals, modes, p["n_conformers"], p["rmsd_A"], p["seed"])])
    return np.column_stack([s0["residue_volume"], s0["residue_score"], ens.mean(0), (ens > 0).mean(0)])


def physics_matrix(chain, feats, shallow):
    """The 22 columns of the boosted physics model."""
    F = core.FEATURES
    X = np.array(feats, dtype=float)
    for j, nm in enumerate(F):
        if nm in LOG_FEATURES:
            X[:, j] = np.log1p(X[:, j])
    X = np.column_stack([_rank(X[:, j]) for j in range(X.shape[1])])
    smooth = smoothing_matrix(chain.xyz[chain.ca_index]) @ X[:, [F.index(n) for n in SMOOTHED]]
    prop = _load("propensity")
    pr = np.array([[prop.get(str(n), 0.0)] for n in chain.res_names])
    sh = np.column_stack([_rank(np.log1p(shallow[:, j])) for j in range(shallow.shape[1])])
    return np.column_stack([X, smooth, pr, sh])


def sequences(chain):
    """One-letter sequence of each chain, in the residue order of the Chain (unknown residues are X)."""
    out, cur, seq = [], None, []
    for key, name in zip(chain.res_keys, chain.res_names):
        cid = key.split("_")[0]
        if cid != cur and seq:
            out.append("".join(seq))
            seq = []
        cur = cid
        seq.append(ONE_LETTER.get(name, "X"))
    if seq:
        out.append("".join(seq))
    return out


def esm_embeddings(seqs, device="cpu", chunk=1000):
    """ESM-2 150M (layer 30) per-residue embeddings of the sequences, stacked [n_residues, 640]."""
    import esm
    import torch

    if "esm" not in _CACHE:
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        model.eval()
        _CACHE["esm"] = (model, alphabet.get_batch_converter())
    model, convert = _CACHE["esm"]
    model = model.to(device)
    out = []
    for seq in seqs:
        e = np.zeros((len(seq), 640), dtype=np.float32)
        for s in range(0, len(seq), chunk):
            part = seq[s:s + chunk]
            _, _, toks = convert([("x", part)])
            with torch.no_grad():
                rep = model(toks.to(device), repr_layers=[30])["representations"][30]
            e[s:s + len(part)] = rep[0, 1:len(part) + 1].cpu().numpy()
        out.append(e)
    return np.vstack(out)


# ------------------------------------------------------------------------------------------------ predict

def predict(structure, chains=None, keep_cofactors=None, model="auto", embeddings=None, device="cpu",
            params=None, progress=None):
    """Cryptic-site prediction for a PDB/mmCIF file (or a core.Chain).

    chains: chain IDs to analyse (default all protein chains).  keep_cofactors: residue names of bound
    cofactors to keep as occupied space (e.g. {"GDP", "MG"}), or "auto" for hetero groups of >= 8 heavy atoms
    that are not crystallization additives; removing a tightly bound cofactor leaves an artificial empty pocket.
    model: "auto", "hybrid", "physics" or "v1".  embeddings: precomputed ESM-2 150M embeddings [n_res, 640].

    Returns a dict: residues, residue_names, probability, sites (best first, each with its residues, kind,
    pocket volumes and a docking box), model, and the per-residue numbers behind them.

    Reproducibility: the sign of each normal mode is arbitrary and the LAPACK build decides it; PyTorch loads its own
    libraries, so in a session where torch was imported before the first prediction a few modes can flip sign and
    the ensemble (and the probabilities) differ slightly from a fresh process.  The published numbers come from fresh
    processes; ESM-2 is imported here only after the physics is done."""
    chain = structure if isinstance(structure, core.Chain) else core.read_chain(
        structure, chains=list(chains) if chains else None, keep_hetero=keep_cofactors)
    if model == "auto":
        model = "hybrid" if (embeddings is not None or have_esm()) else "physics"
    if model not in ("hybrid", "physics", "v1"):
        raise ValueError(f"unknown model {model!r}")
    if model == "hybrid" and embeddings is None and not have_esm():
        raise ImportError("the hybrid model needs PyTorch and fair-esm: pip install 'crypsis[esm]'")
    v1 = core.load_model()
    p = dict(v1.get("params") or {}, **(params or {}))
    feats, extra = core.features(chain, p, progress=progress)
    if model == "v1":
        prob, threshold = core.apply_model(v1, feats), core.THRESHOLD
    else:
        Xp = physics_matrix(chain, feats, shallow_pass(chain, extra))
        if model == "physics":
            prob = _load("physics").predict_proba(Xp)
        else:
            emb = np.asarray(embeddings if embeddings is not None else esm_embeddings(sequences(chain), device),
                             dtype=np.float32)
            if emb.shape != (chain.n_res, 640):
                raise ValueError(f"embeddings must be [{chain.n_res}, 640], got {emb.shape}")
            hyb = _load("hybrid")
            pcs = (emb - hyb.extra["pca_mean"]) @ hyb.extra["pca_components"].T
            prob = hyb.predict_proba(np.column_stack([Xp, pcs]))
        threshold = None
    st = extra["static"]
    big = np.isin(st["label"], np.flatnonzero(st["cluster_volume"] >= extra["params"]["min_volume_A3"]))
    sites = core.sites_from_scores(chain, prob, st["residue_volume"], extra["ens_volume_mean"],
                                   extra["open_fraction"], threshold=threshold, static_points=st["points"][big],
                                   frames=extra["frames"], grid_A=extra["params"]["grid_A"])
    return {"model": model, "residues": chain.res_keys, "residue_names": chain.res_names, "probability": prob,
            "open_fraction": extra["open_fraction"], "static_score": st["residue_score"],
            "pocket_points": st["points"][big], "sites": sites, "extension": core.have_extension(),
            "chain": chain}

#!/usr/bin/env python3
"""Rebuild Crypsis' models and benchmark from the public CryptoBench data (Skrhak et al., Bioinformatics 2025).

Data (https://osf.io/pz4a9/): cryptobench-dataset/{dataset.json,folds.json} and the apo mmCIF files, gzipped,
in <data>/cif/<apo>.cif.gz.  Steps, each cached on disk so later steps can be rerun alone:

    python scripts/cryptobench.py features    --data ~/crypsis_data/cryptobench --work ~/crypsis_data/work
    python scripts/cryptobench.py descriptors --data ... --work ...   # descriptors.DESCRIPTORS
    python scripts/cryptobench.py ensemble    --data ... --work ...   # descriptors.ENSEMBLE
    python scripts/cryptobench.py esm --esm 650M --work ...            # ESM-2 embeddings (separate process)
    python scripts/cryptobench.py cv    --esm 650M --pcs 256 --work ... # 4-fold cross-validation, train folds only
    python scripts/cryptobench.py train --esm 650M --pcs 256 --work ... # -> work/final_models_v3_650M_256.pkl
    python scripts/cryptobench.py test  --esm 650M --pcs 256 --work ... # test split, scored once
    python scripts/export_models.py work/final_models_v3_650M_256.pkl src/crypsis/models 650M

--physics v2 rebuilds the 22-column physics inputs of Crypsis 0.1 (no descriptors, no extra ensembles).

Protocol (CryptoBench's own): for each apo chain every observed residue is scored; a residue is positive when it
is in the apo_pocket_selection of any holo structure paired with that chain.  AUC and AUPRC are computed over all
residues of the split pooled together; confidence intervals resample proteins (bootstrap, 2000 draws).  Sites:
a predicted site (core.sites_from_scores) finds a true site when
it shares at least half of its own residues, or a third of the true site's, with it (and at least 3 residues); each
holo structure's pocket is a true site (101 of the 222 test proteins have more than one), and the result is the
share of proteins whose first site, or one of the first three, finds one of them.

The physics features are computed in their own processes: importing PyTorch before the normal modes can flip the
arbitrary sign of LAPACK eigenvectors (see pipeline.predict), so ESM-2 runs in a separate command.
"""

import argparse
import concurrent.futures as cf
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

AMINO20 = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
ESM_MODELS = {"150M": ("esm2_t30_150M_UR50D", 30, 640), "650M": ("esm2_t33_650M_UR50D", 33, 1280),
              "3B": ("esm2_t36_3B_UR50D", 36, 2560)}


# ------------------------------------------------------------------------------------------------ data

def _dataset(data):
    if _DATA.get("path") != data:
        with open(os.path.join(data, "cryptobench-dataset", "dataset.json")) as fh:
            _DATA.update(path=data, dataset=json.load(fh))
    return _DATA["dataset"]


def _pocket(h, chains):
    """The residues of one holo record's apo pocket on the scored entity's chains (None when there are none)."""
    return {k for k in h["apo_pocket_selection"] if k.split("_")[0] in chains} or None


def pockets_of(data, apo):
    """The distinct cryptic pockets of a scored apo entity (residue-key sets, one per holo structure's pocket), for
    evaluation: every holo record counts with its pocket residues on the entity's chains."""
    holos = _dataset(data)[apo]
    chains = holos[0]["apo_chain"].split("-")
    out = []
    for h in holos:
        p = _pocket(h, set(chains))
        if p and p not in out:
            out.append(p)
    return out


_DATA = {}


def entries(data, split):
    """[(apo, chain, positive residue keys)] of a split: 'test', 'train-0'..'train-3' or 'train' (all four).

    Labels.  Test (evaluation) labels: a residue of the scored entity (the first record's apo_chain, e.g. "C-O") is
    positive when it is in the apo pocket of any holo record of that apo structure.  Training labels keep the rule
    the shipped models were trained with: only records whose apo_chain string equals the first record's count, which
    misses pocket residues of records written with another chain string (258 residues in 16 training entries)."""
    d = _dataset(data)
    with open(os.path.join(data, "cryptobench-dataset", "folds.json")) as fh:
        folds = json.load(fh)
    if split == "all":
        ids = sum(folds.values(), [])
    elif split == "train":
        ids = sum((folds[f"train-{i}"] for i in range(4)), [])
    else:
        ids = folds[split]
    out = []
    for apo in ids:
        holos = d[apo]
        chain = holos[0]["apo_chain"]
        chains = set(chain.split("-"))
        pos = set()
        for h in holos:
            if split == "test":
                pos |= _pocket(h, chains) or set()
            elif h["apo_chain"] == chain:
                pos |= set(h["apo_pocket_selection"])
        out.append((apo, chain, sorted(pos)))
    return out


def fold_of(data):
    folds = json.load(open(os.path.join(data, "cryptobench-dataset", "folds.json")))
    return {apo: name for name, ids in folds.items() for apo in ids}


class CaChain:
    """What pipeline.physics_matrix needs from a Chain: Calpha coordinates and residue names."""

    def __init__(self, ca, names):
        self.xyz, self.ca_index, self.res_names = np.asarray(ca), np.arange(len(ca)), list(names)


# ------------------------------------------------------------------------------------------------ features

def _one_features(job):
    apo, chain_id, pos, data, out_dir = job
    from crypsis import core, pipeline

    path = os.path.join(out_dir, f"{apo}{chain_id}.npz")
    if os.path.exists(path):
        return apo, "cached", 0.0
    t0 = time.time()
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
        if ch.n_res > 1500:
            return apo, "too large", 0.0
        p = dict(core.load_model().get("params") or {})
        feats, extra = core.features(ch, p)
        shallow = pipeline.shallow_pass(ch, extra)
        y = np.array([1 if k in set(pos) else 0 for k in ch.res_keys], dtype=np.int8)
        np.savez_compressed(path, feats=feats, shallow=shallow, y=y, keys=np.array(ch.res_keys),
                            names=np.array(ch.res_names), ca=ch.xyz[ch.ca_index],
                            seqs=np.array(pipeline.sequences(ch)))
        return apo, "ok", time.time() - t0
    except Exception as exc:
        return apo, f"error {type(exc).__name__}: {exc}", 0.0


def cmd_features(a):
    out_dir = os.path.join(a.work, "features")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(apo, ch, pos, a.data, out_dir) for apo, ch, pos in entries(a.data, a.split)]
    if a.limit:
        jobs = jobs[:a.limit]
    t0, status = time.time(), {}
    # one process per structure: a fresh interpreter state for every set of normal modes
    with cf.ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as ex:
        for k, (apo, st, sec) in enumerate(ex.map(_one_features, jobs), 1):
            status[apo] = st
            if st not in ("ok", "cached"):
                print(f"  {apo}: {st}", flush=True)
            if k % 50 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
    json.dump(status, open(os.path.join(a.work, f"features_status_{a.split}.json"), "w"), indent=1)


# The physics inputs added in Crypsis 0.2, each computed by its own command into <work>/<kind>/<id>.npz (key "x").
# The ensemble recomputes the normal modes, so it runs one process per structure, as `features`.
EXTRAS = ["descriptors", "ensemble"]


def _one_extra(job):
    kind, apo, chain_id, data, work = job
    from crypsis import core, descriptors

    ident = f"{apo}{chain_id}"
    out = os.path.join(work, kind, f"{ident}.npz")
    feat = os.path.join(work, "features", f"{ident}.npz")
    if os.path.exists(out) or not os.path.exists(feat):
        return apo, "skipped", 0.0
    t0 = time.time()
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
        z = np.load(feat)
        assert [str(k) for k in z["keys"]] == ch.res_keys, "residues differ from the features file"
        p = dict(core.load_model().get("params") or {})
        s0 = z["feats"][:, core.FEATURES.index("score")]
        if kind == "descriptors":
            x = descriptors.descriptors(ch, static_score=s0, params=p)
        else:
            x = descriptors.ensemble_descriptors(ch, s0, params=p)
        np.savez_compressed(out, x=np.asarray(x, dtype=np.float32))
        return apo, "ok", time.time() - t0
    except Exception as exc:
        return apo, f"error {type(exc).__name__}: {exc}", 0.0


def cmd_extra(a):
    kind = a.command
    os.makedirs(os.path.join(a.work, kind), exist_ok=True)
    jobs = [(kind, apo, ch, a.data, a.work) for apo, ch, _pos in entries(a.data, a.split)]
    if a.limit:
        jobs = jobs[:a.limit]
    t0, status = time.time(), {}
    fresh = {"max_tasks_per_child": 1} if kind == "ensemble" else {}
    with cf.ProcessPoolExecutor(a.workers, **fresh) as ex:
        for k, (apo, st, sec) in enumerate(ex.map(_one_extra, jobs), 1):
            status[apo] = st
            if st.startswith("error"):
                print(f"  {apo}: {st}", flush=True)
            if k % 50 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
    json.dump(status, open(os.path.join(a.work, f"{kind}_status_{a.split}.json"), "w"), indent=1)


def cmd_esm(a):
    import torch

    from crypsis import pipeline

    name, layer, dim = ESM_MODELS[a.esm]
    out_dir = os.path.join(a.work, f"esm_{a.esm}")
    os.makedirs(out_dir, exist_ok=True)
    import esm

    model, alphabet = getattr(esm.pretrained, name)()
    model.eval().to(a.device)
    convert = alphabet.get_batch_converter()
    feat_dir = os.path.join(a.work, "features")
    files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".npz"))
    t0 = time.time()
    for k, f in enumerate(files, 1):
        path = os.path.join(out_dir, f)
        if os.path.exists(path):
            continue
        seqs = [str(s) for s in np.load(os.path.join(feat_dir, f))["seqs"]]
        out = []
        for seq in seqs:
            e = np.zeros((len(seq), dim), dtype=np.float32)
            for s in range(0, len(seq), 1000):
                part = seq[s:s + 1000]
                _, _, toks = convert([("x", part)])
                with torch.no_grad():
                    rep = model(toks.to(a.device), repr_layers=[layer])["representations"][layer]
                e[s:s + len(part)] = rep[0, 1:len(part) + 1].float().cpu().numpy()
            out.append(e)
        np.savez_compressed(path, emb=np.vstack(out).astype(np.float16))
        if k % 50 == 0 or k == len(files):
            print(f"[{k}/{len(files)}] {time.time() - t0:.0f} s", flush=True)
    del pipeline


# ------------------------------------------------------------------------------------------------ matrices

def load_split(a, split, esm=None):
    """Per-protein records of a split: physics inputs, labels and (optionally) embeddings."""
    recs = []
    for apo, ch, pos in entries(a.data, split):
        f = os.path.join(a.work, "features", f"{apo}{ch}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        pos = set(pos)
        y = np.array([1 if str(k) in pos else 0 for k in z["keys"]], dtype=int)   # labels of entries(), see there
        r = {"id": f"{apo}{ch}", "apo": apo, "feats": z["feats"], "shallow": z["shallow"], "y": y,
             "names": [str(n) for n in z["names"]], "ca": z["ca"], "keys": [str(k) for k in z["keys"]]}
        if esm:
            r["emb"] = np.load(os.path.join(a.work, f"esm_{esm}", f"{apo}{ch}.npz"))["emb"].astype(np.float32)
        if a.physics == "v3":
            for kind in EXTRAS:
                r[kind] = np.load(os.path.join(a.work, kind, f"{apo}{ch}.npz"))["x"].astype(float)
        recs.append(r)
    return recs


def fit_propensity(recs):
    """Log-odds of each amino acid among cryptic-site residues against all residues (training records only)."""
    names = np.concatenate([r["names"] for r in recs])
    y = np.concatenate([r["y"] for r in recs])
    out = {}
    for aa in AMINO20:
        m = names == aa
        out[aa] = float(np.log(((y[m].sum() + 1) / (y.sum() + 20)) / ((m.sum() + 1) / (len(y) + 20))))
    return out


def physics(recs, prop):
    """The physics inputs of each record, through the package's own pipeline.physics_matrix (v3) or base_matrix
    (v2, Crypsis 0.1)."""
    from crypsis import pipeline

    out = []
    for r in recs:
        ch = CaChain(r["ca"], r["names"])
        if "descriptors" in r:
            out.append(pipeline.physics_matrix(ch, r["feats"], r["shallow"], r["descriptors"], r["ensemble"],
                                               propensity=prop))
        else:
            out.append(pipeline.base_matrix(ch, r["feats"], r["shallow"], propensity=prop))
    return out


def hgb(seed=0, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier

    p = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
             early_stopping=False, random_state=seed)
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


def fit_models(recs, n_pcs=128, esm=True, hp=None):
    """propensity -> physics model; PCA of the embeddings -> hybrid model (the same recipe as the shipped v2)."""
    from sklearn.decomposition import PCA

    prop = fit_propensity(recs)
    Xp = np.vstack(physics(recs, prop))
    y = np.concatenate([r["y"] for r in recs])
    out = {"prop": prop, "phys": hgb(**(hp or {})).fit(Xp, y)}
    if esm:
        E = np.vstack([r["emb"] for r in recs])
        pca = PCA(n_pcs, random_state=0).fit(E)
        out["pca"] = pca
        out["hyb"] = hgb(**(hp or {})).fit(np.column_stack([Xp, pca.transform(E)]), y)
    return out


def predict(models, recs, kind):
    Xp = physics(recs, models["prop"])
    if kind == "physics":
        return [models["phys"].predict_proba(X)[:, 1] for X in Xp]
    return [models["hyb"].predict_proba(np.column_stack([X, models["pca"].transform(r["emb"])]))[:, 1]
            for X, r in zip(Xp, recs)]


# ------------------------------------------------------------------------------------------------ metrics

def metrics(scores, ys):
    from sklearn.metrics import average_precision_score, roc_auc_score

    s, y = np.concatenate(scores), np.concatenate(ys)
    per = [roc_auc_score(yy, ss) for ss, yy in zip(scores, ys) if 0 < yy.sum() < len(yy)]
    return {"auc": round(float(roc_auc_score(y, s)), 4), "auprc": round(float(average_precision_score(y, s)), 4),
            "median_auc_per_protein": round(float(np.median(per)), 4), "proteins": len(scores),
            "residues": int(len(y)), "positive_rate": round(float(y.mean()), 4)}


def site_hits(scores, recs, data):
    """Share of proteins one of whose cryptic pockets is found by the first predicted site, and by one of the first
    three (core.sites_from_scores on the Calpha trace, the boosted models' top-10% calling)."""
    from crypsis import core

    top1 = top3 = 0
    for sc, r in zip(scores, recs):
        n = len(r["ca"])
        ch = core.Chain(r["ca"], np.arange(n), r["keys"], r["names"], np.arange(n), np.zeros(n))
        zeros = np.zeros(n)
        sites = core.sites_from_scores(ch, np.asarray(sc), zeros, zeros, zeros, threshold=None)
        true = pockets_of(data, r["apo"])
        found = []
        for st in sites[:3]:
            m = set(st["residues"])
            found.append(any(len(m & t) >= 3 and (len(m & t) >= 0.5 * len(m) or len(m & t) >= len(t) / 3.0)
                             for t in true))
        top1 += bool(found[:1] and found[0])
        top3 += any(found)
    return {"site_top1": round(top1 / len(recs), 4), "site_top3": round(top3 / len(recs), 4)}


def bootstrap(scores, ys, n=2000, seed=0, other=None):
    """95% intervals of pooled AUC / AUPRC, resampling proteins; with ``other`` also the paired differences."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    auc, ap, diff, diff_ap = [], [], [], []
    for _ in range(n):
        idx = rng.integers(0, len(scores), len(scores))
        s, y = np.concatenate([scores[i] for i in idx]), np.concatenate([ys[i] for i in idx])
        auc.append(roc_auc_score(y, s))
        ap.append(average_precision_score(y, s))
        if other is not None:
            o = np.concatenate([other[i] for i in idx])
            diff.append(auc[-1] - roc_auc_score(y, o))
            diff_ap.append(ap[-1] - average_precision_score(y, o))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    out = {"auc_ci": ci(auc), "auprc_ci": ci(ap)}
    if other is not None:
        out["auc_diff_ci"] = ci(diff)
        out["auprc_diff_ci"] = ci(diff_ap)
        # two-sided paired bootstrap P of the pooled difference (added after review): the share of resamples on
        # the other side of zero, doubled; at most 1, and no smaller than 1/n
        p = lambda v: float(min(1.0, max(2 * min(np.mean(np.asarray(v) <= 0), np.mean(np.asarray(v) >= 0)), 1 / n)))
        out["auc_diff_p"] = p(diff)
        out["auprc_diff_p"] = p(diff_ap)
    return out


# ------------------------------------------------------------------------------------------------ commands

def cmd_cv(a):
    """4-fold cross-validation on the training folds (each fold held out once)."""
    esm = a.esm if a.model == "hybrid" else None
    folds = {i: load_split(a, f"train-{i}", esm) for i in range(4)}
    hp = json.loads(a.hp) if a.hp else None
    scores, ys, ids = [], [], []
    for i in range(4):
        train = sum((folds[j] for j in range(4) if j != i), [])
        m = fit_models(train, n_pcs=a.pcs, esm=bool(esm), hp=hp)
        fold_scores = predict(m, folds[i], a.model)
        scores += fold_scores
        ys += [r["y"] for r in folds[i]]
        ids += [r["id"] for r in folds[i]]
        print(f"fold {i}: {metrics(fold_scores, [r['y'] for r in folds[i]])}", flush=True)
    res = dict(metrics(scores, ys), model=a.model, esm=esm, pcs=a.pcs, hp=hp, physics=a.physics)
    print(json.dumps(res))
    with open(os.path.join(a.work, "cv_results.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")
    # out-of-fold scores: the inputs of the second (spatial) stage
    if not hp:
        np.savez_compressed(oof_path(a), ids=np.array(ids), **{i: s for i, s in zip(ids, scores)})


def oof_path(a):
    tag = f"{a.physics}_" + (a.model if a.model == "physics" else f"hybrid_{a.esm}_{a.pcs}")
    return os.path.join(a.work, f"oof_{tag}.npz")


# ------------------------------------------------------------------------------------------------ spatial stage

def neighbourhood(s, ca):
    """Second-stage inputs of one protein: its residue scores pooled over spatial neighbourhoods.

    Cryptic sites are compact patches, so a residue surrounded by high scores is more likely to be one of them than
    an isolated high score; the within-protein rank also corrects proteins whose scores are shifted as a whole.
    """
    d = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    logit = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1))
    cols = [logit, (s.argsort().argsort() + 0.5) / len(s), (s - s.mean()) / (s.std() + 1e-6)]
    for r in (6.0, 9.0, 12.0, 16.0):
        w = d < r
        n = w.sum(1)
        cols += [(w @ s) / n, np.where(w, s[None, :], 0).max(1), n / 50.0]
    g = np.exp(-(d / 8.0) ** 2)
    cols.append((g @ s) / g.sum(1))
    return np.column_stack(cols)


def gaussian_smooth(s, ca, sigma=6.0, alpha=0.5):
    d2 = ((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1)
    g = np.exp(-d2 / (2 * sigma ** 2))
    return alpha * s + (1 - alpha) * (g @ s) / g.sum(1)


STACK_HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0)


def cmd_stack(a):
    """Cross-validate the spatial second stage on the out-of-fold scores of `cv` (same folds, train data only)."""
    oof = np.load(oof_path(a))
    folds = {i: [r for r in load_split(a, f"train-{i}") if r["id"] in oof] for i in range(4)}
    for i in folds:
        for r in folds[i]:
            r["s"] = oof[r["id"]]
            r["Z"] = neighbourhood(r["s"], r["ca"])
    allr = sum(folds.values(), [])
    ys = [r["y"] for r in allr]
    print("stage 1     ", metrics([r["s"] for r in allr], ys), flush=True)
    for sigma in (4.0, 6.0, 8.0):
        for alpha in (0.3, 0.5, 0.7):
            sm = [gaussian_smooth(r["s"], r["ca"], sigma, alpha) for r in allr]
            print(f"gauss s={sigma} a={alpha}", metrics(sm, ys), flush=True)
    hp = dict(STACK_HP, **(json.loads(a.hp) if a.hp else {}))
    out = []
    for i in range(4):
        tr = sum((folds[j] for j in range(4) if j != i), [])
        m = hgb(**hp).fit(np.vstack([r["Z"] for r in tr]), np.concatenate([r["y"] for r in tr]))
        out += [m.predict_proba(r["Z"])[:, 1] for r in folds[i]]
    ys = sum(([r["y"] for r in folds[i]] for i in range(4)), [])
    res = dict(metrics(out, ys), stage="spatial", source=os.path.basename(oof_path(a)), hp=hp)
    print("stacked     ", json.dumps(res), flush=True)
    with open(os.path.join(a.work, "cv_results.jsonl"), "a") as fh:
        fh.write(json.dumps(res) + "\n")


def config_tag(a):
    return f"{a.physics}_{a.esm}_{a.pcs}"


def cmd_train(a):
    recs = load_split(a, "train", a.esm)
    hp = json.loads(a.hp) if a.hp else None
    m = fit_models(recs, n_pcs=a.pcs, esm=True, hp=hp)
    path = os.path.join(a.work, f"final_models_{config_tag(a)}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(m, fh)
    print(f"{len(recs)} training proteins -> {path}")


def cmd_test(a):
    """Score the test split once: the shipped models (--shipped) or the models of `train` with the same options,
    raw and with the Gaussian spatial smoothing chosen in cross-validation (--smooth sigma,alpha), each with
    protein-bootstrap intervals, site hits and, when work/test_v2_shipped_scores.npz exists (the Crypsis 0.1 models,
    from `test --shipped --physics v2` with the 0.1 package), the paired differences to the 0.1 hybrid."""
    recs = load_split(a, "test", a.esm)
    ys = [r["y"] for r in recs]
    ids = [r["id"] for r in recs]
    results = {}
    if a.shipped:
        from crypsis import pipeline

        phys, hyb = pipeline._load("physics"), pipeline._load(f"hybrid_{a.esm}")
        Xp = physics(recs, None)
        preds = {"physics (shipped)": [phys.predict_proba(X) for X in Xp],
                 f"hybrid {a.esm} (shipped)": [
                     hyb.predict_proba(np.column_stack([X, (r["emb"] - hyb.extra["pca_mean"]) @ hyb.extra["pca_components"].T]))
                     for X, r in zip(Xp, recs)]}
        out = os.path.join(a.work, f"test_shipped_{a.physics}_{a.esm}.json")
    else:
        tag = config_tag(a)
        m = pickle.load(open(os.path.join(a.work, f"final_models_{tag}.pkl"), "rb"))
        preds = {"physics": predict(m, recs, "physics"), "hybrid": predict(m, recs, "hybrid")}
        if a.smooth:
            sigma, alpha = (float(v) for v in a.smooth.split(","))
            for k in list(preds):
                preds[f"{k}+smooth"] = [gaussian_smooth(sc, r["ca"], sigma, alpha) for sc, r in zip(preds[k], recs)]
        out = os.path.join(a.work, f"test_{tag}.json")
    ref = None
    shipped = os.path.join(a.work, "test_v2_shipped_scores.npz")
    if os.path.exists(shipped):
        z = np.load(shipped, allow_pickle=True)
        pos = {i: k for k, i in enumerate(z["ids"])}
        if all(i in pos for i in ids):
            ref = [z["hybrid"][pos[i]] for i in ids]
    for name, sc in preds.items():
        results[name] = dict(metrics(sc, ys), **site_hits(sc, recs, a.data), **bootstrap(sc, ys, n=a.boot, other=ref))
        print(name, json.dumps(results[name]), flush=True)
    json.dump(results, open(out, "w"), indent=1)
    np.savez_compressed(out.replace(".json", "_scores.npz"), ids=np.array(ids),
                        **{k.split()[0].replace("+", "_"): np.array(v, dtype=object) for k, v in preds.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["features"] + EXTRAS + ["esm", "cv", "stack", "train", "test"])
    ap.add_argument("--data", default=os.path.expanduser("~/crypsis_data/cryptobench"))
    ap.add_argument("--work", default=os.path.expanduser("~/crypsis_data/work"))
    ap.add_argument("--split", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--esm", default="650M", choices=sorted(ESM_MODELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="hybrid", choices=["physics", "hybrid"])
    ap.add_argument("--pcs", type=int, default=256)
    ap.add_argument("--hp", default="", help="JSON of HistGradientBoostingClassifier parameters")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--physics", default="v3", choices=["v2", "v3"],
                    help="physics inputs: v3 (Crypsis 0.2, 49 columns) or v2 (Crypsis 0.1, 22 columns)")
    ap.add_argument("--smooth", default="", help="test: also score with Gaussian spatial smoothing 'sigma,alpha'")
    ap.add_argument("--shipped", action="store_true", help="score the models shipped in src/crypsis/models")
    a = ap.parse_args()
    commands = {"features": cmd_features, "esm": cmd_esm, "cv": cmd_cv, "stack": cmd_stack, "train": cmd_train,
                "test": cmd_test, **{k: cmd_extra for k in EXTRAS}}
    commands[a.command](a)


if __name__ == "__main__":
    main()

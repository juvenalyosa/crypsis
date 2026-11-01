#!/usr/bin/env python3
"""Is it the slow modes, or any perturbation?  Controls for the ensemble-versus-static comparison, added after
review.  Each control is a per-residue score used directly, without learning, on the CryptoBench test split (and
the temporal set with --set temporal):

    python scripts/perturbation_controls.py run --set cryptobench
    python scripts/perturbation_controls.py summary --set cryptobench
    python scripts/perturbation_controls.py extra --set cryptobench      # second round of review (below)
    python scripts/perturbation_controls.py per-protein --set cryptobench

  slow      mean pocket score over 24 conformers along the 30 slowest ANM modes, 3 A Calpha RMSD (the shipped
            ens_score_mean, recomputed here with the same code)
  fast      the same with random combinations of ANM modes 31-60 (higher frequency), same weighting and RMSD
  random    independent isotropic Gaussian displacement of every residue, same RMSD
  jitter    the apo structure itself under 24 random rigid rotations and translations: averages the grid
            discretization only
  dilated   minus the distance from the residue's atoms to the nearest pocket point of the apo structure

All perturbed conformers move every atom of a residue with its Calpha (core.conformers' scheme), use the shipped
seed and the same pocket grid.  The paired differences of `slow` against each control use the same protein
resamples as scripts/cryptobench.py.  Results: <work>/perturbation_controls.json; scores per protein in
<work>/perturbation/<id>.npz.

Second round of review.  `extra` adds, in <work>/perturbation_extra/<id>.npz:

  smooth_<A>   a spatially smooth random deformation that does not come from the elastic network: independent
               Gaussian vectors on the Calpha atoms convolved with a Gaussian kernel of 10 A correlation length
               (exp(-d^2 / 2 l^2) over Calpha distances), scaled to A Calpha RMSD, 24 copies
  slow_<A>, fast_<A>, random_<A>   the controls above at other amplitudes (dose response)
  vbond_<name>   mean absolute change of consecutive Calpha-Calpha distances over the 24 copies (local strain)

`per-protein` evaluates every control as the models are judged elsewhere: per-protein AUC (mean and median, with
paired Wilcoxon tests of `slow` against each control, Holm over the controls) and pooled AUC after ranking each
protein's scores within the protein (the learned models see these columns rank-normalized within a protein).
Results: <work>/perturbation_per_protein.json.
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402

HOME = os.path.expanduser("~/crypsis_data")
SETS = {"cryptobench": (f"{HOME}/cryptobench", f"{HOME}/work"),
        "temporal": (f"{HOME}/temporal_bench", f"{HOME}/work_temporal")}
CONTROLS = ["slow", "fast", "random", "jitter", "dilated"]


def _scaled(chain, disp, rmsd):
    n = disp.shape[0]
    scale = rmsd / max(math.sqrt((disp ** 2).sum() / n), 1e-9)
    return chain.xyz + (disp * scale)[chain.motion_res]


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def _one(job):
    apo, chain_id, data, out = job
    from crypsis import core

    path = os.path.join(out, f"{apo}{chain_id}.npz")
    if os.path.exists(path):
        return apo, "cached"
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
        p = dict(core.DEFAULTS, **(core.load_model().get("params") or {}))
        n, k, rmsd, nconf = ch.n_res, int(p["n_modes"]), p["rmsd_A"], int(p["n_conformers"])
        ca = ch.xyz[ch.ca_index]
        grid = lambda x: core.pocket_grid(x, ch.atom_res, n, p)["residue_score"]
        res = {}
        vals, modes = core.anm_modes(ca, p["anm_cutoff_A"], 2 * k)
        for name, sl in (("slow", slice(0, k)), ("fast", slice(k, 2 * k))):
            if name == "slow":           # exactly the shipped modes and conformers
                v, m = core.anm_modes(ca, p["anm_cutoff_A"], k)
                res[name] = np.mean([grid(x) for x in core.conformers(ch, v, m, nconf, rmsd, p["seed"])], axis=0)
            else:
                v, m = vals[sl], modes[sl]
                res[name] = np.mean([grid(x) for x in core.conformers(ch, v, m, nconf, rmsd, p["seed"])], axis=0)
        rng = np.random.default_rng(p["seed"])
        res["random"] = np.mean([grid(_scaled(ch, rng.normal(size=(n, 3)), rmsd)) for _ in range(nconf)], axis=0)
        centre = ch.xyz.mean(axis=0)
        jit = []
        for _ in range(nconf):
            x = (ch.xyz - centre) @ _rotation(rng).T + centre + rng.uniform(-0.5, 0.5, size=3)
            jit.append(grid(x))
        res["jitter"] = np.mean(jit, axis=0)
        g = core.pocket_grid(ch.xyz, ch.atom_res, n, p)
        keep = np.isin(g["label"], np.flatnonzero(g["cluster_volume"] >= p["min_volume_A3"]))
        pts = g["points"][keep]
        if len(pts):
            from scipy.spatial import cKDTree

            d_atom, _ = cKDTree(pts).query(ch.xyz)
            prot = ch.atom_res >= 0
            dmin = np.full(n, np.inf)
            np.minimum.at(dmin, ch.atom_res[prot], d_atom[prot])
            res["dilated"] = -dmin
        else:
            res["dilated"] = np.zeros(n)
        np.savez_compressed(path, keys=np.array(ch.res_keys), **res)
        return apo, "ok"
    except Exception as exc:
        return apo, f"error {type(exc).__name__}: {exc}"


L_SMOOTH = 10.0
EXTRA = [("slow", 1.0), ("slow", 1.5), ("slow", 5.0), ("fast", 1.0), ("fast", 5.0), ("random", 1.0),
         ("smooth", 1.0), ("smooth", 3.0), ("smooth", 5.0)]


def _vbond(ch, x):
    ca0, ca1 = ch.xyz[ch.ca_index], x[ch.ca_index]
    return float(np.abs(np.linalg.norm(np.diff(ca1, axis=0), axis=1) - np.linalg.norm(np.diff(ca0, axis=0), axis=1)).mean())


def _one_extra(job):
    apo, chain_id, data, out = job
    from crypsis import core

    path = os.path.join(out, f"{apo}{chain_id}.npz")
    if os.path.exists(path):
        return apo, "cached"
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
        p = dict(core.DEFAULTS, **(core.load_model().get("params") or {}))
        n, k, nconf = ch.n_res, int(p["n_modes"]), int(p["n_conformers"])
        ca = ch.xyz[ch.ca_index]
        grid = lambda x: core.pocket_grid(x, ch.atom_res, n, p)["residue_score"]
        vals, modes = core.anm_modes(ca, p["anm_cutoff_A"], 2 * k)
        v_slow, m_slow = core.anm_modes(ca, p["anm_cutoff_A"], k)
        d2 = ((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1)
        kern = np.exp(-d2 / (2 * L_SMOOTH ** 2))
        res = {}
        for name, amp in EXTRA + [("slow", 3.0), ("random", 3.0)]:
            rng = np.random.default_rng(p["seed"])
            if name == "slow":
                xs = core.conformers(ch, v_slow, m_slow, nconf, amp, p["seed"])
            elif name == "fast":
                xs = core.conformers(ch, vals[k:2 * k], modes[k:2 * k], nconf, amp, p["seed"])
            elif name == "random":
                xs = [_scaled(ch, rng.normal(size=(n, 3)), amp) for _ in range(nconf)]
            else:
                xs = [_scaled(ch, kern @ rng.normal(size=(n, 3)), amp) for _ in range(nconf)]
            xs = list(xs)
            res[f"vbond_{name}_{amp:g}"] = np.array(np.mean([_vbond(ch, x) for x in xs]))
            if (name, amp) in EXTRA:
                res[f"{name}_{amp:g}"] = np.mean([grid(x) for x in xs], axis=0)
        np.savez_compressed(path, keys=np.array(ch.res_keys), **res)
        return apo, "ok"
    except Exception as exc:
        return apo, f"error {type(exc).__name__}: {exc}"


def cmd_extra(a):
    out = os.path.join(a.work, "perturbation_extra")
    os.makedirs(out, exist_ok=True)
    jobs = [(apo, ch, a.data, out) for apo, ch, _pos in cb.entries(a.data, "test")]
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as ex:
        for k, (apo, st) in enumerate(ex.map(_one_extra, jobs), 1):
            if st not in ("ok", "cached"):
                print(f"  {apo}: {st}", flush=True)
            if k % 25 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)


def cmd_per_protein(a):
    from scipy.stats import rankdata, wilcoxon
    from sklearn.metrics import roc_auc_score

    import heldout_eval as he
    from crypsis import core

    recs = cb.load_split(SimpleNamespace(data=a.data, work=a.work, physics="v3"), "test")
    sc = {"static": [r["feats"][:, core.FEATURES.index("score")] for r in recs]}
    vb = {}
    for r in recs:
        z = np.load(os.path.join(a.work, "perturbation", f"{r['id']}.npz"))
        assert [str(k) for k in z["keys"]] == r["keys"], r["id"]
        for c in CONTROLS:
            sc.setdefault(c, []).append(z[c])
        f = os.path.join(a.work, "perturbation_extra", f"{r['id']}.npz")
        if os.path.exists(f):
            z = np.load(f)
            assert [str(k) for k in z["keys"]] == r["keys"], r["id"]
            for c in z.files:
                if c.startswith("vbond_"):
                    vb.setdefault(c[6:], []).append(float(z[c]))
                elif c != "keys":
                    sc.setdefault(c, []).append(z[c])
    sc = {c: v for c, v in sc.items() if len(v) == len(recs)}
    ys = [r["y"] for r in recs]
    ok = np.array([0 < y.sum() < len(y) for y in ys])
    per = {c: np.array([roc_auc_score(y, s) if k else np.nan for s, y, k in zip(v, ys, ok)]) for c, v in sc.items()}
    y_all = np.concatenate(ys)
    res = {"proteins": len(recs), "controls": {}, "slow_vs": {}, "vbond_mean": {c: round(float(np.mean(v)), 3)
                                                                               for c, v in vb.items()}}
    for c, v in sc.items():
        rk = np.concatenate([rankdata(s) / len(s) for s in v])
        res["controls"][c] = {"pooled_auc": round(float(roc_auc_score(y_all, np.concatenate(v))), 4),
                              "pooled_auc_rank_normalized": round(float(roc_auc_score(y_all, rk)), 4),
                              "median_auc_per_protein": round(float(np.nanmedian(per[c])), 4),
                              **he.per_protein_means(per[c], per[c], a.boot)}
        res["controls"][c].pop("mean_auprc_per_protein", None)
        res["controls"][c].pop("mean_auprc_per_protein_ci", None)
    others = [c for c in sc if c != "slow"]
    ps = []
    for c in others:
        d = (per["slow"] - per[c])[ok]
        ps.append(float(wilcoxon(d).pvalue))
        res["slow_vs"][c] = {"mean_diff": round(float(d.mean()), 4), "mean_diff_ci": he.per_protein_diff_ci(d, a.boot),
                             "median_diff": round(float(np.median(d)), 4),
                             "slow_wins": int((d > 0).sum()), "n": int(ok.sum()), "wilcoxon_p": ps[-1]}
    for c, q in zip(others, he.holm(ps)):
        res["slow_vs"][c]["wilcoxon_p_holm"] = q
    for est in ("pooled_auc", "pooled_auc_rank_normalized", "mean_auc_per_protein"):
        st, sl, jt = (res["controls"][c][est] for c in ("static", "slow", "jitter"))
        res[f"jitter_recovers_{est}"] = round((jt - st) / (sl - st), 3)
    print(json.dumps(res, indent=1))
    with open(os.path.join(a.work, "perturbation_per_protein.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def cmd_run(a):
    out = os.path.join(a.work, "perturbation")
    os.makedirs(out, exist_ok=True)
    jobs = [(apo, ch, a.data, out) for apo, ch, _pos in cb.entries(a.data, "test")]
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as ex:
        for k, (apo, st) in enumerate(ex.map(_one, jobs), 1):
            if st not in ("ok", "cached"):
                print(f"  {apo}: {st}", flush=True)
            if k % 25 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)


def cmd_summary(a):
    recs = cb.load_split(SimpleNamespace(data=a.data, work=a.work, physics="v3"), "test")
    sc = {c: [] for c in CONTROLS}
    for r in recs:
        z = np.load(os.path.join(a.work, "perturbation", f"{r['id']}.npz"))
        assert [str(k) for k in z["keys"]] == r["keys"], r["id"]
        for c in CONTROLS:
            sc[c].append(z[c])
    ys = [r["y"] for r in recs]
    # the shipped column, as a check that "slow" reproduces it
    from crypsis import core

    shipped = [r["feats"][:, core.FEATURES.index("ens_score_mean")] for r in recs]
    static = [r["feats"][:, core.FEATURES.index("score")] for r in recs]
    res = {"check_slow_equals_shipped": float(max(np.abs(s - t).max() for s, t in zip(sc["slow"], shipped))),
           "static": dict(cb.metrics(static, ys), **cb.bootstrap(static, ys, n=a.boot)), "controls": {}}
    for c in CONTROLS:
        res["controls"][c] = dict(cb.metrics(sc[c], ys), **cb.bootstrap(sc[c], ys, n=a.boot))
        if c != "slow":
            d = cb.bootstrap(sc["slow"], ys, n=a.boot, other=sc[c])
            res["controls"][c]["slow_minus_this"] = {"auc_diff_ci": d["auc_diff_ci"], "auprc_diff_ci": d["auprc_diff_ci"]}
        d = cb.bootstrap(sc[c], ys, n=a.boot, other=static)
        res["controls"][c]["this_minus_static"] = {"auc_diff_ci": d["auc_diff_ci"], "auprc_diff_ci": d["auprc_diff_ci"]}
        print(c, json.dumps(res["controls"][c]), flush=True)
    print("slow reproduces the shipped column to", res["check_slow_equals_shipped"])
    with open(os.path.join(a.work, "perturbation_controls.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "summary", "extra", "per-protein"])
    ap.add_argument("--set", choices=sorted(SETS), default="cryptobench")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    a.data, a.work = SETS[a.set]
    {"run": cmd_run, "summary": cmd_summary, "extra": cmd_extra, "per-protein": cmd_per_protein}[a.command](a)


if __name__ == "__main__":
    main()

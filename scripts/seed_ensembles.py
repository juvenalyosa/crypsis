#!/usr/bin/env python3
"""How much the finite normal-mode ensemble moves the test-split results: the shipped models scored on ensembles
drawn with other random seeds.

    python scripts/seed_ensembles.py features --seed 11 --work-seeds ~/crypsis_data/work_seeds
    python scripts/cryptobench.py test --shipped --esm 650M --pcs 256 --work ~/crypsis_data/work_seeds/seed11
    python scripts/seed_ensembles.py summary --seeds 1 2 3 4      # with seed 7 from the main work directory

Each conformer is a random Gaussian combination of the modes (core.conformers), so flipping a mode's sign is the
same as flipping its random coefficient, whose distribution is symmetric: a flipped sign gives another, equally
likely ensemble, and so does another seed.  The spread over seeds is therefore the scale of what the arbitrary
LAPACK sign changed before Crypsis 0.2.1 fixed it.  Everything that depends on the ensemble is recomputed with the
new seed (the ensemble columns of core.features, pipeline.shallow_pass and descriptors.ensemble_descriptors); the
static descriptors and the ESM-2 embeddings are linked from the main work directory.  Seed 7 is the shipped one.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402


def _one(job):
    apo, chain_id, pos, data, out, seed = job
    from crypsis import core, descriptors, pipeline

    ident = f"{apo}{chain_id}"
    paths = [os.path.join(out, kind, f"{ident}.npz") for kind in ("features", "ensemble")]
    if all(os.path.exists(p) for p in paths):
        return apo, "cached", 0.0
    t0 = time.time()
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
        p = dict(core.load_model().get("params") or {}, seed=seed)
        feats, extra = core.features(ch, p)
        shallow = pipeline.shallow_pass(ch, extra)
        ens = descriptors.ensemble_descriptors(ch, feats[:, core.FEATURES.index("score")], params=p, anm=extra["anm"])
        y = np.array([1 if k in set(pos) else 0 for k in ch.res_keys], dtype=np.int8)
        np.savez_compressed(paths[0], feats=feats, shallow=shallow, y=y, keys=np.array(ch.res_keys),
                            names=np.array(ch.res_names), ca=ch.xyz[ch.ca_index],
                            seqs=np.array(pipeline.sequences(ch)))
        np.savez_compressed(paths[1], x=np.asarray(ens, dtype=np.float32))
        return apo, "ok", time.time() - t0
    except Exception as exc:
        return apo, f"error {type(exc).__name__}: {exc}", 0.0


def cmd_features(a):
    out = os.path.join(a.work_seeds, f"seed{a.seed}")
    for kind in ("features", "ensemble"):
        os.makedirs(os.path.join(out, kind), exist_ok=True)
    for kind in ("descriptors", "esm_650M", "esm_150M"):
        link = os.path.join(out, kind)
        if not os.path.exists(link):
            os.symlink(os.path.join(a.work, kind), link)
    jobs = [(apo, ch, pos, a.data, out, a.seed) for apo, ch, pos in cb.entries(a.data, "test")]
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as ex:
        for k, (apo, st, _sec) in enumerate(ex.map(_one, jobs), 1):
            if st not in ("ok", "cached"):
                print(f"  {apo}: {st}", flush=True)
            if k % 50 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)


def cmd_summary(a):
    dirs = {7: a.work} | {s: os.path.join(a.work_seeds, f"seed{s}") for s in a.seeds}
    keys = ["auc", "auprc", "median_auc_per_protein", "site_top1", "site_top3"]
    rows = {}
    for s, d in dirs.items():
        for esm in ("650M", "150M"):
            with open(os.path.join(d, f"test_shipped_v3_{esm}.json")) as fh:
                for name, v in json.load(fh).items():
                    rows.setdefault(name.split(" (")[0], {})[s] = [v[k] for k in keys]
    out = {}
    for name, by_seed in rows.items():
        x = np.array(list(by_seed.values()))
        out[name] = {"seeds": sorted(by_seed), "per_seed": by_seed,
                     **{k: {"min": float(x[:, j].min()), "max": float(x[:, j].max()),
                            "sd": round(float(x[:, j].std(ddof=1)), 4)} for j, k in enumerate(keys)}}
        print(name, {k: (out[name][k]["min"], out[name][k]["max"]) for k in keys})
    with open(os.path.join(a.work_seeds, "seed_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["features", "summary"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--data", default=os.path.expanduser("~/crypsis_data/cryptobench"))
    ap.add_argument("--work", default=os.path.expanduser("~/crypsis_data/work"))
    ap.add_argument("--work-seeds", default=os.path.expanduser("~/crypsis_data/work_seeds"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    {"features": cmd_features, "summary": cmd_summary}[a.command](a)


if __name__ == "__main__":
    main()

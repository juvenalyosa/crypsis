#!/usr/bin/env python3
"""The PocketMiner validation and test sets (Meller et al., Nat. Commun. 2023), scored as in that paper.

    python scripts/pocketminer_set.py --pm ~/crypsis_data/external/gvp/data/pm-dataset \\
        --cryptobench ~/crypsis_data/work --out ~/crypsis_data/pocketminer/v0.2
    python scripts/pocketminer_set.py ... --src /path/to/crypsis-0.1/src --esm - --out ...   # an older version

--pm is data/pm-dataset of https://github.com/Mickdub/gvp (branch pocket_pred): the cleaned apo structures and, per
protein, a label for every residue (1 cryptic-pocket residue, 0 negative, 2 uncertain).  Labels 1 and 0 of all the
proteins of a set are pooled (2 is left out, as in the paper) for one ROC AUC, with a 95% interval from resampling
proteins.  The label dictionaries are pickled numpy objects; they are read with an unpickler that accepts only numpy
arrays and builtin containers.  "Unseen" test proteins have no chain of >= 50% sequence identity among the
CryptoBench training chains (--cryptobench: the work folder of scripts/cryptobench.py, for their sequences).

Second round of review: P2Rank and PocketMiner scored on the same files and labels, with their per-residue scores
kept (--prep writes the structures as <split>_<id>.pdb; P2Rank and scripts/pocketminer_run.py are run on that folder;
--collect-p2rank converts P2Rank's output; --baselines adds both to the report, unseen subset included).
"""

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from literature_cases import PREDICT  # noqa: E402

ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
           ("numpy._core.multiarray", "scalar"), ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
           ("_codecs", "encode")}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        return super().find_class(module, name)


def load_labels(path):
    """A pickled label dictionary {pdb id: [0/1/2 per residue]} without running arbitrary pickle code."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        np.lib.format._read_array_header(fh, version)
        obj = _Restricted(fh).load()
    return obj.item() if isinstance(obj, np.ndarray) else obj


def structure_file(pm, ident):
    files = {f.lower(): f for f in os.listdir(os.path.join(pm, "apo-structures"))}
    for cand in (f"{ident}_clean_h.pdb", f"{ident[:4]}_clean_h.pdb"):
        if cand.lower() in files:
            return os.path.join(pm, "apo-structures", files[cand.lower()])
    raise FileNotFoundError(ident)


def label_of(labels, ident):
    for k in (ident[:4], ident, ident[:4].upper(), ident[:4].lower(), ident.upper()):
        if k in labels:
            return np.asarray(labels[k])
    raise KeyError(ident)


def unseen(seq, train_seqs, identity=50.0):
    import gemmi

    for t in train_seqs:
        if min(len(t), len(seq)) < 0.5 * max(len(t), len(seq)):
            continue
        al = gemmi.align_string_sequences(list(seq), list(t), [])
        if al.calculate_identity() >= identity:
            return False
    return True


def pooled_auc(scores, ys, n=2000, seed=0):
    from sklearn.metrics import roc_auc_score

    s, y = np.concatenate(scores), np.concatenate(ys)
    rng = np.random.default_rng(seed)
    boot = []
    while len(boot) < n:
        idx = rng.integers(0, len(scores), len(scores))
        yy = np.concatenate([ys[i] for i in idx])
        if 0 < yy.sum() < len(yy):
            boot.append(roc_auc_score(yy, np.concatenate([scores[i] for i in idx])))
    return {"auc": round(float(roc_auc_score(y, s)), 3), "ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
            "proteins": len(scores), "residues": int(len(y)), "positives": int(y.sum())}


def prep(a):
    """Copy every structure of both sets to <dir>/<split>_<id>.pdb, for running P2Rank and PocketMiner on them."""
    import shutil

    os.makedirs(a.prep, exist_ok=True)
    for split in ("val", "test"):
        for ident in (str(i) for i in np.load(os.path.join(a.pm, f"{split}_apo_ids_with_chainids.npy"))):
            shutil.copyfile(structure_file(a.pm, ident), os.path.join(a.prep, f"{split}_{ident}.pdb"))
    print("->", a.prep)


def collect_p2rank(a, raw, out):
    """P2Rank residue probabilities (<raw>/<name>.pdb_residues.csv) in core.read_chain's residue order."""
    from crypsis import core

    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(raw)):
        if not f.endswith(".pdb_residues.csv"):
            continue
        name = f[: -len(".pdb_residues.csv")]
        chain = core.read_chain(os.path.join(a.prep, f"{name}.pdb"))
        prob = {}
        with open(os.path.join(raw, f)) as fh:
            head = [h.strip() for h in fh.readline().split(",")]
            for ln in fh:
                row = dict(zip(head, (v.strip() for v in ln.split(","))))
                prob[f"{row['chain']}_{row['residue_label']}"] = float(row["probability"])
        np.save(os.path.join(out, f"{name}.npy"), np.array([prob.get(k, 0.0) for k in chain.res_keys]))
    print("->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pm", required=True)
    ap.add_argument("--prep", help="second round of review: copy the structures here and exit (see prep)")
    ap.add_argument("--collect-p2rank", nargs=2, metavar=("RAW", "OUT"),
                    help="convert P2Rank output for the --prep files to <OUT>/<split>_<id>.npy and exit")
    ap.add_argument("--baselines", default="",
                    help="name=dir,...: per-residue scores <dir>/<split>_<id>.npy (P2Rank, PocketMiner) to score too")
    ap.add_argument("--cryptobench", required=True, help="scripts/cryptobench.py work folder (training sequences)")
    ap.add_argument("--data", default=os.path.expanduser("~/crypsis_data/cryptobench"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    ap.add_argument("--models", default="physics,hybrid")
    ap.add_argument("--esm", default="650M", help="hybrid language model; '-' for versions without the option")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.src))
    if a.prep and not a.collect_p2rank:
        return prep(a)
    if a.collect_p2rank:
        return collect_p2rank(a, *a.collect_p2rank)
    import cryptobench as cb
    from crypsis import core, pipeline

    baselines = dict(b.split("=", 1) for b in a.baselines.split(",") if b)

    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.abspath(a.src))
    train = [str(s) for apo, ch, _ in cb.entries(a.data, "train")
             if os.path.exists(f := os.path.join(a.cryptobench, "features", f"{apo}{ch}.npz"))
             for s in np.load(f)["seqs"]]
    report = {}
    for split in ("val", "test"):
        ids = [str(i) for i in np.load(os.path.join(a.pm, f"{split}_apo_ids_with_chainids.npy"))]
        labels = load_labels(os.path.join(a.pm, f"{split}_label_dictionary.npy"))
        per = {m: ([], [], []) for m in a.models.split(",") + list(baselines)}
        missing = {b: [] for b in baselines}
        for ident in ids:
            path = structure_file(a.pm, ident)
            lab = label_of(labels, ident)
            chain = core.read_chain(path)
            if len(lab) != chain.n_res:
                raise ValueError(f"{ident}: {len(lab)} labels for {chain.n_res} residues")
            new = unseen("".join(pipeline.sequences(chain)), train)
            keep = lab != 2
            for b, d in baselines.items():
                f = os.path.join(d, f"{split}_{ident}.npy")
                if not os.path.exists(f):
                    missing[b].append(ident)
                    continue
                prob = np.load(f)                                  # float array written by our runs
                if len(prob) != len(lab):
                    missing[b].append(ident)
                    continue
                per[b][0].append(prob[keep])
                per[b][1].append(lab[keep].astype(int))
                per[b][2].append(new)
            for m in a.models.split(","):
                cache = os.path.join(a.out, f"{split}_{ident}_{m}.json")
                if not os.path.exists(cache):
                    chains = ",".join(sorted({k.split("_")[0] for k in chain.res_keys}))
                    res = subprocess.run([sys.executable, "-c", PREDICT, path, chains, "", m, a.esm], env=env,
                                         capture_output=True, text=True, check=True)
                    with open(cache, "w") as fh:
                        fh.write(res.stdout)
                prob = np.asarray(json.load(open(cache))["probability"])
                per[m][0].append(prob[keep])
                per[m][1].append(lab[keep].astype(int))
                per[m][2].append(new)
        for b, miss in missing.items():
            report[f"{split} {b} missing"] = miss
        for m, (scores, ys, news) in per.items():
            if not scores:
                continue
            report[f"{split} {m}"] = pooled_auc(scores, ys)
            if split == "test":
                sel = [i for i, u in enumerate(news) if u]
                report[f"{split} unseen {m}"] = pooled_auc([scores[i] for i in sel], [ys[i] for i in sel])
    for k, v in report.items():
        print(k, json.dumps(v), flush=True)
    json.dump(report, open(os.path.join(a.out, "pocketminer_set.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A geometric baseline without learning, added for the manuscript: fpocket 4.0 on the apo structures.

    python scripts/baselines.py fpocket --fpocket /path/to/fpocket --work ~/crypsis_data/work

Each apo chain is read as Crypsis reads it (core.read_chain: protein heavy atoms, first alternate location, no
ligands or waters) and written with residues numbered 1..n in order, so fpocket's pockets map back exactly onto
Crypsis's residues.  A residue's score is the highest fpocket pocket score ("Score" in <name>_info.txt) among the
pockets it lines (atoms in pockets/pocket<N>_atm.pdb), and 0 when it lines none.  The score was fixed before the
test split was run: fpocket ranks its pockets by this value.  The test split is scored with the same metrics,
protein bootstrap and paired differences as scripts/cryptobench.py.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from crypsis import core  # noqa: E402


def write_pdb(ch, path):
    """Protein atoms of a core.Chain, residue i numbered i+1 on chain A (atom names from the structure file)."""
    lines = []
    for k, (x, y, z) in enumerate(ch.xyz):
        r = ch.atom_res[k]
        if r < 0:
            continue
        name = str(ch.atom_names[k])
        elem = "SE" if name.startswith("SE") else name[0]
        field = f" {name:<3s}" if len(name) < 4 else name          # PDB convention: 1-letter elements in column 14
        lines.append(f"ATOM  {len(lines) + 1:5d} {field} {ch.res_names[r]:>3s} A{r + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\nEND\n")


def parse_scores(out_dir, name, n_res):
    info = open(os.path.join(out_dir, f"{name}_info.txt")).read()
    pocket_score = {int(p): float(s) for p, s in re.findall(r"Pocket (\d+) :\s*\n\s*Score :\s*([-\d.]+)", info)}
    s = np.zeros(n_res)
    pdir = os.path.join(out_dir, "pockets")
    for p, score in pocket_score.items():
        f = os.path.join(pdir, f"pocket{p}_atm.pdb")
        if not os.path.exists(f):
            continue
        for ln in open(f):
            if ln.startswith("ATOM"):
                r = int(ln[22:26]) - 1
                s[r] = max(s[r], score)
    return s, len(pocket_score)


def _one(job):
    apo, chain_id, data, out_dir, fpocket = job
    path = os.path.join(out_dir, f"{apo}{chain_id}.npz")
    if os.path.exists(path):
        return apo, "cached"
    tmp = tempfile.mkdtemp()
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
        name = "protein"
        write_pdb(ch, os.path.join(tmp, f"{name}.pdb"))
        r = subprocess.run([fpocket, "-f", f"{name}.pdb"], cwd=tmp, capture_output=True, text=True, timeout=1800)
        out = os.path.join(tmp, f"{name}_out")
        if r.returncode != 0 or not os.path.isdir(out):
            return apo, f"fpocket failed: {r.stderr[-300:]}"
        s, n_pockets = parse_scores(out, name, ch.n_res)
        np.savez_compressed(path, score=s, keys=np.array(ch.res_keys), n_pockets=n_pockets)
        return apo, "ok"
    except Exception as exc:
        return apo, f"error {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_fpocket(a):
    out_dir = os.path.join(a.work, "fpocket")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(apo, ch, a.data, out_dir, a.fpocket) for apo, ch, _pos in cb.entries(a.data, "test")]
    with cf.ProcessPoolExecutor(a.workers) as ex:
        status = dict(ex.map(_one, jobs))
    bad = {k: v for k, v in status.items() if v not in ("ok", "cached")}
    print(f"{len(status) - len(bad)} structures ok, {len(bad)} failed: {bad}", flush=True)

    recs = cb.load_split(a, "test")
    ref = np.load(os.path.join(a.work, "test_shipped_v3_650M_scores.npz"), allow_pickle=True)  # our own file
    pos = {i: k for k, i in enumerate(ref["ids"])}
    keep = [r for r in recs if os.path.exists(os.path.join(out_dir, f"{r['id']}.npz"))]
    scores = []
    for r in keep:
        z = np.load(os.path.join(out_dir, f"{r['id']}.npz"))
        assert [str(k) for k in z["keys"]] == r["keys"], r["id"]
        scores.append(z["score"])
    ys = [r["y"] for r in keep]
    res = dict(cb.metrics(scores, ys), **cb.site_hits(scores, keep, a.data), **cb.bootstrap(scores, ys, n=a.boot))
    for name in ("physics", "hybrid"):
        other = [np.asarray(ref[name][pos[r["id"]]], dtype=float) for r in keep]
        d = cb.bootstrap(other, ys, n=a.boot, other=scores)        # Crypsis model minus fpocket
        res[f"{name}_minus_fpocket"] = {"auc_diff_ci": d["auc_diff_ci"], "auprc_diff_ci": d["auprc_diff_ci"]}
    res["failed"] = bad
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(a.work, "baseline_fpocket.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fpocket"])
    ap.add_argument("--fpocket", default=shutil.which("fpocket") or "fpocket")
    ap.add_argument("--data", default=os.path.expanduser("~/crypsis_data/cryptobench"))
    ap.add_argument("--work", default=os.path.expanduser("~/crypsis_data/work"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--physics", default="v3")
    a = ap.parse_args()
    {"fpocket": cmd_fpocket}[a.command](a)


if __name__ == "__main__":
    main()

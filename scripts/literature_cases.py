#!/usr/bin/env python3
"""Eight cryptic sites from the literature: apo structure -> holo structure with the drug or ligand bound.

    python scripts/literature_cases.py --pdb ~/crypsis_data/cases/pdb --out ~/crypsis_data/cases/v0.2
    python scripts/literature_cases.py ... --src /path/to/crypsis-0.1/src --out ...   # an older version

The mmCIF files are read from --pdb/<ID>.cif.gz (download them from https://files.rcsb.org/download/<ID>.cif.gz).
The true site is every apo residue with a heavy atom within 4.5 A of the ligand's heavy atoms in the holo structure,
the residues being matched by aligning the two chains' sequences.  Each prediction runs in a fresh process, as the
command line does; reported per case: the AUC within the protein and the rank of the first predicted site that
finds the true one (it shares at least half of its own residues, or a third of the true site's, and at least 3).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

# apo id, apo chains, holo id, holo chain, ligand (name, or name:residue number), cofactors kept, description.
# A ligand present twice is restricted to the copy in the site described (MDM2: nutlin 110 in the p53 cleft; p38a:
# the copy of 362 in the C-lobe lipid pocket, the other one is in the ATP site); TEM-1's two CBT molecules both bind
# the cryptic site.
CASES = [
    ("4LDJ", "A", "6OIM", "A", "MOV", "GDP,MG", "K-Ras G12C + GDP: switch-II pocket (sotorasib)"),
    ("1ZG4", "A", "1PZO", "A", "CBT", None, "TEM-1 beta-lactamase: opens when helices 11/12 separate"),
    ("1MAZ", "A", "2YXJ", "A", "N3C", None, "BCL-xL: BH3 groove (ABT-737)"),
    ("1Z1M", "A", "1RV1", "A", "IMZ:110", None, "MDM2: p53 cleft (nutlin)"),
    ("1V4T", "A", "1V4S", "A", "MRK", None, "Glucokinase: allosteric activator site"),
    ("2CM2", "A", "1T49", "A", "892", None, "PTP1B: allosteric site, needs helix a7 to unfold"),
    ("1WFC", "A", "3HVC", "A", "GG5:362", None, "p38a: C-lobe lipid pocket"),
    ("1DLO", "A,B", "1VRT", "A", "NVP", None, "HIV-1 RT: NNRTI pocket (Tyr181, Tyr188 rotate)"),
]

PREDICT = """
import json, sys
from crypsis.pipeline import predict
path, chains, keep, model, esm = sys.argv[1:6]
kw = {"esm": esm} if esm != "-" else {}
res = predict(path, chains=chains.split(","), keep_cofactors=set(keep.split(",")) if keep else None, model=model, **kw)
json.dump({"residues": res["residues"], "probability": [float(p) for p in res["probability"]],
           "sites": [s["residues"] for s in res["sites"]]}, sys.stdout)
"""


def _protein(path, chains, model=0):
    """[(chain, key, one-letter, heavy-atom coordinates)] of the protein residues, as crypsis.read_chain keys them."""
    import gemmi

    st = gemmi.read_structure(path)
    st.remove_hydrogens()
    out = []
    for ch in st[model]:
        if ch.name not in chains:
            continue
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            if r.het_flag == "H" or info is None or not info.is_amino_acid() or not r.find_atom("CA", "*"):
                continue
            out.append((ch.name, f"{ch.name}_{r.seqid.num}{r.seqid.icode.strip()}", info.one_letter_code.upper(),
                        np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r])))
    return out


def true_site(pdb_dir, apo, apo_chains, holo, holo_chain, ligand, cutoff=4.5):
    """Apo residue keys lining the ligand in the holo structure, sequence-aligned to the apo chain of the same name as
    the holo chain (else the first apo chain: an apo heterodimer's other chain can share the sequence, as HIV-1 RT's
    p51 shares p66's)."""
    import gemmi

    st = gemmi.read_structure(os.path.join(pdb_dir, f"{holo}.cif.gz"))
    st.remove_hydrogens()
    name, _, num = ligand.partition(":")
    lig = np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] if ch.name == holo_chain for r in ch
                    if r.name == name and (not num or r.seqid.num == int(num)) for a in r])
    if not len(lig):
        raise ValueError(f"{ligand} not found in {holo} chain {holo_chain}")
    hres = _protein(os.path.join(pdb_dir, f"{holo}.cif.gz"), [holo_chain])
    lining = {i for i, (_c, _k, _s, xyz) in enumerate(hres)
              if (np.linalg.norm(xyz[:, None, :] - lig[None, :, :], axis=2) <= cutoff).any()}
    site = set()
    for c in [holo_chain if holo_chain in apo_chains else apo_chains[0]]:
        ares = _protein(os.path.join(pdb_dir, f"{apo}.cif.gz"), [c])
        a_seq, h_seq = "".join(r[2] for r in ares), "".join(r[2] for r in hres)
        al = gemmi.align_string_sequences(list(a_seq), list(h_seq), [])
        if al.calculate_identity() < 60.0:
            continue
        i = j = 0
        for op, length in _cigar(al.cigar_str()):
            for _ in range(length):
                if op == "M":
                    if j in lining:
                        site.add(ares[i][1])
                    i, j = i + 1, j + 1
                elif op == "I":
                    i += 1
                else:
                    j += 1
    return site


def _cigar(s):
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            yield ch, int(num)
            num = ""


def found(site, true):
    shared = len(set(site) & true)
    return shared >= 3 and (shared >= 0.5 * len(site) or shared >= len(true) / 3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
                    help="the crypsis source tree to run (default: this one)")
    ap.add_argument("--models", default="physics,hybrid")
    ap.add_argument("--esm", default="650M", help="hybrid language model; '-' for versions without the option")
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.abspath(a.src))
    rows = []
    for apo, chains, holo, hchain, lig, keep, desc in CASES:
        true = true_site(a.pdb, apo, chains.split(","), holo, hchain, lig)
        row = {"apo": apo, "holo": holo, "site": desc, "true_site_residues": len(true)}
        for model in a.models.split(","):
            cache = os.path.join(a.out, f"{apo}_{model}.json")
            if not os.path.exists(cache):
                res = subprocess.run([sys.executable, "-c", PREDICT, os.path.join(a.pdb, f"{apo}.cif.gz"), chains,
                                      keep or "", model, a.esm], env=env, capture_output=True, text=True, check=True)
                with open(cache, "w") as fh:
                    fh.write(res.stdout)
            res = json.load(open(cache))
            y = np.array([k in true for k in res["residues"]])
            rank = next((k for k, s in enumerate(res["sites"], 1) if found(s, true)), None)
            row[model] = {"auc": round(float(roc_auc_score(y, res["probability"])), 2), "site_rank": rank}
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(a.out, "cases.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Crypsis 0.3 hybrid on the kinase domain of every PDB chain of the selected kinases.

  python run_chains.py chain DATA GENE PDB CHAIN   # one chain in a fresh process: sequence, then prediction
  python run_chains.py emb   DATA                  # ESM-2 650M embeddings of the collected sequences
gemmi 0.7.5 segfaulted reading some files after many reads in one process, so every chain runs in its own process
(run_all.sh) and the C-alpha coordinates gemmi returns are checked against the mmCIF table.

The whole deposited chain is analysed (cropping the domain would expose surfaces that the juxtamembrane segment or
the C-terminal tail cover); residues are mapped to UniProt numbering with the SIFTS columns of the PDBe updated mmCIF,
and chains with fewer than MIN_RES observed kinase-domain residues are skipped.  Ligand footprints: non-polymer
groups (>= 6 heavy atoms, not water or a crystallization additive) with any heavy atom within 4 A of the chain.
Interface: the chain's residues within 4 A of other protein chains of the deposited model (no symmetry mates).
fpocket 4.0 runs on the same chain (fpocket_chains.py)."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

if os.environ.get("CRYPSIS_SRC"):           # another Crypsis version's source tree (e.g. a v0.2.1 worktree)
    sys.path.insert(0, os.environ["CRYPSIS_SRC"])
import crypsis  # noqa: E402
from crypsis import core, pipeline  # noqa: E402

MIN_RES = 180
MAX_RESOLUTION = 3.5
FPOCKET = os.environ.get("FPOCKET", "fpocket")             # fpocket 4.0 binary
# gemmi 0.7.5 read_structure segfaults on this entry inside the pipeline process, even in a fresh process
EXCLUDED = {"6p7g"}


TOKEN = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def atom_site(path, columns):
    """Columns of the mmCIF _atom_site loop, parsed in Python (gemmi 0.7.5 get_mmcif_category and cif.Table rows
    corrupted memory on some files, and a later read_structure segfaulted).  Missing values ('?', '.') -> None."""
    names, rows = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                names.append(line.split()[0][len("_atom_site."):])
            elif names:
                if line.startswith(("#", "loop_", "_")):
                    break
                tok = [t[1:-1] if t[0] in "'\"" else t for t in TOKEN.findall(line)]
                if tok:
                    if len(tok) != len(names):
                        raise ValueError(f"_atom_site row with {len(tok)} values for {len(names)} columns")
                    rows.append(tok)
    if not names:
        raise ValueError("no _atom_site loop")
    idx = [names.index(c) for c in columns]
    return [[None if r[i] in ("?", ".") else r[i] for r in rows] for i in idx]


def sifts_map(path, chain, acc, ca=None):
    """(chain_resnum_icode key) -> UniProt number, for residues SIFTS maps to ``acc``.  ``ca``: a dict to fill with
    the first C-alpha coordinates of each residue of the chain, read from the mmCIF table."""
    cols = atom_site(path, ["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code", "pdbx_sifts_xref_db_acc",
                            "pdbx_sifts_xref_db_num", "pdbx_PDB_model_num", "label_atom_id", "Cartn_x", "Cartn_y",
                            "Cartn_z"])
    out, model0 = {}, None
    for r in zip(*cols):
        if model0 is None:
            model0 = r[5]
        if r[5] != model0 or r[0] != chain:
            continue
        key = f"{chain}_{r[1]}{r[2] or ''}"
        if ca is not None and r[6] == "CA" and key not in ca:
            ca[key] = (float(r[7]), float(r[8]), float(r[9]))
        if r[3] != acc or r[4] is None:
            continue
        out[key] = int(r[4])
    return out


def domain_chain(path, chain_id, info):
    """The whole chain and the UniProt number of each residue (None where SIFTS maps none)."""
    acc, (d0, d1) = info["accession"], info["domain"]
    ca = {}
    smap = sifts_map(path, chain_id, acc, ca)
    ch = core.read_chain(path, chains=[chain_id])
    ref = np.array([ca.get(k, (np.nan,) * 3) for k in ch.res_keys])
    if not np.nanmax(np.abs(ch.xyz[ch.ca_index] - ref)) < 1e-3 or np.isnan(ref).any():
        raise RuntimeError("C-alpha coordinates read by gemmi differ from the mmCIF table")
    nums = [smap.get(k) for k in ch.res_keys]
    n_dom = sum(1 for n in nums if n is not None and d0 <= n <= d1)
    if n_dom < MIN_RES:
        raise ValueError(f"only {n_dom} kinase-domain residues")
    return ch, nums


def ligand_footprints(path, sub, nums):
    import gemmi
    from scipy.spatial import cKDTree

    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    tree = cKDTree(sub.xyz)
    out = []
    for ch in st[0]:
        for res in ch:
            if res.het_flag != "H" or res.name in core.WATER_NAMES or res.name in core.ADDITIVES:
                continue
            xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
            if len(xyz) < 6:
                continue
            hit = {int(sub.atom_res[j]) for js in tree.query_ball_point(xyz, 4.0) for j in js}
            if hit:
                out.append({"name": res.name, "chain": ch.name, "num": res.seqid.num, "heavy_atoms": len(xyz),
                            "residues": sorted(nums[i] for i in hit if nums[i] is not None)})
    return out


def interface(path, chain_id, smap):
    """UniProt numbers of the chain's residues within 4 A of other protein chains, and those chains' accessions."""
    import gemmi

    accs = {}
    for ch, ac in zip(*atom_site(path, ["auth_asym_id", "pdbx_sifts_xref_db_acc"])):
        if ac is not None:
            accs.setdefault(ch, ac)
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    model = st[0]
    ns = gemmi.NeighborSearch(model, st.cell, 5).populate()
    hit, partners = set(), set()
    for res in model[chain_id]:
        for a in res:
            for m in ns.find_atoms(a.pos, "\0", radius=4.0):
                cra = m.to_cra(model)
                if cra.chain.name != chain_id:
                    hit.add(f"{chain_id}_{res.seqid.num}{res.seqid.icode.strip()}")
                    partners.add(cra.chain.name)
    return sorted({smap[k] for k in hit if k in smap}), {c: accs.get(c, "?") for c in sorted(partners)}


def seq_key(sub):
    return hashlib.sha1("|".join(pipeline.sequences(sub)).encode()).hexdigest()[:16]


def jobs(data):
    meta = json.loads((data / "meta/kinases.json").read_text())
    for gene, info in meta.items():
        for c in info["chains"]:
            yield gene, info, c


def run_emb(data):
    """Embeddings for the sequences written by the chain step (no structure files are read here)."""
    done = 0
    for f in sorted((data / "seqs").glob("*/*.json")):
        r = json.loads(f.read_text())
        out = data / f"emb/{r['key']}.npy"
        if not out.exists():
            np.save(out, pipeline.esm_embeddings(r["sequences"], size="650M").astype(np.float16))
            done += 1
            if done % 25 == 0:
                print(f"{done} embeddings", flush=True)
    print(f"{done} new embeddings")


def one(data, gene, info, c):
    out = data / f"pred/{gene}/{c['pdb_id']}_{c['chain_id']}.json"
    if out.exists():
        return "cached"
    path = data / f"raw_cif/{c['pdb_id']}_updated.cif"
    sub, nums = domain_chain(path, c["chain_id"], info)
    emb = np.load(data / f"emb/{seq_key(sub)}.npy").astype(np.float32)
    t0 = time.time()
    r = pipeline.predict(sub, model="hybrid", embeddings=emb, esm="650M")
    key2num = dict(zip(sub.res_keys, nums))
    iface, partners = interface(path, c["chain_id"], {k: n for k, n in key2num.items() if n is not None})
    seq = info["sequence"]
    one_letter = pipeline.ONE_LETTER
    mutations = [f"{seq[n - 1]}{n}{one_letter.get(nm, 'X')}" for n, nm in zip(nums, sub.res_names)
                 if n is not None and one_letter.get(nm, "X") not in (seq[n - 1], "X")]
    sites = []
    for s in r["sites"]:
        s = dict(s)
        s["uniprot"] = [key2num[k] for k in s["residues"]]
        sites.append(s)
    rec = {"gene": gene, "pdb_id": c["pdb_id"], "chain": c["chain_id"], "method": c["experimental_method"],
           "resolution": c["resolution"], "uniprot": nums, "residue_names": sub.res_names,
           "probability": np.round(r["probability"], 4).tolist(),
           "open_fraction": np.round(r["open_fraction"], 3).tolist(),
           "static_score": np.round(r["static_score"], 3).tolist(),
           "sites": sites, "ligands": ligand_footprints(path, sub, nums), "mutations": mutations,
           "interface": iface, "interface_partners": partners,
           "seconds": round(time.time() - t0, 1), "crypsis_version": crypsis.__version__}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec))
    return f"{rec['seconds']} s"


def run_chain(data, gene, pdb, chain_id):
    meta = json.loads((data / "meta/kinases.json").read_text())
    info = meta[gene]
    c = next(c for c in info["chains"] if c["pdb_id"] == pdb and c["chain_id"] == chain_id)
    seq_f = data / f"seqs/{gene}/{pdb}_{chain_id}.json"
    if not seq_f.exists():
        sub, _ = domain_chain(data / f"raw_cif/{pdb}_updated.cif", chain_id, info)
        seq_f.parent.mkdir(parents=True, exist_ok=True)
        seq_f.write_text(json.dumps({"key": seq_key(sub), "sequences": pipeline.sequences(sub)}))
    key = json.loads(seq_f.read_text())["key"]
    if not (data / f"emb/{key}.npy").exists():
        return "needs embedding"
    msg = one(data, gene, info, c)
    sys.path.insert(0, str(Path(__file__).parent))   # python -I leaves the script's directory out
    import fpocket_chains

    fpocket_chains.one(data, FPOCKET, gene, info, pdb, chain_id)
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["chain", "emb", "jobs"])
    ap.add_argument("data", type=Path)
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.step == "emb":
        run_emb(a.data)
    elif a.step == "jobs":                 # one line per chain to run: GENE PDB CHAIN
        for g, i, c in jobs(a.data):
            if c["pdb_id"] in EXCLUDED:
                continue
            if c["experimental_method"] != "Solution NMR" and (c["resolution"] or 99) <= MAX_RESOLUTION:
                print(g, c["pdb_id"], c["chain_id"])
    else:
        gene, pdb, chain_id = a.args
        try:
            print(gene, pdb, chain_id, run_chain(a.data, gene, pdb, chain_id), flush=True)
        except Exception as e:
            print(gene, pdb, chain_id, "ERROR", f"{type(e).__name__}: {e}", flush=True)

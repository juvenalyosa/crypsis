"""fpocket 4.0 on the whole chain (protein atoms only, ligands and waters removed) of every chain predicted by
run_chains.py; pockets are stored with their lining residues in UniProt numbering.  Called by run_chains.py, one
chain per process."""
import json
import re
import subprocess
import tempfile
from pathlib import Path

import gemmi

from run_chains import sifts_map


def chain_pdb(path, chain_id, acc, out):
    smap = sifts_map(path, chain_id, acc)
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    for m in range(len(st) - 1, 0, -1):
        del st[m]
    model = st[0]
    for ci in range(len(model) - 1, -1, -1):
        if model[ci].name != chain_id:
            del model[ci]
    model[0].name = "A"                 # PDB format: one-character chain IDs (parse() keys by chain_id)
    st.setup_entities()
    st.write_pdb(str(out))
    return smap


def parse(outdir, name, smap, chain_id):
    info = (outdir / f"{name}_info.txt").read_text()
    pockets = []
    for block in re.split(r"\nPocket ", "\n" + info)[1:]:
        k = int(block.split(":")[0])
        vals = dict((m.group(1).strip(), float(m.group(2))) for m in re.finditer(r"\t([^:\n]+):\s*([-\d.]+)", block))
        res = set()
        for line in (outdir / "pockets" / f"pocket{k}_atm.pdb").read_text().splitlines():
            if line.startswith("ATOM"):
                key = f"{chain_id}_{line[22:26].strip()}{line[26].strip()}"
                if key in smap:
                    res.add(smap[key])
        pockets.append({"pocket": k, "score": vals.get("Score"), "druggability": vals.get("Druggability Score"),
                        "volume": vals.get("Volume"), "hydrophobicity": vals.get("Hydrophobicity score"),
                        "n_spheres": vals.get("Number of Alpha Spheres"), "residues": sorted(res)})
    return pockets


def one(data, fpocket, gene, info, pdb, chain_id):
    out = data / f"fpocket/{gene}/{pdb}_{chain_id}.json"
    if out.exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / f"{pdb}_{chain_id}.pdb"
        smap = chain_pdb(data / f"raw_cif/{pdb}_updated.cif", chain_id, info["accession"], f)
        subprocess.run([fpocket, "-f", str(f)], check=True, capture_output=True, cwd=tmp)
        pockets = parse(Path(tmp) / f"{pdb}_{chain_id}_out", f"{pdb}_{chain_id}", smap, chain_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pockets))

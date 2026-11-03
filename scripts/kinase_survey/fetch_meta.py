"""Kinase-domain boundaries and ATP-binding residues (UniProt) and PDB chains (PDBe SIFTS best_structures)."""
import json
import sys
import urllib.request
from pathlib import Path

KINASES = {"EGFR": "P00533", "ALK": "Q9UM73", "HER2": "P04626", "ROS1": "P08922",
           "MET": "P08581", "RET": "P07949", "ABL1": "P00519", "BRAF": "P15056"}
out = Path(sys.argv[1])


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


meta = {}
for gene, acc in KINASES.items():
    u = get(f"https://rest.uniprot.org/uniprotkb/{acc}.json")
    dom = [f for f in u["features"] if f["type"] == "Domain" and "kinase" in f.get("description", "").lower()]
    atp = [f for f in u["features"] if f["type"] == "Binding site" and "ATP" in json.dumps(f.get("ligand", {}))]
    atp_res = sorted({i for f in atp for i in range(f["location"]["start"]["value"], f["location"]["end"]["value"] + 1)})
    best = get(f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{acc}")[acc]
    d0, d1 = dom[0]["location"]["start"]["value"], dom[0]["location"]["end"]["value"]
    chains = [b for b in best if b["unp_end"] >= d0 + 50 and b["unp_start"] <= d1 - 50]
    meta[gene] = {"accession": acc, "sequence": u["sequence"]["value"], "domain": [d0, d1],
                  "domain_description": dom[0]["description"], "atp_binding": atp_res, "chains": chains}
    print(gene, acc, f"domain {d0}-{d1}", "ATP res", len(atp_res), "chains", len(chains),
          "entries", len({c["pdb_id"] for c in chains}))
(out / "kinases.json").write_text(json.dumps(meta, indent=1))

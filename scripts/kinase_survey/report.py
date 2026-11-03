"""Tables from results/consensus_sites.json: every consensus site called in >= MIN_CALLED of the chains that resolve
it, with its class.

  python report.py DATA"""
import json
import sys
import urllib.request
from pathlib import Path

MIN_CALLED = 0.05


def classify(c):
    if c["atp_site"] >= 0.5:
        return "ATP pocket"
    if c["ligand_entries"] > 0:
        return "ligand observed"
    if c["interface_share"] >= 0.3:
        return "protein interface"
    return "candidate (no ligand in PDB)"


def chem_names(codes, cache_f):
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    for code in codes:
        if code not in cache:
            try:
                with urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}", timeout=30) as r:
                    cache[code] = str(json.load(r)["chem_comp"].get("name", ""))[:80]
            except Exception:
                cache[code] = ""
    cache_f.write_text(json.dumps(cache, indent=1))
    return cache


def main(data):
    res = json.loads((data / "results/consensus_sites.json").read_text())
    rows = []
    for gene, g in res.items():
        for c in g["clusters"]:
            if c["called"] >= MIN_CALLED:
                rows.append(dict(c, cls=classify(c)))
    codes = {name for r in rows for name, _ in r["ligands_top"][:4]}
    names = chem_names(sorted(codes), data / "meta/chem_names.json")
    cols = ["gene", "cls", "called", "top3", "covered_chains", "covered_entries", "kind_forms",
            "fpocket_druggable_share", "fpocket_ds_median", "fpocket_ds_max", "ds_max_entry", "ligand_entries",
            "ligands", "interface_share", "atp_site", "core"]
    lines = ["\t".join(cols)]
    for r in rows:
        ds_max = r["fpocket_ds_max"] or [None, "", ""]
        lines.append("\t".join(str(x) for x in [
            r["gene"], r["cls"], r["called"], r["top3"], r["covered_chains"], r["covered_entries"],
            r["kind"].get("forms with motion", 0.0), r["fpocket_druggable_share"], r["fpocket_ds_median"],
            ds_max[0], f"{ds_max[1]}_{ds_max[2]}", r["ligand_entries"],
            "; ".join(f"{n} ({names.get(n, '')}) x{k}" for n, k in r["ligands_top"][:4]),
            r["interface_share"], r["atp_site"], " ".join(r["core_labels"])]))
    (data / "results/consensus_sites.tsv").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} sites called in >= {MIN_CALLED:.0%} of chains")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

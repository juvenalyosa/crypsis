"""Writing results: residue table, sites, a structure coloured by probability and a PyMOL script."""

import csv
import json
import os


def write_csv(result, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        site_of = {r: s["rank"] for s in result["sites"] for r in s["residues"]}
        w.writerow(["residue", "name", "probability", "open_fraction", "site"])
        for k, n, p, o in zip(result["residues"], result["residue_names"], result["probability"],
                              result["open_fraction"]):
            w.writerow([k, n, f"{p:.4f}", f"{o:.3f}", site_of.get(k, "")])


def write_sites(result, path):
    with open(path, "w") as fh:
        json.dump({"model": result["model"], "sites": result["sites"]}, fh, indent=1)


def write_structure(result, source, path):
    """The input structure with each residue's B-factor set to 100 x its probability (other atoms: 0)."""
    import gemmi

    prob = dict(zip(result["residues"], result["probability"]))
    st = gemmi.read_structure(str(source))
    for ch in st[0]:
        for res in ch:
            v = 100.0 * float(prob.get(f"{ch.name}_{res.seqid.num}{res.seqid.icode.strip()}", 0.0))
            for a in res:
                a.b_iso = v
    st.setup_entities()
    st.write_pdb(str(path))


def write_pymol(result, pdb_path, path, top=3):
    name = os.path.splitext(os.path.basename(pdb_path))[0]
    lines = [f"load {os.path.basename(pdb_path)}, {name}", "hide everything", f"show cartoon, {name}",
             f"spectrum b, white_red, {name}, minimum=0, maximum=100"]
    colours = ["tv_orange", "marine", "forest", "violet", "cyan"]
    for s in result["sites"][:top]:
        sel = " or ".join(f"(chain {r.split('_')[0]} and resi {r.split('_')[1]})" for r in s["residues"])
        lines += [f"select site{s['rank']}, {name} and ({sel})", f"show sticks, site{s['rank']} and not name N+C+O",
                  f"color {colours[(s['rank'] - 1) % len(colours)]}, site{s['rank']} and elem C"]
    lines += ["deselect", f"orient {name}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

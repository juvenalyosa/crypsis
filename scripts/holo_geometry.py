#!/usr/bin/env python3
"""Where each holo ligand sits on the apo structure, for a site state that does not use Crypsis.  Added after
review (both held-out sets).

    python scripts/holo_geometry.py --set cryptobench     # holo mmCIF in <data>/holo_cif/<id>.cif.gz
    python scripts/holo_geometry.py --set temporal        # holo mmCIF in ~/crypsis_data/temporal/raw/<id>.cif.gz

For every holo record of every scored apo entity (dataset.json; the records whose apo pocket lies on the entity's
chains, as in cryptobench.pockets_of), the holo structure is superposed on the apo structure by the Calpha atoms of
the paired pocket residues (apo_pocket_selection / holo_pocket_selection, in order; Kabsch).  The ligand's heavy
atoms (first alternate location) are then placed on the apo entity, read as Crypsis reads it (protein heavy atoms
of the scored chains, core.read_chain), and the script records:

  fit_share_2.5   share of ligand heavy atoms with no apo heavy atom within 2.5 A (also at 2.0 and 3.0 A)
  centroid        ligand centroid in the apo frame (for distance-based site criteria)
  pocket_rmsd     Calpha RMSD of the paired pocket residues after the superposition
  ligand_altloc / pocket_altloc   whether the holo ligand or a holo pocket residue has alternate locations
  covalent        whether a ligand heavy atom lies within 2 A of a polymer heavy atom of the holo structure
  branched        whether the ligand is a residue of a branched (oligosaccharide) entity

Pockets of fewer than 3 residues are superposed on the apo residues within 10 A of them (paired by residue-number
offset).  Per protein, the record marked is_main_holo_structure (or, when it fails, the record of largest pocket
RMSD) gives the ligand-based site state: closed when less than a
third of the ligand fits in the apo structure, open when more than two thirds fit, partly open otherwise (the same
cut-offs as the earlier, Crypsis-based state).  Results: <work>/holo_geometry.json.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cryptobench as cb  # noqa: E402
import heldout_eval as he  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from crypsis import core  # noqa: E402

HOLO = {"cryptobench": os.path.join(he.SETS["cryptobench"][0], "holo_cif"),
        "temporal": os.path.expanduser("~/crypsis_data/temporal/raw")}
CUTS = (2.0, 2.5, 3.0)


def kabsch(p, q):
    """Rotation r and translation t with p @ r.T + t ~ q (least squares)."""
    pc, qc = p.mean(0), q.mean(0)
    u, _s, vt = np.linalg.svd((p - pc).T @ (q - qc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, qc - pc @ r.T


def _key(chain, res):
    return f"{chain}_{res.seqid.num}{res.seqid.icode.strip()}"


def read_holo(path):
    """Calpha positions by residue key, heavy atoms of every residue by (chain, name, number), altloc flags."""
    import gemmi

    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_hydrogens()
    ca, het, alt, poly = {}, {}, set(), []
    for ch in st[0]:
        for res in ch:
            k = _key(ch.name, res)
            if res.entity_type == gemmi.EntityType.Polymer:
                poly.extend((a.pos.x, a.pos.y, a.pos.z) for a in res)
            if any(a.altloc != "\0" for a in res):
                alt.add(k)
            seen, xyz = set(), []
            for a in res:
                if a.name not in seen:                     # first alternate location
                    seen.add(a.name)
                    xyz.append((a.pos.x, a.pos.y, a.pos.z))
                    if a.name == "CA":
                        ca.setdefault(k, (a.pos.x, a.pos.y, a.pos.z))
            het.setdefault((ch.name, res.name, f"{res.seqid.num}{res.seqid.icode.strip()}"), []).extend(xyz)
            if res.entity_type == gemmi.EntityType.Branched:
                alt.add(("branched", ch.name, res.name, f"{res.seqid.num}{res.seqid.icode.strip()}"))
    return ca, het, alt, np.array(poly)


def _extend(pairs, apo_ca, hca, radius=10.0):
    """For pockets of fewer than 3 residues: add the apo residues whose Calpha lies within `radius` of a pocket
    Calpha, paired with the holo residue at the same residue-number offset as the pocket's first pair."""
    (a0, b0) = pairs[0]
    ca_, na = a0.split("_")
    cb_, nb = b0.split("_")
    try:
        off = int(nb) - int(na)
    except ValueError:
        return pairs
    centre = np.array([apo_ca[a] for a, _b in pairs])
    out = list(pairs)
    for k, x in apo_ca.items():
        c, num = k.split("_")
        if c != ca_ or not num.lstrip("-").isdigit() or any(k == a for a, _b in out):
            continue
        if np.min(np.linalg.norm(centre - x, axis=1)) <= radius and f"{cb_}_{int(num) + off}" in hca:
            out.append((k, f"{cb_}_{int(num) + off}"))
    return out


def _one(job):
    from scipy.spatial import cKDTree

    apo, chain_id, data, holo_dir, records = job
    try:
        ch = core.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain_id.split("-"))
    except Exception as exc:
        return apo, chain_id, [], f"apo: {type(exc).__name__}: {exc}"
    apo_ca = {k: ch.xyz[i] for k, i in zip(ch.res_keys, ch.ca_index)}
    tree = cKDTree(ch.xyz)
    chains = set(chain_id.split("-"))
    out, cache = [], {}
    for n, h in enumerate(records):
        rec = {"record": n, "holo": h["holo_pdb_id"], "ligand": h["ligand"], "main": bool(h["is_main_holo_structure"]),
               "pRMSD": h["pRMSD"]}
        if not cb._pocket(h, chains):
            continue
        try:
            if h["holo_pdb_id"] not in cache:
                cache[h["holo_pdb_id"]] = read_holo(os.path.join(holo_dir, f"{h['holo_pdb_id']}.cif.gz"))
            hca, het, alt, poly = cache[h["holo_pdb_id"]]
            pairs = [(a, b) for a, b in zip(h["apo_pocket_selection"], h["holo_pocket_selection"])
                     if a.split("_")[0] in chains and a in apo_ca and b in hca]
            lig = het.get((h["ligand_chain"], h["ligand"], str(h["ligand_index"])))
            if 0 < len(pairs) < 3:
                pairs, rec["superposition"] = _extend(pairs, apo_ca, hca), "extended"
            if len(pairs) < 3 or not lig:
                rec["error"] = "fewer than 3 paired Calpha atoms" if len(pairs) < 3 else "ligand not found"
                out.append(rec)
                continue
            rec["branched"] = ("branched", h["ligand_chain"], h["ligand"], str(h["ligand_index"])) in alt
            if len(poly):                                  # a polymer atom within 2 A: covalently attached
                rec["covalent"] = bool(cKDTree(poly).query(np.asarray(lig))[0].min() < 2.0)
            p = np.array([hca[b] for _a, b in pairs])
            q = np.array([apo_ca[a] for a, _b in pairs])
            r, t = kabsch(p, q)
            lig = np.asarray(lig) @ r.T + t
            dmin, _ = tree.query(lig)
            rec.update({f"fit_share_{c}": round(float((dmin >= c).mean()), 4) for c in CUTS})
            rec.update({f"fit_atoms_{c}": int((dmin >= c).sum()) for c in CUTS})     # exact, for the state
            rec.update(centroid=[round(float(v), 3) for v in lig.mean(0)], ligand_atoms=int(len(lig)),
                       pocket_rmsd=round(float(np.sqrt((((p @ r.T + t) - q) ** 2).sum(1).mean())), 3),
                       pocket_pairs=len(pairs),
                       ligand_altloc=f"{h['ligand_chain']}_{h['ligand_index']}" in alt,
                       pocket_altloc=any(b in alt for _a, b in pairs))
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    return apo, chain_id, out, None


def fit_share(rec, cut="2.5"):
    """Share of the ligand's heavy atoms that fit, from the exact counts (the stored share is rounded)."""
    return rec[f"fit_atoms_{cut}"] / rec["ligand_atoms"]


def state_of(fit):
    """Closed below one third, open above two thirds (compared exactly: 3 * atoms fitting against all atoms)."""
    return "closed" if 3 * fit < 1 - 1e-12 else ("open" if 3 * fit > 2 + 1e-12 else "partly_open")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(he.SETS), required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    data, work = he.SETS[a.set]
    ds = cb._dataset(data)
    jobs = [(apo, ch, data, HOLO[a.set], ds[apo]) for apo, ch, _pos in cb.entries(data, "test")]
    res = {}
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for apo, ch, recs, err in ex.map(_one, jobs):
            main_ = [r for r in recs if r["main"] and "error" not in r] or \
                sorted((r for r in recs if "error" not in r), key=lambda r: -r["pRMSD"])   # main record failed
            res[f"{apo}{ch}"] = {"apo": apo, "records": recs, "error": err,
                                 "state": state_of(fit_share(main_[0])) if main_ else None}
    n_rec = sum(len(v["records"]) for v in res.values())
    bad = sum("error" in r for v in res.values() for r in v["records"])
    states = [v["state"] for v in res.values()]
    summary = {"entities": len(res), "records": n_rec, "records_failed": bad,
               "state_counts": {s: states.count(s) for s in ("closed", "partly_open", "open", None)}}
    print(json.dumps(summary))
    with open(os.path.join(work, "holo_geometry.json"), "w") as fh:
        json.dump({"summary": summary, "entities": res}, fh, indent=1)


if __name__ == "__main__":
    main()

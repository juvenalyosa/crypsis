"""Consensus Crypsis sites across all PDB chains of each kinase.

Sites inside the kinase domain (>= half of their residues in it), ranked 1-5 among the domain's sites, are clustered
by the Jaccard distance of their UniProt residue sets (average linkage, cut at 0.7).  A cluster's core is the residues
present in >= 40% of its members.  Per cluster, over the chains that resolve >= 70% of the core:
  called     share of chains with a member site (top 5); top1/top3 likewise
  pct        median over chains of the mean within-chain percentile of the core residues' probability
  fpocket    best fpocket pocket that covers >= 50% of the core and has >= 25% of its residues in the core: share
             of chains with Druggability Score >= 0.5, median, max
  ligands    entries with a non-polymer ligand contacting >= max(3, 50%) of the core residues
  atp_site   share of the core in the ATP-analogue footprint (residues contacted in >= 20% of the chains that bind
             ATP, ADP or a non-hydrolysable analogue)
  interface  share of chains where >= 30% of the core is within 4 A of another protein chain
  ATP        core residues among UniProt's ATP-binding residues

  python aggregate.py DATA"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

TOP = 5
ATP_LIKE = {"ATP", "ADP", "ANP", "ACP", "AGS", "APC", "AMP"}


def main(data):
    meta = json.loads((data / "meta/kinases.json").read_text())
    out = {}
    for gene, info in meta.items():
        d0, d1 = info["domain"]
        seq, atp = info["sequence"], set(info["atp_binding"])
        recs = [json.loads(p.read_text()) for p in sorted((data / "pred" / gene).glob("*.json"))]
        if not recs:
            continue
        sites, chains = [], []
        for ci, r in enumerate(recs):
            nums = r["uniprot"]
            prob = np.asarray(r["probability"])
            pct = prob.argsort().argsort() / max(len(prob) - 1, 1)
            fp_f = data / f"fpocket/{gene}/{r['pdb_id']}_{r['chain']}.json"
            fp = json.loads(fp_f.read_text()) if fp_f.exists() else None
            chains.append({"rec": r, "pct": dict((n, p) for n, p in zip(nums, pct) if n is not None),
                           "fpocket": fp, "iface": set(r["interface"]), "partners": r["interface_partners"]})
            k = 0
            for s in r["sites"]:
                dom = [n for n in s["uniprot"] if n is not None and d0 <= n <= d1]
                if len(dom) < 0.5 * len(s["uniprot"]) or len(dom) < 4:
                    continue
                k += 1
                if k > TOP:
                    break
                sites.append({"chain": ci, "rank": k, "res": frozenset(dom), "kind": s["kind"],
                              "v0": s["pocket_volume_structure_A3"], "ve": s["pocket_volume_ensemble_A3"]})
        atp_chains = [c for c in chains if any(l["name"] in ATP_LIKE for l in c["rec"]["ligands"])]
        atp_cnt = Counter(x for c in atp_chains for x in {x for l in c["rec"]["ligands"] if l["name"] in ATP_LIKE
                                                          for x in l["residues"]})
        atp_site = {x for x, k in atp_cnt.items() if k >= 0.2 * len(atp_chains)} or atp
        n = len(sites)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sites[i]["res"], sites[j]["res"]
                D[i, j] = D[j, i] = 1 - len(a & b) / len(a | b)
        labels = fcluster(linkage(squareform(D), "average"), 0.7, "distance")
        clusters = []
        for lab in np.unique(labels):
            mem = [sites[i] for i in np.flatnonzero(labels == lab)]
            cnt = Counter(x for m in mem for x in m["res"])
            core = sorted(x for x, c in cnt.items() if c >= 0.4 * len(mem))
            if len(core) < 4:
                continue
            cs = set(core)
            covered = [ci for ci, c in enumerate(chains) if len(cs & set(c["pct"])) >= 0.7 * len(cs)]
            if len(covered) < 3:
                continue
            best_rank = {}
            for m in mem:
                best_rank[m["chain"]] = min(best_rank.get(m["chain"], 99), m["rank"])
            cov = set(covered)
            called = [ci for ci in best_rank if ci in cov]
            pcts = [np.mean([chains[ci]["pct"][x] for x in cs if x in chains[ci]["pct"]]) for ci in covered]
            ds, ds_where = [], []
            for ci in covered:
                fp = chains[ci]["fpocket"]
                if fp is None:
                    continue
                best = max([p["druggability"] for p in fp
                            if len(cs & set(p["residues"])) >= 0.5 * len(cs)
                            and len(cs & set(p["residues"])) >= 0.25 * len(p["residues"])], default=0.0)
                ds.append(best)
                ds_where.append((best, chains[ci]["rec"]["pdb_id"], chains[ci]["rec"]["chain"]))
            lig_entries, lig_names = set(), Counter()
            for ci in covered:
                r = chains[ci]["rec"]
                names = {l["name"] for l in r["ligands"] if len(cs & set(l["residues"])) >= max(3, 0.5 * len(cs))}
                if names:
                    lig_entries.add(r["pdb_id"])
                    lig_names.update(names)
            iface = [ci for ci in covered if len(cs & chains[ci]["iface"]) >= 0.3 * len(cs)]
            partner_accs = Counter(a for ci in iface for a in set(chains[ci]["partners"].values()))
            entries = {chains[ci]["rec"]["pdb_id"] for ci in covered}
            kinds = Counter(m["kind"] for m in mem)
            clusters.append({
                "gene": gene, "core": core, "core_labels": [f"{seq[x - 1]}{x}" for x in core],
                "n_members": len(mem), "covered_chains": len(covered), "covered_entries": len(entries),
                "called": round(len(called) / len(covered), 3),
                "top1": round(sum(best_rank[ci] == 1 for ci in called) / len(covered), 3),
                "top3": round(sum(best_rank[ci] <= 3 for ci in called) / len(covered), 3),
                "pct_median": round(float(np.median(pcts)), 3),
                "kind": {k: round(v / len(mem), 2) for k, v in kinds.items()},
                "volume_structure_median": float(np.median([m["v0"] for m in mem])),
                "volume_ensemble_median": float(np.median([m["ve"] for m in mem])),
                "fpocket_druggable_share": round(float(np.mean([d >= 0.5 for d in ds])), 3) if ds else None,
                "fpocket_ds_median": round(float(np.median(ds)), 3) if ds else None,
                "fpocket_ds_max": max(ds_where) if ds_where else None,
                "ligand_entries": len(lig_entries), "ligand_share": round(len(lig_entries) / len(entries), 3),
                "ligands_top": lig_names.most_common(8),
                "interface_share": round(len(iface) / len(covered), 3),
                "interface_partners": partner_accs.most_common(4),
                "atp_overlap": len(cs & atp), "atp_site": round(len(cs & atp_site) / len(cs), 2),
                "called_entries": sorted({chains[ci]["rec"]["pdb_id"] for ci in called}),
                "top3_examples": sorted((best_rank[ci], chains[ci]["rec"]["pdb_id"], chains[ci]["rec"]["chain"])
                                        for ci in called if best_rank[ci] <= 3)[:12],
            })
        clusters.sort(key=lambda c: -c["called"])
        out[gene] = {"atp_site": sorted(atp_site), "atp_analogue_chains": len(atp_chains), "chains": len(chains), "entries": len({c["rec"]["pdb_id"] for c in chains}),
                     "sites": n, "clusters": clusters}
        print(f"{gene}: {len(chains)} chains, {out[gene]['entries']} entries, {n} sites, {len(clusters)} clusters")
    (data / "results").mkdir(exist_ok=True)
    (data / "results/consensus_sites.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

#!/usr/bin/env python3
"""Temporal hold-out set of cryptic ligand-binding sites, built with CryptoBench's criteria (Skrhak et al.,
Bioinformatics 41(1) btae745) from PDB entries whose *holo* structure was released on or after 2024-06-01
(CryptoBench's newest holo structure: 2024-05-01; newest apo: 2024-05-08).

Output, in <out> (default ~/crypsis_data/temporal):
    dataset.json        CryptoBench's format (apo id -> list of holo records) plus provenance fields
    cif/<apo>.cif.gz    the apo mmCIF files, laid out like CryptoBench's cif/ folder
    raw/<id>.cif.gz     every downloaded mmCIF (holo and apo candidates); cache
    meta/               cached RCSB query results (search hits, GraphQL entry metadata, cluster groups)
    pairs.jsonl         every apo-holo pocket pair evaluated, with all metrics (before filtering)
    stats.json          counts after every step (used by REPORT.md)

Steps, each cached on disk so later steps can be rerun alone:

    python scripts/temporal_set.py validate   # reproduce CryptoBench's pRMSD on random pairs (picks the RMSD)
    python scripts/temporal_set.py apocheck   # what CryptoBench's apo pockets contain (defines "apo")
    python scripts/temporal_set.py cbref      # CryptoBench apo chains: 30% groups and sequences
    python scripts/temporal_set.py search     # new holo entries + all same-UniProt entries (apo candidates)
    python scripts/temporal_set.py download   # mmCIF files of the candidates
    python scripts/temporal_set.py pairs      # every apo-holo pocket pair with its metrics -> pairs.jsonl
    python scripts/temporal_set.py select     # filters, 40% clustering, representatives, extra pockets, output

Definitions (see REPORT.md for the evidence behind each choice):
  * Residue correspondence: UniProt numbering from RCSB's SIFTS-derived entity alignments
    (GraphQL polymer_entity.rcsb_polymer_entity_align, entity label_seq_id -> UniProt position).
  * Binding residues: protein residues of the holo structure (any chain) with a heavy atom within 4.5 A of a
    heavy atom of the ligand residue; the apo pocket is the same UniProt positions in the apo chain(s); every one
    must be observed (equal pocket lengths).
  * Ligand: one non-polymer (or branched-entity) residue with >= 5 observed heavy atoms, not in CryptoBench's
    ignore list (HOH DOD WAT UNK ABA MPD GOL SO4 PO4) and not in BioLiP's artifact list (AHoJ-DB, from which
    CryptoBench was built, only contains BioLiP2 "biologically relevant" interactions).
  * Apo: no hetero group at all (any non-water non-polymer or branched residue, ions and additives included) with a
    heavy atom within 4.5 A of a heavy atom of the apo pocket residues (99.7% of CryptoBench's apo pockets satisfy
    this; see `apocheck`).
  * pRMSD: all-atom RMSD over the matched atoms (same atom names; hydrogens included when both structures have
    them) of the pocket residues after optimal superposition of those same atoms.  `validate` selected this
    definition: it reproduces CryptoBench's pRMSD within 0.05 A for 117/120 random records.
  * Novelty: no RCSB 30% sequence-identity group (rcsb_polymer_entity_group_membership) shared with a CryptoBench
    apo chain, checked for the pair's entities and for every single-accession PDB entity of the same UniProt
    accession; the accession itself must not be a CryptoBench accession.  Max global identity to CryptoBench's apo
    chains (and its shuffled-sequence chance level) is recorded for subset analyses.
  * TM-score: fixed (UniProt) residue correspondence, Calpha, iterative search for the superposition maximising
    TM, d0 = 1.24 (L-15)^(1/3) - 1.8 with L = number of observed residues of the apo chain.
  * Centre shift: distance between the centroids of the pocket's matched heavy atoms after the TM superposition.
  * Rg change: |Rg(apo pocket) - Rg(holo pocket)| / Rg(holo pocket), matched heavy atoms.

Downloaded files are only parsed with gemmi.
"""

import argparse
import collections
import concurrent.futures as cf
import datetime
import gzip
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request

import numpy as np

CUTOFF_DATE = "2024-06-01"
MAX_RES = 2.5
METHODS = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY")
CONTACT = 4.5
MIN_LIGAND_HEAVY = 5
IGNORED = {"HOH", "DOD", "WAT", "UNK", "ABA", "MPD", "GOL", "SO4", "PO4"}
MIN_PRMSD = 2.0
MIN_TM = 0.5
MAX_CENTER_SHIFT = 4.0
MAX_RG_CHANGE = 0.20
MIN_OBSERVED = 50
MAX_APO_LEN = 1500
CLUSTER_ID = 0.40
MIN_REF_LEN = 30                 # CryptoBench reference chains shorter than this (peptides) are ignored for identity
PRMSD_METHOD = "pocket_fit"      # chosen by `validate` (see REPORT.md); implemented inline in evaluate_job

HOME = os.path.expanduser("~")
DEFAULT_OUT = os.path.join(HOME, "crypsis_data", "temporal")
DEFAULT_CB = os.path.join(HOME, "crypsis_data", "cryptobench")
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
FILE_URL = "https://files.rcsb.org/download/{}.cif.gz"
BIOLIP_URL = "https://zhanggroup.org/BioLiP/ligand_list"

AMINO = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE SEC PYL".split())


# ------------------------------------------------------------------------------------------------ RCSB access

def _post(url, payload, tries=5):
    data = json.dumps(payload).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                                                  "User-Agent": "crypsis-temporal-set"})
            with urllib.request.urlopen(req, timeout=300) as fh:
                body = fh.read()
            return json.loads(body) if body else {}
        except Exception as exc:                                     # noqa: BLE001 (network retry)
            if k == tries - 1:
                raise
            print(f"  retry {url} ({exc})", file=sys.stderr)
            time.sleep(5 * (k + 1))


def rcsb_search(nodes, return_type="entry"):
    """All identifiers matching the AND of the given terminal nodes."""
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes}, "return_type": return_type,
         "request_options": {"return_all_hits": True, "results_content_type": ["experimental"]}}
    r = _post(SEARCH_URL, q)
    return [h["identifier"] for h in r.get("result_set", [])]


def term(attr, op, value):
    return {"type": "terminal", "service": "text", "parameters": {"attribute": attr, "operator": op, "value": value}}


QUALITY_NODES = [term("rcsb_entry_info.resolution_combined", "less_or_equal", MAX_RES),
                 term("exptl.method", "in", list(METHODS))]

ENTRY_QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id
 rcsb_accession_info{initial_release_date}
 rcsb_entry_info{resolution_combined}
 exptl{method}
 polymer_entities{
   rcsb_polymer_entity_container_identifiers{entity_id asym_ids auth_asym_ids uniprot_ids}
   rcsb_polymer_entity_align{reference_database_name reference_database_accession
                             aligned_regions{entity_beg_seq_id ref_beg_seq_id length}}
   rcsb_polymer_entity_group_membership{group_id aggregation_method similarity_cutoff}
   entity_poly{rcsb_entity_polymer_type pdbx_seq_one_letter_code_can}}
 nonpolymer_entities{nonpolymer_comp{chem_comp{id} rcsb_chem_comp_info{atom_count_heavy}}}
 branched_entities{rcsb_branched_entity_container_identifiers{entity_id}}
}}"""


def _compact(e):
    """Keep what we need of one GraphQL entry."""
    ents = {}
    for p in e.get("polymer_entities") or []:
        ids = p["rcsb_polymer_entity_container_identifiers"]
        groups = {str(int(g["similarity_cutoff"])): g["group_id"] for g in p.get("rcsb_polymer_entity_group_membership")
                  or [] if g["aggregation_method"] == "sequence_identity"}
        align = []
        for a in p.get("rcsb_polymer_entity_align") or []:
            if a["reference_database_name"] == "UniProt":
                for reg in a["aligned_regions"] or []:
                    align.append((a["reference_database_accession"], reg["entity_beg_seq_id"],
                                  reg["ref_beg_seq_id"], reg["length"]))
        ep = p.get("entity_poly") or {}
        ents[ids["entity_id"]] = {"auth": ids.get("auth_asym_ids") or [], "uniprot": ids.get("uniprot_ids") or [],
                                  "align": align, "groups": groups, "type": ep.get("rcsb_entity_polymer_type"),
                                  "seq": (ep.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")}
    ligs = []
    for n in e.get("nonpolymer_entities") or []:
        c = n["nonpolymer_comp"]
        ligs.append((c["chem_comp"]["id"], (c.get("rcsb_chem_comp_info") or {}).get("atom_count_heavy") or 0))
    res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
    return {"release": e["rcsb_accession_info"]["initial_release_date"][:10], "resolution": res[0],
            "method": [m["method"] for m in e.get("exptl") or []], "entities": ents, "ligands": ligs,
            "branched": len(e.get("branched_entities") or [])}


def entry_meta(ids, cache_path, batch=100):
    """GraphQL metadata of entries, cached in one JSON file (id lower-case -> compact record)."""
    meta = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = sorted({i.lower() for i in ids} - set(meta))
    for k in range(0, len(todo), batch):
        r = _post(GRAPHQL_URL, {"query": ENTRY_QUERY, "variables": {"ids": [i.upper() for i in todo[k:k + batch]]}})
        for e in (r.get("data") or {}).get("entries") or []:
            if e:
                meta[e["rcsb_id"].lower()] = _compact(e)
        for i in todo[k:k + batch]:            # obsolete / missing ids: remember them as None
            meta.setdefault(i, None)
        if (k // batch) % 20 == 19:
            json.dump(meta, open(cache_path, "w"))
            print(f"  metadata {k + batch}/{len(todo)}", file=sys.stderr)
    json.dump(meta, open(cache_path, "w"))
    return meta


def download(ids, folder, workers=4):
    """mmCIF files (gzipped) into folder/<id>.cif.gz; at most `workers` concurrent requests."""
    os.makedirs(folder, exist_ok=True)

    def one(i):
        path = os.path.join(folder, f"{i}.cif.gz")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return i, True
        for k in range(4):
            try:
                req = urllib.request.Request(FILE_URL.format(i.upper()), headers={"User-Agent": "crypsis-temporal-set"})
                with urllib.request.urlopen(req, timeout=300) as fh:
                    body = fh.read()
                gzip.decompress(body)                                      # must be valid gzip
                with open(path + ".part", "wb") as out:
                    out.write(body)
                os.replace(path + ".part", path)
                return i, True
            except Exception:                                          # noqa: BLE001 (network retry)
                time.sleep(3 * (k + 1))
        return i, False

    ok, failed = 0, []
    with cf.ThreadPoolExecutor(workers) as ex:
        for n, (i, good) in enumerate(ex.map(one, sorted(set(ids)))):
            ok += good
            if not good:
                failed.append(i)
            if n % 500 == 499:
                print(f"  downloaded {n + 1}/{len(set(ids))}", file=sys.stderr)
    return ok, failed


def biolip_artifacts(out):
    path = os.path.join(out, "meta", "biolip_artifacts.txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(BIOLIP_URL, headers={"User-Agent": "crypsis-temporal-set"})
        with urllib.request.urlopen(req, timeout=120) as fh:
            open(path, "wb").write(fh.read())
    return {line.split("\t")[0].strip() for line in open(path) if line.strip()}


# ------------------------------------------------------------------------------------------------ structures

class _LazyChains(dict):
    """auth chain -> residue data, parsed on first access (large EM assemblies have dozens of unused chains)."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        if v is None:
            v = self._owner._parse_chain(k)
            dict.__setitem__(self, k, v)
        return v

    def get(self, k, default=None):
        return self[k] if k in self else default

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


class Structure:
    """Protein residues (per auth chain) with heavy atoms, hydrogens and UniProt positions, and hetero groups."""

    def __init__(self, path, meta=None):
        import gemmi
        st = gemmi.read_structure(path)
        st.setup_entities()
        st.remove_alternative_conformations()      # first conformer; hydrogens are kept for the pRMSD (below)
        self._st, self._model = st, st[0]
        self._ents = (meta or {}).get("entities", {})
        self.chains = _LazyChains(self)   # auth chain -> dict(keys, names, entity, unp, atoms, hatoms, ca, index)
        self.hetero = []                  # (chain, name, seqnum+icode, heavy-atom coords)
        for ch in self._model:
            protein = False
            for res in ch:
                et = res.entity_type
                if et == gemmi.EntityType.Polymer:
                    protein = protein or _is_protein_residue(res.name)
                elif et in (gemmi.EntityType.NonPolymer, gemmi.EntityType.Branched) and not res.is_water():
                    self.hetero.append((ch.name, res.name, f"{res.seqid.num}{res.seqid.icode.strip()}",
                                        np.array([a.pos.tolist() for a in res if not a.is_hydrogen()]).reshape(-1, 3)))
            if protein and ch.name not in self.chains:
                dict.__setitem__(self.chains, ch.name, None)
        self._ns = None

    def _parse_chain(self, name):
        import gemmi
        keys, names, unp, atoms, hatoms, ca = [], [], [], [], [], []
        ent_id = None
        for res in self._model[name]:
            if res.entity_type != gemmi.EntityType.Polymer or not _is_protein_residue(res.name):
                continue
            ent_id = res.entity_id
            keys.append(f"{name}_{res.seqid.num}{res.seqid.icode.strip()}")
            names.append(res.name)
            unp.append(_unp(self._ents.get(res.entity_id), res.label_seq))
            atoms.append({a.name: np.array(a.pos.tolist()) for a in res if not a.is_hydrogen()})
            hatoms.append({a.name: np.array(a.pos.tolist()) for a in res if a.is_hydrogen()})
            c = res.find_atom("CA", "*")
            ca.append(np.array(c.pos.tolist()) if c else None)
        return {"keys": keys, "names": names, "entity": ent_id, "unp": unp, "atoms": atoms, "hatoms": hatoms,
                "ca": ca, "index": {k: i for i, k in enumerate(keys)}}

    def near_residues(self, coords, cutoff=CONTACT):
        """{(chain, residue index)} of protein residues with a heavy atom within `cutoff` of any of `coords`
        (asymmetric unit only: no crystal-symmetry images)."""
        import gemmi
        if self._ns is None:
            self._ns = gemmi.NeighborSearch(self._model, gemmi.UnitCell(), 6).populate(include_h=False)
        hit = set()
        for p in coords:
            pos = gemmi.Position(*p)
            for mk in self._ns.find_atoms(pos, "\0", radius=cutoff):
                cra = mk.to_cra(self._model)
                if cra.residue.entity_type != gemmi.EntityType.Polymer or cra.chain.name not in self.chains \
                        or cra.atom.is_hydrogen() or cra.atom.pos.dist(pos) > cutoff:
                    continue
                key = f"{cra.chain.name}_{cra.residue.seqid.num}{cra.residue.seqid.icode.strip()}"
                i = self.chains[cra.chain.name]["index"].get(key)
                if i is not None:
                    hit.add((cra.chain.name, i))
        return hit

    def unp_index(self, chain):
        """UniProt (acc, pos) -> residue index of a chain (first occurrence)."""
        d = self.chains[chain]
        if "unp_index" not in d:
            d["unp_index"] = {}
            for i, u in enumerate(d["unp"]):
                if u is not None:
                    d["unp_index"].setdefault(u, i)
        return d["unp_index"]


def _is_protein_residue(name):
    import gemmi
    if name in AMINO:
        return True
    tab = gemmi.find_tabulated_residue(name)
    return not (tab is not None and tab.is_nucleic_acid())


def _unp(ent, label_seq):
    if not ent or label_seq is None:
        return None
    for acc, ebeg, rbeg, length in ent["align"]:
        if ebeg <= label_seq < ebeg + length:
            return (acc, rbeg + label_seq - ebeg)
    return None


def qualifying_ligand(name, n_heavy, artifacts):
    return n_heavy >= MIN_LIGAND_HEAVY and name not in IGNORED and name not in artifacts


# ------------------------------------------------------------------------------------------------ geometry

def kabsch(x, y):
    """R, t minimising |x R^T + t - y|."""
    cx, cy = x.mean(0), y.mean(0)
    h = (x - cx).T @ (y - cy)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cy - cx @ r.T


def apply(rt, x):
    return x @ rt[0].T + rt[1]


def rmsd(x, y):
    return float(np.sqrt(((x - y) ** 2).sum(1).mean()))


def reject_fit(x, y, cutoff=2.0, cycles=5):
    """PyMOL-style outlier rejection: refit after dropping pairs deviating by more than cutoff x RMSD."""
    idx = np.arange(len(x))
    rt = kabsch(x, y)
    for _ in range(cycles):
        d = np.linalg.norm(apply(rt, x[idx]) - y[idx], axis=1)
        keep = d <= cutoff * np.sqrt((d ** 2).mean())
        if keep.all() or keep.sum() < 3:
            break
        idx = idx[keep]
        rt = kabsch(x[idx], y[idx])
    return rt


def d0_of(length):
    return max(0.5, 1.24 * np.cbrt(length - 15) - 1.8) if length > 21 else 0.5


def tm_score(x, y, length):
    """TM-score of x onto y with a fixed residue correspondence, normalised by `length` (TM-score program's
    heuristic search: seed superpositions on fragments of length n, n/2, n/4 ..., then iterate on the pairs
    closer than d0_search).  Returns (tm, (R, t))."""
    n = len(x)
    d0 = d0_of(length)
    d_search = min(max(d0, 4.5), 8.0)
    best, best_rt = -1.0, None
    frag = n
    frags = []
    while frag >= 4:
        frags.append(frag)
        if frag == 4:
            break
        frag = max(4, frag // 2)
    if n < 4:
        frags = [n]
    for frag in frags:
        step = max(1, (n - frag) // 12)
        for start in range(0, n - frag + 1, step):
            idx = np.arange(start, start + frag)
            seen = set()
            for _ in range(30):
                if len(idx) < 3:
                    break
                rt = kabsch(x[idx], y[idx])
                d = np.linalg.norm(apply(rt, x) - y, axis=1)
                tm = float((1.0 / (1.0 + (d / d0) ** 2)).sum() / length)
                if tm > best:
                    best, best_rt = tm, rt
                cut = d_search
                new = np.where(d < cut)[0]
                while len(new) < 3 and cut < 20:
                    cut += 0.5
                    new = np.where(d < cut)[0]
                key = tuple(new)
                if key in seen:
                    break
                seen.add(key)
                idx = new
    return best, best_rt


def radius_of_gyration(x):
    return float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()))


# ------------------------------------------------------------------------------------------------ pair metrics

def matched_atoms(apo, holo, pairs, hydrogens=False):
    """Matched atoms (same atom name) of paired residues: (apo xyz, holo xyz, atom names).  With hydrogens=True
    hydrogen atoms present in both structures are included (this is what reproduces CryptoBench's pRMSD)."""
    xa, xh, names = [], [], []
    for (ca, ia), (ch, ih) in pairs:
        kinds = ("atoms", "hatoms") if hydrogens else ("atoms",)
        for kind in kinds:
            aa, ah = apo.chains[ca][kind][ia], holo.chains[ch][kind][ih]
            for name in ah:
                if name in aa:
                    xa.append(aa[name])
                    xh.append(ah[name])
                    names.append(name)
    return np.array(xa).reshape(-1, 3), np.array(xh).reshape(-1, 3), names


def chain_correspondence(apo, ca, holo, ch):
    """Residue-index pairs of two chains sharing UniProt positions, both with a Calpha."""
    hi = holo.unp_index(ch)
    out = []
    for i, u in enumerate(apo.chains[ca]["unp"]):
        if u is not None and u in hi and apo.chains[ca]["ca"][i] is not None \
                and holo.chains[ch]["ca"][hi[u]] is not None:
            out.append((i, hi[u]))
    return out


def all_rmsds(apo, holo, pocket_pairs, chain_map):
    """Pocket RMSD under every candidate superposition.  pocket_pairs: [((apo chain, idx), (holo chain, idx))];
    chain_map: {holo chain: apo chain} for the global superpositions."""
    xa, xh, names = matched_atoms(apo, holo, pocket_pairs)
    if len(xa) < 3:
        return None
    gx, gy, gxa, gya = [], [], [], []
    for ch, ca in chain_map.items():
        for ia, ih in chain_correspondence(apo, ca, holo, ch):
            gx.append(apo.chains[ca]["ca"][ia])
            gy.append(holo.chains[ch]["ca"][ih])
            aa, ah = apo.chains[ca]["atoms"][ia], holo.chains[ch]["atoms"][ih]
            for name in ah:
                if name in aa:
                    gxa.append(aa[name])
                    gya.append(ah[name])
    gx, gy, gxa, gya = (np.array(v).reshape(-1, 3) for v in (gx, gy, gxa, gya))
    out = {"none": rmsd(xa, xh)}
    if len(gx) >= 3:
        out["global_ca"] = rmsd(apply(kabsch(gx, gy), xa), xh)
        out["global_ca_reject"] = rmsd(apply(reject_fit(gx, gy), xa), xh)
        out["global_all"] = rmsd(apply(kabsch(gxa, gya), xa), xh)
        out["global_all_reject"] = rmsd(apply(reject_fit(gxa, gya), xa), xh)
        L = len([c for c in apo.chains[next(iter(chain_map.values()))]["ca"] if c is not None])
        _, rt = tm_score(gx, gy, max(L, len(gx)))
        out["global_tm"] = rmsd(apply(rt, xa), xh)
    out["pocket_fit_heavy"] = rmsd(apply(kabsch(xa, xh), xa), xh)
    ya, yh, _ = matched_atoms(apo, holo, pocket_pairs, hydrogens=True)
    out["pocket_fit"] = rmsd(apply(kabsch(ya, yh), ya), yh)
    pa = np.array([apo.chains[c]["ca"][i] for (c, i), _ in pocket_pairs if apo.chains[c]["ca"][i] is not None
                   and holo.chains[_[0]]["ca"][_[1]] is not None]).reshape(-1, 3)
    ph = np.array([holo.chains[h]["ca"][j] for (c, i), (h, j) in pocket_pairs if apo.chains[c]["ca"][i] is not None
                   and holo.chains[h]["ca"][j] is not None]).reshape(-1, 3)
    if len(pa) >= 3:
        out["pocket_ca_fit"] = rmsd(apply(kabsch(pa, ph), xa), xh)
    bb = np.array([n in ("N", "CA", "C", "O") for n in names])
    if bb.sum() >= 3:
        out["pocket_bb_fit"] = rmsd(apply(kabsch(xa[bb], xh[bb]), xa), xh)
    return out


# ------------------------------------------------------------------------------------------------ validate

def cmd_validate(a):
    """Reproduce CryptoBench's pRMSD with each candidate superposition on a random sample of its records."""
    out = a.out
    d = json.load(open(os.path.join(a.cryptobench, "cryptobench-dataset", "dataset.json")))
    recs = [(apo, r) for apo, v in d.items() for r in v]
    rng = random.Random(a.seed)
    sample = rng.sample(recs, a.n)
    ids = {apo for apo, _ in sample} | {r["holo_pdb_id"] for _, r in sample}
    meta = entry_meta(ids, os.path.join(out, "meta", "validate_meta.json"))
    raw = os.path.join(out, "raw")
    download([r["holo_pdb_id"] for _, r in sample], raw)
    rows = []
    for apo_id, r in sample:
        try:
            apo = Structure(os.path.join(a.cryptobench, "cif", f"{apo_id}.cif.gz"), meta.get(apo_id))
            holo = Structure(os.path.join(raw, f"{r['holo_pdb_id']}.cif.gz"), meta.get(r["holo_pdb_id"]))
        except Exception as exc:                                   # noqa: BLE001
            print(apo_id, r["holo_pdb_id"], "unreadable", exc)
            continue
        pairs, unp_agree, missing = [], 0, 0
        for sa, sh in zip(r["apo_pocket_selection"], r["holo_pocket_selection"]):
            ca, ch = sa.split("_")[0], sh.split("_")[0]
            ia = apo.chains.get(ca, {}).get("index", {}).get(sa)
            ih = holo.chains.get(ch, {}).get("index", {}).get(sh)
            if ia is None or ih is None:
                missing += 1
                continue
            pairs.append(((ca, ia), (ch, ih)))
            unp_agree += apo.chains[ca]["unp"][ia] is not None and \
                apo.chains[ca]["unp"][ia] == holo.chains[ch]["unp"][ih]
        if missing or not pairs:
            print(apo_id, r["holo_pdb_id"], "pocket residues missing", missing)
            continue
        cmap = {}
        for (ca, _), (ch, _) in pairs:
            cmap.setdefault(ch, ca)
        res = all_rmsds(apo, holo, pairs, cmap)
        if res is None:
            continue
        rows.append({"apo": apo_id, "holo": r["holo_pdb_id"], "apo_chain": r["apo_chain"],
                     "holo_chain": r["holo_chain"], "n_res": len(pairs), "unp_agree": unp_agree,
                     "pRMSD": r["pRMSD"], **{k: round(v, 3) for k, v in res.items()}})
    methods = [k for k in rows[0] if k not in ("apo", "holo", "apo_chain", "holo_chain", "n_res", "unp_agree",
                                               "pRMSD")]
    ref = np.array([x["pRMSD"] for x in rows])
    summary = {}
    for m in methods:
        ok = [i for i, x in enumerate(rows) if m in x]
        v = np.array([rows[i][m] for i in ok])
        summary[m] = {"n": len(ok), "pearson": float(np.corrcoef(v, ref[ok])[0, 1]),
                      "mae": float(np.abs(v - ref[ok]).mean()), "median_ae": float(np.median(np.abs(v - ref[ok]))),
                      "within_0.05": float((np.abs(v - ref[ok]) <= 0.05).mean()),
                      "within_0.2": float((np.abs(v - ref[ok]) <= 0.2).mean()),
                      "below_2A": int((v < MIN_PRMSD).sum())}
    unp = sum(x["unp_agree"] for x in rows), sum(x["n_res"] for x in rows)
    json.dump({"rows": rows, "summary": summary, "unp_agreement": unp, "seed": a.seed,
               "date": datetime.date.today().isoformat()},
              open(os.path.join(out, "meta", "validate.json"), "w"), indent=1)
    print(f"{len(rows)} pairs; UniProt positions agree for {unp[0]}/{unp[1]} pocket residues")
    for m, s in sorted(summary.items(), key=lambda kv: kv[1]["mae"]):
        print(f"  {m:20s} n={s['n']:3d} r={s['pearson']:.3f} MAE={s['mae']:.3f} medAE={s['median_ae']:.3f} "
              f"|d|<=0.05: {s['within_0.05']:.2f}  |d|<=0.2: {s['within_0.2']:.2f}  <2A: {s['below_2A']}")


# ------------------------------------------------------------------------------------------------ apocheck

def cmd_apocheck(a):
    """What lies within 4.5 A of CryptoBench's apo pockets in the apo structures (all records, local files)."""
    artifacts = biolip_artifacts(a.out)
    d = json.load(open(os.path.join(a.cryptobench, "cryptobench-dataset", "dataset.json")))
    cnt, examples = collections.Counter(), collections.defaultdict(collections.Counter)
    for n, (apo_id, v) in enumerate(d.items()):
        try:
            st = Structure(os.path.join(a.cryptobench, "cif", f"{apo_id}.cif.gz"))
        except Exception:                                          # noqa: BLE001
            cnt["unreadable"] += len(v)
            continue
        from scipy.spatial import cKDTree
        for r in v:
            xyz = []
            for s in r["apo_pocket_selection"]:
                c = s.split("_")[0]
                i = st.chains.get(c, {}).get("index", {}).get(s)
                if i is not None:
                    xyz.extend(st.chains[c]["atoms"][i].values())
            if not xyz:
                cnt["no pocket atoms"] += 1
                continue
            tree = cKDTree(np.array(xyz))
            cats = set()
            for chn, name, num, coords in st.hetero:
                if not len(coords) or min(tree.query(coords)[0]) > CONTACT:
                    continue
                if len(coords) < MIN_LIGAND_HEAVY:
                    cat = "small (<5 heavy atoms, e.g. ions)"
                elif name in IGNORED:
                    cat = "CryptoBench ignore list"
                elif name in artifacts:
                    cat = "BioLiP artifact list"
                else:
                    cat = "qualifying ligand"
                cats.add(cat)
                examples[cat][name] += 1
            for c in cats:
                cnt[c] += 1
            cnt["records"] += 1
            if not cats:
                cnt["empty site"] += 1
            if cats == {"qualifying ligand"} or "qualifying ligand" in cats:
                cnt["any qualifying ligand"] += 1
        if n % 200 == 199:
            print(f"  {n + 1}/{len(d)}", file=sys.stderr)
    res = {"counts": dict(cnt), "examples": {k: v.most_common(25) for k, v in examples.items()}}
    json.dump(res, open(os.path.join(a.out, "meta", "apocheck.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ cbref

def cmd_cbref(a):
    """CryptoBench apo chains: RCSB 30% sequence-identity groups (polymer-entity group membership) and the
    SEQRES sequences of the chains, read from the local apo mmCIF files CryptoBench distributes."""
    import gemmi
    d = json.load(open(os.path.join(a.cryptobench, "cryptobench-dataset", "dataset.json")))
    meta = entry_meta(list(d), os.path.join(a.out, "meta", "cb_meta.json"))
    groups, seqs, missing = collections.defaultdict(set), {}, []
    for apo_id, v in d.items():
        chains = sorted({c for r in v for c in r["apo_chain"].split("-")})
        st = gemmi.read_structure(os.path.join(a.cryptobench, "cif", f"{apo_id}.cif.gz"))
        st.setup_entities()
        m = meta.get(apo_id)
        for c in chains:
            try:
                ent = st.get_entity_of(st[0][c].get_polymer())
            except Exception:                                          # noqa: BLE001
                ent = None
            if ent is not None:
                seqs[f"{apo_id}{c}"] = gemmi.one_letter_code([x.split(",")[0] for x in ent.full_sequence]).upper()
            found = False
            for eid, e in ((m or {}).get("entities") or {}).items():
                if c in e["auth"]:
                    found = True
                    for cut, g in e["groups"].items():
                        groups[cut].add(g)
            if not found:
                missing.append(f"{apo_id}{c}")
    ref = {"groups": {str(k): sorted(v) for k, v in groups.items()}, "seqs": seqs, "no_metadata": missing,
           "date": datetime.date.today().isoformat()}
    json.dump(ref, open(os.path.join(a.out, "meta", "cb_ref.json"), "w"))
    print(f"{len(seqs)} CryptoBench apo chain sequences; 30% groups: {len(groups[30])}; "
          f"chains without RCSB metadata: {len(missing)}")


# ------------------------------------------------------------------------------------------------ search

def eligible_entity(e, cb30):
    """A protein entity with a UniProt alignment whose 30% group is not shared with a CryptoBench apo chain."""
    return e["type"] == "Protein" and e["align"] and e["groups"].get("30") not in cb30


def cmd_search(a):
    """New holo entries (released >= CUTOFF_DATE, resolution <= 2.5 A, X-ray or EM, with a ligand), then all
    entries (any date, same quality) sharing a UniProt accession with one of their eligible protein entities."""
    m = os.path.join(a.out, "meta")
    artifacts = biolip_artifacts(a.out)
    ref = json.load(open(os.path.join(m, "cb_ref.json")))
    cb30 = set(ref["groups"]["30"])
    stats = {"query_date": datetime.date.today().isoformat()}
    path = os.path.join(m, "holo_hits.json")
    if not os.path.exists(path):
        lig = {"type": "group", "logical_operator": "or", "nodes": [
            term("rcsb_entry_info.nonpolymer_entity_count", "greater_or_equal", 1),
            term("rcsb_entry_info.branched_entity_count", "greater_or_equal", 1)]}
        hits = rcsb_search([term("rcsb_accession_info.initial_release_date", "greater_or_equal", CUTOFF_DATE),
                            term("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 1),
                            lig] + QUALITY_NODES)
        json.dump({"date": stats["query_date"], "ids": [h.lower() for h in hits]}, open(path, "w"))
    hits = json.load(open(path))
    stats["query_date"] = hits["date"]
    holo_ids = hits["ids"]
    stats["holo_entries_found"] = len(holo_ids)
    meta = entry_meta(holo_ids, os.path.join(m, "meta.json"))
    holo_ok, accs = [], collections.defaultdict(set)
    n_lig, n_novel = 0, 0
    for i in holo_ids:
        e = meta.get(i)
        if not e or e["release"] < CUTOFF_DATE:
            continue
        if not (any(qualifying_ligand(c, n, artifacts) for c, n in e["ligands"]) or e["branched"]):
            continue
        n_lig += 1
        ents = [x for x in e["entities"].values() if eligible_entity(x, cb30)]
        if not ents:
            continue
        n_novel += 1
        holo_ok.append(i)
        for x in ents:
            for acc in {r[0] for r in x["align"]}:
                accs[acc].add(i)
    stats["holo_entries_with_qualifying_ligand"] = n_lig
    stats["holo_entries_with_novel_protein_entity"] = n_novel
    stats["uniprot_accessions"] = len(accs)
    print(f"holo entries: {len(holo_ids)} found, {n_lig} with a qualifying ligand, {n_novel} with an entity "
          f"outside CryptoBench's 30% groups; {len(accs)} UniProt accessions")
    path = os.path.join(m, "apo_hits.json")
    found = json.load(open(path)) if os.path.exists(path) else {}
    todo = sorted(set(accs) - set(found))
    for k in range(0, len(todo), 100):
        batch = todo[k:k + 100]
        ents = rcsb_search([term("rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers."
                                 "database_accession", "in", batch),
                            term("rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers."
                                 "database_name", "exact_match", "UniProt")] + QUALITY_NODES,
                           return_type="polymer_entity")
        for acc in batch:
            found.setdefault(acc, [])
        tmp = collections.defaultdict(set)
        for eid in ents:
            tmp[eid.split("_")[0].lower()].add(eid)
        found["__entries__"] = sorted(set(found.get("__entries__", [])) | set(tmp))
        json.dump(found, open(path, "w"))
        print(f"  apo search {k + len(batch)}/{len(todo)}: {len(found['__entries__'])} entries", file=sys.stderr)
    all_entries = found.get("__entries__", [])
    meta = entry_meta(all_entries, os.path.join(m, "meta.json"))
    # apo candidates: entries (any date) with an eligible entity carrying the accession (entities in a CryptoBench
    # 30% group could only give excluded pairs, so they are not downloaded)
    by_acc = collections.defaultdict(set)
    for i in all_entries:
        e = meta.get(i)
        if not e or (e["resolution"] or 99) > MAX_RES or not set(e["method"]) & set(METHODS):
            continue
        for x in e["entities"].values():
            if eligible_entity(x, cb30):
                for acc in {r[0] for r in x["align"]}:
                    by_acc[acc].add(i)
    jobs = {}
    for acc, holos in accs.items():
        apos = by_acc.get(acc, set()) | holos
        if len(apos) > 1:
            jobs[acc] = {"holo": sorted(holos), "apo": sorted(apos)}
    json.dump(jobs, open(os.path.join(m, "jobs.json"), "w"))
    files = {i for j in jobs.values() for i in j["holo"] + j["apo"]}
    stats["uniprot_accessions_with_apo_candidates"] = len(jobs)
    stats["candidate_entries"] = len(all_entries)
    stats["files_to_download"] = len(files)
    stats["holo_entries_in_jobs"] = len({i for j in jobs.values() for i in j["holo"]})
    save_stats(a.out, "search", stats)
    print(json.dumps(stats, indent=1))


def save_stats(out, step, d):
    path = os.path.join(out, "stats.json")
    s = json.load(open(path)) if os.path.exists(path) else {}
    s[step] = d
    json.dump(s, open(path, "w"), indent=1)


def cmd_download(a):
    jobs = json.load(open(os.path.join(a.out, "meta", "jobs.json")))
    files = sorted({i for j in jobs.values() for i in j["holo"] + j["apo"]})
    ok, failed = download(files, os.path.join(a.out, "raw"), workers=min(a.workers, 5))
    print(f"{ok}/{len(files)} files; failed: {failed[:20]}")
    save_stats(a.out, "download", {"files": len(files), "ok": ok, "failed": failed})


# ------------------------------------------------------------------------------------------------ pairs

def holo_pockets(hid, st, meta, acc, artifacts):
    """Pockets of the qualifying ligands of a holo structure whose main chain carries UniProt `acc`.
    Returns (pockets, counts)."""
    cnt = collections.Counter()
    out = []
    for lch, lname, lnum, coords in st.hetero:
        if lname in IGNORED or lname in artifacts or len(coords) < MIN_LIGAND_HEAVY:
            continue
        near = st.near_residues(coords)
        if not near:
            continue
        cnt["ligands"] += 1
        per_chain = collections.Counter(c for c, _ in near)
        top = max(per_chain.values())
        tops = sorted(c for c, n in per_chain.items() if n == top)
        primary = lch if lch in tops else tops[0]
        res = sorted(near, key=lambda ci: (ci[0], ci[1]))
        unp = [st.chains[c]["unp"][i] for c, i in res]
        if any(u is None for u in unp):
            cnt["pocket not fully UniProt-mapped"] += 1
            continue
        paccs = {u[0] for (c, i), u in zip(res, unp) if c == primary}
        if len(paccs) != 1 or acc not in paccs:
            cnt["main chain is another protein"] += 1
            continue
        cnt["pockets"] += 1
        out.append({"holo": hid, "ligand": lname, "ligand_chain": lch, "ligand_index": lnum,
                    "ligand_heavy": len(coords), "primary": primary, "res": res, "unp": unp})
    return out, cnt


def _apo_ligand_tree(st, artifacts):
    """Heavy atoms of every non-water hetero group of the apo structure (ions and additives included): CryptoBench's
    apo pockets contain none (see `apocheck`), so AHoJ-DB's apo definition is this strict one."""
    from scipy.spatial import cKDTree
    pts = [c for _, name, _, c in st.hetero if len(c)]
    return cKDTree(np.vstack(pts)) if pts else None


class _LazyCa(dict):
    def __init__(self, st):
        super().__init__()
        self._st = st

    def __missing__(self, chain):
        v = self[chain] = _ca_by_unp(self._st, chain)
        return v


def _ca_by_unp(st, chain):
    d = st.chains[chain]
    return {u: ca for u, ca in zip(d["unp"], d["ca"]) if u is not None and ca is not None}


_WORKER = {}


def evaluate_job(args):
    """All apo-holo pocket pairs of one UniProt accession (cached as pairs/<acc>.json)."""
    acc, job, out, artifacts, loose = args
    path = os.path.join(out, "pairs", f"{acc}.json")
    if os.path.exists(path):
        return acc, "cached"
    if "meta" not in _WORKER:                     # loaded once per worker process
        _WORKER["meta"] = json.load(open(os.path.join(out, "meta", "meta_light.json")))
        _WORKER["cb30"] = set(json.load(open(os.path.join(out, "meta", "cb_ref.json")))["groups"]["30"])
    meta, cb30 = _WORKER["meta"], _WORKER["cb30"]
    raw = os.path.join(out, "raw")
    cnt = collections.Counter()
    t0 = time.time()
    # holo pockets, keeping only what the pairing needs
    pockets, holo_chains = [], {}
    for hid in job["holo"]:
        f = os.path.join(raw, f"{hid}.cif.gz")
        if not os.path.exists(f):
            cnt["holo file missing"] += 1
            continue
        try:
            st = Structure(f, meta.get(hid))
        except Exception:                                              # noqa: BLE001
            cnt["holo unreadable"] += 1
            continue
        if not st.hetero:
            continue
        ps, c = holo_pockets(hid, st, meta.get(hid), acc, artifacts)
        cnt.update({"holo " + k: v for k, v in c.items()})
        for p in ps:
            for c in {c for c, _ in p["res"]}:
                if (hid, c) not in holo_chains:
                    d = st.chains[c]
                    ent = meta[hid]["entities"].get(d["entity"], {})
                    holo_chains[(hid, c)] = {"ca": _ca_by_unp(st, c), "group30": ent.get("groups", {}).get("30"),
                                             "n_obs": sum(1 for u, x in zip(d["unp"], d["ca"])
                                                          if u is not None and x is not None)}
            p["atoms"] = [st.chains[c]["atoms"][i] for c, i in p["res"]]
            p["hatoms"] = [st.chains[c]["hatoms"][i] for c, i in p["res"]]
            p["keys"] = [st.chains[c]["keys"][i] for c, i in p["res"]]
            pockets.append(p)
    rows = []
    if not pockets:
        json.dump({"rows": rows, "counts": cnt, "seconds": time.time() - t0}, open(path, "w"))
        return acc, "no pockets"
    tm_cache = {}
    for aid in job["apo"]:
        f = os.path.join(raw, f"{aid}.cif.gz")
        if not os.path.exists(f):
            cnt["apo file missing"] += 1
            continue
        try:
            apo = Structure(f, meta.get(aid))
        except Exception:                                              # noqa: BLE001
            cnt["apo unreadable"] += 1
            continue
        tree = _apo_ligand_tree(apo, artifacts)
        chains_of = collections.defaultdict(set)          # accession -> apo auth chains (from the metadata)
        for x in (meta.get(aid) or {}).get("entities", {}).values():
            for u in {r[0] for r in x["align"]}:
                chains_of[u].update(x["auth"])
        acc_chains = [c for c in apo.chains if c in chains_of[acc]]
        cas = _LazyCa(apo)
        glob = {}
        for p in pockets:
            if p["holo"] == aid:
                continue
            hc = holo_chains[(p["holo"], p["primary"])]
            others = sorted({c for c, _ in p["res"]} - {p["primary"]})
            for ca in acc_chains:
                cnt["pairs"] += 1
                ui = apo.unp_index(ca)
                if not all(u in ui for (c, _), u in zip(p["res"], p["unp"]) if c == p["primary"]):
                    cnt["pair: pocket residue unobserved in apo"] += 1
                    continue
                # global Calpha superposition of the main chains (to assign the other chains of the pocket)
                gkey = (ca, p["holo"], p["primary"])
                if gkey not in glob:
                    common = [u for u in cas[ca] if u in hc["ca"]]
                    rt = kabsch(np.array([cas[ca][u] for u in common]), np.array([hc["ca"][u] for u in common])) \
                        if len(common) >= 3 else None
                    glob[gkey] = (common, rt)
                common, rt = glob[gkey]
                if rt is None:
                    cnt["pair: < 3 common residues"] += 1
                    continue
                cmap = {p["primary"]: ca}
                ok = True
                for hch in others:
                    need = [u for (c, _), u in zip(p["res"], p["unp"]) if c == hch]
                    hca = holo_chains[(p["holo"], hch)]["ca"]
                    best = None
                    hacc = {u[0] for (c, _), u in zip(p["res"], p["unp"]) if c == hch}
                    for c2 in apo.chains:
                        if c2 in cmap.values() or not any(c2 in chains_of[u] for u in hacc):
                            continue
                        u2 = apo.unp_index(c2)
                        if not all(u in u2 for u in need):
                            continue
                        sel = [u for u in need if u in cas[c2] and u in hca]
                        if not sel:
                            continue
                        dist = float(np.linalg.norm(apply(rt, np.array([cas[c2][u] for u in sel])) -
                                                    np.array([hca[u] for u in sel]), axis=1).mean())
                        if best is None or dist < best[0]:
                            best = (dist, c2)
                    if best is None or best[0] > 10.0:
                        ok = False
                        break
                    cmap[hch] = best[1]
                if not ok:
                    cnt["pair: other pocket chain missing in apo"] += 1
                    continue
                apo_res = []
                for (c, _), u in zip(p["res"], p["unp"]):
                    i = apo.unp_index(cmap[c]).get(u)
                    if i is None:
                        break
                    apo_res.append((cmap[c], i))
                if len(apo_res) != len(p["res"]):
                    cnt["pair: pocket residue unobserved in apo"] += 1
                    continue
                apo_atoms = [apo.chains[c]["atoms"][i] for c, i in apo_res]
                if tree is not None:
                    xyz = np.vstack([np.array(list(at.values())) for at in apo_atoms if at])
                    if (tree.query(xyz, distance_upper_bound=CONTACT)[0] <= CONTACT).any():
                        cnt["pair: apo site occupied by a hetero group"] += 1
                        continue
                xa, xh = [], []
                ya, yh = [], []
                for at_a, at_h, (c, i) in zip(apo_atoms, p["atoms"], apo_res):
                    for name, pos in at_h.items():
                        if name in at_a:
                            xa.append(at_a[name])
                            xh.append(pos)
                for (c, i), hh in zip(apo_res, p["hatoms"]):
                    ha = apo.chains[c]["hatoms"][i]
                    for name, pos in hh.items():
                        if name in ha:
                            ya.append(ha[name])
                            yh.append(pos)
                xa, xh = np.array(xa).reshape(-1, 3), np.array(xh).reshape(-1, 3)
                if len(xa) < 3:
                    cnt["pair: < 3 matched atoms"] += 1
                    continue
                cnt["complete apo pairs"] += 1
                za, zh = (np.vstack([xa, np.array(ya)]), np.vstack([xh, np.array(yh)])) if ya else (xa, xh)
                prmsd = rmsd(apply(kabsch(za, zh), za), zh)
                if prmsd < loose:
                    cnt["pair: pRMSD < %.1f" % loose] += 1
                    continue
                key = (aid, ca, p["holo"], p["primary"])
                if key not in tm_cache:
                    x = np.array([cas[ca][u] for u in common])
                    y = np.array([hc["ca"][u] for u in common])
                    n_apo = sum(1 for v in apo.chains[ca]["ca"] if v is not None)
                    tm, trt = tm_score(x, y, n_apo)
                    tm_cache[key] = (tm, trt)
                tm, trt = tm_cache[key]
                shift = float(np.linalg.norm(apply(trt, xa).mean(0) - xh.mean(0)))
                rg_h = radius_of_gyration(xh)
                rg_change = abs(radius_of_gyration(xa) - rg_h) / rg_h
                apo_chains = sorted(set(cmap.values()))
                ents_a = [meta[aid]["entities"].get(apo.chains[c]["entity"], {}) for c in apo_chains]
                rows.append({
                    "uniprot_id": acc, "holo_pdb_id": p["holo"], "holo_chain": "-".join(sorted(cmap)),
                    "apo_pdb_id": aid, "apo_chain": "-".join(apo_chains), "main_apo_chain": ca,
                    "main_holo_chain": p["primary"],
                    "ligand": p["ligand"], "ligand_index": p["ligand_index"], "ligand_chain": p["ligand_chain"],
                    "ligand_heavy_atoms": p["ligand_heavy"],
                    "apo_pocket_selection": [apo.chains[c]["keys"][i] for c, i in apo_res],
                    "holo_pocket_selection": p["keys"],
                    "pRMSD": round(prmsd, 2), "pRMSD_heavy": round(rmsd(apply(kabsch(xa, xh), xa), xh), 3),
                    "tm_score": round(tm, 4), "center_shift": round(shift, 3), "rg_change": round(rg_change, 4),
                    "n_obs_apo": len(cas[ca]), "n_obs_holo": hc["n_obs"],
                    "apo_length": sum(sum(1 for v in apo.chains[c]["ca"] if v is not None) for c in apo_chains),
                    "apo_group30": sorted({e.get("groups", {}).get("30") for e in ents_a} - {None}),
                    "holo_group30": sorted({holo_chains[(p["holo"], c)]["group30"] for c in cmap} - {None}),
                    "cb30_overlap": any(e.get("groups", {}).get("30") in cb30 for e in ents_a) or
                    any(holo_chains[(p["holo"], c)]["group30"] in cb30 for c in cmap),
                })
    json.dump({"rows": rows, "counts": cnt, "seconds": time.time() - t0}, open(path, "w"))
    return acc, f"{len(rows)} rows"


def cmd_pairs(a):
    m = os.path.join(a.out, "meta")
    jobs = json.load(open(os.path.join(m, "jobs.json")))
    full = json.load(open(os.path.join(m, "meta.json")))
    needed = {i for j in jobs.values() for i in j["holo"] + j["apo"]}
    light = {i: full[i] for i in needed if full.get(i)}
    for e in light.values():                       # JSON keys of the groups dict are strings
        for x in e["entities"].values():
            x["groups"] = {str(k): v for k, v in x["groups"].items()}
            x.pop("seq", None)
    json.dump(light, open(os.path.join(m, "meta_light.json"), "w"))
    os.makedirs(os.path.join(a.out, "pairs"), exist_ok=True)
    artifacts = biolip_artifacts(a.out)
    order = sorted(jobs, key=lambda k: -len(jobs[k]["apo"]) * len(jobs[k]["holo"]))
    todo = [(acc, jobs[acc], a.out, artifacts, a.loose) for acc in order]
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(evaluate_job, t): t[0] for t in todo}
        for n, f in enumerate(cf.as_completed(futs)):
            try:
                acc, msg = f.result()
            except Exception as exc:                                   # noqa: BLE001
                acc, msg = futs[f], f"FAILED {exc!r}"
            if n % 50 == 0 or "FAILED" in msg:
                print(f"  {n + 1}/{len(todo)} {acc}: {msg} ({time.time() - t0:.0f} s)", file=sys.stderr)
    rows, cnt = [], collections.Counter()
    for acc in jobs:
        path = os.path.join(a.out, "pairs", f"{acc}.json")
        if os.path.exists(path):
            r = json.load(open(path))
            rows += r["rows"]
            cnt.update(r["counts"])
    with open(os.path.join(a.out, "pairs.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    save_stats(a.out, "pairs", dict(cnt) | {"rows_pRMSD_ge_%.1f" % a.loose: len(rows)})
    print(json.dumps(dict(cnt), indent=1), len(rows))


# ------------------------------------------------------------------------------------------------ select

_ALIGNER = None


def identity(a, b):
    """Global identity (Needleman-Wunsch, BLOSUM62, gap open -11 / extend -1 as in BLAST, free end gaps): identical
    aligned columns divided by the length of the shorter sequence."""
    global _ALIGNER
    if _ALIGNER is None:
        from Bio import Align
        from Bio.Align import substitution_matrices
        al = Align.PairwiseAligner()
        al.mode = "global"
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
        al.open_gap_score, al.extend_gap_score = -11.0, -1.0
        al.end_gap_score = 0.0
        _ALIGNER = al
    a = re.sub("[^ACDEFGHIKLMNPQRSTVWY]", "X", a.upper())
    b = re.sub("[^ACDEFGHIKLMNPQRSTVWY]", "X", b.upper())
    if not a or not b:
        return 0.0
    aln = _ALIGNER.align(a, b)[0]
    same = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        same += sum(x == y and x != "X" for x, y in zip(a[s1:e1], b[s2:e2]))
    return same / min(len(a), len(b))


def _max_identity(args):
    """Maximum identity of `seq` to the reference chains (>= MIN_REF_LEN residues), and the same for a shuffled copy
    of `seq` (the chance level of a maximum over ~1,300 references)."""
    key, seq, refs = args
    refs = {k: v for k, v in refs.items() if len(v) >= MIN_REF_LEN}
    best, hit = 0.0, None
    for rid, rseq in refs.items():
        v = identity(seq, rseq)
        if v > best:
            best, hit = v, rid
    shuffled = list(seq)
    random.Random(seq).shuffle(shuffled)
    shuffled = "".join(shuffled)
    chance = max(identity(shuffled, rseq) for rseq in refs.values())
    return key, best, hit, chance


def _identity_pair(args):
    i, j, a, b = args
    return i, j, identity(a, b)


def pymol_selection(pdb_id, keys):
    by = collections.OrderedDict()
    for k in keys:
        c, r = k.split("_", 1)
        by.setdefault(c, []).append(r)
    return f"{pdb_id} and ( " + " or ".join(f"(chain {c} and resi {'+'.join(v)})" for c, v in by.items()) + " )"


def cmd_select(a):
    """Filters, novelty, 40% clustering, representatives, additional pockets; writes dataset.json and cif/."""
    m = os.path.join(a.out, "meta")
    meta = json.load(open(os.path.join(m, "meta.json")))
    ref = json.load(open(os.path.join(m, "cb_ref.json")))
    rows = [json.loads(line) for line in open(os.path.join(a.out, "pairs.jsonl"))]
    for r in rows:
        ea, eh = meta[r["apo_pdb_id"]], meta[r["holo_pdb_id"]]
        r.update(apo_release_date=ea["release"], holo_release_date=eh["release"], resolution_apo=ea["resolution"],
                 resolution_holo=eh["resolution"], method_apo=ea["method"][0], method_holo=eh["method"][0])

    # UniProt-level novelty: RCSB's 30% groups can split one protein (e.g. PTP1B entities fall in different groups),
    # so a protein also overlaps CryptoBench when its accession is a CryptoBench accession, or when any
    # single-accession PDB entity of that accession (all entries returned by the search) is in the 30% group of a
    # CryptoBench apo chain.
    cb30 = set(ref["groups"]["30"])
    cbm = json.load(open(os.path.join(m, "cb_meta.json")))
    cbd = json.load(open(os.path.join(a.cryptobench, "cryptobench-dataset", "dataset.json")))
    cb_accs = {x["uniprot_id"] for v in cbd.values() for x in v}
    for apo_id, v in cbd.items():
        chains = {c for x in v for c in x["apo_chain"].split("-")}
        for e in ((cbm.get(apo_id) or {}).get("entities") or {}).values():
            if chains & set(e["auth"]):
                cb_accs.update(e["uniprot"])
    acc_groups = collections.defaultdict(set)
    for e in meta.values():
        for x in (e or {}).get("entities", {}).values():
            accs_x = {t[0] for t in x["align"]}
            if len(accs_x) == 1 and x["groups"].get("30"):
                acc_groups[accs_x.pop()].add(x["groups"]["30"])

    def chain_accs(entry, chains):
        out = set()
        for x in meta[entry]["entities"].values():
            if set(chains) & set(x["auth"]):
                out |= {t[0] for t in x["align"]}
        return out
    for r in rows:
        accs_r = chain_accs(r["apo_pdb_id"], r["apo_chain"].split("-")) | \
            chain_accs(r["holo_pdb_id"], r["holo_chain"].split("-")) | {r["uniprot_id"]}
        r["cb_accession_overlap"] = any(x in cb_accs or acc_groups[x] & cb30 for x in accs_r)

    def summary(rs):
        return {"pairs": len(rs), "apo_chains": len({(r["apo_pdb_id"], r["apo_chain"]) for r in rs}),
                "apo_entries": len({r["apo_pdb_id"] for r in rs}), "holo_entries": len({r["holo_pdb_id"] for r in rs}),
                "uniprot": len({r["uniprot_id"] for r in rs})}

    filters = [
        ("pRMSD >= 2.0 A", lambda r: r["pRMSD"] >= MIN_PRMSD),
        ("TM-score >= 0.5", lambda r: r["tm_score"] >= MIN_TM),
        ("centre shift <= 4 A", lambda r: r["center_shift"] <= MAX_CENTER_SHIFT),
        ("Rg change <= 20%", lambda r: r["rg_change"] <= MAX_RG_CHANGE),
        (">= 50 observed UniProt-mapped residues (apo and holo)",
         lambda r: r["n_obs_apo"] >= MIN_OBSERVED and r["n_obs_holo"] >= MIN_OBSERVED),
        ("apo chain(s) <= 1500 residues", lambda r: r["apo_length"] <= MAX_APO_LEN),
        ("holo released >= %s" % CUTOFF_DATE, lambda r: r["holo_release_date"] >= CUTOFF_DATE),
        ("no RCSB 30% group shared with a CryptoBench apo chain (pair entities)", lambda r: not r["cb30_overlap"]),
        ("UniProt-level novelty (accession not in CryptoBench; no same-accession entity in a CryptoBench 30% group)",
         lambda r: not r["cb_accession_overlap"]),
    ]
    steps = [("pairs with pRMSD >= %.1f A (complete, apo site empty, ligand filter passed)" % a.loose,
              summary(rows))]
    loose_rows = rows
    cur = rows
    for name, f in filters:
        cur = [r for r in cur if f(r)]
        steps.append((name, summary(cur)))
    # what a looser crypticity threshold would give (all other filters unchanged)
    looser = [r for r in loose_rows if all(f(r) for n, f in filters[1:])]
    # maximum global identity of each apo chain to the 1,281 CryptoBench apo chain sequences
    seqs = {}
    for r in looser:
        e = meta[r["apo_pdb_id"]]["entities"]
        for c in r["apo_chain"].split("-"):
            for x in e.values():
                if c in x["auth"]:
                    seqs[f"{r['apo_pdb_id']}{c}"] = x["seq"]
    cache_path = os.path.join(m, "identity_v2.json")       # sequence -> (max identity, CryptoBench chain, chance)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [(v, v, ref["seqs"]) for v in sorted(set(seqs.values())) if v not in cache]
    if todo:
        with cf.ProcessPoolExecutor(a.workers) as ex:
            for n, (k, best, hit, chance) in enumerate(ex.map(_max_identity, todo, chunksize=1)):
                cache[k] = (best, hit, chance)
                if n % 50 == 49:
                    json.dump(cache, open(cache_path, "w"))
                    print(f"  identity {n + 1}/{len(todo)}", file=sys.stderr)
        json.dump(cache, open(cache_path, "w"))
    for r in looser:
        best = max((cache[seqs[f"{r['apo_pdb_id']}{c}"]] for c in r["apo_chain"].split("-")), key=lambda x: x[0])
        r["max_identity_to_cryptobench"] = round(best[0], 4)
        r["closest_cryptobench_chain"] = best[1]
        r["identity_chance_level"] = round(best[2], 4)
    # 40% clustering of the UniProt accessions (one sequence each: the longest apo entity sequence)
    useq = {}
    for r in cur:
        sq = seqs[f"{r['apo_pdb_id']}{r['main_apo_chain']}"]
        if len(sq) > len(useq.get(r["uniprot_id"], "")):
            useq[r["uniprot_id"]] = sq
    accs = sorted(useq)
    parent = list(range(len(accs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    jobs = [(i, j, useq[accs[i]], useq[accs[j]]) for i in range(len(accs)) for j in range(i + 1, len(accs))]
    links = []
    if jobs:
        with cf.ProcessPoolExecutor(a.workers) as ex:
            for i, j, v in ex.map(_identity_pair, jobs, chunksize=64):
                if v >= CLUSTER_ID:
                    parent[find(i)] = find(j)
                    links.append((accs[i], accs[j], round(v, 3)))
    cluster = {acc: f"C{find(i)}" for i, acc in enumerate(accs)}
    for r in cur:
        r["cluster_40"] = cluster[r["uniprot_id"]]
    by_cluster = collections.defaultdict(list)
    for r in cur:
        by_cluster[r["cluster_40"]].append(r)
    reps = []
    for cl, rs in by_cluster.items():
        rs.sort(key=lambda r: (-r["pRMSD"], r["resolution_apo"] or 9, r["apo_pdb_id"], r["holo_pdb_id"]))
        reps.append(rs[0])
    steps.append(("40% clusters -> one representative pair each (max pRMSD)", summary(reps)))
    # additional pockets of the representative apo structures (same apo entry, overlapping apo chains)
    dataset, used = {}, set()
    for rep in sorted(reps, key=lambda r: r["apo_pdb_id"]):
        chains = set(rep["apo_chain"].split("-"))
        extra = [r for r in cur if r["apo_pdb_id"] == rep["apo_pdb_id"] and r is not rep
                 and chains & set(r["apo_chain"].split("-"))]
        recs = dataset.setdefault(rep["apo_pdb_id"], [])
        for k, r in enumerate([rep] + sorted(extra, key=lambda r: -r["pRMSD"])):
            key = (r["holo_pdb_id"], r["holo_chain"], r["ligand_chain"], r["ligand_index"], r["apo_chain"])
            if key in used:
                continue
            used.add(key)
            recs.append(_record(r, main=(k == 0)))
    final_rows = [x for v in dataset.values() for x in v]
    steps.append(("additional pockets of the representative apo structures",
                  {"pairs": len(final_rows), "apo_entries": len(dataset),
                   "holo_entries": len({x["holo_pdb_id"] for x in final_rows}),
                   "uniprot": len({x["uniprot_id"] for x in final_rows})}))
    json.dump(dataset, open(os.path.join(a.out, "dataset.json"), "w"), indent=1)
    for f in os.listdir(os.path.join(a.out, "cif")):           # drop files of an earlier selection
        if f.endswith(".cif.gz") and f[:-7] not in dataset:
            os.remove(os.path.join(a.out, "cif", f))
    for apo_id in dataset:
        src, dst = os.path.join(a.out, "raw", f"{apo_id}.cif.gz"), os.path.join(a.out, "cif", f"{apo_id}.cif.gz")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
    # looser variants (for small sets): same pipeline with pRMSD >= 1.5 A
    def n_clusters(rs):
        return len({cluster.get(r["uniprot_id"], r["uniprot_id"]) for r in rs})
    mains = [v[0] for v in dataset.values()]
    out = {"steps": steps, "identity_links_ge_40": links,
           "final": {"apo_structures": len(dataset), "records": len(final_rows),
                     "apo_new_ge_cutoff": sum(v[0]["apo_release_date"] >= CUTOFF_DATE for v in dataset.values()),
                     "max_identity_lt_30": sum(x["max_identity_to_cryptobench"] < 0.30 for x in mains),
                     "max_identity_lt_10": sum(x["max_identity_to_cryptobench"] < 0.10 for x in mains),
                     "max_identity_within_0.05_of_chance": sum(x["max_identity_to_cryptobench"] -
                                                               x["identity_chance_level"] <= 0.05 for x in mains),
                     "max_identity_ge_40": sum(x["max_identity_to_cryptobench"] >= 0.40 for x in mains),
                     "methods_apo": collections.Counter(x["method_apo"] for x in mains),
                     "methods_holo": collections.Counter(x["method_holo"] for x in final_rows)},
           "looser_pRMSD_1.5": {"pairs": len(looser), "uniprot": len({r["uniprot_id"] for r in looser}),
                                "apo_entries": len({r["apo_pdb_id"] for r in looser})}}
    save_stats(a.out, "select", out)
    for name, sm in steps:
        print(f"{name:75s} {sm}")
    print(json.dumps(out["final"], indent=1))


def _record(r, main):
    rec = {k: r[k] for k in ("uniprot_id", "holo_pdb_id", "holo_chain", "apo_chain", "ligand", "ligand_index",
                             "ligand_chain", "apo_pocket_selection", "holo_pocket_selection")}
    rec["apo_pymol_selection"] = pymol_selection(r["apo_pdb_id"], r["apo_pocket_selection"])
    rec["holo_pymol_selection"] = pymol_selection(r["holo_pdb_id"], r["holo_pocket_selection"])
    rec["pRMSD"] = r["pRMSD"]
    rec["is_main_holo_structure"] = main
    for k in ("holo_release_date", "apo_release_date", "resolution_apo", "resolution_holo", "tm_score",
              "center_shift", "rg_change", "max_identity_to_cryptobench"):
        rec[k] = r[k]
    rec["cryptobench_30pct_cluster_overlap"] = r["cb30_overlap"] or r["cb_accession_overlap"]
    for k in ("identity_chance_level", "closest_cryptobench_chain", "pRMSD_heavy", "method_apo", "method_holo", "ligand_heavy_atoms",
              "main_apo_chain", "main_holo_chain", "n_obs_apo", "apo_length", "cluster_40", "apo_group30",
              "holo_group30"):
        rec[k] = r[k]
    rec["apo_is_new"] = r["apo_release_date"] >= CUTOFF_DATE
    return rec


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["validate", "apocheck", "cbref", "search", "download", "pairs", "select"])
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder (default ~/crypsis_data/temporal)")
    ap.add_argument("--cryptobench", default=DEFAULT_CB, help="CryptoBench folder (dataset + cif/)")
    ap.add_argument("--n", type=int, default=120, help="validate: number of random CryptoBench records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="pairs/select: processes; download: <= 5 connections")
    ap.add_argument("--loose", type=float, default=1.5, help="pairs: keep pairs with pRMSD >= this for analysis")
    a = ap.parse_args()
    a.out = os.path.abspath(os.path.expanduser(a.out))
    for sub in ("meta", "raw", "cif"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    globals()["cmd_" + a.step](a)


if __name__ == "__main__":
    main()

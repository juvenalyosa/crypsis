#!/usr/bin/env python3
"""How new are the temporal set's proteins?  Homology to CryptoBench and exposure to ESM-2 pretraining, from
UniProt canonical sequences (no expression tags).  Added after review.

    python scripts/temporal_novelty.py fetch      # UniProt records of every accession -> ~/crypsis_data/uniprot/
    python scripts/temporal_novelty.py homology   # -> ~/crypsis_data/temporal/novelty.json
    python scripts/temporal_novelty.py homology --query cryptobench_test   # calibration: CryptoBench's own test split

fetch: accession, first public date, canonical sequence and Pfam families of every UniProt accession in the
CryptoBench dataset and the temporal set (UniProt REST API, JSON; nothing is executed from the downloads).

homology: for every temporal protein, local Smith-Waterman alignment (BLOSUM62, gap open 11, extend 1) of its
canonical sequence against the canonical sequence of every CryptoBench *training* accession.  For the five best
scores the alignment is computed: identity over aligned columns and coverage of the shorter sequence.  A protein
has a detectable homologue when one alignment reaches >= 30% identity over >= 50% coverage, and a family overlap when
it shares a Pfam family with any training accession.  Pretraining exposure: the share of accessions first public in
UniProt before 2021-11-17, the date of UniRef release 2021_04 on which ESM-2 was trained, for the temporal set and the
CryptoBench test split.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOME = os.path.expanduser("~/crypsis_data")
CACHE = os.path.join(HOME, "uniprot")
CB = os.path.join(HOME, "cryptobench", "cryptobench-dataset")
TEMPORAL = os.path.join(HOME, "temporal", "dataset.json")
ESM_SNAPSHOT = "2021-11-17"          # UniProt/UniRef release 2021_04 (17 Nov 2021), used to train ESM-2


def accessions():
    with open(os.path.join(CB, "dataset.json")) as fh:
        cb = json.load(fh)
    with open(os.path.join(CB, "folds.json")) as fh:
        folds = json.load(fh)
    with open(TEMPORAL) as fh:
        tp = json.load(fh)
    split = {apo: name for name, ids in folds.items() for apo in ids}
    ids = lambda h: h["uniprot_id"].split("-")           # multi-chain entries join their accessions with "-"
    train = sorted({x for apo, v in cb.items() if split.get(apo, "").startswith("train") for h in v for x in ids(h)})
    test = sorted({x for apo, v in cb.items() if split.get(apo) == "test" for h in v for x in ids(h)})
    temporal = {apo: ids(v[0]) for apo, v in tp.items()}
    return train, test, temporal


def cmd_fetch(_a):
    os.makedirs(CACHE, exist_ok=True)
    train, test, temporal = accessions()
    todo = sorted(set(train) | set(test) | {x for v in temporal.values() for x in v})
    bad = sorted(x for x in todo if not re.fullmatch(r"[A-Z0-9]{6,10}", x))
    todo = [x for x in todo if x not in bad and not os.path.exists(os.path.join(CACHE, f"{x}.json"))]
    print("not UniProt accessions, skipped:", bad)
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        q = urllib.parse.urlencode({"query": " OR ".join(f"accession:{x}" for x in batch), "format": "json",
                                    "fields": "accession,date_created,sequence,xref_pfam", "size": 500})
        try:
            with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/search?{q}", timeout=120) as r:
                res = json.load(r)["results"]
        except urllib.error.HTTPError as exc:
            print("batch failed:", exc, batch[:3], flush=True)
            continue
        for e in res:
            rec = {"accession": e["primaryAccession"],
                   "first_public": e.get("entryAudit", {}).get("firstPublicDate"),
                   "sequence": e.get("sequence", {}).get("value", ""),
                   "pfam": sorted({x["id"] for x in e.get("uniProtKBCrossReferences", []) if x.get("database") == "Pfam"})}
            with open(os.path.join(CACHE, f"{rec['accession']}.json"), "w") as fh:
                json.dump(rec, fh)
        print(f"{i + len(batch)}/{len(todo)} fetched", flush=True)
        time.sleep(0.5)
    missing = [x for x in sorted(set(train) | set(test) | {x for v in temporal.values() for x in v})
               if not os.path.exists(os.path.join(CACHE, f"{x}.json"))]
    print("missing (obsolete or secondary accessions):", missing)


def load(acc):
    path = os.path.join(CACHE, f"{acc}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _aligner():
    from Bio import Align
    from Bio.Align import substitution_matrices

    al = Align.PairwiseAligner()
    al.mode = "local"
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    return al


def _clean(s):
    return "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def _best(job):
    q, targets = job
    al = _aligner()
    qs = _clean(q["sequence"])
    scores = sorted(((al.score(qs, _clean(t["sequence"])), t["accession"]) for t in targets), reverse=True)[:5]
    seqs = {t["accession"]: _clean(t["sequence"]) for t in targets}
    hits = []
    for sc, acc in scores:
        aln = al.align(qs, seqs[acc])[0]
        ident = cols = 0
        for (a0, a1), (b0, b1) in zip(*aln.aligned):
            seg_a, seg_b = qs[a0:a1], seqs[acc][b0:b1]
            ident += sum(x == y for x, y in zip(seg_a, seg_b))
            cols += a1 - a0
        span_q = aln.aligned[0][-1][1] - aln.aligned[0][0][0]
        span_t = aln.aligned[1][-1][1] - aln.aligned[1][0][0]
        cover = max(span_q, span_t) / min(len(qs), len(seqs[acc]))
        hits.append({"target": acc, "score": float(sc), "identity": round(ident / max(cols, 1), 3),
                     "coverage_shorter": round(min(cover, 1.0), 3), "aligned": int(cols)})
    return q["accession"], hits


def cmd_homology(a):
    train, test, temporal = accessions()
    tr = [r for r in (load(x) for x in train) if r and r["sequence"]]
    pf_train = set().union(*[set(r["pfam"]) for r in tr])
    if a.query == "cryptobench_test":                     # the same rule on CryptoBench's test split, to calibrate it
        with open(os.path.join(CB, "dataset.json")) as fh:
            cbd = json.load(fh)
        with open(os.path.join(CB, "folds.json")) as fh:
            temporal = {apo: cbd[apo][0]["uniprot_id"].split("-") for apo in json.load(fh)["test"]}
    queries = sorted({x for v in temporal.values() for x in v})
    recs = [load(x) for x in queries]
    jobs = [(r, tr) for r in recs if r and r["sequence"]]
    out = {"esm_snapshot": ESM_SNAPSHOT, "training_accessions": len(tr), "proteins": {}}
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for k, (acc, hits) in enumerate(ex.map(_best, jobs), 1):
            r = load(acc)
            homolog = any(h["identity"] >= 0.30 and h["coverage_shorter"] >= 0.50 for h in hits)
            shared = sorted(set(r["pfam"]) & pf_train)
            out["proteins"][acc] = {"first_public": r["first_public"], "pfam": r["pfam"], "shared_pfam": shared,
                                    "homolog": homolog, "family_overlap": bool(shared), "top_hits": hits}
            if k % 25 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    for acc in queries:
        if acc not in out["proteins"]:
            out["proteins"][acc] = {"missing": True}
    P = out["proteins"]
    ok = lambda accs, keys: all(not P[x].get("missing") and not any(P[x][k] for k in keys) for x in accs)
    out["apo_novel"] = sorted(apo for apo, accs in temporal.items() if ok(accs, ("homolog", "family_overlap")))
    out["apo_no_homolog"] = sorted(apo for apo, accs in temporal.items() if ok(accs, ("homolog",)))

    def before(accs):
        d = [load(x) for x in accs]
        d = [x["first_public"] for x in d if x and x.get("first_public")]
        return {"n": len(d), "before_snapshot": int(sum(x < ESM_SNAPSHOT for x in d))}

    out["pretraining"] = {"temporal": before(queries), "cryptobench_test": before(test),
                          "cryptobench_train": before(train)}
    out["summary"] = {"proteins": len(queries), "with_homolog": int(sum(p.get("homolog", False) for p in out["proteins"].values())),
                      "with_family_overlap": int(sum(p.get("family_overlap", False) for p in out["proteins"].values())),
                      "novel_apo": len(out["apo_novel"]), "no_homolog_apo": len(out["apo_no_homolog"])}
    print(json.dumps(out["summary"]), json.dumps(out["pretraining"]))
    name = "novelty.json" if a.query == "temporal" else "novelty_cryptobench_test.json"
    with open(os.path.join(HOME, "temporal", name), "w") as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "homology"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--query", choices=["temporal", "cryptobench_test"], default="temporal")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "homology": cmd_homology}[a.command](a)


if __name__ == "__main__":
    main()

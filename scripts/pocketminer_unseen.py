#!/usr/bin/env python3
"""PocketMiner test set: 'unseen' under the homology rule of the temporal set (local BLOSUM62 alignment,
>= 30% identity over >= 50% coverage of the shorter sequence) against the CryptoBench training chains, next to
the >= 50% global-identity rule of pocketminer_set.py; pooled AUC of the four methods on both subsets
(Supplementary Table S2 legend and Note 2).  Needs Biopython, gemmi and scikit-learn:

    python scripts/pocketminer_unseen.py      # after pocketminer_set.py --baselines ... --out .../v0.2.1_rev2
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import cryptobench as cb  # noqa: E402
import pocketminer_set as ps  # noqa: E402
import temporal_novelty as tn  # noqa: E402
from crypsis import core, pipeline  # noqa: E402

D = os.path.expanduser("~/crypsis_data")
PM = f"{D}/external/gvp/data/pm-dataset"
OUT = f"{D}/pocketminer/v0.2.1_rev2"
BASE = {"pocketminer": f"{D}/pocketminer/pmset_pocketminer", "p2rank": f"{D}/pocketminer/pmset_p2rank"}

train = [str(s) for apo, ch, _ in cb.entries(f"{D}/cryptobench", "train")
         if os.path.exists(f := os.path.join(D, "work", "features", f"{apo}{ch}.npz")) for s in np.load(f)["seqs"]]
train = sorted({tn._clean(t) for t in train if len(tn._clean(t)) >= 30})
print("training chain sequences:", len(train))
al = tn._aligner()


def temporal_rule(seq):
    qs = tn._clean(seq)
    best = sorted(((al.score(qs, t), t) for t in train), reverse=True)[:5]
    for _sc, t in best:
        aln = al.align(qs, t)[0]
        ident = cols = 0
        for (a0, a1), (b0, b1) in zip(*aln.aligned):
            ident += sum(x == y for x, y in zip(qs[a0:a1], t[b0:b1]))
            cols += a1 - a0
        span_q = aln.aligned[0][-1][1] - aln.aligned[0][0][0]
        span_t = aln.aligned[1][-1][1] - aln.aligned[1][0][0]
        cover = max(span_q, span_t) / min(len(qs), len(t))
        if ident / max(cols, 1) >= 0.30 and cover >= 0.50:
            return False, round(ident / max(cols, 1), 3), round(min(cover, 1.0), 3)
    return True, None, None


ids = [str(i) for i in np.load(os.path.join(PM, "test_apo_ids_with_chainids.npy"))]
labels = ps.load_labels(os.path.join(PM, "test_label_dictionary.npy"))
per = {m: ([], [], [], []) for m in ("hybrid", "physics", "pocketminer", "p2rank")}
rows = []
for ident in ids:
    chain = core.read_chain(ps.structure_file(PM, ident))
    lab = ps.label_of(labels, ident)
    seq = "".join(pipeline.sequences(chain))
    u50 = ps.unseen(seq, train)
    u30, idn, cov = temporal_rule(seq)
    rows.append((ident, u50, u30, idn, cov))
    keep = lab != 2
    for m in per:
        if m in BASE:
            prob = np.load(os.path.join(BASE[m], f"test_{ident}.npy"))
        else:
            prob = np.asarray(json.load(open(os.path.join(OUT, f"test_{ident}_{m}.json")))["probability"])
        assert len(prob) == len(lab), (ident, m)
        per[m][0].append(prob[keep]); per[m][1].append(lab[keep].astype(int)); per[m][2].append(u50); per[m][3].append(u30)

print("id unseen50 unseenTemporalRule bestIdentity coverage")
for r in rows:
    print(*r)
print("unseen (>=50% global):", sum(r[1] for r in rows), " unseen (temporal rule):", sum(r[2] for r in rows))
for m, (s, y, u50, u30) in per.items():
    for name, flags in (("all", [True] * len(s)), ("unseen50", u50), ("unseen_temporal_rule", u30)):
        sel = [i for i, f in enumerate(flags) if f]
        r = ps.pooled_auc([s[i] for i in sel], [y[i] for i in sel])
        print(f"{m:12s} {name:22s} auc {r['auc']:.3f} [{r['ci'][0]:.3f}-{r['ci'][1]:.3f}] proteins {r['proteins']} "
              f"residues {r['residues']} positives {r['positives']} ({100 * r['positives'] / r['residues']:.1f}%)")

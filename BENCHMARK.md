# Benchmarks and how the models were built

This file records how every number in the README was produced. It covers the data, the protocol, every experiment
tried while developing Crypsis 0.2 (including the ones that did not help) and the final test-set results. All the
scripts are in `scripts/`. Everything runs on a CPU.

## Data

- **CryptoBench** (Škrhák et al., *Bioinformatics* 2025), from <https://osf.io/pz4a9/>:
  - `cryptobench-dataset/{dataset.json, folds.json}`;
  - the apo mmCIF files from `auxiliary-data/cif-files.zip`, gzipped into `<data>/cif/<apo>.cif.gz`.

  It has 1,107 apo structures: a test split of 222 and four training folds of 219–222. One training structure
  (6KQF, 1,500+ residues) is above the size limit of the scripts, so training uses 884 structures.
- **PocketMiner set** (Meller et al., *Nat. Commun.* 2023): `data/pm-dataset` of
  <https://github.com/Mickdub/gvp> (branch `pocket_pred`). It holds the cleaned apo structures and the
  validation (26) and test (35) label dictionaries.
- **Eight literature cases**: apo and holo mmCIF files from the RCSB (`scripts/literature_cases.py` lists them).

## Protocol

- **Labels (CryptoBench's own).** Every observed residue of an apo chain is scored. A residue is positive when it is
  in the `apo_pocket_selection` of any holo structure paired with that chain.
- **Residue metrics.** ROC AUC and AUPRC (average precision) are computed over the residues of all the proteins of
  a split pooled together. 95% confidence intervals come from 2,000 bootstrap resamples of proteins. Paired
  differences between two models use the same resamples.
- **Site metrics.** Each holo structure's pocket is a true site; 101 of the 222 test proteins have more than one.
  A predicted site (`core.sites_from_scores`, top 10% of residues) finds a true site when they share at least 3
  residues and either half of the predicted site's residues or a third of the true site's. Two rates are reported:
  - the share of proteins whose first predicted site finds a true site;
  - the share whose top three predicted sites include one that does.

  Both versions are scored with this code. The README of Crypsis 0.1 reported 44% / 61% (hybrid) with an earlier
  version of the definition; this code gives 45% / 60% for the same 0.1 models.
- **Model selection** uses only the four training folds: 4-fold cross-validation, each fold held out once. The test
  split was scored once for the final configuration. One exception is recorded below under
  [The ensemble columns](#the-ensemble-columns).

## Rebuilding everything

```bash
W=~/crypsis_data/work
python scripts/cryptobench.py features    --split all --work $W   # ANM ensemble + pocket grids (~1 h, 5 workers)
python scripts/cryptobench.py descriptors --split all --work $W   # descriptors.DESCRIPTORS (~10 min)
python scripts/cryptobench.py ensemble    --split all --work $W   # descriptors.ENSEMBLE (~1 h)
python scripts/cryptobench.py esm --esm 650M --work $W            # ESM-2 embeddings (~2 h on a CPU)
python scripts/cryptobench.py esm --esm 150M --work $W
python scripts/cryptobench.py cv    --esm 650M --pcs 256 --work $W
python scripts/cryptobench.py train --esm 650M --pcs 256 --work $W
python scripts/cryptobench.py train --esm 150M --pcs 128 --work $W
python scripts/export_models.py $W/final_models_v3_650M_256.pkl src/crypsis/models 650M
python scripts/export_models.py $W/final_models_v3_150M_128.pkl src/crypsis/models 150M --hybrid-only
python scripts/cryptobench.py test --shipped --esm 650M --pcs 256 --work $W
python scripts/cryptobench.py test --shipped --esm 150M --pcs 128 --work $W
```

`--physics v2` rebuilds the 22 physics columns of Crypsis 0.1. Rebuilt this way, the 0.1 models reproduce the
published ones:

| Model | Published (0.1) | Rebuilt |
|---|---|---|
| Hybrid | AUC 0.881 / AUPRC 0.372 | AUC 0.880 / AUPRC 0.369 |
| Physics | AUC 0.847 / AUPRC 0.283 | AUC 0.847 / AUPRC 0.283 |

## Development log (cross-validation on the training folds)

All rows are 4-fold cross-validation over the 884 training proteins: pooled AUC / AUPRC. A change was kept only
when it helped here.

| Experiment | AUC | AUPRC | Kept |
|---|---|---|---|
| Crypsis 0.1 physics (22 columns), rebuilt | 0.847 | 0.285 | baseline |
| Crypsis 0.1 hybrid (ESM-2 150M, 128 PCs), rebuilt | 0.873 | 0.356 | baseline |
| Hybrid, ESM-2 650M, 64 / 128 / 256 PCs | 0.875 / 0.877 / 0.879 | 0.349 / 0.358 / 0.365 | 650M, 256 PCs |
| + Gaussian smoothing of the scores (σ 6 Å, 70% own score) | 0.881 | 0.368 | no (test: mixed) |
| + learned spatial second stage (neighbourhood scores at 6–16 Å) | 0.882 | 0.367 | no |
| Physics + `DESCRIPTORS` (exposure, GNM, geometry, trimmed side chains) | 0.861 | 0.319 | yes |
| Physics + 5 ensemble variants (1.5 Å, 5 Å, single modes, trimmed) | 0.852 | 0.300 | partly |
| Physics + both | 0.865 | 0.325 | yes |
| … + pockets opened by moving one side chain at a time | 0.865 | 0.326 | no |
| … + 7.5 Å ensemble and 5 Å trimmed ensemble | 0.865 | 0.325 | no |
| … + hydrophobic / polar probe mapping (static, trimmed, 5 Å ensemble) | 0.865 | 0.326 | no |
| … gradient-boosting settings (4 alternatives) | 0.863–0.865 | 0.323–0.326 | defaults |
| Hybrid 650M / 256 PCs + both | 0.886 | 0.378 | yes |
| **Final physics** (22 + 21 descriptors + 6 ensemble columns) | **0.864** | **0.321** | |
| **Final hybrid** (the same + ESM-2 650M, 256 PCs) | **0.886** | **0.379** | |

**Ablations** (drop one group, folds 2 and 3 held out, change in AUPRC):

| Group | Change in AUPRC |
|---|---|
| 0.1 inputs: ensemble pocket columns | −0.022 |
| 0.1 inputs: shallow pass | −0.006 |
| 0.1 inputs: mode mobility | −0.004 |
| 0.1 inputs: B-factor | −0.004 |
| 0.1 inputs: propensity | −0.004 |
| 0.1 inputs: static pocket columns | −0.000 |
| New descriptors: trimmed side chains | −0.010 |
| New descriptors: exposure | −0.009 |
| New descriptors: hydrophobic exposed patch | −0.004 |
| New descriptors: backbone geometry | −0.002 |
| New descriptors: GNM | −0.002 |
| New descriptors: side-chain B-factor | +0.001 (dropped) |
| Ensembles: 5 Å | −0.004 |
| Ensembles: trimmed 3 Å | −0.004 |
| Ensembles: 1.5 Å | −0.001 (dropped) |
| Ensembles: single slow modes | 0.000 (dropped) |

### The ensemble columns

The first test-set run of the final configuration used all 11 ensemble columns: hybrid AUC 0.889 and AUPRC 0.392,
physics 0.863 / 0.314. The five columns that the ablation above had found useless were then removed, to save
computing time. Cross-validation did not change (hybrid 0.886 / 0.379), and the shipped models were trained on
the reduced set. The numbers below are those of the shipped models. The 11-column version was not reinstated
because of its slightly higher test AUPRC: choosing on the test set would bias it.

## CryptoBench test set (222 structures, scored once)

The models are scored as shipped: the exported trees and the package's own input pipeline. The site rates use the
definition above.

| Model | ROC AUC [95% CI] | AUPRC [95% CI] | Median AUC per protein | Site 1st | Site top 3 |
|---|---|---|---|---|---|
| **0.2 hybrid, ESM-2 650M** | **0.889** [0.871–0.905] | **0.388** [0.344–0.433] | **0.928** | **51%** | **64%** |
| 0.2 hybrid, ESM-2 150M | 0.884 [0.866–0.900] | 0.368 [0.326–0.412] | 0.923 | 43% | 59% |
| 0.2 physics | 0.864 [0.843–0.882] | 0.312 [0.274–0.354] | 0.901 | 41% | 55% |
| 0.1 hybrid (ESM-2 150M) | 0.881 [0.862–0.896] | 0.372 [0.328–0.415] | 0.918 | 45% | 60% |
| 0.1 physics | 0.847 [0.827–0.865] | 0.283 [0.246–0.324] | 0.891 | 39% | 50% |

**Paired differences** (95% CI over 2,000 protein resamples):

| Comparison | AUC | AUPRC |
|---|---|---|
| 0.2 physics − 0.1 physics | +0.009 to +0.024 | +0.007 to +0.048 |
| 0.2 hybrid 650M − 0.1 hybrid | +0.003 to +0.014 | −0.003 to +0.034 |
| 0.2 hybrid 150M − 0.1 hybrid | −0.001 to +0.008 | −0.020 to +0.011 |
| 0.2 hybrid 650M − 0.2 hybrid 150M | +0.001 to +0.008 | +0.008 to +0.032 |

What the differences say:
- The physics model improves on both measures.
- The 650M hybrid improves on AUC; its AUPRC gain is not significant.
- The 150M hybrid of 0.2 is equivalent to the 0.1 hybrid.

**Per protein:**

| Model | Proteins with AUC < 0.6 | Proteins with AUC > 0.9 |
|---|---|---|
| 0.1 physics | 17.6% | 47.3% |
| 0.2 physics | 14.4% | 50.5% |
| 0.1 hybrid | 12.6% | 55.9% |
| 0.2 hybrid 650M | **10.8%** | **56.8%** |

**Other methods.** The P2Rank and PocketMiner rows were scored by us on the same files for Crypsis 0.1; they cover
221 of the 222 structures. The others are published numbers on the same test split, so their labels may differ
slightly from these. Seq2Pocket computes AUPRC as the trapezoidal area under the precision–recall curve; on these
scores it differs from average precision by less than 0.001.

| Method | ROC AUC | AUPRC | Source |
|---|---|---|---|
| Seq2Pocket (ESM-2 3B fine-tuned) | 0.894 | 0.475 | Škrhák et al., bioRxiv 2026 |
| **Crypsis 0.2 hybrid (650M)** | **0.889** | **0.388** | this work |
| ESM-2 3B fine-tuned (CryptoBench baseline) | 0.871 | 0.382 | Škrhák et al., 2025 |
| **Crypsis 0.2 physics** (no language model) | **0.864** | **0.312** | this work |
| pLM-NN | 0.86 | 0.36 | Škrhák et al., 2025 |
| P2Rank 2.5.1 | 0.788 | 0.191 | run by us |
| PocketMiner | 0.762 | 0.173 | run by us |

## PocketMiner set

`scripts/pocketminer_set.py`. The labels are pooled as in the PocketMiner paper. "Unseen" test proteins have no
CryptoBench training chain with at least 50% sequence identity: 20 of the 35.

| Pooled ROC AUC [95% CI] | Test, 35 | Unseen test, 20 | Validation, 26 |
|---|---|---|---|
| 0.2 hybrid (650M) | 0.920 [0.868–0.963] | 0.893 [0.813–0.959] | 0.873 [0.724–0.955] |
| 0.2 physics | 0.903 [0.844–0.952] | 0.879 [0.794–0.953] | 0.838 [0.661–0.940] |
| 0.1 hybrid | 0.907 [0.863–0.947] | 0.882 [0.821–0.937] | 0.846 [0.671–0.944] |
| 0.1 physics | 0.877 [0.819–0.936] | 0.848 [0.762–0.935] | 0.803 [0.572–0.935] |

The 0.1 values match those published with Crypsis 0.1 (test hybrid 0.907) within 0.005. The "unseen" subset of
0.1 had 23 proteins, from an earlier definition that was not recorded.

## Eight literature cases

`scripts/literature_cases.py`. The true site is every apo residue within 4.5 Å of the ligand in the holo
structure, matched by sequence alignment. Where a ligand is present twice, only the copy in the described site is
used. Both versions of Crypsis are scored with the same labels; see the README for the table.

## What did not help, and why it matters

Everything in the development log comes from the apo structure: pockets on a grid, normal-mode ensembles of
different amplitudes, side chains cut back, exposure, probe mapping. Past about AUC 0.865 / AUPRC 0.325 for the
physics model, new views of the same structure stopped adding information.

The remaining gap to Seq2Pocket is in AUPRC. Seq2Pocket fine-tunes a 3-billion-parameter language model, and that
is where its gain comes from. Its smoothing step changes AUPRC little.

The next physical step would bring in information the structure does not hold, such as short molecular-dynamics
runs.

# Crypsis

**Find the binding pockets a protein structure is hiding.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
![CryptoBench AUC 0.888](https://img.shields.io/badge/CryptoBench%20AUC-0.888-d94801.svg)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)

*Crypsis* is the biological term for an organism avoiding detection. **Cryptic pockets** do something similar:
they are closed or absent in the apo crystal structure, then open when the protein moves or a ligand binds. Many
targets once called "undruggable" have one, for example the switch-II pocket of K-Ras G12C, the target of
sotorasib.

Crypsis starts from a single structure and keeps the physics explicit at every step:

1. It computes the slow, collective motions of the fold with elastic network models.
2. It builds ensembles of conformers along those motions, with full and with cut-back side chains.
3. In each conformer it finds buried, probe-accessible pocket space on a C++ grid.
4. It adds descriptors of solvent exposure, hinges between moving domains and local backbone geometry.
5. A gradient-boosted model, optionally combined with ESM-2 protein-language-model embeddings, turns these
   descriptors into a probability for every residue.

Each predicted site comes with its pocket volume in the input structure and along the motions. The volumes tell
you whether the pocket is **already open**, **grows with motion** or **forms with motion**. Each site also comes
with a ready-made docking box.

```
$ crypsis predict 4ldj.cif --chains A --keep GDP,MG
crypsis hybrid (ESM-2 650M) model, 170 residues, 2 sites (17 s)
rank  score   n  kind               volume A3 (structure -> ensemble)  residues
   1   2.04   5  forms with motion  15 -> 24                           Val9 Thr58 Ala59 Gly60 Gln61
   2   0.86   4  forms with motion  0 -> 10                            Lys5 Asp38 Ile55 Leu56
```

This run is on K-Ras G12C with GDP bound but no drug (PDB 4LDJ). The switch-II pocket, where sotorasib binds, is
ranked first.

---

## Contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Benchmarks](#benchmarks)
- [Speed and memory](#speed-and-memory)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [Citing](#citing)

## Highlights

- **More accurate than the reference tools on the same structures.** On the CryptoBench test set, the residue
  ROC AUC is 0.888, and the physics model alone (no language model) reaches 0.864. With the earlier labels on
  which P2Rank (0.788) and PocketMiner (0.762) were scored, Crypsis reached 0.889, with about twice their AUPRC.
- **Explainable.** Each site is a real pocket on a grid, with a measured volume in the structure and along the
  motions, not just a residue score.
- **Fast enough for a laptop.** 6–30 s per protein on a 2018 laptop CPU with the C++ grid (see
  [Speed and memory](#speed-and-memory)).
- **Light.** The physics model needs only NumPy, SciPy and gemmi. PyTorch is needed only for the hybrid model.
- **Handles cofactors.** Bound GDP, heme, NAD and similar groups can be kept as occupied space, so removing them
  does not create a fake pocket.
- **Ready for the next step.** Every site comes with a docking box, a PDB coloured by probability and a PyMOL
  script.

## Requirements

| | Minimum | Notes |
|---|---|---|
| Operating system | macOS or Linux | Windows is untested; the NumPy fallback should run there |
| Python | 3.9+ | tested on 3.11 and 3.12 |
| Core dependencies | `numpy`, `scipy`, `gemmi` | installed automatically |
| C++ compiler (optional) | any C++17 compiler | macOS: `xcode-select --install`; Linux: `g++`. Without one, Crypsis uses the NumPy grid, which gives the same results about 5–15× more slowly |
| Hybrid model (optional) | `torch`, `fair-esm` | `pip install "crypsis[esm]"`; downloads the ESM-2 650M weights (about 2.5 GB) on first use, or the 150M weights (about 600 MB) with `--esm 150M` |
| Memory | about 0.25 GB for the physics model; 5.4 GB for the hybrid model with ESM-2 650M, 1.4 GB with 150M | for a 1,000-residue protein |
| GPU | not needed | ESM-2 can use `--device cuda` or `--device mps` |

## Installation

**With the hybrid model** (recommended):

```bash
pip install "crypsis[esm] @ git+https://github.com/juvenalyosa/crypsis"
```

**Physics model only**, with no PyTorch:

```bash
pip install "git+https://github.com/juvenalyosa/crypsis"
```

**In a new conda environment:**

```bash
conda create -n crypsis python=3.12 -y
conda activate crypsis
pip install "crypsis[esm] @ git+https://github.com/juvenalyosa/crypsis"
```

**From source, for development:**

```bash
git clone https://github.com/juvenalyosa/crypsis
cd crypsis
pip install -e ".[esm,test]"
pytest
```

**Check the installation:**

```console
$ crypsis check
C++ grid extension: yes
ESM-2 (hybrid model): yes
```

If the C++ grid extension shows `no`, install a compiler and reinstall with `pip install --force-reinstall
--no-deps ...`. Until then, Crypsis works with the NumPy fallback.

## Quick start

### Command line

```bash
crypsis predict structure.cif                       # all protein chains, best available model
crypsis predict 1dlo.cif --chains A,B               # a heterodimer
crypsis predict 4ldj.cif --keep GDP,MG              # keep bound cofactors as occupied space
crypsis predict 4ldj.cif --keep auto                # ... or every hetero group of >= 8 heavy atoms
crypsis predict structure.pdb --model physics       # hybrid | physics | v1
crypsis predict structure.pdb --esm 150M            # lighter language model (1.4 GB instead of 5.4 GB)
crypsis predict structure.pdb --device mps          # ESM-2 on an Apple GPU (or cuda)
```

| Option | Default | Meaning |
|---|---|---|
| `--chains A,B` | all protein chains | chains to analyse together |
| `--keep auto` or `--keep GDP,MG` | none | cofactors kept as occupied, unscored space |
| `--model` | `auto` | `hybrid` when PyTorch and fair-esm are installed, otherwise `physics`; `v1` is the 3-weight logistic model |
| `--esm` | `650M` | the hybrid's language model: `650M` (more accurate) or `150M` (lighter and faster) |
| `--device` | `cpu` | PyTorch device for ESM-2 |
| `-o DIR` | `crypsis_out` | output folder |
| `--top N` | 5 | number of sites to print |

The input can be a PDB or mmCIF file, gzipped or not. Each run writes four files:

| File | Contents |
|---|---|
| `<name>_residues.csv` | each residue's probability, how often a pocket opened next to it in the ensemble, and its site |
| `<name>_sites.json` | ranked sites: residues, kind, pocket volumes, mean probability and a docking box (`center`, `size` in Å) |
| `<name>_crypsis.pdb` | the input structure with each B-factor set to 100 × probability |
| `<name>_crypsis.pml` | a PyMOL script that colours the structure by probability and shows the top three sites |

`examples/kras_g12c.sh` downloads 4LDJ and runs the prediction; then load `kras_out/4LDJ_crypsis.pml` in PyMOL.

### Python

```python
import crypsis

res = crypsis.predict("4ldj.cif", chains=["A"], keep_cofactors={"GDP", "MG"})
print(res["model"], res["esm"])          # "hybrid 650M", or "physics None" without PyTorch
for site in res["sites"][:3]:
    print(site["rank"], site["kind"], site["residues"], site["docking_box"])

prob = dict(zip(res["residues"], res["probability"]))   # {"A_12": 0.31, ...}
```

If you already have ESM-2 embeddings, pass them as `embeddings=`: 650M, layer 33, a `[n_residues, 1280]` array;
or, with `esm="150M"`, layer 30, `[n_residues, 640]`.

### Keep the cofactors

When a tightly bound cofactor (GDP, heme, NAD, ATP…) is deleted, it leaves an empty, perfectly shaped hole. That
hole then outranks the real cryptic site. With `--keep`, the cofactor counts as occupied space, moves with its
nearest residue in the ensemble, and is not scored. On K-Ras G12C, the switch-II pocket is ranked first only when
GDP is kept.

## Benchmarks

All models were trained and tuned on the CryptoBench training folds only. CryptoBench (Škrhák et al., 2025)
contains apo structures whose cryptic sites are known from holo structures. [BENCHMARK.md](BENCHMARK.md) gives the
protocol, the commands that rebuild every model and number from the public data, and the full development log,
including what did not help.

### CryptoBench test set

The table covers all 222 apo structures (74,046 residues), scored with the shipped models of Crypsis 0.2.1. A
residue is positive when it lines the pocket of any holo record of its apo structure, on any chain of the scored
entity (see [BENCHMARK.md](BENCHMARK.md#protocol); an earlier version of the scripts missed 101 positives in 7
multi-chain entries).
- **AUC and AUPRC** are per residue, with 95% bootstrap confidence intervals over proteins.
- **The site columns** give the share of proteins for which the first predicted site, or one of the first three,
  is a true cryptic site. A protein can have several true sites, one per holo structure.

| Method | ROC AUC [95% CI] | AUPRC [95% CI] | Median AUC per protein | True site ranked 1st | True site in top 3 |
|---|---|---|---|---|---|
| **Crypsis 0.2 hybrid** (physics + ESM-2 650M) | **0.888** [0.870–0.905] | **0.394** [0.350–0.439] | **0.932** | 48% | 62% |
| Crypsis 0.2 hybrid, `--esm 150M` | 0.884 [0.866–0.901] | 0.374 [0.331–0.418] | 0.928 | **50%** | **63%** |
| Crypsis 0.2 physics | 0.864 [0.844–0.882] | 0.323 [0.283–0.364] | 0.895 | 40% | 53% |
| Crypsis 0.1 hybrid (ESM-2 150M) | 0.881 [0.863–0.897] | 0.376 [0.333–0.419] | 0.919 | 46% | 61% |
| Crypsis 0.1 physics | 0.848 [0.828–0.866] | 0.288 [0.251–0.329] | 0.892 | 40% | 50% |
| P2Rank 2.5.1 † | 0.788 [0.769–0.805] | 0.191 [0.163–0.226] | 0.832 | | |
| PocketMiner † | 0.762 [0.738–0.783] | 0.173 [0.149–0.202] | 0.820 | | |

† Run by us on the same files for Crypsis 0.1, with the earlier label rule, on the 221 structures where every
method ran.

**Changes from 0.1 to 0.2.** These are paired bootstrap differences (95% CI):

| Model | AUC | AUPRC | Significant? |
|---|---|---|---|
| Physics | +0.009 to +0.024 | +0.014 to +0.055 | yes, both |
| Hybrid (650M) | +0.002 to +0.013 | +0.001 to +0.036 | yes, both |
| Hybrid (150M) | −0.002 to +0.008 | −0.018 to +0.012 | no |

**Published results on CryptoBench test subsets.** These numbers come from the papers, not from our runs, and
they are not computed on exactly the same residues as the table above:
- Seq2Pocket and its fine-tuned ESM-2 3B baseline evaluate the 184 test structures whose binding sites lie on a
  single chain. On that subset (our runs, post hoc), Crypsis 0.2 scores AUC 0.886 [0.864–0.906] and AUPRC
  0.420 [0.370–0.473].
- pLM-NN, the CryptoBench paper's baseline, is scored on all test structures but chain by chain, and most of that
  paper's statistics come from thresholded (0/1) predictions.

| Method | Test subset | ROC AUC | AUPRC |
|---|---|---|---|
| Seq2Pocket (fine-tuned ESM-2 3B + smoothing; Škrhák et al., 2026) | 184 single-chain | 0.894 | 0.475 |
| **Crypsis 0.2 hybrid** (our run) | 184 single-chain | **0.886** | **0.420** |
| Fine-tuned ESM-2 3B (Škrhák, Gamouh & Hoksza, 2025) | 184 single-chain | 0.871 | 0.382 |
| **Crypsis 0.2 hybrid** (our run) | all 222 | **0.888** | **0.394** |
| pLM-NN, MLP on frozen ESM-2 3B (CryptoBench paper) | all, chain by chain | 0.86 | 0.36 |

Crypsis is not the most precise predictor on this benchmark. Seq2Pocket's fine-tuned 3-billion-parameter language
model ranks residues more precisely (AUPRC 0.475 against 0.420 on the same subset). Crypsis is close in AUC, uses
a frozen and much smaller language model, and adds what a sequence model cannot: explicit pockets with volumes, how they open, and a docking box.

A residue picked at random is a cryptic-site residue 5.4% of the time, so that is the AUPRC of a random
predictor. Per protein, the 0.2 hybrid has both the fewest failures and the most near-perfect predictions:

| Model | Proteins with AUC < 0.6 | Proteins with AUC > 0.9 |
|---|---|---|
| Crypsis 0.1 physics | 17.6% | 47.7% |
| Crypsis 0.2 physics | 14.4% | 49.5% |
| Crypsis 0.1 hybrid | 12.6% | 56.8% |
| Crypsis 0.2 hybrid | 12.6% | **58.1%** |

### Independent benchmark: the PocketMiner set

These proteins come from the PocketMiner paper (Meller et al., 2023). The labels are pooled as in that paper:
pocket residues are positives, and residues of negative-example proteins are negatives. "Unseen" test proteins
have no chain with at least 50% sequence identity in the CryptoBench training folds.

| Method | Test set, 35 proteins | Test proteins unseen in CryptoBench training, 20 | Validation set, 26 proteins |
|---|---|---|---|
| **Crypsis 0.2 hybrid** | **0.919** [0.866–0.963] | **0.888** [0.809–0.954] | 0.854 [0.692–0.946] |
| Crypsis 0.2 physics | 0.891 [0.825–0.946] | 0.863 [0.764–0.945] | 0.816 [0.622–0.932] |
| Crypsis 0.1 hybrid | 0.907 [0.863–0.947] | 0.882 [0.821–0.937] | 0.846 [0.671–0.944] |
| Crypsis 0.1 physics | 0.877 [0.819–0.936] | 0.848 [0.762–0.935] | 0.803 [0.572–0.935] |
| PocketMiner † | 0.868 [0.783–0.935] | – | **0.897** [0.840–0.955] |
| P2Rank † | 0.624 [0.528–0.717] | – | 0.679 [0.544–0.898] |

† Run by us for Crypsis 0.1 on the same files and labels. Their unseen-subset values are left out because that
subset is now defined by sequence identity (`scripts/pocketminer_set.py`).

PocketMiner's own value on its test set (0.87) is reproduced. PocketMiner used the validation set for model
selection, which favours it in the last column. The confidence intervals are wide because the sets are small.

### Eight cryptic sites from the literature

The true site is every apo residue within 4.5 Å of the drug or ligand in the holo structure. Each cell gives the
AUC within the protein and the rank of the first predicted site that finds the true one. Both versions are scored
with the same labels (`scripts/literature_cases.py`). K-Ras is run with GDP kept as a cofactor.

| Protein and site (apo → holo) | 0.1 physics | 0.1 hybrid | 0.2 physics | 0.2 hybrid |
|---|---|---|---|---|
| K-Ras G12C + GDP: switch-II pocket, sotorasib (4LDJ → 6OIM) | 0.89, not found | 0.88, **#1** | 0.84, **#1** | 0.88, **#1** |
| TEM-1 β-lactamase: opens when helices 11/12 separate (1ZG4 → 1PZO) | 0.55, not found | 0.62, **#1** | 0.51, not found | 0.59, not found |
| BCL-xL: BH3 groove, ABT-737 (1MAZ → 2YXJ) | 0.91, **#1** | 0.86, **#1** | 0.91, **#1** | 0.90, **#1** |
| MDM2: p53 cleft, nutlin (1Z1M → 1RV1) | 0.78, not found | 0.83, **#1** | 0.77, not found | 0.79, not found |
| Glucokinase: allosteric activator site (1V4T → 1V4S) | 0.79, #3 | 0.67, not found | 0.67, **#1** | 0.64, not found |
| PTP1B: allosteric site ~20 Å from the active site; helix α7 is not modelled in either structure (2CM2 → 1T49) | 0.39, not found | 0.40, not found | 0.57, not found | 0.59, not found |
| p38α: C-lobe lipid pocket (1WFC → 3HVC) | 0.37, not found | 0.40, not found | 0.62, not found | 0.49, not found |
| HIV-1 RT: NNRTI pocket, Tyr181 and Tyr188 rotate (1DLO → 1VRT) | 0.69, not found | 0.63, not found | 0.66, not found | 0.66, not found |

Eight cases are anecdotes next to 222 test proteins, and here 0.2 does not beat 0.1.
- **0.1 hybrid:** ranks four of the eight sites first.
- **0.2 hybrid:** ranks two first.
- **TEM-1 and MDM2:** the 0.1 hybrid ranks these sites first; the 0.2 hybrid misses them.
- **PTP1B, p38α and HIV-1 RT:** no version finds these sites. They need unfolding, large side-chain rotations, or
  a site absent from the apo crystal.

## Speed and memory

These are wall-clock times for the whole command (reading, ensembles, grid, descriptors, model and output) on a
2018 laptop: Intel Core i7-8850H, 6 cores. Each run is a fresh process, and the hybrid times include loading ESM-2
from the local cache. A recent Apple-silicon laptop is about three times faster.

| Protein | Residues | 0.2 physics | 0.2 hybrid, 150M | 0.2 hybrid, 650M | 0.1 hybrid | Peak memory (physics / 150M / 650M) |
|---|---|---|---|---|---|---|
| Lysozyme (2LZT) | 129 | 5.6 s | 9.1 s | 17.5 s | 6.3 s | 0.08 / 1.40 / 5.42 GB |
| TEM-1 (1ZG4) | 263 | 9.8 s | 14.8 s | 22.3 s | 9.1 s | 0.08 / 1.40 / 5.42 GB |
| Glucokinase (1V4T) | 424 | 16.5 s | 20.8 s | 29.9 s | 13.2 s | 0.10 / 1.41 / 5.44 GB |
| HIV-1 RT heterodimer (1DLO) | 971 | 31.2 s | 36.8 s | 50.3 s | 22.5 s | 0.23 / 1.41 / 5.40 GB |

The 0.2 physics model takes about twice as long as 0.1 because it samples two more ensembles. With less than 8 GB
of RAM, use `--esm 150M`.

## How it works

1. **Motions.** An anisotropic network model is built on the Cα atoms (15 Å cutoff, sparse Hessian). Its 30
   lowest non-trivial modes are kept. A Gaussian network model (7.3 Å) gives the slow and fast fluctuations and the
   hinges, where two parts of the protein move against each other.
2. **Ensembles.** Each conformer is a random thermal combination of the modes, and side chains and kept cofactors
   move with their residue. There are three ensembles of 24 conformers:
   - scaled to a Cα RMSD of 3 Å;
   - scaled to 5 Å;
   - 3 Å with the side chains cut back to Cβ, because many cryptic pockets are hidden by a few side chains.
3. **Pocket grid (C++).** On a 1 Å grid, a point is a pocket point when two conditions hold:
   - a water-sized probe fits there;
   - at least 70% of 30 rays meet protein within 10 Å.

   A second, shallower pass uses 55% of the rays. Connected points form pockets, and each residue gets the score
   of the best pocket it lines.
4. **Descriptors.** For each residue, Crypsis computes:
   - pocket score, volume, opening frequency and gain, in the structure and across each ensemble;
   - pockets in the structure with side chains cut back to Cβ and to Cγ;
   - relative solvent accessibility of the residue and its side chain, and half-sphere exposure;
   - contact numbers at 6–16 Å, and depth relative to the radius of gyration;
   - the hydrophobic fraction of the exposed surface around the residue;
   - the Cα virtual angle and dihedral, and the distance to a chain end;
   - the Gaussian-network fluctuations and the distance to the nearest hinge;
   - B-factor, hydrophobicity, an amino-acid propensity, and patch averages.

   This gives 49 columns in total, rank-normalized within the protein where scales differ between proteins.
5. **Models.**

   | Model | Inputs | Learner | Test AUC |
   |---|---|---|---|
   | `hybrid` | 49 physics columns + 256 principal components of the ESM-2 650M embeddings (layer 33) | gradient-boosted trees | 0.888 |
   | `hybrid --esm 150M` | 49 physics columns + 128 principal components of ESM-2 150M (layer 30) | gradient-boosted trees | 0.884 |
   | `physics` | 49 physics columns | gradient-boosted trees | 0.864 |
   | `v1` | 3 ensemble descriptors | logistic regression | 0.807 * |

   \* Scored with the earlier label rule (see [CryptoBench test set](#cryptobench-test-set)).

6. **Sites.** The top 10% of residues are grouped into sites by Cα proximity (6.5 Å links, within 12 Å of the
   seed residue). Sites are ranked by summed probability. Each site's kind compares its pocket volume in the
   structure with its volume in the ensemble.

The trees are shipped as plain arrays (`src/crypsis/models/*.npz`) and evaluated by a small NumPy evaluator. There
is no pickle and no scikit-learn at prediction time. `scripts/export_models.py` documents the export and checks
that the evaluator reproduces scikit-learn's probabilities exactly.

**Reproducibility.** The sign of each normal mode is arbitrary, and the LAPACK build decides it. Since 0.2.1,
Crypsis fixes every mode's sign (the sum of the cubes of its components is positive), so different computers and
Python environments sample the same ensemble and give the same probabilities. In 0.2.0, another environment could
flip a few modes and change probabilities by up to 0.1.

## Limitations

- **Precision.** A fine-tuned 3-billion-parameter language model (Seq2Pocket) ranks cryptic-site residues more
  precisely: AUPRC 0.475 against 0.420 on the 184 single-chain test structures. Crypsis's physics descriptors seem
  to have reached what the apo structure alone can tell. Adding probe mapping, larger ensembles or single-side-chain moves did not improve them (see
  [BENCHMARK.md](BENCHMARK.md)).
- **Very large rearrangements.** Some sites are missed by every method tested:
  - sites next to segments that are disordered in the crystal (PTP1B: helix α7 is not modelled in either structure);
  - sites that need large side-chain rotations (HIV-1 RT NNRTI pocket);
  - sites absent from the apo crystal structure (p38α lipid pocket).
- **Elastic-network motions.** The ensembles are approximate and only fast to compute; they are not molecular
  dynamics. Use the predicted docking box as a starting point for docking or simulations.
- **Probabilities rank residues within a protein.** Compare sites within one structure, not absolute
  probabilities across proteins.
- **Memory.** The default hybrid model loads ESM-2 650M, about 5.4 GB. `--esm 150M` needs 1.4 GB.

## Also in Leonardo

Crypsis was developed in [Leonardo](https://github.com/juvenalyosa/Leonardo), an open-source molecular modelling
suite for PyMOL, and it is also built in there. In Leonardo the method has a graphical window and an AI-agent
tool, and the predicted docking box goes straight into docking or molecular dynamics.

## Citing

A manuscript describing Crypsis is in preparation; see `CITATION.cff`. Please also cite the work Crypsis builds
on:

- Škrhák, V., Novotný, M., Feidakis, C. P., Krivák, R., & Hoksza, D. (2025). CryptoBench: Cryptic protein–ligand
  binding sites dataset and benchmark. *Bioinformatics, 41*(1), btae745. https://doi.org/10.1093/bioinformatics/btae745
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model.
  *Science, 379*(6637), 1123–1130. https://doi.org/10.1126/science.ade2574
- Atilgan, A. R., et al. (2001). Anisotropy of fluctuation dynamics of proteins with an elastic network model.
  *Biophysical Journal, 80*(1), 505–515. https://doi.org/10.1016/S0006-3495(01)76033-X

The tools and methods in the comparison are:

- Krivák, R., & Hoksza, D. (2018). P2Rank. *Journal of Cheminformatics, 10*, 39.
  https://doi.org/10.1186/s13321-018-0285-8
- Meller, A., et al. (2023). PocketMiner. *Nature Communications, 14*, 1177.
  https://doi.org/10.1038/s41467-023-36699-3
- Škrhák, V., Polák, L., Novotný, M., & Hoksza, D. (2026). Seq2Pocket: augmenting protein language models for
  spatially consistent binding site prediction. *Bioinformatics, 42*(10), btag778.
  https://doi.org/10.1093/bioinformatics/btag778
- Škrhák, V., Gamouh, H., & Hoksza, D. (2025). Hidden in protein sequences: Predicting cryptic binding sites.
  *Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health
  Informatics (ACM-BCB 2025)*. https://doi.org/10.1145/3765612.3767221

## License

MIT. The ESM-2 weights are distributed by Meta under the MIT license.

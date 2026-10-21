# Crypsis

**Find the binding pockets a protein structure is hiding.**

*Crypsis* is the biological term for an organism avoiding detection. Cryptic pockets do something similar: they
are absent or closed in the apo crystal structure, then open when the protein moves or a ligand binds. Many
"undruggable" targets have one, for example the switch-II pocket of K-Ras G12C, the target of sotorasib.

Crypsis takes a single structure and keeps the physics explicit at every step:

1. It computes the low-frequency motions of the fold with an **anisotropic network model**.
2. It builds an ensemble of conformers along those motions.
3. It finds buried, probe-accessible pocket space in every conformer on a **C++ pocket grid**.
4. A gradient-boosted model, optionally combined with **ESM-2** protein-language-model embeddings, turns these
   physical descriptors into a probability for every residue.

Each predicted site is reported with its pocket volume in the input structure and along the motions. That tells
you whether the pocket is **already open**, **grows with motion** or **forms with motion**, and gives you a ready
docking box.

```
$ crypsis predict 4ldj.cif --chains A --keep GDP,MG
crypsis hybrid model, 170 residues, 2 sites (3 s)
rank  score   n  kind               volume A3 (structure -> ensemble)  residues
   1   1.84   5  forms with motion  9 -> 23                            Ser39 Leu56 Thr58 Ala59 Gly60
   2   1.58   5  forms with motion  12 -> 27                           Lys5 Arg68 Tyr71 Thr74 Gly75
```

The input is K-Ras G12C bound to GDP (PDB 4LDJ), a structure without the drug. The switch-II pocket is ranked
first.

## Performance

All models were trained on the CryptoBench training folds only. CryptoBench (Škrhák et al., 2025) contains apo
structures whose cryptic sites were observed in holo structures. The table shows results on the **CryptoBench
test set**: the 221 apo structures on which all methods ran. AUC and AUPRC are per residue. The site columns give
the share of proteins whose true cryptic site is the first, or among the first three, predicted sites.

| Method | ROC AUC | AUPRC | Top-1 site | Top-3 sites |
|---|---|---|---|---|
| **Crypsis hybrid** (physics + ESM-2 150M) | **0.881** | **0.375** | **44%** | **61%** |
| Crypsis physics only | 0.846 | 0.283 | – | – |
| P2Rank 2.5.1 | 0.788 | 0.191 | 34% | 49% |
| PocketMiner | 0.762 | 0.173 | 30% | 44% |

On all 222 test structures the hybrid model reaches a ROC AUC of 0.881 [95% CI 0.862–0.897] and an AUPRC of
0.372 [0.328–0.415]. On the independent PocketMiner test set (35 proteins) the hybrid reaches an AUC of 0.907,
against 0.868 for PocketMiner itself.

These numbers come from the accompanying manuscript, which is in preparation. The weak spots are honest ones:
sites that need a helix to unfold (PTP1B), sites that are absent from the apo crystal structure (p38α lipid
pocket), and sites that open only through large side-chain rotations (HIV-1 RT NNRTI pocket) are missed by every
method tested.

## Install

```bash
pip install "crypsis[esm] @ git+https://github.com/juvenalyosa/crypsis"   # hybrid model (PyTorch + fair-esm)
pip install "git+https://github.com/juvenalyosa/crypsis"                  # physics model only, no PyTorch
```

The core needs only NumPy, SciPy and gemmi.

**C++ extension.** It is compiled during installation when a C++17 compiler is present: Xcode Command Line Tools
on macOS, or g++ on Linux. Without a compiler, Crypsis uses the NumPy version of the same algorithm, which gives
the same results about 15 times more slowly.

**ESM-2 weights.** The hybrid model downloads the ESM-2 150M weights (about 600 MB) on first use. To check what is
available on your installation, run `crypsis check`.

## Use

### Command line

```bash
crypsis predict structure.cif                       # all protein chains, best available model
crypsis predict 1dlo.cif --chains A,B               # a heterodimer
crypsis predict 4ldj.cif --keep GDP,MG              # keep bound cofactors as occupied space
crypsis predict 4ldj.cif --keep auto                # ... or every hetero group of >= 8 heavy atoms
crypsis predict structure.pdb --model physics       # hybrid | physics | v1
crypsis predict structure.pdb --device mps          # run ESM-2 on Apple GPU (or cuda)
```

Each run writes four files:

- `<name>_residues.csv`: the probability for each residue, how often a pocket opened next to it in the ensemble,
  and its site.
- `<name>_sites.json`: the ranked sites, each with its residues, kind, pocket volumes and docking box
  (`center`, `size` in Å).
- `<name>_crypsis.pdb`: the input structure with each B-factor set to 100 × probability.
- `<name>_crypsis.pml`: a PyMOL script that colours the structure by probability and shows the top sites.

### Python

```python
import crypsis

res = crypsis.predict("4ldj.cif", chains=["A"], keep_cofactors={"GDP", "MG"})
print(res["model"])                    # "hybrid" (or "physics" without PyTorch)
for site in res["sites"][:3]:
    print(site["rank"], site["kind"], site["residues"], site["docking_box"])
prob = dict(zip(res["residues"], res["probability"]))
```

If you already have ESM-2 150M embeddings (layer 30, one row of 640 values per residue), pass them as
`embeddings=`.

### Keep the cofactors

When a tightly bound cofactor (GDP, heme, NAD, ATP…) is deleted, it leaves an empty, perfectly shaped hole. That
hole then outranks the real cryptic site. With `--keep`, the cofactor counts as occupied space, moves with its
nearest residue in the ensemble, and is not scored. On K-Ras G12C, the switch-II pocket comes first only with GDP
kept.

## Models

| Model | What it uses | Test AUC |
|---|---|---|
| `hybrid` | 22 physics columns + 128 principal components of ESM-2 150M embeddings, gradient-boosted trees | 0.881 |
| `physics` | 22 physics columns, gradient-boosted trees | 0.846 |
| `v1` | logistic model on 3 ensemble descriptors | 0.807 |

The 22 physics columns are built as follows:

- 14 per-residue descriptors, rank-normalized within the protein: pocket score and volume in the structure and
  the ensemble, opening frequency, gain, burial, hydrophobicity, B-factor, mode mobility, and their patch means.
- 3 of them smoothed over the residue graph (8 Å, σ = 4 Å).
- The residue's amino-acid log-odds propensity.
- A shallower pocket pass (burial ≥ 0.55) in the structure and the ensemble.

The trees are stored as plain arrays (`src/crypsis/models/*.npz`). They are evaluated by a small NumPy evaluator,
so there is no pickle and no scikit-learn at prediction time. `scripts/export_models.py` documents the export and
checks that the evaluator reproduces scikit-learn's probabilities exactly.

**Reproducibility.** Each normal mode is an eigenvector whose sign is arbitrary, and the LAPACK build decides it.
PyTorch loads its own libraries. If `torch` is imported before Crypsis computes the modes, a few modes can flip
sign, which slightly changes the sampled ensemble and therefore the probabilities. Crypsis imports PyTorch only
after the physics is done. For numbers identical to the published ones, run each structure in a fresh process,
which is what the command line does.

## Also in Leonardo

Crypsis is also built into [Leonardo](https://github.com/juvenalyosa/Leonardo), the open-source PyMOL modelling
suite where it was developed. There it has a window and an agent tool, and the predicted site's docking box goes
straight into docking or molecular dynamics.

## Cite

If you use Crypsis, please cite the manuscript (in preparation; see `CITATION.cff`) and the work it builds on:

- Škrhák, V., Novotný, M., Feidakis, C. P., Krivák, R., & Hoksza, D. (2025). CryptoBench: Cryptic protein–ligand
  binding sites dataset and benchmark. *Bioinformatics, 41*(1), btae745. https://doi.org/10.1093/bioinformatics/btae745
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model.
  *Science, 379*(6637), 1123–1130. https://doi.org/10.1126/science.ade2574
- Atilgan, A. R., et al. (2001). Anisotropy of fluctuation dynamics of proteins with an elastic network model.
  *Biophysical Journal, 80*(1), 505–515. https://doi.org/10.1016/S0006-3495(01)76033-X

## License

MIT. The ESM-2 weights are distributed by Meta under the MIT license.

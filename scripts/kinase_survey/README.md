# Kinase survey

Crypsis on every PDB chain of eight kinases (EGFR, ALK, HER2, ROS1, MET, RET, ABL1, BRAF), with fpocket 4.0 and the
deposited ligands as independent evidence. The paper's run used Crypsis 0.2.1 (`CRYPSIS_SRC` pointing at a v0.2.1
checkout) on the PDB as of 2026-11-02.

```
python fetch_meta.py  DATA/meta                     # UniProt kinase domains + PDBe SIFTS chain list
DATA=... CRYPSIS_SRC=... FPOCKET=... ./run_all.sh   # one process per chain: Crypsis, ligands, interface, fpocket
python aggregate.py   DATA                          # consensus sites -> DATA/results/consensus_sites.json
python report.py      DATA                          # -> DATA/results/consensus_sites.tsv
python abl1_myristoyl.py one PDB CHAIN; ... summary # the ABL1 myristoyl pocket
```

Structures are read from `DATA/raw_cif/{pdb}_updated.cif`, the PDBe updated mmCIF files (with SIFTS columns),
decompressed from the downloads in `DATA/raw/`. gemmi 0.7.5 segfaulted after many reads in one process, so every
chain runs in its own process and the `_atom_site` loop is parsed in Python; one entry (6P7G) is excluded because it
segfaulted even then.

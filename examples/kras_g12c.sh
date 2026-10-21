#!/usr/bin/env bash
# K-Ras G12C bound to GDP, without the drug (PDB 4LDJ): the switch-II pocket of sotorasib should come first.
set -euo pipefail
curl -sLO https://files.rcsb.org/download/4LDJ.cif
crypsis predict 4LDJ.cif --chains A --keep GDP,MG -o kras_out
echo "open in PyMOL: cd kras_out && pymol 4LDJ_crypsis.pml"

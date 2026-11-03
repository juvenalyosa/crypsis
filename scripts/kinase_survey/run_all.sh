#!/bin/zsh
# Kinase survey: every chain in its own process; a crash is logged and does not stop xargs.
#   DATA=<run dir> PY=<python> CRYPSIS_SRC=<source tree, optional> FPOCKET=<fpocket binary> run_all.sh [jobs file]
# The paper's run: DATA=~/crypsis_data/kinases_v021 with CRYPSIS_SRC pointing at a v0.2.1 checkout.
PY=${PY:-python}
D=${DATA:-~/crypsis_data/kinases_v021}
S=${0:A:h}
JOBS=${1:-$D/meta/jobs.txt}
[ -n "$1" ] || $PY -I $S/run_chains.py jobs $D > $JOBS
run() { xargs -P 6 -n 3 sh -c "$PY -I $S/run_chains.py chain $D \$0 \$1 \$2 || echo \"\$0 \$1 \$2 CRASH exit \$?\"; exit 0" < $JOBS; }
echo "== pass 1 $(date) $(wc -l < $JOBS) chains"; run > $D/meta/pass1.log 2>&1
echo "== emb $(date)"; $PY -I $S/run_chains.py emb $D
echo "== pass 2 $(date)"; run > $D/meta/pass2.log 2>&1
echo "== done $(date)"

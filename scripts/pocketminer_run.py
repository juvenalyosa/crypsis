#!/usr/bin/env python3
"""Run PocketMiner (Meller et al., Nat. Commun. 2023; model models/pocketminer of https://github.com/Mickdub/gvp,
branch pocket_pred) on every PDB file of a folder, one structure at a time.  Runs in PocketMiner's own environment
(TensorFlow, mdtraj); the files come from `scripts/baselines.py pocketminer-prep`.

    python scripts/pocketminer_run.py <gvp>/src <work>/pocketminer_in <work>/pocketminer_out

Writes <name>.npy with one probability per residue, in file order; scripts/baselines.py pocketminer maps them back.
"""

import os
import sys

src, pdb_dir, out_dir = sys.argv[1:4]
sys.path.insert(0, src)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import mdtraj as md  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from models import MQAModel  # noqa: E402
from util import load_checkpoint  # noqa: E402
from validate_performance_on_xtals import process_strucs  # noqa: E402

# the settings of src/xtal_predict.py
model = MQAModel(node_features=(8, 50), edge_features=(1, 32), hidden_dim=(16, 100), num_layers=4, dropout=0.1)
loaded = False
os.makedirs(out_dir, exist_ok=True)
names = sorted(f[:-4] for f in os.listdir(pdb_dir) if f.endswith(".pdb"))
for k, name in enumerate(names, 1):
    out = os.path.join(out_dir, name + ".npy")
    if os.path.exists(out):
        continue
    try:
        X, S, mask = process_strucs([md.load(os.path.join(pdb_dir, name + ".pdb"))])
        if not loaded:
            load_checkpoint(model, tf.keras.optimizers.Adam(), os.path.join(src, "..", "models", "pocketminer"))
            loaded = True
        p = np.asarray(model(X, S, mask, train=False, res_level=True))[0][: int(mask.sum())]
        np.save(out, p.astype(np.float32))
    except Exception as exc:
        print(name, "failed:", type(exc).__name__, str(exc)[:200], flush=True)
    if k % 25 == 0 or k == len(names):
        print(f"[{k}/{len(names)}]", flush=True)

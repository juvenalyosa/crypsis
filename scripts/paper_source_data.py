#!/usr/bin/env python3
"""Source data of the main figures as CSV files, from the same result files as paper_figures.py.

    python scripts/paper_source_data.py      # benchmark environment -> paper/source_data/fig*.csv

fig1_kras.csv            per-residue apo and ensemble pocket score, hybrid probability, sotorasib-site flag (4LDJ)
fig2_models.csv          pooled ROC AUC and AUPRC with 95% CIs, prevalence, per set and model
fig3_differences.csv     paired differences with 95% CIs, per-protein wins and Holm-adjusted Wilcoxon P values
fig4_perturbation.csv    perturbation controls: pooled metrics and differences to the shipped ensemble
fig5_per_protein.csv     per-protein ROC AUC of every model, with the site state (CryptoBench only)
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cryptobench as cb  # noqa: E402

D = os.path.expanduser("~/crypsis_data")
OUT = os.path.join(HERE, "..", "paper", "source_data")
SETS = {"cryptobench": (f"{D}/cryptobench", "work/heldout_cryptobench.json", "work/perturbation_controls.json"),
        "temporal": (f"{D}/temporal_bench", "work_temporal/heldout_temporal.json",
                     "work_temporal/perturbation_controls.json")}


def load(rel):
    with open(os.path.join(D, rel)) as fh:
        return json.load(fh)


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("->", name, len(rows), "rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "..", "paper", "figures", "data.json")) as fh:
        data = json.load(fh)
    k = data["kras"]
    cols = [c for c in k if isinstance(k[c], list) and len(k[c]) == len(k[next(iter(k))])]
    write("fig1_kras.csv", cols, list(zip(*[k[c] for c in cols])))

    rows, diffs, pert, per = [], [], [], []
    for s, (data_dir, held, pc) in SETS.items():
        h = load(held)["all"]
        for m, v in h["models"].items():
            rows.append([s, m, v["proteins"], v["residues"], v["positive_rate"], v["auc"], *v["auc_ci"], v["auprc"],
                         *v["auprc_ci"], v["site_top1"], *v["site_top1_ci"], v["site_top3"], *v["site_top3_ci"]])
        for m, v in h["differences"].items():
            p = v["per_protein"]
            diffs.append([s, m, *v["auc_diff_ci"], *v["auprc_diff_ci"], p["n"], p["auc_wins"],
                          p["auc_wilcoxon_p_holm"], p["auprc_wins"], p["auprc_wilcoxon_p_holm"]])
        c = load(pc)
        for name, v in [("static", c["static"])] + list(c["controls"].items()):
            d = v.get("slow_minus_this", {})
            pert.append([s, name, v["auc"], *v["auc_ci"], v["auprc"], *v["auprc_ci"], v["positive_rate"],
                         *(d.get("auc_diff_ci") or ["", ""]), *(d.get("auprc_diff_ci") or ["", ""])])
        pp = data["per_protein_auc"][s]
        ids = [f"{apo}_{chain}" for apo, chain, _pos in cb.entries(data_dir, "test")]
        models = [m for m in pp if m != "site_state"]
        assert all(len(pp[m]) == len(ids) for m in models)
        for i, pid in enumerate(ids):
            per.append([s, pid, *[pp[m][i] for m in models], pp["site_state"][i] if "site_state" in pp else ""])
    write("fig2_models.csv", ["set", "model", "proteins", "residues", "positive_rate", "auc", "auc_lo", "auc_hi",
                              "auprc", "auprc_lo", "auprc_hi", "site_top1", "top1_lo", "top1_hi", "site_top3",
                              "top3_lo", "top3_hi"], rows)
    write("fig3_differences.csv", ["set", "comparison", "dauc_lo", "dauc_hi", "dauprc_lo", "dauprc_hi", "proteins",
                                   "auc_wins", "auc_p_holm", "auprc_wins", "auprc_p_holm"], diffs)
    write("fig4_perturbation.csv", ["set", "control", "auc", "auc_lo", "auc_hi", "auprc", "auprc_lo", "auprc_hi",
                                    "positive_rate", "slow_minus_this_dauc_lo", "slow_minus_this_dauc_hi",
                                    "slow_minus_this_dauprc_lo", "slow_minus_this_dauprc_hi"], pert)
    write("fig5_per_protein.csv", ["set", "protein", *models, "site_state"], per)


if __name__ == "__main__":
    main()

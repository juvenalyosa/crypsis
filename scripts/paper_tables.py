#!/usr/bin/env python3
"""LaTeX bodies of the supplementary tables added in the second round of review, written from the result files so
that no number is typed by hand.

    python scripts/paper_tables.py      -> paper/tables/*.tex (\\input by supplementary.tex)

  tableS4_temporal_sensitivity.tex   work_temporal/temporal_sensitivity.json (scripts/temporal_sensitivity.py)
  tableS7b_site_state.tex            work/site_state.json (scripts/site_state.py)
  tableS7c_site_rules.tex            work/site_state.json
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "tables")
D = os.path.expanduser("~/crypsis_data")


def half_up(x, nd=3):
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pfmt(p):
    if p >= 1e-3:
        return f"{float(f'{p:.2g}'):g}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def num(x, nd=3):
    return half_up(x, nd).replace("-", "$-$")


def ci(v):
    return f"[{num(v[0])}--{num(v[1])}]"


def site_state():
    d = json.load(open(os.path.join(D, "work", "site_state.json")))
    names = {"ligand": "Ligand", "fpocket": "fpocket", "crypsis": "Crypsis"}
    sets = {"cryptobench": "CB", "temporal": "Temp."}
    rows = []
    for s, sl in sets.items():
        for df in ("ligand", "fpocket", "crypsis"):
            for st, lab in (("closed", "closed"), ("partly_open", "partly open"), ("open", "open")):
                r = d[s]["definitions"][df]["strata"][st]
                m = r["mean_auc"]
                t = r["tests"]
                pe, hp = t["physics - esm_only_650M"], t["hybrid_650M - physics"]
                rows.append(" & ".join([sl if df == "ligand" and st == "closed" else "",
                                        names[df] if st == "closed" else "", lab, str(r["n"]),
                                        half_up(m["hybrid_650M"]), half_up(m["physics"]), half_up(m["esm_only_650M"]),
                                        half_up(m["p2rank"]) if "p2rank" in m else "--",
                                        ci(pe["mean_diff_ci"]), pfmt(pe["wilcoxon_p_holm"]),
                                        ci(hp["mean_diff_ci"]), pfmt(hp["wilcoxon_p_holm"])]) + " \\\\")
            rows.append("\\addlinespace")
    return "\n".join(rows[:-1])


def site_rules():
    d = json.load(open(os.path.join(D, "work", "site_state.json")))
    models = [("hybrid_650M", "Hybrid 650M"), ("physics", "Physics"), ("esm_only_650M", "ESM-2 650M, PCs"),
              ("p2rank", "P2Rank 2.5.1"), ("pocketminer", "PocketMiner"), ("fpocket", "fpocket 4.0")]
    rules = ["default_top1", "default_top3", "cover50_top1", "dcc8_top1", "dcc8_top3", "dcc4_top1"]
    rows = []
    for s, sl in (("cryptobench", "CryptoBench"), ("temporal", "Temporal")):
        for k, (m, lab) in enumerate(m for m in models if m[0] in d[s]["sites"]):
            r = d[s]["sites"][m]
            rows.append(" & ".join([sl if k == 0 else "", lab] +
                                   [f"{half_up(100 * r[x]['rate'], 0)}\\%" for x in rules]) + " \\\\")
        rows.append("\\addlinespace")
    return "\n".join(rows[:-1])


def temporal_sensitivity():
    d = json.load(open(os.path.join(D, "work_temporal", "temporal_sensitivity.json")))
    lab = {"all": "As built", "records<=10": "$\\leq$10 records per protein", "positives<=15%": "$\\leq$15\\% positives",
           "no_glycan_lipid": "No glycan/covalent/lipid", "no_altloc": "No alternate locations",
           "main_record_only": "Main record only", "curated": "Curated (first three)"}
    rows = []
    for k, r in d.items():
        name = lab.get(k, k.replace("holo_", "Holo in "))
        m = r["models"]
        pe = r["differences"]["physics - esm_only_650M"]
        hp = r["differences"]["hybrid_650M - physics"]
        rows.append(" & ".join([name, str(r["proteins"]), str(r["records"]),
                                f"{half_up(100 * m['physics']['positive_rate'], 1)}\\%",
                                half_up(m["hybrid_650M"]["auc"]), half_up(m["physics"]["auc"]),
                                half_up(m["esm_only_650M"]["auc"]), ci(pe["auc_diff_ci"]), pfmt(pe["auc_diff_p_holm"]),
                                pfmt(pe["per_protein"]["auc_wilcoxon_p_holm"]), ci(hp["auc_diff_ci"])]) + " \\\\")
    return "\n".join(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("tableS7b_site_state.tex", site_state), ("tableS7c_site_rules.tex", site_rules),
                     ("tableS4_temporal_sensitivity.tex", temporal_sensitivity)):
        try:
            body = fn()
        except FileNotFoundError as exc:
            print("skipped", name, exc)
            continue
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(body + "\n")
        print("->", name)


if __name__ == "__main__":
    main()

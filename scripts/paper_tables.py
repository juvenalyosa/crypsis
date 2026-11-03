#!/usr/bin/env python3
"""LaTeX bodies of the supplementary tables added in the second round of review, written from the result files so
that no number is typed by hand.

    python scripts/paper_tables.py      -> paper/tables/*.tex (\\input by supplementary.tex)

  tableS4_temporal_sensitivity.tex   work_temporal/temporal_sensitivity.json (scripts/temporal_sensitivity.py)
  tableS7b_site_state.tex            work/site_state.json (scripts/site_state.py)
  tableS7c_site_rules.tex            work/site_state.json
  tableS10_untrained.tex             work/untrained_physics.json (scripts/untrained_physics.py)
  tableS11a_site_size.tex            work/secondary_sites.json (scripts/secondary_sites.py)
  tableS11b_secondary.tex            work/secondary_sites.json
  tableS12_kinases.tex               kinases_v021/results/consensus_sites.json (scripts/kinase_survey)
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
                                half_up(m["esm_only_650M"]["auc"]), ci(pe["auc_diff_ci"]),
                                ("$\\leq$" if pe["auc_diff_p"] <= 1 / 2000 else "") + pfmt(pe["auc_diff_p_holm"]),
                                pfmt(pe["per_protein"]["auc_wilcoxon_p_holm"]), ci(hp["auc_diff_ci"])]) + " \\\\")
    return "\n".join(rows)


def untrained():
    d = json.load(open(os.path.join(D, "work", "untrained_physics.json")))
    sets = {"cryptobench_train_folds": "CB folds", "cryptobench_test": "CB test", "temporal": "Temporal"}
    lab = {"static (ranked)": "Apo pocket score", "U1 (ranked)": "U1 ensemble pocket score",
           "U2 (ranked)": "U2 ensemble gain", "U3 (ranked)": "U3 rank mean", "hybrid_oof": "Hybrid 650M (out of fold)",
           "physics_trained": "Physics (trained)", "hybrid_trained": "Hybrid 650M (trained)",
           "static (raw)": "Apo pocket score, raw", "U1 (raw)": "U1, raw", "U2 (raw)": "U2, raw"}
    order = ["static (ranked)", "U1 (ranked)", "U2 (ranked)", "U3 (ranked)", "static (raw)", "U1 (raw)", "U2 (raw)",
             "physics_trained", "hybrid_trained", "hybrid_oof"]
    rows = []
    for s, sl in sets.items():
        rs = {r["score"]: r for r in d[s]}
        first = True
        for k in order:
            if k not in rs:
                continue
            r = rs[k]
            rows.append(" & ".join([f"{sl} ({r['proteins']})" if first else "", lab[k],
                                    f"{half_up(r['auc'])} {ci(r['auc_ci'])}", f"{half_up(r['auprc'])} {ci(r['auprc_ci'])}",
                                    half_up(r["auc_per_protein"]), f"{half_up(100 * r['site_top1'], 0)}\\%",
                                    f"{half_up(100 * r['site_top3'], 0)}\\%"]) + " \\\\")
            first = False
        rows.append("\\addlinespace")
    return "\n".join(rows[:-1])


def site_size():
    d = json.load(open(os.path.join(D, "work", "secondary_sites.json")))
    sets = {"cryptobench_oof": "CB folds, OOF", "cryptobench_test": "CB test", "temporal": "Temporal"}
    rows = []
    for s, sl in sets.items():
        b = d[s]["by_size_all_sites"]
        for k, (band, r) in enumerate(b["bands"].items()):
            band_lab = band.replace("-", "--") if not band.endswith("-") else "$\\geq$" + band[:-1]
            rows.append(" & ".join([f"{sl} ({b['proteins']}; {b['sites']})" if k == 0 else "", band_lab,
                                    str(r["sites"]), f"{half_up(100 * r['top3'], 0)}\\%",
                                    f"{half_up(100 * r['top5'], 0)}\\%", half_up(r["pct_median"], 2)]) + " \\\\")
        rows.append("\\addlinespace")
    return "\n".join(rows[:-1])


def site_secondary():
    d = json.load(open(os.path.join(D, "work", "secondary_sites.json")))
    sets = {"cryptobench_oof": "CB folds, OOF", "cryptobench_test": "CB test", "temporal": "Temporal"}
    rows = []
    for s, sl in sets.items():
        r = d[s]
        sm = r["size_matched"]
        rows.append(" & ".join([sl, str(r["proteins"]), str(r["secondary_sites"]),
                                f"{half_up(r['size_median'][0], 0)} / {half_up(r['size_median'][1], 0)}",
                                f"{half_up(100 * r['top5'][0], 0)} / {half_up(100 * r['top5'][1], 0)}\\%",
                                f"{half_up(r['pct_median'][0], 2)} / {half_up(r['pct_median'][1], 2)}",
                                pfmt(r["pct_wilcoxon_p"]) if r["pct_wilcoxon_p"] is not None else "--",
                                str(sm["pairs"]),
                                (f"{half_up(100 * sm['top5'][0], 0)} / {half_up(100 * sm['top5'][1], 0)}\\%"
                                 if sm["pairs"] else "--"),
                                pfmt(sm["binom_p"]) if sm["binom_p"] is not None else "--"]) + " \\\\")
    return "\n".join(rows)


def kinases():
    d = json.load(open(os.path.join(D, "kinases_v021", "results", "consensus_sites.json")))
    rows = []
    for gene, g in d.items():
        first = True
        for c in g["clusters"]:
            if c["called"] < 0.05:
                continue
            cls = ("ATP site" if c["atp_site"] >= 0.5 else "ligand observed" if c["ligand_entries"] > 0
                   else "protein interface" if c["interface_share"] >= 0.3 else "no ligand")
            ds = c["fpocket_ds_max"]
            ligs = ", ".join(n for n, _ in c["ligands_top"][:3]) or "--"
            rows.append(" & ".join([f"{gene} ({g['chains']})" if first else "", " ".join(c["core_labels"]), cls,
                                    half_up(100 * c["called"], 0), half_up(100 * c["top1"], 0),
                                    half_up(100 * c["kind"].get("forms with motion", 0.0), 0),
                                    half_up(100 * c["fpocket_druggable_share"], 0),
                                    f"{half_up(ds[0], 2)} ({ds[1].upper()})" if ds else "--",
                                    str(c["ligand_entries"]), ligs]) + " \\\\")
            first = False
        rows.append("\\addlinespace")
    return "\n".join(rows[:-1])


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("tableS7b_site_state.tex", site_state), ("tableS7c_site_rules.tex", site_rules),
                     ("tableS4_temporal_sensitivity.tex", temporal_sensitivity),
                     ("tableS10_untrained.tex", untrained), ("tableS11a_site_size.tex", site_size),
                     ("tableS11b_secondary.tex", site_secondary), ("tableS12_kinases.tex", kinases)):
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

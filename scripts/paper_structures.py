#!/usr/bin/env python3
"""Figures of predictions on protein structures (Fig. 6 and Supplementary Figs. S3-S5), from the benchmark score
files and the released Crypsis code; rendered with PyMOL.  Second round of review: this replaces the earlier
version, which used one CryptoBench protein per site state and the cryptic-pocket module of Leonardo.

    python scripts/paper_structures.py             # an environment with PyMOL, gemmi, scikit-learn, matplotlib
      -> paper/figures/fig6_structures.pdf, figS3_gallery_cryptobench.pdf, figS4_gallery_temporal.pdf,
         figS5_false_positive.pdf and paper/figures/structures/selection.json (which protein is shown and why)

Inputs: the per-residue scores of scripts/heldout_eval.py (test_shipped_v3_650M_scores.npz for the hybrid and
physics models, control_esm_only_650M_256_scores.npz for the ESM-2 principal-component control), the labels of
cryptobench.entries, and the ligand-based site state and superposition of scripts/holo_geometry.py.

Which proteins (fixed before rendering):
  Fig. 6      temporal set; for each ligand-based site state (open, partly open, closed), the protein closest to
              the joint median of the hybrid per-protein AUC and of the difference physics - ESM-2 control (the
              smallest larger-of-the-two distance of its rank positions from the middle, then identifier), so that
              each row shows typical performance and a typical contrast between the models.  A first rule, the
              median of the difference alone, picked a closed-site protein on which every model failed (5HX6 B,
              AUC 0.21-0.24, far below the group's typical values), and was replaced before the figure was made
  S3, S4      CryptoBench test split (S3) and temporal set (S4); for each site state, the proteins at the 25th, 50th
              and 75th percentiles of the hybrid per-protein AUC (positions round(q (n - 1)) after sorting by AUC,
              then identifier)
  S5          a non-cryptic false positive: among CryptoBench test proteins whose hybrid site 1 finds a
              non-cryptic ligand site of the same apo structure (auxiliary data of CryptoBench) and no cryptic
              site, the one at the lower median of the hybrid AUC

Each row: (1) the apo structure (grey) and the main holo structure superposed on the Calpha atoms of the pocket
(as in holo_geometry.py), with the holo pocket residues (teal) and ligand (orange) and the apo pocket residues
(grey sticks); (2-4) the apo entity coloured by the within-protein percentile of the hybrid, physics and ESM-2
control scores, with the true cryptic-site residues (every holo record) as black Calpha spheres and each model's
site 1 (cryptobench.site_hits' calling) as a surface: purple when it finds a true site, grey when it does not.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
OUT = os.path.join(HERE, "..", "paper", "figures")
D = os.path.expanduser("~/crypsis_data")
SETS = {"cryptobench": (f"{D}/cryptobench", f"{D}/work", f"{D}/cryptobench/holo_cif"),
        "temporal": (f"{D}/temporal_bench", f"{D}/work_temporal", f"{D}/temporal/raw")}
STATES = ("open", "partly_open", "closed")
STATE_NAME = {"open": "Open", "partly_open": "Partly open", "closed": "Closed"}
MODELS = [("hybrid", "Hybrid"), ("physics", "Physics"), ("esm", "ESM-2 only")]
RAMP = ["#eef3fa", "#c6d9f1", "#8fb6e6", "#4f8fd6", "#2a6fc0", "#17488a", "#0b2a57"]
LIGAND, HOLO, HIT, MISS = "#e8590c", "#0ca678", "#ae3ec9", "#868e96"
PX = 640


def hexrgb(c):
    return [int(c[k:k + 2], 16) / 255 for k in (1, 3, 5)]


def ramp(pct):
    """Colour of a within-protein percentile: palest at the median and below, darkest from the 90th."""
    x = min(max((pct - 0.5) / 0.4, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(x), len(RAMP) - 2)
    a, b = np.array(hexrgb(RAMP[i])), np.array(hexrgb(RAMP[i + 1]))
    return a + (b - a) * (x - i)


# ------------------------------------------------------------------------------------------------ data

def load(name):
    """Per protein: keys, Calpha, labels, the three models' scores and AUCs, site state and main holo record."""
    import cryptobench as cb
    from sklearn.metrics import roc_auc_score

    data, work, holo_dir = SETS[name]
    with np.load(os.path.join(work, "test_shipped_v3_650M_scores.npz"), allow_pickle=True) as z:   # own file
        pos = {str(i): k for k, i in enumerate(z["ids"])}
        hyb, phy = z["hybrid"], z["physics"]
    with np.load(os.path.join(work, "control_esm_only_650M_256_scores.npz"), allow_pickle=True) as z:  # own file
        epos = {str(i): k for k, i in enumerate(z["ids"])}
        esm = z["esm_only"]
    with open(os.path.join(work, "holo_geometry.json")) as fh:
        geo = json.load(fh)["entities"]
    ds = cb._dataset(data)
    out = []
    for apo, chain, positives in cb.entries(data, "test"):
        pid = f"{apo}{chain}"
        with np.load(os.path.join(work, "features", f"{pid}.npz")) as z:
            keys, ca, names = [str(k) for k in z["keys"]], z["ca"], [str(n) for n in z["names"]]
        y = np.array([k in set(positives) for k in keys], dtype=int)
        sc = {"hybrid": np.asarray(hyb[pos[pid]], float), "physics": np.asarray(phy[pos[pid]], float),
              "esm": np.asarray(esm[epos[pid]], float)}
        g = geo[pid]
        ok = [x for x in g["records"] if "error" not in x]
        main = [x for x in ok if x["main"]] or sorted(ok, key=lambda x: -x["pRMSD"])
        out.append({"id": pid, "apo": apo, "chain": chain, "keys": keys, "ca": ca, "names": names, "y": y,
                    "scores": sc, "auc": {m: float(roc_auc_score(y, s)) for m, s in sc.items()},
                    "state": g["state"], "geo": main[0], "record": ds[apo][main[0]["record"]],
                    "set": name, "holo_dir": holo_dir, "data": data})
    return out


def site1(p, model):
    """Site 1 of a model (cryptobench.site_hits' calling) and whether it finds a true site (same rule)."""
    import cryptobench as cb
    from crypsis import core

    n = len(p["ca"])
    ch = core.Chain(p["ca"], np.arange(n), p["keys"], p["names"], np.arange(n), np.zeros(n))
    z = np.zeros(n)
    sites = core.sites_from_scores(ch, p["scores"][model], z, z, z, threshold=None)
    if not sites:
        return None, False
    m = set(sites[0]["residues"])
    true = cb.pockets_of(p["data"], p["apo"])
    return sorted(m), any(len(m & t) >= 3 and (len(m & t) >= 0.5 * len(m) or len(m & t) >= len(t) / 3.0)
                          for t in true)


def pick_main(prots):
    """Per state, the protein closest to the joint median of the hybrid AUC and of physics - ESM-2: the smallest
    Chebyshev distance of its two within-state rank positions (scaled to 0-1) from 0.5, then identifier."""
    out = {}
    for s in STATES:
        g = [p for p in prots if p["state"] == s]
        n = len(g)
        r1 = {p["id"]: k / (n - 1) for k, p in enumerate(sorted(g, key=lambda p: (p["auc"]["hybrid"], p["id"])))}
        r2 = {p["id"]: k / (n - 1) for k, p in enumerate(sorted(g, key=lambda p: (p["auc"]["physics"] - p["auc"]["esm"],
                                                                                   p["id"])))}
        out[s] = min(g, key=lambda p: (max(abs(r1[p["id"]] - 0.5), abs(r2[p["id"]] - 0.5)), p["id"]))
    return out


def pick_gallery(prots):
    out = []
    for s in STATES:
        g = sorted((p for p in prots if p["state"] == s), key=lambda p: (p["auc"]["hybrid"], p["id"]))
        for q, lab in ((0.25, "25th"), (0.5, "50th"), (0.75, "75th")):
            out.append((s, lab, g[int(round(q * (len(g) - 1)))]))
    return out


def pick_false_positive(prots):
    data = SETS["cryptobench"][0]
    with open(os.path.join(data, "cryptobench-dataset", "auxiliary-data", "non-cryptic-pockets",
                           "noncryptic-pockets.json")) as fh:
        nc = json.load(fh)
    cand = []
    for p in prots:
        chains = set(p["chain"].split("-"))
        recs = [(h, {k for k in h["apo_pocket_selection"] if k.split("_")[0] in chains}) for h in nc.get(p["apo"], [])]
        recs = [(h, q) for h, q in recs if q]
        if not recs:
            continue
        res, found = site1(p, "hybrid")
        if res is None or found:
            continue
        m = set(res)
        hit = [h for h, q in recs if len(m & q) >= 3 and (len(m & q) >= 0.5 * len(m) or len(m & q) >= len(q) / 3.0)]
        if hit:
            cand.append((p, hit[0]))
    cand.sort(key=lambda t: (t[0]["auc"]["hybrid"], t[0]["id"]))
    return cand[(len(cand) - 1) // 2], len(cand)


def description(path):
    """The first polymer entity's description in an mmCIF file (for the row label)."""
    import gemmi

    b = gemmi.cif.read(path).sole_block()
    for d, t in zip(b.find_values("_entity.pdbx_description"), b.find_values("_entity.type")):
        if gemmi.cif.as_string(t) == "polymer":
            return gemmi.cif.as_string(d)
    return ""


# ------------------------------------------------------------------------------------------------ render

def _resi(num):
    return num.replace("-", "\\-")


def _sel(keys):
    by = {}
    for k in keys:
        c, num = k.split("_", 1)
        by.setdefault(c, []).append(_resi(num))
    return " or ".join(f"(chain {c} and resi {'+'.join(v)})" for c, v in by.items())


def setup(p, rec):
    """Load the apo entity and the holo structure of record `rec`, superposed on the pocket's Calpha atoms."""
    import holo_geometry as hg
    from pymol import cmd

    cmd.reinitialize()
    chains = p["chain"].split("-")
    cmd.load(os.path.join(p["data"], "cif", f"{p['apo']}.cif.gz"), "apo")
    cmd.remove("apo and not (polymer.protein and (" + " or ".join(f"chain {c}" for c in chains) + "))")
    cmd.remove("apo and (hydro or not alt ''+A)")
    holo_path = os.path.join(p["holo_dir"], f"{rec['holo_pdb_id']}.cif.gz")
    cmd.load(holo_path, "holo")
    lig = (f"holo and chain {rec['ligand_chain']} and resn {rec['ligand']} and "
           f"resi {_resi(str(rec['ligand_index']))}")
    alts = sorted({a.alt for a in cmd.get_model(lig).atom})
    keep = alts[0] if alts else ""                     # first conformer of the ligand (some have only B, C or D)
    cmd.remove(f"({lig}) and not alt ''+{keep}" if keep else "none")
    cmd.remove(f"holo and (hydro or solvent or (not alt ''+A and not ({lig})))")
    hca = hg.read_holo(holo_path)[0]
    apo_ca = {f"{a.chain}_{a.resi}": np.array(a.coord) for a in cmd.get_model("apo and name CA").atom}
    pairs = [(a, b) for a, b in zip(rec["apo_pocket_selection"], rec["holo_pocket_selection"])
             if a in apo_ca and b in hca]
    if len(pairs) < 3:
        pairs = hg._extend(pairs, apo_ca, hca)
    r, t = hg.kabsch(np.array([hca[b] for _a, b in pairs]), np.array([apo_ca[a] for a, _b in pairs]))
    xyz = cmd.get_coords("holo")
    cmd.load_coords((xyz @ r.T + t).tolist(), "holo")
    cmd.select("lig", lig)
    cmd.select("hpocket", "holo and (" + _sel(rec["holo_pocket_selection"]) + ")")
    cmd.select("apocket", "apo and (" + _sel([a for a in rec["apo_pocket_selection"] if a.split("_")[0] in chains]) + ")")
    cmd.remove("holo and not (lig or hpocket or (byres (holo and polymer within 8 of lig)))")


def style():
    from pymol import cmd

    cmd.bg_color("white")
    for k, v in {"ray_opaque_background": 1, "antialias": 2, "ray_shadows": 0, "specular": 0.15, "ambient": 0.45,
                 "direct": 0.6, "cartoon_fancy_helices": 1, "cartoon_highlight_color": -1, "depth_cue": 0,
                 "ray_trace_mode": 1, "ray_trace_gain": 0.4, "ray_trace_color": "grey20", "stick_radius": 0.22,
                 "sphere_scale": 0.55, "transparency": 0.4, "surface_quality": 1, "auto_zoom": 0}.items():
        cmd.set(k, v)


def orient():
    """The whole entity, turned so that the ligand faces the camera; returns the view."""
    from pymol import cmd

    cmd.orient("apo")
    lig_c = np.array(cmd.get_coords("lig")).mean(0)
    prot_c = np.array(cmd.get_coords("apo")).mean(0)
    best, view = -1e9, None
    for ax in ("x", "y"):
        for _ in range(12):
            cmd.turn(ax, 30)
            v = cmd.get_view()
            z = (lig_c - prot_c) @ np.array(v[:9]).reshape(3, 3)[:, 2]
            if z > best:
                best, view = z, v
    cmd.set_view(view)
    cmd.zoom("apo", 0)
    return cmd.get_view()


def show_ligand():
    from pymol import cmd

    cmd.show("sticks", "lig")
    cmd.set_color("ligc", hexrgb(LIGAND))
    cmd.color("ligc", "lig and elem C")
    cmd.color("red", "lig and elem O")
    cmd.color("blue", "lig and elem N")
    cmd.set("stick_radius", 0.3, "lig")


def render_holo(png, view):
    from pymol import cmd

    cmd.hide("everything")
    cmd.set_view(view)
    cmd.show("cartoon", "apo or (holo and byres (holo within 8 of lig))")
    cmd.color("grey80", "apo")
    cmd.set_color("holoc", hexrgb(HOLO))
    cmd.color("holoc", "holo")
    cmd.show("sticks", "(apocket or hpocket) and not name N+C+O")
    cmd.color("grey45", "apocket and elem C")
    cmd.color("holoc", "hpocket and elem C")
    show_ligand()
    cmd.zoom("lig or hpocket", 3)
    cmd.png(png, width=PX, height=PX, dpi=300, ray=1)
    crop(png)


def render_model(p, model, png, view):
    from pymol import cmd

    cmd.hide("everything")
    cmd.set_view(view)
    cmd.show("cartoon", "apo")
    s = p["scores"][model]
    pct = (np.argsort(np.argsort(s)) + 0.5) / len(s)
    for k, v in zip(p["keys"], pct):
        name = f"pc{int(round(v * 1000))}"
        cmd.set_color(name, list(ramp(float(v))))
        c, num = k.split("_", 1)
        cmd.color(name, f"apo and chain {c} and resi {_resi(num)}")
    true = [k for k, yy in zip(p["keys"], p["y"]) if yy]
    if true:
        cmd.show("spheres", "apo and name CA and (" + _sel(true) + ")")
        cmd.color("black", "apo and name CA and (" + _sel(true) + ")")
    res, found = site1(p, model)
    if res:
        cmd.create("site", "apo and (" + _sel(res) + ")")
        cmd.hide("everything", "site")
        cmd.show("surface", "site")
        cmd.set_color("sitec", hexrgb(HIT if found else MISS))
        cmd.set("surface_color", "sitec", "site")
    show_ligand()
    cmd.png(png, width=PX, height=PX, dpi=300, ray=1)
    crop(png)
    cmd.delete("site")
    return found


def crop(png, margin=12):
    import matplotlib.pyplot as plt

    img = plt.imread(png)
    ink = np.argwhere(img[:, :, :3].min(axis=2) < 0.97)
    (r0, c0), (r1, c1) = ink.min(0), ink.max(0)
    h = max(r1 - r0, c1 - c0) // 2 + margin
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    pad = np.ones((img.shape[0] + 2 * h, img.shape[1] + 2 * h, img.shape[2]), dtype=img.dtype)
    pad[h:h + img.shape[0], h:h + img.shape[1]] = img
    plt.imsave(png, pad[rc:rc + 2 * h, cc:cc + 2 * h])


def render_row(p, tag):
    """Four images for one protein; returns their paths and whether each model's site 1 finds a true site."""
    from pymol import cmd

    tmp = os.path.join(OUT, "structures")
    os.makedirs(tmp, exist_ok=True)
    imgs = [os.path.join(tmp, f"{tag}_{m}.png") for m in ["holo"] + [m for m, _l in MODELS]]
    if all(os.path.exists(f) for f in imgs) and os.environ.get("REUSE_RENDERS"):
        return imgs, {m: site1(p, m)[1] for m, _l in MODELS}
    setup(p, p["record"])
    style()
    view = orient()
    imgs, found = [os.path.join(tmp, f"{tag}_holo.png")], {}
    render_holo(imgs[0], view)
    for m, _lab in MODELS:
        imgs.append(os.path.join(tmp, f"{tag}_{m}.png"))
        found[m] = render_model(p, m, imgs[-1], view)
    cmd.reinitialize()
    return imgs, found


# ------------------------------------------------------------------------------------------------ figures

def label(p):
    desc = description(os.path.join(p["data"], "cif", f"{p['apo']}.cif.gz"))
    desc = desc if len(desc) <= 34 else desc[:32].rstrip() + "…"
    rec = p["record"]
    return f"{desc}\napo {p['apo'].upper()} {p['chain']}\nholo {rec['holo_pdb_id'].upper()} ({rec['ligand']})"


def figure(rows, name, width_mm, row_mm):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import paper_figures as pf

    pf.plt = plt
    pf.use_helvetica()
    plt.rcParams.update(pf.RC)
    n, foot, head = len(rows), 16.0, 6.0
    height = n * row_mm + foot + head
    fig = plt.figure(figsize=(width_mm * pf.MM, height * pf.MM))
    left = 0.2
    gw = (0.995 - left) / 4
    rh = row_mm / height
    record = []
    for i, (p, text, imgs, found, letter) in enumerate(rows):
        y0 = 1 - head / height - (i + 1) * rh
        fig.text(0.012, y0 + rh * 0.6, text, fontsize=6, va="center", ha="left", linespacing=1.25)
        if letter:
            fig.text(0.0, y0 + rh * 0.98, letter, fontsize=10, fontweight="bold", va="top")
        notes = [f"ligand fits {p['geo']['fit_share_2.5'] * 100:.0f}%, pocket RMSD {p['geo']['pocket_rmsd']:.1f} Å"]
        notes += [f"AUC {p['auc'][m]:.2f}, site 1 {'finds it' if found[m] else 'misses'}" for m, _l in MODELS]
        for j, (img, note) in enumerate(zip(imgs, notes)):
            ax = fig.add_axes([left + j * gw, y0 + rh * 0.13, gw * 0.97, rh * 0.85])
            ax.imshow(plt.imread(img))
            ax.set_axis_off()
            ax.text(0.5, -0.01, note, transform=ax.transAxes, ha="center", va="top", fontsize=5.6)
            if i == 0:
                ax.set_title(["Apo (grey) and holo", "Hybrid", "Physics", "ESM-2 only"][j], fontsize=7, pad=2)
        record.append({"id": p["id"], "set": p["set"], "state": p["state"], "label": text,
                       "auc": {m: round(v, 4) for m, v in p["auc"].items()}, "site1_finds": found,
                       "fit_share_2.5": p["geo"]["fit_share_2.5"], "pocket_rmsd": p["geo"]["pocket_rmsd"],
                       "holo": p["record"]["holo_pdb_id"], "ligand": p["record"]["ligand"]})
    cmap = LinearSegmentedColormap.from_list("pct", RAMP)
    cax = fig.add_axes([0.03, 0.55 * foot / height, 0.22, 0.12 * foot / height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(50, 90))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[50, 70, 90])
    cb.ax.set_xticklabels(["≤50", "70", "≥90"])
    cb.set_label("Score percentile within the protein", fontsize=6.3)
    cb.outline.set_linewidth(0.5)
    handles = [Line2D([], [], marker="o", color="black", lw=0, markersize=3.2, label="true cryptic-site residue (Cα)"),
               Patch(facecolor=HIT, alpha=0.6, label="site 1 finds a true site"),
               Patch(facecolor=MISS, alpha=0.6, label="site 1 misses"),
               Line2D([], [], color=LIGAND, lw=2.2, label="holo ligand"),
               Line2D([], [], color=HOLO, lw=2.2, label="holo pocket and nearby chain")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.32, 0.0), frameon=False, fontsize=6,
               ncol=3, handlelength=1.3, columnspacing=1.0)
    pf.save(fig, name)
    return record


def main():
    import matplotlib

    matplotlib.use("Agg")
    sel = {}
    temporal, cbench = load("temporal"), load("cryptobench")
    rows = []
    for letter, (s, p) in zip("abc", pick_main(temporal).items()):
        imgs, found = render_row(p, f"fig6_{s}")
        rows.append((p, f"{STATE_NAME[s]} site (temporal set)\n" + label(p), imgs, found, letter))
    sel["fig6"] = figure(rows, "fig6_structures.pdf", 180, 44)
    for name, prots, fname in (("cryptobench", cbench, "figS3_gallery_cryptobench.pdf"),
                               ("temporal", temporal, "figS4_gallery_temporal.pdf")):
        rows = []
        for s, q, p in pick_gallery(prots):
            imgs, found = render_row(p, f"{name}_{s}_{q}")
            rows.append((p, f"{STATE_NAME[s]}, {q} percentile\n" + label(p), imgs, found, None))
        sel[name] = figure(rows, fname, 170, 25)
    (p, nc), n_cand = pick_false_positive(cbench)
    imgs, found = render_row(p, "false_positive")
    text = (f"Non-cryptic false positive\n({n_cand} candidates)\n" + label(p) +
            f"\nnon-cryptic site: {nc['ligand']} in {nc['holo_pdb_id'].upper()}")
    sel["false_positive"] = figure([(p, text, imgs, found, None)], "figS5_false_positive.pdf", 170, 40)
    sel["false_positive"][0]["noncryptic_record"] = {k: nc[k] for k in ("holo_pdb_id", "ligand", "apo_pocket_selection")}
    with open(os.path.join(OUT, "structures", "selection.json"), "w") as fh:
        json.dump(sel, fh, indent=1)


if __name__ == "__main__":
    main()

"""Command line: crypsis predict structure.cif [--chains A] [--keep auto] [--model auto] [-o out]"""

import argparse
import os
import sys
import time


def _predict(args):
    from . import output
    from .pipeline import predict

    keep = None if not args.keep else ("auto" if args.keep == "auto" else set(args.keep.split(",")))
    chains = args.chains.split(",") if args.chains else None
    t0 = time.time()

    def progress(k, n):
        if not args.quiet:
            print(f"\r  ensemble {k}/{n}", end="", file=sys.stderr, flush=True)

    res = predict(args.structure, chains=chains, keep_cofactors=keep, model=args.model, device=args.device,
                  progress=progress)
    if not args.quiet:
        print(file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.basename(args.structure)
    for ext in (".gz", ".cif", ".pdb", ".ent", ".mmcif"):
        base = base[:-len(ext)] if base.lower().endswith(ext) else base
    stem = os.path.join(args.out, base)
    output.write_csv(res, stem + "_residues.csv")
    output.write_sites(res, stem + "_sites.json")
    output.write_structure(res, args.structure, stem + "_crypsis.pdb")
    output.write_pymol(res, stem + "_crypsis.pdb", stem + "_crypsis.pml")
    print(f"crypsis {res['model']} model, {len(res['residues'])} residues, {len(res['sites'])} sites "
          f"({time.time() - t0:.0f} s{'' if res['extension'] else ', numpy grid: the C++ extension is not built'})")
    print(f"{'rank':>4} {'score':>6} {'n':>3}  {'kind':<18} {'volume A3 (structure -> ensemble)':<34} residues")
    for s in res["sites"][:args.top]:
        resid = " ".join(f"{n.title()}{k.split('_')[1]}" for k, n in zip(s["residues"], s["residue_names"]))
        vol = f"{s['pocket_volume_structure_A3']:.0f} -> {s['pocket_volume_ensemble_A3']:.0f}"
        print(f"{s['rank']:>4} {s['score']:>6.2f} {s['n_residues']:>3}  {s['kind']:<18} {vol:<34} {resid}")
    print(f"wrote {stem}_residues.csv, _sites.json, _crypsis.pdb (B-factor = 100 x probability), _crypsis.pml")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="crypsis", description="Find cryptic binding pockets in protein structures.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="residue probabilities and ranked cryptic sites for one structure")
    p.add_argument("structure", help="PDB or mmCIF file (gzipped is fine)")
    p.add_argument("--chains", help="comma-separated chain IDs (default: all protein chains)")
    p.add_argument("--keep", help="cofactors to keep as occupied space: 'auto' or names, e.g. GDP,MG")
    p.add_argument("--model", default="auto", choices=["auto", "hybrid", "physics", "v1"],
                   help="auto: hybrid when PyTorch + fair-esm are installed, physics otherwise")
    p.add_argument("--device", default="cpu", help="PyTorch device for ESM-2 (cpu, cuda, mps)")
    p.add_argument("-o", "--out", default="crypsis_out", help="output folder")
    p.add_argument("--top", type=int, default=5, help="sites to print")
    p.add_argument("-q", "--quiet", action="store_true")
    sub.add_parser("check", help="report whether the C++ extension and ESM-2 are available")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        from . import core
        from .pipeline import have_esm

        print("C++ grid extension:", "yes" if core.have_extension() else "no (numpy fallback, ~15x slower)")
        print("ESM-2 (hybrid model):", "yes" if have_esm() else "no (pip install 'crypsis[esm]')")
        return 0
    return _predict(args)


if __name__ == "__main__":
    sys.exit(main())

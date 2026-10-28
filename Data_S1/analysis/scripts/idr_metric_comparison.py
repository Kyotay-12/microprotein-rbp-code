"""Compare per-ORF mean-disorder classification and per-residue IDR fraction.

Two metrics on the SAME predictor (metapredict 3.0.2 per-residue disorder score):

  A. "disordered" if mean(per-residue score) >= 0.5 ;
     ORF-level binary classification.
  B. (Zhang 2026-style metric) IDR_fraction = (residues with score >= 0.5) /
     ORF length ; ORF-level continuous quantity.

Zhang et al. used IUPred3 short mode, whereas this implementation uses
metapredict. It compares the two metric definitions on the same predictor;
it is not a reproduction of an IUPred3-based result.

Output: results/idr_metric_comparison.csv  per-ORF (orf_name, biotype,
                                                    mean_idr, idr_fraction)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import metapredict as meta

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
FASTA = ROOT / "data" / "orfs_7264.fasta"
OUT = ROOT / "results" / "idr_metric_comparison.csv"


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def main():
    s0 = read_step0()
    main = s0[(s0.source_sheet == "S2") & (s0.aa_length <= 100)].copy()

    # Load sequences from FASTA
    seqs = {}
    name = None
    for line in open(FASTA):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        else:
            seqs[name] += line
    # subset to main pop
    main["seq"] = main["orf_name"].map(seqs)
    print(f"main pop n={len(main)}; sequences loaded={int(main['seq'].notna().sum())}", flush=True)
    if main["seq"].isna().any():
        raise ValueError(f"{int(main['seq'].isna().sum())} ORFs lack a sequence")

    # batch metapredict
    print("Running metapredict predict_disorder_batch ...", flush=True)
    seq_list = main["seq"].tolist()
    batch = meta.predict_disorder_batch(seq_list, show_progress_bar=False)
    # batch returns list of [seq, np.array]
    main["idr_fraction"] = [
        float((np.asarray(p[1]) >= 0.5).mean()) for p in batch
    ]
    # sanity: mean_idr already in step0; we recompute from the same batch as cross-check
    main["mean_idr_recompute"] = [float(np.mean(p[1])) for p in batch]
    # tolerance check
    delta = (main["mean_idr_recompute"] - main["mean_idr"]).abs()
    print(f"  mean_idr recompute vs step0: max diff = {delta.max():.6f} (should be ~0)", flush=True)

    # classification under metric A vs distribution under metric B
    A_disordered = (main["mean_idr"] >= 0.5).sum()
    print("\n=== metric A: current paper classification ===")
    print(f"  ORFs with mean(score) >= 0.5: {A_disordered} / {len(main)} "
          f"({100*A_disordered/len(main):.1f}%)")

    print("\n=== metric B: Zhang-style fraction (predictor: metapredict, NOT IUPred3) ===")
    frac = main["idr_fraction"]
    pct = lambda q: float(frac.quantile(q))
    print(f"  IDR fraction: median={frac.median():.3f}  mean={frac.mean():.3f}")
    print(f"    p10={pct(0.10):.3f}  p25={pct(0.25):.3f}  p50={pct(0.50):.3f}  "
          f"p75={pct(0.75):.3f}  p90={pct(0.90):.3f}")
    # how does fraction relate to mean classification?
    print(f"  among ORFs classified disordered (metric A): "
          f"median IDR fraction = {main.loc[main.mean_idr>=0.5,'idr_fraction'].median():.3f}")
    print(f"  among ORFs NOT disordered (metric A): "
          f"median IDR fraction = {main.loc[main.mean_idr<0.5,'idr_fraction'].median():.3f}")
    # cross-table: full disordered (fraction == 1), partial (0 < frac < 1), none (frac == 0)
    full = int((frac == 1.0).sum()); none = int((frac == 0.0).sum())
    high80 = int((frac >= 0.80).sum())
    above50 = int((frac > 0.50).sum())
    print(f"  ORFs with IDR fraction == 1.0 (fully disordered): {full} "
          f"({100*full/len(main):.1f}%)")
    print(f"  ORFs with IDR fraction >= 0.80: {high80} "
          f"({100*high80/len(main):.1f}%)")
    print(f"  ORFs with IDR fraction > 0.50 (majority disordered): {above50} "
          f"({100*above50/len(main):.1f}%)")
    print(f"  ORFs with IDR fraction == 0.0 (no residue passes threshold): {none}")

    # by biotype
    print("\n=== IDR fraction by ORF biotype (median) ===")
    print(main.groupby("orf_biotype").agg(
        n=("orf_name", "size"),
        median_idr_fraction=("idr_fraction", "median"),
        mean_idr_fraction=("idr_fraction", "mean"),
        frac_metric_A_disordered=("mean_idr", lambda s: 100 * (s >= 0.5).mean()),
    ).round(3).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    main[["orf_name", "source_sheet", "orf_biotype", "aa_length",
          "mean_idr", "idr_fraction"]].to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

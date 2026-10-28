"""Build per-transcript exon intervals from GENCODE v35 GTF and report how many
ORF-catalog transcript IDs match (pre-intersect sanity gate).

Runs in .venv_ranges (pandas 2.x + pyranges). Writes the exon table (restricted
to ORF-catalog transcripts) to results/v35_orf_exons.parquet for the intersect.
"""
import sys
from pathlib import Path

import pandas as pd
import pyranges as pr

ROOT = Path(__file__).resolve().parent.parent
GTF = ROOT / "data" / "gencode.v35.annotation.gtf.gz"
STEP0 = ROOT / "results" / "step0_properties.csv"
OUT = ROOT / "results" / "v35_orf_exons.parquet"


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def main():
    df = read_step0()
    orf_tx = df["transcript"].str.split(".").str[0]  # versionless ENST
    orf_tx_set = set(orf_tx)
    print(f"ORF catalog: {len(df)} ORFs, {len(orf_tx_set)} unique transcript IDs", flush=True)

    print("Reading GTF exons ...", flush=True)
    g = pr.read_gtf(str(GTF), as_df=True)
    ex = g[g["Feature"] == "exon"].copy()
    ex["enst"] = ex["transcript_id"].str.split(".").str[0]
    gtf_tx_set = set(ex["enst"].unique())
    print(f"GTF: {len(ex)} exon rows, {len(gtf_tx_set)} unique transcripts", flush=True)

    matched = orf_tx_set & gtf_tx_set
    missing = orf_tx_set - gtf_tx_set
    print(f"\n=== match prediction (pre-intersect) ===", flush=True)
    print(f"  ORF transcripts matched in v35 GTF: {len(matched)} / {len(orf_tx_set)} "
          f"({100*len(matched)/len(orf_tx_set):.2f}%)", flush=True)
    print(f"  ORF transcripts NOT in GTF: {len(missing)}", flush=True)
    if missing:
        # how many ORFs affected, and by biotype
        miss_orfs = df[orf_tx.isin(missing)]
        print(f"  ORFs affected by missing transcript: {len(miss_orfs)}", flush=True)
        print("  by biotype:", flush=True)
        for bt, c in miss_orfs["orf_biotype"].value_counts().items():
            print(f"    {bt}: {c}", flush=True)
        print(f"  examples: {sorted(missing)[:10]}", flush=True)

    # write exon table restricted to ORF transcripts for the intersect
    keep = ex[ex["enst"].isin(orf_tx_set)][
        ["Chromosome", "Start", "End", "Strand", "enst"]
    ].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(OUT, index=False)
    print(f"\n  wrote {len(keep)} exon intervals ({keep['enst'].nunique()} transcripts) -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()

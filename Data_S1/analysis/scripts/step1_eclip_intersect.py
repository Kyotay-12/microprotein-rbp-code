"""eCLIP intersect -> per-ORF binding RBP count, per cell line.

Overlap definitions:
  region  : transcript exons (UTR incl, introns excl)  [primary]
  strand  : required to match (eCLIP is strand-specific)
  overlap : 1 bp
  count   : unique RBP per ORF (via its transcript)     [primary]
  sensitivity: (i) ORF-span-limited region, (ii) peak-count instead of unique RBP

Coordinates: exon parquet (from GTF via pyranges) and narrowPeak (BED) are both
0-based half-open. ORF-span coords come from the Mudge xlsx (1-based inclusive,
'chr'-less) -> converted to 0-based half-open + 'chr' prefix; a self-check reports
the fraction of ORF span that overlaps its own transcript exons (should be ~100%).

Runs in .venv_ranges (pandas 2.x + pyranges). A zero count should be interpreted
as no detected overlap only after transcript coverage in the exon map has
been verified with build_exon_map.py. Writes results/step1_binding.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
EXONS = ROOT / "results" / "v35_orf_exons.parquet"
STEP0 = ROOT / "results" / "step0_properties.csv"
XLSX = ROOT / "data" / "Mudge2022_SupplementaryTables.xlsx"
ECLIP_DIR = ROOT / "data" / "eclip"
OUT = ROOT / "results" / "step1_binding.csv"
CELLS = ["K562", "HepG2"]
SHEETS = ["S2. PHASE I Ribo-seq ORFs", "S3. Single-study Ribo-seq ORFs"]


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    df = pd.read_csv(STEP0, skiprows=nc)
    df["enst"] = df["transcript"].str.split(".").str[0]
    return df


def load_peaks(cell):
    """All reproducible eCLIP peaks for a cell line -> PyRanges with rbp column."""
    rows = []
    for f in sorted((ECLIP_DIR / cell).glob("*.bed.gz")):
        rbp = f.name.split("__")[0]
        d = pd.read_csv(
            f, sep="\t", header=None, compression="gzip",
            usecols=[0, 1, 2, 5], names=["Chromosome", "Start", "End", "Strand"],
            dtype={0: str, 1: int, 2: int, 5: str},
        )
        d["rbp"] = rbp
        rows.append(d)
    allp = pd.concat(rows, ignore_index=True)
    return pr.PyRanges(allp)


def load_orf_spans():
    """ORF genomic blocks from xlsx -> PyRanges (orf_name), 0-based + chr prefix."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    recs = []
    for sheet in SHEETS:
        ws = wb[sheet]
        it = ws.iter_rows(values_only=True)
        header = next(it)
        idx = {h: i for i, h in enumerate(header) if h}
        for row in it:
            name = row[idx["orf_name"]]
            if name is None or not str(name).strip():
                continue
            chrm = str(row[idx["chrm"]])
            strand = row[idx["strand"]]
            starts = str(row[idx["starts"]]).split(";")
            ends = str(row[idx["ends"]]).split(";")
            for s, e in zip(starts, ends):
                recs.append(dict(
                    Chromosome=chrm if chrm.startswith("chr") else "chr" + chrm,
                    Start=int(float(s)) - 1,   # 1-based inclusive -> 0-based half-open
                    End=int(float(e)),
                    Strand=strand,
                    orf_name=str(name),
                ))
    return pr.PyRanges(pd.DataFrame(recs))


def strata(n):
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    return "5+"


def main():
    df = read_step0()
    exon_df = pd.read_parquet(EXONS)
    exon_pr = pr.PyRanges(exon_df)

    # self-check: ORF span overlaps its own transcript exons?
    orf_pr = load_orf_spans()
    print(f"ORF-span blocks: {len(orf_pr.df)} (for {orf_pr.df['orf_name'].nunique()} ORFs)", flush=True)

    for cell in CELLS:
        print(f"\nLoading {cell} eCLIP peaks ...", flush=True)
        peaks = load_peaks(cell)
        npk = len(peaks.df)
        print(f"  {cell}: {npk} peaks, {peaks.df['rbp'].nunique()} unique RBPs", flush=True)

        # --- primary: transcript-exon, strand-matched, 1bp, unique RBP ---
        j = exon_pr.join(peaks, strandedness="same").df
        if len(j):
            per_tx = j.groupby("enst").agg(
                nrbp=("rbp", "nunique"), npeaks=("rbp", "size")
            )
        else:
            per_tx = pd.DataFrame(columns=["nrbp", "npeaks"])
        df[f"{cell}_nrbp_tx"] = df["enst"].map(per_tx["nrbp"]).fillna(0).astype(int)
        df[f"{cell}_npeaks_tx"] = df["enst"].map(per_tx["npeaks"]).fillna(0).astype(int)

        # --- sensitivity: ORF-span region, unique RBP ---
        jo = orf_pr.join(peaks, strandedness="same").df
        if len(jo):
            per_orf = jo.groupby("orf_name").agg(nrbp=("rbp", "nunique"))
        else:
            per_orf = pd.DataFrame(columns=["nrbp"])
        df[f"{cell}_nrbp_orfspan"] = df["orf_name"].map(per_orf["nrbp"]).fillna(0).astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["orf_name", "source_sheet", "orf_biotype", "aa_length", "transcript"] + \
        [c for c in df.columns if any(c.startswith(f"{cell}_") for cell in CELLS)]
    df[keep_cols].to_csv(OUT, index=False)
    print(f"\nWrote {OUT}", flush=True)

    # ===== report =====
    main_pop = df[(df.source_sheet == "S2") & (df.aa_length <= 100)]
    for cell in CELLS:
        col = f"{cell}_nrbp_tx"
        print(f"\n================  {cell}  (transcript-level unique RBP)  ================", flush=True)
        for label, d in [("ALL (n=%d)" % len(df), df), ("MAIN <=100aa&S2 (n=%d)" % len(main_pop), main_pop)]:
            s = d[col]
            st = s.apply(strata).value_counts()
            print(f"  {label}:  mean={s.mean():.2f} median={int(s.median())} max={int(s.max())} "
                  f"frac>0={100*(s>0).mean():.1f}%")
            print(f"      strata  0:{int(st.get('0',0))}  1:{int(st.get('1',0))}  "
                  f"2-4:{int(st.get('2-4',0))}  5+:{int(st.get('5+',0))}")
        # (3) corr peak-count vs unique-RBP
        from scipy.stats import spearmanr, pearsonr
        sp = spearmanr(df[col], df[f"{cell}_npeaks_tx"]).statistic
        pe = pearsonr(df[col], df[f"{cell}_npeaks_tx"]).statistic
        print(f"  corr(unique RBP, peak count): Spearman={sp:.3f} Pearson={pe:.3f}")
        # (4) tx vs orf-span
        d2 = df[[col, f"{cell}_nrbp_orfspan"]]
        same_stratum = (df[col].apply(strata) == df[f"{cell}_nrbp_orfspan"].apply(strata)).mean()
        print(f"  ORF-span vs transcript unique RBP: "
              f"mean orfspan={df[f'{cell}_nrbp_orfspan'].mean():.2f} vs tx={df[col].mean():.2f}; "
              f"same stratum in {100*same_stratum:.1f}% of ORFs; "
              f"Spearman={spearmanr(df[col], df[f'{cell}_nrbp_orfspan']).statistic:.3f}")


if __name__ == "__main__":
    main()

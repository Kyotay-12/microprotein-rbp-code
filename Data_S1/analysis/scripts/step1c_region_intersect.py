"""Region-stratified eCLIP intersect: 5'UTR / CDS / 3'UTR per host transcript.

Regions are defined by the host transcript's CANONICAL CDS (GENCODE GTF feature
types five_prime_utr / CDS / three_prime_utr), NOT the Ribo-seq ORF. So a uORF
(which lives in the 5'UTR) contributes to "5'UTR binding" of its own locus — this
overlap is quantified in the report for the uORF-handling decision.

Non-coding host transcripts (lncRNA / processed_transcript) have no CDS/UTR
features -> flagged as 'noncoding' (region analysis N/A; handled separately).

Per ORF, per cell line, per region: unique RBP count, peak count, region length(bp),
RBP density (unique RBP / region kb). strand-matched, 1bp overlap. No dummy fills.

Runs in .venv_ranges. Writes results/step1c_region_binding.csv and prints the
coverage report. Does not run the regional regression.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr

ROOT = Path(__file__).resolve().parent.parent
GTF = ROOT / "data" / "gencode.v35.annotation.gtf.gz"
STEP0 = ROOT / "results" / "step0_properties.csv"
BIND = ROOT / "results" / "step1_binding.csv"
ECLIP_DIR = ROOT / "data" / "eclip"
OUT = ROOT / "results" / "step1c_region_binding.csv"
CELLS = ["K562", "HepG2"]
# GENCODE GTF uses an undifferentiated 'UTR' feature (no five_prime_utr/three_prime_utr).
# 5'/3' are DERIVED from each transcript's CDS span + strand.
REGIONS = ["utr5", "cds", "utr3"]


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
    rows = []
    for f in sorted((ECLIP_DIR / cell).glob("*.bed.gz")):
        rbp = f.name.split("__")[0]
        d = pd.read_csv(f, sep="\t", header=None, compression="gzip",
                        usecols=[0, 1, 2, 5],
                        names=["Chromosome", "Start", "End", "Strand"],
                        dtype={0: str, 1: int, 2: int, 5: str})
        d["rbp"] = rbp
        rows.append(d)
    return pr.PyRanges(pd.concat(rows, ignore_index=True))


def build_regions(gdf, orf_enst):
    """Return {utr5, cds, utr3} interval DataFrames (Chromosome,Start,End,Strand,enst).

    CDS = GTF 'CDS' features. UTR = GTF 'UTR' features classified to 5'/3' by
    position relative to the transcript's CDS span and strand:
      genomic low-coord side  -> 5'UTR on '+' strand, 3'UTR on '-' strand
      genomic high-coord side -> 3'UTR on '+' strand, 5'UTR on '-' strand
    Only transcripts with a CDS get UTRs classified (non-coding hosts have none).
    """
    g = gdf.copy()
    g["enst"] = g["transcript_id"].str.split(".").str[0]
    g = g[g["enst"].isin(orf_enst)]

    cds = g[g["Feature"] == "CDS"][["Chromosome", "Start", "End", "Strand", "enst"]].reset_index(drop=True)
    # per-transcript CDS span + strand
    span = cds.groupby("enst").agg(cds_min=("Start", "min"), cds_max=("End", "max"),
                                   Strand=("Strand", "first")).reset_index()

    utr = g[g["Feature"] == "UTR"][["Chromosome", "Start", "End", "Strand", "enst"]].copy()
    utr = utr.merge(span[["enst", "cds_min", "cds_max"]], on="enst", how="inner")  # only coding hosts
    low = utr["End"] <= utr["cds_min"]
    high = utr["Start"] >= utr["cds_max"]
    plus = utr["Strand"] == "+"
    is5 = (plus & low) | (~plus & high)
    is3 = (plus & high) | (~plus & low)
    unclassified = int((~(is5 | is3)).sum())
    if unclassified:
        print(f"  WARNING: {unclassified} UTR intervals not cleanly 5'/3' (overlap CDS span?)", flush=True)
    cols = ["Chromosome", "Start", "End", "Strand", "enst"]
    return {"utr5": utr[is5][cols].reset_index(drop=True),
            "cds": cds,
            "utr3": utr[is3][cols].reset_index(drop=True)}


def main():
    s0 = read_step0()
    orf_enst = set(s0["enst"])
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]

    print("Reading GTF ...", flush=True)
    gdf = pr.read_gtf(str(GTF), as_df=True)
    regions = build_regions(gdf, orf_enst)

    # region interval tables + lengths per transcript
    region_pr = {}
    region_len = {}
    for key in REGIONS:
        iv = regions[key]
        region_pr[key] = pr.PyRanges(iv)
        ln = iv.assign(L=iv["End"] - iv["Start"]).groupby("enst")["L"].sum()
        region_len[key] = ln
        print(f"  {key}: {len(iv)} intervals, {iv['enst'].nunique()} transcripts", flush=True)

    # coding vs non-coding host transcript (has CDS?)
    cds_tx = set(region_len["cds"].index)
    s0["host_coding"] = s0["enst"].isin(cds_tx)

    df = s0.merge(bind, on="orf_name")
    for key in REGIONS:
        df[f"len_{key}"] = df["enst"].map(region_len[key]).fillna(0).astype(int)

    # per region per cell: unique RBP + peaks
    for cell in CELLS:
        print(f"\nIntersecting {cell} peaks by region ...", flush=True)
        peaks = load_peaks(cell)
        for key in REGIONS:
            j = region_pr[key].join(peaks, strandedness="same").df
            if len(j):
                agg = j.groupby("enst").agg(nrbp=("rbp", "nunique"), npk=("rbp", "size"))
            else:
                agg = pd.DataFrame(columns=["nrbp", "npk"])
            df[f"{cell}_{key}_nrbp"] = df["enst"].map(agg["nrbp"]).fillna(0).astype(int)
            df[f"{cell}_{key}_npk"] = df["enst"].map(agg["npk"]).fillna(0).astype(int)
            # density = unique RBP / kb (NaN where region length 0 -> region absent)
            ln = df[f"len_{key}"]
            df[f"{cell}_{key}_density"] = np.where(ln > 0, df[f"{cell}_{key}_nrbp"] / (ln / 1000.0), np.nan)

    keep = ["orf_name", "source_sheet", "orf_biotype", "aa_length", "transcript",
            "enst", "host_coding", "K562_nrbp_tx", "HepG2_nrbp_tx"] + \
        [c for c in df.columns if c.startswith("len_") or any(
            c.startswith(f"{cell}_{key}_") for cell in CELLS for key in REGIONS)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(OUT, index=False)
    print(f"\nWrote {OUT}", flush=True)

    # ===================== REGION COVERAGE REPORT =====================
    main_pop = df[(df.source_sheet == "S2") & (df.aa_length <= 100)]
    print(f"\n========== PRE-EXEC REPORT (MAIN pop n={len(main_pop)}) ==========", flush=True)

    # (1) region length distribution (coding hosts only)
    print("\n(1) region length (bp) distribution [coding-host ORFs in main pop]:", flush=True)
    coding_main = main_pop[main_pop["host_coding"]]
    print(f"    coding-host ORFs in main pop: {len(coding_main)}", flush=True)
    for key in REGIONS:
        s = coding_main[f"len_{key}"]
        s = s[s > 0]
        if len(s) == 0:
            print(f"    {key:5s}  n_with_region=0  (no intervals)", flush=True)
            continue
        print(f"    {key:5s}  n_with_region={len(s):4d}  median={int(s.median())}bp  "
              f"p25={int(s.quantile(.25))}  p75={int(s.quantile(.75))}  max={int(s.max())}", flush=True)

    # (2) non-coding hosts + uORF-in-5'UTR overlap + proposal
    print("\n(2) host transcript coding status & uORF region overlap [main pop]:", flush=True)
    print(f"    coding host: {int(main_pop['host_coding'].sum())}  "
          f"non-coding host: {int((~main_pop['host_coding']).sum())}", flush=True)
    print("    non-coding-host ORF biotypes:", flush=True)
    nc = main_pop[~main_pop["host_coding"]]
    for bt, c in nc["orf_biotype"].value_counts().items():
        print(f"      {bt}: {c}", flush=True)
    print("    coding-host ORF biotypes (these get 5'UTR/CDS/3'UTR):", flush=True)
    for bt, c in coding_main["orf_biotype"].value_counts().items():
        print(f"      {bt}: {c}", flush=True)
    # uORF: how big is the ORF relative to its 5'UTR?
    uorf = coding_main[coding_main["orf_biotype"] == "uORF"].copy()
    if len(uorf):
        uorf["orf_bp"] = uorf["aa_length"] * 3
        uorf["frac_of_5utr"] = uorf["orf_bp"] / uorf["len_utr5"].replace(0, np.nan)
        print(f"    uORF in main pop: {len(uorf)}; uORF length as fraction of host 5'UTR: "
              f"median={uorf['frac_of_5utr'].median():.2f} "
              f"(uORF occupies this share of the 5'UTR it is counted in)", flush=True)

    # (3) region-wise unique RBP distribution + correlation with total
    from scipy.stats import spearmanr
    print("\n(3) region-wise unique RBP [main pop, coding hosts] + corr with total nrbp_tx:", flush=True)
    for cell in CELLS:
        print(f"    --- {cell} ---", flush=True)
        for key in REGIONS:
            col = f"{cell}_{key}_nrbp"
            s = coding_main[col]
            tot = coding_main[f"{cell}_nrbp_tx"]
            rho = spearmanr(s, tot).statistic
            print(f"      {key:5s}: mean={s.mean():.2f} median={int(s.median())} "
                  f"frac>0={100*(s>0).mean():.0f}%  Spearman(vs total)={rho:.2f}", flush=True)


if __name__ == "__main__":
    main()

# STATUS: SUPERSEDED
# Historical reconstruction only; use posthoc/canonical for the corrected reference.
# followup10_canonical_lrpf_fixed.py separately repairs the abundance input.
# REASON: transcript abundance for the canonical reference set was drawn from
#         analysis/results/step1_expression.csv, which contains ncORF host
#         transcripts only. Every canonical transcript that received a TPM was
#         therefore an ncORF host, so the "control" population was not
#         independent of the population it was meant to control for.
#         followup10 rebuilds abundance from the ENCODE RSEM quantifications.
# This selection implementation also predates the all-tag and PAR-preserving
# corrections. Its outputs remain inputs to historical comparison checks.

"""Historical canonical-CDS selection, footprint counting and occupancy model.

Approach: sample one canonical CDS per protein-coding gene (the
"appris_principal" / longest CDS in the GENCODE v35 annotation), and
run the IDENTICAL lRPF regression as the ncORF Step 2 main:

   log2(in_frame_RPF_canonical_CDS + 1) ~ log1p(unique_RBP)
                                            + log(CDS_length)
                                            + transcript_GC
                                            + log1p(TPM)
                                          [OLS+HC3]

with the same TPM>=1 filter and same per-cell-line split.

Implementation:
  1. Extract canonical-CDS coordinates from GENCODE v35 GTF.
     Use one CDS per gene, choosing the transcript flagged
     "appris_principal_1" if available, otherwise longest CDS.
     Filter to protein_coding gene_biotype.
     Restrict amino-acid length to 100-1500 aa.
  2. Compute in-frame P-sites by walking spliced CDS in transcription
     direction (same algorithm as ORF, but on CDS exons).
  3. Compute transcript GC from GENCODE v35 transcript FASTA.
  4. Compute unique RBP count from same eCLIP intersect (transcript exons).
  5. Use already-loaded ENCODE polyA+ TPM (versionless ENST).
  6. Run lRPF regression per cell line.

This is computationally heavier than the prior steps because it scans
the entire GENCODE annotation, but uses the same algorithms.

Output: results/followup2_canonical_cds.csv  (regression table) +
        results/followup2_canonical_cds_per_orf.csv (per-CDS data)
"""
import gzip
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
import pyBigWig
import statsmodels.formula.api as smf
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

ROOT = Path(__file__).resolve().parent.parent
GTF = ROOT / "data" / "gencode.v35.annotation.gtf.gz"
TX_FASTA = ROOT / "data" / "gencode.v35.transcripts.fa.gz"
STEP1_EXPR = ROOT / "results" / "step1_expression.csv"
ECLIP_DIR = ROOT / "data" / "eclip"
RIBO_DIR = ROOT / "data" / "riboseq"
# Amino-acid length from CDS nucleotides. "legacy" retains the original
# length convention; "corrected" drops the spurious -1 (see select_canonical_cds).
# Selected by --cds-length; corrected runs write to their own files so that
# neither convention can silently overwrite the other's results.
CDS_LENGTH = "legacy"
OUT_REG = ROOT / "results" / "followup2_canonical_cds.csv"
OUT_PER = ROOT / "results" / "followup2_canonical_cds_per_cds.csv"
RES_DIR = ROOT / "results"

CELLS = ["K562", "HepG2"]
MIN_AA = 100   # exclude very short canonical (overlap with ncORF length range)
MAX_AA = 1500
SAMPLE_PER_GENE = "longest"   # could also do "appris_principal_1" if tag column reliable


def select_canonical_cds(gdf):
    """Return one transcript per gene: prefer appris_principal_1; fallback
    longest CDS span. Return DataFrame with one row per chosen transcript
    plus CDS interval list."""
    # filter to protein_coding genes
    gdf = gdf[gdf["gene_type"] == "protein_coding"].copy()
    cds = gdf[gdf["Feature"] == "CDS"].copy()
    cds["enst"] = cds["transcript_id"].str.split(".").str[0]
    cds["gene_id_v"] = cds["gene_id"].str.split(".").str[0]
    cds["L"] = cds["End"] - cds["Start"]
    # transcript -> total CDS length
    tx_len = cds.groupby("enst", as_index=False).agg(
        cds_len=("L", "sum"),
        gene_id_v=("gene_id_v", "first"))
    # appris tag check
    tx_attr = (gdf[gdf["Feature"] == "transcript"]
                .copy())
    tx_attr["enst"] = tx_attr["transcript_id"].str.split(".").str[0]
    tag_col = None
    for c in tx_attr.columns:
        if c == "tag":
            tag_col = c
            break
    if tag_col is not None:
        tx_attr["is_principal"] = tx_attr["tag"].astype(str).str.contains(
            "appris_principal_1", na=False)
    else:
        tx_attr["is_principal"] = False
    tx_attr = tx_attr[["enst", "is_principal"]].drop_duplicates("enst")
    tx_len = tx_len.merge(tx_attr, on="enst", how="left").fillna({"is_principal": False})
    # per gene: prefer principal; else longest
    tx_len = tx_len.sort_values(
        by=["gene_id_v", "is_principal", "cds_len"],
        ascending=[True, False, False])
    chosen = tx_len.groupby("gene_id_v", as_index=False).first()
    if CDS_LENGTH == "legacy":
        # Kept for reproducibility of the original length convention. The
        # comment this line used to carry ("minus stop") was wrong: a complete
        # GENCODE CDS feature already excludes the stop codon, which is
        # annotated separately and, verified on this GTF, never overlaps the
        # CDS intervals. So this subtracts a codon that was never counted, in
        # BOTH the 100-1500 window below and the log-length covariate.
        chosen["aa_length"] = (chosen["cds_len"] // 3) - 1
    elif CDS_LENGTH == "corrected":
        chosen["aa_length"] = chosen["cds_len"] // 3
    else:
        raise ValueError(f"unknown CDS length convention {CDS_LENGTH!r}")
    chosen["cds_length_convention"] = CDS_LENGTH
    chosen = chosen[(chosen.aa_length >= MIN_AA) & (chosen.aa_length <= MAX_AA)].copy()
    return chosen, cds


def build_transcript_gc_map():
    gc_map = {}
    with gzip.open(TX_FASTA, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            enst = rec.id.split("|")[0].split(".")[0]
            gc_map[enst] = float(gc_fraction(rec.seq))
    return gc_map


def count_inframe_psites(blocks_df, cell):
    """blocks_df has Chromosome, Start, End, Strand, enst columns.
    Returns dict enst -> in-frame P-site sum (sum of phase=0 across spliced CDS).
    Walking in transcription direction; for '-' strand reverse per block."""
    out = {}
    bw_plus = pyBigWig.open(str(RIBO_DIR / f"{cell}_P_sites_plus.bw"))
    bw_minus = pyBigWig.open(str(RIBO_DIR / f"{cell}_P_sites_minus.bw"))
    chr_set = set(bw_plus.chroms().keys())
    by_enst = blocks_df.groupby("enst")
    for enst, sub in by_enst:
        strand = sub["Strand"].iloc[0]
        blocks = list(zip(sub["Chromosome"], sub["Start"], sub["End"]))
        if strand == "+":
            blocks = sorted(blocks, key=lambda b: b[1])
        else:
            blocks = sorted(blocks, key=lambda b: b[1], reverse=True)
        bw = bw_plus if strand == "+" else bw_minus
        in_f = 0.0
        spliced_pos = 0
        for chrom, gs, ge in blocks:
            if chrom not in chr_set:
                in_f = np.nan
                break
            try:
                vals = bw.values(chrom, gs, ge, numpy=True)
            except Exception:
                in_f = np.nan; break
            vals = np.nan_to_num(vals, nan=0.0)
            if strand == "-":
                vals = vals[::-1]
            n = len(vals)
            phases = (np.arange(n) + spliced_pos) % 3
            in_f += float(vals[phases == 0].sum())
            spliced_pos += n
        out[enst] = in_f
    bw_plus.close(); bw_minus.close()
    return out


def main():
    global CDS_LENGTH, OUT_REG, OUT_PER
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cds-length", choices=("legacy", "corrected"),
                    default="legacy",
                    help="amino-acid length convention; 'legacy' reproduces "
                         "the submitted selection and covariate")
    CDS_LENGTH = ap.parse_args().cds_length
    if CDS_LENGTH == "corrected":
        OUT_REG = RES_DIR / "followup15_canonical_cds_corrected_length.csv"
        OUT_PER = RES_DIR / "followup15_canonical_cds_per_cds_corrected_length.csv"
    print(f"CDS length convention: {CDS_LENGTH}", flush=True)
    print("Loading GENCODE v35 GTF ...", flush=True)
    gdf = pr.read_gtf(str(GTF), as_df=True)
    print(f"  total GTF rows: {len(gdf)}", flush=True)
    chosen, cds = select_canonical_cds(gdf)
    print(f"  canonical CDS chosen (1 per gene, {MIN_AA}-{MAX_AA} aa): "
          f"{len(chosen)}", flush=True)

    # CDS blocks for chosen transcripts
    chosen_enst = set(chosen["enst"])
    cds_blocks = cds[cds["enst"].isin(chosen_enst)][
        ["Chromosome", "Start", "End", "Strand", "enst"]].copy()
    print(f"  CDS exon intervals: {len(cds_blocks)}", flush=True)

    # transcript-exon intersect with eCLIP for RBP count
    # use the entire transcript exon set (not CDS only) for binding, matching the ncORF design
    print("\nLoading host-transcript exon intervals (from existing parquet) ...", flush=True)
    # We need EVERY transcript's exons (for canonical comparison this means the chosen ENST set,
    # but the existing parquet only contains ORF-catalog ENSTs). Recompute from GTF.
    ex_df = gdf[gdf["Feature"] == "exon"].copy()
    ex_df["enst"] = ex_df["transcript_id"].str.split(".").str[0]
    ex_df = ex_df[ex_df["enst"].isin(chosen_enst)][
        ["Chromosome", "Start", "End", "Strand", "enst"]].copy()
    print(f"  exon intervals for chosen transcripts: {len(ex_df)}", flush=True)
    exon_pr = pr.PyRanges(ex_df)

    # eCLIP intersect per cell -> per-enst unique RBP count
    for cell in CELLS:
        print(f"\nLoading {cell} eCLIP peaks ...", flush=True)
        peak_rows = []
        for f in sorted((ECLIP_DIR / cell).glob("*.bed.gz")):
            rbp = f.name.split("__")[0].upper()
            d = pd.read_csv(f, sep="\t", header=None, compression="gzip",
                            usecols=[0, 1, 2, 5],
                            names=["Chromosome", "Start", "End", "Strand"],
                            dtype={0: str, 1: int, 2: int, 5: str})
            d["rbp"] = rbp
            peak_rows.append(d)
        peaks = pr.PyRanges(pd.concat(peak_rows, ignore_index=True))
        j = exon_pr.join(peaks, strandedness="same").df
        if len(j):
            per_tx = j.groupby("enst")["rbp"].nunique()
        else:
            per_tx = pd.Series(dtype=int)
        chosen[f"{cell}_nrbp"] = chosen["enst"].map(per_tx).fillna(0).astype(int)
        print(f"  {cell}: median nrbp = {chosen[f'{cell}_nrbp'].median():.0f}, "
              f"max = {chosen[f'{cell}_nrbp'].max()}", flush=True)

    # in-frame P-site sum on CDS per cell
    for cell in CELLS:
        print(f"\nCounting in-frame P-sites for {cell} on canonical CDS ...", flush=True)
        in_f = count_inframe_psites(cds_blocks, cell)
        chosen[f"{cell}_rpf_in"] = chosen["enst"].map(in_f)
        n_ok = int(chosen[f"{cell}_rpf_in"].notna().sum())
        print(f"  {cell}: P-site counted for {n_ok}/{len(chosen)} CDS", flush=True)

    # transcript GC
    print("\nBuilding transcript GC map ...", flush=True)
    gc_map = build_transcript_gc_map()
    chosen["gc"] = chosen["enst"].map(gc_map)

    # TPM (use Step 1 expression file; versionless ENST)
    expr = pd.read_csv(STEP1_EXPR)
    # expr is per-ORF — we need per-transcript TPM directly from RSEM TSVs.
    # The step1_expression.csv keys orf_name not enst; but the values are TPM by transcript
    # which is what we want IF we can recover the enst→TPM map. The CSV has 'enst' column.
    if "enst" in expr.columns:
        tpm_k = expr[["enst", "tpm_K562_polyA"]].drop_duplicates("enst").set_index("enst")["tpm_K562_polyA"].to_dict()
        tpm_h = expr[["enst", "tpm_HepG2_polyA"]].drop_duplicates("enst").set_index("enst")["tpm_HepG2_polyA"].to_dict()
    else:
        tpm_k = tpm_h = {}
    chosen["tpm_K562"] = chosen["enst"].map(tpm_k)
    chosen["tpm_HepG2"] = chosen["enst"].map(tpm_h)
    chosen["l_cdslen"] = np.log(chosen["aa_length"])  # log AA length

    OUT_PER.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(OUT_PER, index=False)
    print(f"\nPer-CDS table written: {OUT_PER}", flush=True)

    # regressions per cell
    rows = []
    for cell in CELLS:
        d = chosen.copy()
        d = d.dropna(subset=[f"{cell}_rpf_in", f"{cell}_nrbp",
                              "gc", f"tpm_{cell}"]).copy()
        d = d[d[f"tpm_{cell}"] >= 1.0].copy()
        if len(d) < 100:
            rows.append(dict(cell=cell, n=len(d), note="too few"))
            continue
        d["l1p_rbp"] = np.log1p(d[f"{cell}_nrbp"])
        d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}"])
        d["lRPF"] = np.log2(d[f"{cell}_rpf_in"] + 1)
        m = smf.ols("lRPF ~ l1p_rbp + l_cdslen + gc + l1p_tpm",
                    data=d).fit(cov_type="HC3")
        beta = m.params["l1p_rbp"]
        lo, hi = m.conf_int().loc["l1p_rbp"]
        # also compute partial R² for l1p_rbp
        m_red = smf.ols("lRPF ~ l_cdslen + gc + l1p_tpm", data=d).fit()
        full_r2 = m.rsquared
        red_r2 = m_red.rsquared
        sp_r2 = full_r2 - red_r2
        partial_r2 = sp_r2 / max(1 - red_r2, 1e-9)
        rows.append(dict(cell=cell, n=int(m.nobs),
                         beta=float(beta), ci_lo=float(lo), ci_hi=float(hi),
                         p_raw=float(m.pvalues["l1p_rbp"]),
                         full_R2=full_r2, reduced_R2=red_r2,
                         semi_partial_R2=sp_r2, partial_R2=partial_r2,
                         median_aa=int(d["aa_length"].median()),
                         median_nrbp=int(d[f"{cell}_nrbp"].median()),
                         median_tpm=float(d[f"tpm_{cell}"].median())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_REG, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== Canonical CDS lRPF regression (TPM>=1) ===")
    print(res.round({"beta": 3, "ci_lo": 3, "ci_hi": 3,
                     "full_R2": 3, "reduced_R2": 3,
                     "semi_partial_R2": 4, "partial_R2": 4,
                     "p_raw": 4, "median_tpm": 2}).to_string(index=False))

    print(f"\nWrote: {OUT_REG}")


if __name__ == "__main__":
    main()

"""Step 0 — establish the property distribution of the 7,264 GENCODE Phase I
Ribo-seq ORFs (microproteins).

Per ORF it assembles:
  (1) mean IDR        — metapredict.predict_disorder   (mean over residues, 0-1)
  (2) mean pLDDT      — metapredict.predict_pLDDT       (mean over residues, 0-100)
  (3) aa_length       — len(orf_sequence without trailing stop)
  (4) transcript GC%  — GC content of the source transcript (GENCODE v35)
  (5) catGRANULE2 LLPS score — read from results/catgranule2_llps_scores.csv
                        (computed separately in the catGRANULE venv)

Sources (also written into the output CSV header):
  - ORF table / sequences : data/Mudge2022_SupplementaryTables.xlsx  (S2 + S3)
  - transcript sequences  : data/gencode.v35.transcripts.fa.gz  (GENCODE v35 = Mudge 2022)
  - LLPS scores           : results/catgranule2_llps_scores.csv

If any ORF fails any computation (e.g. transcript
not found in v35 -> GC undefined), the script writes results/step0_failures.csv,
prints the failure counts, and stops without writing the final table or figures.
Missing predictions are not substituted.
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import metapredict as meta

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data" / "Mudge2022_SupplementaryTables.xlsx"
TX_FASTA = ROOT / "data" / "gencode.v35.transcripts.fa.gz"
LLPS_CSV = ROOT / "results" / "catgranule2_llps_scores.csv"
OUT_CSV = ROOT / "results" / "step0_properties.csv"
FAIL_CSV = ROOT / "results" / "step0_failures.csv"
FIG_DIR = ROOT.parent / "manuscript" / "figures"
SHEETS = ["S2. PHASE I Ribo-seq ORFs", "S3. Single-study Ribo-seq ORFs"]

# --- classification thresholds (explicit, provisional — easy to revise) ---
IDR_DISORDER_THRESHOLD = 0.5      # mean disorder >= this -> disordered
PLDDT_FOLDABLE_THRESHOLD = 70.0   # mean pLDDT >= this (and not disordered) -> foldable
LLPS_PRONE_THRESHOLD = 0.5        # catGRANULE2 score >= this -> LLPS-prone


def load_orf_table():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    rows_out = []
    for sheet in SHEETS:
        ws = wb[sheet]
        it = ws.iter_rows(values_only=True)
        header = next(it)
        idx = {h: i for i, h in enumerate(header) if h}
        for row in it:
            name = row[idx["orf_name"]]
            if name is None or not str(name).strip():
                continue
            seq = str(row[idx["orf_sequence"]]).strip()
            seq_no_stop = seq[:-1] if seq.endswith("*") else seq
            rows_out.append(
                dict(
                    orf_name=str(name),
                    source_sheet="S2" if sheet.startswith("S2") else "S3",
                    transcript=str(row[idx["transcript"]]) if row[idx["transcript"]] else "",
                    orf_biotype=row[idx["orf_biotype"]],
                    gene_name=row[idx["gene_name"]],
                    aa_sequence=seq_no_stop,
                    aa_length=len(seq_no_stop),
                )
            )
    df = pd.DataFrame(rows_out)
    if df["orf_name"].duplicated().any():
        raise ValueError("duplicate orf_name in ORF table")
    return df


def build_transcript_gc_map():
    """versionless ENST -> GC fraction (0-1) of the GENCODE v35 transcript."""
    gc_map = {}
    with gzip.open(TX_FASTA, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            enst = rec.id.split("|")[0].split(".")[0]
            gc_map[enst] = gc_fraction(rec.seq)
    return gc_map


def main():
    print("Loading ORF table ...", flush=True)
    df = load_orf_table()
    print(f"  {len(df)} ORFs (S2={int((df.source_sheet=='S2').sum())}, "
          f"S3={int((df.source_sheet=='S3').sum())})", flush=True)

    # (1)(2) metapredict IDR + pLDDT (batched)
    print("Running metapredict (IDR + pLDDT) ...", flush=True)
    seqs = df["aa_sequence"].tolist()
    dis_batch = meta.predict_disorder_batch(seqs, show_progress_bar=False)
    plddt_batch = meta.predict_pLDDT(seqs)  # accepts a list; returns [seq, array] like the batch fn
    # batch returns list of [seq, np.array]; align by order (metapredict preserves order)
    df["mean_idr"] = [float(np.mean(p[1])) for p in dis_batch]
    df["mean_plddt"] = [float(np.mean(p[1])) for p in plddt_batch]

    # (4) transcript GC
    print("Building GENCODE v35 transcript GC map ...", flush=True)
    gc_map = build_transcript_gc_map()
    print(f"  {len(gc_map)} transcripts indexed", flush=True)
    df["transcript_versionless"] = df["transcript"].str.split(".").str[0]
    df["transcript_gc"] = df["transcript_versionless"].map(gc_map)

    # (5) catGRANULE LLPS scores
    if not LLPS_CSV.exists():
        raise FileNotFoundError(
            f"{LLPS_CSV} not found — run scripts/run_catgranule_scores.py first."
        )
    llps = pd.read_csv(LLPS_CSV)
    df = df.merge(llps, on="orf_name", how="left")

    # --- failure accounting (no dummy fills) ---
    fail = pd.DataFrame({"orf_name": df["orf_name"]})
    fail["missing_gc"] = df["transcript_gc"].isna()
    fail["missing_llps"] = df["catgranule2_llps_score"].isna()
    fail["bad_idr"] = df["mean_idr"].isna()
    fail["bad_plddt"] = df["mean_plddt"].isna()
    fail["any_fail"] = fail[["missing_gc", "missing_llps", "bad_idr", "bad_plddt"]].any(axis=1)
    n_fail = int(fail["any_fail"].sum())

    if n_fail:
        failed = df[fail["any_fail"].values].copy()
        failed = failed.merge(fail, on="orf_name")
        FAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
        failed.to_csv(FAIL_CSV, index=False)
        print("\n*** FAILURES — stopping without writing final table (no dummy fills) ***",
              flush=True)
        print(f"  total ORFs with >=1 failed metric: {n_fail}", flush=True)
        print(f"    missing transcript GC (not in v35): {int(fail['missing_gc'].sum())}", flush=True)
        print(f"    missing catGRANULE LLPS score:      {int(fail['missing_llps'].sum())}", flush=True)
        print(f"    bad mean_idr:                       {int(fail['bad_idr'].sum())}", flush=True)
        print(f"    bad mean_plddt:                     {int(fail['bad_plddt'].sum())}", flush=True)
        print(f"  details -> {FAIL_CSV}", flush=True)
        raise SystemExit(1)

    # --- classify ---
    def classify(r):
        if r.mean_idr >= IDR_DISORDER_THRESHOLD:
            return "disordered"
        if r.mean_plddt >= PLDDT_FOLDABLE_THRESHOLD:
            return "foldable"
        return "intermediate"

    df["structure_group"] = df.apply(classify, axis=1)
    df["llps_prone"] = df["catgranule2_llps_score"] >= LLPS_PRONE_THRESHOLD

    # --- write final table with provenance header ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "orf_name", "source_sheet", "transcript", "gene_name", "orf_biotype",
        "aa_length", "mean_idr", "mean_plddt", "transcript_gc",
        "catgranule2_llps_score", "structure_group", "llps_prone",
    ]
    header_lines = [
        "# Step 0 microprotein properties",
        f"# ORF source: {XLSX.name} (sheets {SHEETS})",
        f"# transcript GC source: {TX_FASTA.name} (GENCODE v35)",
        f"# LLPS source: {LLPS_CSV.name} (catGRANULE 2.0 ROBOT, only_pc/RandomForest)",
        f"# thresholds: IDR>={IDR_DISORDER_THRESHOLD} disordered; "
        f"pLDDT>={PLDDT_FOLDABLE_THRESHOLD} foldable; LLPS>={LLPS_PRONE_THRESHOLD} prone",
    ]
    with open(OUT_CSV, "w") as fh:
        fh.write("\n".join(header_lines) + "\n")
        df[cols].to_csv(fh, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}", flush=True)

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for col, label, fname in [
        ("mean_idr", "mean IDR (metapredict disorder)", "step0_idr_distribution.png"),
        ("mean_plddt", "mean pLDDT (metapredict)", "step0_plddt_distribution.png"),
        ("catgranule2_llps_score", "catGRANULE 2.0 LLPS score", "step0_llps_distribution.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(df[col], bins=60, color="#4477aa", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(label)
        ax.set_ylabel("number of ORFs")
        ax.set_title(f"Step 0: {label}  (n={len(df)})\nsource: {XLSX.name} S2+S3")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=130)
        plt.close(fig)
        print(f"  figure -> {FIG_DIR / fname}", flush=True)

    # --- report ---
    print("\n=== Step 0 summary ===", flush=True)
    print(df[["aa_length", "mean_idr", "mean_plddt", "transcript_gc",
              "catgranule2_llps_score"]].describe().round(3).to_string(), flush=True)
    print("\nStructure group counts (IDR/pLDDT based):", flush=True)
    sg = df["structure_group"].value_counts()
    for g, n in sg.items():
        print(f"  {g:13s} {n:5d}  ({100*n/len(df):.1f}%)", flush=True)
    n_llps = int(df["llps_prone"].sum())
    print(f"\nLLPS-prone (catGRANULE2 >= {LLPS_PRONE_THRESHOLD}): "
          f"{n_llps} ({100*n_llps/len(df):.1f}%)", flush=True)


if __name__ == "__main__":
    main()

"""Step 2 — frame-aware P-site counting per ORF and distribution summaries.

Source: GSE129061 (Calviello et al. 2020, hg38, GENCODE v25 alignment).
ORF coordinates are read from the Mudge 2022 supplementary table (1-based
inclusive, possibly semicolon-separated blocks) and converted to 0-based,
half-open intervals. Blocks are traversed in transcription order. Spliced
positions modulo three define the in-frame and two off-frame counts.
Minus-strand arrays are reversed within each block before accumulation.

Output: analysis/results/step2_rpf_counts.csv. Regression is performed by
separate scripts; this script counts footprints and reports distributions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
import pyBigWig

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data" / "Mudge2022_SupplementaryTables.xlsx"
RIBO = ROOT / "data" / "riboseq"
STEP0 = ROOT / "results" / "step0_properties.csv"
OUT = ROOT / "results" / "step2_rpf_counts.csv"
CELLS = ["K562", "HepG2"]
SHEETS = ["S2. PHASE I Ribo-seq ORFs", "S3. Single-study Ribo-seq ORFs"]


def load_orf_blocks():
    """Return list of dicts: orf_name, chrom, strand, blocks=[(start0, end0),...]."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    out = []
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
            chrom = chrm if chrm.startswith("chr") else "chr" + chrm
            strand = row[idx["strand"]]
            starts = [int(float(s)) for s in str(row[idx["starts"]]).split(";")]
            ends = [int(float(e)) for e in str(row[idx["ends"]]).split(";")]
            blocks = [(s - 1, e) for s, e in zip(starts, ends)]  # 1-based incl -> 0-based half-open
            out.append(dict(orf_name=str(name), chrom=chrom, strand=strand, blocks=blocks))
    return out


def count_framewise(orf, bw, chrom_set):
    """Return (in_frame, off1, off2) for an ORF given an open bigWig (strand-matched)."""
    if orf["chrom"] not in chrom_set:
        return (np.nan, np.nan, np.nan)
    blocks = orf["blocks"]
    if orf["strand"] == "+":
        ordered = sorted(blocks)
    else:
        ordered = sorted(blocks, reverse=True)
    in_f = off1 = off2 = 0.0
    spliced_pos = 0
    for gs, ge in ordered:
        try:
            vals = bw.values(orf["chrom"], gs, ge, numpy=True)
        except Exception:
            return (np.nan, np.nan, np.nan)
        vals = np.nan_to_num(vals, nan=0.0)
        if orf["strand"] == "-":
            vals = vals[::-1]
        n = len(vals)
        phases = (np.arange(n) + spliced_pos) % 3
        in_f += float(vals[phases == 0].sum())
        off1 += float(vals[phases == 1].sum())
        off2 += float(vals[phases == 2].sum())
        spliced_pos += n
    return (in_f, off1, off2)


def main():
    print("Loading ORF blocks from xlsx ...", flush=True)
    orfs = load_orf_blocks()
    print(f"  {len(orfs)} ORFs", flush=True)

    bws = {}
    chrom_sets = {}
    for cell in CELLS:
        for strand_label, strand in [("plus", "+"), ("minus", "-")]:
            f = RIBO / f"{cell}_P_sites_{strand_label}.bw"
            bw = pyBigWig.open(str(f))
            bws[(cell, strand)] = bw
        chrom_sets[cell] = set(bws[(cell, "+")].chroms().keys())

    rows = []
    for orf in orfs:
        rec = dict(orf_name=orf["orf_name"], chrom=orf["chrom"], strand=orf["strand"])
        for cell in CELLS:
            bw = bws[(cell, orf["strand"])]
            in_f, off1, off2 = count_framewise(orf, bw, chrom_sets[cell])
            rec[f"{cell}_rpf_in"] = in_f
            rec[f"{cell}_rpf_off1"] = off1
            rec[f"{cell}_rpf_off2"] = off2
        rows.append(rec)

    for bw in bws.values():
        bw.close()

    df = pd.DataFrame(rows)
    for cell in CELLS:
        df[f"{cell}_rpf_off"] = df[f"{cell}_rpf_off1"] + df[f"{cell}_rpf_off2"]
        df[f"{cell}_rpf_total"] = df[f"{cell}_rpf_in"] + df[f"{cell}_rpf_off"]

    # merge meta from step0
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    s0 = pd.read_csv(STEP0, skiprows=nc)[["orf_name", "source_sheet", "orf_biotype", "aa_length"]]
    df = df.merge(s0, on="orf_name")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}", flush=True)

    # ============= REPORT ON MAIN POP (no regression) =============
    main = df[(df.source_sheet == "S2") & (df.aa_length <= 100)].copy()
    print(f"\n========== Step 2 Phase A — distribution report (MAIN n={len(main)}) ==========",
          flush=True)

    for cell in CELLS:
        s_in = main[f"{cell}_rpf_in"]
        s_off = main[f"{cell}_rpf_off"]
        s_tot = main[f"{cell}_rpf_total"]
        print(f"\n--- {cell} ---", flush=True)
        # in-frame RPF count distribution
        nz = int((s_in == 0).sum())
        print(f"  in-frame RPF count:", flush=True)
        print(f"    zero count (in-frame=0): {nz}  ({100*nz/len(main):.1f}%)", flush=True)
        print(f"    distribution: mean={s_in.mean():.2f}  median={int(s_in.median())} "
              f"p75={int(s_in.quantile(.75))}  p90={int(s_in.quantile(.90))} "
              f"p99={int(s_in.quantile(.99))}  max={int(s_in.max())}", flush=True)
        # total RPF (in + off)
        nz_tot = int((s_tot == 0).sum())
        print(f"  total RPF (in+off): zero count = {nz_tot} ({100*nz_tot/len(main):.1f}%); "
              f"median={int(s_tot.median())} mean={s_tot.mean():.2f}", flush=True)
        # in-frame fraction (where total > 0)
        nonzero = s_tot > 0
        frac = (s_in[nonzero] / s_tot[nonzero])
        print(f"  in-frame fraction (computed where total>0, n={int(nonzero.sum())}):", flush=True)
        print(f"    median={frac.median():.3f}  mean={frac.mean():.3f} "
              f"p25={frac.quantile(.25):.3f}  p75={frac.quantile(.75):.3f}", flush=True)
        # by ORF type
        print(f"  in-frame fraction by ORF type (median; n total / n with total>0):", flush=True)
        for bt, sub in main.groupby("orf_biotype"):
            sit = sub[f"{cell}_rpf_in"]
            tot = sub[f"{cell}_rpf_total"]
            nz_sub = int((sit == 0).sum())
            nzt = (tot > 0)
            if nzt.sum() == 0:
                med_frac = float("nan")
            else:
                med_frac = (sit[nzt] / tot[nzt]).median()
            print(f"    {bt:22s}  n={len(sub):4d}  in=0: {nz_sub:4d} ({100*nz_sub/len(sub):.0f}%)  "
                  f"in-frame frac median={med_frac:.3f}  med RPF in-frame={int(sit.median())}",
                  flush=True)

    print("\n=== STOP — distribution report only, no regression run ===", flush=True)
    print("Next: decide TE formula and zero-handling from this distribution.", flush=True)


if __name__ == "__main__":
    main()


"""Attach K562 / HepG2 transcript TPM to each ORF and report the STRUCTURE of
missingness, reporting abundance coverage before model filtering.

Expression source: RSEM transcript quantifications (GRCh38 V29), 2-rep mean.
  K562  polyA (main):        ENCFF720TFN + ENCFF522DDA
  HepG2 polyA (main):        ENCFF458NJB + ENCFF371KBB
  K562  total (sensitivity): ENCFF012LGS + ENCFF714FKW

Matching: ORF.transcript (versionless ENST) vs RSEM transcript_id (versionless).
Two distinct missing categories are kept separate:
  NA   = ORF's ENST is ABSENT from the quant file (not measured / annotation gap)
  zero = ENST present but TPM == 0 (measured, not detected)

Reports (from data directly, no dummy fills):
  (1) per cell line x ORF biotype cross-tab of NA / zero / expressed
  (2) NA vs zero totals per cell line
  (3) missing rate within MAIN population (<=100aa & S2)
  (4) RBPs measured in both cell lines (from eCLIP manifest)
Writes results/step1_expression.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RNA_DIR = ROOT / "data" / "rnaseq"
STEP0 = ROOT / "results" / "step0_properties.csv"
MANIFEST = ROOT / "results" / "eclip_manifest.tsv"
OUT = ROOT / "results" / "step1_expression.csv"

GROUPS = {
    "K562_polyA":  ["K562_polyA_main_ENCSR000AEO_rep1_ENCFF720TFN.tsv",
                    "K562_polyA_main_ENCSR000AEO_rep2_ENCFF522DDA.tsv"],
    "HepG2_polyA": ["HepG2_polyA_main_ENCSR985KAT_rep1_ENCFF458NJB.tsv",
                    "HepG2_polyA_main_ENCSR985KAT_rep2_ENCFF371KBB.tsv"],
    "K562_total":  ["K562_total_sensitivity_ENCSR109IQO_rep1_ENCFF012LGS.tsv",
                    "K562_total_sensitivity_ENCSR109IQO_rep2_ENCFF714FKW.tsv"],
}


def load_rsem_versionless(path):
    """versionless ENST -> TPM (float).

    GENCODE lists pseudoautosomal (PAR_Y) transcripts twice: the chrX copy
    (ENST.._N) and the chrY copy (ENST.._N_PAR_Y). Both collapse to the same
    versionless ENST. We SUM their TPM (total locus abundance) and verify the
    only collisions are exactly these PAR_Y pairs — any other collision raises.
    """
    grouped = defaultdict(list)  # versionless -> [(full_id, tpm), ...]
    with open(path) as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            tid = row["transcript_id"]
            if not tid.startswith("ENST"):
                continue  # skip spike-ins / tRNAs
            grouped[tid.split(".")[0]].append((tid, float(row["TPM"])))

    out = {}
    n_par = 0
    for enst, items in grouped.items():
        if len(items) == 1:
            out[enst] = items[0][1]
            continue
        # collision: must be a PAR_Y pair (one normal + one *_PAR_Y)
        par = [i for i in items if i[0].endswith("_PAR_Y")]
        if len(items) == 2 and len(par) == 1:
            out[enst] = sum(t for _, t in items)  # sum X + Y copies
            n_par += 1
        else:
            raise ValueError(
                f"non-PAR_Y versionless collision for {enst} in {path.name}: "
                f"{[i[0] for i in items]}"
            )
    return out, n_par


def mean_tpm_for_group(files):
    loaded = [load_rsem_versionless(RNA_DIR / f) for f in files]
    maps = [m for m, _ in loaded]
    par_counts = [p for _, p in loaded]
    keys = set(maps[0])
    for m in maps[1:]:
        if set(m) != keys:
            # same annotation across reps -> identical key set expected
            raise ValueError("replicate transcript sets differ; investigate before averaging")
    merged = {k: float(np.mean([m[k] for m in maps])) for k in keys}
    return merged, par_counts


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def status_for(enst, tpm_map):
    if enst not in tpm_map:
        return "NA"
    return "zero" if tpm_map[enst] == 0.0 else "expressed"


def main():
    print("Loading RSEM TPM maps (2-rep mean) ...", flush=True)
    tpm = {}
    for g, files in GROUPS.items():
        merged, par_counts = mean_tpm_for_group(files)
        tpm[g] = merged
        print(f"  {g}: {len(merged)} transcripts (PAR_Y pairs summed per rep: {par_counts})", flush=True)

    df = read_step0()
    df["enst"] = df["transcript"].str.split(".").str[0]
    for g, m in tpm.items():
        df[f"tpm_{g}"] = df["enst"].map(m)
        df[f"status_{g}"] = df["enst"].apply(lambda e: status_for(e, m))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}", flush=True)

    main_pop = df[(df.source_sheet == "S2") & (df.aa_length <= 100)]

    for cell in ["K562_polyA", "HepG2_polyA"]:
        sc = f"status_{cell}"
        print(f"\n================  {cell}  ================", flush=True)
        # (2) NA vs zero totals
        vc = df[sc].value_counts()
        n = len(df)
        print(f"(2) all ORFs (n={n}):")
        for k in ["expressed", "zero", "NA"]:
            c = int(vc.get(k, 0))
            print(f"      {k:10s} {c:5d}  ({100*c/n:.1f}%)")
        # (1) cross-tab by biotype
        print(f"(1) by ORF biotype  [expressed / zero / NA]:")
        ct = pd.crosstab(df["orf_biotype"], df[sc])
        for bt in ct.index:
            row = ct.loc[bt]
            tot = row.sum()
            exp = int(row.get("expressed", 0)); ze = int(row.get("zero", 0)); na = int(row.get("NA", 0))
            print(f"      {bt:22s} n={tot:5d}  exp={exp:5d}  zero={ze:4d}  NA={na:4d}  "
                  f"(NA {100*na/tot:.1f}%, zero {100*ze/tot:.1f}%)")
        # (3) main population missingness
        m_sc = main_pop[sc].value_counts()
        nm = len(main_pop)
        na_m = int(m_sc.get("NA", 0)); ze_m = int(m_sc.get("zero", 0))
        print(f"(3) MAIN pop (<=100aa & S2, n={nm}): "
              f"NA={na_m} ({100*na_m/nm:.1f}%), zero={ze_m} ({100*ze_m/nm:.1f}%), "
              f"usable(>0)={nm-na_m-ze_m} ({100*(nm-na_m-ze_m)/nm:.1f}%)")

    # (4) common RBPs from manifest
    rbp = defaultdict(set)
    with open(MANIFEST) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rbp[row["cell_line"]].add(row["rbp"])
    common = rbp["K562"] & rbp["HepG2"]
    print(f"\n(4) RBPs: K562={len(rbp['K562'])} HepG2={len(rbp['HepG2'])} "
          f"common(both lines)={len(common)}", flush=True)


if __name__ == "__main__":
    main()


"""Structural-property regressions on a supplied canonical-CDS reference.

The canonical reference is selected by annotation rather than Ribo-seq
detection; its ascertainment therefore differs from the ncORF catalogue.
TPM is rebuilt from complete ENCODE RSEM quantifications rather than an
ncORF-host-restricted expression table: sum within versionless ENST groups,
then take the mean across the two replicates.

MODEL (identical to Step 1 axis (a)):
    outcome ~ log1p(unique_RBP) + log(AA_length) + GC + log1p(TPM)  [OLS+HC3]

Runs in .venv_ranges (statsmodels). Consumes results/followup7_canonical_properties.csv
written by followup7a (which runs in .venv, metapredict).
No dummy values, no imputation. Failures raise.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROPS = RES / "followup7_canonical_properties.csv"
RNASEQ = ROOT / "data" / "rnaseq"
OUT = RES / "followup7_canonical_structural.csv"
QUANT = {
    "K562":  ["K562_polyA_main_ENCSR000AEO_rep1_ENCFF720TFN.tsv",
              "K562_polyA_main_ENCSR000AEO_rep2_ENCFF522DDA.tsv"],
    "HepG2": ["HepG2_polyA_main_ENCSR985KAT_rep1_ENCFF458NJB.tsv",
              "HepG2_polyA_main_ENCSR985KAT_rep2_ENCFF371KBB.tsv"],
}
CELLS = ["K562", "HepG2"]
OUTCOMES = ["mean_plddt", "mean_disorder"]


def tpm_map(fname):
    path = RNASEQ / fname
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", usecols=["transcript_id", "TPM"])
    df = df[df.transcript_id.astype(str).str.startswith("ENST")].copy()
    df["base"] = df.transcript_id.str.split(".").str[0]
    return df.groupby("base")["TPM"].sum().to_dict()


def mean_tpm(files):
    maps = [tpm_map(f) for f in files]
    keys = set(maps[0])
    for m in maps[1:]:
        keys &= set(m)
    if not keys:
        raise RuntimeError("no shared ENST between replicates")
    return {k: float(np.mean([m[k] for m in maps])) for k in keys}


def main():
    if not PROPS.exists():
        raise FileNotFoundError(f"{PROPS} — run followup7a first (.venv)")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--props", type=Path, default=PROPS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    globals()["OUT"] = args.out
    if not args.props.exists():
        raise SystemExit(f"{args.props} does not exist — run "
                         "followup7a_canonical_properties.py first")
    cds = pd.read_csv(args.props)
    print(f"canonical rows with properties : {cds['mean_plddt'].notna().sum()} / {len(cds)}")

    for cell in CELLS:
        m = mean_tpm(QUANT[cell])
        cds[f"tpm_{cell}_fixed"] = cds["base"].map(m)
        print(f"  TPM {cell:5s} rebuilt {cds[f'tpm_{cell}_fixed'].notna().sum():6d} / {len(cds)}"
              f"   (followup2 value present for only {cds[f'tpm_{cell}'].notna().sum()},"
              f" all ncORF hosts)")

    rows = []
    for cell in CELLS:
        d = cds.copy()
        d["nrbp"] = d[f"{cell}_nrbp"]
        d["tpm"] = d[f"tpm_{cell}_fixed"]
        d = d.dropna(subset=OUTCOMES + ["nrbp", "tpm", "gc", "aa_length"])
        d["l1p_rbp"] = np.log1p(d["nrbp"])
        d["l_aalen"] = np.log(d["aa_length"])
        d["l1p_tpm"] = np.log1p(d["tpm"])
        for outcome in OUTCOMES:
            fit = smf.ols(f"{outcome} ~ l1p_rbp + l_aalen + gc + l1p_tpm",
                          data=d).fit(cov_type="HC3")
            red = smf.ols(f"{outcome} ~ l_aalen + gc + l1p_tpm",
                          data=d).fit(cov_type="HC3")
            lo, hi = fit.conf_int().loc["l1p_rbp"]
            sd = d[outcome].std()
            rows.append(dict(
                cell=cell, outcome=outcome, n=len(d),
                beta=fit.params["l1p_rbp"], ci_lo=lo, ci_hi=hi,
                p=fit.pvalues["l1p_rbp"], outcome_sd=sd,
                ci_halfwidth_in_sd=(hi - lo) / 2 / sd,
                full_R2=fit.rsquared, reduced_R2=red.rsquared,
                semi_partial_R2=fit.rsquared - red.rsquared,
                median_nrbp=float(d["nrbp"].median()),
                median_tpm=float(d["tpm"].median())))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print("\n--- Step-1 model on canonical CDSs (OLS + HC3) ---")
    print(out[["cell", "outcome", "n", "beta", "ci_lo", "ci_hi", "p",
               "ci_halfwidth_in_sd", "semi_partial_R2"]].to_string(index=False))
    print(f"\nwritten: {OUT}")
    print("\nci_halfwidth_in_sd is the confidence-interval half-width divided")
    print("by the outcome SD; it is not a fully standardized coefficient.")


if __name__ == "__main__":
    main()

"""Descriptive preparation for ORF-type stratified analysis.

Produces per-orf_biotype descriptive aggregation on the MAIN population
(S2 replicated & aa_length<=100), assesses per-type analyzability for
single-type OLS regression (threshold n>=100), using the directional
hypotheses below. Aggregates directly from CSVs; no
hardcoding.

PRESPECIFIED HYPOTHESES (not external preregistration):
  A. uORF    × property regression -> NULL prediction
     (claim is translation-rate-side; properties of upstream ORFs expected to
      be uncorrelated with their host transcript's RBP binding)
  B. lncRNA  × property regression -> EFFECT+ prediction
     (lncRNA-ORFs may be the subclass where RBP binding shapes property class;
      foldable-bias and dependence on transcript-level RBP regulation more
      plausible biologically here than for uORF/dORF/intORF)
  C. dORF    × property regression -> NULL prediction
     (downstream ORFs, 3'UTR-resident; properties not expected to depend on
      transcript RBP binding magnitude)
  D. intORF  × property regression -> NULL prediction
     (internal/CDS-resident ORFs, often out-of-frame; properties decoupled from
      transcript RBP regulation)

Outcomes per type (where n>=100): pLDDT, catGRANULE (continuous, OLS+HC3 in
the eventual regression). The single hypothesis with EFFECT+ prior is B.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
BIND = ROOT / "results" / "step1_binding.csv"
OUT_DESC = ROOT / "results" / "step1e_type_descriptive.csv"

LEN_CAP = 100

# Prespecified prior per type (matches the 4 hypotheses above)
PRIORS = {
    "uORF":   "null",
    "lncRNA": "effect+",
    "dORF":   "null",
    "intORF": "null",
    # other types: not prespecified; reported descriptively only
}
ANALYZABLE_THRESHOLD = 100


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
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]
    df = s0.merge(bind, on="orf_name")
    main = df[(df.source_sheet == "S2") & (df.aa_length <= LEN_CAP)].copy()
    print(f"MAIN population: n={len(main)} (S2 & aa_length <= {LEN_CAP})\n")

    rows = []
    for bt, sub in main.groupby("orf_biotype"):
        n = len(sub)
        sg = sub["structure_group"].value_counts()
        n_dis = int(sg.get("disordered", 0))
        n_int = int(sg.get("intermediate", 0))
        n_fol = int(sg.get("foldable", 0))
        n_llps = int((sub["catgranule2_llps_score"] >= 0.5).sum())
        # binding pervasiveness per type
        k562_bound = int((sub["K562_nrbp_tx"] > 0).sum())
        hepg2_bound = int((sub["HepG2_nrbp_tx"] > 0).sum())
        rows.append(dict(
            orf_biotype=bt,
            n=n,
            prior=PRIORS.get(bt, "n/a"),
            analyzable=("yes" if n >= ANALYZABLE_THRESHOLD else "no"),
            idr_mean=sub["mean_idr"].mean(),
            idr_median=sub["mean_idr"].median(),
            plddt_mean=sub["mean_plddt"].mean(),
            plddt_median=sub["mean_plddt"].median(),
            catg_mean=sub["catgranule2_llps_score"].mean(),
            catg_median=sub["catgranule2_llps_score"].median(),
            aa_len_median=sub["aa_length"].median(),
            disordered=n_dis, disordered_pct=100 * n_dis / n,
            intermediate=n_int, intermediate_pct=100 * n_int / n,
            foldable=n_fol, foldable_pct=100 * n_fol / n,
            llps_prone=n_llps, llps_prone_pct=100 * n_llps / n,
            k562_bound=k562_bound, k562_bound_pct=100 * k562_bound / n,
            hepg2_bound=hepg2_bound, hepg2_bound_pct=100 * hepg2_bound / n,
        ))
    res = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    OUT_DESC.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DESC, index=False)

    # Formatted display
    pd.set_option("display.width", 200, "display.max_columns", 30)
    cols_show = ["orf_biotype", "n", "prior", "analyzable",
                 "idr_median", "plddt_median", "catg_median",
                 "disordered_pct", "intermediate_pct", "foldable_pct",
                 "llps_prone_pct", "k562_bound_pct", "hepg2_bound_pct"]
    print("Descriptive aggregation by orf_biotype (MAIN, sorted by n):")
    print(res[cols_show].round(2).to_string(index=False))
    print(f"\nWrote {OUT_DESC}")

    # Analyzability summary
    print("\n=== analyzability per prespecified type ===")
    for bt in PRIORS:
        row = res[res.orf_biotype == bt]
        if len(row) == 0:
            print(f"  {bt:8s}: ABSENT from main pop")
            continue
        n = int(row.iloc[0]["n"]); ok = row.iloc[0]["analyzable"]
        print(f"  {bt:8s}: n={n:4d}  prior={PRIORS[bt]:8s}  analyzable={ok}")

    print("\n=== non-prespecified types (reported descriptively only) ===")
    for bt in res["orf_biotype"]:
        if bt not in PRIORS:
            n = int(res[res.orf_biotype == bt].iloc[0]["n"])
            print(f"  {bt}: n={n}")


if __name__ == "__main__":
    main()


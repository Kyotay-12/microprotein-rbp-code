"""Regional joint model for coding-host main-population ORFs.

OLS with HC3 covariance, separately by cell line:
outcome ~ utr5_density + cds_density + utr3_density
          + log(AA_len) + GC + log1p(TPM).

Density is the number of unique RBPs per kilobase of the corresponding
transcript region. Outcomes are mean pLDDT and catGRANULE score. Cases lacking
a region density, abundance or required covariate are dropped.

Output: analysis/results/step1c_regional_results.csv, containing coefficients,
confidence intervals, nominal p-values, adjusted p-values and directional priors.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
REGION = ROOT / "results" / "step1c_region_binding.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
OUT = ROOT / "results" / "step1c_regional_results.csv"
CELLS = ["K562", "HepG2"]
REGIONS = ["utr5", "cds", "utr3"]
PRIOR = {"utr5": "effect", "cds": "neutral", "utr3": "effect"}
OUTCOMES = {"mean_plddt": "pLDDT", "catgranule2_llps_score": "catGRANULE"}


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)[
        ["orf_name", "mean_plddt", "catgranule2_llps_score", "transcript_gc"]]


def main():
    region = pd.read_csv(REGION)
    s0 = read_step0()
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    df = region.merge(s0, on="orf_name").merge(expr, on="orf_name")
    df["l_aalen"] = np.log(df["aa_length"])
    df["gc"] = df["transcript_gc"]

    base = df[(df.source_sheet == "S2") & (df.aa_length <= 100) & (df.host_coding)].copy()
    rows = []
    for cell in CELLS:
        d = base.copy()
        d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
        dens = {r: f"{cell}_{r}_density" for r in REGIONS}
        need = list(dens.values()) + ["l1p_tpm", "l_aalen", "gc",
                                      "mean_plddt", "catgranule2_llps_score"]
        n_before = len(d)
        d = d.dropna(subset=need)
        print(f"\n{cell}: analysis n={len(d)} (coding-host main {n_before}; "
              f"dropped NaN density/TPM: {n_before - len(d)})", flush=True)

        # rename density cols to stable names for the formula
        for r in REGIONS:
            d[f"{r}_density"] = d[dens[r]]
        rhs = "utr5_density + cds_density + utr3_density + l_aalen + gc + l1p_tpm"
        for outcome, olabel in OUTCOMES.items():
            m = smf.ols(f"{outcome} ~ {rhs}", data=d).fit(cov_type="HC3")
            for r in REGIONS:
                term = f"{r}_density"
                beta = m.params[term]
                lo, hi = m.conf_int().loc[term]
                rows.append(dict(cell=cell, outcome=olabel, region=r,
                                 prior=PRIOR[r], beta=beta, ci_lo=lo, ci_hi=hi,
                                 p_raw=m.pvalues[term], R2=m.rsquared, n=len(d)))
            c = m.params; ci = m.conf_int()
            print(f"  [OLS {olabel}] R2={m.rsquared:.3f}  "
                  + "  ".join(f"{r}:b={c[f'{r}_density']:+.3f}[{ci.loc[f'{r}_density'][0]:+.2f},"
                              f"{ci.loc[f'{r}_density'][1]:+.2f}]p={m.pvalues[f'{r}_density']:.2f}"
                              for r in REGIONS), flush=True)

    res = pd.DataFrame(rows)
    # multiple-comparison correction across the 12 regional coefficients
    res["p_BH"] = multipletests(res["p_raw"], method="fdr_bh")[1]
    res["p_bonf"] = multipletests(res["p_raw"], method="bonferroni")[1]
    res = res[["cell", "outcome", "region", "prior", "beta", "ci_lo", "ci_hi",
               "p_raw", "p_BH", "p_bonf", "R2", "n"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("\n================ REGIONAL JOINT MODEL — 12-row result table ================", flush=True)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(res.round({"beta": 3, "ci_lo": 3, "ci_hi": 3, "p_raw": 4, "p_BH": 4,
                     "p_bonf": 4, "R2": 3}).to_string(index=False), flush=True)
    n_sig = int((res["p_bonf"] < 0.05).sum())
    print(f"\n  regional coefficients significant after Bonferroni: {n_sig} / 12", flush=True)
    # any region significant AND reproduced in both cells?
    sig = res[res["p_raw"] < 0.05]
    repro = []
    for (outcome, region), g in sig.groupby(["outcome", "region"]):
        if g["cell"].nunique() == 2 and (np.sign(g["beta"]).nunique() == 1):
            repro.append((outcome, region))
    print(f"  regions significant (raw p<0.05) AND same-sign in BOTH cells: "
          f"{repro if repro else 'none'}", flush=True)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()


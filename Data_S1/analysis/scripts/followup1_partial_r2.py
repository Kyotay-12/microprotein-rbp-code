"""Incremental model fit associated with the log1p(unique_RBP) predictor.

Compares full and reduced models for structural outcomes and the primary
lRPF models (main and lncRNA populations, TPM >= 1). For OLS models,
semi_partial_R2 is full_R2 - reduced_R2, and partial_R2 divides this difference
by 1 - reduced_R2. These describe variance accounted for conditionally on
the included covariates, not a causal contribution.
For logistic models the script instead uses McFadden pseudo-R2 and changes
in that quantity; these must not be interpreted as explained variance.

Output: analysis/results/followup1_partial_r2.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
BIND = ROOT / "results" / "step1_binding.csv"
RPF = ROOT / "results" / "step2_rpf_counts.csv"
OUT = ROOT / "results" / "followup1_partial_r2.csv"

CELLS = ["K562", "HepG2"]
LEN_CAP = 100
COV_RHS = "l_aalen + transcript_gc + l1p_tpm"  # covariates only
FULL_RHS = "l1p_rbp + " + COV_RHS


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def fit_pair(d, outcome, kind="ols"):
    """Return (full_R2, reduced_R2)."""
    if kind == "ols":
        full = smf.ols(f"{outcome} ~ {FULL_RHS}", data=d).fit()
        red = smf.ols(f"{outcome} ~ {COV_RHS}", data=d).fit()
    else:
        full = smf.logit(f"{outcome} ~ {FULL_RHS}", data=d).fit(disp=0)
        red = smf.logit(f"{outcome} ~ {COV_RHS}", data=d).fit(disp=0)
    return float(full.rsquared if kind == "ols" else full.prsquared), \
           float(red.rsquared if kind == "ols" else red.prsquared)


def main():
    s0 = read_step0()
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]
    rpf = pd.read_csv(RPF)[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]]
    df = (s0.merge(expr, on="orf_name").merge(bind, on="orf_name")
                .merge(rpf, on="orf_name"))
    base = df[(df.source_sheet == "S2") & (df.aa_length <= LEN_CAP)].copy()
    base["foldable"] = (base["structure_group"] == "foldable").astype(int)
    base["llps_prone"] = (base["catgranule2_llps_score"] >= 0.5).astype(int)

    rows = []
    # ---------- Step 1 axis (a) ----------
    for cell in CELLS:
        d = base.copy()
        d = d.dropna(subset=[f"tpm_{cell}_polyA", f"{cell}_nrbp_tx",
                              "transcript_gc", "mean_plddt",
                              "catgranule2_llps_score"]).copy()
        d["l1p_rbp"] = np.log1p(d[f"{cell}_nrbp_tx"])
        d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
        d["l_aalen"] = np.log(d["aa_length"])
        for outcome, kind in [("mean_plddt", "ols"),
                              ("catgranule2_llps_score", "ols"),
                              ("foldable", "logit"),
                              ("llps_prone", "logit")]:
            try:
                full_r2, red_r2 = fit_pair(d, outcome, kind=kind)
                sp = full_r2 - red_r2
                p = sp / max(1 - red_r2, 1e-9)
                rows.append(dict(analysis="Step1_axis_a", cell=cell,
                                 outcome=outcome, kind=kind, n=len(d),
                                 full_R2=full_r2, reduced_R2=red_r2,
                                 semi_partial_R2=sp, partial_R2=p))
            except Exception as e:
                rows.append(dict(analysis="Step1_axis_a", cell=cell,
                                 outcome=outcome, kind=kind, n=len(d),
                                 full_R2=np.nan, reduced_R2=np.nan,
                                 semi_partial_R2=np.nan, partial_R2=np.nan,
                                 note=str(e)[:80]))

    # ---------- Step 2 lRPF, TPM >= 1 ----------
    for stratum_name, sub in [("main", base),
                              ("lncRNA", base[base.orf_biotype == "lncRNA"])]:
        for cell in CELLS:
            d = sub.dropna(subset=[f"tpm_{cell}_polyA", f"{cell}_nrbp_tx",
                                    f"{cell}_rpf_in", "transcript_gc"]).copy()
            d = d[d[f"tpm_{cell}_polyA"] >= 1.0].copy()
            if len(d) < 30:
                continue
            d["l1p_rbp"] = np.log1p(d[f"{cell}_nrbp_tx"])
            d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
            d["l_aalen"] = np.log(d["aa_length"])
            d["lRPF"] = np.log2(d[f"{cell}_rpf_in"] + 1)
            full_r2, red_r2 = fit_pair(d, "lRPF", kind="ols")
            sp = full_r2 - red_r2
            p = sp / max(1 - red_r2, 1e-9)
            rows.append(dict(analysis=f"Step2_lRPF_{stratum_name}_TPMge1",
                             cell=cell, outcome="lRPF", kind="ols", n=len(d),
                             full_R2=full_r2, reduced_R2=red_r2,
                             semi_partial_R2=sp, partial_R2=p))

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== partial / semi-partial R² for log1p(unique_RBP) ===")
    print(res.round({"full_R2": 4, "reduced_R2": 4,
                     "semi_partial_R2": 5, "partial_R2": 5}).to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()


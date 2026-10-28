"""ORF-type stratified regression — confirmed design.

Per orf_biotype, per cell, per outcome (pLDDT, catGRANULE):
  outcome ~ log1p(unique_RBP) + log(AA_len) + GC + log1p(TPM)   [OLS + HC3]

Two test groups:
  PRESPECIFIED (8 tests, Bonferroni alpha = 0.05/8 = 0.00625):
    lncRNA  — prior positive association
    dORF    — prior null association
  SENSITIVITY (8 tests, NO multiple correction):
    uORF    — prior null (sensitivity / robustness in 5'UTR context)
    uoORF   — prior null (uORF replication, overlapping 5'UTR+CDS)

Positive verdict per (orf_biotype, outcome): both cells same-sign
  AND (pre-reg) both p_bonf < 0.05  i.e. both p_raw < 0.00625
       (sensitivity) both p_raw < 0.05

The serialized label "pre-reg" is retained for schema compatibility; it
denotes the prespecified group, not external preregistration.

NA-TPM rows dropped per cell line (TPM undefined). NaN handling reported.
All numbers from CSVs directly; no hardcoding. Runs in .venv_ranges.

Source CSVs:
  step0_properties.csv  - outcomes, GC, biotype, length
  step1_expression.csv  - polyA TPM per cell line (NaN = quant absent)
  step1_binding.csv     - K562_nrbp_tx, HepG2_nrbp_tx
"""
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
BIND = ROOT / "results" / "step1_binding.csv"
OUT = ROOT / "results" / "step1e_type_regression_results.csv"

CELLS = ["K562", "HepG2"]
OUTCOMES = {"mean_plddt": "pLDDT", "catgranule2_llps_score": "catGRANULE"}

PRE_REG = [("lncRNA", "effect+"), ("dORF", "null")]
SENSITIVITY = [("uORF", "null"), ("uoORF", "null")]
LEN_CAP = 100
ALPHA_BONF = 0.05 / 8  # 0.00625


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def run(df_type, cell, outcome):
    """Return dict with regression of the l1p_rbp coefficient.

    df_type already filtered to a single orf_biotype within MAIN pop. NA-TPM
    rows are dropped here. Returns n_analysis, beta, ci, p_raw, R2.
    """
    tpm_col = f"tpm_{cell}_polyA"
    rbp_col = f"{cell}_nrbp_tx"
    need = ["aa_length", "transcript_gc", "mean_plddt",
            "catgranule2_llps_score", tpm_col, rbp_col]
    n_before = len(df_type)
    d = df_type.dropna(subset=need).copy()
    n_na = n_before - len(d)
    if len(d) < 20:
        return dict(n=n_before, n_analysis=len(d), n_dropped_na=n_na,
                    beta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_raw=np.nan,
                    R2=np.nan, note="insufficient n")
    d["l1p_rbp"] = np.log1p(d[rbp_col])
    d["l_aalen"] = np.log(d["aa_length"])
    d["gc"] = d["transcript_gc"]
    d["l1p_tpm"] = np.log1p(d[tpm_col])
    m = smf.ols(f"{outcome} ~ l1p_rbp + l_aalen + gc + l1p_tpm",
                data=d).fit(cov_type="HC3")
    beta = m.params["l1p_rbp"]
    lo, hi = m.conf_int().loc["l1p_rbp"]
    return dict(n=n_before, n_analysis=len(d), n_dropped_na=n_na,
                beta=beta, ci_lo=lo, ci_hi=hi, p_raw=m.pvalues["l1p_rbp"],
                R2=m.rsquared, note="")


def main():
    s0 = read_step0()
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]
    df = s0.merge(expr, on="orf_name").merge(bind, on="orf_name")
    main = df[(df.source_sheet == "S2") & (df.aa_length <= LEN_CAP)].copy()
    print(f"MAIN n={len(main)}\n")

    rows = []
    for (bt, prior), kind in [((bt, p), "pre-reg") for bt, p in PRE_REG] + \
                              [((bt, p), "sensitivity") for bt, p in SENSITIVITY]:
        sub = main[main["orf_biotype"] == bt]
        for cell, (outcome, olabel) in product(CELLS, OUTCOMES.items()):
            r = run(sub, cell, outcome)
            rows.append(dict(
                orf_biotype=bt, kind=kind, prior=prior,
                cell=cell, outcome=olabel,
                n=r["n"], n_analysis=r["n_analysis"], n_dropped_na=r["n_dropped_na"],
                beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                p_raw=r["p_raw"], R2=r["R2"], note=r["note"],
            ))
    res = pd.DataFrame(rows)

    # Bonferroni on pre-reg only (8 tests)
    is_pre = res["kind"] == "pre-reg"
    res["p_bonf_prereg"] = np.nan
    if is_pre.any():
        res.loc[is_pre, "p_bonf_prereg"] = multipletests(
            res.loc[is_pre, "p_raw"], method="bonferroni")[1]

    # replicate-pass flag per (orf_biotype, outcome): both cells same-sign and
    # both p significant at the appropriate alpha.
    res["replicate_pass"] = False
    for (bt, outcome, kind), g in res.groupby(["orf_biotype", "outcome", "kind"]):
        if g["cell"].nunique() < 2 or g["beta"].isna().any():
            continue
        same_sign = (np.sign(g["beta"]).nunique() == 1)
        if kind == "pre-reg":
            both_sig = (g["p_bonf_prereg"] < 0.05).all()  # = p_raw < 0.00625
        else:
            both_sig = (g["p_raw"] < 0.05).all()
        if same_sign and both_sig:
            res.loc[g.index, "replicate_pass"] = True

    res = res[["orf_biotype", "kind", "prior", "cell", "outcome",
               "n", "n_analysis", "n_dropped_na",
               "beta", "ci_lo", "ci_hi", "p_raw", "p_bonf_prereg",
               "R2", "replicate_pass", "note"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    # ===== printed report =====
    pd.set_option("display.width", 200, "display.max_columns", 25)
    print("================ Per-type regression results (16 rows) ================")
    print(res.round({"beta": 4, "ci_lo": 4, "ci_hi": 4,
                     "p_raw": 5, "p_bonf_prereg": 4, "R2": 3}).to_string(index=False))

    # lncRNA pre-reg verdict (4 tests)
    print("\n=== lncRNA (PRE-REG effect+) verdict ===")
    sub = res[(res.orf_biotype == "lncRNA")]
    for outcome in ["pLDDT", "catGRANULE"]:
        g = sub[sub.outcome == outcome]
        passed = bool(g["replicate_pass"].iloc[0])
        print(f"  lncRNA × {outcome:10s}  K562 β={g[g.cell=='K562'].beta.iloc[0]:+.4f} "
              f"(p={g[g.cell=='K562'].p_raw.iloc[0]:.3g})  "
              f"HepG2 β={g[g.cell=='HepG2'].beta.iloc[0]:+.4f} "
              f"(p={g[g.cell=='HepG2'].p_raw.iloc[0]:.3g})  "
              f"-> replicate_pass={passed}")

    # dORF pre-reg verdict (4 tests)
    print("\n=== dORF (PRE-REG null) verdict ===")
    sub = res[(res.orf_biotype == "dORF")]
    for outcome in ["pLDDT", "catGRANULE"]:
        g = sub[sub.outcome == outcome]
        passed = bool(g["replicate_pass"].iloc[0])
        print(f"  dORF   × {outcome:10s}  K562 β={g[g.cell=='K562'].beta.iloc[0]:+.4f} "
              f"(p={g[g.cell=='K562'].p_raw.iloc[0]:.3g})  "
              f"HepG2 β={g[g.cell=='HepG2'].beta.iloc[0]:+.4f} "
              f"(p={g[g.cell=='HepG2'].p_raw.iloc[0]:.3g})  "
              f"-> null_broken={passed}")

    # uORF / uoORF sensitivity
    print("\n=== uORF / uoORF (SENSITIVITY, no Bonferroni) ===")
    for bt in ["uORF", "uoORF"]:
        sub = res[res.orf_biotype == bt]
        for outcome in ["pLDDT", "catGRANULE"]:
            g = sub[sub.outcome == outcome]
            passed = bool(g["replicate_pass"].iloc[0])
            print(f"  {bt:6s} × {outcome:10s}  K562 β={g[g.cell=='K562'].beta.iloc[0]:+.4f} "
                  f"(p={g[g.cell=='K562'].p_raw.iloc[0]:.3g})  "
                  f"HepG2 β={g[g.cell=='HepG2'].beta.iloc[0]:+.4f} "
                  f"(p={g[g.cell=='HepG2'].p_raw.iloc[0]:.3g})  "
                  f"-> replicate_pass(raw)={passed}")

    # R² comparison across types
    print("\n=== R² by type/outcome (model explanatory power) ===")
    r2 = res.groupby(["orf_biotype", "outcome"])["R2"].mean().round(3)
    print(r2.to_string())

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()


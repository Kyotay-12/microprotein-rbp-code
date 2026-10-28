"""Step 1 main test (confirmed log1p-continuous model).

Merges step0 properties + expression (polyA TPM) + binding (unique RBP).
Per cell line (K562, HepG2), MAIN population (<=100aa & S2):

  predictors (all continuous): l1p_rbp = log1p(unique_RBP),
                               l_aalen = log(aa_length),
                               gc      = transcript_gc,
                               l1p_tpm = log1p(TPM)
  primary  : pLDDT ~ predictors            [OLS + HC3]
             catGRANULE ~ predictors       [OLS + HC3]
  secondary: foldable(0/1) ~ predictors    [logistic, OR+CI]
             LLPS_prone(0/1) ~ predictors  [logistic, OR+CI]

NA-TPM rows (status==NA) are dropped per cell line (TPM undefined) and counted.
TPM==0 kept (log1p(0)=0). No dummy fills.

Diagnostics emitted FIRST (read before the regressions):
  - cross-tab: binding(0 vs >0) x expression(TPM=0 vs >0)
  - VIF for the 4 predictors

Runs in .venv_ranges (pandas 2.x + statsmodels). Writes results/step1_maintest_results.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
BIND = ROOT / "results" / "step1_binding.csv"
OUT = ROOT / "results" / "step1_maintest_results.csv"
CELLS = ["K562", "HepG2"]
PREDICTORS = ["l1p_rbp", "l_aalen", "gc", "l1p_tpm"]


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def stratum(n):
    if n == 0: return "0"
    if n == 1: return "1"
    if n <= 4: return "2-4"
    if n <= 9: return "5-9"
    if n <= 19: return "10-19"
    return "20+"


def build():
    s0 = read_step0()
    expr = pd.read_csv(EXPR)
    bind = pd.read_csv(BIND)
    expr_cols = ["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA",
                 "status_K562_polyA", "status_HepG2_polyA"]
    bind_cols = ["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx",
                 "K562_npeaks_tx", "HepG2_npeaks_tx",
                 "K562_nrbp_orfspan", "HepG2_nrbp_orfspan"]
    df = s0.merge(expr[expr_cols], on="orf_name").merge(bind[bind_cols], on="orf_name")
    df["foldable"] = (df["structure_group"] == "foldable").astype(int)
    df["llps_prone"] = (df["catgranule2_llps_score"] >= 0.5).astype(int)
    df["l_aalen"] = np.log(df["aa_length"])
    df["gc"] = df["transcript_gc"]
    return df


def fit_ols(d, outcome):
    m = smf.ols(f"{outcome} ~ " + " + ".join(PREDICTORS), data=d).fit(cov_type="HC3")
    return m


def fit_logit(d, outcome):
    m = smf.logit(f"{outcome} ~ " + " + ".join(PREDICTORS), data=d).fit(disp=0)
    return m


def tidy(model, cell, outcome, kind):
    rows = []
    params = model.params
    cis = model.conf_int()
    for term in params.index:
        coef = params[term]
        lo, hi = cis.loc[term]
        p = model.pvalues[term]
        if kind == "logit":
            rows.append(dict(cell=cell, outcome=outcome, kind=kind, term=term,
                             OR=np.exp(coef), or_lo=np.exp(lo), or_hi=np.exp(hi), p=p))
        else:
            rows.append(dict(cell=cell, outcome=outcome, kind=kind, term=term,
                             coef=coef, ci_lo=lo, ci_hi=hi, p=p))
    return rows


def main():
    df = build()
    main_pop = df[(df.source_sheet == "S2") & (df.aa_length <= 100)].copy()
    all_rows = []

    for cell in CELLS:
        tpm_col = f"tpm_{cell}_polyA"
        st_col = f"status_{cell}_polyA"
        rbp_col = f"{cell}_nrbp_tx"
        d = main_pop.copy()
        n_before = len(d)
        # NA = TPM undefined (ENST absent from quant). pandas reads the literal
        # "NA" status string as NaN, so filter on the TPM column directly.
        n_na = int(d[tpm_col].isna().sum())
        d = d[d[tpm_col].notna()].copy()     # drop NA TPM (undefined); keep zero
        d["l1p_rbp"] = np.log1p(d[rbp_col])
        d["l1p_tpm"] = np.log1p(d[tpm_col])

        print(f"\n################  {cell}  ################", flush=True)
        print(f"analysis n = {len(d)} (main pop {n_before}; dropped NA-TPM: {n_na})", flush=True)

        # --- DIAGNOSTIC 1: cross-tab binding(0/>0) x expression(zero/>0) ---
        bound = np.where(d[rbp_col] > 0, ">0 RBP", "0 RBP")
        expr_s = np.where(d[tpm_col] > 0, "TPM>0", "TPM=0")
        ct = pd.crosstab(pd.Series(bound, name="binding"),
                         pd.Series(expr_s, name="expression"), margins=True)
        print("\n[diag] cross-tab binding x expression (analysis sample):", flush=True)
        print(ct.to_string(), flush=True)
        # of the 0-RBP ORFs, fraction that are TPM=0
        zero_rbp = d[d[rbp_col] == 0]
        if len(zero_rbp):
            frac0 = (zero_rbp[tpm_col] == 0).mean()
            print(f"  of 0-RBP ORFs (n={len(zero_rbp)}): {100*frac0:.1f}% are TPM=0 "
                  f"(not expressed -> trivially unbound)", flush=True)

        # --- DIAGNOSTIC 2: VIF ---
        X = sm.add_constant(d[PREDICTORS])
        print("\n[diag] VIF:", flush=True)
        for i, col in enumerate(X.columns):
            print(f"    {col:10s} {variance_inflation_factor(X.values, i):.2f}", flush=True)

        # --- PRIMARY: OLS + HC3 ---
        for outcome in ["mean_plddt", "catgranule2_llps_score"]:
            m = fit_ols(d, outcome)
            all_rows += tidy(m, cell, outcome, "ols")
            c = m.params["l1p_rbp"]; lo, hi = m.conf_int().loc["l1p_rbp"]; p = m.pvalues["l1p_rbp"]
            print(f"\n[OLS {outcome}]  R2={m.rsquared:.3f}  "
                  f"log1p_RBP: beta={c:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.2e}", flush=True)

        # --- SECONDARY: logistic ---
        for outcome in ["foldable", "llps_prone"]:
            try:
                m = fit_logit(d, outcome)
                all_rows += tidy(m, cell, outcome, "logit")
                orr = np.exp(m.params["l1p_rbp"])
                lo, hi = np.exp(m.conf_int().loc["l1p_rbp"])
                p = m.pvalues["l1p_rbp"]
                print(f"[LOGIT {outcome}]  n_events={int(d[outcome].sum())}  "
                      f"log1p_RBP: OR={orr:.3f} [{lo:.3f},{hi:.3f}] p={p:.2e}", flush=True)
            except Exception as e:
                print(f"[LOGIT {outcome}] FAILED: {e}", flush=True)

        # --- dose-response (viz): outcome means by ordered stratum ---
        d["stratum"] = d[rbp_col].apply(stratum)
        order = ["0", "1", "2-4", "5-9", "10-19", "20+"]
        g = d.groupby("stratum").agg(
            n=("orf_name", "size"),
            mean_plddt=("mean_plddt", "mean"),
            mean_catg=("catgranule2_llps_score", "mean"),
            frac_foldable=("foldable", "mean"),
            frac_llps=("llps_prone", "mean"),
        ).reindex(order)
        print("\n[dose-response by stratum]", flush=True)
        print(g.round(3).to_string(), flush=True)

    pd.DataFrame(all_rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

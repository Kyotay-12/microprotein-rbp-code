"""Step 2 Phase B — Translation-Efficiency (TE) regression.

TE = log2((in_frame_RPF + 1) / (TPM + 0.01)) per cell line. Per-stratum OLS+HC3:

    TE ~ log1p(unique_RBP) + log(AA_len) + GC + log1p(TPM)

Strata (each × K562 and HepG2 = 12 tests in total):
  - main           : full MAIN pop (S2 & aa_length<=100)
  - lncRNA_prereg  : lncRNA-derived ORFs subset, PRESPECIFIED, 2-sided
  - uORF / uoORF / dORF / intORF : sensitivity (descriptive, no correction)

Multiple comparison: Bonferroni applied INDEPENDENTLY within each prespecified
family (main: 2 tests; lncRNA_prereg: 2 tests), alpha = 0.05/2 = 0.025. These
two families are independent prespecified hypotheses and do not cross-correct
(same treatment as the Step 1 three-axis prospective specification). Sensitivity strata
report raw p only.

Positive verdict = both cells SAME-SIGN AND both p_bonf < 0.05 (= both p_raw <
0.025 within a 2-test Bonferroni family). For sensitivity strata we apply the
same same-sign + both p_raw<0.05 criterion descriptively (no positive claim).

NA-TPM rows dropped per cell. No dummy fills. All numbers from CSVs directly.
The legacy "lncRNA_prereg" label is retained for schema compatibility and
does not denote external preregistration.
Runs in .venv_ranges.

Source CSVs:
  step0_properties.csv  - source_sheet, aa_length, orf_biotype, GC, mean_plddt, catGRANULE
  step1_expression.csv  - polyA TPM per cell
  step1_binding.csv     - K562_nrbp_tx, HepG2_nrbp_tx
  step2_rpf_counts.csv  - in-frame P-site count per cell
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
BIND = ROOT / "results" / "step1_binding.csv"
RPF = ROOT / "results" / "step2_rpf_counts.csv"
OUT = ROOT / "results" / "step2_TE_results.csv"
CELLS = ["K562", "HepG2"]

# stratum_name -> (biotype filter or None, family for Bonferroni or None)
STRATA = [
    ("main",          None,     "main"),
    ("lncRNA_prereg", "lncRNA", "lncRNA_prereg"),
    ("uORF",          "uORF",   None),
    ("uoORF",         "uoORF",  None),
    ("dORF",          "dORF",   None),
    ("intORF",        "intORF", None),
]
ALPHA_BONF = 0.05 / 2  # 0.025 within each 2-test family


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def fit_one(d, cell):
    """Build TE and fit OLS+HC3. Returns dict of coefficient stats for l1p_rbp."""
    tpm_col = f"tpm_{cell}_polyA"
    rpf_col = f"{cell}_rpf_in"
    rbp_col = f"{cell}_nrbp_tx"
    n_before = len(d)
    d = d.dropna(subset=[tpm_col, rpf_col, rbp_col, "aa_length",
                         "transcript_gc"]).copy()
    n_na = n_before - len(d)
    if len(d) < 20:
        return dict(n=n_before, n_analysis=len(d), n_dropped_na=n_na,
                    beta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_raw=np.nan,
                    R2=np.nan, te_median=np.nan, te_mean=np.nan)
    d["TE"] = np.log2((d[rpf_col] + 1) / (d[tpm_col] + 0.01))
    d["l1p_rbp"] = np.log1p(d[rbp_col])
    d["l_aalen"] = np.log(d["aa_length"])
    d["gc"] = d["transcript_gc"]
    d["l1p_tpm"] = np.log1p(d[tpm_col])
    m = smf.ols("TE ~ l1p_rbp + l_aalen + gc + l1p_tpm", data=d).fit(cov_type="HC3")
    beta = m.params["l1p_rbp"]
    lo, hi = m.conf_int().loc["l1p_rbp"]
    return dict(n=n_before, n_analysis=len(d), n_dropped_na=n_na,
                beta=beta, ci_lo=lo, ci_hi=hi, p_raw=m.pvalues["l1p_rbp"],
                R2=m.rsquared, te_median=float(d["TE"].median()),
                te_mean=float(d["TE"].mean()))


def main():
    s0 = read_step0()
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]
    rpf = pd.read_csv(RPF)[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]]
    df = (s0.merge(expr, on="orf_name").merge(bind, on="orf_name")
                .merge(rpf, on="orf_name"))
    base = df[(df.source_sheet == "S2") & (df.aa_length <= 100)].copy()
    print(f"MAIN pop: n={len(base)}\n", flush=True)

    rows = []
    for stratum_name, biotype, family in STRATA:
        sub = base if biotype is None else base[base.orf_biotype == biotype]
        for cell in CELLS:
            r = fit_one(sub, cell)
            rows.append(dict(
                stratum=stratum_name, family=(family or "sensitivity"),
                biotype=(biotype or "ALL"), cell=cell,
                n=r["n"], n_analysis=r["n_analysis"], n_dropped_na=r["n_dropped_na"],
                beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                p_raw=r["p_raw"], R2=r["R2"],
                te_median=r["te_median"], te_mean=r["te_mean"],
            ))
    res = pd.DataFrame(rows)

    # Bonferroni within each prespecified family (2 tests)
    res["p_bonf"] = np.nan
    for fam in ["main", "lncRNA_prereg"]:
        m = (res["family"] == fam) & res["p_raw"].notna()
        if m.any():
            res.loc[m, "p_bonf"] = multipletests(res.loc[m, "p_raw"],
                                                 method="bonferroni")[1]

    # Replicate-pass per stratum (both cells, same sign + both p significant)
    res["replicate_pass"] = False
    for stratum_name, _, family in STRATA:
        g = res[res["stratum"] == stratum_name]
        if g["cell"].nunique() < 2 or g["beta"].isna().any():
            continue
        same_sign = (np.sign(g["beta"]).nunique() == 1)
        if family in ("main", "lncRNA_prereg"):
            both_sig = (g["p_bonf"] < 0.05).all()
        else:
            both_sig = (g["p_raw"] < 0.05).all()
        if same_sign and both_sig:
            res.loc[g.index, "replicate_pass"] = True

    res = res[["stratum", "family", "biotype", "cell",
               "n", "n_analysis", "n_dropped_na",
               "beta", "ci_lo", "ci_hi", "p_raw", "p_bonf",
               "R2", "te_median", "te_mean", "replicate_pass"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 240, "display.max_columns", 25)
    print("================ Step 2 TE regression — 12-row result table ================",
          flush=True)
    show_cols = ["stratum", "family", "cell", "n_analysis",
                 "beta", "ci_lo", "ci_hi", "p_raw", "p_bonf",
                 "R2", "te_median", "replicate_pass"]
    print(res[show_cols].round({"beta": 4, "ci_lo": 4, "ci_hi": 4,
                                "p_raw": 4, "p_bonf": 4, "R2": 3,
                                "te_median": 2}).to_string(index=False), flush=True)

    # Verdicts
    print("\n=== (1) MAIN test verdict (prespecified, Bonferroni alpha=0.025) ===",
          flush=True)
    m = res[res.stratum == "main"]
    passed = bool(m["replicate_pass"].iloc[0])
    print(f"  MAIN  n_analysis K562={int(m[m.cell=='K562'].n_analysis.iloc[0])} / "
          f"HepG2={int(m[m.cell=='HepG2'].n_analysis.iloc[0])}", flush=True)
    print(f"  K562:  β={m[m.cell=='K562'].beta.iloc[0]:+.4f} "
          f"CI=[{m[m.cell=='K562'].ci_lo.iloc[0]:+.4f}, {m[m.cell=='K562'].ci_hi.iloc[0]:+.4f}]  "
          f"p_raw={m[m.cell=='K562'].p_raw.iloc[0]:.3g}  p_bonf={m[m.cell=='K562'].p_bonf.iloc[0]:.3g}",
          flush=True)
    print(f"  HepG2: β={m[m.cell=='HepG2'].beta.iloc[0]:+.4f} "
          f"CI=[{m[m.cell=='HepG2'].ci_lo.iloc[0]:+.4f}, {m[m.cell=='HepG2'].ci_hi.iloc[0]:+.4f}]  "
          f"p_raw={m[m.cell=='HepG2'].p_raw.iloc[0]:.3g}  p_bonf={m[m.cell=='HepG2'].p_bonf.iloc[0]:.3g}",
          flush=True)
    print(f"  -> replicate_pass = {passed}", flush=True)

    print("\n=== (2) lncRNA prespecified verdict (Bonferroni alpha=0.025, 2-sided) ===",
          flush=True)
    l = res[res.stratum == "lncRNA_prereg"]
    passed_l = bool(l["replicate_pass"].iloc[0])
    print(f"  lncRNA  n_analysis K562={int(l[l.cell=='K562'].n_analysis.iloc[0])} / "
          f"HepG2={int(l[l.cell=='HepG2'].n_analysis.iloc[0])}", flush=True)
    print(f"  K562:  β={l[l.cell=='K562'].beta.iloc[0]:+.4f} "
          f"CI=[{l[l.cell=='K562'].ci_lo.iloc[0]:+.4f}, {l[l.cell=='K562'].ci_hi.iloc[0]:+.4f}]  "
          f"p_raw={l[l.cell=='K562'].p_raw.iloc[0]:.3g}  p_bonf={l[l.cell=='K562'].p_bonf.iloc[0]:.3g}",
          flush=True)
    print(f"  HepG2: β={l[l.cell=='HepG2'].beta.iloc[0]:+.4f} "
          f"CI=[{l[l.cell=='HepG2'].ci_lo.iloc[0]:+.4f}, {l[l.cell=='HepG2'].ci_hi.iloc[0]:+.4f}]  "
          f"p_raw={l[l.cell=='HepG2'].p_raw.iloc[0]:.3g}  p_bonf={l[l.cell=='HepG2'].p_bonf.iloc[0]:.3g}",
          flush=True)
    print(f"  -> replicate_pass = {passed_l}", flush=True)

    print("\n=== (3) R² comparison by stratum (outlier check) ===", flush=True)
    r2_summary = (res.groupby("stratum").agg(
        R2_K562=("R2", lambda s: s.iloc[0]),
        R2_HepG2=("R2", lambda s: s.iloc[-1])))
    print(r2_summary.round(3).to_string(), flush=True)

    print("\n=== (4) ORF type effect direction summary ===", flush=True)
    for st in ["main", "lncRNA_prereg", "uORF", "uoORF", "dORF", "intORF"]:
        g = res[res.stratum == st]
        if g["beta"].isna().any():
            continue
        b_k = g[g.cell == "K562"].beta.iloc[0]
        b_h = g[g.cell == "HepG2"].beta.iloc[0]
        p_k = g[g.cell == "K562"].p_raw.iloc[0]
        p_h = g[g.cell == "HepG2"].p_raw.iloc[0]
        sign_match = (np.sign(b_k) == np.sign(b_h))
        sig = ("**" if (p_k < 0.05 and p_h < 0.05) else
               (" *" if (p_k < 0.05 or p_h < 0.05) else "  "))
        print(f"  {st:14s}  K562 β={b_k:+.4f} (p={p_k:.3g})  "
              f"HepG2 β={b_h:+.4f} (p={p_h:.3g})  "
              f"same-sign={sign_match}  {sig}", flush=True)

    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()


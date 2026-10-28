"""TPM-threshold sensitivity analysis of the lRPF model.

Reports the RBP coefficient at TPM cutoffs 0, 0.5, 1, 2 and 5 for the main
population and lncRNA-derived subset, separately in each cell line.

Model: log2(in_frame_RPF + 1) ~ log1p(unique_RBP) + log(AA_len) + GC
                              + log1p(TPM), fitted by OLS with HC3 covariance.
The TPM >= 1 primary threshold was chosen after examining this sweep.
These overlapping subsets are sensitivity analyses, not independent tests.

Output: analysis/results/step2_tpm_sliding.csv.
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
OUT = ROOT / "results" / "step2_tpm_sliding.csv"

CUTS = [0.0, 0.5, 1.0, 2.0, 5.0]
CELLS = ["K562", "HepG2"]
LEN_CAP = 100


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
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]
    rpf = pd.read_csv(RPF)[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]]
    df = (s0.merge(expr, on="orf_name").merge(bind, on="orf_name")
                .merge(rpf, on="orf_name"))
    base = df[(df.source_sheet == "S2") & (df.aa_length <= LEN_CAP)].copy()
    print(f"main pop n={len(base)}")
    rows = []
    for cell in CELLS:
        for stratum_name, sub in [("main", base),
                                  ("lncRNA", base[base.orf_biotype == "lncRNA"])]:
            d_full = sub.dropna(subset=[f"tpm_{cell}_polyA", f"{cell}_nrbp_tx",
                                         f"{cell}_rpf_in", "transcript_gc"]).copy()
            d_full["l1p_rbp"] = np.log1p(d_full[f"{cell}_nrbp_tx"])
            d_full["l_aalen"] = np.log(d_full["aa_length"])
            d_full["gc"] = d_full["transcript_gc"]
            d_full["l1p_tpm"] = np.log1p(d_full[f"tpm_{cell}_polyA"])
            d_full["lRPF"] = np.log2(d_full[f"{cell}_rpf_in"] + 1)
            for cut in CUTS:
                flt = d_full[d_full[f"tpm_{cell}_polyA"] >= cut]
                if len(flt) < 30:
                    rows.append(dict(stratum=stratum_name, cell=cell,
                                     tpm_cutoff=cut, n=len(flt),
                                     beta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                     p_raw=np.nan, R2=np.nan,
                                     n_kept_pct=100 * len(flt) / len(d_full),
                                     note="insufficient n"))
                    continue
                m = smf.ols("lRPF ~ l1p_rbp + l_aalen + gc + l1p_tpm",
                            data=flt).fit(cov_type="HC3")
                beta = m.params["l1p_rbp"]
                lo, hi = m.conf_int().loc["l1p_rbp"]
                rows.append(dict(stratum=stratum_name, cell=cell,
                                 tpm_cutoff=cut, n=len(flt),
                                 beta=float(beta), ci_lo=float(lo),
                                 ci_hi=float(hi),
                                 p_raw=float(m.pvalues["l1p_rbp"]),
                                 R2=float(m.rsquared),
                                 n_kept_pct=100 * len(flt) / len(d_full),
                                 note=""))
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 25)
    print("\n=== TPM sliding sensitivity (lRPF model β for log1p(unique_RBP)) ===")
    for cell in CELLS:
        for stratum in ["main", "lncRNA"]:
            sub = res[(res.cell == cell) & (res.stratum == stratum)].sort_values("tpm_cutoff")
            print(f"\n--- {cell}  {stratum} ---")
            for _, r in sub.iterrows():
                sig = "**" if r["p_raw"] < 0.001 else ("*" if r["p_raw"] < 0.05 else "ns")
                print(f"  TPM>={r['tpm_cutoff']:<3}  n={int(r['n']):5d} "
                      f"({r['n_kept_pct']:.0f}% kept)  β={r['beta']:+.3f} "
                      f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p_raw']:.3g}  {sig}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()


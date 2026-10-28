"""Associations between host-transcript GC and predicted protein properties.

For each cell line, fit:
   pLDDT       ~ transcript_gc + log(AA_len) + log1p(TPM)        [OLS+HC3]
   catGRANULE  ~ transcript_gc + log(AA_len) + log1p(TPM)        [OLS+HC3]
   foldable    ~ transcript_gc + log(AA_len) + log1p(TPM)        [logistic]
   LLPS_prone  ~ transcript_gc + log(AA_len) + log1p(TPM)        [logistic]

The GC coefficient is per unit GC fraction; the script also reports the
coefficient scaled to a 0.1-unit GC-fraction difference. These are
observational associations, not a test of a causal GC effect.

Output: results/followup4_gc_vs_property.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
OUT = ROOT / "results" / "followup4_gc_vs_property.csv"

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
    df = s0.merge(expr, on="orf_name")
    df["foldable"] = (df["structure_group"] == "foldable").astype(int)
    df["llps_prone"] = (df["catgranule2_llps_score"] >= 0.5).astype(int)
    df["l_aalen"] = np.log(df["aa_length"])
    df["gc"] = df["transcript_gc"]
    base = df[(df.source_sheet == "S2") & (df.aa_length <= LEN_CAP)].copy()
    print(f"main pop n={len(base)}\n")

    rows = []
    for cell in CELLS:
        d = base.copy()
        d = d.dropna(subset=[f"tpm_{cell}_polyA", "gc"]).copy()
        d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
        for outcome, kind in [("mean_plddt", "ols"),
                              ("catgranule2_llps_score", "ols"),
                              ("foldable", "logit"),
                              ("llps_prone", "logit")]:
            try:
                if kind == "ols":
                    m = smf.ols(f"{outcome} ~ gc + l_aalen + l1p_tpm",
                                data=d).fit(cov_type="HC3")
                    beta = m.params["gc"]; lo, hi = m.conf_int().loc["gc"]
                    p = m.pvalues["gc"]; r2 = m.rsquared
                else:
                    m = smf.logit(f"{outcome} ~ gc + l_aalen + l1p_tpm",
                                  data=d).fit(disp=0)
                    beta = m.params["gc"]; lo, hi = m.conf_int().loc["gc"]
                    p = m.pvalues["gc"]; r2 = m.prsquared
                rows.append(dict(cell=cell, outcome=outcome, kind=kind, n=len(d),
                                 gc_beta=float(beta), ci_lo=float(lo),
                                 ci_hi=float(hi), p_raw=float(p),
                                 R2_or_pseudoR2=float(r2),
                                 # per 0.1 GC unit for interpretation
                                 effect_per_0p1_GC=float(beta) * 0.1,
                                 effect_per_0p1_GC_lo=float(lo) * 0.1,
                                 effect_per_0p1_GC_hi=float(hi) * 0.1))
            except Exception as e:
                rows.append(dict(cell=cell, outcome=outcome, kind=kind, n=len(d),
                                 gc_beta=np.nan, p_raw=np.nan, note=str(e)[:80]))

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== GC → structural property (Roginski self-verification) ===")
    cols = ["cell", "outcome", "kind", "n", "gc_beta",
            "ci_lo", "ci_hi", "p_raw", "R2_or_pseudoR2",
            "effect_per_0p1_GC"]
    print(res[cols].round({"gc_beta": 3, "ci_lo": 3, "ci_hi": 3,
                           "R2_or_pseudoR2": 3,
                           "effect_per_0p1_GC": 3,
                           "p_raw": 4}).to_string(index=False))

    # Cross-cell verdict: same sign + both p<0.05 (or both raw p<0.001 stricter)
    print("\n=== cross-cell replicate verdict (same-sign + both p<0.05) ===")
    for outcome in ["mean_plddt", "catgranule2_llps_score", "foldable", "llps_prone"]:
        g = res[res.outcome == outcome]
        if g["gc_beta"].isna().any() or g["cell"].nunique() < 2:
            continue
        same_sign = np.sign(g["gc_beta"]).nunique() == 1
        both_sig = (g["p_raw"] < 0.05).all()
        verdict = "REPLICATES" if (same_sign and both_sig) else "fails"
        print(f"  {outcome:30s}  K562 β={g[g.cell=='K562'].gc_beta.iloc[0]:+.3f} "
              f"(p={g[g.cell=='K562'].p_raw.iloc[0]:.3g})  "
              f"HepG2 β={g[g.cell=='HepG2'].gc_beta.iloc[0]:+.3f} "
              f"(p={g[g.cell=='HepG2'].p_raw.iloc[0]:.3g})  -> {verdict}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

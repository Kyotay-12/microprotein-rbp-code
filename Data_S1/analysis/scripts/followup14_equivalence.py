"""Equivalence testing of structural-property associations (TOST).

The specification is provided in equivalence_specification.txt.
Post hoc analysis: the ordinary regressions were already known. What was fixed
in advance are the bounds and the decision rules.

Model, per cell line, fully standardized within the analysis population:
    z(Y) ~ z(log1p(unique_RBP)) + z(log(AA length)) + z(GC) + z(log1p(TPM))
so the coefficient is SD of outcome per SD of RBP-binding burden.

Outcomes: mean pLDDT and catGRANULE score (continuous only; the binary outcomes
are excluded, see spec §2).

Bounds: primary SESOI +/- 0.10 SD, stringent sensitivity +/- 0.05 SD.
TOST at alpha = 0.05, i.e. equivalence iff the 90% CI lies inside the bound.
SE schemes: transcript-clustered (primary), gene-clustered, single-ORF
transcripts, HC3 (reference).

No dummy values; failures raise.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "followup14_equivalence.csv"
OUTCOMES = {"mean_plddt": "pLDDT", "catgranule2_llps_score": "catGRANULE"}
PRIMARY, STRINGENT, ALPHA = 0.10, 0.05, 0.05


def read_step0():
    nc = 0
    for line in open(RES / "step0_properties.csv"):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(RES / "step0_properties.csv", skiprows=nc)


def build():
    s0 = read_step0()
    cat = pd.read_excel(ROOT / "data" / "Mudge2022_SupplementaryTables.xlsx",
                        sheet_name="S2. PHASE I Ribo-seq ORFs",
                        usecols=["orf_name", "gene_id"])
    bind = pd.read_csv(RES / "step1_binding.csv")
    expr = pd.read_csv(RES / "step1_expression.csv")
    d = (s0.merge(cat, on="orf_name", how="left")
           .merge(bind[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]], on="orf_name")
           .merge(expr[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]], on="orf_name"))
    d = d[(d.source_sheet == "S2") & (d.aa_length <= 100)].copy()
    if d["gene_id"].isna().any():
        raise RuntimeError("ORFs without a stable ENSG")
    d["tx"] = d["transcript"].astype(str).str.split(".").str[0]
    d["gene"] = d["gene_id"].astype(str).str.split(".").str[0]
    return d


def z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("zero or non-finite variance in standardisation")
    return (v - v.mean()) / sd


def tost(beta, se, df, bound):
    """two one-sided tests; equivalence iff both p < alpha"""
    t_lo = (beta + bound) / se          # H0: beta <= -bound
    t_hi = (beta - bound) / se          # H0: beta >= +bound
    p_lo = stats.t.sf(t_lo, df)
    p_hi = stats.t.cdf(t_hi, df)
    return p_lo, p_hi, max(p_lo, p_hi)


def run(df, outcome_col, cell, scheme):
    d = df.dropna(subset=[outcome_col]).copy()
    d["zy"] = z(d[outcome_col])
    d["zrbp"] = z(np.log1p(d["nrbp"]))
    d["zlen"] = z(np.log(d["aa_length"]))
    d["zgc"] = z(d["transcript_gc"])
    d["ztpm"] = z(np.log1p(d["tpm"]))
    f = "zy ~ zrbp + zlen + zgc + ztpm"
    if scheme == "HC3":
        fit = smf.ols(f, data=d).fit(cov_type="HC3")
        dfree, ncl = len(d) - 5, np.nan
    else:
        g = d["tx"] if scheme in ("transcript", "single_ORF") else d["gene"]
        fit = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": g})
        ncl = g.nunique()
        dfree = ncl - 1
    beta = fit.params["zrbp"]
    se = fit.bse["zrbp"]
    ci90 = beta - stats.t.ppf(0.95, dfree) * se, beta + stats.t.ppf(0.95, dfree) * se
    ci95 = beta - stats.t.ppf(0.975, dfree) * se, beta + stats.t.ppf(0.975, dfree) * se
    p_lo, p_hi, p_tost = tost(beta, se, dfree, PRIMARY)
    _, _, p_tost_s = tost(beta, se, dfree, STRINGENT)
    diff_p = fit.pvalues["zrbp"]
    eq10 = (abs(ci90[0]) < PRIMARY) and (abs(ci90[1]) < PRIMARY)
    eq05 = (abs(ci90[0]) < STRINGENT) and (abs(ci90[1]) < STRINGENT)
    if diff_p >= 0.05 and eq10:
        verdict = "bounded near zero"
    elif diff_p >= 0.05 and not eq10:
        verdict = "inconclusive"
    elif diff_p < 0.05 and eq10:
        verdict = "detectable but negligible"
    else:
        verdict = "effect present"
    return dict(cell=cell, outcome=OUTCOMES[outcome_col], scheme=scheme,
                n=len(d), n_cluster=ncl, beta_std=beta, se=se,
                ci90_lo=ci90[0], ci90_hi=ci90[1],
                ci95_lo=ci95[0], ci95_hi=ci95[1],
                p_difference=diff_p, tost_p_lower=p_lo, tost_p_upper=p_hi,
                tost_p=p_tost, **{"equivalent_0.10": eq10},
                tost_p_stringent=p_tost_s, **{"equivalent_0.05": eq05},
                verdict=verdict)


def main():
    base = build()
    rows = []
    for cell in ["K562", "HepG2"]:
        w = base.copy()
        w["nrbp"] = w[f"{cell}_nrbp_tx"]
        w["tpm"] = w[f"tpm_{cell}_polyA"]
        w = w.dropna(subset=["tpm", "nrbp", "transcript_gc", "aa_length"])
        counts = w["tx"].value_counts()
        single = w[w["tx"].isin(counts[counts == 1].index)]
        for col in OUTCOMES:
            for scheme, frame in [("transcript", w), ("gene", w),
                                  ("single_ORF", single), ("HC3", w)]:
                rows.append(run(frame, col, cell, scheme))
    out = pd.DataFrame(rows)
    prim = out[out.scheme == "transcript"].copy()
    out["holm_p_primary"] = np.nan
    out.loc[prim.index, "holm_p_primary"] = multipletests(prim["tost_p"],
                                                          method="holm")[1]
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 240)
    cols = ["cell", "outcome", "scheme", "n", "n_cluster", "beta_std",
            "ci90_lo", "ci90_hi", "tost_p", "equivalent_0.10",
            "equivalent_0.05", "verdict"]
    for sch in ["transcript", "gene", "single_ORF", "HC3"]:
        print(f"\n--- {sch} ---")
        print(out[out.scheme == sch][cols].to_string(index=False))
    ok10 = out[(out.scheme == "transcript")]["equivalent_0.10"].all()
    ok05 = out[(out.scheme == "transcript")]["equivalent_0.05"].all()
    print(f"\nconjunction (all 4 primary models): ±0.10 {ok10} / ±0.05 {ok05}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

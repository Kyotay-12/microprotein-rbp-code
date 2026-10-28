"""Per-stratum lRPF regression with a shared abundance filter and model formula.

    log2(in_frame_RPF + 1) ~ log1p(unique_RBP) + log(AA_length) + GC + log1p(TPM)
    OLS, TPM >= 1, per cell line; HC3 and transcript-clustered SEs both reported.

No dummy values; failures raise.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "followup12_stratified_lrpf.csv"
STRATA = {"main": None, "lncRNA": "lncRNA", "uORF": "uORF",
          "uoORF": "uoORF", "dORF": "dORF", "intORF": "intORF"}


def read_step0():
    nc = 0
    for line in open(RES / "step0_properties.csv"):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(RES / "step0_properties.csv", skiprows=nc)


def main():
    s0 = read_step0()
    bind = pd.read_csv(RES / "step1_binding.csv")
    expr = pd.read_csv(RES / "step1_expression.csv")
    rpf = pd.read_csv(RES / "step2_rpf_counts.csv")
    d = (s0.merge(bind[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]], on="orf_name")
           .merge(expr[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]], on="orf_name")
           .merge(rpf[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]], on="orf_name"))
    d = d[(d.source_sheet == "S2") & (d.aa_length <= 100)].copy()
    d["tx"] = d["transcript"].astype(str).str.split(".").str[0]
    d["l_aalen"] = np.log(d["aa_length"])
    d["gc"] = d["transcript_gc"]

    rows = []
    for cell in ["K562", "HepG2"]:
        w = d.copy()
        w["nrbp"] = w[f"{cell}_nrbp_tx"]
        w["tpm"] = w[f"tpm_{cell}_polyA"]
        w["rpf"] = w[f"{cell}_rpf_in"]
        w = w.dropna(subset=["tpm", "nrbp", "gc", "aa_length", "rpf"])
        w = w[w["tpm"] >= 1].copy()
        w["y"] = np.log2(w["rpf"] + 1)
        w["l1p_rbp"] = np.log1p(w["nrbp"])
        w["l1p_tpm"] = np.log1p(w["tpm"])
        for name, biotype in STRATA.items():
            sub = w if biotype is None else w[w.orf_biotype == biotype]
            if len(sub) < 40:
                print(f"  skip {cell} {name}: n={len(sub)}", flush=True)
                continue
            f = smf.ols("y ~ l1p_rbp + l_aalen + gc + l1p_tpm",
                        data=sub).fit(cov_type="HC3")
            fc = smf.ols("y ~ l1p_rbp + l_aalen + gc + l1p_tpm", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["tx"]})
            lo, hi = f.conf_int().loc["l1p_rbp"]
            clo, chi = fc.conf_int().loc["l1p_rbp"]
            rows.append(dict(stratum=name, cell=cell, n=len(sub),
                             n_tx=sub.tx.nunique(),
                             beta=f.params["l1p_rbp"], ci_lo=lo, ci_hi=hi,
                             p=f.pvalues["l1p_rbp"],
                             ci_lo_clust=clo, ci_hi_clust=chi,
                             p_clust=fc.pvalues["l1p_rbp"], R2=f.rsquared))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(out[["stratum", "cell", "n", "beta", "ci_lo", "ci_hi", "p", "R2"]]
          .to_string(index=False))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()

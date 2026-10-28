"""Followup 11 — unit-of-analysis check: ORFs share host transcripts.

The predictors (unique RBP count, TPM) are properties of the host transcript,
while the regression unit is the ORF. ORFs sharing a host receive identical
host-level predictors. Clustered covariance accounts for within-host
dependence that is not represented by HC3 covariance alone.

Four standard-error/population schemes, reported side by side:
  (a) HC3                          unclustered reference
  (b) cluster-robust by transcript
  (c) cluster-robust by stable ENSG
  (d) one ORF per transcript        retain the longest analysed ORF, use HC3

Run for the two continuous structural-property outcomes (pLDDT and catGRANULE)
and the primary translation-output outcome (lRPF at TPM >= 1). The longest-ORF
selection is not the subset of transcripts carrying exactly one analysed ORF;
that subset is analysed separately in the permutation/equivalence scripts.

No dummy values; failures raise.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "followup11_cluster_robust.csv"


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
    rpf = pd.read_csv(RES / "step2_rpf_counts.csv")
    expected = s0.loc[(s0.source_sheet == "S2") & (s0.aa_length <= 100),
                      "orf_name"]
    if expected.empty or expected.isna().any() or not expected.is_unique:
        raise RuntimeError("Source-selected main-population ORF IDs must be nonempty and unique")
    d = (s0.merge(cat, on="orf_name", how="left")
           .merge(bind[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]], on="orf_name")
           .merge(expr[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]], on="orf_name")
           .merge(rpf[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]], on="orf_name"))
    d = d[(d.source_sheet == "S2") & (d.aa_length <= 100)].copy()
    if (d["orf_name"].isna().any() or not d["orf_name"].is_unique
            or set(d["orf_name"]) != set(expected)):
        raise RuntimeError("Merged main-population ORF IDs differ from the independently filtered source")
    if d["gene_id"].isna().any():
        raise RuntimeError(f"{int(d.gene_id.isna().sum())} ORFs lack a stable ENSG")
    # stable IDs, version-stripped — same blocking variable as followup13/14
    d["tx"] = d["transcript"].astype(str).str.split(".").str[0]
    d["gene"] = d["gene_id"].astype(str).str.split(".").str[0]
    d["foldable"] = (d["structure_group"] == "foldable").astype(int)
    d["l_aalen"] = np.log(d["aa_length"])
    d["gc"] = d["transcript_gc"]
    n_multi = (d.tx.value_counts() > 1).sum()
    print(f"main {len(d)} ORFs on {d.tx.nunique()} transcripts; "
          f"{n_multi} transcripts carry >1 ORF "
          f"({int(d.tx.value_counts()[lambda v: v > 1].sum())} ORFs)", flush=True)
    return d


def fits(df, formula, label, cell, outcome):
    """same model, four standard-error / population treatments"""
    rows = []
    base = smf.ols(formula, data=df)
    for how, kw in [("HC3", dict(cov_type="HC3")),
                    ("cluster_transcript",
                     dict(cov_type="cluster", cov_kwds={"groups": df["tx"]})),
                    ("cluster_gene",
                     dict(cov_type="cluster", cov_kwds={"groups": df["gene"]}))]:
        f = base.fit(**kw)
        lo, hi = f.conf_int().loc["l1p_rbp"]
        rows.append(dict(analysis=label, cell=cell, outcome=outcome, se=how,
                         n=len(df), n_cluster=(df["tx"].nunique() if "transcript" in how
                                               else df["gene"].nunique() if "gene" in how
                                               else np.nan),
                         beta=f.params["l1p_rbp"], ci_lo=lo, ci_hi=hi,
                         p=f.pvalues["l1p_rbp"]))
    # one ORF per transcript: keep the longest
    uniq = df.sort_values("aa_length", ascending=False).drop_duplicates("tx")
    f = smf.ols(formula, data=uniq).fit(cov_type="HC3")
    lo, hi = f.conf_int().loc["l1p_rbp"]
    rows.append(dict(analysis=label, cell=cell, outcome=outcome,
                     se="one_ORF_per_transcript", n=len(uniq), n_cluster=np.nan,
                     beta=f.params["l1p_rbp"], ci_lo=lo, ci_hi=hi,
                     p=f.pvalues["l1p_rbp"]))
    return rows


def main():
    d = build()
    rows = []
    for cell in ["K562", "HepG2"]:
        w = d.copy()
        w["nrbp"] = w[f"{cell}_nrbp_tx"]
        w["tpm"] = w[f"tpm_{cell}_polyA"]
        w["rpf"] = w[f"{cell}_rpf_in"]
        w = w.dropna(subset=["tpm", "nrbp", "gc", "aa_length"])
        w["l1p_rbp"] = np.log1p(w["nrbp"])
        w["l1p_tpm"] = np.log1p(w["tpm"])

        # ---- Step 1: structural-property outcomes ----
        for outcome, col in [("pLDDT", "mean_plddt"),
                             ("catGRANULE", "catgranule2_llps_score")]:
            sub = w.dropna(subset=[col]).copy()
            sub["y"] = sub[col]
            rows += fits(sub, "y ~ l1p_rbp + l_aalen + gc + l1p_tpm",
                         "step1_structural", cell, outcome)

        # ---- Step 2 headline: lRPF at TPM >= 1 (the positive) ----
        sub = w[w["tpm"] >= 1].dropna(subset=["rpf"]).copy()
        sub["y"] = np.log2(sub["rpf"] + 1)
        rows += fits(sub, "y ~ l1p_rbp + l_aalen + gc + l1p_tpm",
                     "step2_lrpf_tpm1", cell, "lRPF")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    for a in out.analysis.unique():
        print(f"\n--- {a} ---")
        print(out[out.analysis == a][["cell", "outcome", "se", "n", "n_cluster",
                                      "beta", "ci_lo", "ci_hi", "p"]]
              .to_string(index=False))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()

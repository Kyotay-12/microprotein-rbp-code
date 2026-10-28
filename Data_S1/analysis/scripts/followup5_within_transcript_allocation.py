"""Follow-up 5: within-transcript ncORF versus canonical-CDS translation.

Question
--------
Does transcript-level RBP binding predict allocation of ribosome footprints to
a non-canonical ORF after accounting for translation of the canonical CDS on
the *same ENST transcript*?

This is a fail-closed re-analysis of existing computed quantities.  It does
not substitute a canonical CDS from another isoform.  Only exact ENST matches
between the Mudge ncORF catalogue and the previously computed GENCODE v35
canonical-CDS table are retained.

Primary population
------------------
Replicated Mudge S2 ncORFs, <=100 aa, TPM >=1, exact ENST match, uORF only.
uORFs are primary because they do not overlap the canonical CDS.  Overlapping
uoORF/intORF/doORF classes would reuse some of the same P-sites in numerator
and denominator and are excluded.  A uORF+dORF sensitivity is also reported.

Models (per cell line; OLS with ENST-clustered covariance)
----------------------------------------------------------
1. allocation:
   log2(ncORF in-frame RPF + 1) ~ log1p(unique RBP)
       + log2(canonical-CDS in-frame RPF + 1)
       + log(ncORF aa length) + log(canonical CDS aa length)
       + transcript GC + log1p(TPM)

2. density ratio:
   log2(((ncORF RPF + 0.5) / ncORF aa length) /
        ((CDS RPF + 0.5) / CDS aa length))
       ~ log1p(unique RBP) + transcript GC + log1p(TPM)

The second model is a sensitivity specification; the pseudocount is explicit.
Results are not interpreted as causal because RBP detection remains
observational and transcript-abundance dependent.
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
CDS = ROOT / "results" / "followup2_canonical_cds_per_cds.csv"
OUT_DATA = ROOT / "results" / "followup5_within_transcript_per_orf.csv"
OUT_MODELS = ROOT / "results" / "followup5_within_transcript_models.csv"

CELLS = ("K562", "HepG2")
TPM_CUTOFF = 1.0


def read_step0():
    n_comments = 0
    with STEP0.open() as handle:
        for line in handle:
            if line.startswith("#"):
                n_comments += 1
            else:
                break
    return pd.read_csv(STEP0, skiprows=n_comments)


def build_exact_match_table():
    s0 = read_step0()
    expr = pd.read_csv(EXPR)[
        ["orf_name", "enst", "tpm_K562_polyA", "tpm_HepG2_polyA"]
    ]
    bind = pd.read_csv(BIND)[
        ["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]
    ]
    rpf = pd.read_csv(RPF)[
        ["orf_name", "K562_rpf_in", "HepG2_rpf_in"]
    ]
    cds = pd.read_csv(CDS)[
        ["enst", "cds_len", "aa_length", "K562_rpf_in", "HepG2_rpf_in"]
    ].rename(
        columns={
            "aa_length": "cds_aa_length",
            "K562_rpf_in": "K562_cds_rpf_in",
            "HepG2_rpf_in": "HepG2_cds_rpf_in",
        }
    )

    data = (
        s0.merge(expr, on="orf_name", validate="one_to_one")
        .merge(bind, on="orf_name", validate="one_to_one")
        .merge(rpf, on="orf_name", validate="one_to_one")
        .merge(cds, on="enst", validate="many_to_one")
    )
    data = data[(data["source_sheet"] == "S2") & (data["aa_length"] <= 100)].copy()
    if data.empty:
        raise RuntimeError("No exact ENST ncORF/canonical-CDS matches")
    if (data["cds_aa_length"] <= 0).any() or (data["aa_length"] <= 0).any():
        raise ValueError("Non-positive ORF length encountered")
    return data


def fit_clustered(formula, data, predictor):
    if data["enst"].nunique() < 30:
        raise ValueError("Too few ENST clusters for model")
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["enst"]}
    )
    lo, hi = model.conf_int().loc[predictor]
    return {
        "beta": float(model.params[predictor]),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_raw": float(model.pvalues[predictor]),
        "r2": float(model.rsquared),
    }


def main():
    data = build_exact_match_table()
    data.to_csv(OUT_DATA, index=False)
    print(
        f"Exact ENST matches: {len(data)} ORFs on {data.enst.nunique()} transcripts"
    )
    print(data["orf_biotype"].value_counts().to_string())

    strata = {
        "uORF_primary": {"uORF"},
        "uORF_dORF_sensitivity": {"uORF", "dORF"},
    }
    rows = []
    for cell in CELLS:
        for stratum, allowed_types in strata.items():
            d = data[data["orf_biotype"].isin(allowed_types)].copy()
            required = [
                f"tpm_{cell}_polyA",
                f"{cell}_nrbp_tx",
                f"{cell}_rpf_in",
                f"{cell}_cds_rpf_in",
                "transcript_gc",
                "aa_length",
                "cds_aa_length",
                "enst",
            ]
            d = d.dropna(subset=required)
            d = d[d[f"tpm_{cell}_polyA"] >= TPM_CUTOFF].copy()
            if len(d) < 100:
                raise ValueError(f"Insufficient observations: {cell} {stratum} n={len(d)}")

            d["l1p_rbp"] = np.log1p(d[f"{cell}_nrbp_tx"])
            d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
            d["l_nc_len"] = np.log(d["aa_length"])
            d["l_cds_len"] = np.log(d["cds_aa_length"])
            d["l_nc_rpf"] = np.log2(d[f"{cell}_rpf_in"] + 1.0)
            d["l_cds_rpf"] = np.log2(d[f"{cell}_cds_rpf_in"] + 1.0)
            d["log_density_ratio"] = np.log2(
                ((d[f"{cell}_rpf_in"] + 0.5) / d["aa_length"])
                / ((d[f"{cell}_cds_rpf_in"] + 0.5) / d["cds_aa_length"])
            )

            specifications = {
                "nc_rpf_unadjusted_for_cds": (
                    "l_nc_rpf ~ l1p_rbp + l_nc_len + transcript_gc + l1p_tpm"
                ),
                "canonical_cds_rpf": (
                    "l_cds_rpf ~ l1p_rbp + l_cds_len + transcript_gc + l1p_tpm"
                ),
                "allocation_cds_adjusted": (
                    "l_nc_rpf ~ l1p_rbp + l_cds_rpf + l_nc_len + "
                    "l_cds_len + transcript_gc + l1p_tpm"
                ),
                "density_ratio": (
                    "log_density_ratio ~ l1p_rbp + transcript_gc + l1p_tpm"
                ),
            }
            for specification, formula in specifications.items():
                result = fit_clustered(formula, d, "l1p_rbp")
                rows.append(
                    {
                        "cell": cell,
                        "stratum": stratum,
                        "specification": specification,
                        "tpm_cutoff": TPM_CUTOFF,
                        "n_orfs": len(d),
                        "n_enst": d["enst"].nunique(),
                        "median_nc_rpf": float(d[f"{cell}_rpf_in"].median()),
                        "median_cds_rpf": float(d[f"{cell}_cds_rpf_in"].median()),
                        **result,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUT_MODELS, index=False)
    pd.set_option("display.width", 180)
    print("\nWithin-transcript models (coefficient for log1p(unique RBP))")
    print(results.to_string(index=False))
    print(f"\nWrote {OUT_DATA}")
    print(f"Wrote {OUT_MODELS}")


if __name__ == "__main__":
    main()

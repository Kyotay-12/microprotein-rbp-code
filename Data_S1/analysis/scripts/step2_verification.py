"""Alternative translation-model specifications and historical permutation.

Evaluates length-normalized RPF ratios, removal of TPM from the ratio-model
covariates, a separate lRPF formulation, reading-frame filters, and transcript
length adjustment. The initial ratio-based TE_raw model is not the article's
primary lRPF model; interpret the model and population columns explicitly.

The predictor-shuffle permutation in this historical script uses 5,000
iterations and TPM >= 0. It has been superseded by the 20,000-iteration,
TPM >= 1, blockwise residual procedure in followup13_freedman_lane.py.
Neither procedure restores biological replicate-level uncertainty.

Inputs under analysis/results/: step0_properties.csv, step1_expression.csv,
step1_binding.csv, step2_rpf_counts.csv and v35_orf_exons.parquet.
P-site bigWigs are obtained separately under analysis/data/riboseq/.
Output: analysis/results/step2_verification.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pyBigWig

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
BIND = ROOT / "results" / "step1_binding.csv"
RPF = ROOT / "results" / "step2_rpf_counts.csv"
EXONS = ROOT / "results" / "v35_orf_exons.parquet"
RIBO_DIR = ROOT / "data" / "riboseq"
OUT = ROOT / "results" / "step2_verification.csv"

CELLS = ["K562", "HepG2"]
EPS_TPM = 0.01
EPS_RPF = 1.0
SEED = 12345
N_PERM = 5000


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def libsize_psites(cell):
    """Total in-genome P-site count from bigWig header.sumData (plus + minus)."""
    total = 0.0
    for strand in ("plus", "minus"):
        bw = pyBigWig.open(str(RIBO_DIR / f"{cell}_P_sites_{strand}.bw"))
        total += float(bw.header()["sumData"])
        bw.close()
    return total


def transcript_lengths():
    """versionless ENST -> total exon length (nt) from v35 exon parquet."""
    ex = pd.read_parquet(EXONS)
    ex["L"] = ex["End"] - ex["Start"]
    return ex.groupby("enst")["L"].sum().to_dict()


def fit_beta(d, formula, x="l1p_rbp"):
    m = smf.ols(formula, data=d).fit(cov_type="HC3")
    return dict(beta=m.params[x], ci_lo=m.conf_int().loc[x, 0],
                ci_hi=m.conf_int().loc[x, 1], p_raw=m.pvalues[x],
                R2=m.rsquared, n=int(m.nobs))


def permutation_p(d, formula, x="l1p_rbp", n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = fit_beta(d, formula, x)["beta"]
    d2 = d.copy()
    null_betas = np.empty(n_perm)
    for i in range(n_perm):
        d2[x] = rng.permutation(d[x].values)
        m = smf.ols(formula, data=d2).fit()  # plain OLS for speed, β unchanged
        null_betas[i] = m.params[x]
    p_emp = float((np.abs(null_betas) >= abs(obs)).mean())
    return dict(obs_beta=obs, null_mean=float(null_betas.mean()),
                null_std=float(null_betas.std()),
                null_max_abs=float(np.abs(null_betas).max()),
                p_emp=p_emp, n_perm=n_perm)


def main():
    print("Loading data ...", flush=True)
    s0 = read_step0()
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]
    rpf = pd.read_csv(RPF)[["orf_name", "K562_rpf_in", "HepG2_rpf_in",
                            "K562_rpf_off", "HepG2_rpf_off"]]
    df = (s0.merge(expr, on="orf_name").merge(bind, on="orf_name")
                .merge(rpf, on="orf_name"))
    df["enst"] = df["transcript"].str.split(".").str[0]
    tx_len = transcript_lengths()
    df["tx_len"] = df["enst"].map(tx_len)
    df["orf_nt"] = df["aa_length"] * 3  # ORF length in nt (excluding stop)
    base = df[(df.source_sheet == "S2") & (df.aa_length <= 100)].copy()
    print(f"  main pop n={len(base)}; transcript length mapped for "
          f"{int(base['tx_len'].notna().sum())}", flush=True)

    libsize = {c: libsize_psites(c) for c in CELLS}
    print(f"  libsize P-sites: {libsize}", flush=True)

    rows = []
    for cell in CELLS:
        d = base.copy()
        d = d.dropna(subset=[f"tpm_{cell}_polyA", f"{cell}_rpf_in",
                              f"{cell}_nrbp_tx", "aa_length", "transcript_gc",
                              "tx_len"]).copy()
        # core derived columns
        d["l1p_rbp"] = np.log1p(d[f"{cell}_nrbp_tx"])
        d["l_aalen"] = np.log(d["aa_length"])
        d["l_txlen"] = np.log(d["tx_len"])
        d["gc"] = d["transcript_gc"]
        d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
        d["lRPF"] = np.log2(d[f"{cell}_rpf_in"] + EPS_RPF)
        # TE variants
        d["TE_raw"] = np.log2((d[f"{cell}_rpf_in"] + EPS_RPF) /
                              (d[f"tpm_{cell}_polyA"] + EPS_TPM))
        d["TE_density"] = np.log2(
            (d[f"{cell}_rpf_in"] / d["orf_nt"] + 1e-6) /
            (d[f"tpm_{cell}_polyA"] + EPS_TPM))
        # RPF in TPM-equivalent: RPF_TPM = RPF / orf_nt * 1e9 / libsize_nt
        # libsize is in P-sites total (not nt total) — use as a scaling constant
        d["TE_bothtpm"] = np.log2(
            ((d[f"{cell}_rpf_in"] / d["orf_nt"]) * 1e9 / libsize[cell] + 1e-6) /
            (d[f"tpm_{cell}_polyA"] + EPS_TPM))

        # in-frame fraction
        tot = d[f"{cell}_rpf_in"] + d[f"{cell}_rpf_off"]
        d["if_frac"] = np.where(tot > 0,
                                d[f"{cell}_rpf_in"] / tot.replace(0, np.nan), np.nan)

        for stratum_label, mask in [("main", np.ones(len(d), dtype=bool)),
                                    ("lncRNA_prereg", d["orf_biotype"] == "lncRNA"),
                                    ("uoORF", d["orf_biotype"] == "uoORF")]:
            sub = d[mask].copy()
            n_full = len(sub)
            if n_full < 30:
                continue

            # --- V1: TE form variants (current model: + log1p(TPM) on RHS, log(AA), GC) ---
            for te_var in ["TE_raw", "TE_density", "TE_bothtpm"]:
                r = fit_beta(sub, f"{te_var} ~ l1p_rbp + l_aalen + gc + l1p_tpm")
                rows.append(dict(check="V1_TE_form", stratum=stratum_label,
                                 cell=cell, variant=te_var, n=r["n"],
                                 beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                 p_raw=r["p_raw"], R2=r["R2"]))

            # --- V2: drop l1p_tpm from RHS, also fit log(RPF) ~ ... + l1p_tpm ---
            r = fit_beta(sub, "TE_raw ~ l1p_rbp + l_aalen + gc")
            rows.append(dict(check="V2_no_tpm_RHS", stratum=stratum_label,
                             cell=cell, variant="TE_raw_noRHS_TPM", n=r["n"],
                             beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             p_raw=r["p_raw"], R2=r["R2"]))
            r = fit_beta(sub, "lRPF ~ l1p_rbp + l_aalen + gc + l1p_tpm")
            rows.append(dict(check="V2_lRPF_with_tpm", stratum=stratum_label,
                             cell=cell, variant="lRPF_RHS_TPM", n=r["n"],
                             beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             p_raw=r["p_raw"], R2=r["R2"]))

            # --- V5: add transcript length as covariate ---
            r = fit_beta(sub, "TE_raw ~ l1p_rbp + l_aalen + l_txlen + gc + l1p_tpm")
            rows.append(dict(check="V5_add_txlen", stratum=stratum_label,
                             cell=cell, variant="TE_raw_with_l_txlen", n=r["n"],
                             beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             p_raw=r["p_raw"], R2=r["R2"]))

            # --- V4: in-frame fraction filter (only for main and uoORF) ---
            if stratum_label in ("main", "uoORF"):
                for cut in [0.5, 0.6]:
                    flt = sub[sub["if_frac"].fillna(0) >= cut]
                    if len(flt) >= 30:
                        r = fit_beta(flt, "TE_raw ~ l1p_rbp + l_aalen + gc + l1p_tpm")
                        rows.append(dict(check=f"V4_ifrac_ge_{cut}",
                                         stratum=stratum_label, cell=cell,
                                         variant=f"in_frame_frac>={cut}",
                                         n=r["n"], beta=r["beta"], ci_lo=r["ci_lo"],
                                         ci_hi=r["ci_hi"], p_raw=r["p_raw"], R2=r["R2"]))

        # --- V3: permutation test on main (TE_raw, default model) ---
        d_main = d.copy()
        d_main["l_aalen_local"] = d_main["l_aalen"]
        formula = "TE_raw ~ l1p_rbp + l_aalen + gc + l1p_tpm"
        perm = permutation_p(d_main, formula)
        rows.append(dict(check="V3_perm_main", stratum="main", cell=cell,
                         variant=f"perm_n={perm['n_perm']}",
                         n=len(d_main), beta=perm["obs_beta"],
                         ci_lo=perm["null_mean"] - 1.96 * perm["null_std"],
                         ci_hi=perm["null_mean"] + 1.96 * perm["null_std"],
                         p_raw=perm["p_emp"], R2=np.nan))
        # permutation for lncRNA
        d_lnc = d[d["orf_biotype"] == "lncRNA"].copy()
        if len(d_lnc) >= 30:
            perm_l = permutation_p(d_lnc, formula)
            rows.append(dict(check="V3_perm_lncRNA", stratum="lncRNA_prereg",
                             cell=cell, variant=f"perm_n={perm_l['n_perm']}",
                             n=len(d_lnc), beta=perm_l["obs_beta"],
                             ci_lo=perm_l["null_mean"] - 1.96 * perm_l["null_std"],
                             ci_hi=perm_l["null_mean"] + 1.96 * perm_l["null_std"],
                             p_raw=perm_l["p_emp"], R2=np.nan))

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 220, "display.max_columns", 25)
    show = res.copy()
    show["beta"] = show["beta"].round(3)
    show["ci_lo"] = show["ci_lo"].round(3)
    show["ci_hi"] = show["ci_hi"].round(3)
    show["R2"] = show["R2"].round(3)
    show["p_raw"] = show["p_raw"].apply(lambda x: f"{x:.2g}" if pd.notna(x) else "n/a")
    print("\n=========== Verification results ===========")
    print(show[["check", "stratum", "cell", "variant", "n",
                "beta", "ci_lo", "ci_hi", "p_raw", "R2"]].to_string(index=False))

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()


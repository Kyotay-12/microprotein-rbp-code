"""Model diagnostics and sensitivity analyses.

V1. VIF of the lRPF model. Per cell, fit:
       log2(in_frame_RPF + 1) ~ log1p(unique_RBP) + log(AA_len) + GC + log1p(TPM)
    report VIF for each predictor (and the const). Also fit the V5 extended
    model adding log(tx_len) and report VIF for all 5 predictors. Flag VIF>5
    (warning) and VIF>10 (severe). If any predictor exceeds 5, drop it and
    refit, report the β shift for log1p(unique_RBP).

V2. lncRNA-ORF TPM distribution and low-TPM dependence. Per cell:
       describe TPM (median, mean, p10/25/50/75/90, range);
       count TPM=0 and TPM<1;
    refit lncRNA lRPF on filtered subsets TPM>=1 and TPM>=5 and report β.
    Same filters applied to main pop for parallel comparison.

V3. dORF ceiling-effect check. Per cell:
       describe unique_RBP distribution in dORF (median, mean, max, p10..90);
       describe TE outcome (log2(in_frame_RPF+1)) distribution including
         top-10% threshold;
       median split dORF by unique_RBP -> Mann-Whitney U on TE between low/high;
       quartile split dORF by unique_RBP -> median TE per Q, fraction in
         top-10%-TE band per Q.
    Same distribution stats reported (briefly) for main, lncRNA, uORF for
    context.

Output:
   results/step2_followup_validation.csv  (long-format: section, item, ...)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
BIND = ROOT / "results" / "step1_binding.csv"
RPF = ROOT / "results" / "step2_rpf_counts.csv"
EXONS = ROOT / "results" / "v35_orf_exons.parquet"
OUT = ROOT / "results" / "step2_followup_validation.csv"
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


def build():
    s0 = read_step0()
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    bind = pd.read_csv(BIND)[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]]
    rpf = pd.read_csv(RPF)[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]]
    df = (s0.merge(expr, on="orf_name").merge(bind, on="orf_name")
                .merge(rpf, on="orf_name"))
    df["enst"] = df["transcript"].str.split(".").str[0]
    ex = pd.read_parquet(EXONS)
    ex["L"] = ex["End"] - ex["Start"]
    tx_len = ex.groupby("enst")["L"].sum().to_dict()
    df["tx_len"] = df["enst"].map(tx_len)
    df["l_aalen"] = np.log(df["aa_length"])
    df["gc"] = df["transcript_gc"]
    base = df[(df.source_sheet == "S2") & (df.aa_length <= LEN_CAP)].copy()
    return base


def prepare(d, cell):
    d = d.dropna(subset=[f"tpm_{cell}_polyA", f"{cell}_nrbp_tx",
                          f"{cell}_rpf_in", "transcript_gc", "tx_len"]).copy()
    d["l1p_rbp"] = np.log1p(d[f"{cell}_nrbp_tx"])
    d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
    d["l_txlen"] = np.log(d["tx_len"])
    d["lRPF"] = np.log2(d[f"{cell}_rpf_in"] + 1)
    return d


def fit_lrpf(d, formula="lRPF ~ l1p_rbp + l_aalen + gc + l1p_tpm"):
    m = smf.ols(formula, data=d).fit(cov_type="HC3")
    beta = m.params.get("l1p_rbp", np.nan)
    if "l1p_rbp" in m.params.index:
        lo, hi = m.conf_int().loc["l1p_rbp"]
        p = m.pvalues["l1p_rbp"]
    else:
        lo = hi = p = np.nan
    return dict(beta=beta, ci_lo=lo, ci_hi=hi, p=p, R2=m.rsquared, n=int(m.nobs))


def vif_table(d, predictors):
    X = sm.add_constant(d[predictors])
    rows = []
    for i, c in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        rows.append((c, vif))
    return rows


def percentiles(s, ps=(10, 25, 50, 75, 90)):
    return {f"p{p}": float(np.percentile(s, p)) for p in ps}


def emit(section, **kw):
    return {"section": section, **kw}


def main():
    base = build()
    print(f"MAIN pop n={len(base)}\n")
    rows = []

    # =============================================================
    # V1 — VIF of lRPF model
    # =============================================================
    print("============== V1: VIF ==============")
    for cell in CELLS:
        d = prepare(base, cell)
        preds4 = ["l1p_rbp", "l_aalen", "gc", "l1p_tpm"]
        preds5 = preds4 + ["l_txlen"]
        # baseline 4-predictor model
        v4 = vif_table(d, preds4)
        print(f"\n{cell}  lRPF 4-predictor VIF (analysis n={len(d)}):")
        for name, val in v4:
            warn = "  WARN(>5)" if val > 5 and name != "const" else ""
            severe = "  SEVERE(>10)" if val > 10 and name != "const" else ""
            print(f"    {name:10s}  VIF={val:.2f}{warn}{severe}")
            rows.append(emit("V1_VIF_4pred", cell=cell, term=name, vif=val))
        # extended 5-predictor (V5 form)
        v5 = vif_table(d, preds5)
        print(f"\n{cell}  lRPF 5-predictor VIF (adds l_txlen):")
        for name, val in v5:
            warn = "  WARN(>5)" if val > 5 and name != "const" else ""
            severe = "  SEVERE(>10)" if val > 10 and name != "const" else ""
            print(f"    {name:10s}  VIF={val:.2f}{warn}{severe}")
            rows.append(emit("V1_VIF_5pred", cell=cell, term=name, vif=val))
        # if any non-const VIF > 5, drop it and refit; compare β
        high = [name for name, val in v4 if val > 5 and name != "const"]
        for name in high:
            keep = [p for p in preds4 if p != name]
            r = fit_lrpf(d, "lRPF ~ " + " + ".join(keep))
            print(f"  -> drop {name}: l1p_rbp β={r['beta']:+.3f} (vs baseline +0.81/+0.85)")
            rows.append(emit("V1_drop_high_VIF", cell=cell, dropped=name,
                             beta=r["beta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             p=r["p"], R2=r["R2"], n=r["n"]))
        if not high:
            rows.append(emit("V1_drop_high_VIF", cell=cell, dropped="(none>5)"))

    # =============================================================
    # V2 — lncRNA TPM distribution and low-TPM dependence
    # =============================================================
    print("\n============== V2: lncRNA TPM distribution + filters ==============")
    lnc = base[base.orf_biotype == "lncRNA"].copy()
    print(f"lncRNA n (pre-NA-drop) = {len(lnc)}")
    for cell in CELLS:
        d_lnc = prepare(lnc, cell)
        d_main = prepare(base, cell)
        tpm = d_lnc[f"tpm_{cell}_polyA"]
        stats = dict(
            n=len(d_lnc),
            mean=float(tpm.mean()), median=float(tpm.median()),
            min=float(tpm.min()), max=float(tpm.max()),
        )
        stats.update(percentiles(tpm))
        zero = int((tpm == 0).sum()); lt1 = int((tpm < 1).sum())
        stats.update(n_TPM_eq_0=zero, frac_TPM_eq_0=zero / len(d_lnc),
                     n_TPM_lt_1=lt1, frac_TPM_lt_1=lt1 / len(d_lnc))
        print(f"\n  {cell} lncRNA TPM: n={len(d_lnc)}  median={stats['median']:.3f}  "
              f"mean={stats['mean']:.3f}  range=[{stats['min']:.3f}, {stats['max']:.1f}]")
        print(f"    p10={stats['p10']:.3f}  p25={stats['p25']:.3f}  p50={stats['p50']:.3f}  "
              f"p75={stats['p75']:.3f}  p90={stats['p90']:.3f}")
        print(f"    TPM=0: {zero} ({100*zero/len(d_lnc):.1f}%)   "
              f"TPM<1: {lt1} ({100*lt1/len(d_lnc):.1f}%)")
        rows.append(emit("V2_lncRNA_TPM_dist", cell=cell, **stats))

        # Refits at different TPM cutoffs (lncRNA and main parallel)
        for cut in (0.0, 1.0, 5.0):
            for stratum_name, d_full in [("lncRNA", d_lnc), ("main", d_main)]:
                flt = d_full[d_full[f"tpm_{cell}_polyA"] >= cut]
                if len(flt) < 30:
                    continue
                r = fit_lrpf(flt)
                tag = "ALL" if cut == 0.0 else f">={cut}"
                rows.append(emit("V2_filter_refit", cell=cell, stratum=stratum_name,
                                 tpm_cutoff=cut, n=r["n"], beta=r["beta"],
                                 ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p=r["p"],
                                 R2=r["R2"]))
                print(f"    lRPF refit  {stratum_name:6s}  TPM{tag:6s}  n={r['n']:5d}  "
                      f"β={r['beta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p']:.3g}")

    # =============================================================
    # V3 — dORF ceiling-effect check
    # =============================================================
    print("\n============== V3: dORF ceiling-effect mechanism check ==============")
    dorf = base[base.orf_biotype == "dORF"].copy()
    main_all = base.copy()
    lnc_all = base[base.orf_biotype == "lncRNA"].copy()
    uorf_all = base[base.orf_biotype == "uORF"].copy()

    for cell in CELLS:
        d_dorf = prepare(dorf, cell)
        d_main_c = prepare(main_all, cell)
        d_lnc_c = prepare(lnc_all, cell)
        d_uorf_c = prepare(uorf_all, cell)

        # context: unique_RBP distribution per group (median/mean/max)
        print(f"\n  --- {cell}  unique_RBP distribution by stratum ---")
        for nm, dd in [("main", d_main_c), ("lncRNA", d_lnc_c),
                       ("uORF", d_uorf_c), ("dORF", d_dorf)]:
            s = dd[f"{cell}_nrbp_tx"]
            p_ = percentiles(s)
            print(f"    {nm:7s}  n={len(dd):5d}  med={int(s.median()):3d}  mean={s.mean():.1f}  "
                  f"max={int(s.max())}  p10={int(p_['p10']):3d}  p90={int(p_['p90']):3d}")
            rows.append(emit("V3_rbp_dist", cell=cell, stratum=nm, n=len(dd),
                             median=float(s.median()), mean=float(s.mean()),
                             max=float(s.max()), **p_))

        # dORF TE outcome distribution
        te = d_dorf["lRPF"]
        top10_thresh = float(d_main_c["lRPF"].quantile(0.90))   # top-10% defined on MAIN distribution
        n_dorf_in_top10 = int((te >= top10_thresh).sum())
        print(f"\n  dORF TE (lRPF) distribution: n={len(d_dorf)}  med={te.median():.2f}  "
              f"mean={te.mean():.2f}  range=[{te.min():.2f}, {te.max():.2f}]")
        print(f"    MAIN-pop top-10% TE threshold: lRPF >= {top10_thresh:.2f}")
        print(f"    dORF in top-10% of MAIN TE: {n_dorf_in_top10}/{len(d_dorf)} "
              f"({100*n_dorf_in_top10/len(d_dorf):.1f}%)")
        rows.append(emit("V3_dORF_TE_dist", cell=cell, n=len(d_dorf),
                          te_median=float(te.median()), te_mean=float(te.mean()),
                          te_min=float(te.min()), te_max=float(te.max()),
                          main_top10_thresh=top10_thresh,
                          dORF_in_main_top10_n=n_dorf_in_top10,
                          dORF_in_main_top10_frac=n_dorf_in_top10 / len(d_dorf)))

        # Median split by unique_RBP -> Mann-Whitney on TE
        med_rbp = d_dorf[f"{cell}_nrbp_tx"].median()
        low = d_dorf[d_dorf[f"{cell}_nrbp_tx"] <= med_rbp]
        high = d_dorf[d_dorf[f"{cell}_nrbp_tx"] > med_rbp]
        # ties at the median: ensure both groups non-empty; if all <= median, use < median
        if len(high) == 0:
            low = d_dorf[d_dorf[f"{cell}_nrbp_tx"] < med_rbp]
            high = d_dorf[d_dorf[f"{cell}_nrbp_tx"] >= med_rbp]
        if len(low) > 0 and len(high) > 0:
            u, p_mw = mannwhitneyu(low["lRPF"], high["lRPF"], alternative="two-sided")
            print(f"  median split (cut at unique_RBP={med_rbp:.0f}):  "
                  f"low (n={len(low)}, med TE={low['lRPF'].median():.2f})  "
                  f"vs high (n={len(high)}, med TE={high['lRPF'].median():.2f})  "
                  f"Mann-Whitney p={p_mw:.3g}")
            rows.append(emit("V3_dORF_median_split", cell=cell, cut=med_rbp,
                             n_low=len(low), te_med_low=float(low["lRPF"].median()),
                             n_high=len(high), te_med_high=float(high["lRPF"].median()),
                             mannwhitney_p=float(p_mw)))

        # Quartile split by unique_RBP
        try:
            q = pd.qcut(d_dorf[f"{cell}_nrbp_tx"], 4, duplicates="drop")
            grp = d_dorf.groupby(q, observed=True).agg(
                n=("orf_name", "size"),
                te_median=("lRPF", "median"),
                te_mean=("lRPF", "mean"),
                rbp_med=(f"{cell}_nrbp_tx", "median"),
            )
            # fraction in top-10% (MAIN reference) per quartile
            grp["frac_in_main_top10"] = d_dorf.groupby(q, observed=True).apply(
                lambda s: (s["lRPF"] >= top10_thresh).mean(),
                include_groups=False).values
            print(f"\n  dORF unique_RBP quartile split:")
            print(grp.round(3).to_string())
            for interval, r in grp.iterrows():
                rows.append(emit("V3_dORF_quartile", cell=cell, quartile=str(interval),
                                 n=int(r["n"]), te_median=float(r["te_median"]),
                                 te_mean=float(r["te_mean"]),
                                 rbp_med=float(r["rbp_med"]),
                                 frac_in_main_top10=float(r["frac_in_main_top10"])))
        except Exception as e:
            print(f"  quartile split failed: {e}")

    # write CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

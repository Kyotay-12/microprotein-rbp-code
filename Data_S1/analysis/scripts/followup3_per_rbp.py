"""Exploratory per-RBP associations with lRPF.

For each assayed RBP, construct a binary
"RBP-X binds the host transcript" variable, and fit:

  log2(in_frame_RPF + 1) ~ bound_X + log(AA_len) + GC + log1p(TPM)
                            [main pop, TPM >= 1, OLS+HC3]

This is an exploratory set of individual association models. Both nominal
and Bonferroni-corrected p-values are reported (alpha/n_tests within each
cell line). Van Nostrand functional-class annotations accompany each RBP.

Per RBP we compute: n_bound (in analysis sample), β, 95% CI, raw p,
Bonferroni-adjusted p (across RBPs within each cell).

Output: results/followup3_per_rbp.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
import openpyxl
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
STEP0 = ROOT / "results" / "step0_properties.csv"
EXPR = ROOT / "results" / "step1_expression.csv"
RPF = ROOT / "results" / "step2_rpf_counts.csv"
EXONS = ROOT / "results" / "v35_orf_exons.parquet"
VN_XLSX = ROOT / "data" / "VanNostrand2020_RBP_functions.xlsx"
ECLIP_DIR = ROOT / "data" / "eclip"
OUT = ROOT / "results" / "followup3_per_rbp.csv"

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


def load_vn_classes():
    wb = openpyxl.load_workbook(VN_XLSX, read_only=True, data_only=True)
    ws = wb["Sheet 1"]
    rows = list(ws.iter_rows(values_only=True))
    sub_header = rows[1]
    cat_to_col = {sub_header[i]: i for i in range(len(sub_header)) if sub_header[i]}
    rbp_classes = {}
    cat_keys = ["Translation regulation", "P-body / stress granules",
                "Splicing regulation", "3' end processing", "RNA modification",
                "RNA stability & decay"]
    for r in rows[2:]:
        if not r[0]:
            continue
        rbp = str(r[0]).strip().upper()
        cls_list = []
        for ck in cat_keys:
            if ck in cat_to_col and r[cat_to_col[ck]] in (1, "1", True):
                cls_list.append(ck)
        rbp_classes[rbp] = "/".join(cls_list) if cls_list else "(none)"
    return rbp_classes


def load_peaks_with_rbp(cell):
    rows = []
    for f in sorted((ECLIP_DIR / cell).glob("*.bed.gz")):
        rbp = f.name.split("__")[0].upper()
        d = pd.read_csv(f, sep="\t", header=None, compression="gzip",
                        usecols=[0, 1, 2, 5],
                        names=["Chromosome", "Start", "End", "Strand"],
                        dtype={0: str, 1: int, 2: int, 5: str})
        d["rbp"] = rbp
        rows.append(d)
    return pr.PyRanges(pd.concat(rows, ignore_index=True))


def main():
    print("Loading metadata ...", flush=True)
    s0 = read_step0()
    s0["enst"] = s0["transcript"].str.split(".").str[0]
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    rpf = pd.read_csv(RPF)[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]]
    df_base = s0.merge(expr, on="orf_name").merge(rpf, on="orf_name")
    df_base["l_aalen"] = np.log(df_base["aa_length"])
    df_base["gc"] = df_base["transcript_gc"]
    base = df_base[(df_base.source_sheet == "S2") & (df_base.aa_length <= LEN_CAP)].copy()
    print(f"  main pop n={len(base)}", flush=True)

    vn = load_vn_classes()
    exon_pr = pr.PyRanges(pd.read_parquet(EXONS))

    all_rows = []
    for cell in CELLS:
        print(f"\nProcessing {cell} ...", flush=True)
        peaks = load_peaks_with_rbp(cell)
        rbps_in_cell = sorted(set(peaks.df["rbp"].unique()))
        print(f"  {cell}: {len(rbps_in_cell)} unique RBPs", flush=True)
        # exon × peaks join (strand-matched)
        j = exon_pr.join(peaks, strandedness="same").df
        # per (enst, rbp) presence -> set
        enst_rbps = j.groupby("enst")["rbp"].agg(set).to_dict()
        # build per-ORF, per-RBP binary
        # restrict to analysis sample: TPM>=1 + non-NA needed cols
        tpm_col = f"tpm_{cell}_polyA"
        rpf_col = f"{cell}_rpf_in"
        d = base.dropna(subset=[tpm_col, rpf_col, "gc"]).copy()
        d = d[d[tpm_col] >= 1.0].copy()
        d["l1p_tpm"] = np.log1p(d[tpm_col])
        d["lRPF"] = np.log2(d[rpf_col] + 1)
        print(f"  analysis sample (TPM>=1) n={len(d)}", flush=True)
        # for each RBP, compute bound binary and fit
        rows = []
        n_total = len(rbps_in_cell)
        for i, rbp in enumerate(rbps_in_cell):
            d["bound"] = d["enst"].apply(
                lambda e: int(rbp in enst_rbps.get(e, set()))
            )
            n_bound = int(d["bound"].sum())
            if n_bound < 10 or n_bound > len(d) - 10:
                rows.append(dict(cell=cell, rbp=rbp,
                                 vn_class=vn.get(rbp, "(unknown)"),
                                 n_bound=n_bound, n_total=len(d),
                                 beta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                 p_raw=np.nan, note="too few bound/unbound"))
                continue
            try:
                m = smf.ols("lRPF ~ bound + l_aalen + gc + l1p_tpm",
                            data=d).fit(cov_type="HC3")
                beta = m.params["bound"]
                lo, hi = m.conf_int().loc["bound"]
                p = m.pvalues["bound"]
                rows.append(dict(cell=cell, rbp=rbp,
                                 vn_class=vn.get(rbp, "(unknown)"),
                                 n_bound=n_bound, n_total=len(d),
                                 beta=float(beta), ci_lo=float(lo),
                                 ci_hi=float(hi), p_raw=float(p), note=""))
            except Exception as e:
                rows.append(dict(cell=cell, rbp=rbp,
                                 vn_class=vn.get(rbp, "(unknown)"),
                                 n_bound=n_bound, n_total=len(d),
                                 beta=np.nan, p_raw=np.nan,
                                 note=str(e)[:80]))
        # Bonferroni within cell
        cell_df = pd.DataFrame(rows)
        valid = cell_df["p_raw"].notna()
        cell_df["p_bonf"] = np.nan
        if valid.any():
            cell_df.loc[valid, "p_bonf"] = multipletests(
                cell_df.loc[valid, "p_raw"], method="bonferroni")[1]
        all_rows.append(cell_df)

    res = pd.concat(all_rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    # Reports
    pd.set_option("display.width", 200, "display.max_columns", 25)
    print("\n=== per-cell summary ===")
    for cell in CELLS:
        sub = res[res.cell == cell].copy()
        sub = sub.dropna(subset=["beta"]).sort_values("beta", ascending=False)
        n_tested = len(sub)
        n_pos_05 = int(((sub["beta"] > 0) & (sub["p_raw"] < 0.05)).sum())
        n_neg_05 = int(((sub["beta"] < 0) & (sub["p_raw"] < 0.05)).sum())
        n_pos_bonf = int(((sub["beta"] > 0) & (sub["p_bonf"] < 0.05)).sum())
        n_neg_bonf = int(((sub["beta"] < 0) & (sub["p_bonf"] < 0.05)).sum())
        print(f"\n  {cell}: n_RBPs tested = {n_tested}")
        print(f"    raw p<0.05: positive β = {n_pos_05}, negative β = {n_neg_05}")
        print(f"    Bonferroni p<0.05: positive β = {n_pos_bonf}, "
              f"negative β = {n_neg_bonf}")
        print(f"    median β = {sub['beta'].median():+.3f}, "
              f"mean β = {sub['beta'].mean():+.3f}")
        # Top 10 positive contributors
        print(f"\n    TOP 10 positive contributors ({cell}):")
        for _, r in sub.head(10).iterrows():
            print(f"      {r['rbp']:8s}  β={r['beta']:+.3f} "
                  f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]  p={r['p_raw']:.2g}  "
                  f"p_bonf={r['p_bonf']:.2g}  n_bound={r['n_bound']:4d}  "
                  f"class={r['vn_class'][:40]}")
        # Bottom 5 (negative)
        print(f"    BOTTOM 5 (most negative β, {cell}):")
        for _, r in sub.tail(5).iterrows():
            print(f"      {r['rbp']:8s}  β={r['beta']:+.3f} "
                  f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]  p={r['p_raw']:.2g}  "
                  f"p_bonf={r['p_bonf']:.2g}  n_bound={r['n_bound']:4d}  "
                  f"class={r['vn_class'][:40]}")

    # Cross-cell consistency for the 76 common RBPs
    print("\n=== cross-cell consistency for common RBPs ===")
    common = (set(res[res.cell == "K562"].rbp) &
              set(res[res.cell == "HepG2"].rbp))
    print(f"  common RBPs: {len(common)}")
    same_sign_count = 0
    both_pos_sig = 0
    for rbp in common:
        k = res[(res.cell == "K562") & (res.rbp == rbp)].iloc[0]
        h = res[(res.cell == "HepG2") & (res.rbp == rbp)].iloc[0]
        if pd.isna(k["beta"]) or pd.isna(h["beta"]):
            continue
        if np.sign(k["beta"]) == np.sign(h["beta"]):
            same_sign_count += 1
            if k["beta"] > 0 and k["p_raw"] < 0.05 and h["p_raw"] < 0.05:
                both_pos_sig += 1
    print(f"  same-sign in both cells: {same_sign_count} / {len(common)} "
          f"({100*same_sign_count/len(common):.0f}%)")
    print(f"  both cells positive AND raw p<0.05: {both_pos_sig}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

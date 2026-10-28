"""Exploratory RBP-knockdown RNA-abundance analysis.

For each RBP–cell-line combination, compare mean RSEM transcript TPM in
knockdown and control, using log2[(KD + 1)/(control + 1)]. Calculate Spearman
correlations with transcript-averaged predicted microprotein properties,
separately for eCLIP-bound and unbound host transcripts.

The paired contrast is rho(bound) - rho(unbound), within the same RBP and
cell line. The pooled Wilcoxon test treats RBP–cell-line pairs as its units.
RBPs represented in both cell lines are not independent RBP identities;
the pooled test does not model this cross-cell dependence and is exploratory.
The legacy n_rbp_paired output column counts RBP–cell-line pairs.

The outcome is RNA abundance, not translation after RBP perturbation.
Failure to detect a difference is not evidence of equivalence.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
KD = ROOT / "data" / "kd"
RES = ROOT / "results"
QUANT = KD / "quant"
MAN = KD / "kd_manifest.tsv"
BIND = RES / "followup8_binding_matrix.parquet"
STEP0 = RES / "step0_properties.csv"
OUT_PER = RES / "followup8_kd_per_rbp.csv"
OUT_SUM = RES / "followup8_kd_summary.csv"
PROPS = {"mean_plddt": "pLDDT", "mean_idr": "IDR",
         "catgranule2_llps_score": "catGRANULE"}
MIN_BOUND = 30


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def main():
    man = pd.read_csv(MAN, sep="\t")
    have = {p.stem for p in QUANT.glob("*.tsv")}
    man = man[man.file.isin(have)]
    print(f"quant files available: {len(have)} / {pd.read_csv(MAN, sep='\t').file.nunique()}")

    # ENCODE serves BOTH an RSEM and a kallisto transcript quantification for
    # every replicate. Pin to RSEM: that is what scripts/step1_expression_report.py
    # uses, so these TPMs are on the same footing as the rest of the paper.
    tpm, fmt = {}, {}
    for f in sorted(have):
        d = pd.read_csv(QUANT / f"{f}.tsv", sep="\t")
        this = d["format"].iloc[0] if "format" in d.columns else "rsem"
        fmt[f] = this
        if this != "rsem":
            continue
        tpm[f] = d.set_index("enst")["TPM"]
    n_kal = sum(1 for v in fmt.values() if v != "rsem")
    print(f"quantifications: {len(tpm)} RSEM used, {n_kal} kallisto ignored")
    man = man[man.file.isin(tpm)]

    # step0_properties.csv already carries the host transcript id
    s0 = read_step0()
    if "transcript" not in s0.columns:
        raise RuntimeError("step0_properties.csv lacks 'transcript'")
    s0 = s0[(s0.source_sheet == "S2") & (s0.aa_length <= 100)].copy()
    s0["enst"] = s0["transcript"].astype(str).str.split(".").str[0]
    prop_cols = [c for c in PROPS if c in s0.columns]
    if len(prop_cols) < 3:
        raise RuntimeError(f"missing property columns; found {prop_cols}")
    tx_prop = s0.groupby("enst")[prop_cols].mean()
    print(f"transcripts with properties: {len(tx_prop)}")

    bm = pd.read_parquet(BIND)
    rows, dropped = [], []
    for (cell, rbp), g in man.groupby(["cell", "rbp"]):
        kdf = [f for f in g[g.role == "KD"].file if f in tpm]
        ctf = [f for f in g[g.role == "control"].file if f in tpm]
        if not kdf or not ctf:
            continue
        # TPM from different quantification pipelines is not comparable;
        # such a KD/control pair is dropped rather than silently mixed.
        fk, fc = {fmt[f] for f in kdf}, {fmt[f] for f in ctf}
        if len(fk | fc) > 1:
            dropped.append((cell, rbp, sorted(fk | fc)))
            continue
        kd = pd.concat([tpm[f] for f in kdf], axis=1).mean(axis=1)
        ct = pd.concat([tpm[f] for f in ctf], axis=1).mean(axis=1)
        idx = kd.index.intersection(ct.index)
        lfc = np.log2((kd[idx] + 1) / (ct[idx] + 1)).rename("lfc")
        bound = set(bm[(bm.cell == cell) & (bm.rbp == rbp)]["enst"])
        df = pd.concat([lfc, tx_prop], axis=1, join="inner").dropna()
        df["bound"] = df.index.isin(bound)
        for grp, sub in [("bound", df[df.bound]), ("unbound", df[~df.bound])]:
            if len(sub) < MIN_BOUND:
                continue
            for pc in prop_cols:
                rho, p = spearmanr(sub["lfc"], sub[pc])
                rows.append(dict(cell=cell, rbp=rbp, group=grp, prop=PROPS[pc],
                                 n=len(sub), rho=rho, p=p,
                                 n_kd_files=len(kdf), n_ctrl_files=len(ctf),
                                 frac_big_lfc=float((sub["lfc"].abs() > 1).mean())))
    if dropped:
        print(f"dropped {len(dropped)} RBPs for mixed quantification pipeline: "
              f"{dropped[:5]}{' ...' if len(dropped) > 5 else ''}")
    per = pd.DataFrame(rows)
    if per.empty:
        raise RuntimeError("no RBP produced a testable comparison")
    per["p_BH"] = np.nan
    for k, g in per.groupby(["group", "prop"]):
        per.loc[g.index, "p_BH"] = multipletests(g["p"], method="fdr_bh")[1]
    per.to_csv(OUT_PER, index=False)

    out = []
    for (grp, prop), g in per.groupby(["group", "prop"]):
        w = wilcoxon(g["rho"]) if len(g) > 10 else None
        out.append(dict(group=grp, prop=prop, n_rbp=len(g),
                        median_rho=g["rho"].median(), mean_rho=g["rho"].mean(),
                        frac_positive=float((g["rho"] > 0).mean()),
                        wilcoxon_p=(w.pvalue if w else np.nan),
                        n_sig_BH=int((g["p_BH"] < 0.05).sum())))
    summ = pd.DataFrame(out).sort_values(["prop", "group"])

    # Decisive statistic: PAIRED within RBP. Comparing p-values between the
    # bound and unbound groups is invalid (different n, different RBP sets).
    # For each RBP that yields both, delta = rho_bound - rho_unbound; if
    # binding carries information delta should be systematically non-zero.
    piv = per.pivot_table(index=["cell", "rbp", "prop"], columns="group",
                          values="rho")
    piv = piv.dropna(subset=["bound", "unbound"])
    piv["delta"] = piv["bound"] - piv["unbound"]
    paired = []
    for prop, g in piv.groupby("prop"):
        w = wilcoxon(g["delta"]) if len(g) > 10 else None
        paired.append(dict(prop=prop, n_rbp_paired=len(g),
                           median_delta=g["delta"].median(),
                           mean_delta=g["delta"].mean(),
                           frac_bound_stronger=float(
                               (g["bound"].abs() > g["unbound"].abs()).mean()),
                           wilcoxon_p=(w.pvalue if w else np.nan)))
    paired = pd.DataFrame(paired)
    paired.to_csv(RES / "followup8_kd_paired.csv", index=False)
    summ.to_csv(OUT_SUM, index=False)
    pd.set_option("display.width", 200)
    print("\n=== per-RBP Spearman(log2FC, property): aggregate ===")
    print(summ.to_string(index=False))
    print("\n=== PAIRED within RBP: rho(bound) - rho(unbound) ===")
    print(paired.to_string(index=False))
    print("\n(frac_bound_stronger = fraction of RBPs where |rho| is larger in the")
    print(" bound set. 0.5 means binding adds nothing.)")
    print(f"\nQC: median fraction of transcripts with |log2FC|>1 = "
          f"{per['frac_big_lfc'].median():.3f}")
    print(f"written: {OUT_PER}\n         {OUT_SUM}")


if __name__ == "__main__":
    main()


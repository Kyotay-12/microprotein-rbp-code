"""Compare per-RBP association estimates across expansion-status groups.

Expansion categories and family annotations are taken from Yasuda (2026).

Source files (in data/yasuda2026_rbp_evo/):
  family_level_correlations_42family_categorized.csv — expansion categories
  expanded_13species_rbp.csv           — per-RBP family annotations
  Homo_sapiens-RBPs.csv                — full human RBP list

For each eCLIP RBP we then test:
  Step 1 (structural property): is the per-RBP β on pLDDT / catGRANULE
    systematically different for expanded vs non-expanded RBPs?
  Step 2 (ribosome occupancy): is the per-RBP β on lRPF systematically
    different for expanded vs non-expanded RBPs?

These are exploratory group comparisons; nonsignificance does not establish
equivalence or refute a biological role for expanded families.

Output: results/followup5_expanded_rbps.csv (per-RBP expansion status + β)
        printed: distribution comparisons (Mann-Whitney + Fisher tests)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pyranges as pr
import openpyxl
from scipy.stats import mannwhitneyu, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EV = ROOT / "data" / "yasuda2026_rbp_evo"
STEP0 = RES / "step0_properties.csv"
EXPR = RES / "step1_expression.csv"
BIND = RES / "step1_binding.csv"
RPF = RES / "step2_rpf_counts.csv"
EXONS = RES / "v35_orf_exons.parquet"
ECLIP_DIR = ROOT / "data" / "eclip"
PER_RBP = RES / "followup3_per_rbp.csv"
OUT = RES / "followup5_expanded_rbps.csv"

CELLS = ["K562", "HepG2"]
LEN_CAP = 100

# Vertebrate-expanded families from the categorized table in Yasuda (2026).
# Source: data/yasuda2026_rbp_evo/family_level_correlations_42family_categorized.csv
# Select rows where category == "Vertebrate-expanded".
EXPANDED_FAMILIES_FILE = (
    Path(__file__).resolve().parent.parent
    / "data" / "yasuda2026_rbp_evo"
    / "family_level_correlations_42family_categorized.csv"
)
_fam_df = pd.read_csv(EXPANDED_FAMILIES_FILE)
EXPANDED_FAMILIES = set(_fam_df[_fam_df["category"] == "Vertebrate-expanded"]["family"].tolist())
print(f"[Followup 5] {len(EXPANDED_FAMILIES)} Vertebrate-expanded families loaded:",
      sorted(EXPANDED_FAMILIES))


def read_step0():
    nc = 0
    for line in open(STEP0):
        if line.startswith("#"):
            nc += 1
        else:
            break
    return pd.read_csv(STEP0, skiprows=nc)


def build_rbp_family_map():
    """Map RBP gene symbol -> family (from Yasuda 2026 human RBPs)."""
    df = pd.read_csv(EV / "expanded_13species_rbp.csv")
    human = df[df["common_name"] == "human"].copy()
    # `protein_name` is uniprot ID like RBM47_HUMAN; extract gene symbol
    human["gene_symbol"] = human["protein_name"].str.split("_").str[0]
    # `family` column has family classification
    # also some have explicit gene_name
    family_map = {}
    for _, r in human.iterrows():
        gene = r["gene_symbol"].upper()
        fam = r["family"] if pd.notna(r["family"]) else "Other"
        if not fam or fam == "":
            fam = "Other"
        family_map[gene] = fam
        # also add explicit gene_name if present
        if pd.notna(r.get("gene_name")) and r["gene_name"]:
            family_map[r["gene_name"].upper()] = fam
    return family_map


def expand_eclip_gene_names_to_paralog_family(eclip_rbps, family_map):
    """Best-effort: match ENCODE eCLIP RBP gene symbols to family annotations.

    Two-stage:
      1. Direct lookup in Yasuda 2026 family_map (built from human RBPs in
         expanded_13species_rbp.csv).
      2. Prefix/pattern heuristic for canonical Vertebrate-expanded family
         members in the categorized source table.
    """
    out = {}
    for rbp in eclip_rbps:
        rbp_u = rbp.upper()
        if rbp_u in family_map:
            out[rbp] = family_map[rbp_u]
            continue
        fam = "Other"
        # canonical Vertebrate-expanded families (Yasuda 2026)
        if rbp_u.startswith("HNRNPK") or rbp_u == "HNRNPK":
            fam = "HNRNPK"  # paralog group separate from main hnRNP
        elif rbp_u.startswith("HNRNP") or rbp_u.startswith("HNRPDL"):
            fam = "hnRNP"
        elif rbp_u.startswith("CELF"):
            fam = "CELF"
        elif rbp_u.startswith("CPEB"):
            fam = "CPEB"
        elif rbp_u.startswith("IGF2BP") or rbp_u.startswith("IMP"):
            fam = "IGF2BP"
        elif rbp_u.startswith("ELAVL") or rbp_u in {"TIAL1", "TIA1"}:
            fam = "ELAV/Hu"
        elif rbp_u.startswith("RBFOX"):
            fam = "RBFOX"
        elif rbp_u.startswith("PABP"):
            fam = "PABP"
        # Additional families represented in the categorized source table
        elif rbp_u.startswith("EXOSC"):
            fam = "Exosome"          # RNase complex, vertebrate-expanded
        elif rbp_u.startswith("DIS3"):
            fam = "DIS3"             # RNase, vertebrate-expanded
        elif rbp_u.startswith("RBM") and rbp_u != "RBM3":
            # Includes RBM5/15/22/47 etc. RBM3 is unrelated cold-shock so excluded conservatively.
            fam = "RBM"
        elif rbp_u.startswith("PTBP") or rbp_u.startswith("PCBP"):
            fam = "PTBP"
        elif rbp_u.startswith("STAU"):
            fam = "Staufen"
        elif rbp_u.startswith("FMR") or rbp_u.startswith("FXR"):
            fam = "FMR"
        elif rbp_u.startswith("LARP"):
            fam = "LARP"
        elif rbp_u.startswith("DHX"):
            fam = "DHX"
        elif rbp_u.startswith("ZC3H"):
            fam = "ZC3H"
        elif rbp_u.startswith("UPF"):
            fam = "UPF"
        elif rbp_u.startswith("ADAR"):
            fam = "ADAR"
        elif rbp_u.startswith("YTHDF") or rbp_u.startswith("YTHDC"):
            fam = "YTHD"
        out[rbp] = fam
    return out


def main():
    print("Loading Yasuda 2026 expanded RBP family map ...", flush=True)
    family_map = build_rbp_family_map()
    print(f"  family map: {len(family_map)} entries", flush=True)

    # Per-RBP results from Followup 3
    pr_df = pd.read_csv(PER_RBP)
    eclip_rbps = sorted(set(pr_df["rbp"].unique()))
    print(f"  eCLIP RBPs: {len(eclip_rbps)} unique across cells", flush=True)

    # Classify each eCLIP RBP by expanded family membership
    rbp_to_family = expand_eclip_gene_names_to_paralog_family(eclip_rbps, family_map)
    fam_counts = pd.Series(rbp_to_family).value_counts()
    print("\n  eCLIP RBP family distribution:", flush=True)
    for fam, n in fam_counts.items():
        flag = "EXPANDED" if fam in EXPANDED_FAMILIES else ""
        print(f"    {fam:20s} {n:3d}  {flag}", flush=True)

    expanded_rbps = {r for r, f in rbp_to_family.items() if f in EXPANDED_FAMILIES}
    nonexpanded_rbps = {r for r, f in rbp_to_family.items() if f not in EXPANDED_FAMILIES}
    print(f"\n  Total expanded-family eCLIP RBPs:    {len(expanded_rbps)}", flush=True)
    print(f"  Total non-expanded eCLIP RBPs:       {len(nonexpanded_rbps)}", flush=True)

    # --- attach expansion status to per-RBP β table ---
    pr_df["family"] = pr_df["rbp"].map(rbp_to_family)
    pr_df["expanded"] = pr_df["family"].isin(EXPANDED_FAMILIES)

    pr_df_save = pr_df[["rbp", "cell", "family", "expanded",
                        "n_bound", "beta", "ci_lo", "ci_hi", "p_raw", "p_bonf"]].copy()
    pr_df_save.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}", flush=True)

    # --- Sub-analysis: list per-RBP β for each expanded family in eCLIP ---
    print("\n  Per-expanded-family eCLIP coverage and TE β:", flush=True)
    fam_summary = pr_df[pr_df.expanded].copy()
    for fam in sorted(EXPANDED_FAMILIES & set(rbp_to_family.values())):
        for cell in CELLS:
            sub = fam_summary[(fam_summary.family == fam) & (fam_summary.cell == cell)]
            sub = sub.dropna(subset=["beta"])
            if len(sub) == 0:
                continue
            members = "  ".join(f"{r.rbp}(β={r.beta:+.2f})" for _, r in sub.iterrows())
            print(f"    {fam:12s} {cell:6s} n={len(sub):2d}  {members}", flush=True)

    # --- Step 2 TE comparison: expanded vs non-expanded β distributions ---
    print("\n" + "="*70, flush=True)
    print("Step 2 TE β: expanded vs non-expanded RBP families", flush=True)
    print("="*70, flush=True)
    for cell in CELLS:
        sub = pr_df[pr_df.cell == cell].dropna(subset=["beta"]).copy()
        exp = sub[sub.expanded]["beta"].values
        nox = sub[~sub.expanded]["beta"].values
        if len(exp) < 5 or len(nox) < 5:
            print(f"\n  {cell}: too few in one group (exp n={len(exp)}, nox n={len(nox)})")
            continue
        u, p_mw = mannwhitneyu(exp, nox, alternative="two-sided")
        print(f"\n  {cell}  (TE β, lRPF):", flush=True)
        print(f"    expanded RBPs (n={len(exp)}):     median β = {np.median(exp):+.3f}  "
              f"mean = {np.mean(exp):+.3f}", flush=True)
        print(f"    non-expanded RBPs (n={len(nox)}): median β = {np.median(nox):+.3f}  "
              f"mean = {np.mean(nox):+.3f}", flush=True)
        print(f"    Mann-Whitney p (two-sided) = {p_mw:.3g}", flush=True)
        # Fisher: significant positive count
        e_mask = sub.expanded
        n_exp_sig = int(((sub.loc[e_mask, "beta"] > 0) & (sub.loc[e_mask, "p_raw"] < 0.05)).sum())
        n_exp_tot = len(exp)
        n_nox_sig = int(((sub.loc[~e_mask, "beta"] > 0) & (sub.loc[~e_mask, "p_raw"] < 0.05)).sum())
        n_nox_tot = len(nox)
        try:
            tab = [[n_exp_sig, n_exp_tot - n_exp_sig],
                   [n_nox_sig, n_nox_tot - n_nox_sig]]
            odds, p_f = fisher_exact(tab, alternative="two-sided")
            print(f"    raw-p<0.05 positive fraction:", flush=True)
            print(f"      expanded:    {n_exp_sig}/{n_exp_tot} ({100*n_exp_sig/n_exp_tot:.0f}%)", flush=True)
            print(f"      non-expanded:{n_nox_sig}/{n_nox_tot} ({100*n_nox_sig/n_nox_tot:.0f}%)", flush=True)
            print(f"      Fisher exact p (two-sided) = {p_f:.3g}, OR = {odds:.2f}", flush=True)
        except Exception:
            pass

    # --- Step 1 structural-property: rerun per-RBP regression on outcomes ---
    print("\n" + "="*70, flush=True)
    print("Step 1 per-RBP β on structural property: expanded vs non-expanded", flush=True)
    print("="*70, flush=True)

    # need raw per-RBP × per-ORF binding matrix again; reuse the F3 framework
    s0 = read_step0()
    s0["enst"] = s0["transcript"].str.split(".").str[0]
    expr = pd.read_csv(EXPR)[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]]
    df_base = s0.merge(expr, on="orf_name")
    df_base["l_aalen"] = np.log(df_base["aa_length"])
    df_base["gc"] = df_base["transcript_gc"]
    base = df_base[(df_base.source_sheet == "S2") & (df_base.aa_length <= LEN_CAP)].copy()
    exon_pr = pr.PyRanges(pd.read_parquet(EXONS))

    structural_rows = []
    for cell in CELLS:
        # build per-ENST set of binding RBPs
        peak_dfs = []
        for f in sorted((ECLIP_DIR / cell).glob("*.bed.gz")):
            rbp = f.name.split("__")[0].upper()
            d = pd.read_csv(f, sep="\t", header=None, compression="gzip",
                            usecols=[0, 1, 2, 5],
                            names=["Chromosome", "Start", "End", "Strand"],
                            dtype={0: str, 1: int, 2: int, 5: str})
            d["rbp"] = rbp
            peak_dfs.append(d)
        peaks = pr.PyRanges(pd.concat(peak_dfs, ignore_index=True))
        j = exon_pr.join(peaks, strandedness="same").df
        enst_rbps = j.groupby("enst")["rbp"].agg(set).to_dict()
        rbps_in_cell = sorted(set(peaks.df["rbp"].unique()))

        # outcomes: mean_plddt, catGRANULE
        for outcome in ["mean_plddt", "catgranule2_llps_score"]:
            d = base.dropna(subset=[f"tpm_{cell}_polyA", "gc", outcome]).copy()
            d["l1p_tpm"] = np.log1p(d[f"tpm_{cell}_polyA"])
            print(f"\n  Fitting {len(rbps_in_cell)} per-RBP models for {cell} × {outcome} ...",
                  flush=True)
            for rbp in rbps_in_cell:
                d["bound"] = d["enst"].apply(lambda e: int(rbp in enst_rbps.get(e, set())))
                n_bound = int(d["bound"].sum())
                if n_bound < 10 or n_bound > len(d) - 10:
                    continue
                try:
                    m = smf.ols(f"{outcome} ~ bound + l_aalen + gc + l1p_tpm",
                                data=d).fit()
                    beta = m.params["bound"]
                    p = m.pvalues["bound"]
                    structural_rows.append(dict(cell=cell, outcome=outcome,
                                                rbp=rbp, n_bound=n_bound,
                                                beta=float(beta),
                                                p_raw=float(p)))
                except Exception:
                    pass

    structural_df = pd.DataFrame(structural_rows)
    structural_df["family"] = structural_df["rbp"].map(rbp_to_family)
    structural_df["expanded"] = structural_df["family"].isin(EXPANDED_FAMILIES)
    structural_df.to_csv(RES / "followup5_structural_per_rbp.csv", index=False)

    # Comparison
    for outcome in ["mean_plddt", "catgranule2_llps_score"]:
        print(f"\n  outcome = {outcome}", flush=True)
        for cell in CELLS:
            sub = structural_df[(structural_df.cell == cell) & (structural_df.outcome == outcome)]
            exp = sub[sub.expanded]["beta"].values
            nox = sub[~sub.expanded]["beta"].values
            if len(exp) < 5 or len(nox) < 5:
                continue
            u, p_mw = mannwhitneyu(exp, nox, alternative="two-sided")
            print(f"    {cell}  expanded (n={len(exp)}): median β = "
                  f"{np.median(exp):+.4f}  |  non-expanded (n={len(nox)}): "
                  f"median β = {np.median(nox):+.4f}  |  Mann-Whitney p = {p_mw:.3g}",
                  flush=True)

    print(f"\nWrote {OUT} and followup5_structural_per_rbp.csv")


if __name__ == "__main__":
    main()

"""Annotate cross-cell per-RBP association groups and compare their properties.

The legacy "robust driver" label denotes an association group with positive
and significant lRPF coefficients in both K562 and HepG2. Group comparisons
cover functional class, expanded-family membership, LLPS propensity and
cell-specific detection patterns.

The script tabulates shared versus cell-specific associations and their
cross-cell coefficient correlation. These groups do not identify causal
drivers or establish generalization to other cell types.

Sources:
  results/followup3_per_rbp.csv             — per-RBP TE β (both cells)
  results/followup5_expanded_rbps.csv       — expansion flag per RBP
  data/yasuda2026_rbp_evo/expanded_13species_rbp.csv  — human RBP LLPS scores
  data/eclip/eCLIP_manifest.tsv             — RBP × cell coverage

Outputs:
  results/followup6_driver_characterization.csv — per-RBP merged annotations
  results/followup6_cell_consistency.csv         — both-cell β correlation
  printed: classification + enrichment tests
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EV = ROOT / "data" / "yasuda2026_rbp_evo"

PER_RBP = RES / "followup3_per_rbp.csv"
EXPANDED = RES / "followup5_expanded_rbps.csv"
HUMAN_LLPS = EV / "expanded_13species_rbp.csv"

OUT = RES / "followup6_driver_characterization.csv"
OUT_CELL = RES / "followup6_cell_consistency.csv"


def load_rbp_llps_per_rbp():
    """LLPS score per human RBP from Yasuda 2026."""
    df = pd.read_csv(HUMAN_LLPS)
    human = df[df["common_name"] == "human"].copy()
    human["gene_symbol"] = human["protein_name"].str.split("_").str[0].str.upper()
    llps = (human.dropna(subset=["llphy_score"])
            .groupby("gene_symbol")["llphy_score"].mean()
            .to_dict())
    return llps


def main():
    pr = pd.read_csv(PER_RBP)
    f5 = pd.read_csv(EXPANDED)[["rbp", "cell", "family", "expanded"]]
    pr = pr.merge(f5, on=["rbp", "cell"], how="left")
    pr["rbp_u"] = pr["rbp"].str.upper()

    llps = load_rbp_llps_per_rbp()
    pr["llps_score"] = pr["rbp_u"].map(llps)
    print(f"  LLPS score annotated for {pr['llps_score'].notna().sum()} / "
          f"{len(pr)} per-cell rows", flush=True)

    # --- Driver classification per cell ---
    pr["positive_sig"] = (pr["beta"] > 0) & (pr["p_raw"] < 0.05) & pr["beta"].notna()
    k562 = pr[pr.cell == "K562"].set_index("rbp")
    hepg2 = pr[pr.cell == "HepG2"].set_index("rbp")
    shared_rbps = k562.index.intersection(hepg2.index)

    # robust driver = sig positive in BOTH cells
    robust_drivers = [r for r in shared_rbps
                      if k562.loc[r, "positive_sig"] and hepg2.loc[r, "positive_sig"]]
    k562_only = [r for r in shared_rbps
                 if k562.loc[r, "positive_sig"] and not hepg2.loc[r, "positive_sig"]]
    hepg2_only = [r for r in shared_rbps
                  if not k562.loc[r, "positive_sig"] and hepg2.loc[r, "positive_sig"]]
    neither = [r for r in shared_rbps
               if not k562.loc[r, "positive_sig"] and not hepg2.loc[r, "positive_sig"]]

    print(f"\n  Both-cell shared eCLIP RBPs: {len(shared_rbps)}", flush=True)
    print(f"    Robust drivers (sig+ in BOTH):  {len(robust_drivers)}", flush=True)
    print(f"    K562-only sig+:                 {len(k562_only)}", flush=True)
    print(f"    HepG2-only sig+:                {len(hepg2_only)}", flush=True)
    print(f"    Neither cell sig+:              {len(neither)}", flush=True)
    print(f"\n  Robust drivers: {sorted(robust_drivers)}", flush=True)
    print(f"\n  K562-only drivers: {sorted(k562_only)}", flush=True)
    print(f"\n  HepG2-only drivers: {sorted(hepg2_only)}", flush=True)

    # --- Cross-cell β correlation (cell-context dependence) ---
    cc = pd.DataFrame({
        "rbp": shared_rbps,
        "beta_K562": [k562.loc[r, "beta"] for r in shared_rbps],
        "beta_HepG2": [hepg2.loc[r, "beta"] for r in shared_rbps],
    }).dropna()
    rho, p_sp = spearmanr(cc["beta_K562"], cc["beta_HepG2"])
    pear = cc[["beta_K562", "beta_HepG2"]].corr().iloc[0, 1]
    print(f"\n  Cross-cell β correlation (n={len(cc)}):", flush=True)
    print(f"    Spearman ρ = {rho:+.3f}  (p = {p_sp:.3g})", flush=True)
    print(f"    Pearson  r = {pear:+.3f}", flush=True)
    cc.to_csv(OUT_CELL, index=False)

    # --- Driver characterization tests ---
    drivers_set = set(robust_drivers)
    cell_specific = set(k562_only) | set(hepg2_only)
    nondrivers_set = set(neither)

    # Build per-RBP table with annotations
    rbp_anno = pr.groupby("rbp").agg({
        "family": "first",
        "expanded": "first",
        "llps_score": "first",
    }).reset_index()
    rbp_anno["driver_class"] = rbp_anno["rbp"].apply(
        lambda r: "robust" if r in drivers_set
        else ("cell_specific" if r in cell_specific
              else ("nondriver" if r in nondrivers_set
                    else "unshared_cell")))
    rbp_anno.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}", flush=True)

    # Test 1: are drivers enriched in expanded RBP families?
    print("\n" + "="*70, flush=True)
    print("Test 1: Expansion enrichment in robust drivers", flush=True)
    print("="*70, flush=True)
    for grp_name, grp_set in [("robust_drivers", drivers_set),
                               ("cell_specific_drivers", cell_specific),
                               ("nondrivers", nondrivers_set)]:
        anno = rbp_anno[rbp_anno["rbp"].isin(grp_set)]
        n_exp = int(anno["expanded"].sum())
        n_tot = len(anno)
        print(f"  {grp_name:25s}: expanded {n_exp}/{n_tot} ({100*n_exp/max(n_tot,1):.0f}%)",
              flush=True)
    # Fisher: robust drivers vs nondrivers, expansion enrichment
    rd = rbp_anno[rbp_anno["rbp"].isin(drivers_set)]
    nd = rbp_anno[rbp_anno["rbp"].isin(nondrivers_set)]
    a = int(rd["expanded"].sum()); b = len(rd) - a
    c = int(nd["expanded"].sum()); d = len(nd) - c
    if (a + b) >= 3 and (c + d) >= 3:
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        print(f"  Fisher (robust vs nondriver expansion): OR={odds:.2f}, p={p:.3g}",
              flush=True)

    # Test 2: family distribution in robust drivers
    print("\n" + "="*70, flush=True)
    print("Test 2: Family distribution in robust drivers vs nondrivers", flush=True)
    print("="*70, flush=True)
    if len(drivers_set) > 0:
        rd_fam = rbp_anno[rbp_anno["rbp"].isin(drivers_set)]["family"].value_counts()
        print("\n  Robust drivers:")
        for f, n in rd_fam.items():
            print(f"    {f:20s} {n}", flush=True)
    if len(nondrivers_set) > 0:
        nd_fam = rbp_anno[rbp_anno["rbp"].isin(nondrivers_set)]["family"].value_counts()
        print("\n  Nondrivers (top 8):")
        for f, n in nd_fam.head(8).items():
            print(f"    {f:20s} {n}", flush=True)

    # Test 3: LLPS score of drivers vs nondrivers
    print("\n" + "="*70, flush=True)
    print("Test 3: LLPS propensity of drivers vs nondrivers", flush=True)
    print("="*70, flush=True)
    rd_llps = rbp_anno[rbp_anno["rbp"].isin(drivers_set)]["llps_score"].dropna()
    nd_llps = rbp_anno[rbp_anno["rbp"].isin(nondrivers_set)]["llps_score"].dropna()
    cs_llps = rbp_anno[rbp_anno["rbp"].isin(cell_specific)]["llps_score"].dropna()
    print(f"  robust drivers   (n={len(rd_llps):2d}): median LLPS = {rd_llps.median() if len(rd_llps) else 'NA':+.3f}",
          flush=True)
    print(f"  cell_specific    (n={len(cs_llps):2d}): median LLPS = {cs_llps.median() if len(cs_llps) else 'NA':+.3f}",
          flush=True)
    print(f"  nondrivers       (n={len(nd_llps):2d}): median LLPS = {nd_llps.median() if len(nd_llps) else 'NA':+.3f}",
          flush=True)
    if len(rd_llps) >= 3 and len(nd_llps) >= 3:
        u, p = mannwhitneyu(rd_llps, nd_llps, alternative="two-sided")
        print(f"  MW (robust vs nondriver): p = {p:.3g}", flush=True)

    # Test 4: top per-RBP β table (combined ranking)
    print("\n" + "="*70, flush=True)
    print("Test 4: Top 12 RBPs by mean(K562 β + HepG2 β)", flush=True)
    print("="*70, flush=True)
    cc_top = cc.copy()
    cc_top["mean_beta"] = (cc_top["beta_K562"] + cc_top["beta_HepG2"]) / 2
    cc_top = cc_top.merge(rbp_anno[["rbp", "family", "expanded", "llps_score"]], on="rbp")
    cc_top = cc_top.sort_values("mean_beta", ascending=False).head(12)
    print("\n  rank  RBP        family          expanded  LLPS    βK562   βHepG2  mean", flush=True)
    for i, r in cc_top.reset_index(drop=True).iterrows():
        print(f"  {i+1:>4}  {r['rbp']:10s} {str(r['family']):14s} "
              f"{str(r['expanded']):8s} {r['llps_score'] if pd.notna(r['llps_score']) else 'NA':>5} "
              f"  {r['beta_K562']:+.2f}   {r['beta_HepG2']:+.2f}   {r['mean_beta']:+.2f}", flush=True)


if __name__ == "__main__":
    main()

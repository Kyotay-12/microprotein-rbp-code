"""Followup 13 — covariate-adjusted residual permutation with exchangeability blocks.

WHY. The parametric p-values for the headline association rest on HC3 standard
errors, and the Ribo-seq signal comes from per-cell-line bigWigs that merge two
biological replicates. A permutation test asks a narrower but assumption-light
question: does the cross-ORF association exceed what an exchangeable null
produces? A naive shuffle of the predictor is not adequate here, because
(i) ORFs on the same host transcript carry identical predictor values and
(ii) the predictor is correlated with the covariates, so shuffling it destroys
the covariate structure as well.

We therefore use Freedman-Lane residual permutation with blocks:

    1. fit the null model      y ~ log(AA_length) + GC + log1p(TPM)
    2. keep fitted values and residuals
    3. permute the residuals BETWEEN BLOCKS, a block being all ORFs on one host
       transcript (or one gene). Blocks are exchanged only with blocks of the
       same size, so the block structure and the size distribution are preserved
    4. y* = null fitted + permuted residuals
    5. refit the full model with log1p(unique_RBP) and recompute the statistic

The statistic is the t-ratio of the RBP coefficient using CLUSTER-ROBUST
standard errors with the same blocks, not beta alone: beta ignores how the
permutation changes the variance.

Three exchangeability schemes, per cell line:
    transcript blocks / gene blocks / transcripts carrying exactly one ORF

Empirical p = (1 + #{|t*| >= |t_obs|}) / (1 + B).

This does NOT restore replicate-level variance; it evaluates whether the
cross-ORF association exceeds a covariate-adjusted exchangeable null.

No dummy values; failures raise.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "followup13_freedman_lane.csv"
B = 20000
SEED = 0


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
    d = (s0.merge(cat, on="orf_name", how="left")
           .merge(bind[["orf_name", "K562_nrbp_tx", "HepG2_nrbp_tx"]], on="orf_name")
           .merge(expr[["orf_name", "tpm_K562_polyA", "tpm_HepG2_polyA"]], on="orf_name")
           .merge(rpf[["orf_name", "K562_rpf_in", "HepG2_rpf_in"]], on="orf_name"))
    d = d[(d.source_sheet == "S2") & (d.aa_length <= 100)].copy()
    if d["gene_id"].isna().any():
        raise RuntimeError(f"{int(d.gene_id.isna().sum())} ORFs lack a stable ENSG")
    # stable IDs, version-stripped; PAR_Y suffixes kept distinct
    d["tx"] = d["transcript"].astype(str).str.split(".").str[0]
    d["gene"] = d["gene_id"].astype(str).str.split(".").str[0]
    d["l_aalen"] = np.log(d["aa_length"])
    d["gc"] = d["transcript_gc"]
    return d


def cluster_t(X, y, XtXi, groups_idx, j=1):
    """coefficient j and its cluster-robust t, given precomputed (X'X)^-1"""
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idx in groups_idx:
        u = X[idx].T @ e[idx]
        meat += np.outer(u, u)
    G, n, k = len(groups_idx), X.shape[0], X.shape[1]
    scale = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = XtXi @ (scale * meat) @ XtXi
    return beta[j], beta[j] / np.sqrt(V[j, j])


def blocks_of(labels):
    order = {}
    for i, g in enumerate(labels):
        order.setdefault(g, []).append(i)
    return [np.array(v) for v in order.values()]


def freedman_lane(df, block_col, rng):
    y = np.log2(df["rpf"].to_numpy() + 1.0)
    ones = np.ones(len(df))
    l1p_rbp = np.log1p(df["nrbp"].to_numpy())
    Z = np.column_stack([ones, df["l_aalen"], df["gc"],
                         np.log1p(df["tpm"].to_numpy())])          # null design
    X = np.column_stack([ones, l1p_rbp, df["l_aalen"], df["gc"],
                         np.log1p(df["tpm"].to_numpy())])          # full design
    XtXi = np.linalg.inv(X.T @ X)
    grp = blocks_of(df[block_col].to_numpy())
    beta_obs, t_obs = cluster_t(X, y, XtXi, grp)

    # null-model fit
    bz = np.linalg.lstsq(Z, y, rcond=None)[0]
    fitted = Z @ bz
    resid = y - fitted

    # blocks exchanged only within their size class
    by_size = {}
    for bi, idx in enumerate(grp):
        by_size.setdefault(len(idx), []).append(bi)

    hits = 0
    for _ in range(B):
        r = np.empty_like(resid)
        for size, members in by_size.items():
            perm = rng.permutation(len(members))
            for a, b in zip(members, perm):
                r[grp[a]] = resid[grp[members[b]]]
        _, t_star = cluster_t(X, fitted + r, XtXi, grp)
        if abs(t_star) >= abs(t_obs):
            hits += 1
    p_emp = (1.0 + hits) / (1.0 + B)
    return dict(n=len(df), n_block=len(grp),
                max_block=max(len(g) for g in grp),
                beta=beta_obs, t_obs=t_obs, n_exceed=hits,
                B=B, p_empirical=p_emp)


def main():
    d = build()
    rng = np.random.default_rng(SEED)
    rows = []
    for cell in ["K562", "HepG2"]:
        w = d.copy()
        w["nrbp"] = w[f"{cell}_nrbp_tx"]
        w["tpm"] = w[f"tpm_{cell}_polyA"]
        w["rpf"] = w[f"{cell}_rpf_in"]
        w = w.dropna(subset=["tpm", "nrbp", "gc", "aa_length", "rpf"])
        w = w[w["tpm"] >= 1].copy()

        # (iii) transcripts carrying EXACTLY ONE analysed ORF
        counts = w["tx"].value_counts()
        single = w[w["tx"].isin(counts[counts == 1].index)].copy()

        for label, frame, col in [("transcript_block", w, "tx"),
                                  ("gene_block", w, "gene"),
                                  ("single_ORF_transcripts", single, "tx")]:
            r = freedman_lane(frame, col, rng)
            r.update(cell=cell, scheme=label)
            rows.append(r)
            print(f"  {cell:6s} {label:24s} n={r['n']:5d} blocks={r['n_block']:5d} "
                  f"max={r['max_block']} beta={r['beta']:+.4f} t={r['t_obs']:+.2f} "
                  f"exceed={r['n_exceed']}/{B} p={r['p_empirical']:.2e}", flush=True)

    out = pd.DataFrame(rows)[["cell", "scheme", "n", "n_block", "max_block",
                              "beta", "t_obs", "n_exceed", "B", "p_empirical"]]
    out.to_csv(OUT, index=False)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()

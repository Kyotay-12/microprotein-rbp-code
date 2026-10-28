"""Generate the 4 main paper figures from result CSVs.

Each panel reads directly from results/* CSVs; no hardcoded values.
Saved at 300 dpi PNG to figures/figure[1-4].png.

  Figure 1  Property landscape, RBP embedding, spatial distribution
  Figure 2  Structural-property associations across four prespecified axes
  Figure 3  Translation-output associations and sensitivity analyses
  Figure 4  Followup analyses (R² comparison, canonical CDS, per-RBP, GC→property)
"""
from pathlib import Path
import hashlib
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT.parent / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INPUT_HASHES = {}
GENERATOR_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_result(name, **kwargs):
    path = RES / name
    INPUT_HASHES[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, **kwargs)

# ---------- visual style ----------
mpl.rcParams.update({
    "font.size": 7,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 7.5,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "figure.dpi": 130,
    "savefig.dpi": 300,
})
# Cell Press print widths: 85 mm (1 col), 114 mm (1.5 col), 174 mm (2 col).
# Every panel figure here is full width. We target 173 mm so the saved raster
# is inside the 174 mm ceiling with a margin, and we do NOT use
# savefig.bbox="tight" — a tight bounding box is computed from whatever the
# artists happen to occupy, so the delivered width would not be the width we
# specified. Layout is fitted inside the fixed canvas with tight_layout instead.
FIG_W_MM = 173.0
FIG_W = FIG_W_MM / 25.4          # inches
MAX_W_MM = 174.0                 # journal ceiling


def save(fig, stem):
    """Write PNG (300 dpi) + vector PDF, and fail if the raster overruns the
    journal's column width."""
    png = FIG / f"{stem}.png"
    fig.savefig(png)
    fig.savefig(FIG / f"{stem}.pdf")
    from PIL import Image
    w_px = Image.open(png).size[0]
    mm = w_px / 300 * 25.4
    if mm > MAX_W_MM:
        raise ValueError(f"{stem}.png is {mm:.1f} mm wide, ceiling is {MAX_W_MM} mm")
    print(f"  wrote {png}  ({mm:.1f} mm at 300 dpi) + .pdf")


K_COL = "#3266cc"   # K562 (blue)
H_COL = "#e67e22"   # HepG2 (orange)
GREY = "#888888"
LIGHT = "#cfd9eb"


# =============================================================
# Figure 1 — property landscape + RBP embedding + spatial
# =============================================================
def figure1():
    # Manual layout, not tight_layout: the two panel legends sit outside their
    # axes, and a layout engine that reserves space for them shrinks the axes
    # unpredictably. Fixed margins keep the delivered geometry deterministic.
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.45),
                             gridspec_kw={"width_ratios": [1.35, 0.85, 0.9]})
    fig.subplots_adjust(left=0.065, right=0.995, top=0.79, bottom=0.26,
                        wspace=0.42)
    for a in axes:
        a.title.set_fontsize(8)
    desc = read_result("step1e_type_descriptive.csv")
    desc = desc[desc["analyzable"] == "yes"].copy()
    desc = desc.sort_values("n", ascending=False).reset_index(drop=True)

    # 1A — stacked bar of structural class by ORF biotype
    ax = axes[0]
    biotypes = desc["orf_biotype"].tolist()
    pos = np.arange(len(biotypes))
    dis = desc["disordered_pct"].values
    inter = desc["intermediate_pct"].values
    fold = desc["foldable_pct"].values
    ax.bar(pos, dis, color="#cc4444", label="disordered", width=0.65)
    ax.bar(pos, inter, bottom=dis, color="#dddd88", label="intermediate", width=0.65)
    ax.bar(pos, fold, bottom=dis + inter, color="#66aacc", label="foldable", width=0.65)
    ax.set_xticks(pos)
    ax.set_xticklabels(biotypes)
    ax.set_ylabel("% of ORFs")
    ax.set_title("A  Structural class composition\nby ORF biotype (main pop)", loc="left")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    hA, lA = ax.get_legend_handles_labels()
    for i, (b, f, n) in enumerate(zip(biotypes, fold, desc["n"].values)):
        ax.text(i, dis[i] + inter[i] + fold[i] + 3,
                f"n={int(n)}\n{f:.1f}% fold",
                ha="center", fontsize=5.5, color="#222", linespacing=1.15)

    # 1B — binding pervasiveness K562 vs HepG2
    ax = axes[1]
    bind = read_result("step1_binding.csv")
    main = bind[(bind.source_sheet == "S2") & (bind.aa_length <= 100)].copy()
    frac_k = (main["K562_nrbp_tx"] > 0).mean() * 100
    frac_h = (main["HepG2_nrbp_tx"] > 0).mean() * 100
    med_k = int(main["K562_nrbp_tx"].median())
    med_h = int(main["HepG2_nrbp_tx"].median())
    if med_k != med_h:
        raise ValueError(f"panel B text assumes one median, got {med_k}/{med_h}")
    max_k = int(main["K562_nrbp_tx"].max())
    max_h = int(main["HepG2_nrbp_tx"].max())
    cells = ["K562", "HepG2"]
    fracs = [frac_k, frac_h]
    bars = ax.bar(cells, fracs, color=[K_COL, H_COL], width=0.55, alpha=0.9)
    for b, v in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, v + 2,
                f"{v:.1f}%", ha="center", fontsize=8, weight="bold")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of ORFs bound by ≥1 RBP")
    ax.set_title("B  RBP embedding\n(host transcript binding)", loc="left")
    # 5.2 pt, not 5.8: at the larger size the box was wider than this panel
    # and its left edge crossed the y-axis spine and the 100 tick
    ax.text(0.5, 0.93,
            f"median {med_k} unique RBP / ORF\n"
            f"(max {max_k} / {max_h}); n = {len(main):,}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=5.2, color="#333", linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor="#bbb",
                      boxstyle="round,pad=0.2", linewidth=0.6))

    # 1C — spatial: tx-exon vs ORF-span mean
    ax = axes[2]
    mean_tx = [main["K562_nrbp_tx"].mean(), main["HepG2_nrbp_tx"].mean()]
    mean_orf = [main["K562_nrbp_orfspan"].mean(), main["HepG2_nrbp_orfspan"].mean()]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, mean_tx, w, color=GREY, label="whole transcript")
    ax.bar(x + w / 2, mean_orf, w, color=LIGHT, label="ORF span only")
    ax.set_xticks(x); ax.set_xticklabels(cells)
    ax.set_ylabel("mean unique RBP / ORF")
    ax.set_title("C  RBP counts\nby interval", loc="left")
    ax.set_ylim(0, max(mean_tx) * 1.42)
    hC, lC = ax.get_legend_handles_labels()
    for i, (tx, orf) in enumerate(zip(mean_tx, mean_orf)):
        pct_outside = (1 - orf / tx) * 100
        ax.text(i, tx * 1.06, f"{pct_outside:.1f}%\nlower",
                ha="center", fontsize=7, color="#222", weight="bold",
                linespacing=1.15)

    # legends in FIGURE coordinates, each centred under its own panel
    fig.legend(hA, lA, loc="lower center", bbox_to_anchor=(0.235, 0.005),
               ncols=3, fontsize=6.5, frameon=False,
               columnspacing=1.0, handlelength=1.1, handletextpad=0.5)
    fig.legend(hC, lC, loc="lower center", bbox_to_anchor=(0.815, 0.005),
               ncols=2, fontsize=6.5, frameon=False,
               columnspacing=1.0, handlelength=1.1, handletextpad=0.5)
    save(fig, "figure1")
    plt.close()



# =============================================================
# Figure 2 — Structural-property associations across four prespecified axes
# =============================================================
def figure2():
    """Four prespecified axes, 2x2 so that every label survives the column width.

    A total binding -> B binding position -> C RBP class -> D ORF context.
    One shared legend; results are carried by the estimates and intervals, not
    by verdict words printed on the panels.
    """
    # Drawn at the final two-column print width (180 mm = 7.09 in) so that the
# on-screen font sizes are the printed font sizes; no shrink-to-fit.
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.2))
    handles = None

    def forest(ax, rows, order, xlabel, title):
        """rows: DataFrame with label / cell / beta / ci_lo / ci_hi"""
        nonlocal handles
        y = np.arange(len(order))
        for cell, color, off in [("K562", K_COL, -0.16), ("HepG2", H_COL, 0.16)]:
            d = rows[rows.cell == cell].set_index("label").reindex(order)
            ax.errorbar(d["beta"], y + off,
                        xerr=[d["beta"] - d["ci_lo"], d["ci_hi"] - d["beta"]],
                        fmt="o", color=color, ecolor=color, markersize=5,
                        capsize=2, label=cell, lw=1.2)
        ax.axvline(0, color="black", lw=0.7, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=7)
        ax.set_ylim(len(order) - 0.55, -0.45)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        if handles is None:
            handles, _ = ax.get_legend_handles_labels()

    # ---- A: total binding ---------------------------------------------------
    a = read_result("step1_maintest_results.csv")
    a = a[a["term"] == "l1p_rbp"].copy()
    lab = {"mean_plddt": "pLDDT\n(raw β)",
           "catgranule2_llps_score": "catGRANULE\n(β × 100)",
           "foldable": "foldable\n(log OR)",
           "llps_prone": "LLPS-prone\n(log OR)"}
    rows = []
    for _, r in a.iterrows():
        # Branch on the MODEL TYPE, never on "coef is NaN": a silent fallback
        # from one column to another is how a wrong scale gets plotted.
        if r["kind"] == "logit":
            # logistic rows carry OR, not a linear coefficient. Point AND both
            # interval bounds are log-transformed so the zero line keeps meaning
            # "no association", as it does for the OLS rows.
            for c in ("OR", "or_lo", "or_hi"):
                if not (np.isfinite(r[c]) and r[c] > 0):
                    raise ValueError(f"{r['cell']} {r['outcome']}: {c}={r[c]!r}"
                                     " is not a positive odds ratio")
            if not (r["or_lo"] <= r["OR"] <= r["or_hi"]):
                raise ValueError(f"{r['cell']} {r['outcome']}: OR outside its CI")
            b, lo, hi = (np.log(r["OR"]), np.log(r["or_lo"]), np.log(r["or_hi"]))
        elif r["kind"] == "ols":
            b, lo, hi = r["coef"], r["ci_lo"], r["ci_hi"]
            if r["outcome"] == "catgranule2_llps_score":
                b, lo, hi = b * 100, lo * 100, hi * 100
        else:
            raise ValueError(f"unknown model kind {r['kind']!r}")
        if not np.isfinite([b, lo, hi]).all():
            raise ValueError(f"non-finite estimate for {r['cell']} {r['outcome']}")
        if not (lo <= b <= hi):
            raise ValueError(f"{r['cell']} {r['outcome']}: estimate outside interval")
        rows.append((lab.get(r["outcome"], r["outcome"]), r["cell"], b, lo, hi))
    A = pd.DataFrame(rows, columns=["label", "cell", "beta", "ci_lo", "ci_hi"])
    forest(axes[0, 0], A, list(lab.values()),
           "coefficient on log1p(unique RBP)",
           "A  Total RBP-binding burden")

    # ---- B: regional density ------------------------------------------------
    b = read_result("step1c_regional_results.csv").copy()
    m = b["outcome"] == "catGRANULE"
    for c in ("beta", "ci_lo", "ci_hi"):
        b.loc[m, c] *= 100
    nice = {"utr5": "5'UTR", "cds": "CDS", "utr3": "3'UTR"}
    b["label"] = [f"{nice[r.region]} → {r.outcome}" for r in b.itertuples()]
    forest(axes[0, 1], b,
           ["5'UTR → pLDDT", "5'UTR → catGRANULE", "CDS → pLDDT",
            "CDS → catGRANULE", "3'UTR → pLDDT", "3'UTR → catGRANULE"],
           "β per unique RBP kb⁻¹\n(catGRANULE × 100)",
           "B  Binding by transcript region")

    # ---- C: prespecified functional classes ---------------------------------
    c = read_result("step1d_classwise_results.csv")
    c = c[c["prior"] == "effect+"].copy()
    m = c["outcome"] == "catGRANULE"
    for col in ("beta", "ci_lo", "ci_hi"):
        c.loc[m, col] *= 100
    c["label"] = c["vn_class"].map({"translation": "Translation regulation\n→ pLDDT",
                                    "pb_sg": "P-body / stress granule\n→ catGRANULE"})
    forest(axes[1, 0], c,
           ["Translation regulation\n→ pLDDT", "P-body / stress granule\n→ catGRANULE"],
           "coefficient   (catGRANULE × 100)",
           "C  Prespecified RBP functional classes")

    # ---- D: ORF biotype -----------------------------------------------------
    d = read_result("step1e_type_regression_results.csv")
    d = d[d["kind"] == "pre-reg"].copy()
    m = d["outcome"] == "catGRANULE"
    for col in ("beta", "ci_lo", "ci_hi"):
        d.loc[m, col] *= 100
    d["label"] = d.apply(lambda r: f"{r['orf_biotype']} → {r['outcome']}", axis=1)
    forest(axes[1, 1], d,
           ["lncRNA → pLDDT", "lncRNA → catGRANULE",
            "dORF → pLDDT", "dORF → catGRANULE"],
           "coefficient   (catGRANULE × 100)",
           "D  ORF-biotype associations")

    fig.legend(handles=handles, labels=["K562", "HepG2"], loc="lower center",
               ncols=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, "figure2")
    plt.close()



# =============================================================
# Figure 3 — Translation-output associations: forest, TPM sensitivity, ratios
# =============================================================
def figure3():
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.8),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    # 3A — forest plot of β for each stratum × cell
    ax = axes[0]
    # Headline specification (lRPF, TPM >= 1) for EVERY stratum, so that all
    # three panels of Figure 3 show the same outcome and the same population.
    # The former TE_raw source (step2_TE_results.csv) is a different outcome and
    # a different n, and is now reported as a sensitivity analysis in the text.
    te = read_result("followup12_stratified_lrpf.csv")
    strata_order = ["main", "lncRNA", "uORF", "uoORF", "dORF", "intORF"]
    _n = te[te.cell == "K562"].set_index("stratum")["n"].to_dict()
    pretty = {k: f"{k}\n(n={_n.get(k, 0):,})" for k in strata_order}
    y_pos = np.arange(len(strata_order))
    for cell, color, off in [("K562", K_COL, -0.18), ("HepG2", H_COL, 0.18)]:
        d = te[(te.cell == cell)].set_index("stratum").reindex(strata_order)
        ax.errorbar(d["beta"], y_pos + off,
                    xerr=[d["beta"] - d["ci_lo"], d["ci_hi"] - d["beta"]],
                    fmt="o", color=color, ecolor=color, markersize=5,
                    capsize=2, label=cell, lw=1.2)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_yticks(y_pos); ax.set_yticklabels([pretty[s] for s in strata_order],
                                              fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("β of log1p(unique RBP) on lRPF")
    ax.set_title("A  Translation output by stratum\n(primary lRPF, TPM ≥ 1)", loc="left")

    # 3B — TPM sliding curve, main pop
    ax = axes[1]
    sl = read_result("step2_tpm_sliding.csv")
    ax.axvspan(0.5, 2.0, color="green", alpha=0.10, label="TPM 0.5–2 range", zorder=0)
    for cell, color in [("K562", K_COL), ("HepG2", H_COL)]:
        d = sl[(sl.cell == cell) & (sl.stratum == "main")].sort_values("tpm_cutoff")
        ax.plot(d["tpm_cutoff"], d["beta"], "o-", color=color, label=cell, lw=1.5)
        ax.fill_between(d["tpm_cutoff"], d["ci_lo"], d["ci_hi"],
                        color=color, alpha=0.15)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.axvline(1.0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("TPM cutoff (≥)")
    ax.set_ylabel("β of log1p(unique RBP) on lRPF")
    ax.set_title("B  TPM sliding sensitivity\n(main pop, lRPF model)", loc="left")
    # anchored to the top of the dashed cutoff line, in the empty upper area:
    # at y ~ 0 it was struck through by the dotted zero line and collided with
    # the 0.0 tick label
    ax.text(1.12, ax.get_ylim()[1] * 0.93, "primary\nTPM ≥ 1",
            ha="left", va="top", fontsize=6.5, color="#444", linespacing=1.2)

    # 3C — fold change curve
    ax = axes[2]
    # Headline coefficient (lRPF at TPM >= 1), NOT the TPM >= 0 variant in
    # step2_verification.csv -- panels A and B are the headline specification
    # and this panel must use the same one.
    lrpf = read_result("step2_tpm_sliding.csv")
    lrpf = lrpf[(lrpf.stratum == "main") & (lrpf.tpm_cutoff == 1.0)]
    # Range capped at 30: the 99th percentile of unique RBPs is 33 (K562) and
    # 26 (HepG2), so 50 would extrapolate past the HepG2 maximum of 35.
    rbp_vals = np.array([0, 1, 5, 10, 20, 30])
    log1p_rbp = np.log1p(rbp_vals)
    for cell, color in [("K562", K_COL), ("HepG2", H_COL)]:
        sub = lrpf[lrpf.cell == cell].iloc[0]
        beta = sub["beta"]; lo = sub["ci_lo"]; hi = sub["ci_hi"]
        fold = 2 ** (beta * log1p_rbp)
        fold_lo = 2 ** (lo * log1p_rbp)
        fold_hi = 2 ** (hi * log1p_rbp)
        ax.plot(rbp_vals, fold, "o-", color=color, label=cell, lw=1.5)
        ax.fill_between(rbp_vals, fold_lo, fold_hi, color=color, alpha=0.15)
    ax.axhline(1, color="black", lw=0.6, ls=":")
    ax.set_xlabel("unique RBPs binding\nhost transcript")
    ax.set_ylabel("back-transformed (RPF + 1) ratio\n(vs zero detected RBPs)")
    ax.set_title("C  Model-implied\n(RPF + 1) ratio", loc="left")
    # linear: symlog compressed 20 and 30 into overlapping tick labels
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    # The two values at 20 RBPs, read off a guide line rather than labelled in
    # place: leader lines from the markers ran across both curves and both
    # confidence bands, and the K562 label landed on the K562 curve itself.
    # The upper-left of this panel is empty, so the numbers go there.
    ax.axvline(20, color="#999", lw=0.5, ls=":", zorder=0)
    top = ax.get_ylim()[1]
    for i, (cell, color) in enumerate([("K562", K_COL), ("HepG2", H_COL)]):
        sub = lrpf[lrpf.cell == cell].iloc[0]
        f = 2 ** (sub["beta"] * np.log1p(20))
        ax.text(0.6, top - 0.06 * (top - 1) - i * 0.11 * (top - 1),
                f"{f:.1f}× at 20 RBPs", ha="left", va="top",
                fontsize=6.5, color=color)

    h, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=h[:2], labels=["K562", "HepG2"], loc="lower center",
               ncols=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.005))
    for a_ in axes:
        a_.title.set_fontsize(8)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "figure3")
    plt.close()



# =============================================================
# Figure 4 — Followup analyses
# =============================================================
def figure4():
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.4),
                              gridspec_kw={"hspace": 0.42, "wspace": 0.28})

    # 4A — partial R² comparison Step 1 vs Step 2
    ax = axes[0, 0]
    pr = read_result("followup1_partial_r2.csv")
    # the Step 1 / Step 2 split is carried by the divider and the two group
    # labels, so the tick labels only need to name the outcome
    rows = [
        ("pLDDT", "Step1_axis_a", "mean_plddt"),
        ("cat-\nGRAN", "Step1_axis_a", "catgranule2_llps_score"),
        ("fold-\nable", "Step1_axis_a", "foldable"),
        ("LLPS-\nprone", "Step1_axis_a", "llps_prone"),
        ("lRPF\nmain", "Step2_lRPF_main_TPMge1", "lRPF"),
        ("lRPF\nlncRNA", "Step2_lRPF_lncRNA_TPMge1", "lRPF"),
    ]
    labels = [r[0] for r in rows]
    sp_k, sp_h = [], []
    for _, ana, out in rows:
        d = pr[(pr.analysis == ana) & (pr.outcome == out)]
        sp_k.append(d[d.cell == "K562"]["semi_partial_R2"].iloc[0] * 100)
        sp_h.append(d[d.cell == "HepG2"]["semi_partial_R2"].iloc[0] * 100)
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, sp_k, w, color=K_COL, label="K562", alpha=0.9)
    ax.bar(x + w / 2, sp_h, w, color=H_COL, label="HepG2", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, linespacing=1.2)
    ax.set_ylabel("ΔR² / Δpseudo-R²\n(× 100; see legend)")
    ax.set_title("A  Incremental model fit\n"
                 "OLS and logistic models", loc="left")
    # divider line between Step 1 and Step 2, with headroom for the two labels
    ax.set_ylim(0, max(sp_k + sp_h) * 1.30)
    ax.axvline(3.5, color="#888", lw=0.5, ls=":")
    ax.text(1.5, ax.get_ylim()[1] * 0.94, "Step 1 (structural)",
            ha="center", va="top", fontsize=6.5, color="#555")
    ax.text(4.5, ax.get_ylim()[1] * 0.94, "Step 2 (translation)",
            ha="center", va="top", fontsize=6.5, color="#555")

    # 4B — canonical CDS vs ncORF β (both at TPM ≥ 1 for apples-to-apples)
    ax = axes[0, 1]
    # followup10 reads canonical TPM from the complete ENCODE quantifications.
    can = read_result("followup10_canonical_lrpf_fixed.csv")
    nc = read_result("step2_tpm_sliding.csv")
    nc = nc[(nc.stratum == "main") & (nc.tpm_cutoff == 1.0)]
    cells = ["K562", "HepG2"]
    nc_beta = [nc[nc.cell == c]["beta"].iloc[0] for c in cells]
    nc_lo = [nc[nc.cell == c]["ci_lo"].iloc[0] for c in cells]
    nc_hi = [nc[nc.cell == c]["ci_hi"].iloc[0] for c in cells]
    can_beta = [can[can.cell == c]["beta"].iloc[0] for c in cells]
    can_lo = [can[can.cell == c]["ci_lo"].iloc[0] for c in cells]
    can_hi = [can[can.cell == c]["ci_hi"].iloc[0] for c in cells]
    x = np.arange(2); w = 0.35
    ax.bar(x - w / 2, nc_beta, w,
           yerr=[np.array(nc_beta) - np.array(nc_lo),
                 np.array(nc_hi) - np.array(nc_beta)],
           color="#aab8d8", label="ncORF main population", capsize=4, ecolor="#444")
    ax.bar(x + w / 2, can_beta, w,
           yerr=[np.array(can_beta) - np.array(can_lo),
                 np.array(can_hi) - np.array(can_beta)],
           color="#3d4d77", label="canonical CDS reference", capsize=4, ecolor="#444")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(cells)
    ax.set_ylabel("β of log1p(unique RBP) on lRPF")
    ax.set_title("B  Canonical CDS reference\n"
                 "Canonical CDS reference", loc="left")
    ax.set_ylim(0, max(can_beta) * 1.62)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.1,
              handletextpad=0.5, borderpad=0.3)
    for i, (n, c, chi) in enumerate(zip(nc_beta, can_beta, can_hi)):
        # clear the canonical error-bar cap, not just the bar top
        ax.text(i, chi + 0.07, f"nc/can = {n/c:.2f}",
                ha="center", fontsize=6.5, color="#222")

    # 4C — per-RBP top drivers forest plot
    ax = axes[1, 0]
    pr3 = read_result("followup3_per_rbp.csv")
    top = []
    for cell in cells:
        sub = pr3[(pr3.cell == cell) & (pr3["beta"] > 0)
                  & (pr3["p_bonf"] < 0.05)].copy()
        sub = sub.sort_values("beta", ascending=False).head(12)
        top.append(sub)
    top_df = pd.concat(top, ignore_index=True)
    rbp_order = top_df.sort_values("beta", ascending=False)["rbp"].unique()[:14]
    y_pos = np.arange(len(rbp_order))
    for cell, color, off in [("K562", K_COL, -0.18), ("HepG2", H_COL, 0.18)]:
        d = pr3[(pr3.cell == cell) & (pr3["rbp"].isin(rbp_order))].set_index(
            "rbp").reindex(rbp_order)
        ax.errorbar(d["beta"], y_pos + off,
                    xerr=[d["beta"] - d["ci_lo"], d["ci_hi"] - d["beta"]],
                    fmt="o", color=color, ecolor=color, markersize=4,
                    capsize=2, label=cell, lw=1.0)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_yticks(y_pos); ax.set_yticklabels(rbp_order, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("β of single-RBP binding on lRPF")
    ax.set_title("C  Single-RBP associations\nwith ncORF ribosome occupancy",
                 loc="left")
    hcell, lcell = ax.get_legend_handles_labels()

    # 4D — GC → property (F4)
    ax = axes[1, 1]
    gc = read_result("followup4_gc_vs_property.csv")
    pretty = {"mean_plddt": "pLDDT", "catgranule2_llps_score": "catGRANULE\n(× 100)",
              "foldable": "foldable\n(log OR)",
              "llps_prone": "LLPS-prone\n(log OR)"}
    outcomes = ["mean_plddt", "catgranule2_llps_score", "foldable", "llps_prone"]
    y_pos = np.arange(len(outcomes))
    for cell, color, off in [("K562", K_COL, -0.18), ("HepG2", H_COL, 0.18)]:
        vals, lo_vals, hi_vals = [], [], []
        for o in outcomes:
            r = gc[(gc.cell == cell) & (gc.outcome == o)].iloc[0]
            scale = 0.1
            if o == "catgranule2_llps_score":
                scale = 0.1 * 100
            vals.append(r["gc_beta"] * scale)
            lo_vals.append(r["ci_lo"] * scale)
            hi_vals.append(r["ci_hi"] * scale)
        ax.errorbar(vals, y_pos + off,
                    xerr=[np.array(vals) - np.array(lo_vals),
                          np.array(hi_vals) - np.array(vals)],
                    fmt="o", color=color, ecolor=color, markersize=5,
                    capsize=2, label=cell, lw=1.2)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_yticks(y_pos); ax.set_yticklabels([pretty[o] for o in outcomes])
    ax.invert_yaxis()
    ax.set_xlabel("effect per 0.1 GC unit")
    ax.set_title("D  Transcript GC associations\n"
                 "Predicted structural properties", loc="left")

    for a_ in axes.ravel():
        a_.title.set_fontsize(8)
    fig.legend(hcell, lcell, loc="lower center", ncols=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, "figure4")
    plt.close()



if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    INPUT_HASHES[str(Path(__file__).relative_to(ROOT.parent))] = GENERATOR_HASH
    for rel, digest in INPUT_HASHES.items():
        assert hashlib.sha256((ROOT.parent / rel).read_bytes()).hexdigest() == digest, rel
    outputs = {str((FIG / f"figure{i}.{ext}").relative_to(ROOT.parent)):
               hashlib.sha256((FIG / f"figure{i}.{ext}").read_bytes()).hexdigest()
               for i in range(1, 5) for ext in ("png", "pdf")}
    (FIG / "figure_manifest.json").write_text(json.dumps({
        "inputs": INPUT_HASHES, "outputs": outputs,
        "note": "Content hashes, not modification times, determine currency."
    }, indent=2) + "\n")
    print(f"\nAll figures written to {FIG}/")

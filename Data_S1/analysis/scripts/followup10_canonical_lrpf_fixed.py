"""Canonical-CDS occupancy regression with transcript abundance from RSEM.

TPM is read from the complete ENCODE RSEM quantifications, not a table
restricted to ncORF host transcripts. The aggregation rule matches
step1_expression_report.py: sum within versionless ENST groups, then take
the mean across the two replicates.

Model:
    log2(in_frame_RPF + 1) ~ log1p(unique_RBP) + log(AA_length) + GC + log1p(TPM)
    OLS + HC3, TPM >= 1, per cell line.

No dummy values; failures raise.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PER = RES / "followup2_canonical_cds_per_cds.csv"
RNASEQ = ROOT / "data" / "rnaseq"
OUT = RES / "followup10_canonical_lrpf_fixed.csv"
QUANT = {
    "K562":  ["K562_polyA_main_ENCSR000AEO_rep1_ENCFF720TFN.tsv",
              "K562_polyA_main_ENCSR000AEO_rep2_ENCFF522DDA.tsv"],
    "HepG2": ["HepG2_polyA_main_ENCSR985KAT_rep1_ENCFF458NJB.tsv",
              "HepG2_polyA_main_ENCSR985KAT_rep2_ENCFF371KBB.tsv"],
}


def tpm_map(fname):
    path = RNASEQ / fname
    if not path.exists():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, sep="\t", usecols=["transcript_id", "TPM"])
    d = d[d.transcript_id.astype(str).str.startswith("ENST")].copy()
    d["base"] = d.transcript_id.str.split(".").str[0]
    return d.groupby("base")["TPM"].sum().to_dict()


def mean_tpm(files):
    maps = [tpm_map(f) for f in files]
    keys = set(maps[0])
    for m in maps[1:]:
        keys &= set(m)
    if not keys:
        raise RuntimeError("no shared ENST between replicates")
    return {k: float(np.mean([m[k] for m in maps])) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cds", type=Path, default=PER,
                    help="per-CDS table to model (default: the legacy-length "
                         "selection written by followup2_canonical_cds.py)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where to write the coefficient table")
    args = ap.parse_args()
    if not args.per_cds.exists():
        raise SystemExit(f"{args.per_cds} does not exist — run "
                         "followup2_canonical_cds.py first")
    print(f"per-CDS table: {args.per_cds.name}", flush=True)
    cds = pd.read_csv(args.per_cds)
    convention = sorted(set(cds.get("cds_length_convention", ["legacy"])))
    if len(convention) != 1:
        raise SystemExit(f"table mixes length conventions: {convention}")
    print(f"CDS length convention: {convention[0]}", flush=True)
    cds["base"] = cds.enst.str.split(".").str[0]
    rows = []
    for cell in QUANT:
        m = mean_tpm(QUANT[cell])
        cds[f"tpm_fix_{cell}"] = cds["base"].map(m)
        old_n = int((cds[f"tpm_{cell}"] >= 1).sum())
        d = cds.copy()
        d["tpm"] = d[f"tpm_fix_{cell}"]
        d["nrbp"] = d[f"{cell}_nrbp"]
        d["rpf"] = d[f"{cell}_rpf_in"]
        d = d.dropna(subset=["tpm", "nrbp", "rpf", "gc", "aa_length"])
        d = d[d["tpm"] >= 1].copy()
        d["lrpf"] = np.log2(d["rpf"] + 1)
        d["l1p_rbp"] = np.log1p(d["nrbp"])
        d["l_cdslen"] = np.log(d["aa_length"])
        d["l1p_tpm"] = np.log1p(d["tpm"])
        fit = smf.ols("lrpf ~ l1p_rbp + l_cdslen + gc + l1p_tpm",
                      data=d).fit(cov_type="HC3")
        red = smf.ols("lrpf ~ l_cdslen + gc + l1p_tpm",
                      data=d).fit(cov_type="HC3")
        lo, hi = fit.conf_int().loc["l1p_rbp"]
        rows.append(dict(cell=cell, n=len(d), n_followup2=old_n,
                         beta=fit.params["l1p_rbp"], ci_lo=lo, ci_hi=hi,
                         p=fit.pvalues["l1p_rbp"], full_R2=fit.rsquared,
                         semi_partial_R2=fit.rsquared - red.rsquared,
                         median_nrbp=float(d.nrbp.median()),
                         median_tpm=float(d.tpm.median())))
        print(f"{cell}: n={len(d)} (followup2 used {old_n})", flush=True)
    out = pd.DataFrame(rows)
    # Only record the convention when it is not the default. Adding a column
    # to the legacy output would change its established schema unnecessarily.
    if convention[0] != "legacy":
        out["cds_length_convention"] = convention[0]
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\n--- canonical CDS lRPF, TPM rebuilt from source ---")
    print(out.to_string(index=False))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

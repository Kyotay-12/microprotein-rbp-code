"""Pre-Step-1: fix the main analysis population and report subclass composition.

All numbers are aggregated directly from results/step0_properties.csv (no
hardcoding). Reports, with per-filter n attrition logged:

  (1) n after  ≤100aa & S2(replicated)  filter
  (2) structure_group (foldable/intermediate/disordered) counts + % in that pop
  (3) LLPS-prone (catGRANULE >= 0.5) count + % in that pop
  (4) reference: S2 overall (no length cap) — same three breakdowns

Purpose: see how much the ≤100aa cap selectively removes the foldable subclass,
which drives the power of the minority-enrichment test in Step 1.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "step0_properties.csv"
LLPS_THRESHOLD = 0.5
LEN_CAP = 100
STRUCT_ORDER = ["foldable", "intermediate", "disordered"]


def read_step0():
    # skip the leading provenance comment lines (start with '#')
    n_comment = 0
    with open(CSV) as fh:
        for line in fh:
            if line.startswith("#"):
                n_comment += 1
            else:
                break
    df = pd.read_csv(CSV, skiprows=n_comment)
    return df


def composition(df, label):
    n = len(df)
    print(f"\n--- {label}  (n={n}) ---")
    if n == 0:
        print("  (empty)")
        return
    # structure groups
    print("  structure_group:")
    vc = df["structure_group"].value_counts()
    for g in STRUCT_ORDER:
        c = int(vc.get(g, 0))
        print(f"    {g:13s} {c:5d}  ({100*c/n:5.1f}%)")
    # any group not in the expected set?
    extra = set(vc.index) - set(STRUCT_ORDER)
    for g in sorted(extra):
        c = int(vc.get(g, 0))
        print(f"    {g:13s} {c:5d}  ({100*c/n:5.1f}%)  [unexpected label]")
    # LLPS-prone (recompute from score, don't trust precomputed bool)
    prone = (df["catgranule2_llps_score"] >= LLPS_THRESHOLD).sum()
    print(f"  LLPS-prone (catGRANULE>={LLPS_THRESHOLD}): {int(prone)} "
          f"({100*prone/n:.1f}%)")


def main():
    df = read_step0()
    print(f"source: {CSV.name}")
    print("\n=== filter attrition ===")
    n0 = len(df)
    print(f"  all ORFs:                 {n0}")
    s2 = df[df["source_sheet"] == "S2"]
    print(f"  after S2 (replicated):    {len(s2)}   (-{n0 - len(s2)})")
    main_pop = s2[s2["aa_length"] <= LEN_CAP]
    print(f"  after & aa_length<={LEN_CAP}:    {len(main_pop)}   (-{len(s2) - len(main_pop)})")

    # (1)(2)(3) main population
    composition(main_pop, f"(1-3) MAIN population: S2 & <={LEN_CAP}aa")
    # (4) reference: S2 overall, no length cap
    composition(s2, "(4) REFERENCE: S2 overall (no length cap)")

    # extra context: how the dropped (S2 & >100aa) ORFs are composed
    dropped = s2[s2["aa_length"] > LEN_CAP]
    composition(dropped, f"(extra) DROPPED by length cap: S2 & >{LEN_CAP}aa")


if __name__ == "__main__":
    main()

"""Followup 8c — per-RBP x per-transcript eCLIP binding matrix.

Step 1 only stored the NUMBER of unique RBPs per ORF. The Tier-0 perturbation
test needs to know WHICH RBP binds WHICH host transcript, so that each
knockdown can be matched to the transcripts that RBP actually binds.

Rebuilt with the same rules as scripts/step1_eclip_intersect.py:
  - transcript exons only (introns excluded), strand must match, 1 bp overlap
  - reproducible peaks only (the BED files already in data/eclip/<cell>/)

Output: results/followup8_binding_matrix.parquet  (cell, rbp, enst)
Failures raise.
"""
from pathlib import Path
import pandas as pd
import pyranges as pr

ROOT = Path(__file__).resolve().parent.parent
EXONS = ROOT / "results" / "v35_orf_exons.parquet"
ECLIP = ROOT / "data" / "eclip"
OUT = ROOT / "results" / "followup8_binding_matrix.parquet"
CELLS = ["K562", "HepG2"]


def main():
    ex = pd.read_parquet(EXONS)
    exr = pr.PyRanges(ex)
    print(f"exon intervals {len(ex)}  transcripts {ex.enst.nunique()}")

    rows = []
    for cell in CELLS:
        beds = sorted((ECLIP / cell).glob("*.bed.gz"))
        if not beds:
            raise FileNotFoundError(f"no eCLIP beds for {cell}")
        for i, f in enumerate(beds, 1):
            rbp = f.name.split("__")[0].upper()
            d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 5],
                            names=["Chromosome", "Start", "End", "Strand"],
                            dtype={0: str})
            if d.empty:
                continue
            hits = pr.PyRanges(d).join(exr, strandedness="same").df
            if hits.empty:
                continue
            for e in hits["enst"].unique():
                rows.append((cell, rbp, e))
            if i % 40 == 0:
                print(f"  {cell} {i}/{len(beds)}", flush=True)
        print(f"{cell}: {len(beds)} eCLIP files processed", flush=True)

    m = pd.DataFrame(rows, columns=["cell", "rbp", "enst"]).drop_duplicates()
    if m.empty:
        raise RuntimeError("empty binding matrix")
    m.to_parquet(OUT, index=False)
    print(f"\nbinding pairs {len(m)}")
    print(m.groupby("cell").agg(rbps=("rbp", "nunique"),
                                transcripts=("enst", "nunique")).to_string())
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

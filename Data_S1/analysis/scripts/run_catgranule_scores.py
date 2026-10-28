"""Compute catGRANULE 2.0 ROBOT per-protein LLPS propensity scores (sequence-only,
only_pc=True / RandomForest) for a FASTA of ORFs.

Runs inside the catGRANULE venv. Must be launched with cwd = the catGRANULE2.0
repo dir because that code hardcodes relative paths './src/...'.

Skips the expensive per-residue profile loop: the protein-level LLPS score comes
only from the 82 mean physico-chemical features + RandomForest, which is what we
need for group classification.

Usage:
    python run_catgranule_scores.py <input.fasta> <output.csv>

Stops (raises) on any NaN score; missing predictions are not substituted.
"""
import os
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO

# This code hardcodes relative paths './src/...', so it must run with cwd = the
# catGRANULE2.0 repo dir. Ensure that dir is importable.
sys.path.insert(0, os.getcwd())

from compute_profiles_and_predictions import (
    get_physical_chemical_properties,
    predict,
    correct_order_columns,
)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_catgranule_scores.py <input.fasta> <output.csv>")
    fasta, out_csv = sys.argv[1], sys.argv[2]

    records = list(SeqIO.parse(fasta, "fasta"))
    ids = [r.id for r in records]
    sequences = [str(r.seq) for r in records]
    if not sequences:
        raise ValueError(f"No sequences parsed from {fasta}")

    scales_dir = "./src/ChemicalPhysicalScales_Py_dictionary"
    classifiers_dir = "./src/TRAINED_MODELS/"

    pc_df = get_physical_chemical_properties(sequences, ids, scales_dir)
    pc_df = pc_df[correct_order_columns[:82]]

    if pc_df.isnull().values.any():
        n_bad = int(pc_df.isnull().any(axis=1).sum())
        bad_ids = pc_df.index[pc_df.isnull().any(axis=1)].tolist()[:20]
        raise ValueError(
            f"{n_bad} ORFs have NaN physico-chemical features (feature computation "
            f"failed). First offenders: {bad_ids}. Stopping per no-dummy-fill rule."
        )

    scores = predict(pc_df, classifiers_dir, only_pc=True)["RandomForest"]
    scores = np.asarray(scores, dtype=float)

    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        raise ValueError(f"{n_nan} NaN LLPS scores produced. Stopping (no dummy fills).")

    out = pd.DataFrame({"orf_name": ids, "catgranule2_llps_score": scores})
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} LLPS scores to {out_csv}", flush=True)
    print(
        f"  score range: min={scores.min():.3f} median={np.median(scores):.3f} "
        f"max={scores.max():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

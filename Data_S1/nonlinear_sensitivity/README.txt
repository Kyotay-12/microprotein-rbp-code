Nonlinear adjustment of TPM and host-RNA length

This code evaluates whether an association between host-RBP count and ncORF
ribosome occupancy is sensitive to nonlinear adjustment of TPM and mature RNA
length. It is not a self-contained data distribution.

Use Python 3.12 and the dependency versions in requirements.txt. Supply both:
  1. A processed ORF-record JSON file.
  2. The intact nonlinear reference distribution, containing file_manifest.json,
     results/cohort_membership.csv, validation/baseline_reference.json and all
     other files listed in that manifest.

The inputs and reference distribution must be obtained separately. The launcher
requires exact input checksums and ordered cohort membership, verifies every
reference file, and stops rather than substituting missing data or references.

From the Data_S1 directory, with your actual paths:

  python3 -B nonlinear_sensitivity/code/run.py --help
  python3 -B nonlinear_sensitivity/code/run.py --check --input-file INPUT_JSON --reference-dir REFERENCE_DIRECTORY
  python3 -B nonlinear_sensitivity/code/run.py --input-file INPUT_JSON --reference-dir REFERENCE_DIRECTORY --output-dir NEW_OUTPUT_DIRECTORY

The check command validates files, dependencies and cohort membership without
model fitting. The full command refits the baseline and nonlinear models,
generates results and figures, and compares all reference CSV tables. Outputs
must be new and separate from code, input and reference directories. Existing
outputs are not overwritten. Do not use python -O.

specification.txt describes the models, spline construction, cluster inference
and sensitivity analyses. This is refitting of processed observations, not
independent reconstruction from raw sequencing data. Figure bytes are not a
numerical reproducibility criterion. See the archive-root licence information.

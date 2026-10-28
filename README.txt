Microprotein host-RNA analysis: source code

Kyota Yasuda and Masashi Fujii

This distribution contains analysis code, methodological specifications,
dependency lists and source-dataset references. It contains no observations,
fitted result tables, completed figures, manuscript or correspondence.
Data and reference results must be supplied separately. Code alone cannot
reproduce a numerical result without those inputs. Do not run scripts in
filename order or use Python's -O option, which disables validation assertions.

VERIFY THE CODE

  python3 -B verify_code.py

This checks the code distribution's inventory and hashes, not model validity.

ENVIRONMENTS

Data_S1/analysis/environments/ contains the recorded upstream environments.
Data_S1/posthoc/requirements.txt covers the derived-data model refits.
Data_S1/posthoc/local_rbp_identity/requirements_original.txt records the local
identity fitting environment. Data_S1/nonlinear_sensitivity/requirements.txt
records the nonlinear sensitivity environment. These lists are not one
interchangeable environment lockfile. Third-party software is not bundled.

MODEL REFITS WITH SEPARATE DATA

DATA_DIR below means the root of an intact, separately supplied Data S1
directory, with its original file_manifest.json. OUTPUT_DIR must be outside
both the code checkout and the supplied data directory. Choose new output
paths. Do not merge private data into a public checkout or commit outputs.
Set DATA_DIR and OUTPUT_DIR to your actual paths and create OUTPUT_DIR itself
before running the commands below. Each named report or
analysis output subdirectory must not already exist.

  python3 -B Data_S1/posthoc/rerun_corrected.py \
    --data-root "$DATA_DIR" --report "$OUTPUT_DIR/model_check.json"

This refits occupancy, host-length, predicted-property, corrected canonical-CDS
and within-host models. Required observations and reference results are checked
against the supplied private inventory. Missing or altered inputs stop the run.

  python3 -B Data_S1/posthoc/local_rbp_identity/replay.py \
    --root "$DATA_DIR/posthoc/local_rbp_identity" \
    --report "$OUTPUT_DIR/local_identity_check.json"

  python3 -B Data_S1/nonlinear_sensitivity/code/run.py \
    --input-file "$DATA_DIR/posthoc/inputs/main_orf_records.json" \
    --reference-dir "$DATA_DIR/nonlinear_sensitivity" --check

  python3 -B Data_S1/nonlinear_sensitivity/code/run.py \
    --input-file "$DATA_DIR/posthoc/inputs/main_orf_records.json" \
    --reference-dir "$DATA_DIR/nonlinear_sensitivity" \
    --output-dir "$OUTPUT_DIR/nonlinear_refit"

For local identity and nonlinear analyses, the reference directory must remain
the complete original distribution: its code, input, specification and result
hashes are independently checked. The executed launcher is the source-code
copy in this repository. Do not replace historical reference files to make a
comparison pass. See the nonlinear README for the command options.

FIGURE CODE

Plotting code reads separately supplied results; it does not fit models or
substitute example data. The corrected canonical panel and paired/length
panels can be generated with:

  python3 -B Data_S1/posthoc/make_figure4_corrected.py \
    --project-root "$DATA_DIR" --outdir "$OUTPUT_DIR/figure4"
  python3 -B Data_S1/posthoc/make_figure5_corrected.py \
    --project-root "$DATA_DIR" --outdir "$OUTPUT_DIR/figure5"

Other original upstream plotting scripts use relative data/output locations.
They require a separate private working directory with the listed inputs;
they are not a data-free quick-start command. Finished plots are not included.

UPSTREAM RECONSTRUCTION

The derived-data refits above are not reprocessing of raw sequencing data.
Data_S1/sources.txt identifies the source datasets, conventions and required
external resources. Upstream preparation additionally requires annotation,
sequence, peak, expression and ribosome-profile inputs and their intermediate
tables. Those inputs are not downloaded by the refit commands above.

Some raw-data producers require MICROPROTEIN_UPSTREAM_ROOT. Additional input
locations are read from the JSON file named by MICROPROTEIN_INPUT_PATHS.
Its keys are host_transcript_metadata, legacy_within_host_records,
legacy_within_host_results, structural_length_results,
structural_length_validation, and local_window_results. Each value must be
the actual absolute file or directory path supplied for that execution.

MICROPROTEIN_VALIDATION_REFERENCE names a separate JSON validation record.
Its counts object must contain the independently established main_orfs and
legacy_canonical counts. Missing references are errors; there are no embedded
empirical count defaults. Such records and upstream intermediates are supplied
with the private reconstruction inputs, not this source-code distribution.

Older selection and model implementations are retained where required by the
comparison/reconstruction code. Use the corrected entry points above rather
than treating filename order as an execution order. Model formulas, thresholds,
post-hoc status, multiplicity families and limitations are retained in the
methodological specifications. No preregistration is asserted.

LICENSING

Original code is MIT licensed. Method documentation is CC BY 4.0. See
LICENSES.txt and Data_S1/LICENSE-CODE / LICENSE-CONTENT. Data and third-party
software retain their own terms and citation requirements. CITATION.cff
identifies the code authors; cite the exact code version used.

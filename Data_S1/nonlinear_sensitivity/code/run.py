"""Refit nonlinear occupancy sensitivity with separately supplied data and references."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent


def verify_files(input_file, reference_dir):
    if not reference_dir.is_dir():
        raise ValueError('The separately supplied reference directory is missing.')
    manifest = json.loads((reference_dir / 'file_manifest.json').read_text())
    records = manifest['files']
    actual = {p.relative_to(reference_dir).as_posix() for p in reference_dir.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(records) | {'file_manifest.json'}:
        raise ValueError('The nonlinear file inventory differs from the distribution.')
    for relative, record in records.items():
        path = reference_dir / relative
        if path.is_symlink() or not path.resolve().is_relative_to(reference_dir):
            raise ValueError(f'Unsafe path: {relative}')
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Changed file: {relative}')
    expected = manifest['shared_input']
    if not input_file.is_file() or input_file.is_symlink():
        raise ValueError('The processed ORF input is missing or is a symbolic link.')
    if input_file.stat().st_size != expected['bytes'] or hashlib.sha256(input_file.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('The processed ORF input differs from the distributed observations.')


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: scientific validation assertions must remain active.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check files and cohorts without fitting models')
    parser.add_argument('--input-file', type=Path, required=True, help='Separately supplied processed ORF observations (JSON)')
    parser.add_argument('--reference-dir', type=Path, required=True, help='Separately supplied intact nonlinear reference distribution, including its manifest, results and validation')
    parser.add_argument('--output-dir', type=Path, help='A new directory outside the code, data and reference directories')
    args = parser.parse_args()
    if args.input_file.is_symlink() or args.reference_dir.is_symlink():
        raise ValueError('Input and reference paths must not be symbolic links.')
    input_file = args.input_file.resolve()
    reference_dir = args.reference_dir.resolve()
    if reference_dir == PACKAGE or reference_dir.is_relative_to(PACKAGE):
        raise ValueError('Reference data must be supplied separately from this code-only distribution.')
    verify_files(input_file, reference_dir)
    output = args.output_dir.resolve() if args.output_dir else None
    if not args.check:
        if output is None:
            parser.error('--output-dir is required for model fitting')
        for protected in [PACKAGE, input_file.parent, reference_dir]:
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError('Output must be separate from the distributed code and data.')
        if output.exists():
            raise FileExistsError('Output already exists; choose a new directory.')
    sys.dont_write_bytecode = True
    os.environ['MICROPROTEIN_INPUT_FILE'] = str(input_file)
    os.environ['MICROPROTEIN_REFERENCE_DIR'] = str(reference_dir)
    if output is not None:
        os.environ['MICROPROTEIN_OUTPUT_DIR'] = str(output)
    import baseline_reproduction as baseline
    import nonlinear_robustness as analysis
    import verify_results
    import numpy, pandas, scipy, statsmodels, patsy, matplotlib
    counts = baseline.verify_cohort()
    info = {'cohorts': counts, 'model_fitting': not args.check,
            'python': sys.version.split()[0],
            'versions': {m.__name__: m.__version__ for m in
                         [numpy, pandas, scipy, statsmodels, patsy, matplotlib]}}
    if args.check:
        print(json.dumps({'status': 'PASS', **info}, indent=2))
        return
    output.mkdir()
    shutil.copyfile(ROOT / 'specification.txt', output / 'protocol.md')
    baseline.main()
    analysis.main()
    report = verify_results.verify(output, reference_dir / 'results')
    verify_files(input_file, reference_dir)
    (output / 'reproduction_check.json').write_text(json.dumps({**report, **info}, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'csv_files': len(report['comparisons']),
                      'output_directory': str(output)}, indent=2))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='microprotein-nonlinear-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', str(Path(cache) / 'matplotlib'))
        os.environ.setdefault('XDG_CACHE_HOME', str(Path(cache) / 'cache'))
        main()

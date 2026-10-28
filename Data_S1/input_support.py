"""Require separately supplied input locations and cohort-validation metadata."""
from pathlib import Path
import hashlib
import json
import os


def _read_config(variable):
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f'{variable} must identify a separately supplied JSON file.')
    path = Path(value).expanduser().resolve(strict=True)
    with path.open() as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError(f'{variable} must contain a JSON object.')
    return record


def expected_count(key):
    counts = _read_config('MICROPROTEIN_VALIDATION_REFERENCE')['counts']
    value = counts[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'Invalid reference count for {key}')
    return value


def input_path(key):
    paths = _read_config('MICROPROTEIN_INPUT_PATHS')
    value = paths[key]
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f'Input location {key} must be an explicit absolute path.')
    return Path(value).resolve(strict=True)


def verify_data_package(root):
    """Validate a complete, separately supplied Data S1 without modifying it."""
    root = Path(root).resolve(strict=True)
    manifest = root / 'file_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('The supplied data directory needs file_manifest.json.')
    records = json.loads(manifest.read_text())['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(records) | {'file_manifest.json'}:
        raise ValueError('Separate data package inventory differs from its manifest.')
    for name, record in records.items():
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe data path: {name}')
        if p.stat().st_size != record['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Changed data/reference file: {name}')
    return root

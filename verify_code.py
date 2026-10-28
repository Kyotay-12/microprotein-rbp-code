"""Check the source-code distribution; no empirical inputs are required."""
from pathlib import Path
import ast
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    records = json.loads((root/'code_manifest.json').read_text())['files']
    expected = set(records) | {'code_manifest.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(root).parts}
    if actual != expected:
        raise ValueError(('Unexpected or missing file', sorted(actual ^ expected)))
    for name, item in records.items():
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError(('Unsafe path', name))
        if p.stat().st_size != item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(('Changed file', name))
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=name)
        if p.suffix == '.json':
            json.loads(p.read_text())
    print(f'PASS: {len(records)} source/method files match the code inventory.')
    print('This is a distribution-integrity check, not a model reproduction result.')


if __name__ == '__main__':
    main()

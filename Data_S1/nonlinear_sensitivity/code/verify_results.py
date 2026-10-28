"""Compare regenerated result tables with separately supplied reference tables."""
from pathlib import Path
import numpy as np
import pandas as pd


def verify(output, reference):
    references = sorted(Path(reference).glob('*.csv'))
    if not references:
        raise ValueError('No reference result tables found.')
    comparisons = []
    for expected_path in references:
        actual = pd.read_csv(Path(output) / expected_path.name)
        expected = pd.read_csv(expected_path)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-8, atol=1e-9)
        maximum = 0.0
        for column in expected:
            a, b = actual[column], expected[column]
            if not pd.api.types.is_numeric_dtype(a) or pd.api.types.is_bool_dtype(a):
                continue
            mask = a.notna()
            if not np.array_equal(mask, b.notna()):
                raise ValueError((expected_path.name, column, 'Missingness differs'))
            x, y = a[mask].to_numpy(), b[mask].to_numpy()
            if column == 'p' or column.startswith('p_'):
                if not np.allclose(x, y, rtol=1e-8, atol=0):
                    raise ValueError((expected_path.name, column, 'Probability differs'))
            if len(x):
                maximum = max(maximum, float(np.max(np.abs(x - y))))
        comparisons.append({'file': expected_path.name, 'rows': len(actual),
                            'max_absolute_difference': maximum, 'status': 'PASS'})
    return {'status': 'PASS', 'comparisons': comparisons,
            'scope': 'Refitting processed observations; not independent raw-data reconstruction.'}

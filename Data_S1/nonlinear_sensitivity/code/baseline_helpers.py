"""Input validation, reference comparison and cluster-robust baseline models."""
import json
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def load_records(path):
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"No records in {path.name}")
    d = pd.DataFrame(records)
    if not d.orf_name.is_unique:
        raise ValueError(f"Nonunique ORF identifiers in {path.name}")
    return d

def compare_models(actual, expected, keys, label):
    actual_by_key = {tuple(r[k] for k in keys): r for r in actual}
    expected_by_key = {tuple(r[k] for k in keys): r for r in expected}
    if len(actual_by_key) != len(actual) or len(expected_by_key) != len(expected) or set(actual_by_key) != set(expected_by_key):
        raise ValueError((label, "Nonunique or changed model membership"))
    count, maximum = 0, 0.0
    def equal(x, y, location):
        nonlocal count, maximum
        if isinstance(x, dict):
            if not isinstance(y, dict) or set(x) != set(y):
                raise ValueError((location, "Changed result fields"))
            for key in x:
                equal(x[key], y[key], location + (key,))
        elif isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                raise ValueError((location, "Changed ordered values"))
            for i, (left, right) in enumerate(zip(x, y)):
                equal(left, right, location + (str(i),))
        elif isinstance(x, (bool, str)) or x is None:
            if x != y:
                raise ValueError((location, x, y))
        elif isinstance(x, (int, float, np.number)):
            if not isinstance(y, (int, float)) or isinstance(y, bool) or not math.isfinite(float(x)) or not math.isfinite(float(y)):
                raise ValueError((location, "Invalid numeric value", x, y))
            field = location[-1]
            probability = field == "p" or field.startswith("p_") or "tost_p" in field or "holm_p" in field
            integer_field = field.startswith("n_") or field == "n" or field in {"rank_X", "t_df", "t_inference_df", "design_columns", "df_inference"}
            okay = (x == y) if integer_field else bool(np.isclose(x, y, rtol=1e-8, atol=0.0 if probability else 1e-10))
            if not okay:
                raise ValueError((location, x, y))
            count += 1
            maximum = max(maximum, abs(float(x) - float(y)))
        else:
            raise TypeError((location, type(x)))
    for key, row in actual_by_key.items():
        equal(row, expected_by_key[key], (label,) + key)
    return {"analysis": label, "models": len(actual), "numeric_comparisons": count, "maximum_absolute_difference": maximum, "status": "PASS"}

def fit_model(w, terms, label, cell):
    model = smf.ols('y ~ ' + terms, data=w, missing='raise')
    assert model.nobs == len(w)
    X, y = model.exog, model.endog
    assert np.linalg.matrix_rank(X) == X.shape[1], f'Rank deficient: {cell} {label}'
    ols = model.fit()
    direct_beta = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(direct_beta, ols.params.to_numpy(), atol=1e-10, rtol=1e-10)
    j = model.exog_names.index('ln_rbp')
    Z = np.delete(X, j, axis=1)
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    bx = np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
    ry, rx = y-Z@by, X[:, j]-Z@bx
    partial_beta = float(rx@ry/(rx@rx))
    assert np.isclose(partial_beta, direct_beta[j], atol=1e-10)
    tss = float(((y-y.mean())**2).sum())
    reduced_sse = float(ry@ry)
    incr_r2 = (reduced_sse-float(ols.ssr))/tss
    vx = float(((X[:, j]-X[:, j].mean())**2).sum()/(rx@rx))
    records = []
    for cluster in ['tx', 'gene']:
        f = model.fit(cov_type='cluster', cov_kwds={'groups': w[cluster], 'use_correction': True, 'df_correction': True}, use_t=True)
        assert f.df_resid_inference == w[cluster].nunique()-1
        lo, hi = f.conf_int().loc['ln_rbp']
        records.append({
            'cell': cell, 'model': label, 'cluster': cluster,
            'n_ORFs': len(w), 'n_transcripts': int(w.tx.nunique()), 'n_genes': int(w.gene.nunique()),
            'beta': float(f.params['ln_rbp']), 'ci95_lo': float(lo), 'ci95_hi': float(hi),
            'SE': float(f.bse['ln_rbp']), 'p': float(f.pvalues['ln_rbp']),
            't_inference_df': int(f.df_resid_inference),
            'full_R2': float(f.rsquared), 'incremental_R2_RBP': incr_r2, 'RBP_VIF': vx,
            'length_beta': float(f.params['ln_tx_length']) if 'ln_tx_length' in f.params else None,
            'length_ci95': f.conf_int().loc['ln_tx_length'].tolist() if 'ln_tx_length' in f.params else None,
            'formula': 'y ~ ' + terms,
        })
    return records


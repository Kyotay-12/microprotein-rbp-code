"""Independent re-analysis: original sources, NumPy OLS/sandwich, manual Holm.

No import of the primary analysis or statsmodels. Patsy supplies only a fixed
cubic B-spline basis. This validation does not modify scientific source files.
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from patsy import dmatrix

PACKAGE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Data_S1')
sys.path.insert(0, str(PACKAGE_ROOT))
from input_support import expected_count, input_path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'properties': ROOT/'analysis/results/step0_properties.csv',
    'binding': ROOT/'analysis/results/step1_binding.csv',
    'expression': ROOT/'analysis/results/step1_expression.csv',
    'catalogue': ROOT/'analysis/data/Mudge2022_SupplementaryTables.xlsx',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read_sources():
    # Index joins preserve the original ORF keys and expose duplicate IDs.
    properties = pd.read_csv(SOURCES['properties'], comment='#').set_index('orf_name', verify_integrity=True)
    properties = properties.loc[(properties.source_sheet == 'S2') & (properties.aa_length <= 100)].copy()
    catalog = pd.read_excel(SOURCES['catalogue'], sheet_name='S2. PHASE I Ribo-seq ORFs', usecols=['orf_name', 'gene_id']).set_index('orf_name', verify_integrity=True)
    properties['gene'] = catalog['gene_id'].reindex(properties.index).str.split('.').str[0]
    properties['tx'] = properties.transcript.str.split('.').str[0]
    assert properties.gene.notna().all()
    for kind, cols in [('binding', ['K562_nrbp_tx', 'HepG2_nrbp_tx']),
                       ('expression', ['tpm_K562_polyA', 'tpm_HepG2_polyA'])]:
        table = pd.read_csv(SOURCES[kind]).set_index('orf_name', verify_integrity=True)
        assert properties.index.isin(table.index).all()
        for col in cols:
            properties[col] = table[col].reindex(properties.index)
    length = pd.read_csv(SOURCES['length']).set_index('enst', verify_integrity=True)
    assert set(length.index) == set(properties.tx)
    properties['nt_length'] = properties.tx.map(length.spliced_length_nt)
    assert properties.nt_length.notna().all() and properties.nt_length.gt(0).all()
    assert properties.tx.map(length.ensg).eq(properties.gene).all()
    assert len(properties) == expected_count('main_orfs')
    return properties


def zs(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    sd = np.sqrt(np.sum((a-a.mean())**2)/(len(a)-1))
    assert sd > 0
    return (a-a.mean())/sd


def fit(frame, outcome, model, scheme, cell):
    w = frame.dropna(subset=[outcome]).copy()
    y = zs(w[outcome])
    xr = zs(np.log1p(w.rbp))
    zhost = zs(np.log(w.nt_length))
    X = np.column_stack([np.ones(len(w)), xr, zs(np.log(w.aa_length)),
                         zs(w.transcript_gc), zs(np.log1p(w.tpm))])
    if model == 'plus_log_length':
        X = np.column_stack([X, zhost])
    elif model == 'plus_length_spline':
        spline = np.asarray(dmatrix('0 + bs(host, df=4, degree=3, include_intercept=False)', {'host': zhost}))
        X = np.column_stack([X, spline])
    n, k = X.shape
    assert np.linalg.matrix_rank(X) == k
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y-X@coef
    bread = np.linalg.inv(X.T@X)
    if scheme == 'HC3':
        leverage = np.einsum('ij,jk,ik->i', X, bread, X)
        scores = X*(resid/(1-leverage))[:, None]
        covariance = bread@(scores.T@scores)@bread
        dfree, ncluster = n-k, None
    else:
        groups, inverse = np.unique(w.gene if scheme == 'gene' else w.tx, return_inverse=True)
        ncluster = len(groups)
        summed_scores = np.zeros((ncluster, k))
        np.add.at(summed_scores, inverse, X*resid[:, None])
        correction = ncluster/(ncluster-1)*(n-1)/(n-k)
        covariance = correction*bread@(summed_scores.T@summed_scores)@bread
        dfree = ncluster-1
    b, se = float(coef[1]), float(np.sqrt(covariance[1, 1]))
    lo90, hi90 = b-stats.t.ppf(.95, dfree)*se, b+stats.t.ppf(.95, dfree)*se
    lo95, hi95 = b-stats.t.ppf(.975, dfree)*se, b+stats.t.ppf(.975, dfree)*se
    pl, pu = float(stats.t.sf((b+.10)/se, dfree)), float(stats.t.cdf((b-.10)/se, dfree))
    p05 = max(float(stats.t.sf((b+.05)/se, dfree)), float(stats.t.cdf((b-.05)/se, dfree)))
    Z = np.delete(X, 1, axis=1)
    reduced_resid = y-Z@np.linalg.lstsq(Z, y, rcond=None)[0]
    sst = float(y@y)
    row = dict(cell=cell, outcome='pLDDT' if outcome == 'mean_plddt' else 'catGRANULE', model=model,
               scheme=scheme, n=n, n_cluster=ncluster, rank_X=k, t_df=dfree,
               beta_std=b, se=se, ci90_lo=float(lo90), ci90_hi=float(hi90), ci95_lo=float(lo95), ci95_hi=float(hi95),
               p_difference_normal=float(2*stats.norm.sf(abs(b/se))),
               p_difference_t=float(2*stats.t.sf(abs(b/se), dfree)),
               tost_p_lower=pl, tost_p_upper=pu, tost_p=max(pl, pu), tost_p_stringent=p05,
               full_R2=1-float(resid@resid)/sst,
               incremental_R2_RBP=float(reduced_resid@reduced_resid-resid@resid)/sst,
               ORF_ids_sha256=hashlib.sha256('\n'.join(sorted(w.index)).encode()).hexdigest())
    row['equivalent_0.10'] = bool(lo90 > -.1 and hi90 < .1)
    row['equivalent_0.05'] = bool(lo90 > -.05 and hi90 < .05)
    return row


def main():
    SOURCES['length'] = input_path('host_transcript_metadata')
    before = {key: digest(path) for key, path in SOURCES.items()}
    original = read_sources()
    rows = []
    models = ['original', 'plus_log_length', 'plus_length_spline']
    for cell in ['K562', 'HepG2']:
        d = original.copy()
        d['tpm'], d['rbp'] = d['tpm_'+cell+'_polyA'], d[cell+'_nrbp_tx']
        d = d.dropna(subset=['tpm', 'rbp', 'aa_length', 'transcript_gc'])
        single = d.loc[d.tx.map(d.tx.value_counts()).eq(1)]
        for outcome in ['mean_plddt', 'catgranule2_llps_score']:
            for model in models:
                for scheme in ['transcript', 'gene', 'single_ORF', 'HC3']:
                    rows.append(fit(single if scheme == 'single_ORF' else d, outcome, model, scheme, cell))
    # Manual Holm: sorted p_i * (m-i), followed by cumulative maxima and cap 1.
    for model in models:
        members = [r for r in rows if r['model'] == model and r['scheme'] == 'transcript']
        assert len(members) == 4
        order = sorted(range(len(members)), key=lambda j: members[j]['tost_p'])
        running = 0.
        for rank, j in enumerate(order):
            running = max(running, (len(order)-rank)*members[j]['tost_p'])
            members[j]['holm_p_primary'] = min(1., running)
            members[j]['equivalent_0.10_holm'] = bool(running < .05)
    # Only after independent calculations are complete, read comparison results.
    observed = json.loads((HERE/'results.json').read_text())
    keyed = {(r['cell'], r['outcome'], r['model'], r['scheme']): r for r in observed['model_results']}
    checks = []
    for row in rows:
        key = tuple(row[c] for c in ['cell', 'outcome', 'model', 'scheme'])
        expected = keyed[key]
        numeric_diff = {}
        for field, value in row.items():
            if isinstance(value, float):
                assert np.isclose(value, expected[field], atol=1e-10, rtol=1e-8), (key, field, value, expected[field])
                numeric_diff[field] = abs(value-expected[field])
            else:
                assert value == expected[field], (key, field, value, expected[field])
        checks.append({'key': list(key), 'status': 'PASS', 'max_absolute_numeric_difference': max(numeric_diff.values())})
    assert len(rows) == len(keyed) == 48
    for key, path in SOURCES.items():
        assert digest(path) == before[key]
    out = {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
        'method': 'Independent keyed source reconstruction, NumPy OLS, manual CR1 and HC3, SciPy t/normal inference, manual Holm; Patsy only for spline basis; no statsmodels or primary analysis import.',
        'scope': 'All 48 conditions, including original/linear/spline × 2 cells × 2 outcomes × 4 covariance/subset schemes.',
        'validator_sha256': digest(Path(__file__)), 'primary_results_sha256': digest(HERE/'results.json'),
        'specification_sha256': digest(HERE/'specification.json'), 'source_sha256': before,
        'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
        'checks': checks, 'independent_results': rows,
    }
    (HERE/'independent_validation.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print('PASS: all 48 independently reconstructed conditions match; all source hashes unchanged.')
    print('Maximum absolute numerical difference:', max(c['max_absolute_numeric_difference'] for c in checks))
    for row in rows:
        if row['scheme'] == 'transcript':
            print(row['model'], row['cell'], row['outcome'], row['beta_std'], row['ci90_lo'], row['ci90_hi'], row['holm_p_primary'])


if __name__ == '__main__':
    main()

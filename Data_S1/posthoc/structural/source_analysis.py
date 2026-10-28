"""Preserve the original TOST estimand while adding measured mature RNA length."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import gzip
import json
import re
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import patsy

PACKAGE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Data_S1')
sys.path.insert(0, str(PACKAGE_ROOT))
from input_support import expected_count, input_path

HERE = Path(__file__).resolve().parent
SPEC = HERE / 'specification.json'
CFG = json.loads(SPEC.read_text())
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT/'analysis/results'
FILES = {
    'properties': RES/'step0_properties.csv',
    'binding': RES/'step1_binding.csv',
    'expression': RES/'step1_expression.csv',
    'catalogue': ROOT/'analysis/data/Mudge2022_SupplementaryTables.xlsx',
    'fasta': ROOT/'analysis/data/gencode.v35.transcripts.fa.gz',
    'legacy_results': RES/'followup14_equivalence.csv',
    'legacy_script': ROOT/'analysis/scripts/followup14_equivalence.py',
    'legacy_spec': ROOT/'equivalence_specification.txt',
}


def digest(p):
    h = sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stable(s):
    return re.sub(r'\.\d+(?=_|$)', '', str(s))


def load():
    p = pd.read_csv(FILES['properties'], comment='#')
    assert p.orf_name.is_unique
    d = p.loc[p.source_sheet.eq('S2') & p.aa_length.le(100)].copy()
    cat = pd.read_excel(FILES['catalogue'], sheet_name='S2. PHASE I Ribo-seq ORFs', usecols=['orf_name', 'gene_id'])
    d = d.merge(cat, on='orf_name', how='left', validate='one_to_one')
    for source, cols in [('binding', ['K562_nrbp_tx', 'HepG2_nrbp_tx']),
                         ('expression', ['tpm_K562_polyA', 'tpm_HepG2_polyA'])]:
        s = pd.read_csv(FILES[source])
        d = d.merge(s[['orf_name']+cols], on='orf_name', how='left', validate='one_to_one')
    assert len(d) == expected_count('main_orfs') and d.gene_id.notna().all()
    d['tx'], d['gene'] = d.transcript.map(stable), d.gene_id.map(stable)
    # Prove that the improved version stripping does not change legacy grouping.
    assert (d.tx == d.transcript.astype(str).str.split('.').str[0]).all()
    assert (d.gene == d.gene_id.astype(str).str.split('.').str[0]).all()
    meta = pd.read_csv(FILES['length_metadata'])
    assert meta.enst.is_unique and set(meta.enst) == set(d.tx)
    d = d.merge(meta[['enst', 'ensg', 'spliced_length_nt']], left_on='tx', right_on='enst', how='left', validate='many_to_one')
    assert (d.gene == d.ensg).all()
    assert d.spliced_length_nt.notna().all() and d.spliced_length_nt.gt(0).all()
    target = set(d.tx)
    lengths, current, n = {}, None, 0
    def finish():
        if current in target:
            assert current not in lengths, f'Duplicate stable FASTA ID {current}'
            lengths[current] = n
    with gzip.open(FILES['fasta'], 'rt') as stream:
        for line in stream:
            if line.startswith('>'):
                finish()
                current = stable(line[1:].split('|')[0].split()[0])
                n = 0
            else:
                n += len(line.strip())
        finish()
    assert set(lengths) == target
    assert (d.tx.map(lengths) == d.spliced_length_nt).all()
    return d


def standardize(v):
    values = np.asarray(v, dtype=float)
    assert np.isfinite(values).all()
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    assert sd > 0 and np.isfinite(sd)
    return (values-mean)/sd, {'mean': mean, 'sd_ddof1': sd}


def prepare(frame, outcome):
    d = frame.dropna(subset=[outcome]).copy()
    columns = {
        'zy': d[outcome], 'zrbp': np.log1p(d.nrbp),
        'zlen': np.log(d.aa_length), 'zgc': d.transcript_gc,
        'ztpm': np.log1p(d.tpm), 'zhostlen': np.log(d.spliced_length_nt),
    }
    scales = {}
    for name, values in columns.items():
        d[name], scales[name] = standardize(values)
    return d, scales


def tost(beta, se, df, margin):
    lower = float(stats.t.sf((beta+margin)/se, df))
    upper = float(stats.t.cdf((beta-margin)/se, df))
    return lower, upper, max(lower, upper)


def fit_one(d, scales, outcome, cell, scheme, model_name):
    formula = CFG['models'][model_name]
    model = smf.ols(formula, data=d, missing='raise')
    X, y = model.exog, model.endog
    rank = np.linalg.matrix_rank(X)
    assert rank == X.shape[1] and len(y) == len(d)
    if scheme == 'HC3':
        fit = model.fit(cov_type='HC3', use_t=False)
        dfree, ncl = len(d)-rank, None
    else:
        groups = d.gene if scheme == 'gene' else d.tx
        fit = model.fit(cov_type='cluster', cov_kwds={'groups': groups, 'use_correction': True, 'df_correction': True}, use_t=False)
        ncl = int(groups.nunique())
        dfree = ncl-1
        assert fit.df_resid_inference == dfree
    beta, se = float(fit.params['zrbp']), float(fit.bse['zrbp'])
    b_direct = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(b_direct, fit.params, atol=1e-10, rtol=1e-10)
    j = model.exog_names.index('zrbp')
    Z = np.delete(X, j, axis=1)
    ry = y-Z@np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = X[:,j]-Z@np.linalg.lstsq(Z, X[:,j], rcond=None)[0]
    assert np.isclose(float(rx@ry/(rx@rx)), beta, atol=1e-10)
    delta_r2 = float((ry@ry-fit.ssr)/np.sum((y-y.mean())**2))
    ci90 = [float(beta-stats.t.ppf(.95,dfree)*se), float(beta+stats.t.ppf(.95,dfree)*se)]
    ci95 = [float(beta-stats.t.ppf(.975,dfree)*se), float(beta+stats.t.ppf(.975,dfree)*se)]
    p_normal = float(fit.pvalues['zrbp'])
    p_t = float(2*stats.t.sf(abs(beta/se), dfree))
    pl, pu, pt = tost(beta, se, dfree, CFG['bounds']['primary'])
    _, _, ps = tost(beta, se, dfree, CFG['bounds']['stringent'])
    eq10 = ci90[0] > -CFG['bounds']['primary'] and ci90[1] < CFG['bounds']['primary']
    eq05 = ci90[0] > -CFG['bounds']['stringent'] and ci90[1] < CFG['bounds']['stringent']
    assert eq10 == (pt < CFG['alpha']) and eq05 == (ps < CFG['alpha'])
    verdict = ('bounded near zero' if eq10 else 'inconclusive') if p_normal >= CFG['alpha'] else ('detectable but negligible' if eq10 else 'effect present')
    return {
        'model': model_name, 'cell': cell, 'outcome': CFG['outcomes'][outcome], 'scheme': scheme,
        'n': len(d), 'n_cluster': ncl, 'n_transcripts': int(d.tx.nunique()), 'n_genes': int(d.gene.nunique()),
        'rank_X': int(rank), 't_df': int(dfree), 'beta_std': beta, 'se': se,
        'ci90_lo': ci90[0], 'ci90_hi': ci90[1], 'ci95_lo': ci95[0], 'ci95_hi': ci95[1],
        'p_difference_normal': p_normal, 'p_difference_t': p_t,
        'tost_p_lower': pl, 'tost_p_upper': pu, 'tost_p': pt,
        'equivalent_0.10': bool(eq10), 'tost_p_stringent': ps, 'equivalent_0.05': bool(eq05),
        'verdict_legacy_convention': verdict,
        'full_R2': float(fit.rsquared), 'incremental_R2_RBP': delta_r2,
        'ORF_ids_sha256': sha256('\n'.join(sorted(d.orf_name)).encode()).hexdigest(),
        'scales': scales, 'formula': formula,
    }


def main():
    FILES['length_metadata'] = input_path('host_transcript_metadata')
    before = {k: {'path': str(p), 'sha256': digest(p)} for k,p in FILES.items()}
    spec_hash = digest(SPEC)
    d = load()
    legacy = pd.read_csv(FILES['legacy_results'])
    assert len(legacy) == 16
    result, coverage = [], []
    for cell in ['K562','HepG2']:
        w = d.copy()
        w['tpm'], w['nrbp'] = w[f'tpm_{cell}_polyA'], w[f'{cell}_nrbp_tx']
        w = w.dropna(subset=['tpm','nrbp','transcript_gc','aa_length']).copy()
        assert (w.tpm >= 0).all() and (w.nrbp >= 0).all()
        counts = w.tx.value_counts()
        single = w[w.tx.isin(counts[counts.eq(1)].index)].copy()
        coverage.append({'cell': cell, 'n_covariate_complete': len(w), 'n_single_ORF': len(single), 'n_TPM_zero_included': int(w.tpm.eq(0).sum())})
        for outcome in CFG['outcomes']:
            for scheme in CFG['schemes']:
                data, scales = prepare(single if scheme == 'single_ORF' else w, outcome)
                for model_name in CFG['models']:
                    result.append(fit_one(data, scales, outcome, cell, scheme, model_name))
    # Families are fixed by model variant. Each contains the same four primary tests.
    for model_name in CFG['models']:
        primary = [r for r in result if r['model']==model_name and r['scheme']=='transcript']
        assert len(primary) == 4
        adjusted = multipletests([r['tost_p'] for r in primary], method='holm')[1]
        for row, p in zip(primary, adjusted):
            row['holm_p_primary'] = float(p)
            row['equivalent_0.10_holm'] = bool(p < CFG['alpha'])
    validation = []
    numerical = ['beta_std','se','ci90_lo','ci90_hi','ci95_lo','ci95_hi','tost_p_lower','tost_p_upper','tost_p','tost_p_stringent']
    for row in [r for r in result if r['model']=='original']:
        old = legacy[(legacy.cell==row['cell']) & (legacy.outcome==row['outcome']) & (legacy.scheme==row['scheme'])]
        assert len(old)==1
        old = old.iloc[0]
        assert int(old.n)==row['n']
        if row['n_cluster'] is not None:
            assert int(old.n_cluster)==row['n_cluster']
        diffs = {c: row[c]-float(old[c]) for c in numerical}
        for c in numerical:
            assert np.isclose(row[c], old[c], rtol=1e-8, atol=1e-11), (row['cell'],row['outcome'],row['scheme'],c)
        assert np.isclose(row['p_difference_normal'], old.p_difference, rtol=1e-8, atol=1e-11)
        for c in ['equivalent_0.10','equivalent_0.05']:
            assert row[c] == bool(old[c])
        assert row['verdict_legacy_convention'] == old.verdict
        if row['scheme']=='transcript':
            assert np.isclose(row['holm_p_primary'],old.holm_p_primary,rtol=1e-8,atol=1e-11)
        validation.append({'cell':row['cell'],'outcome':row['outcome'],'scheme':row['scheme'],'max_abs_numerical_difference':max(abs(v) for v in diffs.values()),'status':'PASS'})
    summaries = []
    for model in CFG['models']:
        rows = [r for r in result if r['model']==model]
        p = [r for r in rows if r['scheme']=='transcript']
        summaries.append({'model':model, 'all_four_primary_equivalent_0.10':all(r['equivalent_0.10'] for r in p),
            'all_four_primary_equivalent_0.10_holm':all(r['equivalent_0.10_holm'] for r in p),
            'all_four_primary_equivalent_0.05_pointwise':all(r['equivalent_0.05'] for r in p),
            'all16_equivalent_0.10_count':sum(r['equivalent_0.10'] for r in rows),
            'all16_equivalent_0.05_count':sum(r['equivalent_0.05'] for r in rows),
            'max_primary_holm_p':max(r['holm_p_primary'] for r in p)})
    for k,p in FILES.items():
        assert digest(p)==before[k]['sha256']
    assert digest(SPEC)==spec_hash
    out = {
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'specification_sha256':spec_hash, 'analysis_script_sha256':digest(Path(__file__)),
        'sources':before, 'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'patsy':patsy.__version__},
        'length_coverage_all_main_hosts':int(d.tx.nunique()), 'coverage':coverage,
        'baseline_validation':validation, 'summaries':summaries, 'model_results':result,
        'limits':CFG['limits'],
    }
    (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Original TOST results reproduced: 16/16; mature lengths verified for all main hosts')
    display = pd.DataFrame(result)
    cols = ['model','cell','outcome','n','beta_std','ci90_lo','ci90_hi','holm_p_primary','equivalent_0.10','equivalent_0.05']
    print(display.loc[display.scheme.eq('transcript'),cols].to_string(index=False))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()

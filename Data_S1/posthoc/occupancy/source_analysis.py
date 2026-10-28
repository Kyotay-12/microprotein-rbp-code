"""Read original inputs and run a pre-documented, post hoc length sensitivity."""
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
from scipy.stats import spearmanr
import statsmodels
import statsmodels.formula.api as smf
import patsy

PACKAGE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Data_S1')
sys.path.insert(0, str(PACKAGE_ROOT))
from input_support import expected_count, input_path

HERE = Path(__file__).resolve().parent
SPEC = HERE / 'specification.json'
cfg = json.loads(SPEC.read_text())
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'analysis/results'
FILES = {
    'properties': RES / 'step0_properties.csv',
    'binding': RES / 'step1_binding.csv',
    'expression': RES / 'step1_expression.csv',
    'rpf': RES / 'step2_rpf_counts.csv',
    'catalogue': ROOT / 'analysis/data/Mudge2022_SupplementaryTables.xlsx',
    'fasta': ROOT / 'analysis/data/gencode.v35.transcripts.fa.gz',
    'legacy': RES / 'followup11_cluster_robust.csv',
}


def digest(p):
    h = sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable(s):
    return re.sub(r'\.\d+(?=_|$)', '', str(s))


def load_inputs():
    d = pd.read_csv(FILES['properties'], comment='#')
    assert d.orf_name.is_unique
    d = d[(d.source_sheet == 'S2') & (d.aa_length <= 100)].copy()
    cat = pd.read_excel(FILES['catalogue'], sheet_name='S2. PHASE I Ribo-seq ORFs', usecols=['orf_name', 'gene_id'])
    assert cat.orf_name.is_unique
    d = d.merge(cat, on='orf_name', how='left', validate='one_to_one')
    for kind, cols in [
        ('binding', ['K562_nrbp_tx', 'HepG2_nrbp_tx']),
        ('expression', ['tpm_K562_polyA', 'tpm_HepG2_polyA']),
        ('rpf', ['K562_rpf_in', 'HepG2_rpf_in'])
    ]:
        part = pd.read_csv(FILES[kind])
        assert part.orf_name.is_unique
        d = d.merge(part[['orf_name'] + cols], on='orf_name', how='left', validate='one_to_one')
    assert len(d) == expected_count('main_orfs'), 'Population differs from supplied validation reference'
    assert d.gene_id.notna().all()
    d['tx'] = d.transcript.map(stable)
    d['gene'] = d.gene_id.map(stable)
    tx = pd.read_csv(FILES['host_metadata'])
    assert tx.enst.is_unique and set(tx.enst) == set(d.tx)
    d = d.merge(tx[['enst', 'ensg', 'spliced_length_nt']], left_on='tx', right_on='enst', how='left', validate='many_to_one')
    assert (d.gene == d.ensg).all(), 'Stable ENSG mapping disagreement'
    assert d.spliced_length_nt.notna().all() and d.spliced_length_nt.gt(0).all()

    # Independently read sequence lengths; GENCODE headers include ENST before |.
    targets = set(d.tx)
    lengths = {}
    key = None
    count = 0
    def finish():
        if key in targets:
            assert key not in lengths, f'Duplicate FASTA stable ENST: {key}'
            lengths[key] = count
    with gzip.open(FILES['fasta'], 'rt') as stream:
        for line in stream:
            if line.startswith('>'):
                finish()
                key = stable(line[1:].split('|')[0].split()[0])
                count = 0
            else:
                count += len(line.strip())
        finish()
    assert set(lengths) == targets, 'Missing FASTA transcript length'
    assert (d.tx.map(lengths) == d.spliced_length_nt).all(), 'FASTA/exon-union lengths disagree'
    d['ln_tx_length'] = np.log(d.spliced_length_nt)
    d['ln_aa'] = np.log(d.aa_length)
    d['gc'] = d.transcript_gc
    return d


BASE = 'ln_rbp + ln_aa + gc + ln_tpm'
MODEL_TERMS = {
    'original': BASE,
    'plus_log_length': BASE + ' + ln_tx_length',
    'plus_length_spline': BASE + ' + bs(ln_tx_length, df=4, degree=3, include_intercept=False)',
    'within_length_groups_pooled': BASE + ' + ln_tx_length + C(length_group)',
    'plus_length_and_ORF_biotype': BASE + ' + ln_tx_length + C(orf_biotype)',
}


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


def main():
    FILES['host_metadata'] = input_path('host_transcript_metadata')
    source_hashes = {k: {'path': str(p), 'sha256': digest(p)} for k, p in FILES.items()}
    spec_hash = digest(SPEC)
    d = load_inputs()
    print(f'Length coverage and FASTA/exon-length agreement: {d.tx.nunique()}/{d.tx.nunique()} main transcripts', flush=True)
    legacy = pd.read_csv(FILES['legacy'])
    records, diagnostics, subsets = [], [], []
    for cell in ['K562', 'HepG2']:
        w = d.copy()
        w['tpm'], w['nrbp'], w['rpf'] = w[f'tpm_{cell}_polyA'], w[f'{cell}_nrbp_tx'], w[f'{cell}_rpf_in']
        w = w.loc[w.tpm.ge(1)].dropna(subset=['tpm', 'nrbp', 'rpf', 'gc', 'aa_length']).copy()
        assert np.isfinite(w[['tpm', 'nrbp', 'rpf', 'gc', 'aa_length', 'spliced_length_nt']]).all().all()
        assert w[['tpm', 'nrbp', 'rpf']].ge(0).all().all()
        w['y'], w['ln_rbp'], w['ln_tpm'] = np.log2(w.rpf+1), np.log1p(w.nrbp), np.log1p(w.tpm)
        for col in ['spliced_length_nt', 'nrbp', 'tpm', 'gene']:
            assert w.groupby('tx')[col].nunique().eq(1).all(), (cell, col)
        hosts = w.drop_duplicates('tx').copy()
        qgroups, edges = pd.qcut(hosts.spliced_length_nt, 5, labels=False, retbins=True, duplicates='raise')
        groups = pd.Series(qgroups.to_numpy()+1, index=hosts.tx)
        w['length_group'] = w.tx.map(groups).astype(int)
        old = legacy[(legacy.analysis == 'step2_lrpf_tpm1') & (legacy.cell == cell) & (legacy.se == 'cluster_gene')].iloc[0]
        original = smf.ols('y ~ ' + BASE, data=w).fit(cov_type='cluster', cov_kwds={'groups': w.gene}, use_t=False)
        assert len(w) == old.n and w.gene.nunique() == old.n_cluster
        assert np.isclose(original.params['ln_rbp'], old.beta, atol=1e-12)
        assert np.allclose(original.conf_int().loc['ln_rbp'], [old.ci_lo, old.ci_hi], atol=1e-12)
        for label, terms in MODEL_TERMS.items():
            records.extend(fit_model(w, terms, label, cell))
        one = w[w.tx.map(w.tx.value_counts()).eq(1)].copy()
        records.extend(fit_model(one, MODEL_TERMS['plus_log_length'], 'single_analysed_ORF_plus_length', cell))
        for group in sorted(w.length_group.unique()):
            sub = w[w.length_group.eq(group)].copy()
            rows = fit_model(sub, MODEL_TERMS['plus_log_length'], f'length_group_{group}', cell)
            for row in rows:
                row.update({'length_min_nt': int(sub.spliced_length_nt.min()), 'length_max_nt': int(sub.spliced_length_nt.max())})
            subsets.extend(rows)
        diagnostics.append({
            'cell': cell, 'length_data_complete_without_row_change': True,
            'legacy_point_estimate_and_z_CI_replicated': True,
            'n_ORFs': len(w), 'n_transcripts': len(hosts),
            'length_nt_quantiles': {str(q): float(v) for q, v in hosts.spliced_length_nt.quantile([0,.25,.5,.75,1]).items()},
            'spearman_host_length_vs_RBP_count_unique_transcripts': float(spearmanr(hosts.spliced_length_nt, hosts.nrbp).statistic),
            'quintile_edges_nt': edges.tolist(),
            'ORF_ids_sha256': sha256('\n'.join(sorted(w.orf_name)).encode()).hexdigest(),
        })
        print(f'{cell}: n={len(w)}, baseline reproduced', flush=True)
    for row in records:
        base = next(r for r in records if r['cell'] == row['cell'] and r['cluster'] == row['cluster'] and r['model'] == 'original')
        row['beta_relative_change_vs_full_population_baseline'] = (row['beta']/base['beta']-1
            if row['model'] != 'single_analysed_ORF_plus_length' else None)
    assert digest(SPEC) == spec_hash
    for name, p in FILES.items():
        assert digest(p) == source_hashes[name]['sha256'], f'Input changed: {p}'
    output = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'specification_sha256': spec_hash, 'analysis_script_sha256': digest(Path(__file__)),
        'sources': source_hashes,
        'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__, 'patsy': patsy.__version__},
        'main_n_ORFs': len(d), 'host_length_verified_n_transcripts': int(d.tx.nunique()),
        'diagnostics': diagnostics, 'model_results': records, 'length_stratum_results': subsets,
        'limitations': cfg['limits'],
    }
    (HERE/'results.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    table = pd.DataFrame(records)
    print(table.loc[table.cluster.eq('gene'), ['cell', 'model', 'n_ORFs', 'beta', 'ci95_lo', 'ci95_hi', 'p', 'incremental_R2_RBP', 'RBP_VIF']].to_string(index=False))
    print('Results written to results.json')


if __name__ == '__main__':
    main()

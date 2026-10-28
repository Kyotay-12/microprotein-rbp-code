"""Fixed same-host comparison models with mature RNA length sensitivity."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import patsy

HERE=Path(__file__).resolve().parent
SPEC=HERE.parent/'specification.json'
FORMULAS={
    'nc_rpf_unadjusted_for_cds':'l_nc_rpf ~ l1p_rbp + l_nc_len + transcript_gc + l1p_tpm',
    'canonical_cds_rpf':'l_cds_rpf ~ l1p_rbp + l_cds_len + transcript_gc + l1p_tpm',
    'allocation_cds_adjusted':'l_nc_rpf ~ l1p_rbp + l_cds_rpf + l_nc_len + l_cds_len + transcript_gc + l1p_tpm',
    'density_ratio':'log_density_ratio ~ l1p_rbp + transcript_gc + l1p_tpm',
}
ADJUSTMENTS={
    'original':'',
    'plus_log_length':' + l_host_len',
    'plus_length_spline':' + bs(l_host_len, df=4, degree=3, include_intercept=False)',
}


def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def population_hash(d):
    return sha256(('\n'.join(sorted(d.orf_name))+'\n').encode()).hexdigest()


def prepare(d,cell,stratum,conservative=False,corrected=False):
    types=['uORF'] if stratum=='uORF_primary' else ['uORF','dORF']
    required=[f'tpm_{cell}_polyA',f'{cell}_nrbp_tx',f'{cell}_rpf_in',f'{cell}_cds_rpf_in','transcript_gc','aa_length','cds_aa_length','enst']
    s=d.loc[d.orf_biotype.isin(types)].dropna(subset=required).copy()
    s=s.loc[s[f'tpm_{cell}_polyA'].ge(1)].copy()
    if 'canonical_model_eligible' in s:
        s=s.loc[s.canonical_model_eligible].copy()
    assert s.own_CDS_overlap_nt.eq(0).all() and s.nc_contiguous_host_path.all()
    if conservative:
        s=s.loc[s.coordinate_conservative].copy()
    assert len(s)>100 and s.enst.notna().all() and s.ensg.notna().all()
    assert s.spliced_length_nt.notna().all() and s.spliced_length_nt.gt(0).all()
    assert s.groupby('enst').ensg.nunique().max()==s.groupby('ensg').enst.nunique().max()==1
    cds_length=s.cds_codon_length if corrected else s.cds_aa_length
    assert (s[[f'{cell}_rpf_in',f'{cell}_cds_rpf_in',f'{cell}_nrbp_tx']]>=0).all().all()
    s['l1p_rbp']=np.log1p(s[f'{cell}_nrbp_tx'])
    s['l1p_tpm']=np.log1p(s[f'tpm_{cell}_polyA'])
    s['l_nc_len']=np.log(s.aa_length)
    s['l_cds_len']=np.log(cds_length)
    s['l_host_len']=np.log(s.spliced_length_nt)
    s['l_nc_rpf']=np.log2(s[f'{cell}_rpf_in']+1)
    s['l_cds_rpf']=np.log2(s[f'{cell}_cds_rpf_in']+1)
    s['log_density_ratio']=np.log2(((s[f'{cell}_rpf_in']+.5)/s.aa_length)/((s[f'{cell}_cds_rpf_in']+.5)/cds_length))
    return s


def fit(s,specification,adjustment,cluster):
    formula=FORMULAS[specification]+ADJUSTMENTS[adjustment]
    m=smf.ols(formula,s).fit(cov_type='cluster',cov_kwds={'groups':s[cluster],'use_correction':True},use_t=False)
    X=m.model.exog; y=m.model.endog
    names=m.model.exog_names
    ix=names.index('l1p_rbp')
    assert int(m.nobs)==len(s) and np.linalg.matrix_rank(X)==X.shape[1]
    beta=float(m.params['l1p_rbp']); se=float(m.bse['l1p_rbp'])
    manual=np.linalg.lstsq(X,y,rcond=None)[0]
    assert np.isclose(beta,manual[ix],rtol=1e-10,atol=1e-11)
    reduced=np.delete(X,ix,axis=1)
    yres=y-reduced@np.linalg.lstsq(reduced,y,rcond=None)[0]
    xres=X[:,ix]-reduced@np.linalg.lstsq(reduced,X[:,ix],rcond=None)[0]
    fwl=float(xres@yres/(xres@xres))
    assert np.isclose(beta,fwl,rtol=1e-10,atol=1e-11)
    nclusters=int(s[cluster].nunique()); df=nclusters-1
    critical=stats.t.ppf(.975,df)
    normalci=m.conf_int().loc['l1p_rbp']
    reduced_r2=1-float(yres@yres/np.sum((y-y.mean())**2))
    return dict(specification=specification,adjustment=adjustment,cluster=cluster,formula=formula,
        n=len(s),n_clusters=nclusters,design_columns=X.shape[1],df_inference=df,population_sha256=population_hash(s),
        beta=beta,SE=se,ci95_lo=beta-critical*se,ci95_hi=beta+critical*se,p_t=float(2*stats.t.sf(abs(beta/se),df)),
        legacy_ci95_normal_lo=float(normalci.iloc[0]),legacy_ci95_normal_hi=float(normalci.iloc[1]),p_normal=float(m.pvalues['l1p_rbp']),
        full_R2=float(m.rsquared),incremental_R2_RBP=float(m.rsquared-reduced_r2),
        RBP_VIF=float(np.sum((X[:,ix]-X[:,ix].mean())**2)/(xres@xres)),condition_number=float(np.linalg.cond(X)),
        canonical_RPF_coefficient=float(m.params['l_cds_rpf']) if 'l_cds_rpf' in m.params else None,
        CDS_length_matches_input=bool(np.allclose(np.exp(s.l_cds_len),s.cds_aa_length)),
        ORF_ids=s.orf_name.tolist())


def main():
    audit=json.loads((HERE/'input_audit.json').read_text())
    assert digest(HERE/'prepared_records.json')==audit['prepared_records_sha256']
    assert digest(SPEC)==audit['specification_sha256']
    assert all(digest(Path(v['path']))==v['sha256'] for v in audit['sources'].values())
    d=pd.DataFrame(json.loads((HERE/'prepared_records.json').read_text()))
    legacy=pd.DataFrame(json.loads(Path(audit['sources']['old_prepared']['path']).read_text()))
    results=[]; common_populations=[]
    def add_models(s,cell,stratum,population,reference_version,adjustments):
        for adjustment in adjustments:
            for key in FORMULAS:
                for cluster in ['enst','ensg']:
                    r=fit(s,key,adjustment,cluster)
                    r.update(cell=cell,stratum=stratum,population=population,reference_version=reference_version,
                             length_definition='legacy' if reference_version=='legacy' else 'corrected_CDS_codon')
                    results.append(r)
    for cell in ['K562','HepG2']:
        for stratum in ['uORF_primary','uORF_dORF_sensitivity']:
            s=prepare(d,cell,stratum)
            add_models(s,cell,stratum,'corrected_reselected','corrected',ADJUSTMENTS)
            old=prepare(legacy,cell,stratum)
            common=sorted(set(s.orf_name)&set(old.orf_name))
            new_common=s.set_index('orf_name',drop=False).loc[common].reset_index(drop=True)
            old_common=old.set_index('orf_name',drop=False).loc[common].reset_index(drop=True)
            assert population_hash(new_common)==population_hash(old_common)
            # Same rows and unchanged micro-side covariates isolate CDS value effects.
            for c in ['enst','ensg','aa_length','transcript_gc','spliced_length_nt',f'{cell}_rpf_in',f'{cell}_nrbp_tx',f'tpm_{cell}_polyA']:
                if pd.api.types.is_numeric_dtype(new_common[c]):
                    assert np.allclose(new_common[c].to_numpy(),old_common[c].to_numpy(),rtol=1e-12,atol=1e-12),c
                else:
                    assert np.array_equal(new_common[c].to_numpy(),old_common[c].to_numpy()),c
            common_populations.append(dict(cell=cell,stratum=stratum,n=len(common),population_sha256=population_hash(new_common),ORF_ids=common))
            add_models(old_common,cell,stratum,'common_host_rows','legacy',ADJUSTMENTS)
            add_models(new_common,cell,stratum,'common_host_rows','corrected',ADJUSTMENTS)
        s=prepare(d,cell,'uORF_primary',conservative=True)
        add_models(s,cell,'uORF_primary','coordinate_conservative','corrected',['original','plus_log_length'])
    primary=[r for r in results if r['population']=='corrected_reselected' and r['stratum']=='uORF_primary' and r['cluster']=='enst' and r['adjustment']=='plus_log_length' and r['specification']=='allocation_cds_adjusted']
    assert len(primary)==2
    for r,p in zip(primary,multipletests([r['p_t'] for r in primary],method='holm')[1]):
        r['holm_p_two_principal_cells']=float(p)
    # Gene and transcript cluster partitions are identical in this matched subset.
    for r in results:
        if r['cluster']!='enst': continue
        keyfields=['cell','stratum','population','length_definition','reference_version','specification','adjustment']
        other=[s for s in results if s['cluster']=='ensg' and all(s[k]==r[k] for k in keyfields)]
        assert len(other)==1
        assert all(np.isclose(r[k],other[0][k],rtol=1e-10,atol=1e-11) for k in ['beta','SE','ci95_lo','ci95_hi','p_t'])
    assert len(results)==320
    out=dict(completed_utc=datetime.now(timezone.utc).isoformat(),specification_sha256=digest(SPEC),
        analysis_script_sha256=digest(Path(__file__)),input_audit_sha256=digest(HERE/'input_audit.json'),
        prepared_records_sha256=digest(HERE/'prepared_records.json'),sources=audit['sources'],
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__,patsy=patsy.__version__),
        previous_results_preserved=True,common_populations=common_populations,
        transcript_and_gene_clusters_identical=True,model_results=results)
    assert all(digest(Path(v['path']))==v['sha256'] for v in audit['sources'].values())
    (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for r in results:
        if r['cluster']=='enst' and r['stratum']=='uORF_primary' and r['adjustment']=='plus_log_length':
            print(f"{r['cell']:5} {r['population']:24} {r['adjustment']:19} {r['specification']:29} n={r['n']} beta={r['beta']:+.6f} CI=[{r['ci95_lo']:+.6f}, {r['ci95_hi']:+.6f}] p={r['p_t']:.6g}")
    print(f'Saved {len(results)} model records; historical inputs/results unchanged')


if __name__=='__main__':
    main()

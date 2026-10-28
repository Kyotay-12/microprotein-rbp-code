"""Canonical models and separate historical-model verification."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE=Path(__file__).resolve().parent;BASE=HERE.parent;R=HERE/'results'
PROJECT=Path(__import__('os').environ.get('MICROPROTEIN_UPSTREAM_ROOT', str(BASE.parents[0]/'upstream'))).resolve();OLD=PROJECT/'analysis/results'
SPEC='a269d3a85af767284f66167c2feabc9d6147bfa09333c805dee9ecd65a15877c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(BASE/'specification.json')!=SPEC:raise ValueError('Specification changed')

def fit(d,outcome):
    X=np.column_stack([np.ones(len(d)),np.log1p(d.nrbp),np.log(d.aa_length),d.gc,np.log1p(d.tpm)])
    y=d[outcome].to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(X)!=5:raise ValueError('Invalid model input')
    b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b
    inv=np.linalg.inv(X.T@X);h=np.einsum('ij,jk,ik->i',X,inv,X)
    if (h>=1).any():raise ValueError('Unit leverage')
    cov=inv@((X.T*(e/(1-h))**2)@X)@inv
    se=np.sqrt(cov[1,1]);z=b[1]/se;crit=norm.ppf(.975)
    den=np.sum((y-y.mean())**2);full=1-np.dot(e,e)/den
    reduced=np.delete(X,1,axis=1);er=y-reduced@np.linalg.lstsq(reduced,y,rcond=None)[0];red=1-np.dot(er,er)/den
    sd=float(d[outcome].std(ddof=1))
    ids=sorted(d.enst.tolist());pop_hash=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
    return dict(n=len(d),n_genes=int(d.ensg.nunique()) if 'ensg' in d else len(d),beta=float(b[1]),SE=float(se),ci_lo=float(b[1]-crit*se),ci_hi=float(b[1]+crit*se),p=float(2*norm.sf(abs(z))),
        full_R2=float(full),reduced_R2=float(red),semi_partial_R2=float(full-red),outcome_sd=sd,
        beta_outcome_SD_per_ln1p_RBP=float(b[1]/sd),ci_halfwidth_in_sd=float(crit*se/sd),
        median_nrbp=float(d.nrbp.median()),median_tpm=float(d.tpm.median()),population_sha256=pop_hash)

def models(per,props,legacy=False):
    rows=[];cohorts={}
    populations=['historical'] if legacy else ['all_phase_valid','complete_only','PAR_excluded']
    for population in populations:
        for cell in ['K562','HepG2']:
            for family,df,outcomes in [('occupancy',per,['lrpf']),('property',props,['mean_plddt','mean_disorder'])]:
                d=df.copy()
                if not legacy:
                    d=d[d.valid_geometry_phase].copy()
                    if population=='complete_only':d=d[d.strict_complete_CDS].copy()
                    if population=='PAR_excluded':d=d[~d.is_PAR_copy].copy()
                d['tpm']=d[cell+'_TPM'];d['nrbp']=d[cell+'_nrbp']
                req=['tpm','nrbp','aa_length','gc']
                if family=='occupancy':
                    d['rpf']=d[cell+'_rpf_in'];req+=['rpf'];d=d.dropna(subset=req);d=d[d.tpm>=1].copy();d['lrpf']=np.log2(d.rpf+1)
                else:d=d.dropna(subset=req+outcomes)
                for outcome in outcomes:
                    row=fit(d,outcome);row.update(population=population,cell=cell,family=family,outcome=outcome,
                        specification='historical_reproduction' if legacy else 'all_tags__preserve_PAR__corrected_codon_length__phase_aware')
                    rows.append(row);cohorts['|'.join([population,cell,outcome])]=sorted(d.enst.tolist())
    return rows,cohorts

per=pd.read_csv(R/'canonical_per_cds.csv');props=pd.read_csv(R/'canonical_properties.csv')
assert per.enst.is_unique and set(per.enst)==set(props.enst)
primary,cohorts=models(per,props)
pd.DataFrame(primary).to_csv(R/'canonical_models.csv',index=False)
(R/'canonical_model_cohorts.json').write_text(json.dumps(cohorts,indent=2)+'\n')

# Historical replay is explicitly separate and never a corrected current input.
oldper=pd.read_csv(OLD/'followup2_canonical_cds_per_cds.csv');oldprops=pd.read_csv(OLD/'followup7_canonical_properties.csv')
quant={
 'K562':['K562_polyA_main_ENCSR000AEO_rep1_ENCFF720TFN.tsv','K562_polyA_main_ENCSR000AEO_rep2_ENCFF522DDA.tsv'],
 'HepG2':['HepG2_polyA_main_ENCSR985KAT_rep1_ENCFF458NJB.tsv','HepG2_polyA_main_ENCSR985KAT_rep2_ENCFF371KBB.tsv']}
sys.path.insert(0,str(BASE/'shared'));from ensembl_ids import stable_ensembl_id
for cell,files in quant.items():
    reps=[]
    for file in files:
        d=pd.read_csv(PROJECT/'analysis/data/rnaseq'/file,sep='\t',usecols=['transcript_id','TPM'])
        d=d[d.transcript_id.str.startswith('ENST')].copy();d['enst']=d.transcript_id.map(lambda x:stable_ensembl_id(x,policy='legacy_first_dot'))
        reps.append(d.groupby('enst').TPM.sum())
    mean=pd.concat(reps,axis=1,join='inner').mean(axis=1)
    oldper[cell+'_TPM']=oldper.enst.map(mean);oldprops[cell+'_TPM']=oldprops.enst.map(mean)
oldmodels,oldcohorts=models(oldper,oldprops,legacy=True)
occ=pd.read_csv(OLD/'followup10_canonical_lrpf_fixed.csv');struct=pd.read_csv(OLD/'followup7_canonical_structural.csv')
checks=[]
for row in oldmodels:
    table=occ if row['family']=='occupancy' else struct
    match=table[table.cell==row['cell']]
    if row['family']=='property':match=match[match.outcome==row['outcome']]
    assert len(match)==1
    target=match.iloc[0]
    for field in ['n','beta','ci_lo','ci_hi','p','full_R2','semi_partial_R2','median_nrbp','median_tpm']:
        ok=bool(np.isclose(row[field],target[field],rtol=1e-8,atol=0 if field=='p' else 1e-10))
        checks.append(dict(cell=row['cell'],outcome=row['outcome'],field=field,passed=ok,recomputed=row[field],source_value=float(target[field])))
legacy_report=dict(specification='Historical last-tag, PAR-collapsed, minus-one-codon selection; old geometry/counts. Replay only, not current corrected reference.',
    source_SHA256={str(p.relative_to(PROJECT)):sha(p) for p in [OLD/'followup2_canonical_cds_per_cds.csv',OLD/'followup7_canonical_properties.csv',OLD/'followup10_canonical_lrpf_fixed.csv',OLD/'followup7_canonical_structural.csv']},
    models=oldmodels,checks=checks,status='PASS' if all(c['passed'] for c in checks) else 'FAIL')
(R/'historical_model_reproduction.json').write_text(json.dumps(legacy_report,indent=2)+'\n')
if legacy_report['status']!='PASS':raise ValueError('Historical regression replay differs')
manifest=dict(specification_SHA256=SPEC,producer_script_sha256=sha(Path(__file__)),
    input_SHA256={p.name:sha(p) for p in [R/'canonical_per_cds.csv',R/'canonical_properties.csv',R/'prediction_provenance.json']},
    output_SHA256={p.name:sha(p) for p in [R/'canonical_models.csv',R/'canonical_model_cohorts.json',R/'historical_model_reproduction.json']},
    current_models=len(primary),historical_numeric_checks=len(checks),historical_reproduction_status='PASS')
(R/'model_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(pd.DataFrame(primary)[['population','cell','outcome','n','beta','SE','ci_lo','ci_hi','p','semi_partial_R2']].to_string(index=False))
print('Historical models reproduced:',len(checks),'checks PASS')

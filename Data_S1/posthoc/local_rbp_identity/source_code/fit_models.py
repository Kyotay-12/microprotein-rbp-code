"""Fit only frozen eligible identity comparisons; no adaptive model search."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from prepare_design import sha, save_json, require

HERE=Path(__file__).resolve().parent.parent
PACKAGE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Data_S1')
sys.path.insert(0, str(PACKAGE_ROOT))
from input_support import input_path


def validate_freeze():
    folder=HERE/'predictor_only'
    manifest=json.loads((folder/'output_manifest.json').read_text())
    for name,r in manifest.items():
        p=folder/name
        require(p.is_file() and sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],
                'Predictor-only output changed:'+name)
    freeze=json.loads((folder/'design_freeze.json').read_text())
    for p,h in freeze['source_hashes'].items():
        require(sha(p)==h,'Frozen source changed:'+p)
    for r in freeze['peak_sources']:
        require(sha(r['path'])==r['sha256'],'Peak source changed:'+r['path'])
    return freeze


def inference(fit,x,y,groups):
    g=pd.Series(groups).nunique(); n,k=x.shape
    require(g>k and n>k,'Too few clusters or rows')
    robust=fit.get_robustcov_results(cov_type='cluster',groups=np.asarray(groups),
                                   use_correction=True,df_correction=True,use_t=True)
    require(robust.df_resid_inference==g-1,'Inference degrees of freedom mismatch')
    a=x.to_numpy(); score=pd.DataFrame(a*fit.resid.to_numpy()[:,None]).groupby(
        np.asarray(groups),sort=True).sum().to_numpy()
    bread=np.linalg.inv(a.T@a)
    manual=g/(g-1)*(n-1)/(n-k)*bread@(score.T@score)@bread
    covariance=np.asarray(robust.cov_params())
    require(np.allclose(covariance,manual,rtol=1e-7,atol=1e-9),'CR1 covariance mismatch')
    ids=[x.columns.get_loc(c) for c in ['A_U','A_D']]
    b=np.asarray(fit.params)[ids]; v=covariance[np.ix_(ids,ids)]
    require(np.linalg.eigvalsh(v).min()>0,'Joint covariance not positive definite')
    f=float(b@np.linalg.solve(v,b)/2); p=float(stats.f.sf(f,2,g-1))
    contrast=np.zeros((2,k)); contrast[0,ids[0]]=1; contrast[1,ids[1]]=1
    wald=robust.wald_test(contrast,use_f=True,scalar=True)
    require(np.isclose(float(wald.statistic),f,rtol=1e-9) and
            np.isclose(float(wald.pvalue),p,rtol=1e-9,atol=1e-15),'Joint test mismatch')
    ci=robust.conf_int(alpha=.05)
    coef=[]
    for j,name in enumerate(x.columns):
        coef.append({'term':name,'beta':float(fit.params.iloc[j]),
                     'se':float(robust.bse[j]),'ci95_lo':float(ci[j,0]),
                     'ci95_hi':float(ci[j,1]),'p_nominal':float(robust.pvalues[j])})
    return {'clusters':int(g),'df_num':2,'df_den':int(g-1),'joint_F':f,
            'joint_p_nominal':p},coef


def main():
    ap=argparse.ArgumentParser();ap.parse_args()
    out=HERE/'results';require(not out.exists(),f'Will not overwrite {out}')
    require(not (HERE/'outcome_fit_freeze.json').exists(),'Already frozen/fitted; do not rerun silently')
    freeze=validate_freeze(); spec=json.loads((HERE/'specification.json').read_text())
    project=Path(freeze['source_project'])
    source=input_path('local_window_results')
    chosen={k:v for k,v in freeze['models'].items() if v['eligible'] and
            freeze['models'][f'{v["cell"]}_{v["rbp"]}_base']['eligible']}
    record={'created_utc':datetime.now(timezone.utc).isoformat(),
            'stage':'After predictor-only selection, before reading RPF outcomes',
            'fit_script_sha256':sha(Path(__file__)),
            'specification_sha256':sha(HERE/'specification.json'),
            'predictor_manifest_sha256':sha(HERE/'predictor_only/output_manifest.json'),
            'eligible_designs':list(chosen),
            'planned_family_size':freeze['planned_primary_tests'],
            'software':{'python':platform.python_version(),'numpy':np.__version__,
                        'pandas':pd.__version__,'scipy':scipy.__version__,
                        'statsmodels':statsmodels.__version__}}
    save_json(HERE/'outcome_fit_freeze.json',record)
    print('Outcome fit specification/code frozen; now reading original RPF counts.',flush=True)
    outcome={}
    for cell in spec['cells']:
        w=pd.read_csv(source/f'{cell}_orf_eligibility.csv',usecols=['orf_name',f'{cell}_rpf_in'])
        require(w.orf_name.is_unique,'Duplicate outcome keys')
        require(w[f'{cell}_rpf_in'].notna().all() and w[f'{cell}_rpf_in'].ge(0).all(),
                'Missing or negative outcome')
        outcome[cell]=w.set_index('orf_name')[f'{cell}_rpf_in']
    records=[];coefficients=[]; predictions={}
    for key,v in freeze['models'].items():
        c,rbp,variant=v['cell'],v['rbp'],v['variant']
        status={'cell':c,'rbp':rbp,'variant':variant,'model_key':key,
                'n':v['n'],'hosts':v['hosts'],'genes':v['genes'],
                'design_eligible':v['eligible']}
        if key not in chosen:
            reason=v['failures'] if not v['eligible'] else ['primary_design_ineligible']
            for cluster in ['transcript','gene']:
                records.append({**status,'cluster':cluster,'status':'not_tested',
                                'reason':';'.join(reason),'joint_p_nominal':None,
                                'joint_p_holm_8':None})
            continue
        rows=pd.read_csv(HERE/'predictor_only'/f'{key}_rows.csv')
        x1=pd.read_csv(HERE/'predictor_only'/f'{key}_design.csv')
        require(list(x1)==v['M1_columns'] and len(x1)==len(rows),'Frozen design grain mismatch')
        x0=x1[v['M0_columns']]
        counts=outcome[c].reindex(rows.orf_name)
        require(counts.notna().all(),'Missing outcome after ID join')
        y=pd.Series(np.log2(counts.to_numpy()+1),index=x1.index,name='y')
        fit0=sm.OLS(y,x0).fit(); fit1=sm.OLS(y,x1).fit()
        sse0=float(np.square(fit0.resid).sum());sse1=float(np.square(fit1.resid).sum())
        sst=float(np.square(y-y.mean()).sum())
        require(sse1<=sse0+1e-8 and sst>0 and sse0>0,'Invalid nested SSE/SST')
        metrics={'sse0':sse0,'sse1':sse1,'sst':sst,'R2_M0':1-sse0/sst,
                 'R2_M1':1-sse1/sst,'delta_R2':(sse0-sse1)/sst,
                 'partial_R2':(sse0-sse1)/sse0,'K0':x0.shape[1],'K1':x1.shape[1]}
        for cluster,col in [('transcript','enst'),('gene','ensg')]:
            infer,coefs=inference(fit1,x1,y,rows[col])
            records.append({**status,'cluster':cluster,'status':'fitted','reason':'',
                            **metrics,**infer})
            coefficients.extend([{**status,'cluster':cluster,**r} for r in coefs])
        pred=rows[['orf_name','enst','ensg']].copy()
        pred['rpf_count']=counts.to_numpy();pred['y']=y
        pred['M0_fitted']=fit0.fittedvalues;pred['M1_fitted']=fit1.fittedvalues
        predictions[key]=pred
    for variant in ['base','biotype_position','host_bound']:
        for cluster in ['transcript','gene']:
            family=[r for r in records if r['variant']==variant and r['cluster']==cluster]
            require(len(family)==freeze['planned_primary_tests'],'Planned multiplicity family changed')
            ps=[r['joint_p_nominal'] if r['status']=='fitted' else 1 for r in family]
            corrected=multipletests(ps,alpha=.05,method='holm')[1]
            for r,p in zip(family,corrected):
                r['joint_p_holm_8']=float(p) if r['status']=='fitted' else None
                r['multiplicity_family']=variant+'_'+cluster+'_8_planned_slots'
    validate_freeze()
    require(sha(Path(__file__))==record['fit_script_sha256'],'Fit code changed during execution')
    out.mkdir()
    save_json(out/'model_results.json',records)
    pd.DataFrame(records).to_csv(out/'model_results.csv',index=False)
    pd.DataFrame(coefficients).to_csv(out/'coefficients.csv',index=False)
    for k,p in predictions.items(): p.to_csv(out/f'{k}_fitted.csv',index=False)
    save_json(out/'provenance.json',record)
    save_json(out/'output_manifest.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}
                                         for p in sorted(out.iterdir()) if p.is_file()})
    for r in records:
        if r['status']=='fitted' and r['cluster']=='transcript':
            print(r['model_key'],'n=',r['n'],'delta_R2=',r['delta_R2'],
                  'p=',r['joint_p_nominal'],'Holm8=',r['joint_p_holm_8'],flush=True)
    print('Finished all planned estimable comparisons; unsupported slots retained.',flush=True)


if __name__=='__main__':main()

"""Read-only replay of all planned local-RBP-identity comparisons.

Requires a separately supplied, unchanged derived-data and reference bundle.
Raw-read/peak/annotation reconstruction is outside this check's scope.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Compare:
    def __init__(self):
        self.checks = 0
        self.max_abs = {}

    def exact(self, got, expected, label):
        self.checks += 1
        require(got == expected, f'{label}: {got!r} != {expected!r}')

    def number(self, got, expected, label, family='numeric'):
        self.checks += 1
        a, b = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                label + ': invalid shape/value')
        self.max_abs[family] = max(self.max_abs.get(family, 0.), float(np.max(np.abs(a-b))))
        require(np.allclose(a, b, atol=2e-9, rtol=2e-7), label + ': numerical mismatch')


def rank(a, reference=0):
    values = np.linalg.svd(a, compute_uv=False)
    tolerance = max(a.shape) * np.finfo(float).eps * max(values[0] if len(values) else 0, reference)
    return int(np.sum(values > tolerance))


def design_from_rows(w, variant):
    # Deterministic model specification, rebuilt from raw derived predictors.
    x = pd.DataFrame({'intercept': np.ones(len(w))})
    for name, source, offset in [('h', 'H', 1), ('aa', 'aa_length', 0),
                                 ('tpm', 'TPM', 1), ('hostlen', 'tx_length', 0),
                                 ('u', 'U_nrbp', 1), ('d', 'D_nrbp', 1)]:
        x[name] = np.log(w[source].to_numpy(dtype=float) + offset)
    x['gc'] = w.transcript_gc.to_numpy(dtype=float)
    x = x[['intercept','h','aa','gc','tpm','hostlen','u','d']]
    omitted = []
    if w.B.nunique() > 1:
        x['B'] = w.B.astype(float)
    else:
        require(w.B.eq(1).all(), 'Candidate absent from all hosts')
        omitted.append('B:constant1')
    require(rank(x.to_numpy()) == len(x.columns), 'Baseline not full rank')
    if variant == 'biotype_position':
        extras = pd.get_dummies(w.orf_biotype, prefix='biotype', drop_first=True, dtype=float)
        extras['relative_start'] = w.t_start.to_numpy(dtype=float) / w.tx_length.to_numpy(dtype=float)
        for column in extras:
            trial = pd.concat([x, extras[[column]]], axis=1)
            if rank(trial.to_numpy()) == len(trial.columns):
                x = trial
            else:
                require(column.startswith('biotype_'), 'Relative start unexpectedly collinear')
                omitted.append(column + ':collinear_dummy')
    return x, pd.concat([x,w[['A_U','A_D']].astype(float)], axis=1), omitted


def validate_design(w, model, guards, compare):
    failures = []
    groups = {'U_positive': w.B.eq(1)&w.A_U.eq(1),
              'D_positive': w.B.eq(1)&w.A_D.eq(1),
              'host_only': w.B.eq(1)&w.A_U.eq(0)&w.A_D.eq(0)}
    if w.B.nunique() > 1:
        groups['host_unbound'] = w.B.eq(0)
    support = {}
    for name, mask in groups.items():
        subset = w.loc[mask]
        support[name] = {'orfs': len(subset), 'transcripts': subset.enst.nunique(),
                         'genes': subset.ensg.nunique()}
        for unit in ['transcripts','genes']:
            if support[name][unit] < guards['min_unique_transcripts_and_genes_per_group']:
                failures.append(f'{name}:{unit}<{guards["min_unique_transcripts_and_genes_per_group"]}')
    compare.exact(support, model['support'], 'support counts')
    compare.exact((len(w),w.enst.nunique(),w.ensg.nunique()),
                  (model['n'],model['hosts'],model['genes']), 'cohort counts')
    try:
        x0,x1,omitted = design_from_rows(w,model['variant'])
    except ValueError as exc:
        failures.append(str(exc))
        compare.exact(failures, model['failures'], 'unconstructable design reason')
        compare.exact(False, model['eligible'], 'unconstructable eligibility')
        return None,None
    compare.exact(list(x0),model['M0_columns'],'M0 columns')
    compare.exact(list(x1),model['M1_columns'],'M1 columns')
    compare.exact(omitted,model['omitted_columns'],'omitted columns')
    a0,a1=x0.to_numpy(),x1.to_numpy()
    z=w[['A_U','A_D']].to_numpy(dtype=float)
    residual=z-a0@np.linalg.lstsq(a0,z,rcond=None)[0]
    r=rank(residual,np.linalg.norm(z,2))
    compare.exact((rank(a0),rank(a1),r),
                  (model['rank_M0'],model['rank_M1'],model['local_added_rank']), 'ranks')
    if r!=2 or rank(a1)!=a1.shape[1]:
        failures.append('local_block_not_rank2')
    else:
        q,s,_=np.linalg.svd(residual,full_matrices=False)
        q=q[:,:2]
        compare.number(s,model['local_singular_values'],'residual singular values')
        for unit in ['enst','ensg']:
            weights=[];eigenmax=[]
            for ids in w.groupby(unit,sort=True).indices.values():
                block=q[ids,:]
                weights.append(np.square(block).sum(axis=0))
                eigenmax.append(float(np.linalg.eigvalsh(block.T@block)[-1]))
            weights=np.stack(weights);eff=1/np.square(weights).sum(axis=0)
            share=weights.max(axis=0);strong=max(eigenmax)
            saved=model['information_concentration'][unit]
            compare.number(eff,saved['effective_clusters_by_direction'],'effective information')
            compare.number(share,saved['max_share_by_direction'],'direction information share')
            compare.number(strong,saved['max_any_direction_share'],'any-direction share')
            if min(eff)<guards['min_effective_clusters_each_residual_svd_direction']:
                failures.append(unit+':low_effective_information_clusters')
            if max(share)>guards['max_residual_information_share_any_cluster'] or strong>guards['max_residual_information_share_any_cluster']:
                failures.append(unit+':information_concentrated')
    compare.exact(failures,model['failures'],'design failures')
    compare.exact(not failures,model['eligible'],'design eligibility')
    return x0,x1


def ols(a,y):
    q,r=np.linalg.qr(a,mode='reduced')
    beta=np.linalg.solve(r,q.T@y)
    resid=y-a@beta
    bread=np.linalg.solve(r,np.eye(r.shape[0]))
    return beta,resid,bread@bread.T


def clustered_inference(a,y,beta,resid,bread,groups,columns):
    codes,labels=pd.factorize(groups,sort=True)
    g,n,k=len(labels),len(y),a.shape[1]
    require(g>k and n>k,'Insufficient clusters')
    scores=np.zeros((g,k))
    np.add.at(scores,codes,a*resid[:,None])
    covariance=g/(g-1)*(n-1)/(n-k)*bread@(scores.T@scores)@bread
    se=np.sqrt(np.diag(covariance));crit=stats.t.ppf(.975,g-1)
    indices=[columns.index('A_U'),columns.index('A_D')]
    local=beta[indices];v=covariance[np.ix_(indices,indices)]
    require(np.linalg.eigvalsh(v).min()>0,'Invalid joint covariance')
    f=float(local@np.linalg.solve(v,local)/2)
    model={'clusters':g,'df_num':2,'df_den':g-1,'joint_F':f,
           'joint_p_nominal':float(stats.f.sf(f,2,g-1))}
    coefficients={name:{'beta':float(beta[j]),'se':float(se[j]),
        'ci95_lo':float(beta[j]-crit*se[j]),'ci95_hi':float(beta[j]+crit*se[j]),
        'p_nominal':float(2*stats.t.sf(abs(beta[j]/se[j]),g-1))}
        for j,name in enumerate(columns)}
    return model,coefficients


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True,help='Separate complete local-RBP-identity data/reference directory')
    ap.add_argument('--report',type=Path,help='Optional JSON outside the distributed tree')
    args=ap.parse_args(argv);root=args.root.resolve();cmp=Compare()
    if not __debug__:raise RuntimeError('Run without -O: validation assertions must remain active.')
    if not (root/'identity_inventory.json').is_file():
        raise FileNotFoundError('The separate identity input needs identity_inventory.json.')
    if args.report:
        require(not args.report.resolve().is_relative_to(root),'Report must be outside distributed subtree')
        require(not args.report.resolve().is_relative_to(Path(__file__).resolve().parents[3]),'Report must be outside the code checkout')
        require(not args.report.exists(),'Will not overwrite an existing report')
    manifest=json.loads((root/'identity_inventory.json').read_text())
    expected=set(manifest['files'])|{'identity_inventory.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    cmp.exact(actual,expected,'identity subtree inventory')
    for rel,record in manifest['files'].items():
        p=root/rel
        require(not p.is_symlink() and p.resolve().is_relative_to(root),'Unsafe inventory path')
        cmp.exact((digest(p),p.stat().st_size),(record['sha256'],record['bytes']),'file bytes '+rel)
    spec=json.loads((root/'specification.json').read_text())
    freeze=json.loads((root/'predictor_only/design_freeze.json').read_text())
    records=json.loads((root/'results/model_results.json').read_text())
    keys={(r['model_key'],r['cluster']):r for r in records}
    expected_keys={(key,cluster) for key in freeze['models'] for cluster in ['transcript','gene']}
    cmp.exact(len(keys),len(records),'unique result keys')
    cmp.exact(set(keys),expected_keys,'all planned statuses retained')
    cmp.exact(len(freeze['models']),len(spec['cells'])*len(spec['candidates'])*3,'planned designs')
    cmp.exact(freeze['planned_primary_tests'],len(spec['cells'])*len(spec['candidates']),'planned family')
    coefficients=pd.read_csv(root/'results/coefficients.csv')
    require(not coefficients.duplicated(['model_key','cluster','term']).any(),'Duplicate coefficient keys')
    host=pd.read_csv(root/'predictor_only/host_binding_identities.csv')
    require(not host.duplicated(['cell','enst']).any(),'Duplicate host identities')
    hostsets={(r.cell,r.enst):set(json.loads(r.rbps)) for r in host.itertuples()}
    support_rows=pd.read_csv(root/'predictor_only/binding_support_counts.csv')
    actual_results={};fitted=0;checked_coefficient_rows=0
    for key,model in freeze['models'].items():
        cell,rbp,variant=model['cell'],model['rbp'],model['variant']
        w=pd.read_csv(root/'predictor_only'/f'{key}_rows.csv')
        require(w.orf_name.is_unique and not w.isna().any().any(),'Missing/duplicate design inputs')
        w['H']=w[f'{cell}_nrbp_tx'];w['TPM']=w[f'tpm_{cell}_polyA']
        for cluster in ['transcript','gene']:
            r=keys[(key,cluster)]
            for name,value in {'cell':cell,'rbp':rbp,'variant':variant,
                    'model_key':key,'n':len(w),'hosts':int(w.enst.nunique()),
                    'genes':int(w.ensg.nunique()),'design_eligible':model['eligible']}.items():
                cmp.exact(r[name],value,'result identity '+name)
        require(w.source_sheet.eq('S2').all() and w.aa_length.between(1,100).all() and w.TPM.ge(1).all(), 'Population mismatch')
        require(w.complete_window_eligible.all() and w.U_available_nt.eq(100).all() and w.D_available_nt.eq(100).all(), 'Window mismatch')
        categories=[]
        for row in w.itertuples():
            h=hostsets[(cell,row.enst)];u=set(json.loads(row.U_rbps));d=set(json.loads(row.D_rbps))
            require(u<=h and d<=h,'Local identities absent from host')
            cmp.exact((len(h),len(u),len(d)),(row.H,row.U_nrbp,row.D_nrbp),'richness counts')
            cmp.exact((int(rbp in h),int(rbp in u),int(rbp in d)),(row.B,row.A_U,row.A_D),'identity indicators')
            category='host_unbound' if not row.B else ('both' if row.A_U and row.A_D else ('U_only' if row.A_U else ('D_only' if row.A_D else 'host_only')))
            categories.append(category)
        cmp.exact(categories,list(w.binding_category),'binding categories')
        if variant=='base':
            saved=support_rows.loc[support_rows.cell.eq(cell)&support_rows.rbp.eq(rbp)].set_index('category')
            cmp.exact(set(saved.index),set(['host_unbound','host_only','U_only','D_only','both']),'support categories')
            for category in saved.index:
                s=w.loc[w.binding_category.eq(category)]
                cmp.exact((len(s),s.enst.nunique(),s.ensg.nunique()),tuple(int(saved.loc[category,c]) for c in ['orfs','transcripts','genes']),'disjoint support')
        x0,x1=validate_design(w,model,spec['design_guards'],cmp)
        if x1 is not None:
            distributed=pd.read_csv(root/'predictor_only'/f'{key}_design.csv')
            cmp.exact(list(distributed),list(x1),'distributed design columns')
            cmp.number(x1.to_numpy(),distributed.to_numpy(),'derived design matrix')
        eligible=model['eligible'] and freeze['models'][f'{cell}_{rbp}_base']['eligible']
        if not eligible:
            reason=model['failures'] if not model['eligible'] else ['primary_design_ineligible']
            for cluster in ['transcript','gene']:
                r=keys[(key,cluster)]
                cmp.exact((r['status'],r['joint_p_nominal'],r['joint_p_holm_8']),('not_tested',None,None),'untestable p-values')
                cmp.exact(r['reason'],';'.join(reason),'untestable reason')
                actual_results[(key,cluster)]=None
            continue
        prediction=pd.read_csv(root/'outcomes'/f'{key}_fitted.csv')
        cmp.exact(w[['orf_name','enst','ensg']].to_dict('records'),prediction[['orf_name','enst','ensg']].to_dict('records'),'outcome row identity')
        counts=prediction.rpf_count.to_numpy(dtype=float)
        require(np.isfinite(counts).all() and (counts>=0).all(),'Invalid real RPF counts')
        y=np.log2(counts+1);cmp.number(y,prediction.y.to_numpy(),'count transform')
        a0,a1=x0.to_numpy(),x1.to_numpy()
        b0,e0,_=ols(a0,y);b1,e1,bread=ols(a1,y)
        cmp.number(a0@b0,prediction.M0_fitted.to_numpy(),'M0 fitted values','fit')
        cmp.number(a1@b1,prediction.M1_fitted.to_numpy(),'M1 fitted values','fit')
        sse0=float(e0@e0);sse1=float(e1@e1);sst=float(np.square(y-y.mean()).sum())
        metrics={'sse0':sse0,'sse1':sse1,'sst':sst,'R2_M0':1-sse0/sst,'R2_M1':1-sse1/sst,
                 'delta_R2':(sse0-sse1)/sst,'partial_R2':(sse0-sse1)/sse0,'K0':a0.shape[1],'K1':a1.shape[1]}
        for cluster,group in [('transcript','enst'),('gene','ensg')]:
            r=keys[(key,cluster)];cmp.exact(r['status'],'fitted','fitted status')
            infer,coefs=clustered_inference(a1,y,b1,e1,bread,w[group].to_numpy(),list(x1))
            for name,value in {**metrics,**infer}.items():
                cmp.number(value,r[name],key+' '+cluster+' '+name,'model')
            cr=coefficients.loc[coefficients.model_key.eq(key)&coefficients.cluster.eq(cluster)].set_index('term')
            cmp.exact(set(cr.index),set(coefs),'coefficient terms')
            checked_coefficient_rows+=len(cr)
            for term,values in coefs.items():
                for name,value in values.items():
                    cmp.number(value,cr.loc[term,name],key+' '+cluster+' '+term+' '+name,'coefficient')
            actual_results[(key,cluster)]=infer['joint_p_nominal'];fitted+=1
    cmp.exact(checked_coefficient_rows,len(coefficients),'all coefficient rows checked')
    for variant in ['base','biotype_position','host_bound']:
        for cluster in ['transcript','gene']:
            family=[r for r in records if r['variant']==variant and r['cluster']==cluster]
            cmp.exact(len(family),freeze['planned_primary_tests'],'fixed multiplicity family size')
            pvalues=np.array([actual_results[(r['model_key'],cluster)] if r['status']=='fitted' else 1. for r in family])
            order=np.argsort(pvalues,kind='stable');running=0.;corrected=np.zeros(len(family))
            for index,which in enumerate(order):
                running=max(running,(len(family)-index)*pvalues[which]);corrected[which]=min(1.,running)
            for r,p in zip(family,corrected):
                if r['status']=='fitted':cmp.number(p,r['joint_p_holm_8'],'Holm fixed family','Holm')
                cmp.exact(r['multiplicity_family'],variant+'_'+cluster+'_8_planned_slots','family labels')
    # CSV is a second distributed rendering, not an independent model.
    csv_rows=list(csv.DictReader((root/'results/model_results.csv').open()))
    cmp.exact(len(csv_rows),len(records),'JSON/CSV row counts')
    for a,b in zip(records,csv_rows):
        for name,value in b.items():
            original=a.get(name)
            if original is None:cmp.exact(value,'','JSON/CSV missingness')
            elif isinstance(original,bool):cmp.exact(value,str(original),'JSON/CSV bool')
            elif isinstance(original,(int,float)):cmp.number(float(value),original,'JSON/CSV number')
            else:cmp.exact(value,str(original),'JSON/CSV text')
    result={'status':'PASS','checks':cmp.checks,'planned_slots':len(records),'fitted_records':fitted,
            'not_tested_records':len(records)-fitted,'max_absolute_numeric_errors':cmp.max_abs,
            'scope':'Observed derived predictors and RPF counts; support/design decisions, fitted cluster-model records, nominal intervals, same-row fit metrics, fixed eight-slot Holm families, and reference CSV/JSON agreement. No raw-data extraction.',
            'replay_sha256':digest(Path(__file__)),
            'identity_inventory_sha256':digest(root/'identity_inventory.json')}
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())

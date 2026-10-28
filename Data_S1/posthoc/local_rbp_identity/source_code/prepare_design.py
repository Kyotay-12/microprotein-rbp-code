"""Real-data, predictor-only extraction and frozen eligibility for identity tests.

No outcome column is read. Scientific CSV/JSON intermediates, not a workbook.
Existing inputs and scripts are read-only. Missing or inconsistent data fail.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Data_S1')
sys.path.insert(0, str(PACKAGE_ROOT))
from input_support import input_path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rank(a, reference=0):
    s = np.linalg.svd(a, compute_uv=False)
    tol = max(a.shape)*np.finfo(float).eps*max(s[0] if len(s) else 0, reference)
    return int(np.sum(s > tol))


def predictor_columns(cell):
    return ['orf_name','enst','ensg','source_sheet','aa_length','transcript_gc',
            f'tpm_{cell}_polyA',f'{cell}_nrbp_tx','tx_length','t_start',
            'orf_biotype','chromosome','strand','gtf_transcript_id',
            'U_nrbp','D_nrbp','U_rbps','D_rbps','U_blocks','D_blocks',
            'U_available_nt','D_available_nt','complete_window_eligible',
            'in_region_disjoint_hosts']


def load_predictors(source, cell):
    w = pd.read_csv(source/f'{cell}_orf_eligibility.csv', usecols=predictor_columns(cell))
    require(w.complete_window_eligible.dtype == bool, 'Eligibility flag not boolean')
    require(w.orf_name.is_unique, 'Duplicate ORF')
    w = w.loc[w.complete_window_eligible].copy().sort_values(['enst','orf_name'])
    require(w.notna().all().all(), 'Missing predictor input')
    require((w.source_sheet == 'S2').all() and w.aa_length.between(1,100).all(),
            'Unexpected analysis population')
    require((w[f'tpm_{cell}_polyA'] >= 1).all(), 'TPM eligibility mismatch')
    require((w.U_available_nt.eq(100) & w.D_available_nt.eq(100)).all(),
            'Incomplete windows')
    for side in ['U','D']:
        w[f'{side}_set'] = w[f'{side}_rbps'].map(json.loads).map(set)
        require((w[f'{side}_set'].map(len) == w[f'{side}_nrbp']).all(),
                'Duplicate/mismatching local labels')
    for target, raw, add in [('h',f'{cell}_nrbp_tx',1),('aa','aa_length',0),
                             ('tpm',f'tpm_{cell}_polyA',1),('hostlen','tx_length',0),
                             ('u','U_nrbp',1),('d','D_nrbp',1)]:
        w[target] = np.log(w[raw]+add)
    w['gc'] = w.transcript_gc
    w['relative_start'] = w.t_start/w.tx_length
    require(np.isfinite(w[['h','aa','tpm','hostlen','u','d','gc',
                          'relative_start']].to_numpy()).all(), 'Nonfinite predictor')
    return w.reset_index(drop=True)


def design(w, variant):
    x = pd.DataFrame({'intercept':np.ones(len(w))}, index=w.index)
    cols = ['h','aa','gc','tpm','hostlen','u','d']
    x = pd.concat([x,w[cols]], axis=1)
    omitted = []
    if w.B.nunique() > 1:
        x['B'] = w.B.astype(float)
    else:
        require(w.B.eq(1).all(), 'Candidate absent from all hosts')
        omitted.append('B:constant1')
    require(rank(x.to_numpy()) == x.shape[1], 'Baseline not full rank')
    if variant == 'biotype_position':
        extras = pd.get_dummies(w.orf_biotype, prefix='biotype', drop_first=True, dtype=float)
        extras['relative_start'] = w.relative_start
        for col in extras:
            new = pd.concat([x,extras[[col]]], axis=1)
            if rank(new.to_numpy()) == new.shape[1]:
                x = new
            else:
                require(col.startswith('biotype_'), 'Relative start unexpectedly collinear')
                omitted.append(col+':collinear_dummy')
    z = w[['A_U','A_D']].astype(float)
    return x, pd.concat([x,z],axis=1), omitted


def assess(w, variant, guards):
    failures = []
    groups = {'U_positive':w.B.eq(1)&w.A_U.eq(1),
              'D_positive':w.B.eq(1)&w.A_D.eq(1),
              'host_only':w.B.eq(1)&w.A_U.eq(0)&w.A_D.eq(0)}
    if w.B.nunique() > 1:
        groups['host_unbound'] = w.B.eq(0)
    support = {}
    for name, flag in groups.items():
        support[name] = {'orfs':int(flag.sum()),
                         'transcripts':int(w.loc[flag,'enst'].nunique()),
                         'genes':int(w.loc[flag,'ensg'].nunique())}
        for unit in ['transcripts','genes']:
            if support[name][unit] < guards['min_unique_transcripts_and_genes_per_group']:
                failures.append(f'{name}:{unit}<{guards["min_unique_transcripts_and_genes_per_group"]}')
    info = {'support':support,'n':len(w),'hosts':int(w.enst.nunique()),
            'genes':int(w.ensg.nunique()),'failures':failures}
    try:
        x0,x1,omitted = design(w,variant)
    except RuntimeError as e:
        failures.append(str(e)); info['eligible']=False
        return info, None, None
    a0,a1 = x0.to_numpy(),x1.to_numpy()
    z = w[['A_U','A_D']].to_numpy(dtype=float)
    residual = z-a0@np.linalg.lstsq(a0,z,rcond=None)[0]
    r = rank(residual,np.linalg.norm(z,2))
    info.update({'omitted_columns':omitted,'rank_M0':rank(a0),'rank_M1':rank(a1),
                 'local_added_rank':r,'M0_columns':list(x0),'M1_columns':list(x1)})
    if r != 2 or rank(a1) != a1.shape[1]:
        failures.append('local_block_not_rank2')
    else:
        q,s,_ = np.linalg.svd(residual,full_matrices=False)
        q = q[:,:2]
        info['local_singular_values']=s.tolist()
        norms = np.linalg.norm(residual,axis=0)
        info['local_normalized_condition']=float(np.linalg.cond(residual/norms))
        info['local_residual_correlation']=float(np.corrcoef(residual.T)[0,1])
        concentrations={}
        for unit in ['enst','ensg']:
            contributions=[]
            for key,idx in w.groupby(unit,sort=True).indices.items():
                qr=q[idx,:]
                contributions.append((str(key),np.square(qr).sum(axis=0),
                                      float(np.linalg.eigvalsh(qr.T@qr)[-1])))
            weights=np.stack([v[1] for v in contributions])
            eff=1/np.square(weights).sum(axis=0)
            maxshare=weights.max(axis=0)
            strongest=max(v[2] for v in contributions)
            concentrations[unit]={'effective_clusters_by_direction':eff.tolist(),
                'max_share_by_direction':maxshare.tolist(),'max_any_direction_share':strongest,
                'top_clusters_any_direction':[{'cluster':v[0],'share':v[2]} for v in
                  sorted(contributions,key=lambda v:(-v[2],v[0]))[:5]]}
            if min(eff)<guards['min_effective_clusters_each_residual_svd_direction']:
                failures.append(unit+':low_effective_information_clusters')
            if max(maxshare)>guards['max_residual_information_share_any_cluster'] or \
                    strongest>guards['max_residual_information_share_any_cluster']:
                failures.append(unit+':information_concentrated')
        info['information_concentration']=concentrations
    info['eligible']=not failures
    return info,x0,x1


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True)
    args=ap.parse_args(); root=args.project.resolve()
    out=HERE/'predictor_only'; require(not out.exists(),f'Will not overwrite {out}')
    spec=json.loads((HERE/'specification.json').read_text())
    source=input_path('local_window_results')
    man=json.loads((source/'output_manifest.json').read_text())
    for name in [f'{c}_orf_eligibility.csv' for c in spec['cells']]+['local_binding_records.csv']:
        require(sha(source/name)==man[name]['sha256'],'Frozen eligibility/local input changed')
    geo=load_module('identity_geometry',HERE/'source_code/geometry.py')
    peaks=load_module('identity_peaks',HERE/'source_code/peaks.py')
    stable=geo._load_stable(HERE/'source_code/ensembl_ids.py')
    source_paths=[HERE/'specification.json',HERE/'literature_candidates.json',Path(__file__),
        source/'output_manifest.json',source/'provenance.json',source/'peak_audit.json',
        source/'local_binding_records.csv',HERE/'source_code/geometry.py',HERE/'source_code/peaks.py',
        HERE/'source_code/ensembl_ids.py',
        root/'analysis/data/gencode.v35.annotation.gtf.gz',
        root/'analysis/results/followup8_binding_matrix.parquet',
        root/'analysis/results/eclip_manifest.tsv']+[source/f'{c}_orf_eligibility.csv' for c in spec['cells']]
    before={str(p):sha(p) for p in source_paths}
    w_by={c:load_predictors(source,c) for c in spec['cells']}
    targets=set().union(*(set(w.enst) for w in w_by.values()))
    tx,failures,_=geo._load_gtf(root/'analysis/data/gencode.v35.annotation.gtf.gz',targets,stable)
    require(not failures and set(tx)==targets,'Missing or ambiguous host geometry')
    for w in w_by.values():
        for r in w.itertuples(index=False):
            t=tx[r.enst]
            require(all(getattr(r,k)==t[k] for k in ['ensg','chromosome','strand','tx_length','gtf_transcript_id']),
                    'Host geometry differs from frozen local source')
    print('Verified PAR-preserving transcript geometry',len(tx),flush=True)
    rows,manifest_prov=peaks._manifest(root)
    for c in spec['cells']:
        require(set(spec['candidates']) <= {r['rbp'] for r in rows if r['cell_line']==c},
                'Candidate not assayed in both cells')
    hostsets={(c,e):set() for c in spec['cells'] for e in targets}
    locus_hosts=defaultdict(list)
    for e,t in tx.items():
        locus_hosts[(t['chromosome'],t['strand'])].append(e)
    localhits={(c,r.orf_name,side):set() for c,w in w_by.items()
               for r in w.itertuples(index=False) for side in ['U','D']}
    file_meta=[]
    for j,row in enumerate(rows,1):
        intervals,meta=peaks._load_peak_intervals(row); file_meta.append(meta)
        indexes={loc:peaks._merged_index(v) for loc,v in intervals.items()}
        c,rbp=row['cell_line'],row['rbp']
        for loc in indexes.keys() & locus_hosts.keys():
            ix=indexes[loc]
            for e in locus_hosts[loc]:
                if any(peaks._overlaps(ix,b['start'],b['end']) for b in tx[e]['exons']):
                    hostsets[(c,e)].add(rbp)
        if rbp in spec['candidates']:
            for r in w_by[c].itertuples(index=False):
                ix=indexes.get((r.chromosome,r.strand))
                if ix is not None:
                    for side in ['U','D']:
                        if any(peaks._overlaps(ix,int(a),int(b)) for a,b in json.loads(getattr(r,side+'_blocks'))):
                            localhits[(c,r.orf_name,side)].add(rbp)
        if j%50==0: print('Original peak files checked',j,'/',len(rows),flush=True)
    oldpeak=json.loads((source/'peak_audit.json').read_text())
    oldfiles=oldpeak['provenance']['peak_files']
    oldhash={r['relative_path']:r['sha256'] for r in oldfiles}
    require({r['relative_path']:r['sha256'] for r in file_meta}==oldhash,'Peak inventory differs from frozen local extraction')
    cache=pd.read_parquet(root/'analysis/results/followup8_binding_matrix.parquet')
    require(list(cache)==['cell','rbp','enst'] and not cache.duplicated().any(),'Host cache schema/duplicates')
    cached={key:set(g.rbp) for key,g in cache.groupby(['cell','enst'])}
    local=pd.read_csv(source/'local_binding_records.csv').set_index(['cell_line','orf_id','side'])
    require(local.index.is_unique,'Duplicate local records')
    checks=0
    for c,w in w_by.items():
        for r in w.itertuples(index=False):
            hs=hostsets[(c,r.enst)]
            require(hs==cached.get((c,r.enst),set()),'Fresh host identity disagrees with existing edge cache')
            require(len(hs)==getattr(r,c+'_nrbp_tx'),'Whole-host richness mismatch')
            checks+=2
            for side in ['U','D']:
                original=getattr(r,side+'_set')
                require(original==set(json.loads(local.loc[(c,r.orf_name,side),'rbps'])), 'Local tables disagree')
                require(original<=hs,'Local identity absent from host')
                require(original & set(spec['candidates'])==localhits[(c,r.orf_name,side)],'Fresh local identity mismatch')
                checks+=3
    for p,h in before.items(): require(sha(p)==h,'Input changed during extraction:'+p)
    out.mkdir()
    pd.DataFrame([{'cell':c,'enst':e,'rbps':json.dumps(sorted(v)),'nrbp':len(v)}
                  for (c,e),v in sorted(hostsets.items())]).to_csv(out/'host_binding_identities.csv',index=False)
    save_json(out/'host_exon_geometry.json',tx)
    freeze={'created_utc':datetime.now(timezone.utc).isoformat(),'source_project':str(root),
            'phase':'predictor-only, no RPF column read or outcome fitted',
            'source_hashes':before,'peak_sources':file_meta,'cross_checks':checks,
            'planned_primary_tests':len(spec['candidates'])*len(spec['cells']),
            'assayed_rbps':{c:sorted({r['rbp'] for r in rows if r['cell_line']==c}) for c in spec['cells']},
            'models':{}}
    counts=[]
    for c,w in w_by.items():
        for candidate in spec['candidates']:
            q=w.copy()
            q['B']=[int(candidate in hostsets[(c,e)]) for e in q.enst]
            q['A_U']=q.U_set.map(lambda s:int(candidate in s))
            q['A_D']=q.D_set.map(lambda s:int(candidate in s))
            require((q.A_U<=q.B).all() and (q.A_D<=q.B).all(),'Identity nesting failure')
            q['binding_category']=np.select([q.B.eq(0),q.A_U.eq(0)&q.A_D.eq(0),
                   q.A_U.eq(1)&q.A_D.eq(0),q.A_U.eq(0)&q.A_D.eq(1)],
                   ['host_unbound','host_only','U_only','D_only'],default='both')
            for category in ['host_unbound','host_only','U_only','D_only','both']:
                g=q.loc[q.binding_category.eq(category)]
                counts.append({'cell':c,'rbp':candidate,'category':category,'orfs':len(g),
                               'transcripts':g.enst.nunique(),'genes':g.ensg.nunique()})
            for variant in ['base','biotype_position','host_bound']:
                s=q.loc[q.B.eq(1)].copy().reset_index(drop=True) if variant=='host_bound' else q.copy()
                status,x0,x1=assess(s,variant,spec['design_guards'])
                key=f'{c}_{candidate}_{variant}'
                freeze['models'][key]={'cell':c,'rbp':candidate,'variant':variant,**status}
                s.drop(columns=['U_set','D_set']).to_csv(out/f'{key}_rows.csv',index=False)
                if x1 is not None:
                    x1.to_csv(out/f'{key}_design.csv',index=False)
                if variant=='base':
                    print(c,candidate,'eligible=',status['eligible'],
                          'support=',status['support'],'reasons=',status['failures'],flush=True)
    pd.DataFrame(counts).to_csv(out/'binding_support_counts.csv',index=False)
    save_json(out/'design_freeze.json',freeze)
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    save_json(out/'output_manifest.json',files)
    print('Predictor-only phase complete. Outcome fitting has NOT run.',flush=True)


if __name__=='__main__': main()

"""Independent source reconstruction, geometry and manual CR1 validation.

Does not import prepare.py, analyze.py or statsmodels. Patsy is used only for
the prespecified B-spline basis; all regressions and sandwich covariance are
computed directly with NumPy. No source file is written.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import numpy as np
import openpyxl
import patsy
from scipy import stats

HERE = Path(__file__).resolve().parent
SPEC_PATH=HERE.parent/'specification.json'
SPEC = json.loads(SPEC_PATH.read_text())
RESULT = json.loads((HERE/'results.json').read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read_csv(p):
    with p.open() as f:return list(csv.DictReader(line for line in f if not line.startswith('#')))


def keyed(rows,key):
    ans={r[key]:r for r in rows}
    assert len(ans)==len(rows),(key,'duplicate source keys')
    return ans


def stable(v):return re.sub(r'\.\d+(?=_|$)','',str(v))


def union(blocks):
    out=[]
    for a,b in sorted({tuple(v) for v in blocks}):
        assert a<b
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out


def overlap(a,b):
    a,b=union(a),union(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total


def build_rows(cds_key='cds'):
    src={k:Path(v['path']) for k,v in RESULT['sources'].items()}
    props=read_csv(src['properties'])
    expr=keyed(read_csv(src['expression']),'orf_name')
    bind=keyed(read_csv(src['binding']),'orf_name')
    rpf=keyed(read_csv(src['rpf']),'orf_name')
    cds=keyed(read_csv(src[cds_key]),'enst')
    meta=keyed(read_csv(src['length']),'enst')
    rows=[]
    for p in props:
        if p['source_sheet']!='S2' or float(p['aa_length'])>100:continue
        name=p['orf_name'];e=expr[name];tx=e['enst']
        if tx not in cds:continue
        c=cds[tx];m=meta[tx]
        r=dict(orf_name=name,enst=tx,ensg=m['ensg'],orf_biotype=p['orf_biotype'],
               aa_length=float(p['aa_length']),transcript_gc=float(p['transcript_gc']),
               cds_aa_length=float(c['aa_length']),cds_len=float(c['cds_len']),
               spliced_length_nt=float(m['spliced_length_nt']))
        for cell in ['K562','HepG2']:
            r[f'tpm_{cell}_polyA']=float(e[f'tpm_{cell}_polyA']) if e[f'tpm_{cell}_polyA'] else math.nan
            r[f'{cell}_nrbp_tx']=float(bind[name][f'{cell}_nrbp_tx'])
            r[f'{cell}_rpf_in']=float(rpf[name][f'{cell}_rpf_in'])
            r[f'{cell}_cds_rpf_in']=float(c[f'{cell}_rpf_in'])
        rows.append(r)
    cat={};w=openpyxl.load_workbook(src['catalogue'],read_only=True,data_only=True)
    it=w['S2. PHASE I Ribo-seq ORFs'].iter_rows(values_only=True);header=next(it)
    ids={r['orf_name'] for r in rows}
    for values in it:
        d=dict(zip(header,values));name=d.get('orf_name')
        if name in ids:cat[name]=d
    w.close();assert len(cat)==len(rows)
    for r in rows:
        d=cat[r['orf_name']]
        assert stable(d['transcript'])==r['enst'] and stable(d['gene_id'])==r['ensg']
        r['chrom']=str(d['chrm']) if str(d['chrm']).startswith('chr') else 'chr'+str(d['chrm'])
        r['strand']=d['strand']
        r['blocks']=list(zip([int(float(x))-1 for x in str(d['starts']).split(';')],
                             [int(float(x)) for x in str(d['ends']).split(';')]))
    return rows,src


def audit_geometry(rows,gtf):
    wanted={r['enst'] for r in rows};tx=defaultdict(lambda:defaultdict(list));glob=defaultdict(list)
    with gzip.open(gtf,'rt') as f:
        for line in f:
            if line.startswith('#'):continue
            z=line.rstrip('\n').split('\t')
            if z[2] not in ['CDS','exon','start_codon','stop_codon','transcript']:continue
            a,b=int(z[3])-1,int(z[4])
            if z[2]=='CDS':glob[(z[0],z[6])].append((a,b))
            fields=re.findall(r'(\w+) "([^"]*)"',z[8]);d=dict(fields)
            t=stable(d.get('transcript_id',''))
            if t not in wanted:continue
            q=tx[t];q[z[2]].append((a,b,z[7]));q['raw'].append(d['transcript_id'])
            q['chrom'].append(z[0]);q['strand'].append(z[6]);q['gene'].append(stable(d['gene_id']))
            q['tag'].extend(v for k,v in fields if k=='tag')
    glob={k:union(v) for k,v in glob.items()}
    for r in rows:
        q=tx[r['enst']];strand=r['strand'];blocks=r['blocks'];cds=q['CDS']
        assert set(q['chrom'])=={r['chrom']} and set(q['strand'])=={strand} and set(q['gene'])=={r['ensg']}
        assert len(set(q['raw']))==1
        cds_nt=sum(b-a for a,b,p in cds);assert cds_nt==r['cds_len']
        exons=union([(a,b) for a,b,p in q['exon']]);assert sum(b-a for a,b in exons)==r['spliced_length_nt']
        path=[];offset=0
        for ea,eb in sorted(exons,reverse=strand=='-'):
            for a,b in sorted(blocks,reverse=strand=='-'):
                if ea<=a and b<=eb:
                    path.append((offset+a-ea,offset+b-ea) if strand=='+' else (offset+eb-b,offset+eb-a))
            offset+=eb-ea
        path.sort()
        inside=overlap(blocks,exons)==sum(b-a for a,b in blocks)
        contiguous=inside and len(path)==len(blocks) and all(path[i][1]==path[i+1][0] for i in range(len(path)-1))
        ordered=sorted(cds,reverse=strand=='-')
        first=int(ordered[0][2]) if ordered and ordered[0][2]!='.' else None
        phase=first is not None;position=0
        for a,b,p in sorted(cds,reverse=strand=='-'):
            phase &= p!='.' and first is not None and int(p)==(first-position)%3;position+=b-a
        stop=union([(a,b) for a,b,p in q['stop_codon']]);start=union([(a,b) for a,b,p in q['start_codon']])
        complete=phase and first==0 and cds_nt%3==0 and sum(b-a for a,b in stop)==3 and sum(b-a for a,b in start)==3 and not ({'cds_start_NF','cds_end_NF'}&set(q['tag']))
        cdsi=[(a,b) for a,b,p in cds]
        own=overlap(blocks,cdsi);other=overlap(blocks,glob[(r['chrom'],strand)])
        stopoverlap=overlap(stop,cdsi)
        raw_exons=[(a,b) for a,b,p in q['exon']]
        valid_geometry=bool(phase and sum(b-a for a,b in union(cdsi))==cds_nt and sum(b-a for a,b in raw_exons)==sum(b-a for a,b in exons) and overlap(cdsi,exons)==cds_nt and stopoverlap==0)
        complete=complete and valid_geometry
        r.update(own_CDS_overlap_nt=own,any_CDS_overlap_nt=other,nc_in_host_exons=inside,
                 nc_contiguous_host_path=contiguous,cds_phase_valid=bool(phase),cds_geometry_valid=valid_geometry,cds_complete=bool(complete),
                 stop_CDS_overlap_nt=stopoverlap,cds_codon_length=cds_nt//3,cds_initial_phase=first,
                 coordinate_conservative=bool(complete and contiguous and own==0 and other==0),
                 legacy_CDS_one_codon_short=False,
                 canonical_model_eligible=bool(valid_geometry and all(math.isfinite(r[f'{cell}_cds_rpf_in']) for cell in ['K562','HepG2'])))
    return rows


def select(rows,m):
    types={'uORF'} if m['stratum']=='uORF_primary' else {'uORF','dORF'}
    cell=m['cell'];out=[]
    for r in rows:
        fields=[f'tpm_{cell}_polyA',f'{cell}_nrbp_tx',f'{cell}_rpf_in',f'{cell}_cds_rpf_in','transcript_gc','aa_length','cds_aa_length']
        if r['orf_biotype'] not in types or not all(math.isfinite(r[k]) for k in fields) or r[f'tpm_{cell}_polyA']<1:continue
        if m['reference_version']=='corrected' and not r['canonical_model_eligible']:continue
        if m['population']=='coordinate_conservative' and not r['coordinate_conservative']:continue
        out.append(r)
    return out


def fit(rows,m):
    cell=m['cell'];a=lambda k:np.array([r[k] for r in rows],dtype=float)
    x=np.log1p(a(f'{cell}_nrbp_tx'));t=np.log1p(a(f'tpm_{cell}_polyA'));gc=a('transcript_gc')
    ln=np.log(a('aa_length'));cdslen=a('cds_codon_length') if m['length_definition']=='CDS_codon' else a('cds_aa_length')
    lc=np.log(cdslen);yn=np.log2(a(f'{cell}_rpf_in')+1);yc=np.log2(a(f'{cell}_cds_rpf_in')+1)
    lh=np.log(a('spliced_length_nt'));s=m['specification']
    if s=='nc_rpf_unadjusted_for_cds':y=yn;cov=[x,ln,gc,t]
    elif s=='canonical_cds_rpf':y=yc;cov=[x,lc,gc,t]
    elif s=='allocation_cds_adjusted':y=yn;cov=[x,yc,ln,lc,gc,t]
    elif s=='density_ratio':
        y=np.log2(((a(f'{cell}_rpf_in')+.5)/a('aa_length'))/((a(f'{cell}_cds_rpf_in')+.5)/cdslen));cov=[x,gc,t]
    else:raise ValueError(s)
    if m['adjustment']=='plus_log_length':cov.append(lh)
    elif m['adjustment']=='plus_length_spline':
        basis=np.asarray(patsy.dmatrix('bs(lh,df=4,degree=3,include_intercept=False)-1',{'lh':lh}))
        cov.extend(basis.T)
    X=np.column_stack([np.ones(len(rows))]+cov);n,k=X.shape
    beta=np.linalg.lstsq(X,y,rcond=None)[0];err=y-X@beta
    inv=np.linalg.inv(X.T@X);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r[m['cluster']]].append(i)
    meat=np.zeros((k,k))
    for ix in groups.values():
        score=X[ix].T@err[ix];meat+=np.outer(score,score)
    G=len(groups);V=inv@meat@inv*(G/(G-1))*((n-1)/(n-k));se=math.sqrt(V[1,1]);b=beta[1]
    crit=stats.t.ppf(.975,G-1);zcrit=stats.norm.ppf(.975)
    Xred=np.delete(X,1,axis=1);redy=y-Xred@np.linalg.lstsq(Xred,y,rcond=None)[0]
    redx=x-Xred@np.linalg.lstsq(Xred,x,rcond=None)[0];tot=np.sum((y-y.mean())**2)
    out=dict(n=n,n_clusters=G,design_columns=k,df_inference=G-1,beta=float(b),SE=se,
             ci95_lo=float(b-crit*se),ci95_hi=float(b+crit*se),p_t=float(2*stats.t.sf(abs(b/se),G-1)),
             legacy_ci95_normal_lo=float(b-zcrit*se),legacy_ci95_normal_hi=float(b+zcrit*se),p_normal=float(2*stats.norm.sf(abs(b/se))),
             full_R2=float(1-err@err/tot),incremental_R2_RBP=float((redy@redy-err@err)/tot),
             RBP_VIF=float(np.sum((x-x.mean())**2)/(redx@redx)),condition_number=float(np.linalg.cond(X)),
             canonical_RPF_coefficient=float(beta[2]) if s=='allocation_cds_adjusted' else None)
    return out


def main():
    hashes={v['path']:sha(Path(v['path'])) for v in RESULT['sources'].values()}
    assert all(hashes[v['path']]==v['sha256'] for v in RESULT['sources'].values())
    assert RESULT['specification_sha256']==sha(SPEC_PATH)
    rows,src=build_rows()
    rows=audit_geometry(rows,src['gtf']);prepared=keyed(json.loads((HERE/'prepared_records.json').read_text()),'orf_name')
    oldrows,_=build_rows('old_cds');oldrows=audit_geometry(oldrows,src['gtf'])
    assert len(rows)==len(prepared)
    for r in rows:
        p=prepared[r['orf_name']]
        for k,v in r.items():
            if k in ['blocks','chrom','strand']:continue
            if isinstance(v,(str,bool)):assert v==p[k],(r['orf_name'],k,v,p[k])
            elif v is None:assert p[k] is None
            elif math.isnan(v):assert p[k] is None or math.isnan(p[k])
            else:assert math.isclose(v,p[k],abs_tol=1e-10,rel_tol=1e-11),(r['orf_name'],k,v,p[k])
    checks=[];maxerr=0;computed=[];number_comparisons=0
    for m in RESULT['model_results']:
        dataset=oldrows if m['reference_version']=='legacy' else rows
        selected=select(dataset,m)
        if m['population']=='common_host_rows':
            neweligible=select(rows,dict(m,reference_version='corrected'))
            oldeligible=select(oldrows,dict(m,reference_version='legacy'))
            names={r['orf_name'] for r in neweligible}&{r['orf_name'] for r in oldeligible}
            selected=sorted([r for r in selected if r['orf_name'] in names],key=lambda r:r['orf_name'])
        actualids=[r['orf_name'] for r in selected]
        assert actualids==m['ORF_ids']
        ph=hashlib.sha256(('\n'.join(sorted(actualids))+'\n').encode()).hexdigest();assert ph==m['population_sha256']
        v=fit(selected,m);local=0
        for k,w in v.items():
            if w is None:assert m[k] is None
            else:
                error=abs(w-m[k]);local=max(local,error)
                tolerance=0.0 if k in {'p_t','p_normal'} else 1e-8
                assert math.isclose(w,m[k],abs_tol=tolerance,rel_tol=1e-8),(m['cell'],m['specification'],m['adjustment'],k,w,m[k])
                number_comparisons+=1
        maxerr=max(maxerr,local)
        key={k:m[k] for k in ['cell','stratum','population','reference_version','length_definition','specification','adjustment','cluster']}
        checks.append(dict(**key,status='PASS',maximum_absolute_difference=local));computed.append(dict(**key,**v))
    assert len(checks)==320
    prim=[i for i,m in enumerate(RESULT['model_results']) if 'holm_p_two_principal_cells' in m];assert len(prim)==2
    sorted_i=sorted(prim,key=lambda i:computed[i]['p_t']);running=0;holm={}
    for j,i in enumerate(sorted_i):running=max(running,min(1,(len(prim)-j)*computed[i]['p_t']));holm[i]=running
    for i in prim:assert math.isclose(holm[i],RESULT['model_results'][i]['holm_p_two_principal_cells'],abs_tol=1e-10)
    txgene_pairs=0
    for i,m in enumerate(RESULT['model_results']):
        if m['cluster']!='enst':continue
        keys=['cell','stratum','population','reference_version','length_definition','specification','adjustment']
        js=[j for j,q in enumerate(RESULT['model_results']) if q['cluster']=='ensg' and all(q[k]==m[k] for k in keys)]
        assert len(js)==1
        for k in ['beta','SE','ci95_lo','ci95_hi','p_t']:assert math.isclose(computed[i][k],computed[js[0]][k],abs_tol=1e-10,rel_tol=1e-10)
        txgene_pairs+=1
    assert all(sha(Path(p))==h for p,h in hashes.items())
    out=dict(status='PASS',failure_count=0,completed_utc=datetime.now(timezone.utc).isoformat(),primary_results_sha256=sha(HERE/'results.json'),
             prepared_records_sha256=sha(HERE/'prepared_records.json'),input_audit_sha256=sha(HERE/'input_audit.json'),
             specification_sha256=sha(SPEC_PATH),validator_sha256=sha(Path(__file__)),
             method='Independent CSV/openpyxl source reconstruction; independent GTF interval, phase, completeness and splice-path validation; NumPy OLS and manually assembled CR1; scipy t/normal tails; Patsy only for spline basis; manual Holm.',
             source_hashes=hashes,source_files_unchanged=True,input_rows_independently_reconstructed=len(rows),
             geometry_rows_independently_checked=len(rows)+len(oldrows),legacy_input_rows_independently_reconstructed=len(oldrows),all_model_records_checked=len(checks),
             maximum_absolute_model_difference=maxerr,numerical_comparisons=number_comparisons,
             transcript_gene_partition_identical_pairs=txgene_pairs,primary_holm=[dict(cell=RESULT['model_results'][i]['cell'],p_t=computed[i]['p_t'],holm_p=holm[i]) for i in prim],
             model_checks=checks,
             limitations=['This check independently reconstructs source joins, geometry and regressions, but does not independently recount P-site tracks; corrected canonical track validation is separate.','Same-host nonoverlap does not provide isoform-specific genomic-track assignment.'])
    (HERE/'independent_validation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['source_hashes','model_checks']},indent=2))


if __name__=='__main__':main()

"""Recompute archived models from derived data; fail on any discrepancy."""
from pathlib import Path
import importlib.util
import json
import math
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parents[1]
DATA=None
sys.path.insert(0,str(HERE.parent))
def load(name):
 p=HERE/name/'source_analysis.py'
 s=importlib.util.spec_from_file_location('posthoc_'+name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def table(path):return pd.DataFrame(json.loads(path.read_text()))
def compare(rows,expected,keys,label):
 actual={tuple(r[k] for k in keys):r for r in rows}
 target={tuple(r[k] for k in keys):r for r in expected}
 assert len(actual)==len(rows) and len(target)==len(expected) and set(actual)==set(target),(label,'row keys')
 checks=0;maxdiff=0.
 def value_equal(x,y,location):
  nonlocal checks,maxdiff
  if isinstance(x,(bool,str)) or x is None:assert x==y,(location,x,y)
  elif isinstance(x,(int,float,np.number)):
   assert isinstance(y,(int,float)) and not isinstance(y,bool),(location,'numeric type')
   assert np.isclose(x,y,rtol=1e-8,atol=1e-10),(location,x,y)
   maxdiff=max(maxdiff,abs(float(x)-float(y)));checks+=1
  elif isinstance(x,dict):
   assert isinstance(y,dict) and set(x)==set(y),(location,'dictionary keys')
   for k in x:value_equal(x[k],y[k],location+(k,))
  elif isinstance(x,list):
   assert isinstance(y,list) and len(x)==len(y),(location,'list length')
   for i,(v,w) in enumerate(zip(x,y)):value_equal(v,w,location+(i,))
  else:raise TypeError((location,type(x)))
 for key,a in actual.items():
  b=target[key]
  for field,x in a.items():
   assert field in b,(label,key,'missing reference field',field)
   value_equal(x,b[field],(label,key,field))
 return dict(analysis=label,status='PASS',models=len(rows),numeric_comparisons=checks,max_absolute_difference=maxdiff)

def occupancy(d):
 m=load('occupancy');rows=[];strata=[]
 d=d.copy();d['ln_tx_length']=np.log(d.spliced_length_nt);d['ln_aa']=np.log(d.aa_length);d['gc']=d.transcript_gc
 for cell in ['K562','HepG2']:
  w=d.copy();w['tpm']=w[f'tpm_{cell}_polyA'];w['nrbp']=w[f'{cell}_nrbp_tx'];w['rpf']=w[f'{cell}_rpf_in']
  w=w.loc[w.tpm.ge(1)].dropna(subset=['tpm','nrbp','rpf','gc','aa_length']).copy()
  w['y']=np.log2(w.rpf+1);w['ln_rbp']=np.log1p(w.nrbp);w['ln_tpm']=np.log1p(w.tpm)
  hosts=w.drop_duplicates('tx');q=pd.qcut(hosts.spliced_length_nt,5,labels=False,duplicates='raise')
  w['length_group']=w.tx.map(pd.Series(q.to_numpy()+1,index=hosts.tx)).astype(int)
  for label,terms in m.MODEL_TERMS.items():rows.extend(m.fit_model(w,terms,label,cell))
  single=w.loc[w.tx.map(w.tx.value_counts()).eq(1)]
  rows.extend(m.fit_model(single,m.MODEL_TERMS['plus_log_length'],'single_analysed_ORF_plus_length',cell))
  for group in sorted(w.length_group.unique()):
   sub=w.loc[w.length_group.eq(group)]
   r=m.fit_model(sub,m.MODEL_TERMS['plus_log_length'],f'length_group_{group}',cell)
   for x in r:x.update(length_min_nt=int(sub.spliced_length_nt.min()),length_max_nt=int(sub.spliced_length_nt.max()))
   strata.extend(r)
 ref=json.loads((DATA/'occupancy/results.json').read_text())
 return [compare(rows,ref['model_results'],['cell','model','cluster'],'occupancy'),compare(strata,ref['length_stratum_results'],['cell','model','cluster'],'occupancy_length_strata')]

def structural(d):
 m=load('structural');rows=[]
 for cell in ['K562','HepG2']:
  w=d.copy();w['tpm']=w[f'tpm_{cell}_polyA'];w['nrbp']=w[f'{cell}_nrbp_tx']
  w=w.dropna(subset=['tpm','nrbp','transcript_gc','aa_length']);counts=w.tx.value_counts();single=w.loc[w.tx.isin(counts[counts.eq(1)].index)]
  for outcome in m.CFG['outcomes']:
   for scheme in m.CFG['schemes']:
    data,scales=m.prepare(single if scheme=='single_ORF' else w,outcome)
    for model in m.CFG['models']:rows.append(m.fit_one(data,scales,outcome,cell,scheme,model))
 for model in m.CFG['models']:
  primary=[r for r in rows if r['model']==model and r['scheme']=='transcript']
  for r,p in zip(primary,multipletests([r['tost_p'] for r in primary],method='holm')[1]):r.update(holm_p_primary=float(p),**{'equivalent_0.10_holm':bool(p<m.CFG['alpha'])})
 ref=json.loads((DATA/'structural/results.json').read_text())
 return [compare(rows,ref['model_results'],['cell','model','outcome','scheme'],'structural')]

if __name__=='__main__':
 raise SystemExit('Use posthoc/rerun_corrected.py with an explicit --data-root; this module supplies statistical helpers.')

"""Source-grounded canonical reconstruction under a fixed correction specification.

Outputs are written beside this script. Supply the documented upstream inputs.
Stages prepare/predict/fit are separate so costly predictions are resumable.
"""
from pathlib import Path
import argparse, gzip, re, json, hashlib, sys, time, math, importlib.metadata
from collections import defaultdict
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'shared'))
from ensembl_ids import stable_ensembl_id, pseudoautosomal_base
PACKAGE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Data_S1')
sys.path.insert(0, str(PACKAGE_ROOT))
from input_support import expected_count
PROJECT=Path(__import__('os').environ.get('MICROPROTEIN_UPSTREAM_ROOT', str(BASE.parents[0]/'upstream'))).resolve()
DATA=PROJECT/'analysis/data'
OLD=PROJECT/'analysis/results'
RESULTS=HERE/'results'
EXPECTED_SPEC='a269d3a85af767284f66167c2feabc9d6147bfa09333c805dee9ecd65a15877c'
CELLS=['K562','HepG2']
QUANT={
 'K562':['K562_polyA_main_ENCSR000AEO_rep1_ENCFF720TFN.tsv','K562_polyA_main_ENCSR000AEO_rep2_ENCFF522DDA.tsv'],
 'HepG2':['HepG2_polyA_main_ENCSR985KAT_rep1_ENCFF458NJB.tsv','HepG2_polyA_main_ENCSR985KAT_rep2_ENCFF371KBB.tsv']}
STANDARD=set('ACDEFGHIKLMNPQRSTVWY')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def seqhash(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(name,obj):
    (RESULTS/name).write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else list(v),allow_nan=False)+'\n')
def source_record(p):return dict(path=str(p.relative_to(PROJECT)),sha256=digest(p),bytes=p.stat().st_size)
def read_csv(name):return pd.read_csv(OLD/name,comment='#')

def gtf():
    tx={}
    with gzip.open(DATA/'gencode.v35.annotation.gtf.gz','rt') as f:
        for line in f:
            if line.startswith('#'):continue
            c=line.rstrip().split('\t')
            if c[2] not in ['transcript','exon','CDS','start_codon','stop_codon']:continue
            pairs=re.findall(r'(\w+) "([^"]*)"',c[8]);a=dict(pairs)
            if a.get('gene_type')!='protein_coding':continue
            k=stable_ensembl_id(a['transcript_id']);g=stable_ensembl_id(a['gene_id'])
            d=tx.setdefault(k,dict(enst=k,ensg=g,raw_transcript_ids=set(),raw_gene_ids=set(),genes=set(),chromosomes=set(),strands=set(),tags=[],features=defaultdict(list)))
            d['raw_transcript_ids'].add(a['transcript_id']);d['raw_gene_ids'].add(a['gene_id']);d['genes'].add(g);d['chromosomes'].add(c[0]);d['strands'].add(c[6])
            if c[2]=='transcript':
                if d.get('has_transcript'):raise ValueError('Duplicate normalized transcript record '+k)
                d['has_transcript']=True;d['tags']=[v for key,v in pairs if key=='tag'];d['gene_name']=a.get('gene_name','')
            else:d['features'][c[2]].append((c[0],int(c[3])-1,int(c[4]),c[6],c[7]))
    return {k:v for k,v in tx.items() if v['features']['CDS']}

def interval_length(blocks):return sum(b[2]-b[1] for b in blocks)
def overlap_nt(a,b):return sum(max(0,min(x[2],y[2])-max(x[1],y[1])) for x in a for y in b if x[0]==y[0])
def geometry(t):
    reasons=[];feats=t['features'];cds=feats['CDS'];exons=feats['exon']
    if len(t['genes'])!=1:reasons.append('multiple_genes')
    if len(t['chromosomes'])!=1:reasons.append('multiple_chromosomes')
    if len(t['strands'])!=1 or next(iter(t['strands'])) not in '+-':reasons.append('invalid_strand')
    strand=next(iter(t['strands']));ordered=sorted(cds,key=lambda v:v[1],reverse=strand=='-')
    for name,blocks in [('CDS',cds),('exon',exons)]:
        if not blocks:reasons.append('missing_'+name)
        s=sorted(blocks,key=lambda v:(v[0],v[1]))
        if any(b[2]<=b[1] for b in s):reasons.append('nonpositive_'+name+'_block')
        if any(a[0]==b[0] and a[2]>b[1] for a,b in zip(s,s[1:])):reasons.append('overlapping_'+name+'_blocks')
    nt=interval_length(cds);phase=None;phase_consistent=True;offset=0
    for i,b in enumerate(ordered):
        if b[4] not in ['0','1','2']:phase_consistent=False;continue
        p=int(b[4])
        if i==0:phase=p
        if phase is None or p!=(phase-offset)%3:phase_consistent=False
        offset+=b[2]-b[1]
    if not phase_consistent or phase is None:reasons.append('invalid_CDS_phase')
    if overlap_nt(cds,exons)!=nt:reasons.append('CDS_outside_exons')
    stopover=overlap_nt(cds,feats['stop_codon'])
    if stopover:reasons.append('stop_overlaps_CDS')
    startlen=interval_length(feats['start_codon']);stoplen=interval_length(feats['stop_codon'])
    complete=(not reasons and phase==0 and nt%3==0 and startlen==stoplen==3 and not {'cds_start_NF','cds_end_NF'}.intersection(t['tags']))
    return dict(cds_len=nt,aa_length=nt//3,first_CDS_phase=phase,CDS_phase_consistent=phase_consistent,
        valid_geometry_phase=not reasons,exclusion_reason=';'.join(reasons),strict_complete_CDS=complete,
        cds_start_NF='cds_start_NF' in t['tags'],cds_end_NF='cds_end_NF' in t['tags'],CDS_nt_mod3=nt%3,
        start_codon_nt=startlen,stop_codon_nt=stoplen,stop_CDS_overlap_nt=stopover,
        spliced_length_nt=interval_length(exons),chromosome=next(iter(t['chromosomes'])),strand=strand)

def selections(tx):
    variants={};selected=None
    for collapse in [True,False]:
        for tags in ['last','all']:
            rec={}
            for k,t in tx.items():
                enst=stable_ensembl_id(next(iter(t['raw_transcript_ids'])),policy='legacy_first_dot') if collapse else k
                gene=stable_ensembl_id(next(iter(t['raw_gene_ids'])),policy='legacy_first_dot') if collapse else t['ensg']
                if enst not in rec:
                    principal=('appris_principal_1' in t['tags']) if tags=='all' else bool(t['tags'] and t['tags'][-1]=='appris_principal_1')
                    rec[enst]=dict(enst=enst,ensg=gene,principal=principal,cds_len=0)
                rec[enst]['cds_len']+=interval_length(t['features']['CDS'])
            frame=pd.DataFrame(rec.values()).sort_values('enst')
            frame=frame.sort_values(['ensg','principal','cds_len','enst'],ascending=[True,False,False,True]).groupby('ensg',as_index=False).first()
            for minus in [0,1]:
                s=frame.copy();s['aa_length']=s.cds_len//3-minus;s=s[s.aa_length.between(100,1500)]
                name=f"{'collapsed' if collapse else 'PAR_preserved'}_{tags}_tags_{'legacy' if minus else 'corrected'}"
                ids=sorted(s.enst)
                variants[name]=dict(n=len(s),principal_n=int(s.principal.sum()),ENSTs=ids,ENST_sha256=seqhash('\n'.join(ids)+'\n'))
                if not collapse and tags=='all' and minus==0:selected=s.copy()
    legacy=read_csv('followup2_canonical_cds_per_cds.csv')
    assert set(legacy.enst)==set(variants['collapsed_last_tags_legacy']['ENSTs']),'Legacy selection not exactly reproduced'
    assert len(legacy)==expected_count('legacy_canonical')
    dump('selection_variants.json',variants)
    return selected

def fasta(path,field):
    seqs={};raws={};cur=None;raw=None;buf=[]
    def flush():
        if cur:
            seq=''.join(buf)
            if cur in seqs and seqs[cur]!=seq:raise ValueError('Conflicting normalized FASTA '+cur)
            seqs[cur]=seq;raws[cur]=raw
    with gzip.open(path,'rt') as f:
        for line in f:
            if line.startswith('>'):
                flush();raw=line[1:].strip().split('|')[field];cur=stable_ensembl_id(raw);buf=[]
            else:buf.append(line.strip())
    flush();return seqs,raws

def tpm():
    result={};qc={}
    for cell,files in QUANT.items():
        reps=[]
        for file in files:
            d=pd.read_csv(DATA/'rnaseq'/file,sep='\t',usecols=['transcript_id','TPM'])
            d=d[d.transcript_id.astype(str).str.startswith('ENST')].copy()
            d['enst']=d.transcript_id.map(stable_ensembl_id)
            if d.TPM.isna().any() or (d.TPM<0).any():raise ValueError('Invalid TPM source '+file)
            reps.append(d.groupby('enst').TPM.sum())
        f=pd.concat(reps,axis=1,join='inner');f.columns=['rep1_TPM','rep2_TPM'];f['TPM']=f.mean(axis=1)
        result[cell]=f;qc[cell]=dict(rep1_ENST=len(reps[0]),rep2_ENST=len(reps[1]),shared_ENST=len(f),X_Y_combined=False)
    dump('TPM_join_QC.json',qc);return result

def binding(tx,ids,cell):
    targets=defaultdict(list)
    for i,k in enumerate(ids):
        for chrom,start,end,strand,_ in tx[k]['features']['exon']:targets[(chrom,strand)].append((start,end,i))
    arrays={key:np.asarray(v,dtype=np.int64) for key,v in targets.items()}
    files=sorted((DATA/'eclip'/cell).glob('*.bed.gz'))
    if not files:raise ValueError('Missing eCLIP panel '+cell)
    by_rbp=defaultdict(list)
    for f in files:by_rbp[f.name.split('__')[0].upper()].append(f)
    counts=np.zeros(len(ids),dtype=np.int64);edge=[]
    for rbp,paths in sorted(by_rbp.items()):
        peaks=defaultdict(list)
        for p in paths:
            with gzip.open(p,'rt') as f:
                for line in f:
                    c=line.rstrip().split('\t');start,end=int(c[1]),int(c[2]);strand=c[5]
                    if end<=start or strand not in '+-':raise ValueError('Invalid peak '+str(p))
                    peaks[(c[0],strand)].append((start,end))
        found=set()
        for key,rows in peaks.items():
            if key not in arrays:continue
            merged=[]
            for start,end in sorted(rows):
                if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
                else:merged.append([start,end])
            p=np.asarray(merged);e=arrays[key]
            j=np.searchsorted(p[:,1],e[:,0],side='right');valid=j<len(p)
            hit=np.zeros(len(e),dtype=bool);hit[valid]=p[j[valid],0]<e[valid,1]
            found.update(e[hit,2].tolist())
        for i in found:counts[i]+=1;edge.append((ids[i],rbp))
    pd.DataFrame(edge,columns=['enst','rbp']).sort_values(['enst','rbp']).to_csv(RESULTS/f'{cell}_binding_edges.csv',index=False)
    return dict(zip(ids,counts)),dict(cell=cell,RBP_count=len(by_rbp),peak_files=len(files),edge_count=len(edge))

def psites(tx,ids,cell,metadata):
    import pyBigWig
    bw={s:pyBigWig.open(str(DATA/'riboseq'/f'{cell}_P_sites_{name}.bw')) for s,name in [('+','plus'),('-','minus')]}
    result={};reasons={}
    for k in ids:
        m=metadata[k]
        if not m['valid_geometry_phase']:
            result[k]=np.nan;reasons[k]='invalid_geometry_or_phase';continue
        blocks=sorted(tx[k]['features']['CDS'],key=lambda x:x[1],reverse=m['strand']=='-')
        track=bw[m['strand']];chroms=track.chroms()
        if m['chromosome'] not in chroms:
            result[k]=np.nan;reasons[k]='missing_track_chromosome';continue
        total=0.;offset=0;problem=''
        for chrom,start,end,strand,_ in blocks:
            if start<0 or end>chroms[chrom]:problem='CDS_outside_track_bounds';break
            try:values=track.values(chrom,start,end,numpy=True)
            except RuntimeError:problem='track_read_error';break
            values=np.nan_to_num(values,nan=0.)
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid P-site values '+k)
            if strand=='-':values=values[::-1]
            positions=np.arange(len(values))+offset
            # Exclude the leading partial-codon segment and honour the first phase.
            mask=(positions>=m['first_CDS_phase'])&((positions-m['first_CDS_phase'])%3==0)
            total+=float(values[mask].sum());offset+=len(values)
        result[k]=np.nan if problem else total;reasons[k]=problem
    for t in bw.values():t.close()
    return result,reasons

def prepare():
    print('Parse GTF and select corrected reference',flush=True)
    tx=gtf();sel=selections(tx);ids=sel.enst.tolist();metadata={k:geometry(tx[k]) for k in ids}
    seqs,_=fasta(DATA/'gencode.v35.pc_translations.fa.gz',1)
    rnas,_=fasta(DATA/'gencode.v35.transcripts.fa.gz',0)
    from Bio.SeqUtils import gc_fraction
    parbases={pseudoautosomal_base(k) for k in tx if k.endswith('_PAR_Y')}
    rows=[]
    for k in ids:
        t=tx[k];m=metadata[k]
        rows.append(dict(enst=k,ensg=t['ensg'],raw_transcript_id=';'.join(sorted(t['raw_transcript_ids'])),raw_gene_id=';'.join(sorted(t['raw_gene_ids'])),
            gene_name=t.get('gene_name',''),is_principal='appris_principal_1' in t['tags'],tags=';'.join(t['tags']),
            is_PAR_copy=pseudoautosomal_base(k) in parbases,is_PAR_Y=k.endswith('_PAR_Y'),
            gc=float(gc_fraction(rnas[k])) if k in rnas else np.nan,
            protein_sequence_sha256=seqhash(seqs[k]) if k in seqs else '',actual_protein_length=len(seqs[k]) if k in seqs else np.nan,
            property_exclusion_reason='missing_protein_sequence' if k not in seqs else ('nonstandard_residues:'+''.join(sorted(set(seqs[k])-STANDARD)) if set(seqs[k])-STANDARD else ''),
            CDS_coordinate_sha256=seqhash(json.dumps(t['features']['CDS'],sort_keys=True)),
            exon_coordinate_sha256=seqhash(json.dumps(t['features']['exon'],sort_keys=True)),**m))
    frame=pd.DataFrame(rows).set_index('enst',drop=False)
    tpms=tpm();bindingqc=[]
    for cell in CELLS:
        print('Recount eCLIP and phase-aware P-sites',cell,flush=True)
        binding_values,bqc=binding(tx,ids,cell);bindingqc.append(bqc)
        counts,reasons=psites(tx,ids,cell,metadata)
        frame[cell+'_nrbp']=frame.enst.map(binding_values)
        frame[cell+'_rpf_in']=frame.enst.map(counts)
        frame[cell+'_RPF_exclusion_reason']=frame.enst.map(reasons)
        for col in ['rep1_TPM','rep2_TPM','TPM']:frame[cell+'_'+col]=frame.enst.map(tpms[cell][col])
    frame.to_csv(RESULTS/'canonical_per_cds.csv',index=False)
    dump('binding_panel_QC.json',bindingqc)
    dump('selected_annotation.json',{k:tx[k] for k in ids})
    dump('selected_protein_sequences.json',{k:seqs[k] for k in ids if k in seqs})
    source_paths=[DATA/'gencode.v35.annotation.gtf.gz',DATA/'gencode.v35.pc_translations.fa.gz',DATA/'gencode.v35.transcripts.fa.gz']
    source_paths += [DATA/'rnaseq'/f for files in QUANT.values() for f in files]
    source_paths += sorted((DATA/'riboseq').glob('*_P_sites_*.bw'))
    source_paths += sorted((DATA/'eclip').glob('*/*.bed.gz'))
    source_paths += [OLD/'followup2_canonical_cds_per_cds.csv',OLD/'followup7_canonical_properties.csv']
    dump('source_provenance.json',dict(specification_sha256=EXPECTED_SPEC,shared_ID_helper_sha256=digest(BASE/'shared/ensembl_ids.py'),
        source_files=[source_record(p) for p in source_paths],python=sys.version,numpy=np.__version__,pandas=pd.__version__))
    # Cache is keyed by actual sequence contents; legacy PAR keys resolve to the
    # X FASTA sequence because the source predictor took the first occurrence.
    cache={};conflicts=[]
    for file in ['followup7_canonical_properties.csv']:
        p=read_csv(file)
        cache_file_sha256=digest(OLD/file)
        for r in p.itertuples():
            if pd.isna(r.mean_plddt) or pd.isna(r.mean_disorder):continue
            k=stable_ensembl_id(r.enst)
            if k not in seqs or int(r.aa_seq_len)!=len(seqs[k]):raise ValueError('Prediction cache sequence mismatch '+k)
            seq=seqs[k]
            if set(seq)-STANDARD:raise ValueError('Unexpected nonstandard cached sequence '+k)
            key=seqhash(seq);record=dict(sequence_sha256=key,sequence=seq,mean_plddt=float(r.mean_plddt),mean_disorder=float(r.mean_disorder),
                predictor='metapredict 3.0.2',source_file='analysis/results/'+file,source_file_sha256=cache_file_sha256,source_ENST=k)
            if key in cache:
                if abs(cache[key]['mean_plddt']-record['mean_plddt'])>1e-7 or abs(cache[key]['mean_disorder']-record['mean_disorder'])>1e-9:conflicts.append(key)
            else:cache[key]=record
    if conflicts:raise ValueError('Prediction cache conflicts '+str(len(set(conflicts))))
    dump('prediction_cache.json',cache)
    dump('preparation_QC.json',dict(selected=len(frame),valid_geometry_phase=int(frame.valid_geometry_phase.sum()),
        geometry_phase_excluded=int((~frame.valid_geometry_phase).sum()),strict_complete=int(frame.strict_complete_CDS.sum()),PAR_records=int(frame.is_PAR_copy.sum()),
        sequence_found=int(frame.actual_protein_length.notna().sum()),property_excluded=int(frame.property_exclusion_reason.ne('').sum()),
        unique_selected_valid_sequences=len({seqhash(seqs[k]) for k in ids if k in seqs and not(set(seqs[k])-STANDARD)}),
        uncached_unique_sequences=len({seqhash(seqs[k]) for k in ids if k in seqs and not(set(seqs[k])-STANDARD)}-set(cache)),
        cache_unique_sequences=len(cache),TPM_matched={c:int(frame[c+'_TPM'].notna().sum()) for c in CELLS},
        RPF_valid={c:int(frame[c+'_rpf_in'].notna().sum()) for c in CELLS}))
    print('Prepared',len(frame),'selected reference records',flush=True)

def predict():
    import metapredict,torch
    if importlib.metadata.version('metapredict')!='3.0.2':raise ValueError('Predictor version differs')
    torch.set_num_threads(1)
    from metapredict import predict_disorder,predict_pLDDT
    cache=json.loads((RESULTS/'prediction_cache.json').read_text());seqs=json.loads((RESULTS/'selected_protein_sequences.json').read_text())
    frame=pd.read_csv(RESULTS/'canonical_per_cds.csv')
    # Real cache checks; no mock sequences or invented output.
    checks=[]
    lengths=sorted(cache,key=lambda key:(len(cache[key]['sequence']),key))
    check_keys=[lengths[int(q*(len(lengths)-1))] for q in [0,.25,.5,.75,1]]
    for key in check_keys:
        c=cache[key];s=c['sequence']
        p=float(np.mean(predict_pLDDT(s)));d=float(np.mean(predict_disorder(s)))
        ok=abs(p-c['mean_plddt'])<1e-6 and abs(d-c['mean_disorder'])<1e-7
        checks.append(dict(sequence_sha256=key,sequence_length=len(s),pLDDT_difference=p-c['mean_plddt'],disorder_difference=d-c['mean_disorder'],passed=ok,
            absolute_tolerance_pLDDT=1e-6,absolute_tolerance_disorder=1e-7))
        if not ok:raise ValueError('Real cache prediction verification failed '+key)
    progress=RESULTS/'new_predictions.json'
    new=json.loads(progress.read_text()) if progress.exists() else {}
    wanted={seqhash(s):s for s in seqs.values() if s and not(set(s)-STANDARD)}
    missing=sorted(set(wanted)-set(cache)-set(new))
    for i,key in enumerate(missing,1):
        s=wanted[key]
        new[key]=dict(sequence_sha256=key,mean_plddt=float(np.mean(predict_pLDDT(s))),mean_disorder=float(np.mean(predict_disorder(s))),
            actual_protein_length=len(s),predictor='metapredict 3.0.2')
        if i%25==0 or i==len(missing):
            dump('new_predictions.json',new);print('Predicted',i,'/',len(missing),'new sequences',flush=True)
    audit=[]
    for index,r in frame.iterrows():
        s=seqs.get(r.enst)
        if not s or set(s)-STANDARD:continue
        key=seqhash(s);pred=cache.get(key,new.get(key))
        if pred is None:raise ValueError('Missing real prediction '+r.enst)
        frame.loc[index,'mean_plddt']=pred['mean_plddt'];frame.loc[index,'mean_disorder']=pred['mean_disorder'];frame.loc[index,'aa_seq_len']=len(s)
        frame.loc[index,'prediction_source']='verified_sequence_cache' if key in cache else 'new_metapredict_inference'
        audit.append(dict(enst=r.enst,sequence_sha256=key,source=frame.loc[index,'prediction_source'],source_file=pred.get('source_file'),source_ENST=pred.get('source_ENST')))
    frame.to_csv(RESULTS/'canonical_properties.csv',index=False)
    pkgroot=Path(metapredict.__file__).resolve().parent
    weight_files=sorted(p for p in pkgroot.rglob('*') if p.is_file() and p.suffix.lower() in ['.pt','.pth','.pkl','.onnx','.h5','.safetensors'])
    dump('prediction_provenance.json',dict(metapredict=importlib.metadata.version('metapredict'),torch=torch.__version__,numpy=np.__version__,
        installed_model_files=[dict(package_relative_path=str(p.relative_to(pkgroot)),sha256=digest(p)) for p in weight_files],
        cached_real_sequence_verification=checks,new_unique_predictions=len(new),rows=audit,
        cache_source_method='Version 3.0.2 source predictor documentation, pinned original protein FASTA content/sequence length, exact sequence hash, source property file SHA, five real sequence re-predictions.'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','predict']);args=ap.parse_args()
    if digest(BASE/'specification.json')!=EXPECTED_SPEC:raise ValueError('Fixed specification changed')
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':prepare()
    else:predict()
if __name__=='__main__':main()

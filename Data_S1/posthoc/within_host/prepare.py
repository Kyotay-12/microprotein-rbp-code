"""Read-only source reconstruction and coordinate/length audit before fitting."""
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
from datetime import datetime, timezone
import bisect
import gzip
import json
import re
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent/'specification.json'
ROOT = Path(__import__('os').environ.get('MICROPROTEIN_UPSTREAM_ROOT', str(HERE.parents[1]/'upstream'))).resolve()
sys.path.insert(0, str(HERE.parent/'shared'))
from ensembl_ids import stable_ensembl_id
PACKAGE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'Data_S1')
sys.path.insert(0, str(PACKAGE_ROOT))
from input_support import input_path
FILES = {
    'properties': ROOT/'analysis/results/step0_properties.csv',
    'expression': ROOT/'analysis/results/step1_expression.csv',
    'binding': ROOT/'analysis/results/step1_binding.csv',
    'rpf': ROOT/'analysis/results/step2_rpf_counts.csv',
    'cds': HERE.parent/'canonical/results/canonical_per_cds.csv',
    'old_cds': ROOT/'analysis/results/followup2_canonical_cds_per_cds.csv',
    'id_helper': HERE.parent/'shared/ensembl_ids.py',
    'catalogue': ROOT/'analysis/data/Mudge2022_SupplementaryTables.xlsx',
    'gtf': ROOT/'analysis/data/gencode.v35.annotation.gtf.gz',
    'fasta': ROOT/'analysis/data/gencode.v35.transcripts.fa.gz',
    'legacy_table': ROOT/'analysis/results/followup5_within_transcript_per_orf.csv',
    'legacy_models': ROOT/'analysis/results/followup5_within_transcript_models.csv',
    'legacy_script': ROOT/'analysis/scripts/followup5_within_transcript_allocation.py',
    'cds_builder': ROOT/'analysis/scripts/followup2_canonical_cds.py',
}


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return stable_ensembl_id(str(value))


def merge_intervals(items):
    out = []
    for start, end in sorted(set(items)):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


def overlap(blocks, union):
    ends = [x[1] for x in union]
    total = 0
    for a, b in blocks:
        pos = bisect.bisect_right(ends, a)
        while pos < len(union) and union[pos][0] < b:
            c, d = union[pos]
            total += max(0, min(b, d)-max(a, c))
            pos += 1
    return total


def load_sources(cds_key='cds'):
    d = pd.read_csv(FILES['properties'], comment='#')
    d = d.loc[d.source_sheet.eq('S2') & d.aa_length.le(100)].copy()
    for key, cols in [
        ('expression', ['enst','tpm_K562_polyA','tpm_HepG2_polyA']),
        ('binding', ['K562_nrbp_tx','HepG2_nrbp_tx']),
        ('rpf', ['K562_rpf_in','HepG2_rpf_in']),
    ]:
        other = pd.read_csv(FILES[key])
        d = d.merge(other[['orf_name']+cols], on='orf_name', validate='one_to_one')
    assert (d.transcript.map(stable)==d.enst).all(), 'Existing microprotein ENST must match PAR-preserving normalization'
    cols=['enst','cds_len','aa_length','K562_rpf_in','HepG2_rpf_in']
    if cds_key=='cds':
        cols+=['ensg','valid_geometry_phase','exclusion_reason','strict_complete_CDS','first_CDS_phase','spliced_length_nt']
    cds = pd.read_csv(FILES[cds_key],low_memory=False)[cols]
    if cds_key=='cds':
        cds=cds.rename(columns={'ensg':'canonical_ensg','exclusion_reason':'source_CDS_exclusion_reason','strict_complete_CDS':'source_CDS_complete','first_CDS_phase':'source_CDS_initial_phase','spliced_length_nt':'source_host_length_nt'})
    assert cds.enst.is_unique and cds.enst.map(stable).eq(cds.enst).all()
    cds = cds.rename(columns={'aa_length':'cds_aa_length','K562_rpf_in':'K562_cds_rpf_in','HepG2_rpf_in':'HepG2_cds_rpf_in'})
    d = d.merge(cds, on='enst', validate='many_to_one')
    if cds_key=='old_cds':
        legacy = pd.read_csv(FILES['legacy_table'])
        pd.testing.assert_frame_equal(d.set_index('orf_name')[legacy.columns.drop('orf_name')].sort_index(), legacy.set_index('orf_name').sort_index(), check_dtype=False, rtol=1e-12, atol=1e-12)
    meta = pd.read_csv(FILES['length'])
    assert meta.enst.is_unique
    d = d.merge(meta[['enst','ensg','spliced_length_nt']], on='enst', how='left', validate='many_to_one')
    assert d.spliced_length_nt.notna().all() and d.spliced_length_nt.gt(0).all()
    cat = pd.read_excel(FILES['catalogue'], sheet_name='S2. PHASE I Ribo-seq ORFs')
    wanted = ['orf_name','gene_id','chrm','strand','starts','ends']
    d = d.merge(cat[wanted], on='orf_name', how='left', validate='one_to_one')
    assert (d.transcript.map(stable)==d.enst).all()
    assert (d.gene_id.map(stable)==d.ensg).all()
    if cds_key=='cds':
        assert d.canonical_ensg.eq(d.ensg).all()
        assert d.source_host_length_nt.eq(d.spliced_length_nt).all()
    return d


def geometry(d):
    targets = set(d.enst)
    tx = {key: defaultdict(list) for key in targets}
    global_cds = defaultdict(list)
    with gzip.open(FILES['gtf'], 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            chrom, feature, start, end, strand, phase, attrs = c[0],c[2],int(c[3])-1,int(c[4]),c[6],c[7],c[8]
            if feature=='CDS':
                global_cds[(chrom,strand)].append((start,end))
            if feature not in {'CDS','exon','start_codon','stop_codon','transcript'}:
                continue
            m = re.search(r'transcript_id "([^"]+)"', attrs)
            if not m or stable(m.group(1)) not in targets:
                continue
            key = stable(m.group(1))
            tx[key][feature].append((start,end,phase))
            tx[key]['chroms'].append(chrom)
            tx[key]['strands'].append(strand)
            gene = re.search(r'gene_id "([^"]+)"', attrs)
            tx[key]['genes'].append(stable(gene.group(1)))
            if feature=='transcript':
                tx[key]['tags'] += re.findall(r'tag "([^"]+)"', attrs)
    unions = {key:merge_intervals(blocks) for key,blocks in global_cds.items()}
    audit = []
    for r in d.itertuples():
        t = tx[r.enst]
        assert len(set(t['chroms']))==len(set(t['strands']))==len(set(t['genes']))==1
        chrom = str(r.chrm) if str(r.chrm).startswith('chr') else 'chr'+str(r.chrm)
        assert chrom==t['chroms'][0] and r.strand==t['strands'][0] and r.ensg==t['genes'][0]
        starts = [int(float(x))-1 for x in str(r.starts).split(';')]
        ends = [int(float(x)) for x in str(r.ends).split(';')]
        assert len(starts)==len(ends)
        blocks = sorted(zip(starts,ends))
        assert all(a<b for a,b in blocks)
        cds = sorted(t['CDS'], reverse=r.strand=='-')
        cds_nt = sum(b-a for a,b,_ in cds)
        assert cds_nt==r.cds_len
        assert r.cds_aa_length == cds_nt//3
        exon_union = merge_intervals([(a,b) for a,b,_ in t['exon']])
        assert sum(b-a for a,b in exon_union)==r.spliced_length_nt
        length = sum(b-a for a,b in blocks)
        nc_in_host = overlap(blocks,exon_union)==length
        # Require consecutive host transcript positions, not just membership.
        offsets = []
        base = 0
        for a,b in exon_union:
            offsets.append((a,b,base)); base += b-a
        path = []
        for a,b in blocks:
            for c,e,k in offsets:
                lo,hi=max(a,c),min(b,e)
                if lo<hi:
                    path.append((k+lo-c,k+hi-c))
        merged_path=merge_intervals(path)
        contiguous = nc_in_host and len(merged_path)==1 and merged_path[0][1]-merged_path[0][0]==length
        cumulative=0
        initial_phase=int(cds[0][2]) if cds and cds[0][2]!='.' else None
        phase_valid=initial_phase is not None
        for a,b,p in cds:
            if p=='.' or initial_phase is None or int(p)!=(initial_phase-cumulative)%3:
                phase_valid=False
            cumulative += b-a
        cds_union = merge_intervals([(a,b) for a,b,_ in cds])
        stop = merge_intervals([(a,b) for a,b,_ in t['stop_codon']])
        start = merge_intervals([(a,b) for a,b,_ in t['start_codon']])
        complete = phase_valid and initial_phase==0 and cds_nt%3==0 and sum(b-a for a,b in start)==3 and sum(b-a for a,b in stop)==3 and 'cds_start_NF' not in t['tags'] and 'cds_end_NF' not in t['tags']
        own = overlap(blocks,cds_union)
        any_overlap = overlap(blocks,unions[(chrom,r.strand)])
        stop_cds_overlap = overlap(stop,cds_union)
        raw_exons=[(a,b) for a,b,_ in t['exon']]
        geometry_valid=bool(phase_valid and sum(b-a for a,b in cds_union)==cds_nt and sum(b-a for a,b in raw_exons)==sum(b-a for a,b in exon_union) and overlap(cds_union,exon_union)==cds_nt and stop_cds_overlap==0)
        complete=complete and geometry_valid
        audit.append(dict(orf_name=r.orf_name, own_CDS_overlap_nt=own, any_CDS_overlap_nt=any_overlap,
            nc_in_host_exons=nc_in_host,nc_contiguous_host_path=contiguous,nc_length_nt=length,
            cds_phase_valid=phase_valid,cds_geometry_valid=geometry_valid,cds_complete=complete,stop_CDS_overlap_nt=stop_cds_overlap,
            cds_initial_phase=initial_phase,cds_codon_length=cds_nt//3,
            coordinate_conservative=bool(complete and contiguous and own==0 and any_overlap==0),
            legacy_CDS_one_codon_short=False))
    return pd.DataFrame(audit)


def main():
    FILES.update({
        'old_prepared': input_path('legacy_within_host_records'),
        'old_results': input_path('legacy_within_host_results'),
        'length': input_path('host_transcript_metadata'),
        'length_record': input_path('structural_length_results'),
        'length_validation': input_path('structural_length_validation'),
    })
    before={key:{'path':str(path),'sha256':digest(path)} for key,path in FILES.items()}
    validated=json.loads(FILES['length_validation'].read_text())
    previous=json.loads(FILES['length_record'].read_text())
    assert validated['status']=='PASS' and digest(FILES['length_record'])==validated['primary_results_sha256']
    assert before['length']['sha256']==previous['sources']['length_metadata']['sha256']
    assert before['fasta']['sha256']==previous['sources']['fasta']['sha256']
    old=load_sources('old_cds')
    oldprepared=pd.DataFrame(json.loads(FILES['old_prepared'].read_text()))
    assert set(old.orf_name)==set(oldprepared.orf_name)
    micro_columns=[c for c in old.columns if c not in {'cds_len','cds_aa_length','K562_cds_rpf_in','HepG2_cds_rpf_in'}]
    pd.testing.assert_frame_equal(old.set_index('orf_name')[list(set(micro_columns)-{'orf_name'})].sort_index(),oldprepared.set_index('orf_name')[list(set(micro_columns)-{'orf_name'})].sort_index(),check_dtype=False,rtol=1e-12,atol=1e-12)
    d=load_sources()
    d=d.merge(geometry(d), on='orf_name', validate='one_to_one')
    oldids=set(old.orf_name);newids=set(d.orf_name)
    d['in_legacy_matched']=d.orf_name.isin(oldids)
    micro_shared=sorted(newids&oldids)
    pd.testing.assert_frame_equal(d.set_index('orf_name').loc[micro_shared,list(set(micro_columns)-{'orf_name'})],old.set_index('orf_name').loc[micro_shared,list(set(micro_columns)-{'orf_name'})],check_dtype=False,rtol=1e-12,atol=1e-12)
    # Invalid canonical phase/count records are explicitly excluded from models.
    assert d.cds_complete.eq(d.source_CDS_complete).all()
    assert d.cds_initial_phase.eq(d.source_CDS_initial_phase).all()
    assert d.valid_geometry_phase.eq(d.cds_geometry_valid).all()
    d['canonical_model_eligible']=d.valid_geometry_phase & d[['K562_cds_rpf_in','HepG2_cds_rpf_in']].notna().all(axis=1)
    d['canonical_exclusion_reason']=np.where(~d.valid_geometry_phase,d.source_CDS_exclusion_reason.fillna('unspecified_CDS_geometry_failure'),np.where(d[['K562_cds_rpf_in','HepG2_cds_rpf_in']].isna().any(axis=1),'missing_CDS_signal',''))
    counts=[]
    for cell in ['K562','HepG2']:
        for stratum,types in [('uORF_primary',['uORF']),('uORF_dORF_sensitivity',['uORF','dORF'])]:
            required=[f'tpm_{cell}_polyA',f'{cell}_nrbp_tx',f'{cell}_rpf_in',f'{cell}_cds_rpf_in','transcript_gc','aa_length','cds_aa_length','enst']
            s=d.loc[d.orf_biotype.isin(types)].dropna(subset=required)
            s=s.loc[s[f'tpm_{cell}_polyA'].ge(1)].copy()
            counts.append(dict(cell=cell,stratum=stratum,n=len(s),n_enst=s.enst.nunique(),n_genes=s.ensg.nunique(),
                own_overlap_ORFs=int(s.own_CDS_overlap_nt.gt(0).sum()),any_overlap_ORFs=int(s.any_CDS_overlap_nt.gt(0).sum()),
                invalid_host_path=int((~s.nc_contiguous_host_path).sum()),incomplete_or_invalid_CDS=int((~s.cds_complete).sum()),
                coordinate_conservative=int(s.coordinate_conservative.sum()),legacy_CDS_one_codon_short=int(s.legacy_CDS_one_codon_short.sum()),
                canonical_model_ineligible=int((~s.canonical_model_eligible).sum()),shared_legacy_ORFs=int(s.in_legacy_matched.sum()),
                zero_nc_signal=int(s[f'{cell}_rpf_in'].eq(0).sum()),zero_CDS_signal=int(s[f'{cell}_cds_rpf_in'].eq(0).sum()),
                transcript_gene_one_to_one=bool(s.groupby('enst').ensg.nunique().max()==1 and s.groupby('ensg').enst.nunique().max()==1)))
    shared=d.set_index('orf_name').loc[micro_shared]
    oldshared=old.set_index('orf_name').loc[micro_shared]
    shared_changes={c:int((~np.isclose(shared[c],oldshared[c],rtol=1e-12,atol=1e-12)).sum()) for c in ['cds_len','cds_aa_length','K562_cds_rpf_in','HepG2_cds_rpf_in']}
    output={'completed_utc':datetime.now(timezone.utc).isoformat(),'specification_sha256':digest(SPEC),
        'preparation_script_sha256':digest(Path(__file__)),'sources':before,'legacy_input_table_exactly_reconstructed':True,
        'length_validation_reused_with_matching_sources':True,'exact_match_n':len(d),'cohort_audit':counts,
        'population_transition':{'old_n':len(oldids),'new_n':len(newids),'shared_n':len(micro_shared),'entrants':sorted(newids-oldids),'exits':sorted(oldids-newids)},
        'common_rows_micro_side_identical':True,'shared_CDS_fields_changed':shared_changes}
    # Derived records are separate from source inputs.
    (HERE/'prepared_records.json').write_text(d.to_json(orient='records',indent=2,double_precision=15))
    output['prepared_records_sha256']=digest(HERE/'prepared_records.json')
    assert all(digest(FILES[key])==rec['sha256'] for key,rec in before.items())
    (HERE/'input_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'exact_match_n':len(d),'cohort_audit':counts},indent=2))


if __name__=='__main__':
    main()

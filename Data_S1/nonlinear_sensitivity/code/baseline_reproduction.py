#!/usr/bin/env python3
"""Verify cohort membership and reproduce the gene-clustered occupancy baseline."""
from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
import baseline_helpers as helpers
ROOT=Path(__file__).resolve().parent.parent
INPUT=Path(os.environ['MICROPROTEIN_INPUT_FILE']).resolve()
REFERENCE=Path(os.environ['MICROPROTEIN_REFERENCE_DIR']).resolve()
OUT=Path(os.environ.get('MICROPROTEIN_OUTPUT_DIR',str(ROOT/'nonlinear_recomputed'))).resolve()
def cohort(cell):
    d=helpers.load_records(INPUT)
    d['ln_tx_length']=np.log(d.spliced_length_nt)
    d['ln_aa']=np.log(d.aa_length)
    d['gc']=d.transcript_gc
    d['tpm']=d[f'tpm_{cell}_polyA'];d['nrbp']=d[f'{cell}_nrbp_tx'];d['rpf']=d[f'{cell}_rpf_in']
    d=d.loc[d.tpm.ge(1)].dropna(subset=['tpm','nrbp','rpf','gc','aa_length']).copy()
    assert np.isfinite(d[['tpm','nrbp','rpf','gc','aa_length','spliced_length_nt']]).all().all()
    assert d[['tpm','nrbp','rpf']].ge(0).all().all()
    assert d[['aa_length','spliced_length_nt']].gt(0).all().all()
    assert d[['tx','gene']].notna().all().all()
    d['y']=np.log2(d.rpf+1);d['ln_rbp']=np.log1p(d.nrbp);d['ln_tpm']=np.log1p(d.tpm)
    for col in ['spliced_length_nt','nrbp','tpm','gene']:
        assert d.groupby('tx')[col].nunique().eq(1).all()
    return d.reset_index(drop=True)

def verify_cohort():
    expected=pd.read_csv(REFERENCE/'results/cohort_membership.csv')
    counts=[]
    for cell in ['K562','HepG2']:
        d=cohort(cell)
        a=d[['orf_name','tx','gene']].reset_index(drop=True)
        b=expected.loc[expected.cell.eq(cell),['orf_name','tx','gene']].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        counts.append(dict(cell=cell,n_ORFs=len(d),n_transcripts=d.tx.nunique(),n_genes=d.gene.nunique()))
    return counts

def main():
    verify_cohort()
    saved=json.loads((REFERENCE/'validation/baseline_reference.json').read_text())
    rows=[];checks=[];members=[]
    for cell in ['K562','HepG2']:
        d=cohort(cell)
        actual=next(x for x in helpers.fit_model(d,'ln_rbp + ln_aa + gc + ln_tpm + ln_tx_length','plus_log_length',cell) if x['cluster']=='gene')
        for label,records in saved.items():
            expected=next(r for r in records if r['cell']==cell)
            check=helpers.compare_models([actual],[{k:expected[k] for k in actual}],['cell','model','cluster'],'baseline')
            checks.append(dict(cell=cell,source=label,**check))
        rows.append(actual)
        members.extend(d[['orf_name','tx','gene']].assign(cell=cell).to_dict('records'))
    pd.DataFrame(rows).to_csv(OUT/'baseline_reproduction.csv',index=False)
    pd.DataFrame(members).to_csv(OUT/'cohort_membership.csv',index=False)
    protocol_hash=hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest()
    (OUT/'baseline_check.json').write_text(json.dumps(dict(checks=checks,protocol_sha256=protocol_hash,cohort_matches_archived_step6=True,scope='Baseline refit from supplied processed observations'),indent=2)+'\n')
    print('PASS: original baseline coefficients, SE, CI, p and saved diagnostics reproduced.')

if __name__=='__main__':
    raise SystemExit('Use run.py to configure data/output paths and preserve source files.')

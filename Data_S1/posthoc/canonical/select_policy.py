"""Read-only access to an explicitly named, previously generated selection.

This command cannot change the fixed corrected analysis or rewrite its inputs.
The three policy flags select one entry from the eight-selection audit record.
"""
from pathlib import Path
import argparse,json,hashlib
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--tag-policy',required=True,choices=['last','all'])
ap.add_argument('--par-policy',required=True,choices=['collapsed','preserve'])
ap.add_argument('--cds-length',required=True,choices=['legacy','corrected'])
ap.add_argument('--include-ids',action='store_true',help='Include the full, sorted stable-transcript-ID selection in stdout.')
args=ap.parse_args()
source=HERE/'results/selection_variants.json'
data=json.loads(source.read_text())
name=f"{'collapsed' if args.par_policy=='collapsed' else 'PAR_preserved'}_{args.tag_policy}_tags_{args.cds_length}"
if name not in data:raise SystemExit('Named selection is absent; run the fixed prepare stage first.')
row=data[name];ids=sorted(row['ENSTs'])
h=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
if len(ids)!=row['n'] or len(set(ids))!=len(ids) or h!=row['ENST_sha256']:raise SystemExit('Selection audit file integrity failed.')
provenance=json.loads((HERE/'results/source_provenance.json').read_text())
gtf=next(x for x in provenance['source_files'] if x['path']=='analysis/data/gencode.v35.annotation.gtf.gz')
out=dict(operation='read_only_selection_lookup',policy=name,
    scientific_scope='Selection membership only. Does not imply a full binding/P-site/property/model run under this policy.',
    source_selection_file='canonical/results/selection_variants.json',source_selection_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
    original_GTF=gtf,n=row['n'],principal_n=row['principal_n'],ENST_sha256=h)
if args.include_ids:out['ENSTs']=ids
print(json.dumps(out,indent=2))

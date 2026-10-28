"""Refit separately supplied derived records and compare private reference results."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, math, tempfile, sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from input_support import verify_data_package
DATA=None

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def function_module(path):
    """Load original imports/functions, not the producer's raw-data main block."""
    parsed=ast.parse(path.read_text())
    first_function=min(n.lineno for n in parsed.body if isinstance(n,ast.FunctionDef))
    selected=[n for n in parsed.body if isinstance(n,ast.FunctionDef) or
        (isinstance(n,(ast.Import,ast.ImportFrom)) and n.lineno<first_function)]
    namespace={'__name__':'canonical_model_functions','__file__':str(path)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def compare(actual,expected,keys,label):
    a={tuple(r[k] for k in keys):r for r in actual}
    b={tuple(r[k] for k in keys):r for r in expected}
    if len(a)!=len(actual) or len(b)!=len(expected) or set(a)!=set(b):raise ValueError((label,'model keys'))
    comparisons=0;max_abs=0.
    def equal(x,y,where):
        nonlocal comparisons,max_abs
        if isinstance(x,dict):
            if set(x)!=set(y):raise ValueError((where,'fields'))
            for k in x:equal(x[k],y[k],where+(k,))
        elif isinstance(x,list):
            if x!=y:raise ValueError((where,'ordered membership'))
        elif isinstance(x,(bool,str)) or x is None:
            if x!=y:raise ValueError((where,x,y))
        elif isinstance(x,(int,float,np.number)):
            if not isinstance(y,(int,float,np.number)) or not math.isfinite(float(x)) or not math.isfinite(float(y)):
                raise ValueError((where,'non-finite/type',x,y))
            # A tiny p-value is never accepted merely because it is below an
            # absolute tolerance. Population counts are compared exactly.
            field=where[-1];atol=0. if field in ['p','p_t','p_normal','holm_p_two_principal_cells'] else 1e-10
            if field in ['n','n_genes','n_clusters','design_columns','df_inference']:
                okay=x==y
            else:okay=np.isclose(x,y,rtol=1e-8,atol=atol)
            if not okay:raise ValueError((where,x,y))
            comparisons+=1;max_abs=max(max_abs,abs(float(x)-float(y)))
        else:raise TypeError((where,type(x)))
    for key,row in a.items():
        if set(row)!=set(b[key]):raise ValueError((label,key,'fields'))
        equal(row,b[key],(label,)+key)
    return {'analysis':label,'status':'PASS','models':len(actual),'numeric_comparisons':comparisons,'max_absolute_difference':max_abs}

def canonical():
    mod=function_module(HERE/'canonical/fit_models.py')
    per=pd.read_csv(DATA/'canonical/results/canonical_per_cds.csv',low_memory=False)
    props=pd.read_csv(DATA/'canonical/results/canonical_properties.csv',low_memory=False)
    actual,cohorts=mod['models'](per,props)
    expected=pd.read_csv(DATA/'canonical/results/canonical_models.csv').to_dict('records')
    if cohorts!=json.loads((DATA/'canonical/results/canonical_model_cohorts.json').read_text()):raise ValueError('canonical populations differ')
    return compare(actual,expected,['population','cell','family','outcome','specification'],'corrected_canonical')

def within_host():
    m=module(HERE/'within_host/analyze.py','corrected_within_host')
    d=pd.DataFrame(json.loads((DATA/'within_host/prepared_records.json').read_text()))
    old=pd.DataFrame(json.loads((DATA/'within_host/reference_cohort.json').read_text()))
    rows=[]
    def add(data,cell,stratum,population,version,adjustments):
        for adj in adjustments:
            for kind in m.FORMULAS:
                for cluster in ['enst','ensg']:
                    row=m.fit(data,kind,adj,cluster)
                    row.update(cell=cell,stratum=stratum,population=population,reference_version=version,
                        length_definition='legacy' if version=='legacy' else 'corrected_CDS_codon')
                    rows.append(row)
    for cell in ['K562','HepG2']:
        for stratum in ['uORF_primary','uORF_dORF_sensitivity']:
            new=m.prepare(d,cell,stratum);legacy=m.prepare(old,cell,stratum)
            add(new,cell,stratum,'corrected_reselected','corrected',m.ADJUSTMENTS)
            common=sorted(set(new.orf_name)&set(legacy.orf_name))
            for source,version in [(legacy,'legacy'),(new,'corrected')]:
                same=source.set_index('orf_name',drop=False).loc[common].reset_index(drop=True)
                add(same,cell,stratum,'common_host_rows',version,m.ADJUSTMENTS)
        conservative=m.prepare(d,cell,'uORF_primary',conservative=True)
        add(conservative,cell,'uORF_primary','coordinate_conservative','corrected',['original','plus_log_length'])
    family=[r for r in rows if r['population']=='corrected_reselected' and r['stratum']=='uORF_primary'
        and r['cluster']=='enst' and r['adjustment']=='plus_log_length' and r['specification']=='allocation_cds_adjusted']
    if len(family)!=2:raise ValueError('Holm family changed')
    for row,p in zip(family,multipletests([r['p_t'] for r in family],method='holm')[1]):row['holm_p_two_principal_cells']=float(p)
    expected=json.loads((DATA/'within_host/results.json').read_text())['model_results']
    return compare(rows,expected,['cell','stratum','population','reference_version','specification','adjustment','cluster'],'corrected_within_host')

def main():
    global DATA
    if not __debug__:raise RuntimeError('Run without -O: validation assertions must remain active.')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True,help='Separately supplied complete Data S1 directory')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    data_root=verify_data_package(args.data_root)
    DATA=data_root/'posthoc'
    inputs=[DATA/'inputs/main_orf_records.json',DATA/'within_host/prepared_records.json',DATA/'within_host/reference_cohort.json',DATA/'canonical/results/canonical_per_cds.csv',DATA/'canonical/results/canonical_properties.csv']
    for p in inputs:
        if not p.is_file():raise FileNotFoundError(f'Required separate input: {p}')
    report=args.report or Path(tempfile.mkdtemp(prefix='microprotein_reproduction_'))/'report.json'
    for protected in [HERE.parent.parent,data_root]:
        if report.resolve().is_relative_to(protected):raise ValueError('Write the report outside code and private input directories')
    if report.exists():raise FileExistsError('The report already exists; choose a new path.')
    previous=module(HERE/'support/host_length_reproduction.py','host_length_reproduction');previous.HERE=HERE;previous.DATA=DATA
    records=previous.table(DATA/'inputs/main_orf_records.json')
    checks=previous.occupancy(records)+previous.structural(records)+[canonical(),within_host()]
    result={'status':'PASS','scope':'Model refitting from distributed derived records with the original statistical functions; not independent raw-data reconstruction or an independent statistical implementation.',
        'checks':checks,'total_models':sum(r['models'] for r in checks),
        'derived_input_sha256':{str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    verify_data_package(data_root)
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));print('Reproduction report:',report)

if __name__=='__main__':main()

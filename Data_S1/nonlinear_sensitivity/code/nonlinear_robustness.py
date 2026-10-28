#!/usr/bin/env python3
"""Natural-spline sensitivity analysis of ribosome occupancy."""
from pathlib import Path
import hashlib,json,sys,warnings
import numpy as np
import pandas as pd
import patsy,scipy,statsmodels
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baseline_reproduction import cohort,ROOT,OUT

VARS=['ln_tpm','ln_tx_length']
BASE=['Intercept','ln_rbp','ln_aa','gc','ln_tpm','ln_tx_length']

class Smooth:
    def __init__(self,x,df):
        self.df=df
        self.probs=np.arange(1,df)/df
        self.knots=np.quantile(x,self.probs,method='linear')
        self.bounds=np.array([np.min(x),np.max(x)])
        assert np.unique(np.r_[self.bounds[0],self.knots,self.bounds[1]]).size==df+1
        b=patsy.dmatrix("0 + cr(x, knots=knots, lower_bound=lo, upper_bound=hi, constraints='center')",dict(x=x,knots=self.knots,lo=self.bounds[0],hi=self.bounds[1]))
        self.info=b.design_info
        b=np.asarray(b)
        assert b.shape[1]==df
        z=np.column_stack([np.ones(len(x)),x])
        self.projection=np.linalg.lstsq(z,b,rcond=None)[0]
        residual=b-z@self.projection
        _,sing,v=np.linalg.svd(residual,full_matrices=False)
        assert np.sum(sing>sing[0]*1e-10)==df-1
        self.mapping=v[:df-1].T*np.sqrt(len(x))/sing[:df-1]
        nl=residual@self.mapping
        zfull=np.column_stack([z,nl])
        self.space_error=float(np.max(np.abs(b-zfull@np.linalg.lstsq(zfull,b,rcond=None)[0])))
        self.orth_error=float(np.max(np.abs(z.T@nl)))
        self.linear_error=float(np.max(np.abs(x-np.column_stack([np.ones(len(x)),b])@np.linalg.lstsq(np.column_stack([np.ones(len(x)),b]),x,rcond=None)[0])))
        assert self.space_error<1e-8 and self.orth_error<1e-6 and self.linear_error<1e-8
        factor=next(iter(self.info.factor_infos.values()))
        tr=next(iter(factor.state['transforms'].values()))
        self.center_constraint=tr._constraints.tolist()
    def transform(self,x):
        b=np.asarray(patsy.build_design_matrices([self.info],dict(x=np.asarray(x),knots=self.knots,lo=self.bounds[0],hi=self.bounds[1]))[0])
        return (b-np.column_stack([np.ones(len(x)),x])@self.projection)@self.mapping
    def record(self,cell,var):
        return dict(cell=cell,variable=var,smooth_df=self.df,quantile_probabilities=self.probs.tolist(),interior_knots=self.knots.tolist(),boundary_knots=self.bounds.tolist(),centering_constraint=self.center_constraint,projection=self.projection.tolist(),rotation_scaling=self.mapping.tolist(),column_space_error=self.space_error,orthogonality_error=self.orth_error,linear_null_error=self.linear_error)

def design(d,smooths,model):
    x=pd.DataFrame({'Intercept':np.ones(len(d)),**{c:d[c].to_numpy() for c in BASE[1:]}},index=d.index)
    for var,include in [('ln_tpm',model in ['M1','M3']),('ln_tx_length',model in ['M2','M3'])]:
        if include:
            b=smooths[var].transform(d[var].to_numpy())
            for j in range(b.shape[1]):x[f'{var}_nonlinear_{j+1}']=b[:,j]
    return x

def fit(d,x):
    assert np.linalg.matrix_rank(x)==x.shape[1]
    f=sm.OLS(d.y,x,missing='raise').fit(cov_type='cluster',cov_kwds={'groups':d.gene,'use_correction':True,'df_correction':True},use_t=True)
    assert f.df_resid_inference==d.gene.nunique()-1
    return f

def row(f,d,cell,model,df):
    lo,hi=f.conf_int().loc['ln_rbp']
    return dict(cell=cell,model=model,smooth_df=df,n_ORFs=len(d),n_transcripts=d.tx.nunique(),n_genes=d.gene.nunique(),beta_R=float(f.params.ln_rbp),SE=float(f.bse.ln_rbp),ci95_lo=float(lo),ci95_hi=float(hi),p=float(f.pvalues.ln_rbp),inference_df=int(f.df_resid_inference),R2=float(f.rsquared),adjusted_R2=float(f.rsquared_adj))

def savefig(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    check=json.loads((OUT/'baseline_check.json').read_text())
    assert hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest()==check['protocol_sha256']
    assert all(c['status']=='PASS' for c in check['checks'])
    results=[];tests=[];diagnostics=[];observation=[];knots=[];parameters=[];influences=[];tail=[];tail_members=[];distributions=[];curves=[];correlations=[];coef=[]
    fits={};data={};smooth_cache={}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for cell in ['K562','HepG2']:
            d=cohort(cell);data[cell]=d;hosts=d.drop_duplicates('tx')
            for var in ['tpm','spliced_length_nt','nrbp']:
                q=hosts[var].quantile([0,.01,.25,.5,.75,.95,.99,1])
                distributions.append(dict(cell=cell,variable=var,unit='unique_host',n=len(hosts),**{f'q{p:g}':float(v) for p,v in q.items()}))
            for var in ['tpm','spliced_length_nt']:
                rho=stats.spearmanr(hosts[var],hosts.nrbp).statistic
                correlations.append(dict(cell=cell,variable=var,rho_with_nRBP=float(rho),n_hosts=len(hosts)))
            smooth_cache[cell]={}
            configs=[('M0',0)]+[(m,df) for df in [4,3,5] for m in ['M1','M2','M3']]
            for df in [4,3,5]:
                smooth_cache[cell][df]={}
                for var in VARS:
                    sp=Smooth(d[var].to_numpy(),df);smooth_cache[cell][df][var]=sp
                    parameters.append(sp.record(cell,var))
                    inverse=np.expm1 if var=='ln_tpm' else np.exp
                    for prob,value in zip(np.r_[0,sp.probs,1],np.r_[sp.bounds[0],sp.knots,sp.bounds[1]]):
                        knots.append(dict(cell=cell,variable=var,smooth_df=df,quantile=float(prob),transformed_value=float(value),original_value=float(inverse(value)),kind='boundary' if prob in [0,1] else 'interior'))
            for model,df in configs:
                sp=smooth_cache[cell].get(df,{})
                x=design(d,sp,model);f=fit(d,x);fits[cell,model,df]=f
                r=row(f,d,cell,model,df);results.append(r)
                beta0=results[-1]['beta_R'] if model=='M0' else fits[cell,'M0',0].params.ln_rbp
                r.update(delta_beta=r['beta_R']-beta0,absolute_delta_beta=abs(r['beta_R']-beta0),relative_delta_beta=(r['beta_R']-beta0)/abs(beta0))
                # Check spline fit is identical to direct centered-cr parameterization.
                raw=x[BASE].drop(columns=[v for v in VARS if any(c.startswith(v+'_nonlinear') for c in x)]).to_numpy()
                for var in VARS:
                    if any(c.startswith(var+'_nonlinear') for c in x):
                        raw=np.column_stack([raw,np.asarray(patsy.build_design_matrices([sp[var].info],dict(x=d[var].to_numpy(),knots=sp[var].knots,lo=sp[var].bounds[0],hi=sp[var].bounds[1]))[0])])
                direct=np.linalg.lstsq(raw,d.y,rcond=None)[0]
                assert np.allclose(raw@direct,f.fittedvalues,atol=1e-9,rtol=1e-9)
                # OLS influence screens; inference remains clustered.
                ordinary=sm.OLS(d.y,x).fit();inf=ordinary.get_influence();h=inf.hat_matrix_diag
                xs=x.to_numpy().copy();xs[:,1:]=(xs[:,1:]-xs[:,1:].mean(axis=0))/xs[:,1:].std(axis=0)
                diagnostics.append(dict(cell=cell,model=model,smooth_df=df,n_columns=x.shape[1],rank=np.linalg.matrix_rank(x),condition_number_raw=np.linalg.cond(x),condition_number_standardized=np.linalg.cond(xs),max_leverage=h.max(),leverage_2p_n=int((h>2*x.shape[1]/len(d)).sum()),leverage_3p_n=int((h>3*x.shape[1]/len(d)).sum()),max_cooks_distance=inf.cooks_distance[0].max(),residual_sd=np.std(f.resid,ddof=x.shape[1]),natural_basis_prediction_max_difference=float(np.max(np.abs(raw@direct-f.fittedvalues)))))
                obs=d[['orf_name','tx','gene','tpm','spliced_length_nt','nrbp']].copy()
                obs=obs.assign(cell=cell,model=model,smooth_df=df,leverage=h,cooks_distance=inf.cooks_distance[0],residual=f.resid.to_numpy())
                observation.extend(obs.to_dict('records'))
                for term,value in f.params.items():coef.append(dict(cell=cell,model=model,smooth_df=df,term=term,beta=value,SE=f.bse[term]))
                blocks={var:[i for i,c in enumerate(x.columns) if c.startswith(var+'_nonlinear')] for var in VARS}
                if model=='M3':blocks['both_joint']=blocks[VARS[0]]+blocks[VARS[1]]
                for var,ix in blocks.items():
                    if not ix:continue
                    restriction=np.eye(x.shape[1])[ix]
                    contrast=restriction@f.cov_params().to_numpy()@restriction.T
                    assert np.linalg.matrix_rank(contrast)==len(ix)
                    w=f.f_test(restriction)
                    assert w.df_denom==d.gene.nunique()-1 and w.df_num==len(ix)
                    manual=float((restriction@f.params).T@np.linalg.solve(contrast,restriction@f.params)/len(ix))
                    assert np.isclose(float(w.fvalue),manual,rtol=1e-9)
                    tests.append(dict(cell=cell,model=model,smooth_df=df,nonlinearity_block=var,F=float(w.fvalue),numerator_df=int(w.df_num),denominator_df=int(w.df_denom),p=float(w.pvalue),null='nonlinear components = 0; linear terms retained'))
            # Complete predeclared tail sensitivity, using full-cohort mapping.
            thresholds={v:float(hosts[v].quantile(.99,interpolation='linear')) for v in ['tpm','spliced_length_nt','nrbp']}
            bad={v:set(d.loc[d[v].gt(cut),'gene']) for v,cut in thresholds.items()}
            bad['union']=set.union(*bad.values())
            for rule,genes in bad.items():
                subset=d.loc[~d.gene.isin(genes)]
                subrows=[]
                for model,df in [('M0',0),('M3',4)]:
                    x=design(subset,smooth_cache[cell][4],model);f=fit(subset,x)
                    r=row(f,subset,cell,model,df)
                    full=fits[cell,model,df].params.ln_rbp
                    r.update(rule=rule,threshold=thresholds.get(rule,np.nan),n_excluded_ORFs=len(d)-len(subset),n_excluded_genes=d.gene.nunique()-subset.gene.nunique(),n_excluded_transcripts=d.tx.nunique()-subset.tx.nunique(),delta_beta_vs_corresponding_full=r['beta_R']-full,relative_delta_vs_corresponding_full=(r['beta_R']-full)/abs(full))
                    subrows.append(r)
                for r in subrows:r['relative_delta_vs_subset_M0']=(r['beta_R']-subrows[0]['beta_R'])/abs(subrows[0]['beta_R'])
                tail.extend(subrows)
                tail_members.extend(d.loc[d.gene.isin(genes),['orf_name','tx','gene']].assign(cell=cell,rule=rule).to_dict('records'))
            # Exact gene deletion through normal-equation block update; validate extremes by explicit refit.
            for model,df in [('M0',0),('M3',4)]:
                x=design(d,smooth_cache[cell][4],model);a=x.to_numpy();y=d.y.to_numpy();ata=a.T@a;aty=a.T@y;b=fits[cell,model,df].params.ln_rbp
                local=[]
                for gene,ix in d.groupby('gene').indices.items():
                    ag=a[ix];yg=y[ix]
                    deleted=np.linalg.solve(ata-ag.T@ag,aty-ag.T@yg)[1]
                    local.append(dict(cell=cell,model=model,smooth_df=df,deleted_gene=gene,n_deleted_ORFs=len(ix),beta_without_gene=deleted,delta_beta=deleted-b,relative_delta_beta=(deleted-b)/abs(b)))
                for r in [min(local,key=lambda z:z['beta_without_gene']),max(local,key=lambda z:z['beta_without_gene'])]:
                    keep=d.gene.ne(r['deleted_gene'])
                    explicit=np.linalg.lstsq(a[keep],y[keep],rcond=None)[0][1]
                    assert np.isclose(explicit,r['beta_without_gene'],atol=1e-9)
                influences.extend(local)
            # Mean curves on observed support; no spline refit on prediction grid.
            fixed={c:float(hosts[c].median()) for c in ['ln_rbp','gc','ln_tpm','ln_tx_length']};fixed['ln_aa']=float(d.ln_aa.median())
            for var in VARS:
                grid=np.linspace(d[var].min(),d[var].max(),240)
                new=pd.DataFrame({c:np.repeat(v,len(grid)) for c,v in fixed.items()});new[var]=grid
                q1,q99=hosts[var].quantile([.01,.99])
                for model,df in [('M0',0),('M3',4)]:
                    f=fits[cell,model,df];xx=design(new,smooth_cache[cell][4],model)
                    mean=np.asarray(xx@f.params);se=np.sqrt(np.einsum('ij,jk,ik->i',xx,f.cov_params(),xx));critical=stats.t.ppf(.975,f.df_resid_inference)
                    for i,value in enumerate(grid):curves.append(dict(cell=cell,variable=var,model=model,transformed_x=value,original_x=float(np.expm1(value) if var=='ln_tpm' else np.exp(value)),mean_prediction=mean[i],ci95_lo=mean[i]-critical*se[i],ci95_hi=mean[i]+critical*se[i],sparse_tail=bool(value<q1 or value>q99),**{'fixed_'+c:v for c,v in fixed.items() if c!=var}))
        warn=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
    tables={'model_results':results,'nonlinearity_tests':tests,'model_diagnostics':diagnostics,'observation_diagnostics':observation,'spline_knots':knots,'all_coefficients':coef,'gene_deletion_influence':influences,'tail_sensitivity':tail,'tail_excluded_membership':tail_members,'host_distributions':distributions,'host_RBP_correlations':correlations,'adjusted_curves':curves}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    pd.DataFrame(results).loc[lambda x:x.smooth_df.isin([0,4])].to_csv(OUT/'primary_coefficient_comparison.csv',index=False)
    pd.DataFrame(results).loc[lambda x:x.smooth_df.gt(0)].to_csv(OUT/'df_sensitivity.csv',index=False)
    (OUT/'spline_parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(protocol_sha256=check['protocol_sha256'],python=sys.version,versions=dict(numpy=np.__version__,pandas=pd.__version__,patsy=patsy.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__,matplotlib=matplotlib.__version__),warnings=warn,tail_quantile_unit='unique_host',knot_quantile_unit='ORF',inference='unweighted OLS; gene CR1; t/F G-1',models=len(results)),indent=2)+'\n')
    # Figures generated only after all fixed fits are complete.
    res=pd.DataFrame(results);curve=pd.DataFrame(curves)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharex=True)
    for ax,cell in zip(axs,['K562','HepG2']):
        for df,color,offset in [(3,'#82b3cc',-.16),(4,'#23658a',0),(5,'#d58b39',.16)]:
            g=res.loc[res.cell.eq(cell)&res.smooth_df.eq(df)]
            xx=np.array([int(v[1]) for v in g.model])+offset
            ax.errorbar(g.beta_R,xx,xerr=np.vstack([g.beta_R-g.ci95_lo,g.ci95_hi-g.beta_R]),fmt='o',color=color,label=f'Smooth df={df}',capsize=2)
        b=res.loc[res.cell.eq(cell)&res.model.eq('M0')].iloc[0]
        ax.errorbar(b.beta_R,0,xerr=[[b.beta_R-b.ci95_lo],[b.ci95_hi-b.beta_R]],fmt='s',color='black',capsize=3,label='M0 baseline')
        ax.axvline(b.beta_R,color='gray',ls=':',lw=1);ax.axvline(0,color='gray',lw=1)
        ax.set(yticks=[0,1,2,3],yticklabels=['M0: linear','M1: TPM spline','M2: length spline','M3: both splines'],xlabel='RBP coefficient (95% gene-cluster t CI)',title=cell)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axs[1].legend(fontsize=8,loc='lower left');savefig(fig,'rbp_coefficient_comparison')
    fig,axs=plt.subplots(2,2,figsize=(11,7.5))
    for ci,cell in enumerate(['K562','HepG2']):
        hosts=data[cell].drop_duplicates('tx')
        for vi,var in enumerate(VARS):
            ax=axs[ci,vi];inv=np.expm1 if var=='ln_tpm' else np.exp
            q1,q99=hosts[var].quantile([.01,.99]);lo,hi=data[cell][var].agg(['min','max'])
            for model,color in [('M0','#777777'),('M3','#23658a')]:
                g=curve.loc[curve.cell.eq(cell)&curve.variable.eq(var)&curve.model.eq(model)]
                ax.plot(g.original_x,g.mean_prediction,label=model,color=color)
                ax.fill_between(g.original_x,g.ci95_lo,g.ci95_hi,color=color,alpha=.12)
            ax.axvspan(inv(lo),inv(q1),color='orange',alpha=.12);ax.axvspan(inv(q99),inv(hi),color='orange',alpha=.12)
            ax.plot(inv(hosts[var]),np.repeat(ax.get_ylim()[0],len(hosts)),'|',alpha=.12,color='black',markersize=3)
            ax.set(xscale='log',xlabel='TPM' if var=='ln_tpm' else 'Host RNA length (nt)',ylabel='Predicted log2(RPF + 1)',title=f'{cell} | fixed median covariates')
            ax.legend(fontsize=8);ax.grid(alpha=.15)
    savefig(fig,'adjusted_TPM_length_curves')
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ci,cell in enumerate(['K562','HepG2']):
        hosts=data[cell].drop_duplicates('tx')
        axs[ci,0].hist(hosts.ln_tpm,bins=35,color='#23658a',alpha=.8)
        axs[ci,0].set(xlabel='ln(1 + TPM)',ylabel='Number of hosts',title=f'{cell}: TPM distribution')
        for vi,var in enumerate(VARS):
            ax=axs[ci,vi+1];ax.scatter(hosts[var],hosts.nrbp,s=8,alpha=.25,color='#23658a')
            ax.set(xlabel='ln(1 + TPM)' if var=='ln_tpm' else 'ln(host RNA length)',ylabel='Host RBP count',title=cell)
    savefig(fig,'host_covariate_diagnostics')
    print(res[['cell','model','smooth_df','beta_R','ci95_lo','ci95_hi','p','relative_delta_beta']].to_string(index=False))
    print('Warnings:',warn)
if __name__=='__main__':main()

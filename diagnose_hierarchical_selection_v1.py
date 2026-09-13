from pathlib import Path
import pandas as pd,numpy as np,json
root=Path(__file__).resolve().parent;run=root/'data/model_runs/hierarchical_fund_v1/20260913T034800Z';out=root/'data/audit/hierarchical_selection_v1/20260913T034800Z';out.mkdir(parents=True,exist_ok=False)
reg=pd.read_csv(root/'data/model_ready_hierarchical_v1/20260913T033945Z/registry.csv').set_index('Instrument');rows=[];details=[]
for fam in ['logistic','random_forest']:
    for layer in ['sleeve','stock']:
        p=pd.read_csv(run/f'{fam}_{layer}_predictions.csv');p=p[p.prediction.notna()]
        target=pd.read_csv(run/f'{layer}_targets.csv')
        key='primary_group' if layer=='sleeve' else 'Instrument';y='sleeve_excess_ai' if layer=='sleeve' else 'stock_excess_sleeve'
        x=p.rename(columns={'entity':key}).merge(target,on=['Date',key],validate='one_to_one').dropna(subset=[y])
        if layer=='stock':x['primary_group']=x.Instrument.map(reg.primary_group)
        groupkeys=['Date'] if layer=='sleeve' else ['Date','primary_group']
        for keys,z in x.groupby(groupkeys):
            date=keys[0] if isinstance(keys,tuple) else keys
            if len(z)<2 or z.prediction.nunique()<2 or z[y].nunique()<2:continue
            ic=z.prediction.rank().corr(z[y].rank());details.append(dict(family=fam,layer=layer,Date=date,group=keys[1] if layer=='stock' else 'all_sleeves',count=len(z),rank_ic=ic))
        dd=pd.DataFrame(details);v=dd[dd.family.eq(fam)&dd.layer.eq(layer)].groupby('Date').rank_ic.mean().sort_index().to_numpy();n=len(v);center=v-v.mean();L=20
        longvar=np.dot(center,center)/n
        for k in range(1,min(L,n-1)+1):longvar+=2*(1-k/(L+1))*np.dot(center[k:],center[:-k])/n
        se=np.sqrt(max(longvar,0)/n);rng=np.random.default_rng(5360);means=[]
        for _ in range(500):
            starts=rng.integers(0,n-L+1,size=int(np.ceil(n/L)));ii=np.concatenate([np.arange(s,s+L) for s in starts])[:n];means.append(v[ii].mean())
        rows.append(dict(family=fam,layer=layer,days=n,mean_daily_rank_ic=v.mean(),hac20_se=se,hac20_t=v.mean()/se if se>0 else np.nan,block20_ci_low=np.quantile(means,.025),block20_ci_high=np.quantile(means,.975),note='stock IC averaged equally across eligible nonconstant multi-stock sleeves each day; single-stock Network not assessable'))
pd.DataFrame(details).to_csv(out/'daily_group_rank_ic.csv',index=False);pd.DataFrame(rows).to_csv(out/'rank_ic_summary.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))

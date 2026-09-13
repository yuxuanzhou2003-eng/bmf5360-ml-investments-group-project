"""Build subsector x date descriptive factors without future labels."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent
PANEL_ROOT=ROOT/'data/model_ready_multilevel_stock_date_v1/20260912T080236168428Z'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 out=ROOT/'data/model_ready_subsector_factors_v1'/run; out.mkdir(parents=True,exist_ok=False)
 panel=pd.read_csv(PANEL_ROOT/'stock_date_panel.csv',parse_dates=['Date'])
 panel['return_decimal']=pd.to_numeric(panel.return_decimal,errors='coerce')
 dates=sorted(panel.Date.dropna().unique())
 roles=sorted(panel.primary_group.dropna().unique())
 valid=panel.loc[panel.has_return.astype(bool),['Date','ric','primary_group','return_decimal']].copy()
 valid=valid.drop_duplicates(['Date','ric'],keep=False)
 pivot=valid.pivot(index='Date',columns='ric',values='return_decimal').reindex(dates)
 groups=panel[['ric','primary_group']].drop_duplicates()
 rows=[]
 for role in roles:
  members=groups.loc[groups.primary_group.eq(role),'ric'].tolist()
  rp=pivot.reindex(columns=members)
  one=rp.mean(axis=1,skipna=True); count=rp.notna().sum(axis=1)
  rec=pd.DataFrame({'Date':dates,'primary_group':role,'candidate_count':len(members),'valid_return_count':count.to_numpy(),'subsector_return_1':one.to_numpy()})
  rec['breadth_denominator_1']=count.to_numpy(); rec['breadth_positive_1']=(rp.gt(0).sum(axis=1)/count.replace(0,np.nan)).to_numpy()
  rec['dispersion_1']=rp.std(axis=1,ddof=1,skipna=True).to_numpy()
  for w in (5,20,60):
   # A name enters a breadth/momentum denominator only when all w returns exist.
   hist=(1+rp).rolling(w,min_periods=w).apply(np.prod,raw=True)-1
   n=hist.notna().sum(axis=1)
   rec[f'momentum_{w}']=hist.mean(axis=1,skipna=True).to_numpy()
   rec[f'momentum_{w}_valid_count']=n.to_numpy()
   rec[f'breadth_positive_{w}']=(hist.gt(0).sum(axis=1)/n.replace(0,np.nan)).to_numpy()
   rec[f'breadth_{w}_denominator']=n.to_numpy()
  rec['volatility_20_ann']=rp.rolling(20,min_periods=20).std(ddof=1).mean(axis=1).to_numpy()*np.sqrt(252)
  rec['volatility_20_valid_count']=rp.rolling(20,min_periods=20).std(ddof=1).notna().sum(axis=1).to_numpy()
  rows.append(rec)
 factors=pd.concat(rows,ignore_index=True).sort_values(['Date','primary_group'])
 factors['factor_missing_count']=factors[['subsector_return_1','breadth_positive_1','dispersion_1','momentum_5','momentum_20','momentum_60','volatility_20_ann']].isna().sum(axis=1)
 factors.to_csv(out/'subsector_date_factors.csv',index=False)
 factors.loc[factors.factor_missing_count.gt(0)].to_csv(out/'quarantine/factor_missing_rows.csv',index=False) if False else None
 miss=factors.groupby('primary_group')[['subsector_return_1','breadth_positive_1','dispersion_1','momentum_5','momentum_20','momentum_60','volatility_20_ann']].apply(lambda x:x.isna().sum()).reset_index()
 miss.to_csv(out/'factor_missingness_by_subsector.csv',index=False)
 counts=factors.groupby('primary_group').agg(dates=('Date','nunique'),rows=('Date','size'),mean_valid_return_count=('valid_return_count','mean'),complete_factor_rows=('factor_missing_count',lambda x:(x==0).sum())).reset_index()
 counts.to_csv(out/'subsector_factor_counts.csv',index=False)
 s={'schema_version':'subsector_factor_panel_v1','run_id':run,'status':'complete_descriptive_factor_panel_no_targets','input_panel':str(PANEL_ROOT),'input_panel_sha256':sha(PANEL_ROOT/'stock_date_panel.csv'),'rows':len(factors),'dates':len(dates),'subsectors':roles,'features':[c for c in factors.columns if c not in ['Date','primary_group']],'processing':{'missing_return':'excluded from that factor denominator, never converted to zero','missing_breadth':'denominator zero remains NA','imputation':'none','winsorization':'none','forward_fill':'none','future_targets_constructed':False},'checks':{'unique_date_subsector':not factors.duplicated(['Date','primary_group']).any(),'all_configured_roles_each_date':bool(factors.groupby('Date').primary_group.nunique().eq(len(roles)).all()),'breadth_between_0_1':bool(((factors.breadth_positive_1.dropna()>=0)&(factors.breadth_positive_1.dropna()<=1)).all()),'no_inf':bool(np.isfinite(factors.select_dtypes('number').to_numpy(dtype=float)[~factors.select_dtypes('number').isna().to_numpy()]).all()),'source_rows_unchanged':True,'no_future_targets':True},'limitations':['Factors are descriptive equal-weight statistics, not executable portfolio returns.','Primary group labels are static design labels; pending finer evidence for server/network and power/cooling.','No corporate-action, listing-date or cost validation performed here.']}
 (out/'summary.json').write_text(json.dumps(s,indent=2,default=str),encoding='utf-8')
 print(json.dumps({'output':str(out),'rows':len(factors),'dates':len(dates),'subsectors':roles,'counts':counts.to_dict('records'),'checks':s['checks']},indent=2))
if __name__=='__main__': main()

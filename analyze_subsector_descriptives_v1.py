"""Descriptive subsector analysis; no targets, ranking or model selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'data/model_ready_subsector_factors_v1/20260912T080445965839Z/subsector_date_factors.csv'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); out=ROOT/'data/analysis/subsector_descriptives_v1'/run; out.mkdir(parents=True,exist_ok=False)
 f=pd.read_csv(INPUT,parse_dates=['Date'])
 daily=f.pivot(index='Date',columns='primary_group',values='subsector_return_1').sort_index()
 rows=[]
 for g in daily:
  s=daily[g].dropna(); wealth=(1+s).cumprod(); dd=wealth/wealth.cummax()-1
  rows.append({'primary_group':g,'observations':len(s),'mean_daily_return':s.mean(),'daily_volatility':s.std(ddof=1),'annualized_return_simple':s.mean()*252,'annualized_volatility':s.std(ddof=1)*np.sqrt(252),'total_return':wealth.iloc[-1]-1,'max_drawdown':dd.min(),'positive_day_fraction':(s>0).mean(),'first_date':s.index.min(),'last_date':s.index.max()})
 summary=pd.DataFrame(rows).sort_values('primary_group'); summary.to_csv(out/'subsector_performance_summary.csv',index=False)
 corr=daily.corr(min_periods=60); corr.to_csv(out/'subsector_return_correlation.csv')
 breadth=f.pivot(index='Date',columns='primary_group',values='breadth_positive_20').sort_index(); breadth.describe().T.to_csv(out/'subsector_breadth_summary.csv')
 vol=f.pivot(index='Date',columns='primary_group',values='volatility_20_ann').sort_index(); vol.describe().T.to_csv(out/'subsector_volatility_summary.csv')
 s={'schema_version':'subsector_descriptives_v1','run_id':run,'status':'complete_descriptive_no_targets','input':str(INPUT),'input_sha256':sha(INPUT),'rows':len(f),'dates':f.Date.nunique(),'groups':f.primary_group.nunique(),'processing':{'rows_deleted':0,'imputation':'none','winsorization':'none','forward_fill':'none','target_read':False,'model_selection':False},'checks':{'unique_date_group':not f.duplicated(['Date','primary_group']).any(),'all_returns_finite':bool(np.isfinite(daily.to_numpy(dtype=float)[~daily.isna().to_numpy()]).all()),'all_groups_retained':set(f.primary_group)==set(daily.columns),'no_future_targets':True},'limitations':['Descriptive in-sample period statistics do not establish predictive power.','Return averages use valid observations and are not executable start-of-period portfolio returns.','Static role labels and historical membership limitations remain.']}
 (out/'summary.json').write_text(json.dumps(s,indent=2,default=str),encoding='utf-8')
 print(json.dumps({'output':str(out),'summary':summary.to_dict('records'),'checks':s['checks']},indent=2,default=str))
if __name__=='__main__': main()

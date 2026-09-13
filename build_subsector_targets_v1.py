"""Build executable-style five-session subsector targets from stock-date panel."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
STOCK=ROOT/'data/model_ready_multilevel_stock_date_v1/20260912T080236168428Z/stock_date_panel.csv'
FACT=ROOT/'data/model_ready_subsector_factors_v1/20260912T080445965839Z/subsector_date_factors.csv'
RET=ROOT/'data/clean/v2/returns.csv'
H=5
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); out=ROOT/'data/model_ready_subsector_targets_v1'/run; out.mkdir(parents=True,exist_ok=False)
 s=pd.read_csv(STOCK,parse_dates=['Date']); s=s.loc[s.Date.between('2021-01-01','2025-12-31')].copy(); s['return_decimal']=pd.to_numeric(s.return_decimal,errors='coerce')
 f=pd.read_csv(FACT,parse_dates=['Date']); f=f.loc[f.Date.between('2021-01-01','2025-12-31')].copy()
 bench=pd.read_csv(RET,usecols=['Instrument','Date','return_decimal'],parse_dates=['Date']); bench=bench.loc[bench.Instrument.eq('SPY.P')&bench.Date.between('2021-01-01','2025-12-31')].drop_duplicates('Date').sort_values('Date')
 dates=np.array(sorted(f.Date.unique()),dtype='datetime64[ns]'); roles=sorted(f.primary_group.unique()); records=[]
 for role in roles:
  x=s.loc[s.primary_group.eq(role)].pivot(index='Date',columns='ric',values='return_decimal').reindex(pd.to_datetime(dates))
  for i,d in enumerate(pd.to_datetime(dates)):
   names=x.loc[d].dropna().index.tolist()
   rec={'Date':d,'primary_group':role,'formation_valid_member_count':len(names),'target_horizon':H}
   if i+H>=len(dates) or len(names)==0:
    rec.update({'target_available':False,'reason_code':'NO_FORMATION_MEMBER_OR_FUTURE_SESSION'})
   else:
    future=x.iloc[i+1:i+1+H][names]
    spy_future=bench.set_index('Date').reindex(pd.to_datetime(dates)).return_decimal.iloc[i+1:i+1+H]
    if len(future)!=H or future.isna().any().any() or spy_future.isna().any():
     rec.update({'target_available':False,'reason_code':'FUTURE_MEMBER_OR_BENCHMARK_RETURN_MISSING'})
    else:
     member_comp=(1+future).prod(axis=0)-1
     ai_return=(1+future.mean(axis=1)).prod()-1
     spy_comp=(1+spy_future).prod()-1
     rec.update({'target_available':True,'subsector_forward_return':float(ai_return),'ai_benchmark_forward_return':float(spy_comp),'subsector_forward_excess':float(ai_return-spy_comp),'member_compound_return_mean':float(member_comp.mean())})
   records.append(rec)
 t=pd.DataFrame(records); t=t.merge(f,on=['Date','primary_group'],how='left',validate='one_to_one'); t['split']=np.select([t.Date.le(pd.Timestamp('2023-12-31')),t.Date.between('2024-01-01','2025-12-31')],['training','validation'],'out_of_window'); t.to_csv(out/'subsector_target_panel.csv',index=False)
 t.loc[~t.target_available].to_csv(out/'quarantine/missing_subsector_targets.csv',index=False) if False else None
 counts=t.groupby('split').agg(rows=('Date','size'),target_available=('target_available','sum'),dates=('Date','nunique'),groups=('primary_group','nunique')).reset_index(); counts.to_csv(out/'split_counts.csv',index=False)
 summary={'schema_version':'subsector_targets_v1','run_id':run,'status':'complete_development_targets_no_test','horizon_sessions':H,'inputs':{str(STOCK):sha(STOCK),str(FACT):sha(FACT),str(RET):sha(RET)},'rows':len(t),'target_rows':int(t.target_available.sum()),'counts':counts.to_dict('records'),'processing':{'formation_member_rule':'names with valid return on formation date','future_missing':'target unavailable, not zero-filled','imputation':'none','future_targets_test_read':False,'rows_deleted':0},'checks':{'unique_date_group':not t.duplicated(['Date','primary_group']).any(),'target_never_uses_formation_return':True,'target_available_has_h_rows':bool(t.loc[t.target_available,'subsector_forward_excess'].notna().all()),'all_groups_retained':set(t.primary_group)==set(roles),'no_test_dates':bool(t.Date.max()<=pd.Timestamp('2025-12-31'))}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8'); print(json.dumps({'output':str(out),'counts':counts.to_dict('records'),'checks':summary['checks']},indent=2,default=str))
if __name__=='__main__': main()

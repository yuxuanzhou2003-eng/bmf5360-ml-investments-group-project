"""Build a chronological event/neighbor panel; no strategy performance claims."""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'access_probe'
USE_CLEAN='--clean' in sys.argv
OUT=ROOT/'data'/('diffusion_pilot_clean' if USE_CLEAN else 'diffusion_pilot')
OUT.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'diffusion_pilot_config.json').read_text(encoding='utf-8'))

def read_groups(stem):
    if USE_CLEAN:
        table={'pilot_returns':'returns','pilot_actuals':'actuals','pilot_estimates':'estimates'}[stem]
        frame=pd.read_csv(ROOT/'data/clean/v1'/f'{table}.csv')
        if table=='returns':frame=frame.loc[frame.Instrument!=cfg['benchmark']]
        return frame
    frames=[]
    for i in range(3):
        path=RAW/f'{stem}_{i}.csv'
        if not path.exists(): raise RuntimeError(f'Missing required input: {path.name}')
        frames.append(pd.read_csv(path,index_col=0))
    return pd.concat(frames,ignore_index=True)

returns=read_groups('pilot_returns')
if USE_CLEAN:
    benchmark=pd.read_csv(ROOT/'data/clean/v1/returns.csv')
    benchmark=benchmark.loc[benchmark.Instrument==cfg['benchmark']]
else:
    benchmark=pd.read_csv(RAW/'pilot_benchmark.csv',index_col=0)
returns=pd.concat([returns,benchmark],ignore_index=True)
returns['Date']=pd.to_datetime(returns['Date'],errors='coerce').dt.normalize()
returns['Total Return']=pd.to_numeric(returns['Total Return'],errors='coerce')/100
duplicate_returns=int(returns.duplicated(['Instrument','Date']).sum())
if duplicate_returns: raise RuntimeError('Duplicate instrument-date returns require investigation')
wide=returns.pivot(index='Date',columns='Instrument',values='Total Return').sort_index()
wide=wide.loc[wide.index.notna() & wide[cfg['benchmark']].notna()]
if (wide.dropna(how='all') < -1).any().any(): raise RuntimeError('Return below -100%; check units')
wide.to_csv(OUT/'returns_decimal.csv')

actuals=read_groups('pilot_actuals').rename(columns={'Report Date':'announcement','Period End Date':'period_end','Earnings Per Share - Actual':'actual'})
estimates=read_groups('pilot_estimates').rename(columns={'Calc Date':'snapshot','Period End Date':'period_end','Earnings Per Share - Mean':'consensus','Earnings Per Share - Standard Deviation':'dispersion','Earnings Per Share - Number of Included Estimates':'analysts'})
for frame,col in [(actuals,'announcement'),(actuals,'period_end'),(estimates,'snapshot'),(estimates,'period_end')]:
    frame[col]=pd.to_datetime(frame[col],format='mixed',errors='coerce')
for frame,cols in [(actuals,['actual']),(estimates,['consensus','dispersion','analysts'])]:
    for col in cols: frame[col]=pd.to_numeric(frame[col],errors='coerce')
actuals=actuals.dropna(subset=['announcement','period_end','actual']).drop_duplicates(['Instrument','announcement','period_end']).sort_values('announcement')
actuals=actuals.loc[actuals.announcement.between(pd.Timestamp(cfg['event_start']),pd.Timestamp(cfg['event_end'])+pd.Timedelta(days=1),inclusive='left')]
event_rows=[]
edges=[]
for _,event in actuals.iterrows():
    source=event.Instrument
    day=event.announcement.normalize()
    row={'source':source,'announcement':event.announcement,'period_end':event.period_end,'actual':event.actual}
    candidates=estimates.loc[(estimates.Instrument==source)&(estimates.period_end==event.period_end)&(estimates.snapshot<day)&estimates.consensus.notna()].sort_values('snapshot')
    if candidates.empty:
        row['status']='no_matching_preannouncement_quarter';event_rows.append(row);continue
    latest=candidates.iloc[-1]
    row.update(snapshot=latest.snapshot,consensus=latest.consensus,dispersion=latest.dispersion,analysts=latest.analysts,snapshot_age_days=(day-latest.snapshot).days)
    if row['snapshot_age_days']>14:
        row['status']='stale_snapshot';event_rows.append(row);continue
    row['eps_difference']=event.actual-latest.consensus
    row['standardized_surprise']=row['eps_difference']/latest.dispersion if latest.dispersion>0 else np.nan
    if source not in wide:
        row['status']='source_returns_missing';event_rows.append(row);continue
    hist=wide.loc[wide.index<day].tail(cfg['graph_lookback_sessions'])
    market=hist[cfg['benchmark']]
    residuals=pd.DataFrame(index=hist.index)
    for ticker in cfg['instruments']:
        if ticker not in hist: continue
        valid=hist[[ticker,cfg['benchmark']]].dropna()
        if len(valid)<cfg['graph_min_observations']: continue
        beta=valid[ticker].cov(valid[cfg['benchmark']])/valid[cfg['benchmark']].var()
        residuals[ticker]=hist[ticker]-beta*market
    if source not in residuals:
        row['status']='insufficient_graph_history';event_rows.append(row);continue
    corr=residuals.corr(min_periods=cfg['graph_min_observations'])[source].drop(labels=[source]).dropna()
    selected=corr.abs().sort_values(ascending=False).head(cfg['neighbors_per_event']).index
    entry_i=int(wide.index.searchsorted(day,side='right'))
    end_i=entry_i+cfg['forward_sessions']
    if end_i>=len(wide):
        row['status']='label_window_incomplete';event_rows.append(row);continue
    row.update(status='matched',graph_last_date=hist.index.max(),entry_close=wide.index[entry_i],exit_close=wide.index[end_i],neighbor_count=len(selected))
    event_rows.append(row)
    for receiver in selected:
        future=wide.iloc[entry_i+1:end_i+1][[receiver,cfg['benchmark']]]
        edge=dict(row,receiver=receiver,residual_correlation=float(corr[receiver]))
        complete=len(future)==cfg['forward_sessions'] and future.notna().all().all()
        edge['label_complete']=complete
        edge['receiver_forward_return']=float((1+future[receiver]).prod()-1) if complete else np.nan
        edge['benchmark_forward_return']=float((1+future[cfg['benchmark']]).prod()-1) if complete else np.nan
        edge['forward_benchmark_excess']=edge['receiver_forward_return']-edge['benchmark_forward_return']
        own_events=actuals.loc[actuals.Instrument==receiver,'announcement'].dt.normalize()
        edge['receiver_event_coverage_present']=len(own_events)>0
        edge['ex_post_own_announcement_overlap']=bool(own_events.between(day,wide.index[end_i]).any()) if len(own_events) else None
        edges.append(edge)

events=pd.DataFrame(event_rows)
panel=pd.DataFrame(edges)
if len(panel):
    panel['sample_id']=[hashlib.sha256(f'{s}|{a}|{p}|{r}'.encode()).hexdigest()[:24] for s,a,p,r in zip(panel.source,panel.announcement,panel.period_end,panel.receiver)]
    if panel.sample_id.duplicated().any():raise RuntimeError('Duplicate sample IDs')
    feature_cols=['sample_id','source','receiver','announcement','period_end','snapshot','actual','consensus','dispersion','analysts','snapshot_age_days','eps_difference','standardized_surprise','graph_last_date','residual_correlation']
    label_cols=['sample_id','entry_close','exit_close','label_complete','receiver_forward_return','benchmark_forward_return','forward_benchmark_excess']
    diagnostic_cols=['sample_id','receiver_event_coverage_present','ex_post_own_announcement_overlap']
    panel[feature_cols].to_csv(OUT/'features.csv',index=False)
    panel[label_cols].to_csv(OUT/'labels.csv',index=False)
    panel[diagnostic_cols].to_csv(OUT/'diagnostics_ex_post.csv',index=False)
events.to_csv(OUT/'events.csv',index=False)
panel.to_csv(OUT/'event_neighbor_panel.csv',index=False)
matched=events.loc[events.status=='matched'] if len(events) else pd.DataFrame()
checks={
    'unique_return_keys':duplicate_returns==0,
    'snapshots_strictly_before_event_day':bool(len(matched) and (matched.snapshot<matched.announcement.dt.normalize()).all()),
    'graph_history_strictly_before_event_day':bool(len(matched) and (matched.graph_last_date<matched.announcement.dt.normalize()).all()),
    'entry_strictly_after_event_day':bool(len(matched) and (matched.entry_close>matched.announcement.dt.normalize()).all()),
    'no_self_edges':bool(len(panel) and (panel.source!=panel.receiver).all()),
}
summary={
    'purpose':cfg['purpose'],
    'raw_return_rows':len(returns),'return_sessions':len(wide),'instruments_with_returns':int(wide.notna().any().sum()),
    'return_non_null_by_instrument':wide.notna().sum().to_dict(),
    'actual_events':len(actuals),'companies_with_actual_events':actuals.Instrument.nunique(),'companies_without_actual_events':sorted(set(cfg['instruments'])-set(actuals.Instrument)),
    'estimate_rows':len(estimates),'valid_estimate_rows':int(estimates.consensus.notna().sum()),'event_status_counts':events.status.value_counts().to_dict() if len(events) else {},
    'neighbor_panel_rows':len(panel),'complete_label_rows':int(panel.label_complete.sum()) if len(panel) else 0,
    'own_event_overlap_rows':int(panel.ex_post_own_announcement_overlap.sum()) if len(panel) else 0,
    'unknown_receiver_event_coverage_rows':int((~panel.receiver_event_coverage_present).sum()) if len(panel) else 0,
    'checks':checks,
    'limitations':['Hand-selected surviving stocks; historical universe unresolved','EPS adjustment and as-reported revisions not independently verified','Vendor announcement timezone not confirmed','Weekly consensus may be up to 14 days old','Residual correlation graph is statistical association, not a supply-chain graph or causal evidence','Overlap flag uses future realized events for diagnostic stratification ONLY; never a feature or trading filter','TR.TotalReturn units assumed percent; corporate-action methodology still needs independent reconciliation','Pilot dates are development data, not a final test set','No predictive model fitted and no tradable portfolio backtested']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['return_non_null_by_instrument','limitations']},ensure_ascii=False,indent=2))
if not all(checks.values()): raise RuntimeError('Chronology or structure checks failed')

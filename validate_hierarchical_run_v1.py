"""Independent reconstruction of labels, features and daily accounting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
panel=ROOT/'data/model_ready_hierarchical_v1/20260913T033945Z';run=ROOT/'data/model_runs/hierarchical_fund_v1/20260913T034800Z';out=ROOT/'data/audit/hierarchical_run_v1/20260913T034800Z';out.mkdir(parents=True,exist_ok=False)
s=pd.read_csv(panel/'stock_daily.csv',parse_dates=['Date']);l=pd.read_csv(run/'daily_ledger.csv',parse_dates=['Date']);w=pd.read_csv(run/'daily_weights.csv',parse_dates=['Date']);t=pd.read_csv(run/'target_weights.csv',parse_dates=['formation_session','execution_session']);mac=pd.read_csv(panel/'macro_daily.csv',parse_dates=['Date']).set_index('Date');dates=pd.DatetimeIndex(sorted(s.Date.unique()));ret=s.pivot(index='Date',columns='Instrument',values='return_decimal');close=s.pivot(index='Date',columns='Instrument',values='close');checks={}
checks['30_complete_unique_ledgers']=l.groupby(['strategy','cost_bps']).size().eq(502).all() and not l.duplicated(['strategy','cost_bps','Date']).any()
checks['no_negative_targets']=t.target_weight.ge(0).all();checks['stockcap_at_orders']=t.target_weight.le(.1+1e-12).all();checks['sleevecap_at_orders']=t.groupby(['strategy','execution_session','primary_group']).target_weight.sum().le(.4+1e-12).all();checks['cash_nonnegative']=l.cash_weight.ge(-1e-12).all()
checks['execution_exactly_next_session']=all(dates.get_loc(e)-dates.get_loc(f)==1 for f,e in t[['formation_session','execution_session']].drop_duplicates().itertuples(index=False,name=None))
checks['cash_stock_sum_one']=np.allclose(l.ai_weight+l.cash_weight,1,atol=1e-12)
maxerror=0.;costerror=0.;weighttargeterror=0.
for strategy,x in l[l.cost_bps.eq(10)].groupby('strategy'):
    x=x.sort_values('Date');weights=w[w.strategy.eq(strategy)].pivot(index='Date',columns='Instrument',values='weight').reindex(columns=ret.columns)
    prevw=np.zeros(ret.shape[1]);prevcash=1.;prevnav=1.
    for row in x.itertuples():
        d=row.Date;i=dates.get_loc(d);r=ret.loc[d].to_numpy();active=prevw>1e-14;assert np.isfinite(r[active]).all()
        cr=mac.loc[d,'dgs3mo_level']/100*(d-dates[i-1]).days/365
        precost=prevnav*(1+np.dot(prevw[active],r[active])+prevcash*cr)
        maxerror=max(maxerror,abs((precost-row.cost_dollars)-row.nav));costerror=max(costerror,abs(row.cost_dollars-.001*row.turnover*precost))
        tw=t[t.strategy.eq(strategy)&t.execution_session.eq(d)]
        if len(tw):weighttargeterror=max(weighttargeterror,np.abs(weights.loc[d]-tw.set_index('Instrument').target_weight.reindex(ret.columns)).max())
        prevw=weights.loc[d].to_numpy();prevcash=row.cash_weight;prevnav=row.nav
checks['daily_accounting_reconstruction']=maxerror<1e-9;checks['cost_reconstruction']=costerror<1e-9;checks['order_to_actual_weights']=weighttargeterror<1e-9
labels=pd.read_csv(run/'stock_targets.csv',parse_dates=['Date']);finite=labels[labels.forward_return.notna()].sample(200,random_state=5360);labelerror=0.
for row in finite.itertuples():
    i=dates.get_loc(row.Date);actual=np.prod(1+ret[row.Instrument].iloc[i+2:i+7])-1;labelerror=max(labelerror,abs(actual-row.forward_return))
checks['200_sample_delayed_labels']=labelerror<1e-10
features=pd.read_csv(run/'stock_features.csv',parse_dates=['Date']);sample=features[features.feature_eligible].sample(100,random_state=5360);ferror=0.
for row in sample.itertuples():
    i=dates.get_loc(row.Date);actual=np.prod(1+ret[row.Instrument].iloc[i-19:i+1])-1;ferror=max(ferror,abs(actual-row.mom20))
checks['100_sample_past_only_mom20']=ferror<1e-10
fits=pd.read_csv(run/'fit_audit.csv');checks['all_training_exits_purged']=(pd.to_datetime(fits.train_last_exit)<'2024-01-01').all()
checks['no2026_records']=l.Date.lt('2026-01-01').all();checks['no_ledger_gap_rows']=pd.read_csv(run/'ledger_gaps.csv').empty
cashret=pd.Series({d:mac.loc[d,'dgs3mo_level']/100*(d-dates[dates.get_loc(d)-1]).days/365 for d in sorted(l.Date.unique())})
risk=[]
for (strategy,cost),x in l.groupby(['strategy','cost_bps']):
    x=x.sort_values('Date');excess=x.return_net.to_numpy()-cashret.reindex(x.Date).to_numpy();risk.append(dict(strategy=strategy,cost_bps=cost,sharpe_excess_cash=np.mean(excess)/np.std(excess,ddof=1)*np.sqrt(252),annualized_vol=x.return_net.std()*np.sqrt(252)))
pd.DataFrame(risk).to_csv(out/'risk_adjusted_metrics.csv',index=False)
# External SPY comparator: same first execution date, then hold; 10bps entry cost.
spy=pd.read_csv(panel/'spy_daily.csv',parse_dates=['Date']).set_index('Date').return_decimal
firstexec=t.execution_session.min();cash=1.;equity=0.;prev=1.;bench=[]
for d in cashret.index:
    equity*=1+spy.loc[d];cash*=1+cashret.loc[d]
    if d==firstexec:equity=cash/(1+.001);cash=0.
    nav=cash+equity;bench.append(dict(Date=d,nav=nav,return_net=nav/prev-1));prev=nav
benchmark=pd.DataFrame(bench);benchmark.to_csv(out/'spy_comparator_10bps.csv',index=False)
nav=benchmark.nav.to_numpy();ex=benchmark.return_net.to_numpy()-cashret.to_numpy();benchmarkmetrics=dict(total_return=nav[-1]-1,sharpe_excess_cash=np.mean(ex)/np.std(ex,ddof=1)*np.sqrt(252),max_drawdown=np.min(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1),note='External 100% SPY benchmark exempt from individual-stock cap; same first execution delay and10bps entry cost.')
summary=dict(checks={k:bool(v) for k,v in checks.items()},all_pass=all(checks.values()),maximum_errors=dict(daily_nav=maxerror,cost=costerror,order_weight=weighttargeterror,sampled_labels=labelerror,sampled_momentum=ferror),spy_comparator=benchmarkmetrics,limitations=['Tests independently reconstruct sample labels and full 10bps ledgers; not independent vendor truth.','Run performance.csv original sharpe assumes zero risk-free; risk_adjusted_metrics.csv provides cash-excess Sharpe for final report.'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

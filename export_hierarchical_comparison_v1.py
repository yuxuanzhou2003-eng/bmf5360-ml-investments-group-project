"""Export the screenshot's cash-excess Sharpe comparison from a completed run."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def stats(frame, cash):
    q=frame.sort_values('Date');r=q.return_net.to_numpy();ex=r-cash.reindex(q.Date).to_numpy()
    assert np.isfinite(r).all() and np.isfinite(ex).all()
    nav=np.cumprod(1+r);sd=ex.std(ddof=1)
    return dict(days=len(q),total_return=nav[-1]-1,annualized_vol=r.std(ddof=1)*np.sqrt(252),sharpe_excess_cash=ex.mean()/sd*np.sqrt(252) if sd>1e-12 else np.nan,max_drawdown=np.min(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel-dir',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    ledger=pd.read_csv(a.run_dir/'daily_ledger.csv',parse_dates=['Date']);target=pd.read_csv(a.run_dir/'target_weights.csv',parse_dates=['execution_session'])
    mac=pd.read_csv(a.panel_dir/'macro_daily.csv',parse_dates=['Date']).set_index('Date').sort_index();days=pd.DatetimeIndex(mac.index)
    assert not days.has_duplicates
    cash=pd.Series(mac.dgs3mo_level.to_numpy()/100*np.r_[np.nan,np.diff(days.values)/np.timedelta64(1,'D')]/365,index=days)
    labels={'equal_sleeves':'等赛道对照','momentum':'简单动量','logistic_all':'Logistic 三层','random_forest_all':'随机森林三层'};rows=[]
    for strategy,label in labels.items():
        q=ledger[ledger.strategy.eq(strategy)&ledger.cost_bps.eq(10)]
        assert len(q)>0 and not q.Date.duplicated().any()
        rows.append(dict(strategy=strategy,label=label,cost_bps=10,**stats(q,cash)))
    evaluation=pd.DatetimeIndex(sorted(ledger.Date.unique()));first=target.execution_session.min()
    spy=pd.read_csv(a.panel_dir/'spy_daily.csv',parse_dates=['Date']).set_index('Date').return_decimal
    balance=1.;equity=0.;last=1.;result=[]
    for date in evaluation:
        assert np.isfinite(spy.loc[date]) and np.isfinite(cash.loc[date])
        equity*=1+spy.loc[date];balance*=1+cash.loc[date]
        if date==first:equity=balance/1.001;balance=0.
        nav=equity+balance;result.append(dict(Date=date,return_net=nav/last-1));last=nav
    rows.append(dict(strategy='SPY',label='SPY 对照',cost_bps=10,**stats(pd.DataFrame(result),cash)))
    out=pd.DataFrame(rows);a.output_dir.mkdir(parents=True,exist_ok=False);out.to_csv(a.output_dir/'comparison.csv',index=False)
    lines=['| 策略 | 年化波动率 | 净 Sharpe（相对现金） | 最大回撤 |','|---|---:|---:|---:|']
    for row in out.itertuples():lines.append(f'| {row.label} | {row.annualized_vol:.2%} | {row.sharpe_excess_cash:.2f} | {row.max_drawdown:.2%} |')
    (a.output_dir/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out.to_string(index=False))
if __name__=='__main__':main()

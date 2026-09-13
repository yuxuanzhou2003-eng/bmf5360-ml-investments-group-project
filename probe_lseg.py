"""Small read-only entitlement probes. Credentials stay in the local .env."""
import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
credentials = dotenv_values(ROOT / '.env')
secrets = [v for v in credentials.values() if v]

def redact(value):
    value = str(value)
    for secret in secrets:
        value = value.replace(secret, '[REDACTED]')
    return value

class SafeStream:
    def __init__(self, target): self.target = target
    def write(self, message): return self.target.write(redact(message))
    def flush(self): self.target.flush()
    def isatty(self): return False

sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)
import lseg.data as ld
import pandas as pd

OUT = ROOT / 'data' / 'access_probe'
OUT.mkdir(parents=True, exist_ok=True)
mode = sys.argv[1] if len(sys.argv) > 1 else 'core'
records = []

def save():
    (OUT / f'{mode}_summary.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')

def probe(name, request, callback):
    record = {'name': name, 'request': request, 'tested_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    records.append(record)
    save()
    print('START', name, flush=True)
    def timeout():
        record.update(status='timeout', error='Request exceeded 90 seconds; availability unconfirmed.')
        save()
        print('TIMEOUT', name, flush=True)
        os._exit(2)
    timer = threading.Timer(90, timeout)
    timer.start()
    started = time.monotonic()
    try:
        df = callback()
        if df is None:
            record.update(status='connected' if name == 'desktop_session' else 'empty')
        else:
            df = df.replace(r'^\s*$', pd.NA, regex=True)
            df.to_csv(OUT / f'{name}.csv')
            record.update(status='returned' if len(df) else 'empty', rows=len(df), columns=[str(c) for c in df.columns], non_null={str(c):int(df[c].notna().sum()) for c in df.columns}, index_min=str(df.index.min()), index_max=str(df.index.max()))
    except Exception as exc:
        record.update(status='error', error=redact(exc)[:1500])
    finally:
        timer.cancel()
        record['elapsed_seconds'] = round(time.monotonic()-started, 2)
        save()
        print(json.dumps(record, ensure_ascii=False), flush=True)

probe('desktop_session', {'session':'desktop.workspace'}, lambda: ld.open_session(name='desktop.workspace', app_key=credentials.get('LSEG_APP_KEY')) and None)
if records[-1]['status'] == 'error':
    sys.exit(1)
try:
    if mode == 'diffusion_pilot':
        cfg=json.loads((ROOT / 'diffusion_pilot_config.json').read_text(encoding='utf-8'))
        universe=cfg['instruments']
        for i in range(0,len(universe),10):
            batch=universe[i:i+10]
            args=dict(universe=batch,fields=['TR.TotalReturn.date','TR.TotalReturn'],parameters={'SDate':cfg['warmup_start'],'EDate':cfg['price_end'],'Frq':'D'})
            probe(f'pilot_returns_{i//10}',args,lambda a=args: ld.get_data(**a))
            args=dict(universe=batch,fields=['TR.EPSActValue.announcedate','TR.EPSActValue.periodenddate','TR.EPSActValue'],parameters={'SDate':cfg['event_start'],'EDate':cfg['event_end'],'Frq':'FQ','Period':'FQ0'})
            probe(f'pilot_actuals_{i//10}',args,lambda a=args: ld.get_data(**a))
            args=dict(universe=batch,fields=['TR.EPSMean.calcdate','TR.EPSMean.periodenddate','TR.EPSMean','TR.EPSStdDev','TR.EPSNumIncEstimates'],parameters={'SDate':'2014-12-01','EDate':cfg['event_end'],'Frq':'W','Period':'FQ1'})
            probe(f'pilot_estimates_{i//10}',args,lambda a=args: ld.get_data(**a))
        args=dict(universe=[cfg['benchmark']],fields=['TR.TotalReturn.date','TR.TotalReturn'],parameters={'SDate':cfg['warmup_start'],'EDate':cfg['price_end'],'Frq':'D'})
        probe('pilot_benchmark',args,lambda: ld.get_data(**args))
    if mode == 'diffusion_dates':
        for date in ['2015-01-02','2017-01-03']:
            args=dict(universe=[f'0#.SPX({date})'],fields=['TR.RIC','TR.CommonName'])
            probe('iso_chain_'+date,args,lambda a=args: ld.get_data(**a))
        args=dict(universe=['.SPX'],fields=['TR.IndexConstituentRIC(SDate=2015-01-02)','TR.IndexConstituentName(SDate=2015-01-02)'])
        probe('field_dated_constituents',args,lambda: ld.get_data(**args))
    if mode == 'diffusion_discovery':
        for date in ['20150102','20170103']:
            args=dict(universe=[f'0#.SPX({date})'],fields=['TR.RIC','TR.CommonName'])
            probe('dated_chain_'+date,args,lambda a=args: ld.get_data(**a))
        args=dict(universe=['AAPL.O','MSFT.O'],fields=['TR.EPSActValue.announcedate','TR.EPSActValue.periodenddate','TR.EPSActValue'],parameters={'SDate':'2015-01-01','EDate':'2015-12-31','Frq':'FQ','Period':'FQ0'})
        probe('quarterly_actuals_with_period',args,lambda: ld.get_data(**args))
        for period in ['FQ0','FQ1']:
            args=dict(universe=['AAPL.O','MSFT.O'],fields=['TR.EPSMean.calcdate','TR.EPSMean.periodenddate','TR.EPSMean','TR.EPSStdDev','TR.EPSNumIncEstimates'],parameters={'SDate':'2015-01-20','EDate':'2015-01-27','Frq':'D','Period':period})
            probe('quarterly_estimates_'+period,args,lambda a=args: ld.get_data(**a))
    if mode == 'core':
        for label, start, end in [('recent','2025-01-01','2025-01-15'),('historical','2010-01-01','2010-01-15')]:
            args = dict(universe=['SPY.P','TLT.O','GLD.P','AAPL.O'], fields=['TRDPRC_1','ACVOL_UNS'], interval='daily', start=start, end=end)
            probe('prices_'+label, args, lambda a=args: ld.get_history(**a))
        args = dict(universe=['SPY.P','TLT.O','GLD.P'],fields=['TR.TotalReturn.date','TR.TotalReturn'], parameters={'SDate':'2010-01-01','EDate':'2010-01-15','Frq':'D'})
        probe('total_returns',args, lambda: ld.get_data(**args))
    if mode == 'fundamental':
        args = dict(universe=['AAPL.O','MSFT.O'], fields=['TR.EPSMean.calcdate','TR.EPSMean','TR.EPSStdDev','TR.EPSNumIncEstimates'],parameters={'SDate':'2015-01-01','EDate':'2015-03-31','Frq':'M','Period':'FY1'})
        probe('historical_estimates',args,lambda: ld.get_data(**args))
        args = dict(universe=['AAPL.O','MSFT.O'], fields=['TR.EPSActValue.announcedate','TR.EPSActValue'],parameters={'SDate':'2015-01-01','EDate':'2016-01-01','Frq':'FQ','Period':'FQ0'})
        probe('earnings_actuals',args,lambda: ld.get_data(**args))
        args = dict(universe=['AAPL.O','MSFT.O'],fields=['TR.TotalAssets.date','TR.TotalAssets','TR.Revenue'],parameters={'SDate':'2014-01-01','EDate':'2016-01-01','Frq':'FY'})
        probe('fundamentals',args,lambda: ld.get_data(**args))
    if mode in ('news_macro', 'screening'):
        for label,start,end in [('recent','2026-09-01','2026-09-06'),('historical','2015-01-01','2015-01-15')]:
            args=dict(query='R:AAPL.O AND Language:LEN',start=start,end=end,count=10)
            probe('news_'+label,args,lambda a=args: ld.news.get_headlines(**a))
        args=dict(universe=['.VIX','EUR=','JPY=','US10YT=RR'],fields=['TRDPRC_1'],interval='daily',start='2010-01-01',end='2010-01-15')
        probe('macro_fx',args,lambda: ld.get_history(**args))
    if mode in ('universe', 'screening'):
        args=dict(universe=['.SPX'],fields=['TR.IndexConstituentRIC','TR.IndexConstituentName'],parameters={'SDate':'2010-01-04','EDate':'2010-01-04'})
        probe('historical_constituents',args,lambda: ld.get_data(**args))
    if mode == 'screening':
        args=dict(universe=['SPY.P','TLT.O','GLD.P'],fields=['TRDPRC_1'],interval='daily',start='2010-01-01',end='2025-12-31')
        probe('long_history_coverage',args,lambda: ld.get_history(**args))
    if mode == 'fallback':
        args=dict(universe=['SPY.P','AAPL.O'],fields=['TR.PriceClose.date','TR.PriceClose'],parameters={'SDate':'2010-01-01','EDate':'2010-01-15','Frq':'D'})
        probe('pricing_fundamental_endpoint',args,lambda: ld.get_data(**args))
    if mode == 'detail':
        for label,ric,fields in [('vix','.VIX',['TRDPRC_1']),('fx_spot','EUR=',['BID','ASK']),('treasury_yield','US10YT=RR',['B_YLD_1','YLDTOMAT'])]:
            args=dict(universe=[ric],fields=fields,interval='daily',start='2010-01-01',end='2010-01-15')
            probe(label,args,lambda a=args: ld.get_history(**a))
        for label,ric,params in [('spx_current','.SPX',{}),('spx_historical_retry','.SPX',{'SDate':'2010-01-04'}),('ftse_historical','.FTSE',{'SDate':'2010-01-04'})]:
            args=dict(universe=[ric],fields=['TR.IndexConstituentRIC','TR.IndexConstituentName'],parameters=params)
            probe(label,args,lambda a=args: ld.get_data(**a))
        def story_probe():
            source=pd.read_csv(OUT / 'news_historical.csv')
            result=ld.news.get_story(source['storyId'].iloc[0],format=ld.news.Format.TEXT)
            (OUT / 'historical_story.txt').write_text(result or '',encoding='utf-8')
            return pd.DataFrame([{'characters':len(result or ''),'nonempty':bool(result and result.strip())}])
        probe('historical_news_body',{'source':'first headline from news_historical.csv'},story_probe)
finally:
    ld.close_session()

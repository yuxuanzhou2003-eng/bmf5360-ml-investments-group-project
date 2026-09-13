"""Read-only probe: can we retrieve returns/earnings for delisted RICs, and what happens on the final day."""
import sys, json, os, threading, hashlib
from pathlib import Path
from datetime import datetime, timezone
from dotenv import dotenv_values
ROOT = Path(__file__).resolve().parent
cred = dotenv_values(ROOT/'.env'); secrets=[v for v in cred.values() if v]
def redact(x):
    x=str(x)
    for s in secrets: x=x.replace(s,'[REDACTED]')
    return x
class S:
    def __init__(s,t):s.t=t
    def write(s,m):return s.t.write(redact(m))
    def flush(s):s.t.flush()
    def isatty(s):return False
sys.stdout=S(sys.stdout); sys.stderr=S(sys.stderr)
import pandas as pd, lseg.data as ld
OUT = ROOT/'data'/'raw'/'universe_probe'; OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    ('SWY.N^A15',  '2014-06-01', '2015-06-30', 'Safeway: Albertsons cash+CVR takeover, Jan 2015'),
    ('ALTR.OQ^L15','2015-06-01', '2016-03-31', 'Altera: Intel all-cash $54/sh, Dec 2015'),
    ('ABMD.OQ^L22','2022-09-01', '2023-03-31', 'Abiomed: J&J cash tender $380/sh, Dec 2022'),
    ('AKS.N^C20',  '2019-09-01', '2020-06-30', 'AK Steel: Cleveland-Cliffs all-stock, Mar 2020'),
    ('AET.N^K18',  '2018-06-01', '2019-03-31', 'Aetna: CVS cash+stock, Nov 2018'),
]
def cached(name, request, func):
    dest, meta = OUT/f'{name}.csv', OUT/f'{name}.meta.json'
    if dest.exists() and meta.exists():
        print('CACHE', name, flush=True); return pd.read_csv(dest)
    rec = {'name':name,'request':request,'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
    t=threading.Timer(120, lambda: os._exit(2)); t.start()
    try:
        df = func()
        if df is None or not len(df):
            rec.update(status='empty', rows=0); print('EMPTY', name, flush=True)
            (OUT/f'{name}.meta.json').write_text(json.dumps(rec,indent=2), encoding='utf-8'); return None
        df = df.replace(r'^\s*$', pd.NA, regex=True); df.to_csv(dest, index=False)
        rec.update(status='success', rows=len(df), columns=[str(c) for c in df.columns], sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        meta.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
        print('OK', name, len(df), flush=True); return df
    except Exception as e:
        rec.update(status='error', error=redact(e)[:400]); print('ERROR', name, rec['error'], flush=True)
        (OUT/f'{name}.meta.json').write_text(json.dumps(rec,indent=2), encoding='utf-8'); return None
    finally: t.cancel()

t=threading.Timer(120, lambda: os._exit(2)); t.start()
try: ld.open_session(name='desktop.workspace', app_key=cred.get('LSEG_APP_KEY'))
finally: t.cancel()
findings=[]
try:
    for ric, start, end, note in CASES:
        tag = ric.replace('.','_').replace('^','_')
        args = {'universe':[ric], 'fields':['TR.TotalReturn.date','TR.TotalReturn'], 'parameters':{'SDate':start,'EDate':end,'Frq':'D'}}
        df = cached(f'delisted_ret_{tag}', args, lambda a=args: ld.get_data(**a))
        f = {'ric':ric,'note':note,'returns_rows':0}
        if df is not None and len(df):
            df['Date']=pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df['tr']=pd.to_numeric(df['Total Return'], errors='coerce')
            v=df.dropna(subset=['Date','tr'])
            f.update(returns_rows=len(v), first=str(v.Date.min().date()) if len(v) else None,
                     last=str(v.Date.max().date()) if len(v) else None,
                     last5=[{'d':str(d.date()),'ret_pct':round(x,3)} for d,x in zip(v.Date.tail(5), v.tr.tail(5))],
                     max_abs_pct=round(v.tr.abs().max(),2))
        args2 = {'universe':[ric], 'fields':['TR.EPSActValue.date','TR.EPSActValue','TR.EPSActValue.periodenddate'], 'parameters':{'SDate':start,'EDate':end,'Period':'FQ0','Frq':'FQ'}}
        df2 = cached(f'delisted_eps_{tag}', args2, lambda a=args2: ld.get_data(**a))
        f['eps_rows'] = 0 if df2 is None else int(df2.notna().any(axis=1).sum())
        findings.append(f); print(json.dumps(f, ensure_ascii=False), flush=True)
finally:
    ld.close_session()
(OUT/'delisted_findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding='utf-8')
print('DELISTED PROBE COMPLETE', flush=True)

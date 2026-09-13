"""Read-only batched collection for the reconstructed 2015-2026 S&P 500 universe.
Writes only to data/raw/universe_v2/. Cached + checksummed, safe to interrupt and resume."""
import sys, json, os, time, threading, hashlib
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

OUT = ROOT/'data'/'raw'/'universe_v2'; OUT.mkdir(parents=True, exist_ok=True)
WARMUP_START = '2013-11-01'
STUDY_END = '2026-09-07'
part = sys.argv[1] if len(sys.argv) > 1 else 'all'

SPECS = {
 'returns':   (25, ['TR.TotalReturn.date','TR.TotalReturn'], {'Frq':'D'}),
 'actuals':   (25, ['TR.EPSActValue.announcedate','TR.EPSActValue.periodenddate','TR.EPSActValue'], {'Frq':'FQ','Period':'FQ0'}),
 'estimates': (10, ['TR.EPSMean.calcdate','TR.EPSMean.periodenddate','TR.EPSMean','TR.EPSStdDev','TR.EPSNumIncEstimates'], {'Frq':'W','Period':'FQ1'}),
}

def cached(name, request, func):
    dest, meta = OUT/f'{name}.csv', OUT/f'{name}.meta.json'
    if dest.exists() and meta.exists():
        rec = json.loads(meta.read_text(encoding='utf-8'))
        if rec.get('request') != request: raise RuntimeError(f'Cache request mismatch: {name}')
        if rec.get('sha256') and hashlib.sha256(dest.read_bytes()).hexdigest() != rec['sha256']: raise RuntimeError(f'Cache checksum mismatch: {name}')
        print('CACHE', name, rec.get('rows'), flush=True); return
    rec = {'name': name, 'request': request, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
    st = time.monotonic(); tm = threading.Timer(300, lambda: os._exit(2)); tm.start()
    try:
        df = func()
        if df is None or not len(df):
            rec.update(status='empty', rows=0)
        else:
            df = df.replace(r'^\s*$', pd.NA, regex=True)
            df.to_csv(dest, index=False)
            rec.update(status='success', rows=len(df), columns=[str(c) for c in df.columns], sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        rec['elapsed_s'] = round(time.monotonic()-st, 1)
        meta.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
        print('OK', name, rec.get('rows'), f"{rec['elapsed_s']}s", flush=True)
    except Exception as e:
        rec.update(status='error', error=redact(e)[:400], elapsed_s=round(time.monotonic()-st,1))
        (OUT/f'{name}.error.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
        print('ERROR', name, rec['error'], flush=True)
    finally: tm.cancel()

spans = pd.read_csv(ROOT/'data/audit/universe_rebuild/universe_spans_2015_2026.csv')
spans['fetch_end'] = pd.to_datetime(spans.fetch_end)
live = spans[~spans.delisted_ric].ric.tolist()
dead = spans[spans.delisted_ric].copy()
dead['end_year'] = dead.fetch_end.dt.year

groups = [('live', live, WARMUP_START, STUDY_END)]
for year, g in dead.groupby('end_year'):
    groups.append((f'dead{year}', g.ric.tolist(), WARMUP_START, g.fetch_end.max().strftime('%Y-%m-%d')))

tm = threading.Timer(150, lambda: os._exit(2)); tm.start()
try: ld.open_session(name='desktop.workspace', app_key=cred.get('LSEG_APP_KEY'))
finally: tm.cancel()
try:
    if part in ('returns','all'):
        args = {'universe':['SPY.P'], 'fields':SPECS['returns'][1], 'parameters':dict(SDate=WARMUP_START, EDate=STUDY_END, **SPECS['returns'][2])}
        cached('benchmark_returns', args, lambda a=args: ld.get_data(**a))
    for kind, (size, fields, extra) in SPECS.items():
        if part not in (kind, 'all'): continue
        total = sum((len(rics)+size-1)//size for _, rics, _, _ in groups)
        done = 0
        for tag, rics, sd, ed in groups:
            for i in range(0, len(rics), size):
                batch = rics[i:i+size]; done += 1
                args = {'universe': batch, 'fields': fields, 'parameters': dict(SDate=sd, EDate=ed, **extra)}
                print(f'[{kind} {done}/{total}]', tag, i, flush=True)
                cached(f'{kind}_{tag}_{i//size:03d}', args, lambda a=args: ld.get_data(**a))
finally:
    ld.close_session()
print('COLLECTION COMPLETE:', part, flush=True)

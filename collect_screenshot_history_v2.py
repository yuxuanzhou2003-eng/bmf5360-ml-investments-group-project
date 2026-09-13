"""Fresh raw vendor responses for locally evidenced screenshot additions."""
import hashlib,json,threading,os
from datetime import datetime,timezone
from pathlib import Path
from collect_universe_v3 import ROOT,CREDS,redact,SafeStream,FULL_ADJUSTMENTS,PRICE_FIELDS
import sys
sys.stdout=SafeStream(sys.stdout);sys.stderr=SafeStream(sys.stderr)
import pandas as pd
import lseg.data as ld
out=ROOT/'data/raw/screenshot_additions_v2'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out.mkdir(parents=True)
registry=ROOT/'data/audit/screenshot_additions/20260913T030723Z/additional_candidates.csv'
rics=pd.read_csv(registry).matched_local_ric.tolist()
records=[]
def save():
    (out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
def call(name,args,fn):
    rec=dict(name=name,request=args,time=datetime.now(timezone.utc).isoformat(),status='running');records.append(rec);save()
    def timeout():
        rec.update(status='timeout');save();os._exit(2)
    timer=threading.Timer(45,timeout);timer.start()
    try:
        df=fn(**args)
        if df is None:raise RuntimeError('No dataframe returned')
        path=out/f'{name}.csv';df.to_csv(path)
        rec.update(status='returned' if len(df) else 'empty',rows=len(df),columns=[str(x) for x in df.columns],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(name,rec['status'],len(df),flush=True)
        return df
    except Exception as exc:rec.update(status='error',error=redact(exc));print(name,rec['status'],flush=True)
    finally:timer.cancel();save()
try:
    call('session',{},lambda: (ld.open_session(name='desktop.workspace',app_key=CREDS.get('LSEG_APP_KEY')),pd.DataFrame())[1])
    identity=call('identity',dict(universe=rics,fields=['TR.CommonName','TR.ExchangeName','TR.PriceCurrency']),ld.get_data)
    if identity is None or identity.empty:raise RuntimeError('Identity request failed; history not requested')
    for i in range(0,len(rics),3):
        batch=rics[i:i+3]
        call(f'prices_{i//3}',dict(universe=batch,fields=PRICE_FIELDS,interval='1D',start='2020-01-01',end='2025-12-31',adjustments=FULL_ADJUSTMENTS),ld.get_history)
        call(f'returns_{i//3}',dict(universe=batch,fields=['TR.TotalReturn.date','TR.TotalReturn'],parameters={'SDate':'2020-01-01','EDate':'2025-12-31','Frq':'D'}),ld.get_data)
finally:
    ld.close_session()
    print(out,flush=True)
    entry=f'\n\n## Screenshot additions vendor refresh {datetime.now(timezone.utc).isoformat()}\nExecuted collect_screenshot_history_v2.py SHA256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}. Input {registry.relative_to(ROOT)}; output {out.relative_to(ROOT)}. Nine locally evidenced RICs requested, raw prices with explicit adjustments {FULL_ADJUSTMENTS}, daily TR.TotalReturn in vendor units, 2020-01-01 to 2025-12-31. Raw vendor responses preserved with request/status/count/hash manifest. No cleaning, imputation, deduplication, currency conversion, company exclusion or model run. Returned row counts do not establish complete coverage. Identity and corporate-action review remain required.\n'
    with (ROOT/'DATA_PROCESSING_LOG.md').open('a',encoding='utf-8') as f:f.write(entry)
    with (ROOT/'AI_USE_LOG.md').open('a',encoding='utf-8') as f:f.write(entry)

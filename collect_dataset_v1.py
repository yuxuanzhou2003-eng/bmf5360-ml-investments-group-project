"""Read-only, cached supplemental collection. Never prints credentials or news bodies."""
import hashlib
import json
import sys
import threading
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent
secrets=[v for v in dotenv_values(ROOT/'.env').values() if v]
def redact(x):
    x=str(x)
    for s in secrets: x=x.replace(s,'[REDACTED]')
    return x
class SafeStream:
    def __init__(self,target):self.target=target
    def write(self,s):return self.target.write(redact(s))
    def flush(self):self.target.flush()
    def isatty(self):return False
sys.stdout=SafeStream(sys.stdout);sys.stderr=SafeStream(sys.stderr)
import pandas as pd
import lseg.data as ld
from lseg.data.content import news

OUT=ROOT/'data'/'raw'/'supplement_v1'
OUT.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'diffusion_pilot_config.json').read_text(encoding='utf-8'))
manifest=[]
def get(name,request,func,kind='csv'):
    dest=OUT/f'{name}.{kind}'
    meta=OUT/f'{name}.meta.json'
    if dest.exists() and meta.exists():
        record=json.loads(meta.read_text(encoding='utf-8'))
        if record['request']!=request:raise RuntimeError('Cache request mismatch')
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('Cache checksum mismatch')
        print('CACHE',name,flush=True)
        manifest.append(record)
        return pd.read_csv(dest) if kind=='csv' else json.loads(dest.read_text(encoding='utf-8'))
    print('FETCH',name,flush=True)
    timer=threading.Timer(120,lambda:os._exit(2));timer.start()
    record={'name':name,'request':request,'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        result=func()
        if kind=='csv':
            result=result.replace(r'^\s*$',pd.NA,regex=True)
            result.to_csv(dest,index=False)
            record['rows']=len(result)
        else:
            dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        record.update(status='success',sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        meta.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print('OK',name,record.get('rows','json'),flush=True)
        return result
    except Exception as exc:
        record.update(status='error',error=redact(exc)[:1200]);print('ERROR',name,record['error'],flush=True)
        return None
    finally:
        timer.cancel();manifest.append(record)
        (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

timer=threading.Timer(90,lambda:os._exit(2));timer.start()
try:ld.open_session(name='desktop.workspace',app_key=dotenv_values(ROOT/'.env').get('LSEG_APP_KEY'))
finally:timer.cancel()
try:
    fields=['TR.RIC','TR.CommonName','TR.OrganizationID','TR.ISIN','TR.TRBCIndustry','TR.TRBCIndustryCode','TR.TRBCEconomicSector']
    args={'universe':cfg['instruments'],'fields':fields}
    get('company_master_current',args,lambda:ld.get_data(**args))
    args={'universe':['FDX.N','FDX'],'fields':['TR.RIC','TR.CommonName','TR.OrganizationID']}
    get('fdx_identity',args,lambda:ld.get_data(**args))
    args={'universe':['.SPX'],'fields':['TR.IndexJLConstituentRIC','TR.IndexJLConstituentChange','TR.IndexJLConstituentChangeDate'],'parameters':{'SDate':'2015-01-01','EDate':'2017-12-31'}}
    get('index_changes_probe',args,lambda:ld.get_data(**args))
    args={'universe':['.SPX'],'fields':['TR.IndexJLConstituentChangeDate','TR.IndexJLConstituentRIC','TR.IndexJLConstituentName','TR.IndexJLConstituentituentChange'],'parameters':{'SDate':'2015-01-01','EDate':'2017-12-31','IC':'B'}}
    get('index_changes_with_direction',args,lambda:ld.get_data(**args))
    args={'universe':['0#.SPX(20150102)'],'fields':['TR.PriceClose'],'parameters':{'SDate':'2015-01-02','EDate':'2015-01-02'}}
    get('dated_chain_with_price_parameters',args,lambda:ld.get_data(**args))
    query='('+' OR '.join('R:'+s for s in cfg['instruments'])+') AND Language:LEN AND Source:RTRS'
    def items(raw):
        if isinstance(raw,dict):return raw.get('data',[])
        return [item for part in raw for item in part.get('data',[])]
    def fetch_window(start,end,depth=0):
        name='news_'+start.strftime('%Y%m%dT%H%M%S')+'_'+end.strftime('%Y%m%dT%H%M%S')
        request={'query':query,'count':1000,'date_from':start.isoformat(),'date_to':end.isoformat()}
        def call():
            response=news.headlines.Definition(**request).get_data()
            if response.is_success is False:raise RuntimeError('News content-layer response not successful')
            return response.data.raw
        raw=get(name,request,call,'json')
        if raw is None:return
        count=len(items(raw))
        print('HEADLINES',count,flush=True)
        # Retain every parent response; cleaner deduplicates overlapping versions.
        parts=[raw] if isinstance(raw,dict) else raw
        limits=[p['meta']['pageLimit'] for p in parts if isinstance(p.get('meta'),dict) and p['meta'].get('pageLimit')]
        page_limit=min(limits) if limits else 100
        if count>=page_limit:
            if depth>=8:raise RuntimeError('Saturated news window: completeness unresolved')
            mid=start+(end-start)/2
            fetch_window(start,mid,depth+1);fetch_window(mid,end,depth+1)
    for start,end in [('2014-12-01','2014-12-08'),('2014-12-08','2014-12-15'),('2014-12-15','2014-12-22'),('2014-12-22','2014-12-29'),('2014-12-29','2015-01-01')]:
        fetch_window(datetime.fromisoformat(start),datetime.fromisoformat(end))
finally:
    ld.close_session()
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

"""Credential-redacted, run-scoped desktop connection probe."""
import sys, os, json, threading
from pathlib import Path
from datetime import datetime, timezone
from dotenv import dotenv_values
root=Path(__file__).resolve().parent
cfg=dotenv_values(root/'.env')
def redact(x):
    x=str(x)
    for v in cfg.values():
        if v:x=x.replace(v,'[REDACTED]')
    return x
class Safe:
    def __init__(self,s):self.s=s
    def write(self,x):return self.s.write(redact(x))
    def flush(self):self.s.flush()
sys.stdout=Safe(sys.stdout);sys.stderr=Safe(sys.stderr)
out=root/'data/audit/screenshot_lseg_access'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out.mkdir(parents=True)
def save(status,error=''):
    (out/'session.json').write_text(json.dumps(dict(status=status,error=redact(error),time=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
def timeout():
    save('timeout','Desktop session did not open within 40 seconds');os._exit(2)
timer=threading.Timer(40,timeout);timer.start()
try:
    import lseg.data as ld
    ld.open_session(name='desktop.workspace',app_key=cfg.get('LSEG_APP_KEY'))
    df=ld.get_data(universe=['CAT.N'],fields=['TR.CommonName'])
    if df is None or df.empty:raise RuntimeError('No identity data returned; session access unverified')
    df.to_csv(out/'identity_probe.csv',index=False)
    save('data_request_returned');ld.close_session();print('data_request_returned')
except Exception as e:
    save('error',e);print(redact(e))
finally:timer.cancel()
print(out)

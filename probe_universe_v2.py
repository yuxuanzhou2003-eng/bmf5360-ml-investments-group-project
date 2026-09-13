"""Read-only S&P 500 historical universe probe. Writes only to data/raw/universe_probe/."""
import hashlib
import json
import os
import sys
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
import pandas as pd
import lseg.data as ld

OUT = ROOT / 'data' / 'raw' / 'universe_probe'
OUT.mkdir(parents=True, exist_ok=True)
ANCHOR_START, ANCHOR_END = 2010, 2026

def cached(name, request, func):
    dest, meta = OUT / f'{name}.csv', OUT / f'{name}.meta.json'
    if dest.exists() and meta.exists():
        record = json.loads(meta.read_text(encoding='utf-8'))
        if record['request'] != request: raise RuntimeError(f'Cache request mismatch: {name}')
        if hashlib.sha256(dest.read_bytes()).hexdigest() != record['sha256']: raise RuntimeError(f'Cache checksum mismatch: {name}')
        print('CACHE', name, record.get('rows'), flush=True)
        return pd.read_csv(dest)
    print('FETCH', name, flush=True)
    record = {'name': name, 'request': request, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
    timer = threading.Timer(180, lambda: os._exit(2)); timer.start()
    try:
        df = func()
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df.to_csv(dest, index=False)
        record.update(status='success', rows=len(df), columns=[str(c) for c in df.columns], sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        meta.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('OK', name, len(df), flush=True)
        return df
    except Exception as exc:
        record.update(status='error', error=redact(exc)[:1000])
        (OUT / f'{name}.error.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('ERROR', name, record['error'], flush=True)
        return None
    finally:
        timer.cancel()

timer = threading.Timer(120, lambda: os._exit(2)); timer.start()
try: ld.open_session(name='desktop.workspace', app_key=credentials.get('LSEG_APP_KEY'))
finally: timer.cancel()

try:
    args = {'universe': ['.SPX'], 'fields': ['TR.IndexConstituentRIC', 'TR.IndexConstituentName']}
    cached('spx_current_anchor', args, lambda: ld.get_data(**args))
    for year in range(ANCHOR_START, ANCHOR_END + 1):
        # NOTE: only the 'ituentituent' spelling returns the Joiner/Leaver Change column; the
        # correctly spelled TR.IndexJLConstituentChange silently drops it. Verified 2026-09-07.
        args = {'universe': ['.SPX'],
                'fields': ['TR.IndexJLConstituentChangeDate', 'TR.IndexJLConstituentRIC', 'TR.IndexJLConstituentName', 'TR.IndexJLConstituentituentChange'],
                'parameters': {'SDate': f'{year}-01-01', 'EDate': f'{year}-12-31', 'IC': 'B'}}
        cached(f'spx_jl_{year}', args, lambda a=args: ld.get_data(**a))
finally:
    ld.close_session()
print('PROBE COMPLETE', flush=True)

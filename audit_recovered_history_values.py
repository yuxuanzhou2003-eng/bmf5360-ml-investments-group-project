"""Read-only development history sanity checks; no labels or imputation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
import numpy as np

root=Path(__file__).resolve().parent
key_run=root/'data/audit/local_history_keys/20260912T075125849398Z'
out=root/'data/audit/recovered_history_values'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True,exist_ok=False)
keys=pd.read_csv(key_run/'candidate_date_key_coverage.csv')
names=set(keys.Instrument)
def read(rel,cols):
    parts=[]
    for c in pd.read_csv(root/rel,usecols=cols,chunksize=150000):
        parts.append(c.loc[c.Instrument.isin(names)&c.Date.between('2021-01-01','2025-12-31')])
    return pd.concat(parts,ignore_index=True)
p=read('data/clean/v3/20260909T012417705069Z/prices.csv',['Instrument','Date','TRDPRC_1','raw_file'])
r=read('data/clean/v2/returns.csv',['Instrument','Date','return_decimal','raw_file'])
x=keys.merge(p.rename(columns={'raw_file':'price_source'}),on=['Instrument','Date'],how='left',validate='one_to_one').merge(r.rename(columns={'raw_file':'return_source'}),on=['Instrument','Date'],how='left',validate='one_to_one')
x['positive_finite_close']=np.isfinite(x.TRDPRC_1)&x.TRDPRC_1.gt(0)
x['finite_return']=np.isfinite(x.return_decimal)
x['return_ge_minus_one']=x.return_decimal.ge(-1)
x['source_pointers_present']=x.price_source.notna()&x.return_source.notna()
x['basic_value_checks_pass']=x.both_source_keys&x.positive_finite_close&x.finite_return&x.return_ge_minus_one&x.source_pointers_present
x['large_move_review_only']=x.return_decimal.abs().gt(.5)
x.to_csv(out/'row_value_audit.csv',index=False)
x.loc[~x.basic_value_checks_pass|x.large_move_review_only].to_csv(out/'review_queue.csv',index=False)
agg=x.groupby('Instrument').agg(record_days=('Date','size'),both_keys=('both_source_keys','sum'),basic_pass=('basic_value_checks_pass','sum'),added_days=('recoverable_key_candidate','sum'),large_moves=('large_move_review_only','sum'))
agg.to_csv(out/'company_value_audit.csv')
s={'status':'complete_basic_value_audit_not_model_ready','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'key_run':str(key_run.relative_to(root)),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'rows':len(x),'companies':x.Instrument.nunique(),'both_keys':int(x.both_source_keys.sum()),'basic_pass':int(x.basic_value_checks_pass.sum()),
 'recovered_basic_pass':int((x.recoverable_key_candidate&x.basic_value_checks_pass).sum()),'review_queue_rows':int((~x.basic_value_checks_pass|x.large_move_review_only).sum()),
 'large_moves_review_only':int(x.large_move_review_only.sum()),
 'rules':{'missing':'preserved, not filled','large_move':'absolute daily return > 50%, review only; not rejection','deletion':0,'targets_generated':0},
 'limitations':['Positive prices and finite returns are sanity checks only, not full corporate-action or survivorship validation.',
 'No listing dates or historical RIC continuity established. Source paths are retained; source rows not independently reconstructed.',
 'Inputs parse identifier/date and selected financial columns, output dates restricted to 2021-2025. No target file read.']}
(out/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps({'output':str(out),**s},indent=2))

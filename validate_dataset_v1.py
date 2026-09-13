"""Independent persisted-artifact checks, including recomputation of every target."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
base=ROOT/'data';checks={}
for entry in json.loads((base/'raw/pilot_v1/manifest.json').read_text()):
    checks['raw_hash_'+entry['file']]=hashlib.sha256((base/'raw/pilot_v1'/entry['file']).read_bytes()).hexdigest()==entry['sha256']
for path in (base/'raw/supplement_v1').glob('*.meta.json'):
    entry=json.loads(path.read_text());data=path.with_name(path.name.replace('.meta.json','.json'))
    if not data.exists():data=path.with_name(path.name.replace('.meta.json','.csv'))
    checks['supplement_hash_'+entry['name']]=hashlib.sha256(data.read_bytes()).hexdigest()==entry['sha256']
quality=json.loads((base/'audit/v1/quality_report.json').read_text(encoding='utf-8'))
for table in ['returns','actuals','estimates','news_versions']:
    stats=quality[table];checks['row_conservation_'+table]=stats['input_rows']==stats['clean_rows']+stats['quarantined_rows']
f=pd.read_csv(base/'diffusion_pilot_clean/features.csv');y=pd.read_csv(base/'diffusion_pilot_clean/labels.csv')
d=pd.read_csv(base/'diffusion_pilot_clean/diagnostics_ex_post.csv')
checks['feature_schema_has_no_labels']=not set(f).intersection(set(y)-{'sample_id'}) and not set(f).intersection(set(d)-{'sample_id'})
checks['unique_sample_ids']=all(not x.sample_id.duplicated().any() for x in [f,y,d])
checks['matching_sample_ids']=set(f.sample_id)==set(y.sample_id)==set(d.sample_id)
dates={k:pd.to_datetime(f[k]) for k in ['snapshot','announcement','graph_last_date']}
checks['snapshot_before_event']=bool((dates['snapshot']<dates['announcement'].dt.normalize()).all())
checks['network_before_event']=bool((dates['graph_last_date']<dates['announcement'].dt.normalize()).all())
rr=pd.read_csv(base/'clean/v1/returns.csv');rr['Date']=pd.to_datetime(rr.Date)
wide=rr.pivot(index='Date',columns='Instrument',values='return_decimal').sort_index()
joined=f[['sample_id','receiver','announcement']].merge(y,on='sample_id',validate='one_to_one')
matches=[]
for row in joined.itertuples():
    entry=pd.Timestamp(row.entry_close);exit_=pd.Timestamp(row.exit_close)
    v=wide.loc[(wide.index>entry)&(wide.index<=exit_),[row.receiver,'SPY.P']]
    got=(1+v[row.receiver]).prod()-1-((1+v['SPY.P']).prod()-1)
    matches.append(len(v)==5 and v.notna().all().all() and entry>pd.Timestamp(row.announcement).normalize() and np.isclose(got,row.forward_benchmark_excess,atol=1e-12,rtol=0))
checks['all_forward_targets_recomputed']=bool(all(matches))
n=pd.read_csv(base/'clean/v1/news_versions.csv')
first=pd.to_datetime(n.first_created,format='mixed',utc=True);available=pd.to_datetime(n.availability_utc,format='mixed',utc=True);version=pd.to_datetime(n.version_created,format='mixed',utc=True)
checks['news_not_backdated']=bool((available==version).all() and (available>=first).all())
checks['news_inside_collection_window']=bool(available.between(pd.Timestamp('2014-12-01',tz='UTC'),pd.Timestamp('2015-01-01',tz='UTC'),inclusive='left').all())
checks['news_unsaturated_window_coverage']=quality['news_requested_window_covered_by_unsaturated_subwindows']
master=pd.read_csv(base/'clean/v1/company_master_current_only.csv')
checks['current_industry_not_authorized_for_history']=bool((master.historical_use_allowed==False).all())
checks['no_missing_returns_imputed']=quality['returns']['input_rows']==quality['returns']['clean_rows']
result={'passed':sum(bool(x) for x in checks.values()),'total':len(checks),'failed':[k for k,v in checks.items() if not v],'checks':{k:bool(v) for k,v in checks.items()},'scope':'Engineering integrity only, not proof of PIT correctness or alpha'}
(base/'audit/v1/validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if result['failed']:raise RuntimeError('Dataset validation failed')

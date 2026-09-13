"""Audit identifier/date coverage only. No price/return values or targets read."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out = ROOT / 'data/audit/local_history_keys' / run_id
out.mkdir(parents=True, exist_ok=False)
registry_path = ROOT / 'data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv'
reg = pd.read_csv(registry_path)
names = set(reg.ric)
sources = {
    'prices': ('data/clean/v3/20260909T012417705069Z/prices.csv', 'Date'),
    'returns': ('data/clean/v2/returns.csv', 'Date'),
    'legacy_panel': ('data/model_ready_ai_pool_daily_v1/20260910T060000000000Z/metadata.csv', 'formation_session'),
}
tables, manifest = {}, []
for label, (rel, date_col) in sources.items():
    path = ROOT / rel
    parts = []
    scanned = 0
    for chunk in pd.read_csv(path, usecols=['Instrument', date_col], chunksize=150000):
        scanned += len(chunk)
        keep = chunk.Instrument.isin(names) & chunk[date_col].between('2021-01-01', '2025-12-31')
        parts.append(chunk.loc[keep].rename(columns={date_col: 'Date'}))
    df = pd.concat(parts, ignore_index=True)
    duplicate = df.duplicated(['Instrument', 'Date'], keep=False)
    df.loc[duplicate].to_csv(out / f'{label}_duplicate_keys.csv', index=False)
    if duplicate.any():
        raise ValueError(f'Duplicate keys in {label}; audit stopped, no silent deduplication')
    tables[label] = df
    manifest.append({'source': label, 'path': rel, 'read_columns': ['Instrument', date_col],
                     'scanned_key_rows': scanned, 'selected_key_rows': len(df),
                     'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})

price = tables['prices'].assign(price_key=True)
ret = tables['returns'].assign(return_key=True)
keys = price.merge(ret, on=['Instrument','Date'], how='outer', validate='one_to_one')
keys = keys.merge(tables['legacy_panel'].assign(legacy_key=True), on=['Instrument','Date'], how='outer', validate='one_to_one')
for col in ['price_key','return_key','legacy_key']:
    # These are presence flags from merges, not missing financial observations.
    keys[col] = keys[col].eq(True)
keys['both_source_keys'] = keys.price_key & keys.return_key
keys['recoverable_key_candidate'] = keys.both_source_keys & ~keys.legacy_key
keys = keys.merge(reg[['ric','primary_group','member_from','member_to']], left_on='Instrument', right_on='ric', validate='many_to_one')
keys['outside_legacy_span'] = ~keys.Date.between(keys.member_from, keys.member_to)
keys['status'] = keys.apply(lambda r: 'LOCAL_KEYS_OUTSIDE_LEGACY_PANEL_NOT_YET_VALUE_VALIDATED' if r.recoverable_key_candidate else ('LEGACY_PANEL_KEY' if r.legacy_key else 'ONLY_ONE_SOURCE_KEY'), axis=1)
keys.to_csv(out / 'candidate_date_key_coverage.csv', index=False)
rows=[]
for rec in reg.to_dict('records'):
    g=keys.loc[keys.Instrument.eq(rec['ric'])]
    rows.append({'ric':rec['ric'],'name':rec['canonical_name'],'legacy_primary_group':rec['primary_group'],
                 'legacy_member_from':rec['member_from'],'legacy_member_to':rec['member_to'],
                 'price_key_days':int(g.price_key.sum()),'return_key_days':int(g.return_key.sum()),
                 'both_source_key_days':int(g.both_source_keys.sum()),'legacy_panel_days':int(g.legacy_key.sum()),
                 'additional_local_key_days':int(g.recoverable_key_candidate.sum()),
                 'first_both_key_date':g.loc[g.both_source_keys,'Date'].min(),
                 'last_both_key_date':g.loc[g.both_source_keys,'Date'].max(),
                 'listing_dates_verified':False,'financial_values_validated':False})
companies=pd.DataFrame(rows)
companies.to_csv(out/'company_history_coverage.csv',index=False)
summary={'run_id':run_id,'status':'complete_key_coverage_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'inputs':manifest,'registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'candidate_count':len(reg),'source_both_key_rows':int(keys.both_source_keys.sum()),
         'legacy_panel_rows':int(keys.legacy_key.sum()),'additional_local_key_rows':int(keys.recoverable_key_candidate.sum()),
         'companies_with_additional_keys':int(companies.additional_local_key_days.gt(0).sum()),
         'checks':{'all_candidates_retained':len(companies)==len(reg),'unique_keys':not keys.duplicated(['Instrument','Date']).any(),
                   'date_window':bool(keys.Date.between('2021-01-01','2025-12-31').all()),'no_financial_value_columns_read':True},
         'limitations':['Key presence is not evidence of a valid price or return, tradability, listing date or historical AI membership.',
                        'No source values or target files were read. Large source file hashes intentionally not calculated; size and modification time recorded.',
                        'Boolean absent merge flags are False; no financial missing value was filled. Raw inputs unchanged.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'output':str(out),**{k:summary[k] for k in ['candidate_count','source_both_key_rows','legacy_panel_rows','additional_local_key_rows','companies_with_additional_keys','checks']}},indent=2))
print(companies.loc[companies.ric.isin(['CIEN.N','MRVL.OQ','VRT.N','PLTR.OQ'])].to_string(index=False))

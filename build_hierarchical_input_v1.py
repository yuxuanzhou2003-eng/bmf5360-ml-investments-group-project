"""Transparent US screenshot panel, no forward-fill or future selection."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
out=ROOT/'data/model_ready_hierarchical_v1'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True);(out/'quarantine').mkdir()
base=ROOT/'data/audit/screenshot_taxonomy_v1/20260913025822Z'
c=pd.read_csv(base/'candidate_entities.csv').fillna('');roles=pd.read_csv(base/'role_memberships.csv').fillna('')
mapping=dict(zip(c.loc[c.matched_local_ric.ne(''),'entity_key'],c.loc[c.matched_local_ric.ne(''),'matched_local_ric']))
for rel in ['data/audit/screenshot_additions/20260913T030723Z/additional_candidates.csv','data/audit/screenshot_discovery_review_v2/us_additions.csv']:
    x=pd.read_csv(ROOT/rel);mapping.update(dict(zip(x.entity_key,x.matched_local_ric)))
reg=c[c.entity_key.isin(mapping)].copy();reg['Instrument']=reg.entity_key.map(mapping)
first=roles.sort_values('source_role_id').drop_duplicates('entity_key').set_index('entity_key')
reg['primary_group']=reg.entity_key.map(first.source_group);reg['subsector']=reg.entity_key.map(first.source_subgroup)
for symbol,group in {'NVDA.OQ':'Processor','AMD.OQ':'Processor','AVGO.OQ':'Semi Production','MRVL.OQ':'Semi Production','QCOM.OQ':'Semi Production','INTC.OQ':'Semi Production','HPE.N':'Server','VRT.N':'Internal Power / Cooling'}.items():
    mask=reg.Instrument.eq(symbol);reg.loc[mask,'primary_group']=group
    entity=reg.loc[mask,'entity_key'].iloc[0];rr=roles[roles.entity_key.eq(entity)&roles.source_group.eq(group)]
    if len(rr):reg.loc[mask,'subsector']=rr.iloc[0].source_subgroup
assert len(reg)==34 and reg.Instrument.is_unique
reg.to_csv(out/'registry.csv',index=False);roles.to_csv(out/'all_source_roles.csv',index=False)
c[~c.entity_key.isin(mapping)].assign(execution_status='OUTSIDE_CURRENT_US_VERIFIED_EXECUTION_SCOPE_RETAINED').to_csv(out/'quarantine/global_candidates_not_executed.csv',index=False)
sources=[]
def read_local(rel,cols):
    path=ROOT/rel;parts=[];count=0
    for x in pd.read_csv(path,usecols=cols,chunksize=100000):
        count+=len(x);parts.append(x[x.Instrument.isin(list(reg.Instrument)+['SPY.P'])&x.Date.between('2020-01-01','2025-12-31')].copy())
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    sources.append(dict(path=rel,rows=count,sha256=h.hexdigest()))
    return pd.concat(parts,ignore_index=True)
r=read_local('data/clean/v2/returns.csv',['Instrument','Date','return_decimal','raw_file'])
p=read_local('data/clean/v3/20260909T012417705069Z/prices.csv',['Instrument','Date','TRDPRC_1','raw_file'])
fresh_r=[];fresh_p=[]
for run in ['20260913T032531Z','20260913T033702Z']:
    directory=ROOT/'data/raw/screenshot_additions_v2'/run
    for file in directory.glob('returns_*.csv'):
        x=pd.read_csv(file);x['return_decimal']=pd.to_numeric(x['Total Return'],errors='coerce')/100;x['raw_file']=str(file.relative_to(ROOT));fresh_r.append(x[['Instrument','Date','return_decimal','raw_file']])
    for file in directory.glob('prices_*.csv'):
        x=pd.read_csv(file,header=[0,1],index_col=0)
        for ric in x.columns.get_level_values(0).unique():
            fresh_p.append(pd.DataFrame(dict(Instrument=ric,Date=x.index,TRDPRC_1=pd.to_numeric(x[(ric,'TRDPRC_1')],errors='coerce').to_numpy(),raw_file=str(file.relative_to(ROOT)))))
    sources.append(dict(path=str(directory.relative_to(ROOT)),manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()))
fr=pd.concat(fresh_r,ignore_index=True);fp=pd.concat(fresh_p,ignore_index=True)
refresh=set(fr.Instrument);r=r[~r.Instrument.isin(refresh)];p=p[~p.Instrument.isin(refresh)]
r=pd.concat([r,fr],ignore_index=True);p=pd.concat([p,fp],ignore_index=True)
def unique(x,field,kind):
    dup=x.duplicated(['Instrument','Date'],keep=False)
    x.loc[dup].to_csv(out/f'quarantine/{kind}_duplicate_original_rows.csv',index=False)
    conflicts=x.groupby(['Instrument','Date'])[field].nunique(dropna=False).gt(1)
    conflictkeys=set(conflicts[conflicts].index)
    z=x.drop_duplicates(['Instrument','Date']).copy()
    mask=pd.Series([(a,b) in conflictkeys for a,b in zip(z.Instrument,z.Date)],index=z.index)
    z.loc[mask,field]=np.nan
    return z,dict(duplicate_source_rows=int(dup.sum()),conflicting_keys=len(conflictkeys))
r,rc=unique(r,'return_decimal','returns');p,pc=unique(p,'TRDPRC_1','prices')
calendar=sorted(r.loc[r.Instrument.eq('SPY.P'),'Date'].unique());assert len(calendar)>1400
r[r.Instrument.eq('SPY.P')].to_csv(out/'spy_daily.csv',index=False)
grid=pd.MultiIndex.from_product([calendar,reg.Instrument],names=['Date','Instrument']).to_frame(index=False)
grid=grid.merge(reg[['Instrument','primary_group','subsector']],on='Instrument',validate='many_to_one').merge(r,on=['Instrument','Date'],how='left',validate='one_to_one').rename(columns={'raw_file':'return_source'}).merge(p,on=['Instrument','Date'],how='left',validate='one_to_one').rename(columns={'TRDPRC_1':'close','raw_file':'price_source'}).sort_values(['Instrument','Date'])
grid['observation_status']=np.where(np.isfinite(grid.return_decimal)&grid.return_decimal.ge(-1)&np.isfinite(grid.close)&grid.close.gt(0),'VALID','MISSING_OR_INVALID')
grid.loc[grid.observation_status.ne('VALID')].to_csv(out/'quarantine/missing_or_invalid_rows.csv',index=False)
price_change=grid.groupby('Instrument').close.pct_change(fill_method=None)
grid['return_price_discrepancy']=grid.return_decimal-price_change
flag=grid.return_price_discrepancy.abs().gt(.15)|grid.return_decimal.abs().gt(.4)
grid.loc[flag].to_csv(out/'quarantine/corporate_action_review_flags.csv',index=False)
grid.to_csv(out/'stock_daily.csv',index=False)
macrofile=ROOT/'data/clean/daily_macro_v1/20260910T062700000000Z/macro_features.csv'
m=pd.read_csv(macrofile);m=m[m.spy_session.between('2020-01-01','2025-12-31')];assert (m.reference_spy_session<m.spy_session).all()
features=['vix_level','vix_change_5d','dgs2_level','dgs2_change_20d','dgs10_change_20d','term_spread_10y_2y','broad_dollar_change_20d','dgs3mo_level']
m[['spy_session','reference_spy_session']+features].rename(columns={'spy_session':'Date'}).to_csv(out/'macro_daily.csv',index=False)
m.to_csv(out/'macro_provenance.csv',index=False)
cov=grid.groupby(['Instrument','primary_group']).agg(rows=('Date','size'),valid=('observation_status',lambda x:int(x.eq('VALID').sum())),missing_return=('return_decimal',lambda x:int(x.isna().sum())),missing_close=('close',lambda x:int(x.isna().sum()))).reset_index();cov.to_csv(out/'coverage.csv',index=False)
summary=dict(status='structural_panel_built_action_flags_require_review',companies=len(reg),groups=reg.primary_group.value_counts().to_dict(),rows=len(grid),dates=len(calendar),valid_rows=int(grid.observation_status.eq('VALID').sum()),flagged_rows=int(flag.sum()),duplicate_returns=rc,duplicate_prices=pc,sources=sources,macro_sha256=hashlib.sha256(macrofile.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rules='Fresh15company vendor sources replace old version for those instruments, not cherry-picked by date; remaining19 use existing local. Raw retained. No filling/winsorization. USD listed universe; currency metadata pending. One execution group each; static current classification not PIT.')
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
for log in ['DATA_PROCESSING_LOG.md','AI_USE_LOG.md']:
    with (ROOT/log).open('a',encoding='utf-8') as f:f.write(f'\n\n## Hierarchical input build {datetime.now(timezone.utc).isoformat()}\nExecuted build_hierarchical_input_v1.py. Input hashes, rules/counts in {out.relative_to(ROOT)}/summary.json. Output 34-company full grid {len(grid)} rows; valid {summary["valid_rows"]}; action flags {int(flag.sum())}. Invalid rows/duplicates retained in quarantine; no imputation, winsorization, or raw deletion. Global out-of-execution candidates retained with scope reason. Mapping rule preregistered in protocol. Not yet certified for model execution.\n')
print(out);print(json.dumps(summary,indent=2))

"""Versioned raw snapshot, explicit quarantine, clean tables and data audit."""
import hashlib
import itertools
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'/'pilot_v1'
CLEAN=ROOT/'data'/'clean'/'v1'
AUDIT=ROOT/'data'/'audit'/'v1'
for directory in [RAW,CLEAN,AUDIT]:directory.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'diffusion_pilot_config.json').read_text(encoding='utf-8'))
records=json.loads((ROOT/'data/access_probe/diffusion_pilot_summary.json').read_text(encoding='utf-8'))
meta={r['name']:r for r in records}
raw_manifest=[]
for stem in ['pilot_returns','pilot_actuals','pilot_estimates']:
    for i in range(3):
        src=ROOT/'data/access_probe'/f'{stem}_{i}.csv';dst=RAW/src.name
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise RuntimeError('Raw snapshot changed; create a new dataset version')
        if not dst.exists():shutil.copy2(src,dst)
        raw_manifest.append({'file':dst.name,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'request':meta[src.stem]['request'],'retrieved_at_utc':meta[src.stem]['tested_at_utc']})
src=ROOT/'data/access_probe/pilot_benchmark.csv';dst=RAW/src.name
if dst.exists() and dst.read_bytes()!=src.read_bytes():raise RuntimeError('Benchmark snapshot changed')
if not dst.exists():shutil.copy2(src,dst)
raw_manifest.append({'file':dst.name,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'request':meta[src.stem]['request'],'retrieved_at_utc':meta[src.stem]['tested_at_utc']})
(RAW/'manifest.json').write_text(json.dumps(raw_manifest,indent=2),encoding='utf-8')

audit={};quarantines=[]
def load(prefix):
    frames=[]
    for path in sorted(RAW.glob(prefix+'*.csv')):
        df=pd.read_csv(path,index_col=0).replace(r'^\s*$',pd.NA,regex=True)
        df['raw_file']=path.name;df['raw_row']=np.arange(len(df));frames.append(df)
    return pd.concat(frames,ignore_index=True)

def clean(name,df,dates,numeric,required,keys,positive=(),ordered_dates=()):
    original_cols=[c for c in df if c not in ['raw_file','raw_row']]
    stats={'input_rows':len(df)}
    for col in dates:df[col]=pd.to_datetime(df[col],format='mixed',errors='coerce')
    for col in numeric:df[col]=pd.to_numeric(df[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    reason=pd.Series('',index=df.index)
    invalid=df[list(required)].isna().any(axis=1)
    reason.loc[invalid]='missing_or_unparseable_required_field'
    for col in positive:reason.loc[(reason=='')&(df[col]<0)]='negative_'+col
    exact=df.duplicated(original_cols,keep='first')
    reason.loc[(reason=='')&exact]='exact_duplicate'
    eligible=reason==''
    conflicts=df.loc[eligible].duplicated(list(keys),keep=False)
    reason.loc[conflicts.index[conflicts]]='conflicting_duplicate_key'
    if ordered_dates:
        first,last=ordered_dates
        reason.loc[(reason=='')&(df[last]<df[first])]='timestamp_order_contradiction'
    bad=df.loc[reason!=''].copy();bad['rejection_reason']=reason.loc[reason!=''];bad['table']=name
    bad.to_csv(AUDIT/f'{name}_quarantine.csv',index=False)
    quarantines.append(bad)
    good=df.loc[reason==''].copy().sort_values(list(keys)).reset_index(drop=True)
    good.to_csv(CLEAN/f'{name}.csv',index=False)
    stats.update(clean_rows=len(good),quarantined_rows=len(bad),reasons=bad.rejection_reason.value_counts().to_dict(),unique_key=not good.duplicated(list(keys)).any())
    if stats['input_rows']!=stats['clean_rows']+stats['quarantined_rows']:raise RuntimeError('Row accounting mismatch')
    audit[name]=stats
    return good

r=pd.concat([load('pilot_returns'),load('pilot_benchmark')],ignore_index=True)
r=clean('returns',r,['Date'],['Total Return'],['Instrument','Date','Total Return'],['Instrument','Date'])
r['return_decimal']=r['Total Return']/100
if (r.return_decimal < -1).any():raise RuntimeError('Returns below -100%')
r.to_csv(CLEAN/'returns.csv',index=False)
r.loc[r.return_decimal.abs()>.2].to_csv(AUDIT/'extreme_returns_review.csv',index=False)
a=clean('actuals',load('pilot_actuals'),['Report Date','Period End Date'],['Earnings Per Share - Actual'],['Instrument','Report Date','Period End Date','Earnings Per Share - Actual'],['Instrument','Report Date','Period End Date'])
e=clean('estimates',load('pilot_estimates'),['Calc Date','Period End Date'],['Earnings Per Share - Mean','Earnings Per Share - Standard Deviation','Earnings Per Share - Number of Included Estimates'],['Instrument','Calc Date','Period End Date','Earnings Per Share - Mean'],['Instrument','Calc Date','Period End Date'],['Earnings Per Share - Standard Deviation','Earnings Per Share - Number of Included Estimates'])
coverage=[]
bench_dates=set(r.loc[r.Instrument==cfg['benchmark'],'Date'])
for ric in cfg['instruments']+[cfg['benchmark']]:
    rr=r.loc[r.Instrument==ric];aa=a.loc[a.Instrument==ric];ee=e.loc[e.Instrument==ric]
    coverage.append({'instrument':ric,'return_rows':len(rr),'first_return':rr.Date.min(),'last_return':rr.Date.max(),'missing_benchmark_sessions':len(bench_dates-set(rr.Date)),'actual_rows':len(aa),'estimate_rows':len(ee),'zero_return_fraction':float(rr.return_decimal.eq(0).mean()),'extreme_abs_return_gt_20pct':int(rr.return_decimal.abs().gt(.2).sum())})
pd.DataFrame(coverage).to_csv(AUDIT/'coverage_by_company.csv',index=False)

# News: retain versions, use versionCreated for availability, and separate tags from verified business links.
SUP=ROOT/'data/raw/supplement_v1'
news_rows=[];windows=[]
for path in sorted(SUP.glob('news_*.json')):
    if path.name.endswith('.meta.json'):continue
    raw=json.loads(path.read_text(encoding='utf-8'))
    parts=[raw] if isinstance(raw,dict) else raw
    all_items=[item for part in parts for item in part.get('data',[])]
    request=json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))['request']
    windows.append({'file':path.name,'start':request['date_from'],'end':request['date_to'],'rows':len(all_items),'saturated':len(all_items)>=100})
    for j,item in enumerate(all_items):
        node=item.get('newsItem',{});im=node.get('itemMeta',{});cm=node.get('contentMeta',{})
        codes=sorted({s.get('_qcode','')[2:] for s in cm.get('subject',[]) if s.get('_qcode','').startswith('R:')})
        title=(im.get('title') or [{}])[0].get('$','')
        news_rows.append({'story_id':item.get('storyId'),'first_created':im.get('firstCreated',{}).get('$'),'version_created':im.get('versionCreated',{}).get('$'),'headline':title,'rics_json':json.dumps(codes),'raw_file':path.name,'raw_row':j})
if news_rows:
    news_df=pd.DataFrame(news_rows)
    for col in ['first_created','version_created']:news_df[col]=pd.to_datetime(news_df[col],format='mixed',utc=True,errors='coerce')
    # Null strings are rejected by the same audit mechanism; versions are never backdated to firstCreated.
    news_df=news_df.replace(r'^\s*$',pd.NA,regex=True)
    n=clean('news_versions',news_df,[],[],['story_id','first_created','version_created','headline'],['story_id','version_created'],ordered_dates=('first_created','version_created'))
    n['availability_utc']=n.version_created
    n['family_id']=n.story_id.str.replace(r':\d+$','',regex=True)
    n['candidate_roundup']=n.headline.str.contains(r'BUZZ|MARKET[S]?[- ]|STOCKS[- ]|DIARY|FACTBOX|BRIEF|SUMMARY',case=False,regex=True,na=False)
    n.to_csv(CLEAN/'news_versions.csv',index=False)
    pairs=[]
    for row in n.itertuples():
        inside=sorted(set(json.loads(row.rics_json))&set(cfg['instruments']))
        for left,right in itertools.combinations(inside,2):
            pairs.append({'left':left,'right':right,'family_id':row.family_id,'story_id':row.story_id,'availability_utc':row.availability_utc,'candidate_roundup':row.candidate_roundup,'tagged_company_count':len(json.loads(row.rics_json))})
    pairs_df=pd.DataFrame(pairs,columns=['left','right','family_id','story_id','availability_utc','candidate_roundup','tagged_company_count'])
    pairs_df=pairs_df.sort_values('availability_utc').drop_duplicates(['left','right','family_id'],keep='first')
    pairs_df['relation_status']='unreviewed_news_cotag_not_verified_business_link'
    pairs_df.to_csv(CLEAN/'news_cotag_candidates.csv',index=False)
    audit['news_candidate_pairs']={'rows':len(pairs_df),'distinct_pairs':int(pairs_df[['left','right']].drop_duplicates().shape[0]),'not_business_relationships':True}
    pd.DataFrame(windows).to_csv(AUDIT/'news_windows.csv',index=False)
    leaves=[w for w in windows if not w['saturated']]
    merged=[]
    for w in sorted(leaves,key=lambda x:x['start']):
        s=pd.Timestamp(w['start']);t=pd.Timestamp(w['end'])
        if merged and s<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],t)
        else:merged.append([s,t])
    audit['news_requested_window_covered_by_unsaturated_subwindows']=len(merged)==1 and merged[0][0]<=pd.Timestamp('2014-12-01') and merged[0][1]>=pd.Timestamp('2015-01-01')

master=SUP/'company_master_current.csv'
if master.exists():
    m=pd.read_csv(master);m['snapshot_retrieved_utc']=json.loads(master.with_suffix('.meta.json').read_text())['retrieved_at_utc'];m['historical_use_allowed']=False
    m.to_csv(CLEAN/'company_master_current_only.csv',index=False)
audit['dataset_status']='engineering_clean_not_final_research_ready'
audit['blockers']=['survivor-biased pilot universe','historical industry/relationship validity not established','EPS original-vintage and adjustment methodology unverified','vendor earnings timezone unverified','news only December 2014 supplement, not full 2015-2017','transaction and shorting costs not collected']
audit['config_sha256']=hashlib.sha256((ROOT/'diffusion_pilot_config.json').read_bytes()).hexdigest()
(AUDIT/'quality_report.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))

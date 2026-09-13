"""Build stock x date research panel from local history, preserving missing rows."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REG = ROOT / 'data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv'
RET = ROOT / 'data/clean/v2/returns.csv'
PRICE = ROOT / 'data/clean/v3/20260909T012417705069Z/prices.csv'
START, END = pd.Timestamp('2021-01-01'), pd.Timestamp('2025-12-31')
ROLES = ['gpu_accelerator','ai_semiconductor','memory_hbm_storage','server_network','cloud_software','data_center_power_cooling','robotics_autonomy']
SUBSECTOR = {
    'gpu_accelerator':'accelerator_processor', 'ai_semiconductor':'supporting_semiconductor',
    'memory_hbm_storage':'memory_storage', 'server_network':'server_network_optical',
    'cloud_software':'cloud_ai_software', 'data_center_power_cooling':'data_center_power_cooling',
    'robotics_autonomy':'robotics_autonomy'
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'data/model_ready_multilevel_stock_date_v1'/run_id
    out.mkdir(parents=True,exist_ok=False); (out/'quarantine').mkdir()
    reg=pd.read_csv(REG,parse_dates=['member_from','member_to'])
    reg=reg[['ric','canonical_name','primary_group','member_from','member_to','pit_evidence_status','delisted_ric']].copy()
    reg['proposed_subsector']=reg.primary_group.map(SUBSECTOR)
    reg['subsector_status']=np.where(reg.primary_group.isin(['server_network','data_center_power_cooling']),'COARSE_ROLE_PENDING_EVIDENCE','ROLE_TO_SUBSECTOR_STATIC_MAP')
    reg.to_csv(out/'company_taxonomy_snapshot.csv',index=False)
    r=pd.read_csv(RET,usecols=['Instrument','Date','return_decimal','raw_file'],parse_dates=['Date'])
    r=r.loc[r.Date.between(START,END)&r.Instrument.isin(reg.ric)].copy()
    r['return_decimal']=pd.to_numeric(r.return_decimal,errors='coerce')
    r=r.drop_duplicates(['Instrument','Date'],keep=False)
    p=pd.read_csv(PRICE,usecols=['Instrument','Date','TRDPRC_1','BID','ASK','raw_file'],parse_dates=['Date'])
    p=p.loc[p.Date.between(START,END)&p.Instrument.isin(reg.ric)].copy()
    p=p.drop_duplicates(['Instrument','Date'],keep=False)
    sessions=pd.read_csv(RET,usecols=['Instrument','Date'],parse_dates=['Date'])
    sessions=sessions.loc[sessions.Instrument.eq('SPY.P')&sessions.Date.between(START,END),['Date']].drop_duplicates().sort_values('Date')
    if sessions.empty: raise ValueError('No SPY sessions in window')
    grid=sessions.assign(_k=1).merge(reg.assign(_k=1),on='_k').drop(columns='_k')
    panel=grid.merge(r.rename(columns={'raw_file':'return_source'}),left_on=['ric','Date'],right_on=['Instrument','Date'],how='left',validate='one_to_one').drop(columns=['Instrument'])
    panel=panel.merge(p.rename(columns={'raw_file':'price_source'}),left_on=['ric','Date'],right_on=['Instrument','Date'],how='left',validate='one_to_one').drop(columns=['Instrument'])
    panel['legacy_span_active']=panel.Date.between(panel.member_from,panel.member_to)
    panel['has_return']=panel.return_decimal.notna()&np.isfinite(panel.return_decimal)
    panel['has_close']=panel.TRDPRC_1.notna()&np.isfinite(panel.TRDPRC_1)&panel.TRDPRC_1.gt(0)
    panel['has_two_sided_quote']=panel.BID.notna()&panel.ASK.notna()&panel.BID.gt(0)&panel.ASK.ge(panel.BID)
    panel['basic_observation_available']=panel.has_return&panel.has_close
    panel['reason_code']=np.select([panel.basic_observation_available,~panel.has_return&~panel.has_close,~panel.has_return,~panel.has_close],['VALID_LOCAL_OBSERVATION','NO_RETURN_AND_CLOSE','NO_RETURN_OBSERVATION','NO_VALID_CLOSE'],default='REVIEW')
    panel['formation_session']=panel.Date
    panel['split']=np.select([panel.Date.le(pd.Timestamp('2023-12-31')),panel.Date.between('2024-01-01','2025-12-31')],['training','development_validation'],'out_of_window')
    panel=panel[['formation_session','Date','ric','canonical_name','primary_group','proposed_subsector','subsector_status','pit_evidence_status','delisted_ric','member_from','member_to','legacy_span_active','return_decimal','TRDPRC_1','BID','ASK','return_source','price_source','has_return','has_close','has_two_sided_quote','basic_observation_available','reason_code','split']]
    panel.to_csv(out/'stock_date_panel.csv',index=False)
    panel.loc[~panel.basic_observation_available].to_csv(out/'quarantine/missing_stock_date_observations.csv',index=False)
    cov=panel.groupby(['ric','primary_group','proposed_subsector'],as_index=False).agg(total_date_rows=('Date','size'),return_rows=('has_return','sum'),close_rows=('has_close','sum'),basic_rows=('basic_observation_available','sum'),legacy_span_rows=('legacy_span_active','sum'),first_local_date=('Date','min'),last_local_date=('Date','max'))
    cov.to_csv(out/'company_date_coverage.csv',index=False)
    role=panel.groupby(['formation_session','primary_group'],as_index=False).agg(candidate_rows=('ric','size'),basic_rows=('basic_observation_available','sum'),return_rows=('has_return','sum'),close_rows=('has_close','sum'))
    role.to_csv(out/'date_subsector_coverage.csv',index=False)
    counts=panel.groupby('split',dropna=False).agg(rows=('ric','size'),companies=('ric','nunique'),dates=('Date','nunique'),basic_rows=('basic_observation_available','sum'),return_rows=('has_return','sum'),close_rows=('has_close','sum')).reset_index()
    counts.to_csv(out/'split_counts.csv',index=False)
    summary={'schema_version':'multilevel_stock_date_panel_v1','run_id':run_id,'status':'complete_structural_panel_not_model_ready','window':{'start':str(START.date()),'end':str(END.date())},'inputs':{str(REG):sha(REG),str(RET):sha(RET),str(PRICE):sha(PRICE)},'candidate_count':len(reg),'date_count':len(sessions),'panel_rows':len(panel),'basic_observation_rows':int(panel.basic_observation_available.sum()),'missing_observation_rows':int((~panel.basic_observation_available).sum()),'companies_with_any_basic_rows':int(panel.groupby('ric').basic_observation_available.any().sum()),'counts':counts.to_dict('records'),'processing':{'row_deletion':'none; full 49 x SPY-session grid retained','return_imputation':'none','price_imputation':'none','winsorization':'none','forward_fill':'none','identifier_remapping':'none','future_targets_constructed':False},'checks':{'all_registry_candidates_retained':len(reg)==49,'unique_grid_keys':not panel.duplicated(['ric','Date']).any(),'dates_in_window':bool(panel.Date.between(START,END).all()),'no_nonfinite_valid_returns':bool(np.isfinite(panel.loc[panel.has_return,'return_decimal']).all()),'no_invalid_positive_closes':bool((panel.loc[panel.has_close,'TRDPRC_1']>0).all()),'no_future_target_read':True},'limitations':['The old member span is annotated, not used to remove rows; it is not an IPO or AI-role date.','Static primary groups are research labels and server/network plus power/cooling remain coarse pending evidence.','No listing/ric continuity, corporate-action or tradability validation is asserted by this panel.','No target, model output or future-period label was read or constructed.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'output':str(out),'status':summary['status'],'panel_rows':len(panel),'basic_rows':int(panel.basic_observation_available.sum()),'missing_rows':int((~panel.basic_observation_available).sum()),'companies_with_any_basic_rows':summary['companies_with_any_basic_rows'],'checks':summary['checks']},indent=2))
if __name__=='__main__': main()

"""Build ex-ante features separately from delayed, equal-sleeve AI targets."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STOCK = ROOT / 'data/model_ready_multilevel_stock_date_v1/20260912T080236168428Z/stock_date_panel.csv'
FACT = ROOT / 'data/model_ready_subsector_factors_v1/20260912T080445965839Z/subsector_date_factors.csv'
MACROFILE = ROOT / 'data/clean/daily_macro_v1/20260910T062700000000Z/macro_features.csv'
TECH = ['subsector_return_1', 'momentum_5', 'momentum_20', 'momentum_60', 'volatility_20_ann', 'breadth_positive_20', 'dispersion_1']
MACRO = ['vix_level', 'vix_change_5d', 'dgs2_level', 'dgs2_change_20d', 'dgs10_change_20d', 'term_spread_10y_2y', 'broad_dollar_change_20d', 'dgs3mo_level']
SERIES = ['VIXCLS', 'DGS2', 'DGS10', 'DGS3MO', 'DTWEXBGS']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    out = ap.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    stock = pd.read_csv(STOCK, parse_dates=['Date'])
    assert stock.Date.between('2021-01-01', '2025-12-31').all()
    assert not stock.duplicated(['Date', 'ric']).any()
    assert stock.groupby('ric').primary_group.nunique().eq(1).all()
    f = pd.read_csv(FACT, parse_dates=['Date']).sort_values(['Date', 'primary_group']).reset_index(drop=True)
    assert f.Date.between('2021-01-01', '2025-12-31').all()
    assert not f.duplicated(['Date', 'primary_group']).any()
    days = pd.DatetimeIndex(sorted(f.Date.unique()))
    roles = sorted(f.primary_group.unique())
    assert len(roles) == 7 and f.groupby('Date').size().eq(7).all()
    assert days.equals(pd.DatetimeIndex(sorted(stock.Date.unique())))
    # Feature stage is complete and saved before any label is constructed.
    for c in TECH:
        f[c + '_csz'] = (f[c] - f.groupby('Date')[c].transform('mean')) / f.groupby('Date')[c].transform('std').replace(0, np.nan)
    meta = ['spy_session', 'reference_spy_session'] + [s + suffix for s in SERIES for suffix in ['_value_observation_date', '_carry_age_spy_sessions', '_reason_code']]
    m = pd.read_csv(MACROFILE, usecols=meta + MACRO, parse_dates=['spy_session', 'reference_spy_session'])
    m = m.loc[m.spy_session.between('2021-01-01', '2025-12-31')].copy()
    assert not m.spy_session.duplicated().any()
    assert (m.reference_spy_session < m.spy_session).all()
    macro_audit = []
    for s in SERIES:
        vdate = pd.to_datetime(m[s + '_value_observation_date'])
        assert (vdate.dropna() <= m.loc[vdate.notna(), 'reference_spy_session']).all()
        ages = m[s + '_carry_age_spy_sessions']
        macro_audit.append({'series': s, 'dates': len(m), 'value_date_missing': int(vdate.isna().sum()),
            'max_carry_sessions': float(ages.max()), 'carry_over5_sessions': int(ages.gt(5).sum()),
            'reason_counts': m[s + '_reason_code'].value_counts(dropna=False).to_dict()})
    f = f.merge(m.rename(columns={'spy_session': 'Date'}), on='Date', how='left', validate='many_to_one').sort_values(['Date', 'primary_group']).reset_index(drop=True)
    required = [c + '_csz' for c in TECH] + MACRO
    rowok = np.isfinite(f[required]).all(axis=1)
    f['feature_eligible'] = rowok.groupby(f.Date).transform('all')
    f['feature_missing_fields'] = [','.join(np.asarray(required)[~np.isfinite(x)]) for x in f[required].to_numpy()]
    f.to_csv(out / 'features.csv', index=False)
    print('Saved target-independent features', flush=True)
    observations_ok = np.isfinite(stock.return_decimal) & stock.return_decimal.ge(-1) & np.isfinite(stock.TRDPRC_1) & stock.TRDPRC_1.gt(0)
    obs_ex = stock.loc[~observations_ok].copy()
    obs_ex['formation_exclusion_reason'] = 'MISSING_OR_INVALID_KNOWN_RETURN_OR_CLOSE'
    obs_ex.to_csv(out / 'source_observation_exclusions.csv', index=False)
    target_rows, member_rows = [], []
    for role in roles:
        group = stock.loc[stock.primary_group.eq(role)]
        rr = group.pivot(index='Date', columns='ric', values='return_decimal').reindex(days)
        pp = group.pivot(index='Date', columns='ric', values='TRDPRC_1').reindex(index=days, columns=rr.columns)
        ra, pa, tickers = rr.to_numpy(), pp.to_numpy(), np.asarray(rr.columns)
        for i, d in enumerate(days):
            good = np.isfinite(ra[i]) & (ra[i] >= -1) & np.isfinite(pa[i]) & (pa[i] > 0)
            names = tickers[good]
            member_rows.extend({'Date': d, 'primary_group': role, 'ric': ric, 'initial_weight_within_sleeve': 1 / len(names)} for ric in names)
            rec = {'Date': d, 'primary_group': role, 'formation_valid_member_count': len(names),
                'entry_session': days[i + 1] if i + 1 < len(days) else pd.NaT,
                'target_end_session': days[i + 6] if i + 6 < len(days) else pd.NaT,
                'subsector_forward_return': np.nan, 'sleeve_target_available': False,
                'missing_entry_rics': '', 'missing_exit_rics': '', 'missing_future_return_rics': '', 'target_reason': ''}
            if i + 6 >= len(days):
                rec['target_reason'] = 'FUTURE_SESSIONS_OUTSIDE_DEVELOPMENT'
            elif not len(names):
                rec['target_reason'] = 'NO_FORMATION_MEMBERS'
            else:
                future = ra[i + 2:i + 7, :][:, good]
                entry, end = pa[i + 1, good], pa[i + 6, good]
                badentry = ~np.isfinite(entry) | (entry <= 0)
                badexit = ~np.isfinite(end) | (end <= 0)
                badret = (~np.isfinite(future) | (future < -1)).any(axis=0)
                rec['missing_entry_rics'] = '|'.join(names[badentry])
                rec['missing_exit_rics'] = '|'.join(names[badexit])
                rec['missing_future_return_rics'] = '|'.join(names[badret])
                if badentry.any() or badexit.any() or badret.any():
                    rec['target_reason'] = 'FORMATION_HOLDING_FUTURE_DATA_MISSING_OR_INVALID'
                else:
                    rec['subsector_forward_return'] = float(((1 + future).prod(axis=0) - 1).mean())
                    rec['sleeve_target_available'] = True
            target_rows.append(rec)
    t = pd.DataFrame(target_rows).sort_values(['Date', 'primary_group']).reset_index(drop=True)
    t['target_available'] = t.groupby('Date').sleeve_target_available.transform('all')
    t['ai_benchmark_forward_return'] = t.groupby('Date').subsector_forward_return.transform('mean').where(t.target_available)
    t['subsector_forward_excess'] = (t.subsector_forward_return - t.ai_benchmark_forward_return).where(t.target_available)
    t['y'] = (t.subsector_forward_excess > 0).astype(float).where(t.target_available)
    t['split'] = np.where(t.Date.lt('2024-01-01'), 'training', 'validation')
    t['boundary_purged'] = t.split.eq('training') & t.target_end_session.ge('2024-01-01')
    t['model_eligible'] = t.target_available & ~t.boundary_purged & f.feature_eligible
    t.to_csv(out / 'targets.csv', index=False)
    members = pd.DataFrame(member_rows)
    members.to_csv(out / 'formation_members.csv', index=False)
    excl = f[['Date', 'primary_group', 'feature_eligible', 'feature_missing_fields']].merge(t, on=['Date', 'primary_group'], validate='one_to_one')
    excl['feature_date_incomplete'] = ~excl.feature_eligible
    excl['target_date_incomplete'] = ~excl.target_available
    excl.loc[excl.feature_date_incomplete | excl.target_date_incomplete | excl.boundary_purged].to_csv(out / 'exclusions.csv', index=False)
    counts = []
    for split in ['training', 'validation']:
        mask = t.split.eq(split)
        counts.append({'split': split, 'rows': int(mask.sum()), 'dates': int(t.loc[mask, 'Date'].nunique()),
            'feature_eligible_rows': int((mask & f.feature_eligible).sum()), 'target_available_rows': int((mask & t.target_available).sum()),
            'boundary_purged_rows': int((mask & t.boundary_purged).sum()), 'model_eligible_rows': int((mask & t.model_eligible).sum())})
    checks = {'full_8785_keys': len(f) == len(t) == 8785, '49_candidates_retained': stock.ric.nunique() == 49,
        'feature_target_keys_equal': f[['Date', 'primary_group']].equals(t[['Date', 'primary_group']]),
        'macro_lag_prior': bool((f.reference_spy_session < f.Date).all()), 'no_test_dates': bool(t.Date.lt('2026-01-01').all()),
        'no_test_label_end': bool(t.target_end_session.dropna().lt('2026-01-01').all()),
        'purged_training_end': bool(t.loc[t.model_eligible & t.split.eq('training'), 'target_end_session'].lt('2024-01-01').all()),
        'seven_group_evaluation': bool(t.loc[t.target_available].groupby('Date').size().eq(7).all())}
    summary = {'status': 'complete_diagnostic_panel' if all(checks.values()) else 'failed_checks',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'inputs': {str(p): sha(p) for p in [STOCK, FACT, MACROFILE]}, 'script_sha256': sha(__file__),
        'counts': counts, 'source_stock_rows': len(stock), 'source_instruments': stock.ric.nunique(), 'feature_rows': len(f),
        'formation_member_rows': len(members), 'formation_excluded_observations': len(obs_ex),
        'entry_missing_sleeve_rows': int(t.missing_entry_rics.ne('').sum()), 'exit_missing_sleeve_rows': int(t.missing_exit_rics.ne('').sum()),
        'future_return_missing_sleeve_rows': int(t.missing_future_return_rics.ne('').sum()), 'macro_audit': macro_audit,
        'checks': checks, 'processing': {'deleted_source_rows': 0, 'new_imputation': 'none', 'new_forward_fill': 'none; inherited macro as-of carry is audited',
        'technical_scaling': 'full ex-ante date universe, ddof1', 'volatility_semantics': 'mean individual-member historical volatility, not sleeve volatility',
        'targets': 't+1 close entry t+6 close exit; returns t+2 through t+6; equal7 initial sleeves', 'test_labels_read': False},
        'limitations': ['Static role membership and source company-action/price continuity remain unverified.',
            'Current-vintage FRED data and conservative lag are not historical release-time proof.',
            'Future completeness censors diagnostic labels only; no liquidation or trading-cost ledger is implemented.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=int), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=int), flush=True)
    assert all(checks.values())


if __name__ == '__main__':
    main()

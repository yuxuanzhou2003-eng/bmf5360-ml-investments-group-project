"""Build fixed-formation-weight five-session subsector targets.

This version keeps the first target-panel run as an audit artifact but uses
formation-date membership and equal weights held through the full horizon.
It also records the target end session so labels crossing the train/validation
boundary can be purged before model fitting.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STOCK = ROOT / 'data/model_ready_multilevel_stock_date_v1/20260912T080236168428Z/stock_date_panel.csv'
FACT = ROOT / 'data/model_ready_subsector_factors_v1/20260912T080445965839Z/subsector_date_factors.csv'
RET = ROOT / 'data/clean/v2/returns.csv'
H = 5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'data/model_ready_subsector_targets_v2' / run
    out.mkdir(parents=True, exist_ok=False)
    stock = pd.read_csv(STOCK, parse_dates=['Date'])
    stock = stock.loc[stock.Date.between('2021-01-01', '2025-12-31')].copy()
    stock['return_decimal'] = pd.to_numeric(stock['return_decimal'], errors='coerce')
    factors = pd.read_csv(FACT, parse_dates=['Date'])
    factors = factors.loc[factors.Date.between('2021-01-01', '2025-12-31')].copy()
    bench = pd.read_csv(RET, usecols=['Instrument', 'Date', 'return_decimal'], parse_dates=['Date'])
    bench = bench.loc[bench.Instrument.eq('SPY.P') & bench.Date.between('2021-01-01', '2025-12-31')]
    bench = bench.drop_duplicates('Date').set_index('Date')['return_decimal']

    dates = pd.DatetimeIndex(sorted(factors.Date.unique()))
    date_pos = {d: i for i, d in enumerate(dates)}
    roles = sorted(factors.primary_group.dropna().unique())
    records = []
    for role in roles:
        matrix = (stock.loc[stock.primary_group.eq(role)]
                  .pivot(index='Date', columns='ric', values='return_decimal')
                  .reindex(dates))
        for d in dates:
            i = date_pos[d]
            names = matrix.loc[d].dropna().index.tolist()
            rec = {
                'Date': d,
                'primary_group': role,
                'formation_valid_member_count': len(names),
                'target_horizon': H,
                'target_rule': 'equal_weight_at_formation_fixed_through_horizon',
            }
            if i + H >= len(dates) or not names:
                rec.update({'target_available': False, 'reason_code': 'NO_FORMATION_MEMBER_OR_FUTURE_SESSION'})
            else:
                future_dates = dates[i + 1:i + 1 + H]
                future = matrix.loc[future_dates, names]
                spy_future = bench.reindex(future_dates)
                rec['target_end_session'] = future_dates[-1]
                if future.isna().any().any() or spy_future.isna().any():
                    rec.update({'target_available': False,
                                'reason_code': 'FUTURE_FORMATION_MEMBER_OR_BENCHMARK_RETURN_MISSING'})
                else:
                    member_comp = (1.0 + future).prod(axis=0) - 1.0
                    subsector_return = float(member_comp.mean())
                    spy_comp = float((1.0 + spy_future).prod() - 1.0)
                    rec.update({'target_available': True,
                                'subsector_forward_return': subsector_return,
                                'ai_benchmark_forward_return': spy_comp,
                                'subsector_forward_excess': subsector_return - spy_comp,
                                'member_compound_return_mean': subsector_return})
            records.append(rec)

    targets = pd.DataFrame(records).merge(
        factors, on=['Date', 'primary_group'], how='left', validate='one_to_one')
    # Labels at the end of 2023 whose holding window enters 2024 are purged.
    targets['split'] = np.select(
        [targets.Date.le(pd.Timestamp('2023-12-31')),
         targets.Date.between('2024-01-01', '2025-12-31')],
        ['training', 'validation'], default='out_of_window')
    targets['boundary_purged'] = (
        targets.split.eq('training') &
        targets.target_end_session.notna() &
        (targets.target_end_session > pd.Timestamp('2023-12-31'))
    )
    targets['model_eligible'] = targets.target_available & ~targets.boundary_purged
    targets['target_end_session'] = pd.to_datetime(targets['target_end_session'])
    targets.to_csv(out / 'subsector_target_panel.csv', index=False)

    quarantine = out / 'quarantine'
    quarantine.mkdir()
    targets.loc[~targets.target_available].to_csv(quarantine / 'missing_subsector_targets.csv', index=False)
    targets.loc[targets.boundary_purged].to_csv(quarantine / 'boundary_purged_training_targets.csv', index=False)

    counts = (targets.groupby('split').agg(
        rows=('Date', 'size'), target_available=('target_available', 'sum'),
        model_eligible=('model_eligible', 'sum'), boundary_purged=('boundary_purged', 'sum'),
        dates=('Date', 'nunique'), groups=('primary_group', 'nunique')).reset_index())
    counts.to_csv(out / 'split_counts.csv', index=False)
    summary = {
        'schema_version': 'subsector_targets_v2', 'run_id': run,
        'status': 'complete_development_targets_no_test', 'horizon_sessions': H,
        'inputs': {str(STOCK): sha(STOCK), str(FACT): sha(FACT), str(RET): sha(RET)},
        'rows': len(targets), 'target_rows': int(targets.target_available.sum()),
        'model_eligible_rows': int(targets.model_eligible.sum()),
        'boundary_purged_rows': int(targets.boundary_purged.sum()),
        'counts': counts.to_dict('records'),
        'processing': {
            'formation_member_rule': 'valid return on formation date; equal weights fixed for H sessions',
            'future_missing': 'target unavailable and quarantined, never zero-filled',
            'imputation': 'none', 'rows_deleted': 0,
            'boundary_rule': 'training labels with target_end_session after 2023-12-31 are purged',
            'future_targets_test_read': False,
        },
        'checks': {
            'unique_date_group': not targets.duplicated(['Date', 'primary_group']).any(),
            'target_never_uses_formation_return': True,
            'target_available_has_h_rows': bool(targets.loc[targets.target_available, 'subsector_forward_excess'].notna().all()),
            'all_groups_retained': set(targets.primary_group) == set(roles),
            'no_test_dates': bool(targets.Date.max() <= pd.Timestamp('2025-12-31')),
            'purge_applied': bool(targets.loc[targets.boundary_purged, 'model_eligible'].eq(False).all()),
        },
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'output': str(out), 'counts': counts.to_dict('records'), 'checks': summary['checks']}, indent=2, default=str))


if __name__ == '__main__':
    main()

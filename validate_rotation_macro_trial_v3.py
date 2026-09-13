"""Independent panel checks, including sampled label reconstruction from stocks."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STOCK = ROOT / 'data/model_ready_multilevel_stock_date_v1/20260912T080236168428Z/stock_date_panel.csv'
FACT = ROOT / 'data/model_ready_subsector_factors_v1/20260912T080445965839Z/subsector_date_factors.csv'
MACRO = ROOT / 'data/clean/daily_macro_v1/20260910T062700000000Z/macro_features.csv'
TECH = ['subsector_return_1', 'momentum_5', 'momentum_20', 'momentum_60',
        'volatility_20_ann', 'breadth_positive_20', 'dispersion_1']
M = ['vix_level', 'vix_change_5d', 'dgs2_level', 'dgs2_change_20d',
     'dgs10_change_20d', 'term_spread_10y_2y', 'broad_dollar_change_20d', 'dgs3mo_level']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    feat = pd.read_csv(args.panel_dir / 'features.csv', parse_dates=['Date'])
    target = pd.read_csv(args.panel_dir / 'targets.csv', parse_dates=['Date', 'entry_session', 'target_end_session'])
    source = pd.read_csv(FACT, parse_dates=['Date']).set_index(['Date', 'primary_group']).sort_index()
    f = feat.set_index(['Date', 'primary_group']).sort_index()
    checks = {}
    checks['features_full_source_keys'] = f.index.equals(source.index)
    checks['features_unique'] = not f.index.duplicated().any()
    checks['source_seven_groups_daily'] = bool(source.reset_index().groupby('Date').primary_group.nunique().eq(7).all())
    checks['features_development_only'] = bool(feat.Date.between('2021-01-01', '2025-12-31').all())
    for col in TECH:
        expect = (source[col] - source.groupby(level=0)[col].transform('mean')) / source.groupby(level=0)[col].transform('std').replace(0, np.nan)
        checks['raw_' + col] = bool(np.allclose(f[col], source[col], equal_nan=True))
        checks['ex_ante_z_' + col] = bool(np.allclose(f[col + '_csz'], expect, equal_nan=True, atol=1e-10))
    macro = pd.read_csv(MACRO, usecols=['spy_session', 'reference_spy_session'] + M, parse_dates=['spy_session', 'reference_spy_session'])
    macro = macro.loc[macro.spy_session.between('2021-01-01', '2025-12-31')].set_index('spy_session')
    for col in M:
        checks['macro_' + col] = bool(np.allclose(feat[col], feat.Date.map(macro[col]), equal_nan=True))
    checks['macro_lag_strict'] = bool((macro.reference_spy_session < macro.index).all())
    eligible = np.isfinite(feat[[c + '_csz' for c in TECH] + M]).all(axis=1)
    eligible = eligible.groupby(feat.Date).transform('all')
    checks['eligibility_only_known_features'] = bool((eligible.to_numpy() == feat.feature_eligible.to_numpy()).all())
    t = target.set_index(['Date', 'primary_group']).sort_index()
    checks['targets_full_keys'] = t.index.equals(f.index)
    checks['targets_development_only'] = bool(target.Date.between('2021-01-01', '2025-12-31').all())
    checks['split_by_date'] = bool((target['split'] == np.where(target.Date.le(pd.Timestamp('2023-12-31')), 'training', 'validation')).all())
    expected_purge = target.Date.le(pd.Timestamp('2023-12-31')) & target.target_end_session.ge(pd.Timestamp('2024-01-01'))
    checks['purge_matches_exit'] = bool((expected_purge == target.boundary_purged).all())
    usable = target.loc[target.target_available].copy()
    checks['nonempty_evaluation_targets'] = not usable.empty
    checks['all7_target_dates'] = bool(usable.groupby('Date').primary_group.nunique().eq(7).all())
    expected_bench = usable.groupby('Date').subsector_forward_return.transform('mean')
    checks['benchmark_equal7_not_spy'] = bool(np.allclose(usable.ai_benchmark_forward_return, expected_bench))
    checks['excess_difference'] = bool(np.allclose(usable.subsector_forward_excess, usable.subsector_forward_return - expected_bench))
    checks['y_relative_ai'] = bool((usable.y == (usable.subsector_forward_excess > 0).astype(int)).all())

    stock = pd.read_csv(STOCK, parse_dates=['Date'])
    days = pd.DatetimeIndex(sorted(stock.Date.unique()))
    checks['stock_calendar_equals_factor_calendar'] = days.equals(pd.DatetimeIndex(sorted(feat.Date.unique())))
    checks['targets_no_test_exit'] = bool(target.target_end_session.dropna().lt(pd.Timestamp('2026-01-01')).all())
    # Uniform deterministic sample plus split boundaries and all unavailable in-range dates.
    indices = set(np.linspace(0, len(days) - 7, 32, dtype=int).tolist())
    indices.update(i for i, d in enumerate(days) if pd.Timestamp('2023-12-18') <= d <= pd.Timestamp('2024-01-05'))
    indices.update(days.get_indexer(target.loc[~target.target_available & target.target_end_session.notna(), 'Date'].unique()).tolist())
    rows = []
    for group, g in stock.groupby('primary_group'):
        r = g.pivot(index='Date', columns='ric', values='return_decimal').reindex(days)
        price = g.pivot(index='Date', columns='ric', values='TRDPRC_1').reindex(days)
        for i in sorted(indices):
            if i < 0 or i + 6 >= len(days):
                continue
            good = np.isfinite(r.iloc[i]) & r.iloc[i].ge(-1) & np.isfinite(price.iloc[i]) & price.iloc[i].gt(0)
            names = good.index[good].tolist()
            observed = t.loc[(days[i], group)]
            rr = r.loc[days[i + 2:i + 7], names]
            pp = price.loc[[days[i + 1], days[i + 6]], names]
            known = bool(names) and bool(np.isfinite(rr).all().all() and rr.ge(-1).all().all() and np.isfinite(pp).all().all() and pp.gt(0).all().all())
            expected = float(((1 + rr).prod(axis=0) - 1).mean()) if known else np.nan
            # Complete-date gating may mask an individually valid sleeve, so compare all evaluated rows.
            comparison = bool(np.isclose(observed.subsector_forward_return, expected, atol=1e-10)) if observed.target_available else True
            rows.append({'Date': days[i], 'primary_group': group, 'formation_members': len(names),
                         'independent_return': expected, 'stored_return': observed.subsector_forward_return,
                         'target_available': bool(observed.target_available), 'sample_match': comparison,
                         'independent_sleeve_available': known,
                         'available_requires_known': not observed.target_available or known,
                         'entry_match': observed.entry_session == days[i + 1],
                         'exit_match': observed.target_end_session == days[i + 6]})
    sample = pd.DataFrame(rows)
    sample['independent_date_available'] = sample.groupby('Date').independent_sleeve_available.transform('all')
    checks['sampled_availability_matches_all7'] = bool((sample.target_available == sample.independent_date_available).all())
    for c in ['sample_match', 'available_requires_known', 'entry_match', 'exit_match']:
        checks[c] = bool(sample[c].all())
    sample.to_csv(args.output_dir / 'sampled_target_reconstruction.csv', index=False)
    summary = {'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks,
               'check_count': len(checks), 'passed_count': sum(checks.values()),
               'sampled_targets': len(sample), 'evaluated_sampled_targets': int(sample.target_available.sum()),
               'scope': 'Full-key/source-feature/macro/purge/benchmark checks; independently sampled stock target reconstruction. No corporate-action, vintage or model-performance certification.',
               'panel': str(args.panel_dir.resolve()), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

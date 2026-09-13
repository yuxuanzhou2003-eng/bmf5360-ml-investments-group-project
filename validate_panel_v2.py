"""Independent persisted-panel checks; no source data or targets are changed."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    panel = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'data/panel_v2'
    f = pd.read_csv(panel / 'features.csv')
    l = pd.read_csv(panel / 'labels.csv').set_index('sample_id')
    d = pd.read_csv(panel / 'diagnostics_ex_post.csv').set_index('sample_id')
    events = pd.read_csv(panel / 'events.csv')
    cfg = json.loads((ROOT / 'panel_v2_config.json').read_text())
    r = pd.read_csv(ROOT / 'data/clean/v2/returns.csv', usecols=['Instrument', 'Date', 'return_decimal'])
    r['Date'] = pd.to_datetime(r.Date)
    wide = r.pivot(index='Date', columns='Instrument', values='return_decimal').sort_index()
    wide = wide.loc[wide[cfg['benchmark']].notna()]
    iv = pd.read_csv(ROOT / 'data/audit/universe_rebuild/membership_intervals.csv')
    iv['start'] = pd.to_datetime(iv.start); iv['end'] = pd.to_datetime(iv.end)
    spells = {ric: list(zip(g.start, g.end)) for ric, g in iv.groupby('ric')}
    checks = {}
    checks['nonempty_features'] = len(f) > 0
    checks['unique_feature_ids'] = not f.sample_id.duplicated().any()
    checks['unique_label_ids'] = not l.index.duplicated().any()
    checks['unique_diagnostic_ids'] = not d.index.duplicated().any()
    checks['same_sample_ids'] = set(f.sample_id) == set(l.index) == set(d.index)
    forbidden = {'label_complete', 'forward_benchmark_excess', 'receiver_forward_return',
                 'benchmark_forward_return', 'ex_post_own_announcement_overlap'}
    checks['no_target_or_ex_post_fields_in_features'] = not bool(forbidden & set(f.columns))
    if not checks['same_sample_ids'] or not checks['unique_label_ids']:
        raise ValueError('Cannot align panel IDs safely')
    l = l.loc[f.sample_id].reset_index()
    day = pd.to_datetime(f.announcement).dt.normalize()
    snap = pd.to_datetime(f.snapshot)
    graph = pd.to_datetime(f.graph_last_date)
    checks['strict_prior_snapshot'] = bool((snap < day).all())
    checks['snapshot_age_limit'] = bool(((day - snap).dt.days <= cfg['max_snapshot_age_days']).all())
    checks['strict_prior_graph'] = bool((graph < day).all())
    checks['no_self_edges'] = bool((f.source != f.receiver).all())
    checks['source_membership_exact_day'] = all(any(s <= dt <= e for s,e in spells.get(ric,[])) for ric,dt in zip(f.source,day))
    checks['receiver_membership_exact_day'] = all(any(s <= dt <= e for s,e in spells.get(ric,[])) for ric,dt in zip(f.receiver,day))
    checks['edge_threshold'] = bool((f.residual_correlation.abs() >= cfg['neighbor_selection']['min_abs_correlation'] - 1e-12).all())
    counts = f.groupby(['source','announcement','period_end']).size()
    checks['max_neighbor_count'] = bool((counts <= cfg['neighbor_selection']['max_neighbors']).all())
    checks['all_selected_edges_preserved'] = int(pd.to_numeric(events.get('neighbor_count', pd.Series(dtype=float)),errors='coerce').sum()) == len(f)
    entry = pd.to_datetime(l.entry_close); expiry = pd.to_datetime(l.exit_close)
    positions = wide.index.get_indexer(entry)
    ends = wide.index.get_indexer(expiry)
    h = cfg['forward_sessions']
    checks['label_window_sessions'] = bool(((positions >= 0) & (ends == positions+h)).all())
    if not checks['label_window_sessions']:
        raise ValueError('Invalid target session indices')
    future = positions[:,None] + np.arange(1,h+1)
    receiver_col = wide.columns.get_indexer(f.receiver)
    assert (receiver_col >= 0).all()
    values = wide.to_numpy(dtype=float)[future, receiver_col[:,None]]
    market = wide[cfg['benchmark']].to_numpy()[future]
    complete = np.isfinite(values).all(axis=1)
    expected = np.where(complete, np.prod(1+values,axis=1)-1, np.nan)
    benchmark = np.prod(1+market,axis=1)-1
    checks['label_completeness_recomputed'] = np.array_equal(complete, l.label_complete.to_numpy())
    for name, v in [('receiver_forward_return',expected),('benchmark_forward_return',benchmark),('forward_benchmark_excess',expected-benchmark)]:
        checks[name+'_recomputed'] = bool(np.allclose(l[name],v,equal_nan=True,rtol=1e-10,atol=1e-12))
    # Independent least-squares beta for each security, then correlation only
    # over mutually observed residuals. This does not call the panel builder.
    tested = 0
    correlation_ok = True
    for row in f.sample(min(40,len(f)), random_state=5360).itertuples():
        hist = wide.loc[wide.index < pd.Timestamp(row.announcement).normalize()].tail(cfg['graph_lookback_sessions'])
        residuals = []
        for ric in [row.source,row.receiver]:
            pair = hist[[ric,cfg['benchmark']]].dropna()
            x = np.column_stack([np.ones(len(pair)), pair[cfg['benchmark']]])
            coefficient = np.linalg.lstsq(x, pair[ric], rcond=None)[0]
            residuals.append(pd.Series(pair[ric].to_numpy()-x@coefficient,index=pair.index))
        pair = pd.concat(residuals,axis=1).dropna()
        correlation_ok &= len(pair) >= cfg['graph_min_observations'] and np.isclose(pair.iloc[:,0].corr(pair.iloc[:,1]),row.residual_correlation,atol=1e-9)
        tested += 1
    checks['sampled_residual_correlations_independent_ols'] = bool(correlation_ok)
    result = {'checked_at_utc':datetime.now(timezone.utc).isoformat(),'rows':len(f),'complete_labels':int(complete.sum()),
              'correlations_independently_checked':tested,'passed':sum(bool(v) for v in checks.values()),'total':len(checks),
              'failed':[k for k,v in checks.items() if not v],'checks':{k:bool(v) for k,v in checks.items()},
              'scope':'Engineering calculations only; does not verify vendor timezone, EPS vintages, or executable trading prices.',
              'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [panel/'features.csv',panel/'labels.csv',panel/'events.csv',panel/'diagnostics_ex_post.csv']}}
    out = ROOT / 'data/audit/panel_v2' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','input_sha256']},indent=2))
    print(str(out))
    if result['failed']:
        raise SystemExit('Panel verification failed')


if __name__ == '__main__':
    main()

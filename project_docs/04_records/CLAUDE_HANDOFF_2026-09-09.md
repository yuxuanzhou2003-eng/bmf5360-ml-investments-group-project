# BMF5360 project handoff — 2026-09-09

## 1. User intent and working rules

This is a master's-level BMF5360 Machine Learning in Investments group project. The current research idea is an earnings-information-diffusion fund: when company A announces an earnings surprise, use a pre-event relation graph to predict the subsequent returns of economically/statistically related receiver companies B/C/D, then turn the predictions into a tradable cross-sectional strategy.

The user wants the work to be rigorous rather than merely “fancy.” Weak or failed results must be reported honestly. Every data action must be perceptible and logged, especially:

- delisting treatment;
- stock splits and other company actions;
- missing-value handling and model imputation;
- row removal, quarantine, eligibility masks and boundary purges;
- any revised target or entry timing.

Do not silently delete, replace, backfill, winsorize or clip data. Preserve master rows and express exclusions through explicit flags/reason codes whenever possible. Update `DATA_PROCESSING_LOG.md` and `AI_USE_LOG.md` after material work.

The holdout remains sealed:

- training: 2015-01-01 to 2020-12-31 announcement dates;
- validation: 2021-01-01 to 2022-12-31;
- test: 2023-01-01 to 2026-06-30.

Do **not** calculate, inspect or report test targets, predictions, metrics, charts or strategy performance until the development specification is frozen. Reading a source file that contains later market history for mechanical calendar construction is not permission to score test observations. Any target horizon must be purged if its exit crosses into the next split.

## 2. Environment and repository state

- Workspace: `D:\3 Study 学习资料\2E 金融研二(上)资料\BMF5360 Machine Learning in Investments\Group project_2.0`
- Python: `.venv\Scripts\python.exe`
- ML dependencies are pinned in `requirements-ml.txt`.
- `scikit-learn==1.9.0` was installed into `.venv`; system Python was not changed.
- LSEG runs locally and credentials are in the existing environment. Never print `.env` or secrets.
- This directory was checked earlier and is not currently a Git repository. Do not assume commits or branches exist.
- Preserve MCP, plugin, provider and credential configuration.

## 3. Frozen, validated data lineage

### Clean v3 market/fundamental data

Current clean run:

`data/clean/v3/20260909T012417705069Z/`

The clean daily price file is:

`data/clean/v3/20260909T012417705069Z/prices.csv`

It contains adjusted LSEG fields including `TRDPRC_1`, `OPEN_PRC`, volume, bid/ask, turnover, daily membership and `price_adjustments`. The observed adjustment string includes `exchangeCorrection,manualCorrection,CCH,CRE,RPO,RTS`. Do not infer that this is a dividend total-return series; open-price experiments must explicitly state they are price-return experiments.

Independent clean-data audit:

`data/audit/validate_clean_v3/20260909T015809930464Z/validation.json`

### Panel v3

Builder and configuration:

- `build_panel_v3.py`
- `panel_v3_config.json`

Frozen panel:

`data/panel_v3/20260909T035245922341Z/`

Important files:

- `events.csv`
- `features.csv`
- `labels.csv`
- `execution_inputs.csv`
- `event_flow.csv`
- `entry_missing_rows.csv`
- `summary.json`

Key definitions:

- event key: `(source, announcement, period_end)`;
- one edge/sample is source announcement → receiver stock;
- formal entry session is the first SPY market session strictly after the announcement calendar day;
- graph, surprise and liquidity/model features use information from `t0-1` or earlier;
- entry-day execution fields are isolated in `execution_inputs.csv` and must not be model-selection features;
- 5-session label is close-to-close from entry close to the fifth following session;
- no physical row deletion or filling was performed.

Counts:

- 28,995 events;
- 107,532 edges;
- 107,499 complete 5-session labels;
- 33 incomplete labels;
- 15 unique samples have at least one missing entry execution field;
- exact entry missing counts: 9 price rows, 12 close values, 9 volume values, 15 quotes, 9 dollar-volume values;
- 7 edges whose receiver was no longer an S&P 500 member at entry remain in the master data.

Independent panel validator:

- script: `validate_panel_v3.py`
- audit: `data/audit/panel_v3/20260909T041022558174Z/validation.json`
- result: 31/31 checks passed, including independent price/label lookup recomputation.

### Model-ready v1

Builder/configuration:

- `build_model_ready_v1.py`
- `model_ready_v1_config.json`

Frozen model-ready run:

`data/model_ready_v1/20260909T064151673764Z/`

It retains all 107,532 master rows and creates 45 backward-looking candidate features. It does not pre-impute, winsorize, clip or physically delete observations.

Split counts before supervised eligibility:

- training: 54,421 edges / 11,626 events;
- validation: 19,415 / 3,961;
- test: 33,696 / 6,936.

The event-level boundary purge correctly identifies three events / 14 edges:

- `PAYX.OQ`, announcement 2020-12-23: 5 edges;
- `KMX.N`, announcement 2022-12-22: 5 edges;
- `PAYX.OQ`, announcement 2022-12-22: 4 edges.

Eligibility issues remain explicit:

- 17 entry-trade-ineligible edges;
- 33 incomplete labels;
- 213 rows with at least one model feature missing;
- 207 are missing standardized surprise and derived interactions;
- 4 are missing a receiver return/rank/lagged-liquidity group;
- 2 are missing lagged spread only.

Final `supervised_model_eligible` counts:

- training: 54,241;
- validation: 19,392;
- test: 33,637.

Independent model-ready validator:

- script: `validate_model_ready_v1.py`
- final audit: `data/audit/model_ready_v1/20260909T064816383676Z/validation.json`
- result: 31/31 passed, including sampled independent recomputation of 5,632 return features and 1,536 liquidity features.

An earlier warning run and a false-failure validator run remain preserved and documented; do not present them as final:

- `data/audit/model_ready_v1/20260909T064041853892Z/`
- `data/audit/model_ready_v1/20260909T064653604512Z/`

## 4. Baseline model result: technically valid, economically weak

Scripts/configuration:

- `run_baseline_models_v1.py`
- `baseline_model_v1_config.json`
- `validate_baseline_models_v1.py`

Final baseline output:

`data/model_runs/baseline_v1/20260909T070830647279Z/`

Independent audit:

`data/audit/baseline_v1/20260909T071048556782Z/validation.json`

Audit result: 13/13 passed. Saved models were reloaded, predictions/metrics and CV selection were reproduced, and no test rows were scored.

Models used event-equal sample weights and training-only forward CV. Selected parameters were:

- receiver Ridge alpha 0.01;
- network Ridge alpha 100;
- HistGradientBoosting: learning rate 0.03, 7 leaves, L2 10, 200 iterations, minimum leaf 50.

Validation results for 2021–2022:

| Model | Weighted MSE | Weighted R² | Pooled Spearman | Mean within-event Spearman |
|---|---:|---:|---:|---:|
| Training weighted mean | 0.0019788 | -0.0071 | n/a | n/a |
| Calibrated simple network | 0.0019789 | -0.0072 | 0.0115 | 0.0100 |
| Receiver-only Ridge | 0.0019624 | 0.0012 | 0.0960 | 0.0159 |
| Network Ridge | 0.0019662 | -0.0007 | 0.0743 | 0.0172 |
| Network HGB | 0.0019915 | -0.0136 | 0.0209 | -0.0017 |

The predeclared champion rule selected network Ridge by mean within-event Spearman, but its improvement over receiver-only Ridge is only about 0.0013 and its MSE/R² are worse. This is not credible evidence of network alpha.

Important retrospective diagnosis from a read-only Luna agent:

- validation target mean is about 0.00443 versus about 0.00060 in training, a roughly 37–38 bp level shift;
- predictions stay close to the training mean and miss this shift;
- only 3.55% of validation observations have `|target| > 10%`, but they contribute roughly 37–38% of weighted MSE;
- receiver volatility relates more to target magnitude than signed direction;
- receiver Ridge's pooled IC of 0.096 is largely cross-event; true within-event IC is only 0.0159;
- receiver Ridge within-event IC is about 0.0414 in 2021 and -0.0097 in 2022;
- network Ridge within-event IC is about 0.0359 in 2021 and -0.0016 in 2022;
- missing values are too rare to explain poor validation quality;
- the network fields receive small fitted coefficients relative to receiver momentum/volatility.

## 5. Exploratory diagnostics completed after the weak baseline

These are development-stage diagnostics with multiple specifications. They are not confirmatory evidence and must not be cherry-picked.

### Horizon and confidence filters

Script:

`diagnose_signal_horizons_v1.py`

Latest corrected output:

`data/analysis/validation_signal_diagnostics_v1/20260909T073556574431Z/`

The earlier run `20260909T072158563895Z` did not apply horizon-specific boundary purging for the 10-session target and is superseded. The corrected run derives an exit date for every horizon and masks cross-boundary targets. Purged row counts are:

- horizons 1/2/3/5: 0;
- horizon 10: 77 rows in total across training and validation.

All recorded checks passed, including unchanged inputs and no test rows.

Main finding: the raw `network_signal = standardized_surprise × residual_correlation` is near zero or unstable across horizons. The visually strongest validation subgroup is high surprise (training q75) + positive correlation at 10 sessions, with validation mean within-event IC about 0.02845 and top-minus-bottom target about 45.7 bp. The same specification is almost flat in training, so it must not be selected as a discovered alpha rule.

### Directional lead-lag graph

Script:

`diagnose_lead_lag_v1.py`

Latest corrected output:

`data/analysis/lead_lag_diagnostics_v1/20260909T073416575653Z/`

The script computes market-adjusted residual histories using only dates strictly before each announcement day, then estimates source residual at day t → receiver residual at t+1, the reverse direction and their asymmetry. The temporal constraint is programmatically checked.

The same horizon-specific purge correction applies: 0 rows at 5 sessions and 77 total rows at 10 sessions. All checks passed.

Main result:

- 5-day lead signal: training pooled/event IC about -0.0052/-0.0017; validation -0.0064/+0.0078;
- 10-day lead signal after purge: training +0.0059/-0.0008; validation -0.0179/+0.0233.

The sign and ranking definition disagree between training and validation. This does not rescue the current graph thesis.

## 6. Newly written but not yet run: next-session-open diagnostic

Script:

`diagnose_next_open_targets_v1.py`

Status at handoff: **written immediately before this handoff, not yet compiled or executed, and therefore not validated**. Review it before running.

Purpose:

- test whether entering only at the next-session close discarded the main information-diffusion window;
- define theoretical entry at the first market-session open strictly after the announcement calendar day;
- calculate receiver open-to-exit-close price return minus SPY open-to-exit-close price return;
- examine horizons 0, 1, 2, 3, 5 and 10 sessions;
- preserve all development rows and record missing entry open, missing/nonpositive exit prices, boundary masks and target availability;
- fit only on training and evaluate only validation using the fixed model families and Ridge alphas inherited from the existing close-target baseline;
- write a visible median-imputation log without changing stored source features.

Expected output root:

`data/analysis/next_open_target_diagnostics_v1/`

Before running, inspect these points:

1. Compile: `.venv\Scripts\python.exe -m py_compile diagnose_next_open_targets_v1.py`.
2. Verify MultiIndex `reindex` works with `NaT` exit keys and does not reorder rows.
3. Confirm the chosen horizon convention: horizon 0 is entry-day open → entry-day close; horizon 1 is entry-day open → next-session close, etc.
4. Confirm every horizon independently masks an exit on/after 2021-01-01 for training or on/after 2023-01-01 for validation.
5. Add an independent recomputation/check before treating its outputs as reliable.
6. State clearly that official-open execution is theoretical, ignores auction/slippage, and uses price return rather than dividend total return.
7. If the script fails, preserve the failure in the processing log. Do not quietly patch and omit the incident.

## 7. Recommended next decision sequence

1. Review, compile and run `diagnose_next_open_targets_v1.py` on training/validation only.
2. Independently validate its exact price lookups, target arithmetic, row ordering, split boundaries and output hashes.
3. Append both corrected horizon runs, the superseded pre-purge runs, the lead-lag result and the open-target run to:
   - `DATA_PROCESSING_LOG.md`
   - `DATASET_STATUS.md`
   - `RESEARCH_PROTOCOL.md`
   - `AI_USE_LOG.md`
   - `DATA_DICTIONARY.md` if new persisted fields require definitions.
4. Decide whether the thesis survives. Require training/validation directional consistency and economically interpretable within-event ranking. Do not promote a subgroup because it looks good only in validation.
5. If next-open targets remain weak, treat the contemporaneous residual-correlation graph as rejected in its current form. The clean data and pipeline remain reusable.
6. The strongest disciplined pivot is a two-stage model:
   - event-level component predicting the average receiver response;
   - within-event component predicting which receiver outperforms its event peers.
   This directly addresses the finding that pooled IC is mostly cross-event while the investment action requires within-event selection.
7. Other bounded experiments, all chosen with training-only CV, are recency weighting, Huber/volatility-scaled targets, predeclared correlation-sign interactions and lagged-liquidity/cost stress tests. Keep the experiment count small and log every one.
8. Freeze the final specification and strategy rules before touching the test set.

## 8. Research interpretation to carry forward

The data engineering is substantially ready and independently audited. The current alpha hypothesis is not. This distinction matters in the report: a high-quality master's project can credibly document why a plausible network signal fails, then make a disciplined economic pivot using the same point-in-time infrastructure. It should not hide instability behind a complex model or a validation-only subgroup.

The current evidence suggests:

- receiver characteristics contain some predictive structure;
- contemporaneous residual correlation adds little incremental value;
- pooled prediction quality overstates the relevant within-event ranking ability;
- regime and target-mean shifts matter;
- tail returns dominate regression loss;
- next-open timing is the last direct test of whether the current target starts too late.

## 9. Existing project documentation

Read these before changing data or methodology:

- `DATA_PROCESSING_LOG.md`
- `DATASET_STATUS.md`
- `RESEARCH_PROTOCOL.md`
- `DATA_DICTIONARY.md`
- `AI_USE_LOG.md`
- `AGENTS.md`

The documentation is current through the validated baseline. The post-baseline diagnostics and the newly discovered/corrected 10-session boundary issue still need to be appended.

## 10. Non-negotiable cautions

- Do not open the 2023–2026 test results.
- Do not use entry-day close, spread, volume or dollar volume as predictive features for a trade decided before that information is available.
- Do not treat `price_adjustments` as proof of dividend total-return coverage.
- Do not infer delisting from missing forward returns; use an explicit reason such as unknown missing-return cause until separately verified.
- Do not use future index membership to form historical edges.
- Do not delete invalid/missing rows from the master files; store flags and counts.
- Do not report network Ridge as a convincing champion merely because the predeclared rule mechanically chose it.
- Do not reuse the superseded pre-purge 10-day diagnostics.

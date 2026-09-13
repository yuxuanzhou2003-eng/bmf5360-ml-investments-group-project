# AI Portfolio Backtest Pre-registration Specification v1

**Status:** pre-registered design; no portfolio backtest has been executed under this specification  
**Freeze date:** 2026-09-10 (Asia/Singapore; all run identifiers remain UTC identifiers)  
**Primary strategy:** long-only AI-industry stock sleeve with a SPY beta hedge  
**Secondary registered strategy:** market-neutral AI top/bottom portfolio  
**Signal horizon:** 21 SPY sessions  
**Rebalance interval:** every 21 SPY sessions

This document fixes the portfolio mapping before any test-period target, prediction, or performance result is opened. It is a design contract for a later run, not a report of portfolio results. Any material change to a threshold, schedule, universe, model, cost rule, risk target, or missing-data rule requires a new version and a new pre-registration entry.

## 1. Fixed inputs and provenance

The frozen primary prediction model is the `technical_plus_ai_state/logistic` artifact from:

```text
data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/
models/technical_plus_ai_state_logistic.joblib
```

The model artifact SHA-256 is `5e12d0688b8a93f3905d1794c87e6cac2a654d642d005044cd7670160bbd2783`. Its training/validation run is marked `complete_training_validation_only`; the run produced validation predictions only. The frozen model uses the 21 technical features plus the 21 independently audited incremental AI-industry-state features listed in the run's `feature_sets.json`. The stored imputer and scaler are part of the artifact and are applied exactly as saved.

The upstream model-ready input is `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/`, with the model-run summary reporting 49 literal RICs and `membership_source_mode=exploratory_static_candidate`. The model was trained on 2015–2020 primary H21 anchors and validated on 2021–2022 primary H21 anchors. No refit, rolling update, probability recalibration, or hyperparameter search is allowed during this portfolio mapping.

The point-in-time design inputs are:

| Input | Frozen version or path | SHA-256 / status |
|---|---|---|
| AI pool specification | `AI_POOL_SPEC.md` v1.1 | `7d2cb81904d1102ae5f8f5f6a6729deda4ba9c31b69a9fba33eb416d0acd77e3` |
| Supply-chain taxonomy | `AI_SUPPLY_CHAIN_TAXONOMY_v2.md` v2.0 | `23adb148418e8b6ee229902c744659879e9acf3ff3fe8e516c90fd1d23b19fbc` |
| Candidate list | `AI_POOL_EXPANSION_CANDIDATES_v1.md` | `0467b1e5ac765357b11a17baeee0e10bc46c512e55399c12bfec29e508505e62` |
| Candidate registry | `data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv` | `b165060c7d61be0a3ff31f3b337dda1730f0e154efb1a4ac1b61cf693f038cde` |
| Model configuration | `ai_pool_daily_state_models_v1_1_config.json` | `d0b92845f6297e2fb2ef4cfeaf8c0c4881b7e80ab3511f47c812ee0398de9177` |

The registry contains 49 unique literal RICs in seven primary groups. Nine have `direct_local_pit` role evidence and 40 have `provisional_static_source` evidence. The latter are included in the registered exploratory candidate pool used by the frozen model-ready data, but are never described as complete historical PIT AI membership. The registry's group, evidence-status, `member_from`, `member_to`, and delisting fields are carried into every portfolio audit. The known `JNPR.N^G25` delisting span ending 2025-07-08 is retained as a historical security record.

The test period is 2023-01-01 through 2026-06-30. Its future targets remain sealed until this specification has been approved and the test run is explicitly opened. Writing this specification does not open, parse, aggregate, score, rank, or predict any test target.

## 2. Research object and signal

For an eligible stock (i) at formation session (F_k), the frozen model supplies

\[
p_{i,k}=P(R_{i,(F_k,X_k]}-R_{SPY,(F_k,X_k]}>0),
\]

where the position is entered at the close of formation session (F_k) and (X_k) is the 21st SPY session after (F_k). This matches the frozen model-ready label convention `entry_session=formation_session`. The score is computed before the formation close from features ending on the prior SPY session. It is a probability of a positive 21-session stock excess return. It is not treated as a calibrated probability after 2020, and it is not a return forecast in dollars.

The model's stored preprocessing handles feature-level missing values with the training-only median and missing indicators already embedded in the joblib pipeline. No new imputer, scaler, calibration model, clipping rule, or replacement value is fitted in the portfolio stage.

All-daily inference is required. A prediction is generated for every SPY session with a model-ready row, including sessions that are not rebalance sessions. A daily inference row uses only features whose source reference is strictly before that formation session, as required by the model-ready audit. Daily scores are saved for coverage and diagnostics; positions change only on the fixed 21-session schedule or after a mandatory delisting/identity exit. Applying the frozen model daily is an operational extension of a model trained on non-overlapping H21 anchors; it does not create additional training observations.

For ranking, higher (p_{i,k}) is better. Exact ties are resolved by ascending literal RIC, so the result is deterministic and independent of file order.

## 3. SPY-session calendar and holding convention

SPY is the master calendar. Let (S_0,S_1,ldots) be the ordered sessions in the relevant evaluation split.

The validation split is 2021-01-01 through 2022-12-31 and the test split is 2023-01-01 through 2026-06-30. The first available SPY sessions are used as split anchors (2021-01-04 and 2023-01-03 in the current calendar). For each split independently,

```text
F_k = S_(21k)                         formation and close execution session
X_k = the 21st SPY session after F_k  exit/rebalance execution at close
```

Only formations for which (F_k) and (X_k) are inside the same split are eligible for a complete 21-session holding block. The last incomplete block is omitted from that split's performance sample. Positions are flat at the split boundary; a validation holding is never carried into the test return series. If a formation cannot form a portfolio, the calendar is not re-anchored to obtain a more convenient date; the next scheduled (F_{k+1}) remains 21 SPY sessions later.

The score at (F_k) is computed before the formation close using the row ending at the prior SPY session. A market-on-close order is submitted from that frozen signal and executes at the close of (F_k), using the close and quote available for execution-cost measurement. Holdings are effective after the (F_k) close and earn returns over ((F_k,X_k]). At (X_k=F_{k+1}), the old block is marked through the close and the next block's net trade is executed at that same close. There are no intraday trades and no daily signal-driven turnover.

## 4. Universe, membership, and group identity

The screening universe is exactly the 49 literal RICs in the frozen candidate registry. No name, ticker, current company name, ETF holding, return, volatility, liquidity value, or model output can add a security or establish membership.

At (F_k), a candidate passes the registry membership gate only when:

1. its literal RIC is present in the registry and the model-ready membership snapshot;
2. `member_from <= F_k <= member_to` and the snapshot marks it `membership_active`; and
3. its registry `candidate_status` is retained with its original `pit_evidence_status` and reason code.

The primary supply-chain group is the registry `primary_group`, one of `gpu_accelerator`, `ai_semiconductor`, `memory_hbm_storage`, `server_network`, `cloud_software`, `data_center_power_cooling`, or `robotics_autonomy`. Secondary group labels are retained for diagnostics but do not double count a security for caps. Group identity never changes because of a later business description, return, or prediction.

Static-source candidates are not silently upgraded to historical PIT members. Their inclusion is explicitly the registered exploratory-static pool used to train the frozen model. A strict `direct_local_pit`-only analysis would be a separate universe/model specification and is not substituted after results are observed.

Delisting, membership expiry, missing data, and a failed execution gate are row-level status reasons. They do not physically delete the RIC from the candidate registry or historical audit tables.

## 5. Pre-trade eligibility

All pre-trade eligibility inputs are measured using observations available through (F_k-1), the SPY session immediately before formation. No future entry-day close, volume, quote, or return is used to decide the signal or rank.

A candidate must satisfy every item below to enter the eligible ranking set:

| Gate | Fixed rule |
|---|---|
| Active membership | Registry/model-ready `membership_active=True` at (F_k) and inside its recorded member span |
| Price | Adjusted close on (F_k-1) is at least USD 5.00 |
| Price history | At least 15 non-missing, positive adjusted closes in the trailing 20 SPY sessions |
| Dollar liquidity | Median trailing-20 dollar volume is at least USD 20 million; the existing source label (`TRNOVR_UNS` or `TRDPRC_1_times_ACVOL_UNS`) is retained |
| Quote coverage | At least 15 non-missing two-sided quotes in the trailing 20 SPY sessions |
| Spread | Median trailing-20 quoted bid/ask spread is no more than 50 basis points |
| Risk history | At least 100 valid stock/SPY return pairs in the trailing 126 sessions, finite `beta_126`, finite `volatility_60_ann`, and finite `idio_vol_126_ann` |
| Prediction | A finite prediction is available from the stored model pipeline at (F_k) |

Blank, invalid, or non-positive values fail the relevant gate; they are not changed to zero. No forward fill, interpolation, winsorization, cross-sectional imputation, or date substitution is used in eligibility. Eligibility flags and reason codes are retained for every registry candidate even when the row is excluded from ranking.

At the actual (F_k) close, a new stock trade also requires a positive adjusted close and a valid two-sided quote or the explicitly registered trailing-spread fallback in Section 10. A failed execution gate prevents that new order; it does not cause a replacement name to be selected using future information. Existing positions are handled by the carry, forced-exit, and delisting rules below.

## 6. Primary portfolio: long-only AI sleeve plus SPY hedge

The primary portfolio is a long-only stock sleeve. All stock positions have non-negative weights. The only permitted short exposure is the SPY hedge used to remove the stock sleeve's estimated market beta. The stock sleeve's maximum gross exposure is 100% of NAV before volatility scaling; leverage is never added when estimated volatility is below target.

### 6.1 Top-quintile selection and confidence rule

Let (N_k) be the number of eligible candidates with a finite prediction at (F_k). A new target can be formed only when (N_k \ge 20). Define

```text
q_k = max(5, ceil(0.20 * N_k))       target number of stock slots
b_k = ceil(0.30 * N_k)              expanded turnover-buffer rank
```

The strict entry set is the top (q_k) names by (p_{i,k}) with (p_{i,k} \ge 0.55). The 0.55 threshold is a fixed five-percentage-point confidence margin above the neutral probability and is not tuned on validation.

At a later rebalance, an existing long position is protected from a signal exit when its current rank is at most (b_k) and (p_{i,k} \ge 0.50). Protected positions are sorted by descending probability and occupy slots first, up to (q_k). Remaining slots are filled by the highest-ranked strict-entry names, subject to group and single-name caps. A protected name is never added solely because it was held previously if it fails the active-membership, price, liquidity, or risk gate.

If fewer than five positions can be supported after the confidence and group rules, a scheduled signal rebalance is skipped and the existing portfolio is carried subject to mandatory exits. At the first formation of a split, this means the portfolio stays in cash. There is no fallback to momentum, a 0.50 probability, or an unselected name merely to fill slots.

### 6.2 Stock weights and supply-chain caps

The selected long names start with equal stock-sleeve weights. Weights are then reduced and redistributed pro rata among uncapped selected names until all constraints hold:

* each individual stock weight is at most 10% of NAV;
* each primary supply-chain group is at most 30% of NAV on the long stock sleeve;
* total long stock weight is at most 100% of NAV; and
* weights remain non-negative.

Redistribution follows descending model rank and the same deterministic cap order; it never introduces a name outside the eligible strict-entry or protected set. If caps prevent the sleeve from reaching 100%, the residual remains cash. The post-cap weights are recorded before the SPY hedge and before risk scaling.

### 6.3 Beta hedge and volatility target

The stock sleeve beta is

\[
\widehat\beta_{stock,k}=\sum_i w_{i,k}\,\texttt{beta\_126}_{i,k}.
\]

The initial SPY hedge is (w_{SPY,k}=-\widehat\beta_{stock,k}), treating SPY beta as one. The hedge is capped at 100% of NAV in absolute value. If the required hedge is larger, the entire stock sleeve is first multiplied by (1/|\widehat\beta_{stock,k}|), and the hedge is recomputed, so the hedge cap is respected without leverage. A positive hedge is allowed when the stock sleeve's beta is negative.

For volatility scaling, use the sample covariance of clean daily total returns over the trailing 126 SPY sessions, annualized by 252. Estimate one covariance matrix from dates on which every selected stock and SPY has a valid return; require at least 100 common observations. This listwise-complete rule avoids a non-positive-semidefinite matrix produced by mixing different pairwise samples. The ex-ante volatility of the stock plus hedge weights is

\[
\widehat\sigma_k=\sqrt{252\,w_k^\top\Sigma_k w_k}.
\]

The fixed annual volatility target is 10%. Apply

```text
lambda_k = min(1.0, 0.10 / sigma_hat_k)
final risky weights = lambda_k * pre-risk weights
```

The portfolio is never levered up to reach the target. Cash is the residual signed NAV after the stock sleeve and SPY hedge; cash earns zero in the backtest. If the covariance or beta inputs required for a new target are unavailable or non-finite, the signal rebalance is skipped with `RISK_ESTIMATE_UNAVAILABLE`; no risk estimate is imputed. The realized beta after a turnover-limited transition is reported and is not retroactively corrected.

## 7. Secondary registered portfolio: market-neutral top/bottom

The secondary variant is registered before any performance is seen. It is not used to replace the primary strategy.

Use the same (N_k), (q_k), (b_k), eligibility gates, daily frozen predictions, 0.55/0.45 confidence margins, and 30% rank buffer. The strict long set is the top (q_k) names with (p_{i,k}\ge0.55); the strict short set is the bottom (q_k) names with (p_{i,k}\le0.45). Long and short sets are disjoint by construction. Each side needs at least five names to form a new target; otherwise that side is carried and no new side is forced.

The stock legs begin with equal weights and have 50% long gross and 50% short gross. The same iterative allocation rule applies, with these caps:

* absolute single-name weight at most 10% of NAV;
* each primary group at most 20% of NAV on each side; and
* each side gross exposure at most 50% of NAV.

The two stock legs are then beta-hedged with SPY using the same beta estimate, 100% hedge cap, 10% annual volatility target, and no-leverage rule as the primary portfolio. This makes the reported secondary portfolio beta-neutral at the target before a turnover-limited transition while retaining the top-versus-bottom stock signal. Short stock weights incur the borrow charge in Section 10. The market-neutral result is labelled theoretical unless a separate point-in-time borrow-availability file is later supplied.

## 8. Turnover and rebalance buffer

The rank/confidence buffer above is the signal hysteresis rule. A separate fixed execution limit applies to signal-driven trades. Let (w^{old}) include stocks and the SPY hedge, and let (w^{target}) be the post-cap, post-beta, post-volatility target. Define

\[
T_k=\frac{1}{2}\sum_i|w^{target}_{i,k}-w^{old}_{i,k}|.
\]

The maximum signal-driven one-way turnover at one scheduled rebalance is 50% of NAV. If (T_k>0.50), trade the convex fraction (0.50/T_k) of the target transition and carry the remainder:

```text
trade_fraction = min(1.0, 0.50 / T_k)
w_trade = w_old + trade_fraction * (w_target - w_old)
```

The newly constructed target always satisfies the name and group caps. The realised pre-trade portfolio may have moved above a cap because of price drift, and a 50% turnover-limited transition may not return it fully inside the cap in one rebalance. Therefore the backtest reports both target-cap checks and the maximum realised name/group weights; an excess in realised weights is a monitored carry condition rather than evidence that the target allocator violated its rule. Mandatory delisting, identity, or unpriced-security exits may exceed the signal turnover limit; they are separately flagged and reported. No unscheduled signal trade is allowed because a non-rebalance-day prediction looks better.

## 9. Prediction and missing-data rules

The following rules are fixed and apply equally to validation and test:

1. If all required model feature columns exist and the stored pipeline returns a finite probability in ([0,1]), use that probability even when one or more source feature values are missing and the stored training imputer supplies the documented median/indicator transformation.
2. If the model row is absent, has a duplicate key, has a non-finite model output, or lacks a required feature column, record `PREDICTION_UNAVAILABLE` and exclude the name from a new ranking. Do not substitute 0.50, zero, a cross-sectional mean, or a last prediction.
3. A missing prediction for an existing holding does not trigger a signal exit. The current holding is carried to the next scheduled rebalance, unless a separate membership, delisting, corporate-action, price, or risk rule requires an exit.
4. If fewer than 20 finite-prediction eligible names remain, no new target is formed. If fewer than five strict/protected long names (or five names on either secondary side) remain, the affected signal target is skipped or carried as specified above.
5. A missing daily prediction on a non-rebalance day has no effect on an already-held position, but its date and reason are included in inference coverage.
6. A missing portfolio return, SPY return, execution price, or mark is never changed to zero. If an open position cannot be marked or liquidated under the registered fallback rules, the run is marked `INCOMPLETE_UNRESOLVED_NAV`, NAV stops on that date, and no full-period headline performance metric is reported. The unresolved row and reason remain in the audit output.

All exclusions, carries, skipped formations, and unresolved marks remain in audit outputs with row-level reason codes. Missing values are distinct from genuine observed zeros.

## 10. Execution, spread slippage, financing, and borrow

The portfolio is marked with the frozen clean daily total-return series. Returns are not treated as executable prices. Trades occur at the (F_k) close or at a mandatory-exit close.

When a two-sided quote is available, a buy executes at the ask and a sell executes at the bid. Let (s_{i,t}) be the full quoted bid/ask spread in basis points. The spread cost for a trade of absolute weight change (|\Delta w_{i,t}|) is

\[
\text{spread\_cost}_{i,t}=|\Delta w_{i,t}|\frac{s_{i,t}}{2\times10{,}000}.
\]

This charges half the full spread per one-way trade and therefore charges a full quoted spread for a complete round trip. The same rule applies to the SPY hedge. There is no separate commission, tax, or market-impact term in the base case because the available data do not provide a point-in-time order-size impact model. A pre-declared sensitivity adds 5 basis points and 10 basis points per one-way dollar traded; these sensitivities are descriptive and cannot select a strategy.

If the execution-day quote is missing but the formation-time trailing-20 median quote is available from at least 15 valid observations, use that median as the effective spread and flag `SPREAD_FALLBACK_TRAILING_MEDIAN`. This is a cost proxy, not a replacement of a source price. If neither an execution quote nor the registered trailing median exists, a new signal trade is not executed and is recorded as `SPREAD_UNAVAILABLE`. A mandatory delisting exit uses the last valid quote/spread proxy when available; if no defensible exit price exists, the position remains `UNRESOLVED_EXIT_PRICE` rather than being assigned a zero or arbitrary price.

The secondary market-neutral variant charges a fixed base stock-borrow fee of 300 basis points per annum on absolute short notional, prorated by 252 SPY sessions:

\[
\text{borrow\_cost}_t=\frac{0.03}{252}\sum_{i\in short}|w_{i,t}|.
\]

There is no stock-loan rebate and no cash interest. The 300-basis-point assumption is a fixed paper-trading assumption because the frozen inputs do not include point-in-time borrow inventory. If a future implementation supplies borrow availability, a missing or unavailable borrow record makes that short ineligible and is recorded as `BORROW_UNVERIFIED`; it is never imputed. Report 0 and 600 basis-point borrow sensitivities without changing selection or declaring a new primary.

## 11. Delisting, identity, and corporate actions

Delisted RICs remain in the historical universe through their last recorded tradable span. A RIC is not removed merely because it delists. Once `member_to` has passed, no new position is opened. An existing position is forced to close on the last available SPY session on or before `member_to`, subject to the mandatory-exit rule and a recorded `DELISTING_FORCED_EXIT` reason.

If the clean returns table supplies a vendor delisting or terminal return, that return is used. If it does not, the position is liquidated at the last available positive adjusted close and the last valid spread proxy, with `DELISTING_PRICE_FALLBACK` recorded. No fabricated zero return, recovery return, or forward-filled price is introduced. A holding that disappears without either a terminal return or a defensible last price remains an unresolved mark and is excluded from complete-case performance metrics with the limitation reported.

The price and return inputs already contain the documented provider/company-action adjustment. Stock splits and reverse splits are handled by applying the same share conversion to the simulated share quantity and using the adjusted price/return series; no split factor is multiplied into a clean price, return, or EPS field a second time. A mechanical split is not recorded as an economic gain or loss. Dividends are included only through the frozen total-return series and are not added a second time as cash.

An RIC identity change, merger, or successor is not guessed from a name or ticker. Unless an explicit frozen identity mapping exists, the old literal RIC is closed and the new literal RIC is a new candidate subject to its own registry span and eligibility gates. All corporate-action rows, exits, share conversions, and source flags remain auditable.

## 12. Portfolio return construction

For each SPY session, first mark stock and SPY positions using clean total returns, then subtract spread execution costs on that session and any secondary short-borrow fee. The base portfolio return is the change in net marked NAV after those charges. Report separately:

* gross stock-sleeve return before all trading costs;
* stock-sleeve return after spread cost;
* SPY-hedge contribution;
* short-borrow contribution for the secondary variant;
* total cost drag in basis points; and
* final net portfolio return.

The primary and secondary portfolios are evaluated separately by validation and test split. A split is not concatenated across a flat boundary for a headline result. The incomplete final holding block is not converted into a shorter horizon. The 21-session block return is reported as an additional horizon-consistent diagnostic, while the primary performance series is the daily net NAV return.

## 13. Performance, risk, and coverage metrics

All annualization uses 252 SPY sessions. The headline metrics for the primary long-only-plus-hedge portfolio are:

1. cumulative net and gross return, annualized geometric return, annualized volatility, Sharpe ratio using zero cash return, Sortino ratio, maximum drawdown, drawdown duration, and Calmar ratio;
2. daily and 21-session hit rate, mean/median return, downside deviation, 5% historical VaR and expected shortfall, skewness, and kurtosis;
3. beta to SPY, intercept/alpha from a fixed daily OLS diagnostic, tracking error, information ratio, and (R^2);
4. average, median, 95th-percentile, and maximum stock gross exposure, SPY hedge exposure, signed net exposure, realized beta, and realized volatility;
5. one-way and round-trip turnover, annualized turnover, number of rebalances, skipped formations, carried targets, mandatory exits, and incomplete/unresolved marks;
6. spread cost, fallback-spread cost, cost per unit of turnover, and cost sensitivity results; and
7. active-name count, selected-name count, each primary-group weight, the largest single-name weight, rank/confidence coverage, prediction missingness, membership evidence status, delisting coverage, and eligibility reason counts.

For the market-neutral variant, additionally report long-leg and short-leg gross/net returns, long-short spread before and after borrow, gross exposure, short exposure, borrow cost, borrow-availability flags, and the same group/name/beta/volatility diagnostics. For both variants, all results are accompanied by the number of valid daily return observations and the dates omitted for missing marks; no complete-case metric is presented as full-period coverage when it is not.

Model discrimination metrics from the frozen run (ROC-AUC, PR-AUC, Brier, rank IC, and the diagnostic top-minus-bottom spread) are reported separately from portfolio performance. Portfolio results must not be used to relabel the frozen model or to select among the registered feature sets.

## 14. Required audit outputs and acceptance checks

The later backtest must create a new run directory and preserve raw and prior outputs. At minimum it must write:

* `daily_inference.csv`: session, literal RIC, model version, prediction, feature availability, imputer-use flags, and reason codes;
* `eligibility.csv`: all 49 candidates per scheduled formation with membership, price, liquidity, quote, risk, prediction, and exclusion flags;
* `portfolio_targets.csv`: selected names, rank, confidence status, group, pre-cap weight, post-cap weight, hedge, volatility scale, and target reason;
* `trades.csv`: old/new weights, side, execution price, spread source, spread cost, borrow cost, turnover, and mandatory-exit flags;
* `holdings_daily.csv` and `nav_daily.csv`: signed holdings, cash, gross/net exposure, marked returns, cost components, NAV, and missingness; and
* `summary.json`, `coverage_audit.csv`, and a run-specific entry in `DATA_PROCESSING_LOG.md`.

The audit must verify, at minimum, strict prior-session feature cutoffs, 21-session formation spacing, no cross-split holdings, literal-RIC identity, no physical deletion of registry rows, top/bottom rank determinism, confidence and buffer rules, single-name/group caps, beta-hedge arithmetic, volatility scaling, turnover cap, quote/cost arithmetic, borrow fee arithmetic, split handling, delisting reason codes, finite-prediction handling, and absence of test-target reads before the test is opened.

## 15. Fixed choices and change control

The following values are frozen and were chosen as simple capacity and risk conventions rather than validation-optimized values: 21-session signal horizon, 21-session rebalance, top/bottom 20% ranking, minimum 20 eligible names, minimum five positions per active side, 0.55/0.45 confidence margins, 30% rank buffer, 50% one-way signal turnover cap, 10% single-name cap, 30% primary long-group cap, 20% secondary side-group cap, USD 5 price floor, USD 20 million trailing median dollar-volume floor, 15/20 observation minimum, 50-basis-point median-spread ceiling, 100-observation beta/risk history, zero target SPY beta, 10% annual volatility target, 100% SPY hedge cap, half-quoted-spread base trading cost, 300-basis-point annual short-borrow base cost, and no leverage-up.

No threshold may be selected by scanning validation or test outcomes. A new primary choice, a different candidate universe, strict PIT membership treatment, a different execution timing, or any new cost/risk assumption creates `AI_PORTFOLIO_BACKTEST_SPEC_v2.md` and preserves this v1 document and its audit trail.

The realised-weight sentence in Section 8 was clarified after the preliminary validation implementation exposed price-drift and turnover-limited carry. This corrects the earlier mathematical claim that a convex transition automatically preserves caps when the old realised portfolio itself is no longer feasible. It changes no model, threshold, target allocator, trading rule, cost, validation observation, or performance result.

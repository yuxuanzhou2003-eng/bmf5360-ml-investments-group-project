# AI Infrastructure Timing Model — Development Protocol v1

**Status:** superseded for model estimation by v2; retained as a 2015–2022 data-engineering and coverage audit. No model was trained under this protocol.  
**Scope:** development data only (2015-01-01 to 2022-12-31). The 2023-01-01 to 2026-06-30 future-return labels remain sealed.

## 1. Investment question

The strategy is not a cross-sectional stock picker. It asks one market-timing question each trading day:

> Should capital be allocated to a pre-defined AI-infrastructure equity basket for the next five trading sessions, rather than to SPY?

The action is therefore simple and investable: hold the AI-infrastructure basket when the model probability exceeds a pre-specified threshold; otherwise hold SPY. This lets VIX, interest rates, dollar conditions and the basket's own market state contribute to the decision legitimately: they describe the state of the **whole industry**, rather than being incorrectly interpreted as stock-specific factors.

## 2. Research universe and index construction

### 2.1 Source and role map

The user-provided `AI Infrastructure Map` is the economic map for the research universe. Its relevant parts are hyperscalers/data centres and the physical stack below them: accelerators and processors, memory/storage, servers, networking, internal power/cooling and power supply. The local candidate registry already implements a documented, wider version of that map: 49 RICs in seven roles (`gpu_accelerator`, `ai_semiconductor`, `memory_hbm_storage`, `server_network`, `cloud_software`, `data_center_power_cooling`, `robotics_autonomy`). It was selected by role, not by realised return.

### 2.2 Primary basket for this study

The primary research basket will be an **AI Infrastructure Core**. It uses the first six roles above and excludes `robotics_autonomy` from the primary index because robotics is a separate downstream adoption theme rather than direct data-centre infrastructure. A later robustness basket may add it, but it cannot replace the primary basket after validation results are known.

To avoid a handful of mega-cap cloud firms determining every result, the index will use equal sleeve weights across the six roles, then equal weights among the eligible names inside each sleeve. A name's daily return is used only when a valid return exists on that day; missing observations are never converted to zero. Within a sleeve, the available names are reweighted and the count is recorded. A sleeve with no eligible name makes that day's index value missing and is retained with a reason code rather than silently filled.

### 2.3 Historical-universe limitation

The selection is a current, economically reasoned infrastructure map. The existing registry has direct local point-in-time evidence for 9 of 49 names and provisional static issuer-role evidence for 40. It should therefore be described as a **static-role research universe**, not as a historically complete point-in-time AI index. The sealed test will assess the frozen model conditional on this universe; it cannot remove the universe hindsight limitation. Recent entrants and delisted securities remain in the registry and coverage audit; they are not discarded merely because they are unavailable in part of history.

## 3. Target and decision schedule

### 3.1 Primary target

At close on trading day *t*, define the forward five-session excess return:

`R_AI,t→t+5 − R_SPY,t→t+5`.

The binary target is 1 when this quantity is strictly positive and 0 otherwise. The model observes only information available on or before close *t*. A five-session horizon supports a daily process while avoiding the mismatch of treating 21-day overlapping labels as 503 independent validation observations.

### 3.2 Execution and overlap

The model may produce a daily probability, but the primary economic implementation will rebalance every five sessions. Its historical training folds will use a five-session embargo after each validation segment. Reported uncertainty will use calendar-block resampling of at least five sessions. A 21-session target is a **secondary robustness check**, not a tuning target.

## 4. Feature set, frozen before results

The primary feature set is deliberately compact (maximum 15 raw inputs). It has three groups:

| Group | Examples | Purpose |
|---|---|---|
| AI-basket state | 5/20/60-day excess momentum versus SPY; 20/60-day volatility; 60-day drawdown | trend and risk of the traded object |
| AI-basket participation | fractions with positive 5/20/60-day return; cross-sectional dispersion | whether the move is broad or narrow |
| Market regime | SPY 20/60-day momentum and volatility; VIX level/change; 10y–2y term spread; broad-dollar 20-day change | regime information for industry timing |

No individual-stock feature, analyst estimate, future return, event outcome or test-period target may enter this model. Features are calculated using only returns and macro observations dated on or before the formation session. Macro series are joined by their last published value known by the formation session; their release timing and missingness must be audited.

Missing values will **not** be forward-filled, zero-filled or winsorised. The primary models may use a training-fold median imputer with a missingness indicator; each such action will be stated in the run log along with before/after counts. If a feature is structurally unavailable in training, it is excluded with a reason code rather than imputed.

## 5. Model ladder

Logistic regression is the interpretable benchmark, not the claimed final model. The locked comparison ladder is:

1. **Economic comparators:** always hold the AI basket; always hold SPY; and a simple 20-day AI-basket excess-momentum rule.
2. **Regularised Logistic regression:** L2 model with training-only scaling/imputation.
3. **Elastic-net Logistic regression:** the sparse linear alternative; its penalty is selected only in purged training folds.
4. **Shallow gradient-boosted tree model:** maximum depth 2 and a large minimum leaf size, to test limited nonlinearity and interactions without allowing a high-capacity tree search.

The final candidate is selected by pre-specified, dependence-aware validation criteria, not by the highest pooled ROC-AUC. Random forest may be added later as a labelled exploratory diagnostic only; it is not in the primary model-selection competition.

## 6. Validation, selection and reporting

Training is 2015–2020 and validation is 2021–2022. Inside training, tuning uses chronological purged folds and never randomly mixes dates. The test-period labels remain inaccessible until the basket, target, feature list, models, threshold and portfolio rules are frozen.

Primary validation evidence:

- block-bootstrap confidence interval for the change in five-session excess-return performance versus SPY;
- probability calibration and Brier score;
- ROC-AUC reported with a five-session calendar-block interval, never an IID p-value;
- annualised strategy return, volatility, Sharpe, maximum drawdown, turnover and transaction-cost assumptions for the fixed five-session execution rule.

The final chosen model must beat the simple momentum rule on at least two robust validation dimensions, one of which is a cost-aware economic result. If no model clears that bar, the conclusion is that the benchmark is adequate and the complex model is not adopted.

## 7. Planned data-processing audit before first run

The next build will create a separate, versioned basket panel from existing development-period prices and macro files. It will report: candidate and eligible company counts by sleeve and date; each excluded/retained row and reason; index coverage; missingness; any reweighting caused by unavailable names; exact feature definitions; target availability; fold boundaries; and hashes of inputs/scripts/configuration. It will not change raw data or open sealed test targets.

# AI Infrastructure ML Fund — strategy architecture v1

Status: design frozen for the next development stage; no new performance result is implied.

## Mandate

The fund invests in companies linked to AI infrastructure. The candidate map defines the researchable universe; it does not require every company to be held. The fund may hold SPY or cash when the industry signal is weak. The primary objective is risk-adjusted excess return versus SPY after stated turnover and transaction costs.

## Portfolio layers

### Industry gate

An industry-level model estimates whether the AI infrastructure opportunity is attractive over the next registered horizon. It controls total AI exposure (0%, 50%, or 100% in the first implementation) and leaves the remainder in SPY or cash. This layer uses only basket/market state variables and never individual stock rankings.

### Subsector allocation

The candidate taxonomy is refined into accelerator/processor, supporting semiconductor, memory/HBM/storage, server, networking/optical, cloud/AI software, data-centre operator, power/electrical, and cooling. A company can have multiple economic tags, but portfolio weights must use a single pre-registered primary sleeve or a documented split rule so multi-tag names are not double counted.

The subsector model ranks sleeves using lagged relative return, volatility, breadth and validated fundamental/event aggregates. It assigns weights subject to a maximum sleeve weight, a minimum liquidity rule and a turnover cap. A fixed equal-sleeve portfolio remains the comparator.

### Stock selection

Within each selected sleeve, an individual-stock model estimates either forward relative return or the probability of outperforming the sleeve/benchmark. Only features that vary by stock and are available before the decision time enter this model. A long-only implementation holds the top-ranked names with a volatility or risk-budget weight; a long-short version is a later research branch and cannot be mixed into the primary backtest after validation results are seen.

### Event overlay

Events are first evaluated in a standalone event study. Event type, announcement time, pre-event expectation and post-event abnormal return remain separate fields. Only event definitions with a documented first-public timestamp and stable training-period reaction can become model inputs. The overlay can reduce or increase a stock/sleeve weight within registered bounds; it cannot silently replace the base model.

## Model and benchmark ladder

The baseline ladder is: always-SPY; always-AI-universe exposure; equal-sleeve AI portfolio; simple momentum rotation; then one model per layer. The first ML models are regularised Logistic for classification and Ridge for return prediction. A shallow tree model is a confirmation model. All preprocessing, feature selection, thresholds and weight mapping are fitted only in chronological training folds.

## Attribution and risk controls

Every portfolio result is decomposed into industry-gate contribution, subsector allocation contribution, stock-selection contribution and event-overlay contribution. The accounting is additive at the daily return ledger level and reports gross and net results separately.

The initial risk rules are a maximum AI exposure, sleeve caps, stock caps, volatility scaling, beta monitoring against SPY, a turnover limit and explicit spread/impact costs. Missing price or quote data never becomes zero return. If a holding is unavailable, the ledger records the reason and applies the pre-registered execution rule; it does not silently reweight after seeing the realised return.

## Information boundary

Training is 2021–2023, validation is 2024–2025, and 2026 H1 remains sealed. Because 2024–2025 has already been inspected in earlier development work, v1 architecture changes must be described as a new development iteration rather than a clean confirmation. The static AI role map also has a historical point-in-time limitation. These constraints are reported with all results.

## Next implementation gate

Before any new model comparison, reconstruct the stock×date table from locally available history, add the primary and proposed subsector labels, check corporate-action and identifier continuity, and produce a separate event coverage audit. A model may enter the fund backtest only after those tables pass structural and timing checks.

# AI Infrastructure Timing Model — Development Protocol v2 (AI-era sample)

**Status:** active. This replaces v1 for model estimation, while preserving v1 and its outputs unchanged as an engineering audit.  
**Research question:** each trading day, should the portfolio hold a pre-defined AI Infrastructure Core basket or SPY for the next five sessions?

## Why the sample begins in 2021

The fund is about the modern AI-infrastructure investment cycle. Using 2015–2020 to estimate its behaviour would give excessive weight to a period before the current generative-AI capital-expenditure cycle. The revised sample starts 2021-01-01, while retaining the same role-based static basket, sleeve-equal weighting, feature definitions, five-session label and non-imputation rules recorded in v1.

## Frozen split and information boundary

| Role | Formation dates | Use |
|---|---|---|
| Training | 2021-01-01 to 2023-12-31 | model fitting and chronological inner-fold tuning only |
| Validation | 2024-01-01 to 2025-12-31 | one-time model comparison and rule selection |
| Final test | 2026-01-01 to 2026-06-30 | sealed until universe, features, model, threshold and cost rules are frozen |

The v1 sealed period (2023–2026) is not overwritten or retrospectively relabelled. v2 creates a new, separately versioned development-only panel through 2025 and a separate sealed 2026 schedule with no future-return values. It must never merge v1 and v2 targets.

## Model and feature rules

The primary basket has six direct-infrastructure roles: GPU/accelerators, AI semiconductors, memory/storage, server/networking, cloud software and data-centre power/cooling. It excludes downstream robotics from the primary basket. Roles have equal weights; valid names inside a role have equal weights. Missing returns are not zero-filled, forward-filled or otherwise imputed.

The 15 raw inputs are exactly those frozen in `ai_infrastructure_timing_v1_config.json`: basket excess momentum over 5/20/60 sessions; basket 20-session volatility and 60-session drawdown; basket breadth at 5/20/60 sessions and dispersion; SPY momentum/volatility; VIX level/change; 10y–2y term spread; and dollar change. Any additional feature is a later, training-only sandbox proposal and cannot enter v2 after validation is examined.

The first model run is a fixed L2 Logistic benchmark against always-AI, always-SPY and a 20-session basket-excess-momentum rule. Later candidates are elastic-net Logistic and a shallow, constrained gradient-boosted tree. Their parameters are selected only in chronological training folds with a five-session embargo.

## Validation and limitations

Signals may be calculated daily. The primary economic rule trades every five sessions, so reported holding-period returns do not overlap. Validation uncertainty uses calendar-block resampling; IID p-values are not headline evidence. Cost-aware performance is required before a model can be adopted.

The universe remains a current static-role map: 9 of 49 registry names have direct local PIT evidence and 40 have provisional static issuer-role evidence. The model test can measure performance conditional on this fixed map; it does not remove historical-universe hindsight.

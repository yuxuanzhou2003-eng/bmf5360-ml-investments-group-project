# ML4T factor-engineering ideas adapted to the AI Infrastructure timing study

**Purpose:** design note written before the first AI Infrastructure Core model run. It is not evidence that any factor works.

## Source consulted

- Stefan Jansen, [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading), especially the repository's financial-feature-engineering material and its research-design/feature-engineering release notes, consulted 2026-09-12.

ML4T's transferable idea is to construct factors from a financial mechanism, compare them against simple benchmarks, and test their incremental contribution under time-aware validation. It is not an instruction to add every technical indicator in its factor library.

## Mapping to this project

| ML4T-style factor family | AI Infrastructure timing version | Why it may matter | Status |
|---|---|---|---|
| Trend | AI-basket excess return versus SPY over 5, 20 and 60 sessions | captures persistence of the industry trade | primary |
| Short-versus-long horizon | 5/20 and 20/60 momentum spreads | distinguishes acceleration from a mature trend or potential reversal | sandbox candidate |
| Risk / drawdown | basket realised volatility and 60-session drawdown | measures the cost and fragility of holding the basket | primary |
| Breadth / dispersion | fraction of constituents with positive return at 5/20/60 sessions; cross-company one-day dispersion | distinguishes broad infrastructure participation from a narrow mega-cap move | primary |
| Market regime | SPY momentum/volatility, VIX, term spread and dollar change | industry-timing conditions common to the whole basket | primary |
| Liquidity / microstructure | median quoted spread and dollar volume across eligible basket members | may proxy implementation stress, rather than alpha | later diagnostic only |
| Latent regimes | two- or three-state HMM using basket and market returns | possible descriptive regime layer | deferred; not a primary model input |

## Rules adopted from this review

1. **Factor families first.** Highly correlated lookbacks are not treated as independent discoveries. The primary specification is capped at 15 raw features across four economic families.
2. **Feature selection happens inside training only.** A sandbox candidate must show consistent direction and incremental value across chronological, purged training folds before it can enter a later version of the model. Validation data cannot be used to add factors.
3. **Simple comparators stay in the study.** The ML model must be compared with always holding the basket, SPY, and a 20-session excess-momentum rule.
4. **Signal and strategy are assessed separately.** Classification/calibration measures describe a forecast; net return, turnover, drawdown and transaction costs determine whether a trading rule is viable.
5. **No unbounded factor search.** Alpha101-style bulk enumeration, many transforms of the same return series, and neural networks are excluded from the first development round because they would create a severe multiple-testing problem in this sample.

## Immediate implication

The already drafted `AI_INFRASTRUCTURE_TIMING_PROTOCOL_v1.md` remains appropriate for the first run. The short-versus-long momentum spread and liquidity/stress measures will be written to a separate training-only factor sandbox only after the primary 15-feature benchmark and its simple economic comparators have been evaluated.

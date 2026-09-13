# Calculation review — 2026-09-12

Status: retrospective source-code review while explaining the model to the user. No data transformations or model reruns performed. Previous v2 model outputs are provisional and must not be treated as independently validated strategy results.

Confirmed implementation issues:

1. Training labels are constructed across the full development calendar and then split by formation date. No exit-date purge is applied at the 2023/2024 boundary. Final training labels therefore include validation-period returns.
2. Breadth uses per_name.gt(0).mean(). Missing rolling returns become False before averaging, incorrectly counting unavailable names as non-positive observations.
3. Excess momentum compounds daily return differences. This is not the difference between separately compounded basket and SPY horizon returns.
4. Reported sharpe_zero_rf divides CAGR by annualised volatility. Standard zero-risk-free Sharpe uses arithmetic mean periodic return divided by periodic standard deviation, scaled by sqrt(periods/year).
5. Basket weights depend on same-day return availability. This descriptive index does not establish executable start-of-period weights. Daily sleeve/name reweighting is omitted from switch counts and costs.
6. Signals use formation-close data while target returns start from the same close. Executable trade timing has not been established. Five-session endpoint drawdowns omit intra-period drawdowns.
7. Existing validator repeats some builder formulas and omits these checks. Its 10 PASS checks do not validate the complete financial pipeline.

Scope limitations: the screenshot has not been fully transcribed and mapped; the 41-name subset is from the existing registry, not a verified match to all screenshot companies. Registry membership dates need their economic meaning checked before reuse. Daily 694/497 observations have overlapping five-session labels and are not independent samples. Approximately 100 non-overlapping validation holdings are not automatically independent either.

Next action: correct definitions, purge and execution conventions and independently test them before additional model comparison. No new performance claim is supported by this review.

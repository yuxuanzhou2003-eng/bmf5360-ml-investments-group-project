# Report positioning and next-stage research plan

## Recommended framing

The project is a reusable machine-learning investment-research framework. The AI supply-chain universe is the first completed pilot, not a claim that the final fund must be an AI-only fund.

Suggested working title:

> From ML Signals to Investable Strategies: An AI Supply-Chain Pilot Study

The report should answer a broader question: how can a research team turn a financial hypothesis, an auditable investable universe, point-in-time information and ML predictions into a strategy that can be evaluated honestly?

## What the AI pilot has established

- A 49-RIC US-listed AI supply-chain exploratory universe can be documented by economic role, membership span, evidence source and delisting treatment.
- A daily H21 prediction pipeline can preserve a training/validation split, apply training-only preprocessing and keep final test labels sealed.
- Technical, market and macro factors can be documented with formulas, units, lookbacks and information timing.
- Logistic Regression is a valid supervised-ML benchmark. In this pilot it outperformed the matched Random Forest reference.
- Pooled ROC-AUC and a tradable cross-sectional stock-selection signal are different claims and must be separated.

## What the pilot has not established

- The 49-company exploratory universe is not yet a complete historical point-in-time AI universe: 40 of 49 companies have provisional static evidence rather than direct local point-in-time evidence.
- The pooled technical-plus-macro ROC-AUC of 0.5616 is not evidence that macro variables improve stock selection. Twenty macro variables and seven technical variables are common within formation date. The 21-session block-bootstrap confidence interval for the pooled macro AUC lift includes zero.
- Same-date selection evidence is weak: the macro-minus-technical pair-weighted within-date AUC lift is 0.00624 with a block-bootstrap interval crossing zero; the rank-IC increment also crosses zero.
- The current top-minus-bottom spread is a diagnostic, not a costed portfolio NAV, annual return or Sharpe ratio.
- The sealed 2023-01 to 2026-06 test labels have not been opened for this pilot.

## Recommended report narrative

1. **Research framework.** Define a fund-research workflow from universe formation through deployment constraints.
2. **AI pilot.** Explain why AI supply-chain roles are a useful first universe: rich economic structure, observable market data and a clear need for disciplined membership rules.
3. **Data and leakage controls.** Show the F-1 information boundary, H21 target, train/validation split, preserved missing values and audit trail.
4. **Model baseline.** Use Logistic Regression and Random Forest as transparent, matched comparisons.
5. **Key learning.** Separate sector timing from within-sector selection. Do not equate pooled AUC with investable alpha.
6. **Decision framework.** Evaluate later candidate strategies using the same criteria rather than selecting the highest in-sample metric.
7. **Next stage.** Select one final strategy, freeze its universe, target, model and trading rules, complete validation-period portfolio analysis, then perform one final sealed test.

## Criteria for choosing the final strategy

| Criterion | Evidence required before selection |
| --- | --- |
| Economic mechanism | A falsifiable reason why returns or risk should differ from a benchmark. |
| Point-in-time feasibility | Dated universe membership and feature availability, including delistings and exclusions. |
| Sample adequacy | Enough independent dates/events after accounting for holding-period overlap. |
| ML contribution | A pre-specified improvement over a simple model or rule, measured with dependence-aware uncertainty. |
| Implementation | A frozen portfolio rule, costs, turnover, capacity/borrow assumptions and risk controls. |
| Test discipline | One untouched final period after all design choices are fixed. |

## Reporting language to use

Use: “The AI pilot provides a leakage-controlled, reproducible modelling prototype and identifies a possible mixed timing/selection signal.”

Do not use: “The AI model proves stable alpha,” “macro factors improve stock selection,” or “the AI fund is ready for investment.”

## Linked package materials

- `BMF5360_AI_Baseline_Model_Calculations.xlsx`: company, factor, model and audit tables.
- `BMF5360_AI_Baseline_Model_Report.docx`: readable technical explanation and limitations.
- `audits/timing_selection_validation/`: validation-only timing-versus-selection audit code and results.
- `code/`: frozen technical-only and technical-plus-macro baseline runner.
- `DATA_PROCESSING_LOG.md` and `AI_USE_LOG.md`: cumulative provenance.

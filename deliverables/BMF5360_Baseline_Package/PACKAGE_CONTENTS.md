# BMF5360 baseline submission package

This package is the development-and-validation deliverable for the AI supply-chain daily baseline. It contains no sealed test labels and no new test predictions or metrics.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `BMF5360_AI_Baseline_Model_Calculations.xlsx` | Company universe, selection logic, factor dictionary, validation metrics, coefficients, audit and sources. The summary labels pooled AUC as a mixed timing/selection diagnostic. | `c8b30cf7bf43a0e80614499fa8f5d2988b77ae3c8d3fa20fcf87a04dfa7b0a2a` |
| `BMF5360_AI_Baseline_Model_Report.docx` | Plain-language technical explanation, including the timing-versus-selection and block-bootstrap limitations. | `31b753f164cf055da5dc2b578e5882d0ec14d78b346273834801e66a213d2620` |
| `README.md` and `code/` | Reproducible training/validation runner, locked configuration and dependencies. | Runner: `e830737f5997d07af42501e7f940f728039b027f53c922037de7fed842bd012d` |
| `BASELINE_LOGISTIC_MODEL_CARD.md` | Detailed model-card record for the technical-plus-macro Logistic benchmark. | `3fb266645ffc93ebd28c0c146c671b8fa7c32852b7d4245145538c42ccc76678` |
| `DATA_PROCESSING_LOG.md` | Cumulative record of material data processing, exclusions, missing-value handling and audit checks. | `d646e05688d771b8d7a5db1e17f967ac78546f41829a0fd2526aea9c87570cfb` |
| `AI_USE_LOG.md` | Cumulative material AI-assistance log required by the course. | `b0da010915695e53d52d416ca0357fd0cce441ec614f30732ceddd9b08904586` |
| `REPORT_POSITIONING_AND_NEXT_STAGE.md` | Report narrative: AI as a completed pilot within a broader strategy-research framework, with decision criteria for the final strategy. | See file-level hash if an external submission checksum is needed. |
| `audits/timing_selection_validation/` | Validation-only code and results separating pooled timing/selection diagnostics from within-date stock-selection evidence. | No test labels, raw LSEG data or fitted model artifacts. |

The raw LSEG data are not copied into this package because of size and data-source licensing/provenance constraints. The code README names the exact frozen local input versions needed for an authorised reproducibility run. The runner uses only `targets_dev.csv`; it never opens `targets_test_sealed.csv`.

Validation figures are research evidence, not an investment performance claim. In particular, ROC-AUC is a classification-ranking statistic, and the current top-bottom spread is not yet a costed portfolio NAV or Sharpe ratio.

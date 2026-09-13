# BMF5360 Group Project: ML Stock-Selection Pilot

Course project for NUS BMF5360 Machine Learning in Investments. Current deliverable is the midterm report (Week 5).

## Where things are

| Item | Path |
|---|---|
| Midterm report (current draft, v4) | `deliverables/Midterm_Report_GroupX_v4.docx` / `.pdf` |
| Report outline and decision log | `deliverables/MIDTERM_PROPOSAL_OUTLINE.md` |
| Report build scripts and figures | `_artifact_work/build_midterm_report.py`, `_artifact_work/build_figures.py`, `_artifact_work/figures/` |
| Baseline submission package (code, model card, Excel, logs) | `deliverables/BMF5360_Baseline_Package/` |
| Research protocol, specs, feasibility notes | `RESEARCH_PROTOCOL.md`, `AI_*.md`, `project_docs/` |
| Data processing and AI-use logs | `DATA_PROCESSING_LOG.md`, `AI_USE_LOG.md` |
| Model run used in the report | `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/` |
| Validation portfolio run | `data/backtests/ai_portfolio_v1/20260910T133000000000Z/` |
| Independent audits | `data/audit/` (selected runs) |
| Collection, cleaning, modelling, validation scripts | top-level `*.py` |

## What is not in the repo

Raw and cleaned LSEG market data (`data/raw`, `data/clean` except FRED macro), model-ready panels, and the sealed test targets are excluded for size and data-licensing reasons. The `.env` file with the LSEG App Key is never committed. Scripts expect the project virtual environment (`requirements-lseg.txt`, `requirements-ml.txt`) and a local LSEG Workspace session.

## Key numbers in the report

Validation 2021-2022, 992 anchors: market + industry-state Logistic ROC-AUC 0.5799 (market-only 0.5395; paired bootstrap lift interval [0.0007, 0.0801]). Pre-registered hedged portfolio -3.10% net; +14.6% in the 21-session windows after executed rebalances, -15.4% in the four skipped-rebalance extension windows. Test period 2023-01 to 2026-06 remains sealed.

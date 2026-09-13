# AI-Use Log — BMF5360

### EPS field semantics probes (2026-09-09)

OpenAI Codex and a delegated GPT-5.6 Luna research task reviewed official LSEG examples and created an evidence-ranked field candidate matrix. Codex then ran 18 isolated local LSEG field probes for AAPL.OQ/NVDA.OQ plus five follow-up calls after a combined-field dataframe error. It audited currency, scale, fperiod, ActType candidates and alternative EPS measures. The NVIDIA 2024 comparison showed the IBES Reported-selector value matching issuer non-GAAP split scale while alternative diluted/normalized fields matched GAAP scale. A first audit-output run failed JSON serialization and was preserved; a new run passed 15 checks and ten independent checks. No returned field was merged into the clean data or panel, and no model/backtest was run.

### LSEG stock-split collection and audit (2026-09-09)

OpenAI Codex identified LSEG corporate-action fields from official/developer documentation, created a raw-only collector, ran a three-security probe, then collected stock-split events for all 782 requested RICs through the local LSEG Workspace session. The run produced 807 raw rows: 113 valid events across 88 RICs and 694 retained blank-event placeholders. Codex created a separate audit with factor/terms consistency, duplicate, row-conservation, input-hash and common-scale checks, and independently recomputed ten checks. No split factor was applied to actuals, estimates, returns, labels, or positions. Issuer comparisons for Apple, Tesla and NVIDIA and the remaining basis/vintage/date limitations require group review.

Codex also matched later split events to every source event in the frozen feature panel without reading labels or returns. It flagged 1,336 of 22,523 source events and 6,347 of 107,532 edges as having at least one later split, verified common-scale z-score invariance, retained 42 zero-dispersion events as missing, and independently recomputed the exposure counts. This is a review inventory, not an error classification or a data correction.

### Coverage split inventory completion (2026-09-08)

Documented the completed read-only audit under `data/audit/coverage_split_inventory/20260908T1128053483021Z/`. Independent review confirmed all 17 missing-actual and 14 missing-estimate instruments were retained; 26 incomplete labels across 12 receiver RICs have unknown causes. Training/validation/test edge rows are 54,421/19,415/33,696, complete labels 54,411/19,412/33,683, edge events 11,626/3,961/6,936, with zero boundary crossings. Eight rows lack entry-day return observations and require price/execution review; this is not proof of non-trading. No source edits, imputation, new LSEG retrieval, model run, or performance analysis occurred.

### Research protocol and field semantics review (2026-09-08)

Recorded `RESEARCH_PROTOCOL.md` as design-only: train 2015–2020, validation 2021–2022, test 2023-01-01–2026-06-30, whole-event boundary purging, and a proposed verified-ET-date-close entry rule; none was applied. Recorded `FIELD_SEMANTICS_REVIEW.md`, based on prior probes and official LSEG materials without fresh retrieval or input transformation. Timezone, EPS basis/units/split adjustment and vintage remain unresolved; actual/mean/std are not approved inputs. Coverage split inventory was subsequently completed; see the completion entry above.

This cumulative log records material AI assistance. Entries require group review before submission.

### LSEG Data Item Browser operation and field evidence (2026-09-09)

- At the user's request, Codex operated the already-running local LSEG Workspace through reversible window and keyboard control, opened Data Item Browser, selected `AAPL.OQ`, and inspected `TR.EPSActValue`.
- Preserved screenshots and hashes under `data/audit/lseg_data_item_browser/20260909T012702710Z/`. The field definition states normalization to I/B/E/S default currency and corporate actions including stock splits. Visible Actual Type choices were `Reported`, `Restated`, `GoForward`, `All`, and `Latest`; visible parameter defaults were documented in `FIELD_NOTE.md`.
- Updated the field-semantics conclusion: no manual split multiplication should be applied to `TR.EPSActValue`; `Reported` remains an I/B/E/S analyst-basis actual and is not asserted to equal issuer GAAP diluted EPS. Point-in-time vintage, announcement timezone, and common actual/consensus basis remain unresolved.
- No dataset row was changed, deleted, imputed, or reclassified. No model or backtest was run. Group review pending.

| Date | Tool | User request | Material assistance | Verification and status |
|---|---|---|---|---|
| 2026-09-07 | OpenAI Codex | Search GitHub for reusable ML investment repositories; consider LSEG data. | Searched primary GitHub sources, compared ML4T ETF case study, Qlib, Systematic Sector Alpha, LSEG market-regime example and LSEG Python data examples. Proposed an ETF rotation research design and outlined data/validation requirements. | Repository pages and selected source material inspected; no strategy executed, no results reproduced, no LSEG account access or data availability verified. Recommendations are preliminary; group review pending. |

## Sources consulted

### Panel v2 engineering output documentation (2026-09-08)

- Documented the retrospective engineering panel run `data/panel_v2/20260908T053814601256Z/summary.json` and independent validation `data/audit/panel_v2/20260908T111211716224Z/validation.json` / `output_profile.*`. Recorded 29,003 events, 22,824 matched snapshots, 22,732 graph-ready events, 92 insufficient-history events, 209 zero-neighbor events, 22,523 events with at least one edge, 107,532 rows, 107,506 complete labels, 26 incomplete labels across 12 receiver RICs, 10,615 negative-correlation edges, 21/21 validation checks and 40 independently recomputed OLS correlations.
- Recorded the implemented rules and non-actions: 126-session/100-observation history, `abs(rho) >= 0.3`, max 5 neighbors, negative edges allowed, exact announcement-day membership, no imputation, no delisted-only deletion, no identifier substitution, no timezone conversion, and EPS raw fields trace-only. Announcement timezone remains unresolved; the 17 actual and 14 estimate company-level gaps remain a coverage limitation. No model or backtest was run.
- At the time of that entry, explicitly documented that the original ex-post overlap diagnostic had a false/unknown semantic issue and was pending correction. The subsequent correction is recorded below; full model-ready status is still not claimed.

### Panel v2 diagnostic correction (2026-09-08)

- Reviewed the corrected panel at `data/panel_v2/20260908T111211716224Z/`, its diagnostic audit, and independent validation `data/audit/panel_v2/20260908T111836060858Z/validation.json`. The correction changed only 762 rows across 11 receiver RICs from `False` to `NA/Unknown`; corrected counts are True 37,808, False 68,962, Unknown 762.
- Confirmed that the original five output hashes and events/features/labels contents are unchanged; 8/8 correction tests and 21/21 independent validation checks passed. False means no observed match and does not prove no event; available actuals do not prove full coverage. No model or backtest was run.

### Historical membership, field semantics and engineering panel follow-up (2026-09-08)

- User authorized missing-data and historical-timing verification followed by network/label preparation, with processing visibility. Luna subtasks were interrupted by usage limits; explicitly selected Terra fallbacks resumed only unfinished work after inspecting saved outputs.
- Astra identified and corrected order-dependent rollback of an EVHC.N^L16 same-day Joiner/Leaver pair. Original code/data summaries were backed up. Atomic daily changes remove one false membership interval; 777 membership flags and 8 event eligibility statuses changed, with all return values and raw rows preserved. Revised eligibility 22,831; matched snapshots 22,824. Original 782-RIC collection manifest retained; corrected 781-RIC eligibility spans saved separately.
- Existing semantics probe: three RICs, five API requests; unsupported origtimezone/ActType candidates recorded as failures, not confirmations. Apple primary-source EPS 3.06 versus returned 0.765 is consistent with a later four-for-one split; it does not establish full vintage semantics. A read-only ET-vs-UTC sensitivity found 111 date differences and 216 entry-close differences among 32,415 actual rows. Timezone remains unknown.
- Terra completed seven targeted identity/EPS calls. FDX and FDX.N identify the same organization but remain blank for actuals/estimates; FDXF is a distinct organization, as are current and historical DD identifiers. Selected historical EPS requests remain blank. No recovered rows merged, numerical imputation, or entity substitution performed.
- Panel code preparation separates engineering calculations from unverified economic timing. Astra prepared an independent persisted-output validator using full forward-return recomputation and sampled independent OLS residual correlations. Actual panel execution/results will be logged only once completed; no investment performance claim is implied.

### v2 recovery, cleaning and event-readiness analysis (2026-09-08)

- User authorized continuing data analysis with transparent, cumulative processing records. Astra coordinated and reviewed; Luna handled bounded raw-data audit, recovery and cleaner/validator changes. Luna later hit a usage limit; Terra was explicitly spawned as the user-authorized fallback for unfinished event-readiness analysis.
- Two missing estimates batches were recovered: 6,710 + 1,009 rows. 84/84 estimate batches now have successful metadata; raw estimates total 463,871 rows. Recovery bookkeeping threw a duplicate-key TypeError after successful first installations, causing two redundant second requests. Existing successful files and original errors were preserved; corrected recovery verification records the actual outcome.
- Backed up cleaner/validator code; fixed duplicate-conflict handling, retained per-table audit provenance, archived old outputs, and made missing required validation inputs fail explicitly. Synthetic edge cases and a separate parent regression check passed.
- Executed full clean and independent validation. Returns 2,256,958→2,081,816; actuals 33,206→32,415; estimates 463,871→424,101. All excluded rows retained with reasons. Returns and actuals byte-identical to archived prior outputs. No numerical imputation, winsorization, or exclusion solely for delisting. 223/223 engineering checks passed; this does not establish historical point-in-time correctness or predictive value.
- Parent review caught a mismatched denominator in the initial read-only coverage report (all observed securities divided by eligible securities). The analysis was corrected to use the intersection with the eligible population; the original ratio must not be described as coverage.
- No model fitted or portfolio backtested. Event-readiness completion and figures must be taken from its actual run report, not assumed from this work plan. Group review pending.
- Subsequent executed event-readiness run 20260908T013521791934Z (Terra fallback): 29,003 events in the inclusive date window; 22,839 eligible at announcement; 22,832 matched prior snapshots, 4 missing quarters, 3 stale snapshots. 44 zero-dispersion events retained with unavailable standardized surprise. Astra independently recomputed match decisions/consensus with merge_asof and all 22,788 computable standardized surprises; results agreed. Missing-company coverage remains a limitation; no graph, future labels or model used.

### Data-processing visibility and retrospective documentation (2026-09-08)

- User requested an explanation and durable record of every material data-processing step, including removals, delisted securities, and missing-value handling.
- Codex added project AGENTS.md rules and delegated retrospective DATA_PROCESSING_LOG.md documentation to a GPT-5.6 Luna child. This records existing code and persisted evidence, separating planned, implemented, executed, and verified work.
- The preceding read-only v2 review found incomplete estimates/panel artifacts, partial audit overwriting and skipped checks, and inconsistent membership dates in graph selection. These are review findings, not fixes or a successful pipeline rerun.
- No investment data was modified, no missing values were imputed, and no LSEG collection or model/backtest was run in this documentation step. Group review pending.

### Versioned collection and cleaning (2026-09-07)

- User requested another run and prioritization of data collection/cleaning and dataset status.
- Codex added collect_dataset_v1.py, clean_dataset_v1.py, validate_dataset_v1.py and run_data_pipeline.py; extended the existing panel builder to consume clean tables and separate features, targets and ex-post diagnostics.
- Collected current company metadata, a 205-row historical index-change sample, a rejected dated-chain cross-check and December 2014 Reuters metadata with RIC tags. Detected server news cap of 100 and split time windows to cover the requested month. Cached responses include query, retrieval time and SHA-256.
- Snapshot raw pilot data, normalize missing/date/numeric values, quarantine invalid/conflicting rows, deduplicate news versions and retain provenance. Final clean counts: returns 31,496; actuals 349; estimates 4,669; news versions 1,131. Quarantined two FDX blank placeholders, 1,301 duplicate news rows and seven news timestamp contradictions. News tags yield 465 pair-story records / 101 distinct candidate pairs, not verified business links.
- Rebuilt 1,047 event-neighbor records; independent target recomputation, schema separation, chronology and file checks passed (61 total checks, many per-file). Fixed mixed-precision timestamp parsing discovered during validation; preserved real large-return observations for review. Clean valid panel equals prior valid panel after key sorting.
- DATASET_STATUS.md and DATA_DICTIONARY.md document limitations: selected surviving stocks; unresolved initial universe, announcement timezone, EPS vintage/adjustment, historical industry, full news history and implementation costs. No models or return claims. Group review pending.

### Information diffusion engineering pilot (2026-09-07)

- User said continue after choosing the information-diffusion direction. Codex ran bounded LSEG data probes and created a 30-stock, 2015–2017 engineering sample, with earlier return warmup and later label coverage.
- Created diffusion_pilot_config.json and build_diffusion_pilot.py; extended probe_lseg.py; produced INFORMATION_DIFFUSION_PILOT.md and local raw/processed evidence.
- Retrieved 31,496 stock/benchmark return rows, 4,669 valid weekly quarterly-consensus rows and 349 events from 29 companies. Matched 349 same-quarter preannouncement snapshots and constructed 1,047 event-neighbor records. Five chronology/structure checks passed. FDX.N event coverage missing; 25 receiver-overlap records marked unknown, not false.
- Found 328 overlapping receiver announcements and potentially weak/economically ambiguous correlation links. Historical dated-chain requests matched current constituents and were rejected as historical-universe evidence. No model training or portfolio backtest; hand-picked universe is survivor-biased. EPS adjustment, timezones and point-in-time completeness remain unresolved. Group review pending.
- Reference checked: https://community.developers.lseg.com/discussion/48816/query-about-epsmean-and-associated-parameters ; https://community.developers.lseg.com/discussion/96654/date-format ; https://community.developers.lseg.com/discussion/75644/get-data-for-specific-date . API examples were treated as hypotheses and verified against actual returned data.

### Information diffusion proposal (2026-09-07)

- User expressed interest in the information-diffusion fund idea.
- Codex reviewed the original Economic Links and Predictable Returns paper abstract/publisher record and author repository descriptions for Temporal Relational Stock Ranking and HIST. Created INFORMATION_DIFFUSION_PROPOSAL.md: earnings-anchored graph research design, baseline comparisons, execution-aligned targets, leakage controls and data gates.
- Scope and methods are proposals, not empirical findings or final user-approved specifications. No new data requests, training or backtests in this step. Historical universe and matched quarterly expectations remain unresolved; graph attention is not causal evidence. Group review pending.

### Local LSEG access tests (2026-09-07)

- User authorized use of the local .env and running LSEG Workspace to test data access.
- Codex created a read-only probe script, isolated dependency environment, dependency list, ignore rules and LSEG_DATA_FEASIBILITY.md. Credentials remained local and were redacted from tool output; no trades or external messages were sent.
- Verified three ETFs' 2010–2025 daily prices, sampled total returns and volume, two firms' historical EPS consensus/dispersion/coverage, earnings actuals with report timestamps, accounting values, news headlines and one historical article body, EUR spot quotes and Treasury yields.
- .VIX history returned permission denial; 2010 S&P 500 constituents returned empty member values despite current constituents working. FTSE historical query returned 67 members, completeness unverified. Historical options and DSWS were not tested.
- Corrected an initial empty-string non-null counting issue after inspecting constituent CSVs; no assertion of point-in-time completeness or strategy profitability. Suggested prioritizing projects 1, 2 and 6 based on tested access. Group review pending.

### Follow-up: ten advanced project candidates (2026-09-07)

- User requested ten master's-level alternatives, accepted ML4T as a reference, and allowed larger datasets.
- Codex searched primary repository and LSEG documentation pages and proposed: regime-conditioned multi-asset allocation; uncertainty-aware allocation; decision-focused portfolio learning; conditional latent-factor equities; ML-filtered statistical arbitrage; earnings-announcement drift; news/price multimodal equities; volatility risk premium; FX carry/momentum; and deep hedging.
- Suggested sample sizes, models, strategy rules, difficulty comparisons and rankings are proposed research designs, not verified empirical findings. Linked repositories include both research workflows and individual model/data components; none has been reproduced in this session. NUS data entitlements remain unverified. Group review pending.
- Additional sources: https://github.com/scikit-learn-contrib/MAPIE ; https://github.com/jankrepl/deepdow ; https://github.com/bkelly-lab/ipca ; https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/case_studies/us_firm_characteristics ; https://github.com/stefan-jansen/machine-learning-for-trading/tree/second-edition/09_time_series_models ; https://github.com/LSEG-API-Samples/Article.DataLibrary.Python.NewsSentimentAnalysis ; https://github.com/ProsusAI/finBERT ; https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/case_studies/sp500_options ; https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/case_studies/fx_pairs ; https://github.com/hansbuehler/deephedging ; https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/actuals ; https://www.lseg.com/en/data-analytics/financial-data/company-data/ibes-estimates?elqCampaignId=20667

- https://github.com/stefan-jansen/machine-learning-for-trading/tree/main/case_studies/etfs
- https://github.com/microsoft/qlib
- https://github.com/areebarshad/systematic-sector-alpha
- https://github.com/LSEG-API-Samples/Article.RD.Python.MarketRegimeDetectionUsingStatisticalAndMLBasedApproaches
- https://github.com/LSEG-API-Samples/Example.DataLibrary.Python
- https://github.com/LSEG-API-Samples/Article.DataLibrary.Python.QuickReferenceGuide

For future entries, record code or text reused, source versions, changes made, checks performed, and the reviewing team member. Do not treat upstream backtest figures as this group's results.

### Estimates gap recovery (2026-09-08)

- OpenAI Codex reconstructed the 84 estimates batch requests from `collect_universe_v2.py` and the persisted universe spans, then created `recover_estimates_v2.py` to run a bounded recovery for only the two failed batches. Credentials were read from the local `.env` for LSEG Workspace access and were never printed.
- The run generated a pre-fetch plan, recovered `estimates_live_049` and `estimates_dead2015_000`, preserved both prior `.error.json` files, and wrote request/response audit artifacts under `data/audit/estimate_recovery/20260907T172009Z/`. No cleaning, imputation, deduplication, row removal, security exclusion, or model/backtest was performed.
- Independent post-fetch verification matched every expected batch's request, physical row count, metadata status, and SHA-256: 84/84 batches, 463,871 rows and 782 observed instruments. The original run report is retained; it records a bookkeeping TypeError after first-attempt installation and the consequent redundant second requests. A corrected no-network verification records both targets installed successfully on attempt 1. Group review pending.
## 2026-09-09 — v3 timing, execution-data collection and clean layer

- AI designed and executed isolated LSEG probes for earnings timing/vintage, AAPL stock-split price adjustments, execution/liquidity fields, and the AMCR selector anomaly. Every request and direct response was saved with hashes; no credentials were written to outputs.
- AI implemented a versioned, resumable v3 collector with fixed plans, request identity checks, atomic CSV/metadata installation, preserved error sidecars, and a corrected 781-RIC universe. It collected 39 actual batches and 163 price batches. One daily-estimates batch was retained as a pilot after its runtime showed that complete continuous daily retrieval would be disproportionate for the core version.
- AI implemented the v3 cleaning and independent validation workflow. Material judgments requiring team review are: conservative next-session entry, use of weekly consensus as the core with daily sensitivity deferred, tagged archival fallback for 29 AMCR events, exclusion of EVHC under corrected membership, structural-wide-padding quarantine, and tagged `close × volume` fallback when vendor turnover is missing.
- AI did not train a model, select hyperparameters, construct a portfolio, run a backtest, or write performance claims in this stage. The group must review all field interpretations, processing logs, and archived-source decisions before submission.

### v3 event-neighbour panel and audit (2026-09-09)

- OpenAI Codex migrated the engineering panel builder to the versioned v3 actual, weekly estimate and price/liquidity inputs while retaining the verified total-return table for historical graphs and labels. It implemented the conservative first-session-after-announcement entry rule and separated prior-session liquidity features, future labels, entry-day EOD execution data and ex-post diagnostics.
- Codex built `data/panel_v3/20260909T035245922341Z/`: 28,995 event audit rows and 107,532 source-receiver samples. It did not force neighbors below the correlation threshold, fill missing snapshots/prices/returns, or delete incomplete labels.
- Codex created `validate_panel_v3.py` and independently recomputed all five-session receiver and SPY returns, exact actual/estimate/price joins, sample keys and provenance. Final audit `data/audit/panel_v3/20260909T041022558174Z/` passed 31/31 checks, hashes the validator code, and includes `event_flow.csv` plus all 15 samples with entry-data missingness. An initial validator execution failed on an array-handling bug before producing an audit; the validator was corrected and rerun. No panel input or output was modified by validation.
- A delegated read-only review independently confirmed all 107,532 keys and labels, exact estimate/price joins, 4 missing lagged price rows, 9 missing entry price rows, AMCR source tags and complete EVHC exclusion. It also identified schema improvements to implement in the later eligibility layer: a standalone receiver column and explicit dollar-volume availability flag.
- Material judgments for group review are the 126-session residual-correlation graph, absolute 0.30 threshold, maximum five neighbors, weekly consensus with 14-day freshness, next-session-close entry, and the pending execution-eligibility/cost fallback rules. No model, portfolio, backtest or performance claim was produced.

### Model-ready v1 feature and split layer (2026-09-09)

- Codex created `model_ready_v1_config.json` and `build_model_ready_v1.py`, then generated `data/model_ready_v1/20260909T064151673764Z/`. The run retains all 107,532 panel samples and creates 45 backward-looking candidate features, event-level splits, boundary-purge flags, entry eligibility, label availability and feature availability without fitting preprocessing or a model.
- Codex separated contemporaneous entry eligibility from future label completeness. It identified 3 source events / 14 edges crossing the next split boundary, 17 entry-ineligible rows, 33 incomplete labels, 213 rows with at least one feature missing, and 107,270 rows meeting the current supervised-model eligibility components. No row was physically deleted and no missing value was filled.
- Codex created `validate_model_ready_v1.py`. Final audit `data/audit/model_ready_v1/20260909T064816383676Z/` passed 31/31 checks, including full eligibility/split recomputation and deterministic independent recomputation of 5,632 return-related and 1,536 liquidity feature values across 256 samples. An earlier validator audit `20260909T064653604512Z` preserved two false failures caused by strict pandas boolean dtype comparison; value mismatches were zero and the source data were unchanged.
- A delegated read-only review found that the conservative v3 entry rule creates three boundary-crossing events that the v2 split inventory did not have. This correction was incorporated before any model training. No test-period model selection, model fit, portfolio construction or performance claim occurred.

### Training/validation baseline models (2026-09-09)

- Codex installed and pinned scikit-learn 1.9.0 in the project virtual environment and recorded the complete ML package versions in `requirements-ml.txt`. Existing LSEG, provider, MCP and plugin configuration was not changed.
- Codex created a frozen baseline configuration and trained a simple calibrated network score, receiver-only Ridge, network-augmented Ridge and constrained histogram gradient boosting model. Hyperparameters were selected with three forward training-only folds and event-equal weights; preprocessing was fitted inside each fold. The final development run used training and validation only and produced no test prediction or metric.
- Validation results were weak and were reported as such. Network Ridge ranked first by the predeclared mean event Spearman (0.0172) but had negative weighted R² and only a 0.0013 event-rank increment over receiver-only Ridge. Receiver-only Ridge had stronger pooled rank correlation and slightly positive weighted R²; HGB was worse. No alpha, significance or fund-performance claim was made.
- Codex created `validate_baseline_models_v1.py`; audit `data/audit/baseline_v1/20260909T071048556782Z/` passed 13/13 checks by reloading every saved model, reproducing all validation predictions and metrics, checking CV selection and hashes, and confirming that predictions contain only 2021–2022 rows.

### Post-baseline diagnostics and next-open target test (2026-09-09)

- Claude Opus 5 took over from the previous OpenAI Codex session using the handoff document `CLAUDE_HANDOFF_2026-09-09.md`. It reviewed, compiled and ran `diagnose_next_open_targets_v1.py`, which Codex had written but never executed or validated.
- Before running, Claude verified three mechanical risks named in the handoff: the script compiles; MultiIndex `reindex` with `NaT` exit keys returns NaN instead of raising; `DataFrame.stack()` raises no FutureWarning under pandas 2.3.3. It also found that the script's own `targets_finite_when_available` check has no discriminating power, because `stack()` drops NaN by default so the check is always true. The diagnostic script was left unmodified and a real masked check was added to the independent validator instead.
- The command timed out in the foreground at 600 seconds and was moved to the background by the CLI, which executed the script twice and produced two run directories. All six output files are byte-identical by SHA-256. `20260909T075928064059Z` is the official run; `20260909T075826878720Z` is retained as duplicate-execution and determinism evidence. This incident is recorded rather than quietly cleaned up.
- Claude wrote `validate_next_open_targets_v1.py`, an independent validator that recomputes entry opens with a merge instead of a MultiIndex reindex, locates SPY sessions with searchsorted instead of get_indexer, recomputes exit sessions, boundary purges, target availability and target values for all six horizons, and independently refits all three models at horizons 1 and 5 to reproduce the stored validation predictions and metrics. Audit `data/audit/next_open_targets_v1/20260909T080525524443Z/` passed 26/26 checks.
- The result is negative and is reported as such. Moving theoretical entry from the next session close to the next session open does not recover the relation signal: the raw network signal stays within ±0.007 mean event Spearman with inconsistent training/validation signs, and network Ridge event-level IC stays in the 0.0121–0.0219 range, the same noise magnitude as the close-target baseline.
- Claude appended the two corrected horizon/lead-lag diagnostic runs, their superseded pre-purge predecessors, and this open-target run to `DATA_PROCESSING_LOG.md`, `DATASET_STATUS.md`, `RESEARCH_PROTOCOL.md`, `DATA_DICTIONARY.md` and this log. No raw, clean, panel or model-ready file was modified, and no test-period target, prediction, metric or strategy result was computed.
- The decision on whether to pivot to a two-stage event/within-event model, change the relation definition, or narrow the research question is left to the group; `RESEARCH_PROTOCOL.md` v0.6 records the options and the constraints, and freezes no new specification.

### Relation-data feasibility probes (2026-09-09)

- After the residual-correlation graph failed its development-stage tests, Claude Opus 5 wrote three read-only probe scripts (`probe_relationships_v1/v2/v3.py`) and ran 34 independent LSEG requests over three sample RICs to test whether economically motivated relation edges are available under the current license. Each candidate field family was requested separately so that one unresolvable field could not hide the availability of others.
- Confirmed available: broker-level analyst coverage via `TR.RecEstBrokerName` with genuine activation-date history in both as-of and range form, and institutional ownership via `TR.InvestorFullName` retrievable as of a historical date. Confirmed unavailable: per-broker EPS estimates, where the broker-name column returns empty.
- Claude identified two point-in-time hazards that must be handled before any ownership graph is built. `Holdings Filing Date` is the holding period end, not the disclosure date — 99.3% of as-of 2015-12-31 rows are month-end and 79% equal the as-of date itself — so building edges on that date would embed roughly 45 days of look-ahead against actual 13F disclosure. Ownership range requests fail with a gateway timeout, so history can only be pulled one as-of snapshot per quarter. Neither a lag rule nor a collection plan was implemented; both are written up for user approval.
- Claude also recorded what could NOT be concluded. Thirteen supply-chain field spellings all failed with "unable to resolve field", which is a name failure and not a permission denial, so the finding is recorded as "these spellings are unavailable" rather than "this license has no supply-chain data". TRBC as-of 2015 returned values identical to current ones, which cannot distinguish a stable classification from an ignored SDate; point-in-time behaviour is recorded as unverified.
- One execution incident is recorded rather than cleaned up: the round-2 range request timed out at 90 seconds and the script exited by design, leaving twelve candidates untested. Claude rewrote the probe to be resumable so that a single timeout costs only the current candidate, then reran and obtained an explicit gateway-timeout error for that request.
- Findings are written to `RELATION_DATA_FEASIBILITY.md` and `DATA_PROCESSING_LOG.md`. No graph, model, portfolio or backtest was produced, no existing project data file was read or modified, and the test period remains sealed.

### Independent analyst-model validation (2026-09-09/10)

- Codex created `validate_analyst_models_v1.py` to independently audit the frozen analyst-model run `data/model_runs/analyst_v1/20260909T172214155509Z/`. It checked output completeness and recorded input/config/runner/output/model hashes, rebuilt training/validation counts and rolling-fold boundaries from eligibility plus metadata, and verified no sample/event/time leakage.
- The validator independently recalculated validation pooled Spearman, equal-weight mean event Spearman and event-weighted MSE for all seven models from the saved validation predictions. It independently rebuilt `analyst_hgb` minus `context_ridge` paired event deltas and the configured announcement-month block bootstrap (seed 5360, 2,000 replications), and checked confirmatory versus exploratory labels.
- Audit `data/audit/analyst_model_v1/20260909T173749815485Z/` passed 45/45 checks. Training/validation counts were 55,763/19,764 rows and 11,213/3,965 events; nominee paired event delta was -0.0238882164409603 with 95% CI [-0.0522669870810809, 0.0081128930757617]. `targets.csv` was hashed and its header inspected only; no test future-return row was parsed, aggregated, scored or predicted, and no model-ready/model-run/runner file was modified.

- The validator was rerun after removing an internal summary alias and adding a retained eligibility-audit row/ID/inclusion check. Final audit `data/audit/analyst_model_v1/20260909T174131297373Z/` passed 46/46 checks; the earlier 45/45 audit remains retained as a superseded run record. The final run preserved the same independent validation metrics and announcement-month bootstrap values, and still parsed no test target rows.

### Independent AI-fundamentals probe v2 validation (2026-09-10)

- OpenAI Codex created and ran `validate_ai_fundamentals_probe_v2.py` (SHA-256 `6e58164f782f7c7be00fe5a5e162d6cb23dccbb9052cff438da6ff5dac294019`) against the frozen raw run `data/raw/ai_fundamentals_probe_v2/20260909T174626399754Z/`. The validator independently recomputed plan/request/metadata/CSV hashes, request statuses, row/column/non-null counts, date ranges, duplicate diagnostics, period-end alignment, announcement/update lags, and raw-file before/after hashes.
- The final audit is `data/audit/ai_fundamentals_probe_v2/20260909T180200648853Z/` with `validation.json`, `validation.csv` and `findings.md`; 25/26 checks passed. The only retained failure is a physical CSV versus metadata non-null-count mismatch in R&D `Financial Period Absolute` (raw 45/46 versus metadata and probe summary 46/46); no file was corrected or overwritten. The earlier superseded validator output `20260909T180004095495Z/` remains retained.
- The audit classified 54 annual slots independently by `Instrument + Income Statement Period End Date`: R&D 45 matched slots plus 9 WMT no-row slots and one unkeyed blank row; Revenue, IS dates and five TR.F control families each cover 54/54. It found announcement-period lag 11–755 days and Last Update after announcement in all 54 rows with lag 903–3,808 days. Emitted annual `.date`/`Date` values match period-end keys in this probe, but were not treated as available dates.
- Monthly market-cap value/date responses each contain 648 rows (108 per instrument) with identical Instrument order and a complete 108-month date response. Positional pairing is recorded only as a candidate; formal join approval remains false and the audit recommends a future same-request value+date retrieval. No test target, return, factor, label, model or portfolio data was read or constructed; no deletion, imputation, deduplication, exclusion, unit change or timezone write-back occurred.

### Independent AI-fundamentals PIT probe v3 validation (2026-09-10)

- Codex created and ran `validate_ai_fundamentals_pit_v3.py` (SHA-256 `bac2fcb0ac9660ce046c6a1a1f512b540be8e2fef5d2a4686210d473de21be07`) against `data/raw/ai_fundamentals_pit_probe_v3/20260909T175636988821Z/`, with v2 current-probe comparison input `data/raw/ai_fundamentals_probe_v2/20260909T174626399754Z/`. The validator read only raw probe artifacts, recomputed request/metadata/CSV hashes, structure, parsed-valid date diagnostics, duplicates, annual keys and v2 differences, and wrote `data/audit/ai_fundamentals_pit_probe_v3/20260909T181610854824Z/validation.json`, `validation.csv` and `findings.md`.
- Final audit passed 28/28 checks. The v3 run retained 35 raw files and 1,395 CSV rows across 6 instruments; raw file set, sizes and SHA-256 were unchanged before/after validation. Metadata non-null counts were reported alongside raw-memory blank-aware counts and parsed-valid date counts, so the probe's dataframe `notna` convention was not treated as CSV damage. An earlier superseded audit directory `20260909T181315318416Z/` remains retained after validator logic was corrected; no raw input was modified.
- The v3 `ReportingState=Orig` versus v2 current comparison found 52 different values among 369 matched non-missing value pairs and 11 different IS Orig Announce Date cells; Last Update and Period End dates had 0 differences. These are sample observations only and do not establish the vendor's full original-vintage rule.
- The validator documented the 269-row business-segment structure: 54 Instrument + period/date groups, 58 blank-code/blank-name null-total candidates covering all groups, 0 named total-token rows, 44 blank-code named rows, 4 duplicate-total groups with 4 extra rows, reconciliation distributions, and per-instrument name-set changes. It explicitly records that AI keyword classification is not frozen. Combined market cap has 648/648 unique Instrument + Date keys with 108 months per instrument from 2014-01 through 2022-12.
- No cleaning, imputation, winsorization, forward fill, deduplication, row deletion, company exclusion, identifier mapping, unit conversion, return/label/factor construction, test-target read, model fit, portfolio construction or performance calculation was performed. No quarantine was created.

### Independent AI-factor collection trial validation (2026-09-09/10)

- OpenAI Codex created `validate_ai_factor_collection_trial_v1.py` and ran it read-only against `data/raw/ai_factor_v1/20260909T181731782199Z/`. The validator read only the frozen plan, collector manifest, summary and the three returned `rd_000`–`rd_002` request/metadata/CSV triples, plus the current collector/config/corrected spans files for provenance hashes and independent 781-RIC/154-delisted counts. It never opened LSEG or issued a request.
- The audit output is `data/audit/ai_factor_collection_trial_v1/20260909T183401961679Z/` (`validation.json`, `validation.csv`, `findings.md`); validator SHA-256=`eef6d9c936962e9085e9c23c347271235cc4eb7965f524b2495a1fccdcd779fa`. The plan has 357 requests = 11 fundamental families × 32 batches + 5 ETF requests. The trial returned exactly `rd_000`–`rd_002`, each requested 25 literal RICs; requested and returned unions are both 75 with no unexpected instruments. Physical CSV rows total 501.
- The validator reported physical non-empty versus empty CSV cells separately: `rd_000` 684/44, `rd_001` 622/58, `rd_002` 541/55, for 1,847/157 overall. Metadata/manifest in-memory `non_null` counts were retained as provenance; their `Financial Period Absolute` count exceeds physical non-empty by 11/12/14 because empty returned strings serialize as blank CSV cells. No imputation or conversion was applied, and no literal zero was reclassified as missing.
- Raw request, metadata and CSV hash chains passed. The raw trial file set, sizes and SHA-256 were identical before and after validation; no raw row was changed and no quarantine was created. The collector summary flags show no test-target read, future-return generation, label generation or factor/model output. Delisted RICs were retained and no identifier mapping, deletion, deduplication, winsorization, unit/date/timezone transformation or label filtering occurred.
- The collector's exit 1 is recorded as expected `--limit 3` partial completion: 3 attempted, 3 successful/empty, 0 errors, 354 pending. The previous plan-only run `20260909T181534358580Z` is retained because a script hash change caused failure before any request; the formal trial is `20260909T181731782199Z`.
- Final status is `failed_checks` with 48/50 checks because the current worktree collector/config hashes (`f591b7d8319a468dc6acd80c6d845833e6dd5620f2c751e5f0dcfd43c2b57931` / `c29fa12e22c9128f2fd966bd4b5a11ee1d13446cf3bde5dacc5e62dc236bf929`) differ from the hashes recorded by the formal trial (`0ef90dd426551a58d14bb311214d4663a62fd5b5261d27dea110a1367f0c0a47` / `6e77856b3e8bcbb3abaeb35a20345e16f055cd99ce628009acf481975d150e85`). This is retained post-trial provenance drift, not raw corruption. Superseded validator outputs `20260909T183053432313Z` and `20260909T183233667125Z` remain preserved after the physical-missingness and per-family membership logic was corrected.

### AI-factor collector repair, full raw collection and cleaning plan (2026-09-10)

- Luna reviewed the collector design and identified a retry-state bug; the collector was revised to distinguish a retryable error/timeout sidecar from a tampered success artifact, and to hash both canonical request content and request-file bytes. Seven offline state-machine tests passed. The monthly market-cap family is explicitly documented as a market-data exception that omits `ReportingState=Orig`.
- Codex ran the repaired collector against LSEG using the frozen 781-RIC corrected-span universe. Run `data/raw/ai_factor_v1/20260909T183801359195Z/` returned all 357 requests (11 fundamental families × 32 batches plus 5 ETF requests), 244,383 physical rows, and no errors/timeouts. All RICs, including 154 delisted flags, were retained; no clean table, label, model or portfolio was produced.
- A quick independent disk scan recomputed all 1,071 request/meta/CSV physical hashes and verified family counts, 781-RIC span counts, and false test-seal flags. This is recorded as a structural cross-check while the per-field audit artifact is pending; it is not being represented as model readiness.
- Codex authored `AI_FACTOR_CLEANING_PLAN.md` v0.1 before any clean transformation. It specifies blank-aware NA handling, explicit period/announcement joins, Orig-only reporting fundamentals, segment duplicate/reconciliation reason codes, no imputation/winsorization/deletion, retained delisted rows, and price-return versus total-return separation. No clean-stage transformation has yet run.

### Research-objective revision (2026-09-10)

User clarified that the fund should invest specifically within an AI-industry stock pool. Codex updated the research protocol and created `AI_POOL_SPEC.md` v1.0 defining point-in-time Core AI and AI ecosystem membership, ambiguity handling, announcement cutoffs, and required audit outputs. This is a specification change only; no raw data or model results were modified.

### AI pool point-in-time membership builder (2026-09-10)

- Codex reviewed the existing membership builder/config against `AI_POOL_SPEC.md` v1.0 and the raw run `data/raw/ai_factor_v1/20260909T183801359195Z/`. The builder was corrected to read only the three configured families (`segments_fy0_*`, `is_dates_*`, `f_tot_revenue_*`) while recording 261 co-located but unrequested CSVs as ignored inventory; raw collector family IDs are explicitly mapped in config and all request/meta/CSV hashes plus `ReportingState=Orig` are rechecked.
- Codex added independent-word-boundary matching for the frozen `AI` taxonomy term and corrected the reconciliation fallback for a non-finite unique null-total candidate. The run directory is created only after classification/input checks; summary hash indexing excludes self-referential summary/gate hashes.
- Final dry-run `data/clean/ai_pool_v1/20260910T023105114403Z/` and execute run `data/clean/ai_pool_v1/20260910T023221976393Z/` completed on the full 96 configured raw files. Execute output contains 37,865 audit rows and 9,135 membership rows across 781 literal RICs: 16 Core rows, 53 Ecosystem rows, 8,916 ambiguous rows and 150 membership-unknown rows; Core/Ecosystem overlap is 2 rows. The audit CSV is stored under the configured `data/audit/ai_pool_v1/` run directory. The independent post-run checks found no known-key duplicates, no cutoff violations, no member rows missing keywords, and no missing audit raw provenance.
- Final execute gate passed 15/15 checks; raw inputs were unchanged. No deletion, company exclusion, imputation, forward/backward fill, winsorization, identifier mapping, future-return/label/model/portfolio read or generation was performed. Summary/gate and output hashes are recorded in the run-scoped artifacts and `DATA_PROCESSING_LOG.md`.

### AI pool controls-only model-ready layer (20260910T030424673919Z)

- OpenAI Codex created and ran `build_ai_pool_baseline_model_ready_v1.py` against the latest membership run `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv` plus the existing clean returns/prices and a newly completed clean AI-factor controls snapshot. The output is `data/model_ready_ai_pool_v1/20260910T030424673919Z` with `678` active AI-pool security-month rows across `9` literal RICs and `138` formation months.
- Membership is applied point-in-time using the latest known dated status and corrected span interval; ambiguous/unknown source states are retained in `membership_snapshot_audit.csv` and are not carried forward as members. No name-based RIC mapping, company merge, delisted exclusion, imputation, forward fill, winsorization or event study was used.
- Development labels were written only for training/validation rows (`405` rows). Test rows (`273`) have no target values; test seal check=`True`. Formal fundamental controls remain blocked because clean inputs report unit/currency/market-cap semantics as UNVERIFIED; candidate ratios are retained and tagged for review.

### AI pool controls-only model-ready layer (20260910T030458992545Z)

- OpenAI Codex created and ran `build_ai_pool_baseline_model_ready_v1.py` against the latest membership run `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv` plus the existing clean returns/prices and a newly completed clean AI-factor controls snapshot. The output is `data/model_ready_ai_pool_v1/20260910T030458992545Z` with `678` active AI-pool security-month rows across `9` literal RICs and `138` formation months.
- Membership is applied point-in-time using the latest known dated status and corrected span interval; ambiguous/unknown source states are retained in `membership_snapshot_audit.csv` and are not carried forward as members. No name-based RIC mapping, company merge, delisted exclusion, imputation, forward fill, winsorization or event study was used.
- Development labels were written only for training/validation rows (`405` rows). Test rows (`273`) have no target values; test seal check=`True`. Formal fundamental controls remain blocked because clean inputs report unit/currency/market-cap semantics as UNVERIFIED; candidate ratios are retained and tagged for review.

### AI pool controls-only baseline models (20260910T030948780628Z)

- OpenAI Codex fit controls-only Logistic and fixed-parameter Random Forest using `data/model_ready_ai_pool_v1/20260910T030458992545Z/model_features.csv` plus training/validation labels from `data/model_ready_ai_pool_v1/20260910T030458992545Z/targets_dev.csv`. Training rows=242; validation rows=142; no event study input was used.
- Preprocessing was fit on training rows only with median imputation and missing indicators; Logistic additionally used StandardScaler. Validation metrics and probability rankings are in `data/model_runs/ai_pool_baseline_v1/20260910T030948780628Z`. Test target rows were not opened, and test predictions/metrics are zero by construction.

### AI pool controls-only baseline models (20260910T031100000000Z)

- OpenAI Codex fit controls-only Logistic and fixed-parameter Random Forest using `data/model_ready_ai_pool_v1/20260910T030458992545Z/model_features.csv` plus training/validation labels from `data/model_ready_ai_pool_v1/20260910T030458992545Z/targets_dev.csv`. Training rows=242; validation rows=142; no event study input was used.
- Preprocessing was fit on training rows only with median imputation and missing indicators; Logistic additionally used StandardScaler. Validation metrics and probability rankings are in `data/model_runs/ai_pool_baseline_v1/20260910T031100000000Z`. Test target rows were not opened, and test predictions/metrics are zero by construction.

### AI pool baseline independent validation (20260910T031200000000Z)

- OpenAI Codex independently recomputed development validation metrics and checked the model artifacts for model run `data/model_runs/ai_pool_baseline_v1/20260910T031100000000Z`. No test target values were read; audit output is `data/audit/ai_pool_baseline_v1/20260910T031200000000Z/summary.json`.

### AI pool controls-only model-ready layer (20260910T031500000000Z)

- OpenAI Codex created and ran `build_ai_pool_baseline_model_ready_v1.py` against the latest membership run `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv` plus the existing clean returns/prices and a newly completed clean AI-factor controls snapshot. The output is `data/model_ready_ai_pool_v1/20260910T031500000000Z` with `678` active AI-pool security-month rows across `9` literal RICs and `138` formation months.
- Membership is applied point-in-time using the latest known dated status and corrected span interval; ambiguous/unknown source states are retained in `membership_snapshot_audit.csv` and are not carried forward as members. No name-based RIC mapping, company merge, delisted exclusion, imputation, forward fill, winsorization or event study was used.
- Development labels were written only for training/validation rows (`405` rows). Test rows (`273`) have no target values; test seal check=`True`. Formal fundamental controls remain blocked because clean inputs report unit/currency/market-cap semantics as UNVERIFIED; candidate ratios are retained and tagged for review.

### AI pool controls-only baseline models (20260910T031600000000Z)

- OpenAI Codex fit controls-only Logistic and fixed-parameter Random Forest using `data/model_ready_ai_pool_v1/20260910T031500000000Z/model_features.csv` plus training/validation labels from `data/model_ready_ai_pool_v1/20260910T031500000000Z/targets_dev.csv`. Training rows=242; validation rows=142; no event study input was used.
- Preprocessing was fit on training rows only with median imputation and missing indicators; Logistic additionally used StandardScaler. Validation metrics and probability rankings are in `data/model_runs/ai_pool_baseline_v1/20260910T031600000000Z`. Test target rows were not opened, and test predictions/metrics are zero by construction.

### AI pool baseline independent validation (20260910T031700000000Z)

- OpenAI Codex independently recomputed development validation metrics and checked the model artifacts for model run `data/model_runs/ai_pool_baseline_v1/20260910T031600000000Z`. No test target values were read; audit output is `data/audit/ai_pool_baseline_v1/20260910T031700000000Z/summary.json`.

### AI pool controls-only model-ready layer (20260910T031900000000Z)

- OpenAI Codex created and ran `build_ai_pool_baseline_model_ready_v1.py` against the latest membership run `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv` plus the existing clean returns/prices and a newly completed clean AI-factor controls snapshot. The output is `data/model_ready_ai_pool_v1/20260910T031900000000Z` with `678` active AI-pool security-month rows across `9` literal RICs and `138` formation months.
- Membership is applied point-in-time using the latest known dated status and corrected span interval; ambiguous/unknown source states are retained in `membership_snapshot_audit.csv` and are not carried forward as members. No name-based RIC mapping, company merge, delisted exclusion, imputation, forward fill, winsorization or event study was used.
- Development labels were written only for training/validation rows (`405` rows). Test rows (`273`) have no target values; test seal check=`True`. Formal fundamental controls remain blocked because clean inputs report unit/currency/market-cap semantics as UNVERIFIED; candidate ratios are retained and tagged for review.

### AI pool controls-only baseline models (20260910T032000000000Z)

- OpenAI Codex fit controls-only Logistic and fixed-parameter Random Forest using `data/model_ready_ai_pool_v1/20260910T031900000000Z/model_features.csv` plus training/validation labels from `data/model_ready_ai_pool_v1/20260910T031900000000Z/targets_dev.csv`. Training rows=242; validation rows=142; no event study input was used.
- Preprocessing was fit on training rows only with median imputation and missing indicators; Logistic additionally used StandardScaler. Validation metrics and probability rankings are in `data/model_runs/ai_pool_baseline_v1/20260910T032000000000Z`. Test target rows were not opened, and test predictions/metrics are zero by construction.

### AI pool baseline independent validation (20260910T032100000000Z)

- OpenAI Codex independently recomputed development validation metrics and checked the model artifacts for model run `data/model_runs/ai_pool_baseline_v1/20260910T032000000000Z`. No test target values were read; audit output is `data/audit/ai_pool_baseline_v1/20260910T032100000000Z/summary.json`.

### AI pool baseline independent validation (20260910T032200000000Z)

- OpenAI Codex independently recomputed development validation metrics and checked the model artifacts for model run `data/model_runs/ai_pool_baseline_v1/20260910T032000000000Z`. No test target values were read; audit output is `data/audit/ai_pool_baseline_v1/20260910T032200000000Z/summary.json`.

### AI pool membership taxonomy v1.1 dry-run (20260910T035952879271Z)

- OpenAI Codex synchronized `ai_pool_membership_pit_v1_config.json` and `build_ai_pool_membership_pit_v1.py` with the user-preregistered `AI_POOL_SPEC.md` v1.1 broad taxonomy: compute/accelerator, HBM/AI memory, AI server/network/data-center, cloud/software platform, robotics/autonomy, and power/cooling/equipment. Generic words remain ambiguous unless an explicit preregistered phrase is present; standalone `AI` uses strict case-insensitive `\\bAI\\b` matching.
- The builder retains Core and legacy ecosystem evidence (`matched_keywords`, `broad_category_hits`, `legacy_ecosystem_hits`, reason codes, and Orig announcement-date cutoff fields) and compares the eligible literal-RIC union to the hash-verified v1.0 baseline `20260910T023221976393Z`. It does not read or create future returns, labels, predictions, model outputs, portfolios, or event-study data.
- Final full dry-run output is `data/clean/ai_pool_v1/20260910T035952879271Z/` with audit directory `data/audit/ai_pool_v1/20260910T035952879271Z/`; it read 96 configured Orig raw files (57,058 raw rows; 37,865 segment rows) and wrote no membership/audit tables. Counts were 16 Core rows, 53 broad rows, 56 legacy-ecosystem-match rows across 7 instruments, 8,916 ambiguous rows, 150 membership-unknown rows, 0 new RICs vs v1.0, and 10 overlapping eligible RICs. All six broad categories are present in the summary, including zero-count HBM/memory and power/cooling categories.
- Final hashes: spec `7d2cb81904d1102ae5f8f5f6a6729deda4ba9c31b69a9fba33eb416d0acd77e3`; config `0b1ba515f44a953f55de5e243045d0bcbd130eac70784b46b353307a7c09f5ee`; builder `b93be43b74f353bd436c59112c978c5425983c1e07c1cc6f142dac493b91ad63`; dry-run stats `eb30d3e6f4dc1982997a5be6c81bbb6ef322b9c8a59cdb44629c25ecaea95d90`; summary `5c081484c8ac835806769eb923505b1ddf8adac938a78ace37d049c08c085dee`; gate `2604fe1e3b7a0784784c83782c6afb5662a24db0cb9446180ca5deeba0b55780`.
- Raw and sidecar hashes were independently rechecked, raw before/after hashes were equal, all 19 dry-run gate checks passed, and the prior v1.0 run was preserved unchanged. No rows were deleted, quarantined, deduplicated, imputed, winsorized, or identifier-mapped.

### AI pool daily controls-only model-ready layer (20260910T044900000000Z)

- OpenAI Codex built a new daily model-ready run `data/model_ready_ai_pool_daily_v1/20260910T044900000000Z` from `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv`, clean returns and clean prices. Current validation uses `10` requested literal RICs, `9` active RICs, `14133` daily pool rows and H21 labels only for development splits.
- Daily controls end at the prior SPY session; the H21 target is future stock cumulative return minus future SPY cumulative return. Test future returns are not looked up or labeled; candidate list can be supplied later with --candidate-list without changing monthly runs.

### AI pool daily controls-only baseline models (20260910T045000000000Z)

- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `data/model_ready_ai_pool_daily_v1/20260910T044900000000Z`. Training anchors=243; validation anchors=150; validation dates are chronological and no event study input was used.
- Preprocessing was fit on non-overlapping training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `data/model_runs/ai_pool_daily_baseline_v1/20260910T045000000000Z`.

### AI supply-chain candidate expansion (20260910T045000Z)

- OpenAI Codex inspected the existing 781-RIC corrected universe, AI pool v1 outputs, LSEG raw segment/announcement families, and the project ETF files. The inspection found 781 unique literal RICs, 10 existing AI-pool member RICs, and ETF price files for AIQ/BOTZ/IGV/SOXX/XLK without constituent holdings.
- OpenAI Codex authored `AI_SUPPLY_CHAIN_TAXONOMY_v2.md`, `ai_pool_expansion_v1_config.json`, `build_ai_pool_expansion_candidates_v1.py`, and `validate_ai_pool_expansion_candidates_v1.py`. The frozen design contains seven supply-chain groups and 49 candidate RICs selected by documented business role, with no return/target/label/model-output inputs.
- OpenAI Codex executed `data/audit/ai_pool_expansion_v1/20260910T045000Z`. It retained all 781 screening rows, selected 49 design candidates, retained 732 not-selected rows with a reason code, retained the one delisted candidate JNPR.N^G25, and physically deleted zero rows. Nine candidates have local LSEG segment-keyword plus Orig announcement-date evidence; 40 are marked provisional static-source candidates because issuer pages were reference-only and not downloaded.
- OpenAI Codex executed the independent validator; `independent_validation.json` reports `independently_verified`. The validator confirmed candidate/config RIC equality, row counts, group counts, raw hashes, summary/gate hashes, and zero reads of returns/targets/labels/model outputs. These artifacts are design/audit outputs and do not claim a completed 49-RIC PIT membership or backtest.
- OpenAI Codex created `data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_instruments_49ric.csv` as a mechanical column-name derivative (`ric` copied to `Instrument`) for the existing daily builder interface. It contains the same 49 literal RICs and does not alter membership eligibility or PIT evidence status; SHA-256=`e93a5164db123078c6cf826e12ea17a4b3a18ebdfd6a3dc1e2a937a079a99088`.

### AI pool daily baseline independent validation (20260910T045200000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T045000000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T045200000000Z/summary.json`.

### AI pool daily controls-only model-ready layer (20260910T050000000000Z)

- OpenAI Codex built a new daily model-ready run `data/model_ready_ai_pool_daily_v1/20260910T050000000000Z` from `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv`, clean returns and clean prices. Current validation uses `10` requested literal RICs, `9` active RICs, `14133` daily pool rows and H21 labels only for development splits.
- Daily controls end at the prior SPY session; the H21 target is future stock cumulative return minus future SPY cumulative return. Test future returns are not looked up or labeled; candidate list can be supplied later with --candidate-list without changing monthly runs.

### AI pool daily controls-only baseline models (20260910T051000000000Z)

- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `data/model_ready_ai_pool_daily_v1/20260910T050000000000Z`. Training anchors=243; validation anchors=150; validation dates are chronological and no event study input was used.
- Preprocessing was fit on non-overlapping training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `data/model_runs/ai_pool_daily_baseline_v1/20260910T051000000000Z`.

### AI pool daily baseline independent validation (20260910T052000000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T051000000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T052000000000Z/summary.json`.

### AI pool daily controls-only baseline models (20260910T053000000000Z)

- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `data/model_ready_ai_pool_daily_v1/20260910T050000000000Z`. Training anchors=243; validation anchors=150; validation dates are chronological and no event study input was used.
- Preprocessing was fit on non-overlapping training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `data/model_runs/ai_pool_daily_baseline_v1/20260910T053000000000Z`.

### AI pool daily controls-only baseline models (20260910T053100000000Z)

- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `data/model_ready_ai_pool_daily_v1/20260910T050000000000Z`. Training anchors=243; validation anchors=150; validation dates are chronological and no event study input was used.
- Preprocessing was fit on non-overlapping training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `data/model_runs/ai_pool_daily_baseline_v1/20260910T053100000000Z`.

### AI pool daily baseline independent validation (20260910T054100000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T053100000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T054100000000Z/summary.json`.

### AI pool daily controls-only model-ready layer (20260910T052000000000Z)

- OpenAI Codex built a new daily model-ready run `data/model_ready_ai_pool_daily_v1/20260910T052000000000Z` from `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv`, clean returns and clean prices. Current validation uses `49` requested literal RICs, `9` active RICs, `14133` daily pool rows and H21 labels only for development splits.
- Daily controls end at the prior SPY session; the H21 target is future stock cumulative return minus future SPY cumulative return. Test future returns are not looked up or labeled; candidate list can be supplied later with --candidate-list without changing monthly runs.

### AI pool daily controls-only model-ready layer (20260910T052100000000Z)

- OpenAI Codex built a new daily model-ready run `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z` from `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv`, clean returns and clean prices. Current validation uses `49` requested literal RICs, `49` active RICs, `110829` daily pool rows and H21 labels only for development splits. Membership mode=`exploratory_static_candidate`; PIT status=`not_full_pit_provisional_static_candidate_registry`.
- Daily controls end at the prior SPY session; the H21 target is future stock cumulative return minus future SPY cumulative return. Test future returns are not looked up or labeled; registry AI-role evidence is retained in metadata and provisional static candidates must not be treated as a complete PIT backtest.

### AI pool daily controls-only model-ready layer (20260910T060000000000Z)

- OpenAI Codex built a new daily model-ready run `data/model_ready_ai_pool_daily_v1/20260910T060000000000Z` from `data/clean/ai_pool_v1/20260910T023221976393Z/ai_pool_membership_pit.csv`, clean returns and clean prices. Current validation uses `49` requested literal RICs, `49` active RICs, `110829` daily pool rows and H21 labels only for development splits. Membership mode=`exploratory_static_candidate`; PIT status=`not_full_pit_provisional_static_candidate_registry`.
- Daily controls end at the prior SPY session; the H21 target is future stock cumulative return minus future SPY cumulative return. Test future returns are not looked up or labeled; registry AI-role evidence is retained in metadata and provisional static candidates must not be treated as a complete PIT backtest.

### AI pool daily controls-only baseline models (20260910T052200000000Z)

- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `data/model_ready_ai_pool_daily_v1/20260910T050000000000Z`. Training anchors=243; validation anchors=150; validation dates are chronological and no event study input was used.
- Preprocessing was fit on non-overlapping training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `data/model_runs/ai_pool_daily_baseline_v1/20260910T052200000000Z`.

### AI pool daily controls-only baseline models (20260910T052300000000Z)

- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z` using `conservative_non_overlap_anchors`. Training anchors=2334; validation anchors=992; validation dates are chronological and no event study input was used.
- Preprocessing was fit on selected training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `data/model_runs/ai_pool_daily_baseline_v1/20260910T052300000000Z`.

### AI pool daily controls-only baseline models (20260910T052400000000Z)

- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z` using `all_daily_with_date_purge`. Training anchors=48821; validation anchors=20747; validation dates are chronological and no event study input was used.
- Preprocessing was fit on selected training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `data/model_runs/ai_pool_daily_baseline_v1/20260910T052400000000Z`.

### AI pool daily baseline independent validation (20260910T052500000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T052300000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T052500000000Z/summary.json`.

### AI pool daily baseline independent validation (20260910T052600000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T052400000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T052600000000Z/summary.json`.

### AI pool daily baseline independent validation (20260910T052700000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked `conservative_non_overlap_anchors` training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T052300000000Z` using model-ready `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T052700000000Z/summary.json`.

### AI pool daily baseline independent validation (20260910T052800000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked `all_daily_with_date_purge` training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T052400000000Z` using model-ready `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T052800000000Z/summary.json`.

### AI pool daily baseline independent validation (20260910T061500000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked `conservative_non_overlap_anchors` training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T052300000000Z` using model-ready `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T061500000000Z/summary.json`.

### AI pool daily baseline independent validation (20260910T061600000000Z)

- OpenAI Codex independently recomputed the daily H21 validation metrics and checked `all_daily_with_date_purge` training-only imputer statistics and model artifacts for `data/model_runs/ai_pool_daily_baseline_v1/20260910T052400000000Z` using model-ready `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z`. No test target values were read; audit output is `data/audit/ai_pool_daily_baseline_v1/20260910T061600000000Z/summary.json`.

### Daily audit supersession notice (20260910T061700000000Z)

- Early audits `20260910T052500000000Z` and `20260910T052600000000Z` are superseded because their validator used the wrong hard-coded model-ready path. Replacement audits `20260910T061500000000Z` and `20260910T061600000000Z` derive the model-ready path from each model summary and independently verify successfully. Historical files remain preserved.

### FRED daily macro raw collection and independent validation (20260910T062000000000Z)

OpenAI Codex implemented and ran a raw-only collector against the official FRED `fredgraph.csv` endpoint for VIXCLS, DGS3MO, DGS2, DGS10, BAMLH0A0HYM2, BAMLC0A0CM, and DTWEXBGS over 2013-11-01 through 2026-09-07. The designated raw run `data/raw/daily_macro_v1/20260910T061957615542Z/` returned 7/7 HTTP 200 responses; URL, retrieval time, content type, bytes, SHA-256, and attempt history are preserved in the manifest. Two earlier interrupted config-only directories remain preserved and were not used. An independent raw validator checked 75 transport/schema/date/value checks, including exact rehashes and provider missing-token counts; all passed. No prices, test targets, model results, or downstream features were read, and no missing value was imputed or transformed. FRED current CSV is not a complete vintage database, so revision and release-time limitations remain. Scripts: `collect_daily_macro_v1.py` SHA-256 `7046dfc2f78c394d9d023ecb31731cb561158d1f728e6556d995bf4590621581`; `validate_daily_macro_collection_v1.py` SHA-256 `5f30478babd04b710c575201cde0f50929e6a2d3f90e61f76691c57965b18c06`. Group review pending before downstream feature use.

### FRED daily macro feature build and independent validation (20260910T063100000000Z)

OpenAI Codex built `data/clean/daily_macro_v1/20260910T062700000000Z/` from the designated raw FRED run `data/raw/daily_macro_v1/20260910T061957615542Z/` and the raw SPY benchmark date column. The complete table has 3,228 SPY sessions; `macro_features_train_valid.csv` has 2,014 rows for 2015–2022. The table retains raw source tokens, latest source date, effective valid observation date, first available date, calendar/session carry ages and reason codes. All engineered features use the prior SPY session, with no imputation or future fill. BAMLH0A0HYM2 and BAMLC0A0CM first become valid on 2023-09-11, so their train/validation values are unavailable; their columns remain auditable in the full table while six credit features are excluded from the recommended train/validation list under `TRAINING_COVERAGE_UNAVAILABLE`. An independent validator reconstructed the as-of values and arithmetic from raw bytes and passed 158/158 checks; audit output is `data/audit/daily_macro_v1/20260910T063000000000Z/summary.json`. No labels, returns, test targets, model results, predictions, metrics, or artifacts were read. FRED current CSV is not a complete vintage database and lacks historical release timestamps; revision and timing limitations remain. A first audit serialization failure at `data/audit/daily_macro_v1/20260910T062854896764Z/` is preserved and superseded by the passing audit. Scripts: `build_daily_macro_features_v1.py` SHA-256 `b8f059b95e5f8840eb1bd7e10168bad6637f6341a9d6f4938a9b4c6f20bf1afa`; `validate_daily_macro_features_v1.py` SHA-256 `36a7a081bc27a390886e00e5c2d5ad8c2e1abeb6b88b6c38adfed7e11fb97446`. Group review pending for downstream model integration.

### AI industry state features v1 (20260910T144000000000Z)

OpenAI Codex implemented and ran `build_ai_daily_macro_features_v1.py` against the formal 49-RIC expanded daily input `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/model_features.csv`, clean SPY/pool prices, and raw SOXX/XLK/AIQ/BOTZ/IGV prices. The run generated `data/clean/ai_daily_macro_features_v1/20260910T144000000000Z/ai_state_features.csv` (2,889 formation-session rows), `pool_member_coverage.csv` (110,829 member rows), `etf_coverage.csv` (14,445 ETF-session rows), a summary and hash manifest; all features use the strictly prior SPY session, and no targets, labels, future returns, or model-run files were read. ETF history before inception remains NA with reason codes; no imputation, forward/backward fill, winsorization, or source row deletion occurred. Pool low-coverage sessions were zero because valid active-member counts stayed at least 27; member-level missing MA60/return coverage remains explicitly recorded. `validate_ai_daily_macro_features_v1.py` independently rechecked hashes, keys, lag ordering, pool denominators, breadth/dispersion/equal-weight recomputations, and input immutability; status=`independently_verified`, failed checks=`0`. Config SHA-256=`f73a87b187ce7b2525a8cc9234f27442ed7b3d11d7f461d49d4b969ad986b528`; builder SHA-256=`d98adc84ea5f6dd6ed39c7e4b2f701b1cad32c17f89b659f389823bc5ce18f59`; final feature table SHA-256=`2c8a196263ffb09351caa461942de23764191dab0ce5a7d8f4db307e4544649c`.

### Enhanced AI-pool daily state models (20260910T064229538382Z)

- OpenAI Codex created and ran `run_ai_pool_daily_state_models_v1.py` for four feature groups: technical_only, technical_plus_macro, technical_plus_ai_state and full_state. Inputs were the model-ready `20260910T052100000000Z`, macro `20260910T062700000000Z` and independently verified AI state `20260910T144000000000Z` runs.
- The primary sample used 21-session non-overlap anchors. RF hyperparameters were selected only from purged training walk-forward folds; validation labels were used only for the final comparison. The sealed test-target file was never opened and no test predictions or metrics were produced.
- Median imputation and missing indicators were fit within each training fold; AIQ/BOTZ/IGV pre-listing missingness was retained and reported. Credit-spread six-column macro features were excluded. AI-state technical duplicates and the `spy_return_1` alias were dropped before merging, as recorded in the run summary. Outputs: `data/model_runs/ai_pool_daily_state_models_v1/20260910T064229538382Z`.

### Enhanced daily input audit (20260910T065044522242Z)

OpenAI Codex created and ran the read-only audit script `audit_ai_daily_enhanced_inputs_v1.py` (v1.0.0, SHA-256 `4480db27137d99e129c58e04c3c68f05de093eec0ece2fa43e1df7b1eaec9d76`) for model-ready `20260910T052100000000Z`, macro `20260910T062700000000Z`, and AI state `20260910T144000000000Z`. The audit checked literal key uniqueness, strict prior-SPY alignment, development/test date separation, target/future-name guards, primary non-overlap anchor missingness, and duplicate controls. It did not open `targets_test_sealed.csv`, `targets_dev.csv`, model runs, predictions, or metrics, and it made no input changes or model fits.

The audit output is `data/audit/ai_daily_enhanced_inputs_v1/20260910T065044522242Z/` with status `PASS` (32 PASS, 3 WARN, 2 non-blocking FAIL, 0 blocking failures). Safe candidates are 20 macro features and 21 incremental AI-state features after excluding six all-missing development credit-spread fields, 18 high-missing ETF history fields, and the duplicated/source-inconsistent SPY controls. Full missingness, exclusions, hashes, and limitations are recorded in `summary.json`, `checks.csv`, `anchor_feature_missingness.csv`, and `feature_inventory.csv`.

### Enhanced AI-pool daily state models (20260910T065216293899Z)

- OpenAI Codex created and ran `run_ai_pool_daily_state_models_v1.py` for four feature groups: technical_only, technical_plus_macro, technical_plus_ai_state and full_state. Inputs were the model-ready `20260910T052100000000Z`, macro `20260910T062700000000Z` and independently verified AI state `20260910T144000000000Z` runs.
- The primary sample used 21-session non-overlap anchors. RF hyperparameters were selected only from purged training walk-forward folds; validation labels were used only for the final comparison. The sealed test-target file was never opened and no test predictions or metrics were produced.
- Median imputation and missing indicators were fit within each training fold; AIQ/BOTZ/IGV pre-listing missingness was retained and reported. Credit-spread six-column macro features were excluded. AI-state technical duplicates and the `spy_return_1` alias were dropped before merging, as recorded in the run summary. Outputs: `data/model_runs/ai_pool_daily_state_models_v1/20260910T065216293899Z`.

### Enhanced AI-pool model independent validation (20260910T064727226020Z)

- OpenAI Codex created and ran `validate_ai_pool_daily_state_models_v1.py` (SHA-256=`ad8441ce0940454e30cb96e52c3390112c1607419051169104dd558f854c40e1`) as a read-only audit of model run `data/model_runs/ai_pool_daily_state_models_v1/20260910T064229538382Z`. The audit passed `72/72` checks with `0` failures and wrote `data/audit/ai_pool_daily_state_models_v1/20260910T064727226020Z/validation.json` and `validation.csv`.
- The audit independently checked the formal 49-RIC model-ready input path `20260910T052100000000Z`, source/output hashes, `992` validation predictions, four feature sets × two models, persisted validation metric-table structure, RF selected parameters and purged training fold evidence, strict `last_fit_exit < validation_start`, feature duplicate/alias exclusions, and six credit-spread exclusions. It did not open, read, hash, or aggregate `targets_test_sealed.csv`, and it produced no test predictions or metrics.
- Earlier audits `20260910T064619260294Z` and `20260910T064652127085Z` are superseded by the passing `20260910T064727226020Z` audit; their files remain preserved for provenance.

### Enhanced AI-pool daily state models (20260910T070105846190Z)

- OpenAI Codex created and ran `run_ai_pool_daily_state_models_v1.py` for four feature groups: technical_only, technical_plus_macro, technical_plus_ai_state and full_state. Inputs were the model-ready `20260910T052100000000Z`, macro `20260910T062700000000Z` and independently verified AI state `20260910T144000000000Z` runs.
- The primary sample used 21-session non-overlap anchors. RF hyperparameters were selected only from purged training walk-forward folds; validation labels were used only for the final comparison. The sealed test-target file was never opened and no test predictions or metrics were produced.
- Median imputation and missing indicators were fit within each training fold; AIQ/BOTZ/IGV pre-listing features were excluded by the independently audited coverage-safe registry; remaining structural missingness is retained. Credit-spread six-column macro features were excluded. AI-state technical duplicates and the `spy_return_1` alias were dropped or excluded upstream, as recorded in the run summary. Outputs: `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z`.

### Enhanced AI-pool daily state models (20260910T070000000000Z)

- OpenAI Codex created and ran `run_ai_pool_daily_state_models_v1.py` for four feature groups: technical_only, technical_plus_macro, technical_plus_ai_state and full_state. Inputs were the model-ready `20260910T052100000000Z`, macro `20260910T062700000000Z` and independently verified AI state `20260910T144000000000Z` runs.
- The primary sample used 21-session non-overlap anchors. RF hyperparameters were selected only from purged training walk-forward folds; validation labels were used only for the final comparison. The sealed test-target file was never opened and no test predictions or metrics were produced.
- Median imputation and missing indicators were fit within each training fold; AIQ/BOTZ/IGV pre-listing features were excluded by the independently audited coverage-safe registry; remaining structural missingness is retained. Credit-spread six-column macro features were excluded. AI-state technical duplicates and the `spy_return_1` alias were dropped or excluded upstream, as recorded in the run summary. Outputs: `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z`.

### Model-run role clarification (20260910T070500000000Z)

- Primary result: `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/`, independently validated at `data/audit/ai_pool_daily_state_models_v1_1/20260910T070500000000Z/` with 74/74 checks passing.
- `20260910T064229538382Z` and `20260910T065216293899Z` remain preserved as `expanded_state_robustness` runs with superseded markers. The separately produced `20260910T070105846190Z` coverage-safe run and its audit remain supplemental reproducibility records; no baseline or source file was overwritten.
### AI产业链日频宏观与行业状态模型结果说明（2026-09-10）

- OpenAI Codex 汇总并解释经独立核验的宏观、AI行业状态和覆盖一致日频模型结果，生成 `AI_DAILY_STATE_MODEL_RESULTS_v1.md`。文档使用既有 run artifacts 中的样本计数、验证指标、数据处理规则和限制；没有打开测试目标，也没有新增模型选择或测试期结论。

### AI portfolio pre-registration timing review（2026-09-10）

- OpenAI Codex 审阅 `AI_PORTFOLIO_BACKTEST_SPEC_v1.md`，将执行时点纠正为形成日前一交易日特征、形成日收盘成交，与 model-ready 的 `entry_session=formation_session` 标签一致；风险协方差改为至少 100 个共同完整历史交易日，未解决的持仓标记会使 NAV run 失败而非静默删日。未执行组合回测，也未读取测试目标。

### AI portfolio backtest pre-registration specification v1 (2026-09-10T10:46:47.142579Z)

- OpenAI Codex read the project protocol, AI pool specification, frozen `AI_SUPPLY_CHAIN_TAXONOMY_v2.0`, 49-RIC candidate registry, and the summary/audit/config artifacts for the frozen `technical_plus_ai_state/logistic` model run `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z`. The model artifact, registry, taxonomy, and upstream input versions are recorded in `AI_PORTFOLIO_BACKTEST_SPEC_v1.md`.
- Codex pre-registered a fixed portfolio mapping with the long-only AI stock sleeve plus SPY beta hedge as the primary strategy and the top/bottom market-neutral sleeve as a secondary registered variant. Fixed choices include all-daily inference from the frozen training artifact, 21-SPY-session formation/holding blocks and rebalance, top/bottom quintile ranking, confidence and rank/turnover buffers, seven supply-chain group caps, single-name caps, price/liquidity/quote gates, zero-beta and 10% volatility targets, spread-based execution costs, a stated paper borrow assumption for the secondary short leg, delisting/corporate-action handling, and missing-prediction rules.
- No test target file was opened, parsed, aggregated, scored, ranked, or predicted. No portfolio backtest, return calculation, performance metric, or test-period model selection was performed. No raw data, model artifact, candidate registry, or prior audit output was modified; only the new specification and this cumulative log entry were written. The exploratory-static membership limitation (9 direct-local-PIT candidates and 40 provisional static candidates) and absent point-in-time borrow inventory are explicitly disclosed for later review.

### AI portfolio validation-input audit (20260910T100300Z)

- OpenAI Codex performed a read-only, run-scoped audit at `data/audit/ai_portfolio_inputs_v1/20260910T100300Z` using the 49-RIC candidate registry, clean v3 prices, daily model-ready metadata/eligibility and archived stock-split records. It covered `21671` validation schedule rows and `992` non-overlapping supervised anchor rows; no validation prediction was read or used, and no sealed test target was read.
- The audit retained missing close/volume/quote/dollar-volume rows and model-ready non-anchor rows with stable reason codes, retained the single delisted candidate's history, and found `6` candidate split events in 2021–2022. It proposed a spread-plus-square-root-participation cost rule for later review; no strategy, portfolio or performance result was produced.

### Primary AI daily validation diagnostics (20260910T104859915072Z)

- OpenAI Codex implemented and ran `diagnose_ai_daily_primary_v1.py` (analysis version `1.0.0`, SHA-256=`9d1ad0ad1460a054c2d333e07f88c6033491b06db34898c0f2dddbad3c9980b0`) against the designated validation predictions from model run `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z` and its corresponding `targets_dev.csv`. The run-scoped copy, `summary.json`, `summary.md`, and all CSV tables are under `data/analysis/ai_daily_primary_diagnostics_v1/20260910T070000000000Z`.
- The analysis reconciled 992 validation predictions one-to-one to development targets (71,619 target rows loaded; 44 instruments; 89 dates), computed 2021/2022 and quarterly diagnostics, ten fixed probability calibration bins/ECE, and 2,000 paired date-cluster bootstrap replicates per scope/comparator with fixed seed base 5360. It focused on the pre-specified `technical_plus_ai_state/logistic` and compared against technical-only Logistic and Random Forest without threshold tuning or model selection.
- The diagnostic deliberately did not open, read, aggregate, score, rank, or predict any sealed test target. No fitted model artifact or source data was modified. Results remain development validation comparisons; only 23 of 89 dates have usable within-date rank IC, 29 dates have at least two securities for a descriptive spread, and zero dates meet a ten-per-side gate. The upstream exploratory static/provisional PIT membership limitation remains applicable.

### AI portfolio validation-input audit (20260910T100500Z)

- OpenAI Codex performed a read-only, run-scoped audit at `data/audit/ai_portfolio_inputs_v1/20260910T100500Z` using the 49-RIC candidate registry, clean v3 prices, daily model-ready metadata/eligibility and archived stock-split records. It covered `21671` validation schedule rows and `992` non-overlapping supervised anchor rows; no validation prediction was read or used, and no sealed test target was read.
- The audit retained missing close/volume/quote/dollar-volume rows and model-ready non-anchor rows with stable reason codes, retained the single delisted candidate's history, and found `6` candidate split events in 2021–2022. It proposed a spread-plus-square-root-participation cost rule for later review; no strategy, portfolio or performance result was produced.

### AI portfolio validation-input audit (20260910T110000000000Z)

- OpenAI Codex performed a read-only, run-scoped audit at `data/audit/ai_portfolio_inputs_v1/20260910T110000000000Z` using the 49-RIC candidate registry, clean v3 prices, daily model-ready metadata/eligibility and archived stock-split records. It covered `21671` validation schedule rows and `992` non-overlapping supervised anchor rows; no validation prediction was read or used, and no sealed test target was read.
- The audit retained missing close/volume/quote/dollar-volume rows and model-ready non-anchor rows with stable reason codes, retained the single delisted candidate's history, and found `6` candidate split events in 2021–2022. It proposed a spread-plus-square-root-participation cost rule for later review; no strategy, portfolio or performance result was produced.

### AI portfolio validation-input audit (20260910T101500Z)

- OpenAI Codex performed a read-only, run-scoped audit at `data/audit/ai_portfolio_inputs_v1/20260910T101500Z` using the 49-RIC candidate registry, clean v3 prices, daily model-ready metadata/eligibility and archived stock-split records. It covered `21671` validation schedule rows and `992` non-overlapping supervised anchor rows; no validation prediction was read or used, and no sealed test target was read.
- The audit retained missing close/volume/quote/dollar-volume rows and model-ready non-anchor rows with stable reason codes, retained the single delisted candidate's history, and found `6` candidate split events in 2021–2022. It proposed a spread-plus-square-root-participation cost rule for later review; no strategy, portfolio or performance result was produced.


### AI portfolio validation backtest (20260910T113500000000Z)

- OpenAI Codex implemented and ran the frozen validation-only primary portfolio. It generated all-daily scores, 21-session targets, beta hedge, volatility scale, turnover cap, spread costs and daily NAV without opening sealed test targets. Outputs: `data/backtests/ai_portfolio_v1/20260910T113500000000Z`.


### Formal AI portfolio validation backtest (20260910T125000000000Z)

- OpenAI Codex implemented and ran the validation-only v1.1 portfolio with F-1 ranking gates, post-ranking execution checks, detailed audit outputs, transaction costs, beta hedge, volatility scaling and turnover controls. It did not open sealed test targets. Outputs: `data/backtests/ai_portfolio_v1/20260910T125000000000Z`.

### AI portfolio independent validation audits (20260910T121500000000Z, 20260910T122000000000Z, 20260910T130000000000Z)

- OpenAI Codex created `validate_ai_portfolio_validation_v1.py` and independently reconstructed the preliminary and formal validation-only ledgers from holdings, realised returns, trades and costs. It identified realised-weight drift and the preliminary omission of initial formation cost from daily risk metrics, then confirmed the formal run with 13 PASS, 1 non-blocking WARN and 0 FAIL. No sealed test target was opened.

### Failed formal portfolio environment attempt (20260910T124500000000Z)

- OpenAI Codex attempted the formal runner with system scikit-learn 1.7.2; the 1.9.0 model artifact was incompatible and execution stopped before an output directory was created. It then switched to the project `.venv` matching scikit-learn 1.9.0. No source data or sealed test targets were modified or opened.

### Current formal AI portfolio validation and audit (20260910T133000000000Z / 20260910T133500000000Z)

- OpenAI Codex reran the formal validation portfolio after clarifying target versus realised caps and adding software/version/output provenance. It then independently reconstructed the ledger and verified 14 checks: 13 PASS, one non-blocking realised-weight-drift WARN, and zero FAIL. It updated the readable results document and current-run pointer without opening sealed test targets or tuning a validation threshold.

### Technical-plus-macro Logistic baseline model card (2026-09-10)

- OpenAI Codex read the frozen validation metrics, feature registry, model configuration and fitted Logistic artifact for `technical_plus_macro/logistic`, then created `BASELINE_LOGISTIC_MODEL_CARD.md`. It identified the requested ROC-AUC as 0.561559, documented the 41 raw inputs, training-only preprocessing, time split, validation metrics and standardized coefficient interpretation. This was read-only analysis of existing development artifacts; no model was retrained and no sealed test target was opened.


### Formal AI portfolio validation backtest (20260910T133000000000Z)

- OpenAI Codex implemented and ran the validation-only v1.1 portfolio with F-1 ranking gates, post-ranking execution checks, detailed audit outputs, transaction costs, beta hedge, volatility scaling and turnover controls. It did not open sealed test targets. Outputs: `data/backtests/ai_portfolio_v1/20260910T133000000000Z`.

### Baseline report content extraction (2026-09-10T20:25:39+08:00)

- OpenAI Codex performed a read-only, retrospective extraction from the frozen 49-RIC candidate/taxonomy materials, the daily state model configuration, the validation metrics and the Logistic model card, then wrote `deliverables/BMF5360_Baseline_Package/support/report_content.md` using `apply_patch`. The draft records the seven taxonomy groups, 9 direct-local-PIT and 40 provisional-static candidates, 21 technical and 20 macro inputs, six excluded credit-spread fields, the H21 split, Pipeline parameters and exact validation comparisons including ROC-AUC `0.5615590709322795`.
- Candidate registry counts and `validation_metrics.csv` were cross-checked. No model was retrained, no source data or model artifact was modified, no test target was opened, and no data transformation or row deletion was performed. Output SHA-256: `868e696f4dbcca4be62e486862748c2a00cc1e883429f0c6ed523a75a0fe97d1`.

### Baseline deliverable code package (2026-09-10T20:26:40+08:00)

- OpenAI Codex created the isolated package `deliverables/BMF5360_Baseline_Package/` with a concise runner, locked configuration, pinned requirements and README for technical-only / technical-plus-macro Logistic and same-table Random Forest comparisons. The implementation is derived from the audited daily state runner and retains its H21 non-overlap, training-only purged RF tuning and training-fitted preprocessing rules.
- This was a packaging/static-validation stage only: Python AST and JSON parsing passed; no model was retrained, no data table was transformed, no source or model artifact was modified, and no sealed test target was opened or hashed. Runtime outputs, hashes and checks must be taken from the package run's own `summary.json` and `audit.json`.

### Baseline package naming alignment (2026-09-10T20:32:30+08:00)

- Codex aligned the packaged model artifact keys with the audited run's underscore filenames (`technical_only_logistic.joblib`, `technical_plus_macro_random_forest.joblib`). This was a source-code-only edit followed by AST/CLI static checks; no model or data was run and no sealed test target was opened.

### Submission artefact assembly and strategy assessment (2026-09-11)

- OpenAI Codex assembled an Excel calculation workbook and a Chinese Word explanation from already audited company-selection, factor-definition, Logistic model and validation artefacts. It also reviewed three user-provided Word documents about feasibility, package quality and an S&P 500 inclusion event strategy. The assessment distinguishes document claims from locally verified project artefacts; no claim in the event-strategy document was treated as independently reproduced.
- The workbook has 8 user-facing sheets and a formula-error scan with zero matches. The code package runner was corrected so its documented safe run-id can pass validation before data loading; Python syntax, configuration JSON and CLI help were checked. No model was retrained, no data rows were changed, and the sealed test target was not opened.

### Submission archive (2026-09-11)

- OpenAI Codex created and verified the 10-file submission ZIP for the baseline package. It excludes raw LSEG data, model inputs, test targets and temporary artefacts. No data processing or model execution occurred.

### Timing versus selection validation audit (2026-09-11)

- OpenAI Codex created and ran two validation-only audit scripts after the user raised concern that common SPY/macro variables could inflate pooled AUC. The scripts used saved validation predictions and development-only feature/calendar tables, performed 21-session calendar-block bootstrap diagnostics, isolated same-date AUC/rank IC, and checked validation-period spread missingness. No test target was read, no model was retrained or selected, and no source data was modified.

- OpenAI Codex then revised the baseline Excel and Word delivery wording so that pooled AUC is no longer described as verified stock-selection improvement. The revision cites the block-bootstrap and same-date diagnostics; it makes no new investment-performance claim.

### Report positioning consolidation (2026-09-11)

- OpenAI Codex created a report-positioning document that treats the AI supply-chain work as a completed pilot within a wider ML strategy-research framework. It copied the validation-only timing/selection audit code and result tables into the deliverable package for future report drafting. No data, model or test target was changed or opened.

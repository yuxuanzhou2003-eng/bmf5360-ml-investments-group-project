# Data-processing transparency

The user requires visibility into every material data-processing step. These rules apply to all agents working on this project.

- Before executing a processing stage, explain in plain Chinese its purpose, planned rules, and expected outputs. After execution, report actual effects and validation results. Do not silently chain material transformations without explaining them.
- Record each stage in DATA_PROCESSING_LOG.md: run time, input/output versions and paths, script/config version or hashes, rule and rationale, before/after row and instrument counts, affected rows, quarantine location, checks, limitations, and execution status.
- Explicitly disclose deletions, missing-value handling, imputation, deduplication, outlier handling, units, date/timezone changes, identifier mapping, membership selection, snapshot matching, graph selection, and label filtering. Distinguish row removal from company exclusion and missing values from genuine zero.
- Never remove securities solely because they are delisted. Explain membership eligibility and delisted-data coverage separately. Retain missing-data and failed-fetch coverage in audit records.
- Preserve raw data and provenance. Retain excluded rows with reason codes. Do not overwrite prior audit results with partial-stage results; keep run-specific records and a clearly scoped current summary.
- Do not silently introduce imputation, winsorization, forward filling, or other changes to research assumptions. Explain proposed choices and their bias implications before applying them. User visibility does not require repeated approval for already authorized and documented routine processing.
- Distinguish proposed rules, implemented code, executed runs, and independently verified results. Never claim a complete pipeline passed when a required table or check was skipped.
- Retrospective documentation must say it is retrospective; do not invent historical run counts, approvals, or authorship. Mark unavailable evidence explicitly.
- Maintain the cumulative AI_USE_LOG.md for material assistance. Document changes without fabricating other tools' activity.

# Model routing

- Root: GPT-6 Astra / low for decomposition, architecture, difficult questions, review, and final decisions.
- Delegate ordinary bounded work to GPT-5.6 Luna, normally with max reasoning. If Luna is unavailable, use GPT-5.6 Terra / max. Set explicit spawn overrides; do not silently inherit Astra for routine work.
- Back up existing config.toml and AGENTS.md before modifying either. Preserve MCP, plugin, provider, credentials, and unrelated configuration.

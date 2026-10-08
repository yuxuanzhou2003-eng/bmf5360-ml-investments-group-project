# All-sector model-ready CSV v1

This package freezes the verified 5,214-company panel before model fitting. The three large files are gzip-compressed CSVs and retain every company-date grid row. Empty fields are missing values, not zero. No imputation, winsorization, company removal, model fit, or backtest occurs here.

- `model_ready_warmup_2020.csv.gz`: rolling-feature warm-up only.
- `model_ready_train_2021_2023.csv.gz`: training partition.
- `model_ready_validation_2024_2025.csv.gz`: validation partition.
- `macro_current_vintage_sensitivity_2020_2025.csv`: separate sensitivity input; not strict PIT.
- `universe_5214.csv`: security, identity, venue, history, and SSP/SCN coverage evidence.
- `feature_dictionary.csv`: predictor, target, audit, provenance, and excluded-feature roles.

The strict main predictor list excludes `volume_log1p`, `volume_change_1d`, and `volume_z20`. Raw share volume remains for audit and later execution research. `supervised_eligible` requires an eligible label and all 34 strict market features, but it does not require complete fundamental or valuation coverage. Any model imputation must be fit inside training folds and documented separately. No 2026 data is included.

## GitHub download

This public repository stores the three compressed model-ready tables with Git LFS. After cloning, run `git lfs pull` to download their contents. The tables are frozen LSEG-derived research data uploaded for this course project under the team's authorized access; they do not include credentials or raw vendor API responses. Do not treat this repository as a substitute for an LSEG licence or redistribute the data beyond the project's permitted use.

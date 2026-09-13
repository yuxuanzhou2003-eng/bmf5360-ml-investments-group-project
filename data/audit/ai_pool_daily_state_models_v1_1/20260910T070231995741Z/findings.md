# Independent validation — 20260910T070105846190Z

- **Validation status**: `PASS`
- **Validation time (UTC)**: `2026-09-10T07:02:31.998772+00:00`
- **Model run**: `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z`
- **Scope**: persisted training/validation artifacts, source hash chain, primary sample reconstruction, and training-only RF purge evidence.
- **Sealed test policy**: `targets_test_sealed.csv` was not opened, read, hashed, or aggregated; test rows read = 0.

## Check results

| Check | Status | Detail |
|---|---|---|
| `summary_run_id` | PASS | summary identifies the requested model run |
| `summary_complete_status` | PASS | runner status is complete and development-only |
| `sealed_target_not_opened` | PASS | sealed test target is declared unopened |
| `no_test_outputs` | PASS | run contains no test predictions or metrics |
| `input_hash_model_features` | PASS | persisted input hash unchanged for model_features |
| `input_hash_eligibility` | PASS | persisted input hash unchanged for eligibility |
| `input_hash_targets_dev` | PASS | persisted input hash unchanged for targets_dev |
| `input_hash_model_ready_summary` | PASS | persisted input hash unchanged for model_ready_summary |
| `input_hash_macro_features_train_valid` | PASS | persisted input hash unchanged for macro_features_train_valid |
| `input_hash_macro_summary` | PASS | persisted input hash unchanged for macro_summary |
| `input_hash_ai_state_features` | PASS | persisted input hash unchanged for ai_state_features |
| `input_hash_ai_state_summary` | PASS | persisted input hash unchanged for ai_state_summary |
| `input_hash_config` | PASS | persisted input hash unchanged for config |
| `input_hash_runner` | PASS | persisted input hash unchanged for runner |
| `input_hash_ai_state_safe_feature_audit` | PASS | persisted input hash unchanged for ai_state_safe_feature_audit |
| `runner_hash_chain` | PASS | runner hash recorded in summary matches current script |
| `config_hash_chain` | PASS | config hash recorded in summary matches current config |
| `output_hash_validation_predictions.csv` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/validation_predictions.csv |
| `output_hash_validation_metrics.csv` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/validation_metrics.csv |
| `output_hash_feature_importance.csv` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/feature_importance.csv |
| `output_hash_feature_sets.json` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/feature_sets.json |
| `output_hash_rf_tuning_evidence.json` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/rf_tuning_evidence.json |
| `output_hash_technical_only_logistic.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/technical_only_logistic.joblib |
| `output_hash_technical_only_random_forest.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/technical_only_random_forest.joblib |
| `output_hash_technical_plus_macro_logistic.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/technical_plus_macro_logistic.joblib |
| `output_hash_technical_plus_macro_random_forest.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/technical_plus_macro_random_forest.joblib |
| `output_hash_technical_plus_ai_state_logistic.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/technical_plus_ai_state_logistic.joblib |
| `output_hash_technical_plus_ai_state_random_forest.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/technical_plus_ai_state_random_forest.joblib |
| `output_hash_full_state_logistic.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/full_state_logistic.joblib |
| `output_hash_full_state_random_forest.joblib` | PASS | model-run output hash matches for data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070105846190Z/models/full_state_random_forest.joblib |
| `feature_set_names` | PASS | all four registered feature groups are present |
| `feature_set_columns_unique` | PASS | stable technical-first deduplication |
| `credit_spread_excluded` | PASS | six credit-spread columns are absent from effective features |
| `target_columns_excluded` | PASS | target and forward-return columns are absent from features |
| `technical_duplicate_exclusions` | PASS | technical duplicates are explicitly excluded before AI merge |
| `numeric_alias_exclusion` | PASS | spy_return_1 alias is excluded in favor of technical spy_momentum_1 |
| `effective_ai_feature_count` | PASS | effective AI feature count agrees with feature sets |
| `eight_metric_rows` | PASS | four feature sets each have logistic and RF validation metrics |
| `validation_prediction_row_count` | PASS | validation predictions cover the persisted primary validation sample |
| `validation_only_predictions` | PASS | prediction table contains validation rows only |
| `eight_probability_columns` | PASS | all model validation probabilities are present and finite |
| `independent_count_model_ready_master_rows` | PASS | recomputed model_ready_master_rows from persisted development inputs |
| `independent_count_model_ready_master_instruments` | PASS | recomputed model_ready_master_instruments from persisted development inputs |
| `independent_count_model_ready_training_rows` | PASS | recomputed model_ready_training_rows from persisted development inputs |
| `independent_count_model_ready_validation_rows` | PASS | recomputed model_ready_validation_rows from persisted development inputs |
| `independent_count_model_ready_test_rows` | PASS | recomputed model_ready_test_rows from persisted development inputs |
| `independent_count_development_target_rows_loaded` | PASS | recomputed development_target_rows_loaded from persisted development inputs |
| `independent_count_primary_training_rows` | PASS | recomputed primary_training_rows from persisted development inputs |
| `independent_count_primary_validation_rows` | PASS | recomputed primary_validation_rows from persisted development inputs |
| `independent_count_primary_training_instruments` | PASS | recomputed primary_training_instruments from persisted development inputs |
| `independent_count_primary_validation_instruments` | PASS | recomputed primary_validation_instruments from persisted development inputs |
| `independent_count_primary_training_sessions` | PASS | recomputed primary_training_sessions from persisted development inputs |
| `independent_count_primary_validation_sessions` | PASS | recomputed primary_validation_sessions from persisted development inputs |
| `independent_primary_anchor_keys_unique` | PASS | recomputed primary anchors are unique per instrument/session |
| `tuning_feature_set_names` | PASS | RF tuning evidence covers all feature sets |
| `tuning_grid_technical_only` | PASS | RF candidate grid is the pre-registered grid |
| `tuning_purge_technical_only` | PASS | RF evidence reproduces exit-session purged expanding folds and no label window crosses validation start |
| `tuning_selected_technical_only` | PASS | selected RF parameters are from the registered training-only grid |
| `tuning_grid_technical_plus_macro` | PASS | RF candidate grid is the pre-registered grid |
| `tuning_purge_technical_plus_macro` | PASS | RF evidence reproduces exit-session purged expanding folds and no label window crosses validation start |
| `tuning_selected_technical_plus_macro` | PASS | selected RF parameters are from the registered training-only grid |
| `tuning_grid_technical_plus_ai_state` | PASS | RF candidate grid is the pre-registered grid |
| `tuning_purge_technical_plus_ai_state` | PASS | RF evidence reproduces exit-session purged expanding folds and no label window crosses validation start |
| `tuning_selected_technical_plus_ai_state` | PASS | selected RF parameters are from the registered training-only grid |
| `tuning_grid_full_state` | PASS | RF candidate grid is the pre-registered grid |
| `tuning_purge_full_state` | PASS | RF evidence reproduces exit-session purged expanding folds and no label window crosses validation start |
| `tuning_selected_full_state` | PASS | selected RF parameters are from the registered training-only grid |
| `macro_session_key_coverage` | PASS | all primary formation sessions have a macro key |
| `ai_state_session_key_coverage` | PASS | all primary formation sessions have an AI-state key |
| `safe_ai_feature_list_matches_audit` | PASS | coverage-safe AI features exactly match the independently audited safe list |
| `runner_audit_flags` | PASS | runner audit agrees with independent validation |
| `runner_audit_test_seal` | PASS | runner audit seals test and input mutation policy |
| `processing_log_appended` | PASS | DATA_PROCESSING_LOG contains this run |
| `ai_use_log_appended` | PASS | AI_USE_LOG contains this run |

## Findings

- No failures. The persisted run passed all independent checks within the stated scope.

## Processing limits

- The validator verifies persisted development artifacts and source hashes; it does not inspect sealed test target rows or make any test performance claim.
- Input hash equality verifies that the runner did not mutate listed source files between the runner read and this validation.
- Structural pre-listing missingness is validated through the run summary and effective feature schema; imputer fitting remains inside persisted model pipelines.

# Superseded audit run

This run (`20260910T065022447921Z`) used an early audit-script version whose split-gate condition incorrectly required the pooled `all_anchors` row to be a split-specific row. It therefore serialized zero safe candidates even though the underlying checks and missingness calculations were otherwise the same.

The run is retained for provenance and must not be used as the input decision. The corrected, independently re-read run is:

`data/audit/ai_daily_enhanced_inputs_v1/20260910T065044522242Z/`

No input file was modified, deleted, imputed, or re-written by either audit run.

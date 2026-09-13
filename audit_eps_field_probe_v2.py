"""Audit two immutable EPS field probes without modifying research data."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "data/raw/eps_field_probe_v2/20260908T165319452604Z"
FOLLOWUP = ROOT / "data/raw/eps_alternative_probe/20260908T165548467804Z"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data/audit/eps_field_semantics" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


inputs = sorted(PRIMARY.glob("*")) + sorted(FOLLOWUP.glob("*"))
input_files = [path for path in inputs if path.is_file()]
before = {str(path.relative_to(ROOT)): sha256(path) for path in input_files}
manifest1 = json.loads((PRIMARY / "manifest.json").read_text(encoding="utf-8"))
manifest2 = json.loads((FOLLOWUP / "manifest.json").read_text(encoding="utf-8"))


def read(run: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(run / f"{name}.csv")


base_a = read(PRIMARY, "actual_base")
reported_a = read(PRIMARY, "actual_reported")
currency_a = read(PRIMARY, "actual_currency_suffix")
scale_a = read(PRIMARY, "actual_scale_suffix")
effective_a = read(PRIMARY, "actual_effective_suffix")
fperiod_a = read(PRIMARY, "actual_fperiod_suffix")
restated = read(PRIMARY, "actual_restated_candidate")
goforward = read(PRIMARY, "actual_goforward_candidate")
scale0_a, scale6_a = read(PRIMARY, "actual_usd_scale0"), read(PRIMARY, "actual_usd_scale6")
base_e = read(PRIMARY, "estimate_base")
currency_e = read(PRIMARY, "estimate_currency_suffix")
scale_e = read(PRIMARY, "estimate_scale_suffix")
fperiod_e = read(PRIMARY, "estimate_fperiod_suffix")
scale0_e, scale6_e = read(PRIMARY, "estimate_usd_scale0"), read(PRIMARY, "estimate_usd_scale6")
diluted = read(FOLLOWUP, "diluted_ex_extra")
normalized = read(FOLLOWUP, "normalized_diluted")
reported_full = read(FOLLOWUP, "reported_actual_full")
restated_full = read(FOLLOWUP, "restated_with_period")
goforward_full = read(FOLLOWUP, "goforward_with_period")

pd.testing.assert_frame_equal(base_a, reported_a)
pd.testing.assert_frame_equal(base_a, scale0_a)
pd.testing.assert_frame_equal(base_a, scale6_a)
pd.testing.assert_frame_equal(base_e, scale0_e)
pd.testing.assert_frame_equal(base_e, scale6_e)

comparison = reported_full.merge(diluted, on="Instrument", suffixes=("_ibes", "_diluted"))
comparison = comparison.merge(normalized, on="Instrument", suffixes=("", "_normalized"))
comparison = comparison.rename(columns={
    "Earnings Per Share - Actual": "ibes_actual_reported_selector",
    "Diluted EPS Excluding Extraordinary Items": "diluted_eps_excl_extra",
    "EPS Normalized (dil.)": "eps_normalized_diluted",
})
comparison["ibes_minus_diluted"] = (
    comparison["ibes_actual_reported_selector"] - comparison["diluted_eps_excl_extra"]
)
comparison["issuer_reference_note"] = comparison["Instrument"].map({
    "AAPL.OQ": "Issuer diluted EPS 1.53; no GAAP/non-GAAP conclusion from rounding difference.",
    "NVDA.OQ": "Issuer GAAP diluted EPS 5.98 and non-GAAP 6.12 before 10-for-1 split.",
})
comparison.to_csv(OUT / "field_value_comparison.csv", index=False)

activation_call = next(call for call in manifest1["calls"] if call["name"] == "actual_activation_suffix")
checks = {
    "primary_call_accounting_17_returned_1_error": manifest1["returned_calls"] == 17 and manifest1["error_calls"] == 1,
    "followup_all_5_returned": manifest2["returned_calls"] == 5 and manifest2["error_calls"] == 0,
    "default_equals_reported_36_rows": len(base_a) == 36,
    "actual_currency_all_usd": len(currency_a) == 36 and set(currency_a["Currency"]) == {"USD"},
    "estimate_currency_all_usd": len(currency_e) == 18 and set(currency_e["Currency"]) == {"USD"},
    "actual_scale_suffix_all_blank": scale_a["SCALE"].isna().all(),
    "estimate_scale_suffix_all_blank": scale_e["SCALE"].isna().all(),
    "scale_parameter_0_6_no_value_change": True,
    "restated_and_goforward_blank": restated.iloc[:, -1].isna().all() and goforward.iloc[:, -1].isna().all()
                                       and restated_full.iloc[:, -1].isna().all()
                                       and goforward_full.iloc[:, -1].isna().all(),
    "fperiod_populated": fperiod_a.iloc[:, -1].notna().all() and fperiod_e.iloc[:, -1].notna().all(),
    "effective_suffix_blank": effective_a["EFFECTIVEDATE"].isna().all(),
    "activation_suffix_has_no_distinct_return_column": activation_call["columns"] == list(base_a.columns),
    "alternative_values_nonmissing": comparison[["ibes_actual_reported_selector", "diluted_eps_excl_extra",
                                                   "eps_normalized_diluted"]].notna().all().all(),
    "diluted_equals_normalized_in_two_row_sample": bool(np.allclose(
        comparison["diluted_eps_excl_extra"], comparison["eps_normalized_diluted"])),
}
after = {str(path.relative_to(ROOT)): sha256(path) for path in input_files}
checks["inputs_unchanged"] = before == after
summary = {
    "run_id": RUN_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "primary_probe": str(PRIMARY.relative_to(ROOT)), "followup_probe": str(FOLLOWUP.relative_to(ROOT)),
    "primary_calls_returned": manifest1["returned_calls"], "primary_calls_error": manifest1["error_calls"],
    "followup_calls_returned": manifest2["returned_calls"], "followup_calls_error": manifest2["error_calls"],
    "default_reported_equal_rows": len(base_a), "actual_currency_rows_usd": len(currency_a),
    "estimate_currency_rows_usd": len(currency_e),
    "nvidia_2024_values": comparison.loc[comparison["Instrument"].eq("NVDA.OQ")].to_dict("records"),
    "input_hashes_before": before, "input_hashes_after": after,
    "audit_script_sha256": sha256(Path(__file__)), "checks": checks,
    "limitations": [
        "Field acceptance and returned labels do not replace Data Item Browser definitions.",
        "Blank scale/effective output does not prove the concept is unavailable elsewhere.",
        "Reported selector is not renamed GAAP; NVIDIA sample differs from diluted/normalized fields.",
        "No field from this probe is merged into clean data or the panel.",
    ],
}
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2,
               default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n",
    encoding="utf-8",
)
if not all(checks.values()):
    raise RuntimeError({key: value for key, value in checks.items() if not value})
print(json.dumps({"run_id": RUN_ID, "checks": len(checks), "nvidia": summary["nvidia_2024_values"]}, indent=2))

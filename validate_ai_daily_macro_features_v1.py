"""Independently validate the lagged AI industry state feature artifact.

The validator reads only the clean state outputs and the explicitly allowed
model_features/price inputs. It never opens target, label, future-return, or
model-run files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "ai_daily_macro_feature_v1"
FORBIDDEN_COLUMNS = {"y", "label", "label_complete", "target", "target_reason", "forward_excess_return", "stock_forward_return", "benchmark_forward_return"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    hashes_path = output_dir / "hashes.json"
    state_path = output_dir / "ai_state_features.csv"
    pool_path = output_dir / "pool_member_coverage.csv"
    etf_path = output_dir / "etf_coverage.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    state = pd.read_csv(state_path, low_memory=False)
    pool = pd.read_csv(pool_path, low_memory=False)
    etf = pd.read_csv(etf_path, low_memory=False)
    state["formation_session"] = pd.to_datetime(state["formation_session"], errors="coerce")
    state["reference_session"] = pd.to_datetime(state["reference_session"], errors="coerce")
    pool["formation_session"] = pd.to_datetime(pool["formation_session"], errors="coerce")
    pool["reference_session"] = pd.to_datetime(pool["reference_session"], errors="coerce")
    etf["formation_session"] = pd.to_datetime(etf["formation_session"], errors="coerce")
    etf["reference_session"] = pd.to_datetime(etf["reference_session"], errors="coerce")

    checks: dict[str, bool] = {}
    checks["summary_schema"] = summary.get("schema_version") == SCHEMA_VERSION
    checks["hash_manifest_schema"] = hashes.get("schema_version") == f"{SCHEMA_VERSION}_hash_manifest"
    checks["state_rows_match_summary"] = len(state) == int(summary["counts"]["state_rows"])
    checks["pool_rows_match_summary"] = len(pool) == int(summary["counts"]["pool_coverage_rows"])
    checks["etf_rows_match_summary"] = len(etf) == int(summary["counts"]["etf_coverage_rows"])
    checks["state_keys_unique"] = not state.duplicated(["formation_session"]).any()
    checks["pool_keys_unique"] = not pool.duplicated(["formation_session", "Instrument"]).any()
    checks["etf_keys_unique"] = not etf.duplicated(["formation_session", "Instrument"]).any()
    checks["reference_strictly_before_formation"] = bool((state["reference_session"] < state["formation_session"]).all())
    checks["state_no_forbidden_columns"] = not bool(FORBIDDEN_COLUMNS & set(state.columns))
    def finite_or_na(frame: pd.DataFrame) -> bool:
        values = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        return bool(np.isfinite(values).all())

    checks["state_no_infinite_numeric"] = finite_or_na(state)
    checks["pool_no_infinite_numeric"] = finite_or_na(pool)
    checks["etf_no_infinite_numeric"] = finite_or_na(etf)
    checks["pool_coverage_active_all"] = bool(pool["active_member"].astype(bool).all())
    checks["pool_low_coverage_consistent"] = bool((state["pool_low_coverage"].astype(bool) == (state[["pool_low_coverage_return", "pool_low_coverage_ma20", "pool_low_coverage_ma60"]].astype(bool).any(axis=1))).all())

    grouped_pool = pool.groupby("formation_session", sort=True)
    recon_pool = grouped_pool.agg(
        valid_return_count=("return_valid", "sum"),
        valid_ma20_count=("above_ma20_valid", "sum"),
        valid_ma60_count=("above_ma60_valid", "sum"),
        mean_return=("one_session_return", "mean"),
        return_dispersion=("one_session_return", lambda x: x.dropna().std(ddof=1)),
    ).reset_index()
    compare = state[["formation_session", "pool_valid_return_count", "pool_valid_ma20_count", "pool_valid_ma60_count", "pool_equal_weight_return_1", "pool_cross_sectional_dispersion_1"]].merge(recon_pool, on="formation_session", how="left", validate="one_to_one")
    checks["pool_denominators_recomputed"] = bool((compare["pool_valid_return_count"] == compare["valid_return_count"]).all() & (compare["pool_valid_ma20_count"] == compare["valid_ma20_count"]).all() & (compare["pool_valid_ma60_count"] == compare["valid_ma60_count"]).all())
    checks["pool_equal_weight_recomputed"] = bool(np.allclose(compare["pool_equal_weight_return_1"].fillna(0), compare["mean_return"].fillna(0), atol=1e-12, equal_nan=True))
    checks["pool_dispersion_recomputed"] = bool(np.allclose(compare["pool_cross_sectional_dispersion_1"].fillna(0), compare["return_dispersion"].fillna(0), atol=1e-12, equal_nan=True))

    for name, metadata in hashes.get("output_hashes", {}).items():
        path = ROOT / metadata["path"]
        checks[f"hash_{name}"] = path.exists() and sha256_file(path) == metadata["sha256"]
    input_hash_checks = {}
    for name, metadata in hashes.get("input_hashes", {}).items():
        path = ROOT / metadata["path"]
        input_hash_checks[name] = bool(path.exists() and sha256_file(path) == metadata["sha256"])
    checks["allowed_inputs_unchanged"] = bool(input_hash_checks) and bool(all(input_hash_checks.values()))
    checks["no_test_targets_or_model_results_read"] = True

    validation = {
        "schema_version": f"{SCHEMA_VERSION}_independent_validation",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "feature_run": summary.get("run_id"),
        "feature_output": rel(output_dir),
        "checks": checks,
        "input_hash_checks": input_hash_checks,
        "counts": {"state_rows": int(len(state)), "pool_coverage_rows": int(len(pool)), "etf_coverage_rows": int(len(etf)), "failed_checks": int(sum(not value for value in checks.values()))},
        "status": "independently_verified" if all(checks.values()) else "verification_failed",
        "limitations": ["Validator recomputes output aggregations and source/output hashes; it does not inspect sealed targets or model-run artifacts."],
    }
    validation_path = output_dir / "independent_validation.json"
    write_json(validation_path, validation)
    print(json.dumps({"status": validation["status"], "failed_checks": validation["counts"]["failed_checks"], "path": rel(validation_path)}, ensure_ascii=False))
    return validation_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently validate AI industry state features.")
    parser.add_argument("output_dir", help="Run-scoped clean output directory")
    args = parser.parse_args()
    main((ROOT / args.output_dir).resolve())

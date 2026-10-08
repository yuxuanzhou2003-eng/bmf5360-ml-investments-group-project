"""Independently reread and validate the frozen model-ready CSV package."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "deliverables/All_Sector_Model_Ready_CSV_20261006/v1"
PARTITIONS = {
    "warmup": ("model_ready_warmup_2020.csv.gz", 253, "2020-01-02", "2020-12-31"),
    "train": ("model_ready_train_2021_2023.csv.gz", 753, "2021-01-04", "2023-12-29"),
    "validation": ("model_ready_validation_2024_2025.csv.gz", 502, "2024-01-02", "2025-12-31"),
}
USECOLS = [
    "source_input_position", "Instrument", "date", "partition", "target_return_5d", "target_up",
    "date_in_fixed_calendar", "label_future_complete", "label_within_partition", "label_eligible",
    "history_scope_eligible", "sector_available", "strict_market_feature_available_34",
    "score_eligible", "supervised_eligible", "pit_feature_available_m1", "m1_supervised_eligible",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().eq("true").to_numpy()


def validate_partition(name: str, spec: tuple[str, int, str, str]) -> dict:
    filename, sessions, expected_min, expected_max = spec
    path = PACKAGE / filename
    rows = 0
    positions: Counter[int] = Counter()
    dates: Counter[str] = Counter()
    last_key: tuple[int, str] | None = None
    score_rows = supervised_rows = m1_rows = 0
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=150_000, low_memory=False):
        rows += len(chunk)
        position = pd.to_numeric(chunk["source_input_position"], errors="raise").astype(int)
        date = chunk["date"].astype(str)
        positions.update(position.tolist())
        dates.update(date.tolist())
        if not chunk["partition"].eq(name).all():
            raise AssertionError(f"partition value mismatch in {filename}")
        if date.str.startswith("2026").any():
            raise AssertionError(f"2026 date found in {filename}")
        keys = list(zip(position.tolist(), date.tolist()))
        if last_key is not None and keys[0] <= last_key:
            raise AssertionError(f"key order/uniqueness failure across chunks in {filename}")
        if any(right <= left for left, right in zip(keys, keys[1:])):
            raise AssertionError(f"key order/uniqueness failure within chunk in {filename}")
        last_key = keys[-1]

        fixed = as_bool(chunk["date_in_fixed_calendar"])
        history = as_bool(chunk["history_scope_eligible"])
        sector = as_bool(chunk["sector_available"])
        market = as_bool(chunk["strict_market_feature_available_34"])
        score = as_bool(chunk["score_eligible"])
        future = as_bool(chunk["label_future_complete"])
        within = as_bool(chunk["label_within_partition"])
        label = as_bool(chunk["label_eligible"])
        supervised = as_bool(chunk["supervised_eligible"])
        pit = as_bool(chunk["pit_feature_available_m1"])
        m1 = as_bool(chunk["m1_supervised_eligible"])
        target = pd.to_numeric(chunk["target_return_5d"], errors="coerce").to_numpy(float)
        target_up = pd.to_numeric(chunk["target_up"], errors="coerce").to_numpy(float)
        target_valid = np.isfinite(target) & np.isin(target_up, [0.0, 1.0])
        if not np.array_equal(score, fixed & history & sector & market):
            raise AssertionError(f"score_eligible identity failure in {filename}")
        if not np.array_equal(supervised, score & future & within & label & target_valid):
            raise AssertionError(f"supervised_eligible identity failure in {filename}")
        if not np.array_equal(m1, supervised & pit):
            raise AssertionError(f"m1_supervised_eligible identity failure in {filename}")
        both_missing = np.isnan(target) & np.isnan(target_up)
        if not np.all(both_missing | target_valid):
            raise AssertionError(f"target missingness/value failure in {filename}")
        if target_valid.any() and not np.array_equal(target_up[target_valid], (target[target_valid] > 0).astype(float)):
            raise AssertionError(f"target direction failure in {filename}")
        score_rows += int(score.sum())
        supervised_rows += int(supervised.sum())
        m1_rows += int(m1.sum())

    expected_rows = 5214 * sessions
    if rows != expected_rows:
        raise AssertionError(f"row count failure in {filename}: {rows} != {expected_rows}")
    if len(positions) != 5214 or set(positions.values()) != {sessions}:
        raise AssertionError(f"company grid failure in {filename}")
    if len(dates) != sessions or set(dates.values()) != {5214}:
        raise AssertionError(f"date grid failure in {filename}")
    if min(dates) != expected_min or max(dates) != expected_max:
        raise AssertionError(f"date boundary failure in {filename}")
    return {
        "file": filename,
        "rows": rows,
        "instruments": len(positions),
        "sessions": len(dates),
        "date_min": min(dates),
        "date_max": max(dates),
        "score_eligible": score_rows,
        "supervised_eligible": supervised_rows,
        "m1_supervised_eligible": m1_rows,
        "sha256": sha(path),
    }


def main() -> None:
    manifest_path = PACKAGE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = {row["path"]: row["sha256"] for row in manifest["outputs"]}
    hash_checks = {name: sha(PACKAGE / name) == value for name, value in expected_hashes.items()}
    if not all(hash_checks.values()):
        raise AssertionError("one or more frozen output hashes changed")
    partition_results = {name: validate_partition(name, spec) for name, spec in PARTITIONS.items()}
    for name, result in partition_results.items():
        if result["supervised_eligible"] != manifest["partition_counts"][name]["supervised_eligible"]:
            raise AssertionError(f"supervised count differs from export receipt: {name}")

    macro = pd.read_csv(PACKAGE / "macro_current_vintage_sensitivity_2020_2025.csv", low_memory=False)
    macro_dates = pd.to_datetime(macro["spy_session"], errors="raise")
    if len(macro) != 1508 or macro_dates.duplicated().any() or macro_dates.min().strftime("%Y-%m-%d") != "2020-01-02" or macro_dates.max().strftime("%Y-%m-%d") != "2025-12-31":
        raise AssertionError("macro sidecar date grid failure")
    if not as_bool(macro["sensitivity_only"]).all() or as_bool(macro["strict_pit_eligible"]).any():
        raise AssertionError("macro vintage flags failure")

    universe = pd.read_csv(PACKAGE / "universe_5214.csv", dtype=str, keep_default_na=False)
    if len(universe) != 5214 or universe["source_input_position"].duplicated().any():
        raise AssertionError("universe uniqueness failure")
    dictionary = pd.read_csv(PACKAGE / "feature_dictionary.csv", dtype=str, keep_default_na=False)
    excluded = dictionary.loc[dictionary["column"].isin(["volume_log1p", "volume_change_1d", "volume_z20"])]
    if len(excluded) != 3 or excluded["included_in_strict_main_predictors"].str.lower().ne("false").any():
        raise AssertionError("share-volume exclusion contract failure")

    report = {
        "schema": "all_sector_model_ready_csv_independent_validation_v1",
        "status": "passed",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha(manifest_path),
        "validator_sha256": sha(Path(__file__)),
        "frozen_output_hash_checks": hash_checks,
        "partitions": partition_results,
        "macro_rows": len(macro),
        "universe_rows": len(universe),
        "dictionary_rows": len(dictionary),
        "checks": {
            "all_files_reread": True,
            "fixed_company_date_grid": True,
            "strict_key_order_and_uniqueness": True,
            "partition_and_date_boundaries": True,
            "no_2026_dates": True,
            "target_direction_and_missingness": True,
            "eligibility_mask_identities": True,
            "macro_vintage_flags": True,
            "share_volume_exclusion_contract": True,
        },
    }
    validation_path = PACKAGE / "independent_validation.json"
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete = {
        "status": "complete_and_independently_validated",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": report["manifest_sha256"],
        "independent_validation_sha256": sha(validation_path),
    }
    (PACKAGE / "COMPLETE.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

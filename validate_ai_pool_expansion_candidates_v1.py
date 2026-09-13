"""Independent validation for an AI supply-chain candidate-registry run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


VALIDATOR_VERSION = "1.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="ai_pool_expansion_v1_config.json")
    parser.add_argument("--output-root", default="data/audit/ai_pool_expansion_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = root / args.output_root / args.run_id
    summary_path = out / "summary.json"
    gate_path = out / "gate_report.json"
    registry_path = out / "candidate_registry.csv"
    screening_path = out / "universe_screening.csv"
    evidence_path = out / "local_segment_evidence.csv"
    source_path = out / "source_catalog.csv"
    quarantine_path = out / "candidate_quarantine.csv"
    required = [summary_path, gate_path, registry_path, screening_path, evidence_path, source_path, quarantine_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing run outputs: " + ", ".join(missing))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    spans_path = root / config["inputs"]["membership_spans"]
    spans = read_csv(spans_path)
    registry = read_csv(registry_path)
    screening = read_csv(screening_path)
    evidence = read_csv(evidence_path)
    source_rows = read_csv(source_path)
    quarantine = read_csv(quarantine_path)
    configured_rics = [row["ric"] for row in config["candidates"]]
    registry_rics = [row["ric"] for row in registry]
    screening_selected = {row["ric"] for row in screening if row.get("screen_status") == "selected_design_candidate"}
    screening_not_selected = {row["ric"] for row in screening if row.get("screen_status") == "not_selected_design_candidate"}
    span_rics = [row.get("ric", "") for row in spans]
    current_hashes = summary["input_hashes"]
    raw_hash_checks: dict[str, bool] = {}
    for family in ["raw_segment_files", "raw_announcement_files"]:
        for path_value, expected in current_hashes[family].items():
            path = root / Path(path_value)
            raw_hash_checks[path_value] = path.exists() and sha256(path) == expected

    checks = {
        "run_id_matches": summary.get("run_id") == args.run_id and gate.get("run_id") == args.run_id,
        "config_rics_equal_registry": set(configured_rics) == set(registry_rics),
        "registry_row_count_49": len(registry) == 49,
        "registry_ric_unique": len(registry_rics) == len(set(registry_rics)),
        "spans_row_count_781": len(spans) == 781,
        "spans_ric_unique": len(span_rics) == len(set(span_rics)),
        "screening_row_count_matches_spans": len(screening) == len(spans),
        "screening_selected_equals_registry": screening_selected == set(registry_rics),
        "screening_not_selected_retained": len(screening_not_selected) == 732,
        "quarantine_empty_and_explicit": len(quarantine) == 0 and quarantine_path.exists(),
        "seven_groups_present": len(set(row["primary_group"] for row in registry)) == 7,
        "source_rows_match_registry": len(source_rows) == len(registry) and {row["ric"] for row in source_rows} == set(registry_rics),
        "direct_local_rows_have_pit_date": all(
            row.get("pit_date_available") == "True" for row in evidence if row.get("ric") in {r["ric"] for r in registry if r.get("pit_evidence_status") == "direct_local_pit"}
        ),
        "summary_output_hashes_match": all(
            summary["output_hashes"].get(path.name) == sha256(path)
            for path in [registry_path, screening_path, evidence_path, source_path, quarantine_path]
        ),
        "gate_summary_hash_matches": gate.get("summary_sha256") == sha256(summary_path),
        "raw_input_hashes_unchanged": bool(raw_hash_checks) and all(raw_hash_checks.values()),
        "no_forbidden_output_columns": not any(
            forbidden in set(registry[0]) | set(screening[0])
            for forbidden in ["return", "returns", "target", "targets", "label", "labels", "model_output", "prediction"]
        ),
        "no_return_target_label_reads_recorded": all(summary["counts"].get(key, -1) == 0 for key in ["return_rows_read", "target_rows_read", "label_rows_read", "model_output_rows_read"]),
        "summary_status_complete": summary.get("status") == "complete_design_candidate_registry",
        "gate_status_complete": gate.get("status") == "complete_design_candidate_registry",
    }
    recomputed = {
        "registry_rows": len(registry),
        "registry_instruments": len(set(registry_rics)),
        "screening_rows": len(screening),
        "screening_instruments": len({row["ric"] for row in screening}),
        "selected_rows": len(screening_selected),
        "not_selected_rows": len(screening_not_selected),
        "evidence_rows": len(evidence),
        "source_rows": len(source_rows),
        "quarantine_rows": len(quarantine),
        "group_counts": dict(sorted(Counter(row["primary_group"] for row in registry).items())),
        "direct_local_pit_candidates": sum(row.get("pit_evidence_status") == "direct_local_pit" for row in registry),
        "provisional_static_source_candidates": sum(row.get("pit_evidence_status") == "provisional_static_source" for row in registry),
    }
    result = {
        "schema_version": "ai_pool_expansion_candidates_independent_validation_v1",
        "run_id": args.run_id,
        "validator": {"path": Path(__file__).relative_to(root).as_posix(), "version": VALIDATOR_VERSION, "sha256": sha256(Path(__file__))},
        "input_run": out.relative_to(root).as_posix(),
        "recomputed": recomputed,
        "raw_input_hash_checks": raw_hash_checks,
        "checks": checks,
        "status": "independently_verified" if all(checks.values()) else "validation_failed",
        "limitations": [
            "This validator confirms file-level and registry-level consistency; it does not make current issuer pages historical PIT evidence.",
            "It does not re-read or score returns, targets, labels, or model outputs.",
        ],
    }
    result_path = out / "independent_validation.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

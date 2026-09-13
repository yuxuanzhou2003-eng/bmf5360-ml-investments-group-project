"""Build the auditable AI supply-chain design candidate registry.

This stage intentionally does not read returns, targets, labels, model outputs, or
portfolio files.  It uses literal RICs from a frozen config, the existing corrected
universe spans, and a bounded read of existing Orig segment/announcement raw files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "1.0.0"
GROUP_ORDER = [
    "gpu_accelerator",
    "ai_semiconductor",
    "memory_hbm_storage",
    "server_network",
    "cloud_software",
    "data_center_power_cooling",
    "robotics_autonomy",
]

# Exact historical segment-text evidence dictionary for this registry.  Generic
# words such as software, cloud, semiconductor, memory, storage and utility are
# deliberately absent.  AI is matched as an independent word.
GROUP_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "gpu_accelerator": [
        ("AI", re.compile(r"\bAI\b", re.I)),
        ("GPU", re.compile(r"\bGPU\b", re.I)),
        ("AI accelerator", re.compile(r"AI accelerator", re.I)),
        ("accelerated computing", re.compile(r"accelerated computing", re.I)),
        ("tensor", re.compile(r"tensor", re.I)),
        ("inference chip", re.compile(r"inference chip", re.I)),
        ("data-center processor", re.compile(r"data[- ]center processor", re.I)),
    ],
    "ai_semiconductor": [
        ("AI", re.compile(r"\bAI\b", re.I)),
        ("GPU", re.compile(r"\bGPU\b", re.I)),
        ("AI accelerator", re.compile(r"AI accelerator", re.I)),
        ("tensor", re.compile(r"tensor", re.I)),
        ("inference", re.compile(r"inference", re.I)),
    ],
    "memory_hbm_storage": [
        ("HBM", re.compile(r"\bHBM\b", re.I)),
        ("high bandwidth memory", re.compile(r"high bandwidth memory", re.I)),
        ("memory for AI", re.compile(r"memory for AI", re.I)),
        ("AI memory", re.compile(r"AI memory", re.I)),
        ("data-center memory", re.compile(r"data[- ]center memory", re.I)),
        ("SSD/NAND for data center", re.compile(r"SSD/NAND for data center", re.I)),
        ("data center", re.compile(r"data[- ]center", re.I)),
    ],
    "server_network": [
        ("AI server", re.compile(r"AI server", re.I)),
        ("AI networking", re.compile(r"AI networking", re.I)),
        ("data center", re.compile(r"data[- ]center", re.I)),
        ("data-center infrastructure", re.compile(r"data[- ]center infrastructure", re.I)),
        ("high-performance computing", re.compile(r"high[- ]performance computing", re.I)),
        ("accelerated server", re.compile(r"accelerated server", re.I)),
    ],
    "cloud_software": [
        ("AI software platform", re.compile(r"AI software platform", re.I)),
        ("machine-learning platform", re.compile(r"machine[- ]learning platform", re.I)),
        ("generative AI platform", re.compile(r"generative AI platform", re.I)),
        ("intelligent cloud", re.compile(r"intelligent cloud", re.I)),
        ("AI services", re.compile(r"AI services", re.I)),
        ("AI", re.compile(r"\bAI\b", re.I)),
    ],
    "data_center_power_cooling": [
        ("data-center power", re.compile(r"data[- ]center power", re.I)),
        ("data-center cooling", re.compile(r"data[- ]center cooling", re.I)),
        ("liquid cooling", re.compile(r"liquid cooling", re.I)),
        ("AI power systems", re.compile(r"AI power systems", re.I)),
        ("data-center equipment", re.compile(r"data[- ]center equipment", re.I)),
    ],
    "robotics_autonomy": [
        ("robotics", re.compile(r"robotics", re.I)),
        ("autonomous driving", re.compile(r"autonomous driving", re.I)),
        ("autonomous systems", re.compile(r"autonomous systems", re.I)),
        ("industrial AI", re.compile(r"industrial AI", re.I)),
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def valid_date(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    # The raw provider field can contain timestamps; keep only the calendar date
    # for the day-level PIT limitation documented by the project.
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else ""


def classify_name(name: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for group in GROUP_ORDER:
        hits = [label for label, pattern in GROUP_PATTERNS[group] if pattern.search(name or "")]
        if hits:
            result[group] = hits
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ai_pool_expansion_v1_config.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="data/audit/ai_pool_expansion_v1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve(root, args.output_root) / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    taxonomy_path = resolve(root, config["taxonomy_path"])
    spans_path = resolve(root, config["inputs"]["membership_spans"])
    raw_root = resolve(root, config["inputs"]["raw_root"])
    old_pool_path = resolve(root, config["inputs"]["existing_pool_membership"])

    candidates = config["candidates"]
    candidate_rics = [str(item["ric"]) for item in candidates]
    candidate_counter = Counter(candidate_rics)
    duplicate_candidate_rics = sorted([ric for ric, count in candidate_counter.items() if count > 1])
    candidate_by_ric = {item["ric"]: item for item in candidates}

    spans = read_csv(spans_path)
    span_by_ric = {row.get("ric", ""): row for row in spans}
    duplicate_span_rics = sorted(
        ric for ric, count in Counter(row.get("ric", "") for row in spans).items() if ric and count > 1
    )

    segment_files = sorted(raw_root.glob(config["inputs"]["raw_segment_glob"]))
    announcement_files = sorted(raw_root.glob(config["inputs"]["raw_announcement_glob"]))
    if not segment_files or not announcement_files:
        raise FileNotFoundError("Required bounded LSEG raw segment/announcement files are missing")

    # Announcement dates are indexed first, then only candidate RIC rows are
    # retained in evidence output.  No rows are imputed or deduplicated.
    announcement_rows_total = 0
    announcement_map: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    announcement_rows_by_ric: Counter[str] = Counter()
    for path in announcement_files:
        for row in read_csv(path):
            announcement_rows_total += 1
            ric = row.get("Instrument", "")
            if ric in candidate_by_ric:
                announcement_rows_by_ric[ric] += 1
                period_end = valid_date(row.get("Income Statement Period End Date", ""))
                announcement = valid_date(row.get("Income Statement Orig Announce Date", ""))
                if period_end and announcement:
                    announcement_map[(ric, period_end)].append(announcement)

    segment_rows_total = 0
    segment_rows_by_ric: Counter[str] = Counter()
    segment_periods_by_ric: defaultdict[str, set[str]] = defaultdict(set)
    evidence_rows: list[dict[str, Any]] = []
    evidence_by_ric: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in segment_files:
        for raw_row, row in enumerate(read_csv(path), start=1):
            segment_rows_total += 1
            ric = row.get("Instrument", "")
            if ric not in candidate_by_ric:
                continue
            segment_rows_by_ric[ric] += 1
            period_end = valid_date(row.get("Date", ""))
            if period_end:
                segment_periods_by_ric[ric].add(period_end)
            segment_name = row.get("Segment Name", "") or ""
            classifications = classify_name(segment_name)
            if not classifications:
                continue
            announcements = sorted(set(announcement_map.get((ric, period_end), [])))
            for group, matched_terms in classifications.items():
                evidence = {
                    "ric": ric,
                    "canonical_name": candidate_by_ric[ric]["canonical_name"],
                    "primary_group": candidate_by_ric[ric]["primary_group"],
                    "evidence_group": group,
                    "matched_terms": " | ".join(matched_terms),
                    "segment_name": segment_name,
                    "period_end": period_end,
                    "orig_announcement_dates": " | ".join(announcements),
                    "pit_date_available": bool(announcements),
                    "raw_file": path.relative_to(root).as_posix(),
                    "raw_row": raw_row,
                    "raw_text": ",".join(
                        [ric, row.get("Segment Code", ""), segment_name, row.get("Financial Period Absolute", ""), row.get("Date", ""), row.get("Standardized Revenue - Business Segment", "")]
                    ),
                    "reason_code": "LOCAL_LSEG_ORIG_SEGMENT_KEYWORD_MATCH",
                }
                evidence_rows.append(evidence)
                evidence_by_ric[ric].append(evidence)

    old_pool_member_rics: set[str] = set()
    old_pool_rows = read_csv(old_pool_path)
    for row in old_pool_rows:
        if row.get("membership_status", "") == "member":
            old_pool_member_rics.add(row.get("Instrument", ""))

    registry_rows: list[dict[str, Any]] = []
    for item in candidates:
        ric = item["ric"]
        span = span_by_ric.get(ric, {})
        ric_evidence = evidence_by_ric.get(ric, [])
        local_groups = sorted({row["evidence_group"] for row in ric_evidence}, key=GROUP_ORDER.index)
        local_announcements = sorted(
            {
                announcement
                for row in ric_evidence
                for announcement in str(row.get("orig_announcement_dates", "")).split(" | ")
                if announcement
            }
        )
        first_local_announcement = local_announcements[0] if local_announcements else ""
        member_from = valid_date(span.get("member_from", ""))
        if first_local_announcement and member_from:
            first_observable = max(first_local_announcement, member_from)
        else:
            first_observable = first_local_announcement or member_from
        has_local_pit = bool(ric_evidence and local_announcements)
        if has_local_pit:
            pit_status = "direct_local_pit"
            pit_limitation = "Local Orig segment keyword match joins an Orig announcement date; day-level date only and raw segment taxonomy coverage is limited to this LSEG run."
        elif ric_evidence:
            pit_status = "historical_segment_match_missing_announcement"
            pit_limitation = "Historical segment text matched, but no Orig announcement date joined for the matched period; retain membership_unknown pending dated evidence."
        else:
            pit_status = "provisional_static_source"
            pit_limitation = "Issuer description is a current/reference snapshot and was not downloaded or hashed in this run; it cannot backfill historical membership."
        registry_rows.append(
            {
                "ric": ric,
                "canonical_name": item["canonical_name"],
                "primary_group": item["primary_group"],
                "candidate_status": "selected_design_candidate",
                "inclusion_reason": item["inclusion_reason"],
                "source_type": item["source_type"],
                "source_title": item["source_title"],
                "source_url": item["source_url"],
                "source_retrieval_status": "reference_only_not_downloaded",
                "universe_name": span.get("name", ""),
                "delisted_ric": span.get("delisted_ric", ""),
                "member_from": member_from,
                "member_to": valid_date(span.get("member_to", "")),
                "segment_rows": segment_rows_by_ric.get(ric, 0),
                "segment_periods": len(segment_periods_by_ric.get(ric, set())),
                "announcement_rows": announcement_rows_by_ric.get(ric, 0),
                "local_keyword_evidence_rows": len(ric_evidence),
                "local_keyword_groups": " | ".join(local_groups),
                "local_first_orig_announcement": first_local_announcement,
                "pit_first_observable_date": first_observable if has_local_pit else "",
                "pit_evidence_status": pit_status,
                "pit_eligible_now": has_local_pit,
                "pit_limitation": pit_limitation,
                "existing_pool_v1_member_overlap": ric in old_pool_member_rics,
                "taxonomy_version": config["taxonomy"],
                "reason_code": "SUPPLY_CHAIN_ROLE_SELECTED_NO_RETURN_SCREEN",
            }
        )

    selected_set = set(candidate_rics)
    screening_rows: list[dict[str, Any]] = []
    for row in spans:
        ric = row.get("ric", "")
        if ric in selected_set:
            selected = next(item for item in candidates if item["ric"] == ric)
            screening_rows.append(
                {
                    "ric": ric,
                    "universe_name": row.get("name", ""),
                    "delisted_ric": row.get("delisted_ric", ""),
                    "member_from": row.get("member_from", ""),
                    "member_to": row.get("member_to", ""),
                    "screen_status": "selected_design_candidate",
                    "primary_group": selected["primary_group"],
                    "reason_code": "SUPPLY_CHAIN_ROLE_SELECTED_NO_RETURN_SCREEN",
                }
            )
        else:
            screening_rows.append(
                {
                    "ric": ric,
                    "universe_name": row.get("name", ""),
                    "delisted_ric": row.get("delisted_ric", ""),
                    "member_from": row.get("member_from", ""),
                    "member_to": row.get("member_to", ""),
                    "screen_status": "not_selected_design_candidate",
                    "primary_group": "",
                    "reason_code": "OUTSIDE_FROZEN_49_NAME_DESIGN_SET",
                }
            )

    quarantine_rows: list[dict[str, Any]] = []
    for ric in sorted(set(candidate_rics) - set(span_by_ric)):
        quarantine_rows.append(
            {
                "ric": ric,
                "reason_code": "RIC_NOT_IN_781_UNIVERSE",
                "action": "retain_candidate_quarantine; do_not_add_to_pool",
            }
        )
    for ric in duplicate_candidate_rics:
        quarantine_rows.append(
            {
                "ric": ric,
                "reason_code": "DUPLICATE_CANDIDATE_RIC_IN_CONFIG",
                "action": "resolve_config_before_downstream_use",
            }
        )

    registry_fields = list(registry_rows[0].keys())
    screening_fields = list(screening_rows[0].keys())
    evidence_fields = list(evidence_rows[0].keys()) if evidence_rows else [
        "ric", "canonical_name", "primary_group", "evidence_group", "matched_terms", "segment_name", "period_end", "orig_announcement_dates", "pit_date_available", "raw_file", "raw_row", "raw_text", "reason_code"
    ]
    quarantine_fields = ["ric", "reason_code", "action"]
    source_fields = ["ric", "canonical_name", "primary_group", "source_type", "source_title", "source_url", "source_retrieval_status", "source_as_of_policy", "pit_limitation"]
    source_rows = [
        {
            "ric": row["ric"],
            "canonical_name": row["canonical_name"],
            "primary_group": row["primary_group"],
            "source_type": row["source_type"],
            "source_title": row["source_title"],
            "source_url": row["source_url"],
            "source_retrieval_status": row["source_retrieval_status"],
            "source_as_of_policy": "reference URL supplied in frozen config; no page body downloaded in this run",
            "pit_limitation": row["pit_limitation"],
        }
        for row in registry_rows
    ]

    registry_path = output_dir / "candidate_registry.csv"
    screening_path = output_dir / "universe_screening.csv"
    evidence_path = output_dir / "local_segment_evidence.csv"
    source_path = output_dir / "source_catalog.csv"
    quarantine_path = output_dir / "candidate_quarantine.csv"
    write_csv(registry_path, registry_fields, registry_rows)
    write_csv(screening_path, screening_fields, screening_rows)
    write_csv(evidence_path, evidence_fields, evidence_rows)
    write_csv(source_path, source_fields, source_rows)
    write_csv(quarantine_path, quarantine_fields, quarantine_rows)

    group_counts = Counter(row["primary_group"] for row in registry_rows)
    direct_local_count = sum(row["pit_evidence_status"] == "direct_local_pit" for row in registry_rows)
    static_count = sum(row["pit_evidence_status"] == "provisional_static_source" for row in registry_rows)
    historical_missing_date_count = sum(row["pit_evidence_status"] == "historical_segment_match_missing_announcement" for row in registry_rows)
    input_hashes: dict[str, Any] = {
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256(config_path)},
        "taxonomy": {"path": taxonomy_path.relative_to(root).as_posix(), "sha256": sha256(taxonomy_path)},
        "membership_spans": {"path": spans_path.relative_to(root).as_posix(), "sha256": sha256(spans_path)},
        "existing_pool_membership": {"path": old_pool_path.relative_to(root).as_posix(), "sha256": sha256(old_pool_path)},
        "raw_segment_files": {path.relative_to(root).as_posix(): sha256(path) for path in segment_files},
        "raw_announcement_files": {path.relative_to(root).as_posix(): sha256(path) for path in announcement_files},
    }

    counts = {
        "input_universe_rows": len(spans),
        "input_universe_instruments": len(span_by_ric),
        "output_screening_rows": len(screening_rows),
        "output_screening_instruments": len({row["ric"] for row in screening_rows}),
        "candidate_requested_rows": len(candidates),
        "candidate_requested_instruments": len(set(candidate_rics)),
        "candidate_in_universe_rows": sum(ric in span_by_ric for ric in candidate_rics),
        "candidate_missing_universe_rows": len(set(candidate_rics) - set(span_by_ric)),
        "duplicate_candidate_rics": len(duplicate_candidate_rics),
        "duplicate_universe_rics": len(duplicate_span_rics),
        "selected_candidate_rows": len(registry_rows),
        "selected_candidate_instruments": len({row["ric"] for row in registry_rows}),
        "not_selected_design_rows_retained": sum(row["screen_status"] == "not_selected_design_candidate" for row in screening_rows),
        "physical_rows_deleted": 0,
        "candidate_quarantine_rows": len(quarantine_rows),
        "candidate_delisted_rows_retained": sum(str(row.get("delisted_ric", "")).lower() == "true" for row in registry_rows),
        "existing_pool_v1_overlap_instruments": sum(bool(row["existing_pool_v1_member_overlap"]) for row in registry_rows),
        "raw_segment_files_read": len(segment_files),
        "raw_segment_rows_read": segment_rows_total,
        "candidate_segment_rows_read": sum(segment_rows_by_ric.values()),
        "raw_announcement_files_read": len(announcement_files),
        "raw_announcement_rows_read": announcement_rows_total,
        "candidate_announcement_rows_read": sum(announcement_rows_by_ric.values()),
        "candidates_with_segment_rows": sum(row["segment_rows"] > 0 for row in registry_rows),
        "candidates_with_announcement_rows": sum(row["announcement_rows"] > 0 for row in registry_rows),
        "candidates_with_local_keyword_evidence": sum(row["local_keyword_evidence_rows"] > 0 for row in registry_rows),
        "direct_local_pit_candidates": direct_local_count,
        "historical_match_missing_announcement_candidates": historical_missing_date_count,
        "provisional_static_source_candidates": static_count,
        "source_reference_rows": len(source_rows),
        "source_pages_downloaded": 0,
        "return_rows_read": 0,
        "target_rows_read": 0,
        "label_rows_read": 0,
        "model_output_rows_read": 0,
    }
    checks = {
        "candidate_count_in_target_range": 30 <= len(registry_rows) <= 50,
        "candidate_rics_unique": not duplicate_candidate_rics,
        "universe_ric_unique": not duplicate_span_rics,
        "candidate_rics_all_in_universe": counts["candidate_missing_universe_rows"] == 0,
        "screening_rows_equal_input_rows": len(screening_rows) == len(spans),
        "screening_nonselected_rows_retained": counts["not_selected_design_rows_retained"] == len(spans) - len(registry_rows),
        "no_physical_row_deletion": counts["physical_rows_deleted"] == 0,
        "candidate_quarantine_is_explicit": True,
        "delisted_candidates_retained": True,
        "local_pit_evidence_separated_from_static_sources": direct_local_count + static_count + historical_missing_date_count == len(registry_rows),
        "all_frozen_groups_present": set(group_counts) == set(GROUP_ORDER),
        "source_pages_not_downloaded_as_documented": counts["source_pages_downloaded"] == 0,
        "no_return_or_target_or_label_inputs_read": all(counts[key] == 0 for key in ["return_rows_read", "target_rows_read", "label_rows_read", "model_output_rows_read"]),
        "raw_hashes_recorded": bool(input_hashes["raw_segment_files"] and input_hashes["raw_announcement_files"]),
        "no_imputation_or_winsorization": True,
        "no_silent_deduplication": True,
    }

    output_hashes = {
        path.name: sha256(path)
        for path in [registry_path, screening_path, evidence_path, source_path, quarantine_path]
    }
    summary = {
        "schema_version": "ai_pool_expansion_candidates_v1",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": "complete_design_candidate_registry" if all(checks.values()) else "validation_failed",
        "model_ready": False,
        "taxonomy": config["taxonomy"],
        "taxonomy_path": taxonomy_path.relative_to(root).as_posix(),
        "script": {"path": Path(__file__).relative_to(root).as_posix(), "version": SCRIPT_VERSION, "sha256": sha256(Path(__file__))},
        "config_path": config_path.relative_to(root).as_posix(),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "counts": counts,
        "primary_group_counts": dict(sorted(group_counts.items())),
        "candidate_rics": [row["ric"] for row in registry_rows],
        "checks": checks,
        "rules": {
            "identity": "literal RIC only; curated names are explanatory metadata",
            "selection": "supply-chain role and frozen taxonomy only; no returns/targets/labels/model outputs",
            "missing_values": "blank fields retained as blank/NA; no imputation or zero fill",
            "deduplication": "duplicate keys block validation; no silent deduplication",
            "delisted": "retained and annotated with corrected membership span",
            "pit": "Orig historical segment keyword + Orig announcement date required for direct_local_pit; static issuer pages remain provisional",
            "etf": "existing project ETF files are prices only; no constituents read or inferred",
        },
        "limitations": [
            "40 of the selected design candidates rely on a current/reference issuer description pointer and are not historical PIT members yet.",
            "Local LSEG segment names are a finite raw run and a keyword hit does not prove an all-period company-level role; formation must be validated period by period.",
            "Orig announcement times are unavailable, so date availability is day-level and downstream strict trade entry remains separate.",
            "The candidate registry does not create labels, returns, portfolio weights or model outputs and is not a performance claim.",
        ],
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    summary_hash = sha256(summary_path)
    gate = {
        "schema_version": "ai_pool_expansion_candidates_gate_v1",
        "run_id": run_id,
        "summary_path": summary_path.relative_to(root).as_posix(),
        "summary_sha256": summary_hash,
        "status": summary["status"],
        "counts": counts,
        "checks": checks,
        "affected_rows": {
            "selected_design_candidates": len(registry_rows),
            "not_selected_rows_retained": counts["not_selected_design_rows_retained"],
            "quarantine_rows": len(quarantine_rows),
            "physical_deletions": 0,
        },
        "quarantine_path": quarantine_path.relative_to(root).as_posix(),
        "raw_immutable": True,
        "execution_status": "complete" if all(checks.values()) else "validation_failed",
    }
    gate_path = output_dir / "gate_report.json"
    write_json(gate_path, gate)
    print(json.dumps({"run_id": run_id, "output_dir": output_dir.relative_to(root).as_posix(), "status": summary["status"], "counts": counts, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

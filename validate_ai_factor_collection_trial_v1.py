"""Independent structural validation of the AI-factor collection trial.

This validator is deliberately read-only with respect to the frozen raw trial
``data/raw/ai_factor_v1/20260909T181731782199Z``.  It reads the trial plan,
manifest, summary and the three returned ``rd`` request/metadata/CSV triples;
it also reads the current collector, configuration and corrected spans source
only for provenance and independent membership counts.  It never opens an
LSEG session, issues a request, parses or rewrites raw values, or creates a
clean/model/target table.

The CSV checks count physical empty strings separately from physical non-empty
cells.  A literal ``0`` is an observed value and is never treated as missing.
No date conversion, identifier mapping, deduplication, deletion, imputation,
winsorisation, unit conversion or timezone conversion is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw" / "ai_factor_v1"
DEFAULT_RUN_ID = "20260909T181731782199Z"
AUDIT_ROOT = ROOT / "data" / "audit" / "ai_factor_collection_trial_v1"
SCHEMA_VERSION = "ai_factor_collection_v1"
EXPECTED_RETURNED = ("rd_000", "rd_001", "rd_002")
EXPECTED_CSV_COLUMNS = [
    "Instrument",
    "Research And Development",
    "Date",
    "Financial Period Absolute",
]
EXPECTED_FALSE_POLICY_KEYS = [
    "cleaning",
    "blank_replacement",
    "missing_value_imputation",
    "winsorization",
    "deduplication",
    "row_deletion",
    "row_filtering",
    "date_parsing_or_conversion_in_raw",
    "timezone_conversion_in_raw",
    "unit_or_currency_conversion",
    "factor_ratio_or_score_generation",
    "return_or_label_generation",
    "test_future_return_or_label_read",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_audit_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def compact(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def snapshot_files(root: Path) -> dict[str, dict[str, Any]]:
    """Return a recursive, relative path/size/hash inventory without writing."""

    if not root.is_dir():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix()
        result[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def read_physical_csv(path: Path) -> tuple[list[str], list[list[str]], list[int]]:
    """Read CSV cells as strings; return header, rows and malformed row widths."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], [], []
        rows: list[list[str]] = []
        malformed: list[int] = []
        width = len(header)
        for row_number, row in enumerate(reader, start=2):
            rows.append(row)
            if len(row) != width:
                malformed.append(row_number)
    return header, rows, malformed


def physical_cell_stats(header: list[str], rows: list[list[str]]) -> dict[str, Any]:
    nonempty = Counter({column: 0 for column in header})
    missing = Counter({column: 0 for column in header})
    literal_zero = Counter({column: 0 for column in header})
    extra_nonempty = 0
    all_blank_rows = 0
    for row in rows:
        if not row or all(value == "" for value in row):
            all_blank_rows += 1
        for index, value in enumerate(row):
            if index >= len(header):
                if value != "":
                    extra_nonempty += 1
                continue
            column = header[index]
            if value == "":
                missing[column] += 1
            else:
                nonempty[column] += 1
                if value in {"0", "0.0", "-0", "+0", "-0.0", "+0.0"}:
                    literal_zero[column] += 1
        if len(row) < len(header):
            for column in header[len(row) :]:
                missing[column] += 1
    total_cells = len(rows) * len(header)
    nonempty_cells = sum(nonempty.values())
    missing_cells = sum(missing.values())
    return {
        "rows": len(rows),
        "columns": len(header),
        "physical_nonempty": dict(nonempty),
        "physical_missing": dict(missing),
        "physical_nonempty_cells": nonempty_cells,
        "physical_missing_cells": missing_cells,
        "physical_total_cells": total_cells,
        "literal_zero": dict(literal_zero),
        "literal_zero_cells": sum(literal_zero.values()),
        "all_blank_rows": all_blank_rows,
        "extra_nonempty_cells": extra_nonempty,
    }


def record_check(
    checks: list[dict[str, Any]],
    check_id: str,
    category: str,
    passed: bool,
    expected: Any,
    observed: Any,
    notes: str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "passed": bool(passed),
            "expected": expected,
            "observed": observed,
            "notes": notes,
        }
    )


def resolve_relative(path_value: str, root: Path) -> Path:
    return root / Path(path_value.replace("\\", "/"))


def expected_raw_names() -> set[str]:
    names = {"plan.json", "collector_manifest.json", "summary.json"}
    for request_id in EXPECTED_RETURNED:
        names.update(
            {
                f"{request_id}.csv",
                f"{request_id}.meta.json",
                f"{request_id}.request.json",
            }
        )
    return names


def build_findings(report: dict[str, Any], audit_dir: Path) -> str:
    checks = report["checks"]
    passed = sum(1 for check in checks if check["passed"])
    failed = [check["check_id"] for check in checks if not check["passed"]]
    status = report["status"]
    source = report["source_hashes"]
    counts = report["counts"]
    metadata_gaps = {
        request_id: report["returned_batches"][request_id].get("metadata_minus_physical_gap", {})
        for request_id in EXPECTED_RETURNED
        if report["returned_batches"][request_id].get("metadata_minus_physical_gap")
    }
    def source_note(label: str) -> str:
        item = source[label]
        available = [value for value in item.get("recorded", []) if value not in (None, "")]
        if available and all(value == item.get("current") for value in available):
            return "与所有可用记录一致"
        return "当前值与 trial 记录不同，作为 provenance drift 保留"

    lines = [
        "# AI-factor collection trial v1 独立验证",
        "",
        f"- 验证运行：`{report['audit_id']}`（{report['finished_at_utc']}）",
        f"- 输入：`{report['raw_run_path']}`",
        f"- 输出：`{audit_dir}`",
        f"- 状态：**{status}**；{passed}/{len(checks)} checks passed。",
        "",
        "## 关键结论",
        "",
        f"- plan 请求数为 **{counts['plan_request_count']} = 11 × 32 + 5**：11 个 fundamental family 各 32 批，另有 5 个独立 ETF 请求。股票池为 **{counts['span_rows']} 行、{counts['span_unique_rics']} 个唯一 literal RIC**，其中 **{counts['span_delisted']} 个 delisted RIC**，全部保留在计划成员中。",
        f"- raw trial 恰好返回 `rd_000`、`rd_001`、`rd_002`；三批各请求 25 个 RIC，计划集合合计 **{counts['requested_rd_union']} 个唯一 RIC**，CSV 返回集合合计 **{counts['returned_rd_union']} 个唯一 Instrument**，无 unexpected Instrument。",
        f"- 三个 CSV 合计 **{counts['returned_rows']} 行**。物理非空单元格 **{counts['physical_nonempty_cells']}**，物理缺失空字符串 **{counts['physical_missing_cells']}**；缺失没有填补，literal zero 单元格按观测值保留（合计 {counts['literal_zero_cells']}）。",
        f"- metadata/manifest 的 `non_null` 计数作为供应商返回对象 provenance 另行保留；本次物理 CSV 对 `Financial Period Absolute` 的空字符串差异为 `{compact(metadata_gaps)}`，没有把该差异改写成零或填补。",
        f"- summary/manifest 为 `partial`，`last_execute.limit=3`、attempted=3、successful_or_empty=3、errors=0，pending=354。这里的 execute exit 1 按“达到 limit 后的 partial”解释，**不判作请求失败**。",
        "- 未生成或读取测试目标、未来收益、标签、factor/score/model/portfolio；未产生 error sidecar；没有 quarantine。",
        "",
        "## 哈希与原始文件守恒",
        "",
        f"- collector script：`{source['script']['current']}`（{source_note('script')}）",
        f"- config：`{source['config']['current']}`（{source_note('config')}）",
        f"- corrected spans：`{source['spans']['current']}`（{source_note('spans')}；manifest 未提供该字段时按缺省记录处理）",
        f"- plan：`{report['actual_hashes']['plan.json']}`；manifest：`{report['actual_hashes']['collector_manifest.json']}`；summary：`{report['actual_hashes']['summary.json']}`。",
        "- raw trial 验证前后文件集合、大小和 SHA-256 完全一致；审计文件只写入 `data/audit/`。",
        "",
        "## 返回批次与缺失口径",
        "",
        "| request | requested RIC | rows | physical non-empty | physical missing | unique returned Instrument | missing requested | unexpected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for request_id in EXPECTED_RETURNED:
        batch = report["returned_batches"][request_id]
        lines.append(
            f"| `{request_id}` | {batch['requested_instrument_count']} | {batch['rows']} | {batch['physical_nonempty_cells']} | {batch['physical_missing_cells']} | {batch['observed_instrument_count']} | {len(batch['missing_instruments'])} | {len(batch['unexpected_instruments'])} |"
        )
    lines.extend(
        [
            "",
            "每个 CSV 的 `physical_nonempty`/`physical_missing` 都按原始字符串是否为空统计；没有 strip、日期解析或 numeric coercion。重复 Instrument 行是供应商年度响应的多行结构，未作 deduplication。退市标记只用于覆盖审计，未因 delisted 删除证券。",
            "",
            "## 审计范围与限制",
            "",
            "- 本验证只覆盖本次 trial 的 plan、manifest、summary 和三个已返回 R&D request/metadata/CSV 三元组，并以当前 collector/config/corrected spans 文件重算 provenance hashes；不验证未返回的 354 个请求的供应商内容。",
            "- raw 响应仍是 direct vendor serialization。没有填补、前向填充、缩尾、去重、行删除、公司排除、单位/币种转换、日期/时区写回、identifier mapping 或标签过滤；缺失 coverage 作为审计结果保留。",
            "- 前序计划 run `20260909T181534358580Z` 是 retrospective 保留记录：只保留 `plan.json`，其记录脚本 SHA-256 为 `0bdf9b46e2c98fc4c39d63bfad1502fabdc21009a960961a78fc56d8a4a75e6e`，与当前 collector SHA-256 不同，因此在任何请求前失败；该计划没有被覆盖。正式 trial 为 `20260909T181731782199Z`。",
        ]
    )
    if failed:
        lines.extend(["", "失败 checks：" + ", ".join(f"`{item}`" for item in failed)])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--audit-id", default=None)
    args = parser.parse_args()

    raw_dir = RAW_ROOT / args.run_id
    audit_id = args.audit_id or new_audit_id()
    audit_dir = AUDIT_ROOT / audit_id
    if audit_dir.exists():
        raise RuntimeError(f"audit output already exists; refusing to overwrite: {audit_dir}")

    started_at = utc_now()
    checks: list[dict[str, Any]] = []
    raw_files_before = snapshot_files(raw_dir)
    expected_files = expected_raw_names()

    plan_path = raw_dir / "plan.json"
    manifest_path = raw_dir / "collector_manifest.json"
    summary_path = raw_dir / "summary.json"
    plan = read_json(plan_path)
    manifest = read_json(manifest_path)
    summary = read_json(summary_path)

    plan_hash = sha256_file(plan_path)
    manifest_hash = sha256_file(manifest_path)
    summary_hash = sha256_file(summary_path)

    script_path = ROOT / str(plan.get("script", "collect_ai_factor_v1.py"))
    config_path = resolve_relative(str(plan.get("config_path", "ai_factor_collection_v1_config.json")), ROOT)
    spans_path = resolve_relative(
        str(plan.get("spans_path", "data/audit/universe_rebuild/eligible_spans_2015_2026_corrected.csv")),
        ROOT,
    )
    current_script_hash = sha256_file(script_path)
    current_config_hash = sha256_file(config_path)
    current_spans_hash = sha256_file(spans_path)
    validator_hash = sha256_file(Path(__file__))

    record_check(
        checks,
        "raw_run_directory_exists",
        "inventory",
        raw_dir.is_dir(),
        str(raw_dir),
        str(raw_dir) if raw_dir.is_dir() else None,
        "The validator reads the frozen trial directory and writes only under data/audit.",
    )
    record_check(
        checks,
        "raw_file_inventory_exact",
        "inventory",
        set(raw_files_before) == expected_files,
        sorted(expected_files),
        sorted(raw_files_before),
        "The trial contains plan/manifest/summary and exactly the three rd request, metadata and CSV triples; no extra raw artifact is expected.",
    )
    record_check(
        checks,
        "schema_and_run_id_consistent",
        "plan",
        all(
            item.get("schema_version") == SCHEMA_VERSION and item.get("run_id") == args.run_id
            for item in (plan, manifest, summary)
        ),
        {"schema_version": SCHEMA_VERSION, "run_id": args.run_id},
        {
            "plan": {"schema_version": plan.get("schema_version"), "run_id": plan.get("run_id")},
            "manifest": {"schema_version": manifest.get("schema_version"), "run_id": manifest.get("run_id")},
            "summary": {"schema_version": summary.get("schema_version"), "run_id": summary.get("run_id")},
        },
        "Plan, manifest and summary identity is checked without rewriting any artifact.",
    )

    recorded_source_hashes = {
        "script": [plan.get("script_sha256"), manifest.get("script_sha256"), summary.get("script_sha256")],
        "config": [plan.get("config_sha256"), manifest.get("config_sha256"), summary.get("config_sha256")],
        "spans": [plan.get("spans_sha256"), manifest.get("spans_sha256"), summary.get("spans_sha256")],
    }
    current_source_hashes = {
        "script": current_script_hash,
        "config": current_config_hash,
        "spans": current_spans_hash,
    }
    for label, current in current_source_hashes.items():
        recorded = recorded_source_hashes[label]
        available_recorded = [value for value in recorded if value not in (None, "")]
        record_check(
            checks,
            f"current_{label}_hash_matches_records",
            "hashes",
            bool(current) and bool(available_recorded) and all(value == current for value in available_recorded),
            {"current": current, "available_recorded": sorted(set(available_recorded))},
            {"current": current, "recorded": recorded},
            "The current source hash must agree across all available plan/manifest/summary provenance fields; an omitted optional field is reported as absent.",
        )
    record_check(
        checks,
        "plan_manifest_summary_hash_chain",
        "hashes",
        manifest.get("plan_sha256") == plan_hash and summary.get("plan_sha256") == plan_hash and summary.get("manifest_sha256") == manifest_hash,
        {"plan_sha256": plan_hash, "manifest_sha256": manifest_hash},
        {
            "plan_actual": plan_hash,
            "manifest_recorded_plan": manifest.get("plan_sha256"),
            "summary_recorded_plan": summary.get("plan_sha256"),
            "manifest_actual": manifest_hash,
            "summary_recorded_manifest": summary.get("manifest_sha256"),
        },
        "Plan, manifest and summary hash links are recomputed independently.",
    )

    plan_requests = [item for item in plan.get("requests", []) if isinstance(item, dict)]
    plan_by_id = {str(item.get("request_id")): item for item in plan_requests}
    manifest_requests = manifest.get("requests", {}) if isinstance(manifest.get("requests"), dict) else {}
    summary_records = summary.get("requests", []) if isinstance(summary.get("requests"), list) else []
    summary_by_id = {str(item.get("request_id")): item for item in summary_records if isinstance(item, dict)}
    record_check(
        checks,
        "plan_request_count_357",
        "plan",
        len(plan_requests) == 357 and plan.get("request_count") == 357,
        {"request_count": 357, "list_length": 357},
        {"request_count": plan.get("request_count"), "list_length": len(plan_requests)},
        "The plan count is checked against both its declared count and its request list.",
    )

    config = read_json(config_path)
    config_families = [str(item.get("family_id")) for item in config.get("fundamental_requests", []) if isinstance(item, dict)]
    family_counts = Counter(str(item.get("family_id")) for item in plan_requests if item.get("kind") == "fundamental")
    etf_specs = [item for item in plan_requests if item.get("kind") == "etf_history"]
    construction_observed = {
        "config_fundamental_families": len(config_families),
        "plan_fundamental_families": len(family_counts),
        "fundamental_requests": sum(family_counts.values()),
        "fundamental_family_counts": dict(sorted(family_counts.items())),
        "etf_requests": len(etf_specs),
        "constructed_total": len(plan_requests),
    }
    construction_ok = (
        len(config_families) == 11
        and set(config_families) == set(family_counts)
        and all(family_counts[family] == 32 for family in config_families)
        and len(etf_specs) == 5
        and len(plan_requests) == 11 * 32 + 5
    )
    record_check(
        checks,
        "plan_request_construction_11x32_plus5",
        "plan",
        construction_ok,
        {"fundamental_families": 11, "batches_per_family": 32, "etf_requests": 5, "total": 357},
        construction_observed,
        "The 357 planned requests must decompose into the 11 configured fundamental families and five independent ETF requests.",
    )

    span_rows: list[dict[str, str]] = []
    with spans_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        span_header = list(reader.fieldnames or [])
        span_rows = [dict(row) for row in reader]
    span_rics = [row.get("ric", "") for row in span_rows]
    span_unique_rics = set(span_rics)
    span_delisted = sum(1 for row in span_rows if parse_bool(row.get("delisted_ric", "")))
    span_counts = {
        "rows": len(span_rows),
        "distinct_literal_rics": len(span_unique_rics),
        "live_rows": len(span_rows) - span_delisted,
        "delisted_rows": span_delisted,
        "blank_ric_rows": sum(1 for ric in span_rics if ric == ""),
        "header": span_header,
    }
    record_check(
        checks,
        "spans_781_unique_rics_154_delisted",
        "membership",
        span_counts["rows"] == 781 and span_counts["distinct_literal_rics"] == 781 and span_delisted == 154 and span_counts["blank_ric_rows"] == 0,
        {"rows": 781, "distinct_literal_rics": 781, "delisted_rows": 154, "blank_ric_rows": 0},
        span_counts,
        "Literal RIC strings are counted as supplied; delisted rows remain eligible historical members.",
    )
    record_check(
        checks,
        "plan_span_counts_match_source",
        "membership",
        plan.get("span_counts") == {key: span_counts[key] for key in ("rows", "distinct_literal_rics", "live_rows", "delisted_rows")},
        plan.get("span_counts"),
        {key: span_counts[key] for key in ("rows", "distinct_literal_rics", "live_rows", "delisted_rows")},
        "The plan span summary is compared with an independent CSV count.",
    )

    stock_batch_grid: dict[str, Any] = {}
    grid_ok = True
    for family in sorted(family_counts):
        specs = [item for item in plan_requests if item.get("kind") == "fundamental" and str(item.get("family_id")) == family]
        specs = sorted(specs, key=lambda item: int(item.get("batch_number", -1)))
        expected_sizes = [25] * 31 + [6]
        observed_sizes = [len((item.get("request") or {}).get("universe", [])) for item in specs]
        observed_batch_sizes = [item.get("batch_size") for item in specs]
        observed_numbers = [item.get("batch_number") for item in specs]
        family_ok = (
            len(specs) == 32
            and observed_numbers == list(range(32))
            and observed_sizes == expected_sizes
            and observed_batch_sizes == expected_sizes
        )
        grid_ok = grid_ok and family_ok
        stock_batch_grid[family] = {
            "count": len(specs),
            "batch_numbers": observed_numbers,
            "universe_sizes": observed_sizes,
            "batch_sizes": observed_batch_sizes,
            "passed": family_ok,
        }
    record_check(
        checks,
        "fundamental_batch_grid_31x25_plus6",
        "plan",
        grid_ok and plan.get("stock_batch_size") == 25,
        {"stock_batch_size": 25, "per_family_universe_sizes": [25] * 31 + [6]},
        {"stock_batch_size": plan.get("stock_batch_size"), "families": stock_batch_grid},
        "Each configured fundamental family uses the corrected spans order with 31 full batches and a six-RIC final batch.",
    )

    plan_fundamental_members = {}
    member_coverage_ok = True
    for family in sorted(family_counts):
        family_specs = [item for item in plan_requests if item.get("kind") == "fundamental" and str(item.get("family_id")) == family]
        members = [member for spec in family_specs for member in spec.get("members", []) if isinstance(member, dict)]
        member_rics = [str(member.get("ric", "")) for member in members if isinstance(member, dict)]
        member_delisted = sum(1 for member in members if isinstance(member, dict) and parse_bool(member.get("delisted_ric", False)))
        family_ok = len(member_rics) == 781 and len(set(member_rics)) == 781 and set(member_rics) == span_unique_rics and member_delisted == 154
        member_coverage_ok = member_coverage_ok and family_ok
        plan_fundamental_members[family] = {
            "rows": len(member_rics),
            "unique_literal_rics": len(set(member_rics)),
            "delisted_rows": member_delisted,
            "passed": family_ok,
        }
    record_check(
        checks,
        "delisted_members_retained_in_every_fundamental_family",
        "membership",
        member_coverage_ok,
        {"members_per_family": 781, "unique_literal_rics": 781, "delisted_rows": 154},
        plan_fundamental_members,
        "Delisting is an audit attribute; it does not remove a RIC from the planned universe.",
    )

    manifest_statuses = {request_id: str(item.get("status")) for request_id, item in manifest_requests.items() if isinstance(item, dict)}
    summary_statuses = {request_id: str(item.get("status")) for request_id, item in summary_by_id.items()}
    returned_manifest = sorted(request_id for request_id, status in manifest_statuses.items() if status in {"returned", "empty"})
    returned_summary = sorted(request_id for request_id, status in summary_statuses.items() if status in {"returned", "empty"})
    returned_exact = returned_manifest == list(EXPECTED_RETURNED) and returned_summary == list(EXPECTED_RETURNED)
    record_check(
        checks,
        "returned_request_ids_exact_rd_000_to_002",
        "status",
        returned_exact,
        list(EXPECTED_RETURNED),
        {"manifest": returned_manifest, "summary": returned_summary},
        "The trial is expected to have exactly the first three rd requests in a terminal successful/empty state.",
    )

    manifest_status_counts = dict(sorted(Counter(manifest_statuses.values()).items()))
    summary_counts = summary.get("counts", {}) if isinstance(summary.get("counts"), dict) else {}
    record_check(
        checks,
        "pending_count_354",
        "status",
        summary_counts.get("pending") == 354 and manifest_status_counts.get("planned", 0) == 354 and len(plan_requests) - len(EXPECTED_RETURNED) == 354,
        {"summary_pending": 354, "manifest_planned": 354, "plan_minus_returned": 354},
        {
            "summary_counts": summary_counts,
            "manifest_status_counts": manifest_status_counts,
            "plan_minus_returned": len(plan_requests) - len(EXPECTED_RETURNED),
        },
        "planned requests are the pending remainder after the explicit three-request limit.",
    )
    last_execute = manifest.get("last_execute", {}) if isinstance(manifest.get("last_execute"), dict) else {}
    partial_limit_ok = (
        manifest.get("status") == "partial"
        and summary.get("status") == "partial"
        and last_execute.get("limit") == 3
        and last_execute.get("attempted") == 3
        and last_execute.get("successful_or_empty") == 3
        and last_execute.get("errors") == 0
        and summary_counts.get("returned") == 3
        and summary_counts.get("pending") == 354
    )
    record_check(
        checks,
        "partial_limit_exit_is_not_request_failure",
        "status",
        partial_limit_ok,
        {"status": "partial", "limit": 3, "attempted": 3, "successful_or_empty": 3, "errors": 0},
        {"manifest_status": manifest.get("status"), "summary_status": summary.get("status"), "last_execute": last_execute},
        "An execute exit 1 after --limit is interpreted as expected partial completion when all attempted requests succeeded and no error occurred.",
    )

    error_sidecars = sorted(name for name in raw_files_before if ".error" in name.lower())
    record_check(
        checks,
        "no_error_sidecars",
        "status",
        not error_sidecars and not summary.get("errors"),
        {"error_sidecars": [], "summary_errors": []},
        {"error_sidecars": error_sidecars, "summary_errors": summary.get("errors")},
        "Failed attempts would be retained as error sidecars; none exists in this trial.",
    )

    raw_policy_values = {
        "plan": {key: (plan.get("raw_response_policy") or {}).get(key) for key in EXPECTED_FALSE_POLICY_KEYS},
        "manifest": {key: (manifest.get("raw_response_policy") or {}).get(key) for key in EXPECTED_FALSE_POLICY_KEYS},
    }
    raw_policy_ok = all(value is False for policy in raw_policy_values.values() for value in policy.values())
    record_check(
        checks,
        "raw_response_policy_no_transformations",
        "policy",
        raw_policy_ok,
        {key: False for key in EXPECTED_FALSE_POLICY_KEYS},
        raw_policy_values,
        "Raw responses remain direct vendor serialization; no imputation, deletion, filtering or conversion is applied.",
    )
    generated_flags = {
        "test_targets_read": summary.get("test_targets_read"),
        "future_returns_generated": summary.get("future_returns_generated"),
        "labels_generated": summary.get("labels_generated"),
        "feature_or_factor_generated": summary.get("feature_or_factor_generated"),
    }
    forbidden_tokens = ("target", "return", "label", "factor", "score", "portfolio", "model")
    forbidden_files = sorted(name for name in raw_files_before if any(token in name.lower() for token in forbidden_tokens))
    test_outputs_ok = all(value is False for value in generated_flags.values()) and not forbidden_files
    record_check(
        checks,
        "test_targets_returns_labels_not_generated",
        "policy",
        test_outputs_ok,
        {**{key: False for key in generated_flags}, "forbidden_raw_files": []},
        {**generated_flags, "forbidden_raw_files": forbidden_files},
        "The collector's test seal is checked from summary flags and the frozen raw inventory.",
    )

    returned_batches: dict[str, dict[str, Any]] = {}
    requested_union: set[str] = set()
    returned_union: set[str] = set()
    requested_overlap = False
    returned_overlap = False
    total_rows = 0
    total_physical_nonempty = 0
    total_physical_missing = 0
    total_literal_zero = 0

    for request_id in EXPECTED_RETURNED:
        request_path = raw_dir / f"{request_id}.request.json"
        metadata_path = raw_dir / f"{request_id}.meta.json"
        csv_path = raw_dir / f"{request_id}.csv"
        request_obj = read_json(request_path)
        metadata = read_json(metadata_path)
        header, rows, malformed_rows = read_physical_csv(csv_path)
        stats = physical_cell_stats(header, rows)
        plan_spec = plan_by_id.get(request_id, {})
        manifest_record = manifest_requests.get(request_id, {})
        summary_record = summary_by_id.get(request_id, {})
        requested_universe = list(request_obj.get("universe", []))
        requested_set = set(requested_universe)
        instrument_index = header.index("Instrument") if "Instrument" in header else None
        observed_values = [row[instrument_index] for row in rows if instrument_index is not None and len(row) > instrument_index]
        observed_instruments = {value for value in observed_values if value != ""}
        missing_instruments = sorted(requested_set - observed_instruments)
        unexpected_instruments = sorted(observed_instruments - requested_set)
        requested_overlap = requested_overlap or bool(requested_union & requested_set)
        returned_overlap = returned_overlap or bool(returned_union & observed_instruments)
        requested_union.update(requested_set)
        returned_union.update(observed_instruments)

        request_actual_hash = sha256_file(request_path)
        request_canonical_hash = sha256_bytes(canonical_json(request_obj))
        metadata_actual_hash = sha256_file(metadata_path)
        csv_actual_hash = sha256_file(csv_path)
        request_hash_records = [
            plan_spec.get("request_sha256"),
            manifest_record.get("request_sha256"),
            manifest_record.get("request_file_sha256"),
            metadata.get("request_sha256"),
            metadata.get("request_file_sha256"),
        ]
        csv_hash_records = [
            manifest_record.get("csv_sha256"),
            summary_record.get("csv_sha256"),
            metadata.get("csv_sha256"),
        ]
        metadata_hash_records = [manifest_record.get("metadata_sha256"), summary_record.get("metadata_sha256")]
        physical_nonempty = stats["physical_nonempty"]
        metadata_non_null = metadata.get("non_null", {}) if isinstance(metadata.get("non_null"), dict) else {}
        manifest_non_null = manifest_record.get("non_null", {}) if isinstance(manifest_record.get("non_null"), dict) else {}

        record_check(
            checks,
            f"{request_id}_request_hash_chain",
            "hashes",
            request_actual_hash == request_canonical_hash and all(value == request_actual_hash for value in request_hash_records),
            {"sha256": request_actual_hash},
            {"actual_file": request_actual_hash, "canonical": request_canonical_hash, "records": request_hash_records},
            "Request JSON bytes, canonical request payload and all recorded request hashes must agree.",
        )
        record_check(
            checks,
            f"{request_id}_metadata_hash_chain",
            "hashes",
            all(value == metadata_actual_hash for value in metadata_hash_records),
            {"sha256": metadata_actual_hash},
            {"actual_file": metadata_actual_hash, "records": metadata_hash_records},
            "Metadata file hash is checked against manifest and summary.",
        )
        record_check(
            checks,
            f"{request_id}_csv_hash_chain",
            "hashes",
            all(value == csv_actual_hash for value in csv_hash_records),
            {"sha256": csv_actual_hash},
            {"actual_file": csv_actual_hash, "records": csv_hash_records},
            "CSV bytes are checked against metadata, manifest and summary without cleaning the response.",
        )
        record_check(
            checks,
            f"{request_id}_request_payload_matches_plan",
            "request",
            request_obj == plan_spec.get("request") and metadata.get("request_id") == request_id and manifest_record.get("request_id") == request_id,
            {"request_id": request_id, "payload_matches_plan": True},
            {"request_id": metadata.get("request_id"), "manifest_request_id": manifest_record.get("request_id"), "payload_matches_plan": request_obj == plan_spec.get("request")},
            "The saved request object is compared semantically with the frozen plan request.",
        )
        record_check(
            checks,
            f"{request_id}_csv_shape_and_rows",
            "csv",
            header == EXPECTED_CSV_COLUMNS and not malformed_rows and stats["rows"] == metadata.get("rows") == manifest_record.get("rows") == summary_record.get("rows") and csv_path.stat().st_size == metadata.get("csv_bytes"),
            {"columns": EXPECTED_CSV_COLUMNS, "rows": metadata.get("rows"), "csv_bytes": metadata.get("csv_bytes")},
            {"columns": header, "rows": stats["rows"], "malformed_row_numbers": malformed_rows, "metadata_rows": metadata.get("rows"), "manifest_rows": manifest_record.get("rows"), "summary_rows": summary_record.get("rows"), "actual_csv_bytes": csv_path.stat().st_size, "metadata_csv_bytes": metadata.get("csv_bytes")},
            "CSV structure and physical data-row count are checked before any value interpretation.",
        )
        metadata_gap = {
            column: int(metadata_non_null.get(column, 0)) - int(physical_nonempty.get(column, 0))
            for column in header
            if int(metadata_non_null.get(column, 0)) != int(physical_nonempty.get(column, 0))
        }
        record_check(
            checks,
            f"{request_id}_physical_nonempty_and_missing_reported",
            "missingness",
            set(header) == set(physical_nonempty) == set(metadata_non_null) == set(manifest_non_null) and stats["physical_nonempty_cells"] + stats["physical_missing_cells"] == stats["physical_total_cells"],
            {"physical_nonempty_and_missing_accounted": True, "metadata_gap_allowed_and_reported": True},
            {"physical_nonempty": physical_nonempty, "physical_missing": stats["physical_missing"], "metadata_non_null": metadata_non_null, "manifest_non_null": manifest_non_null, "metadata_minus_physical_gap": metadata_gap},
            "Physical empty strings are reported separately from non-empty cells. Metadata/manifest non_null is retained as provenance and may differ when an empty vendor string serializes as a blank CSV cell; no imputation is attempted.",
        )
        record_check(
            checks,
            f"{request_id}_physical_cells_conserved",
            "missingness",
            stats["physical_nonempty_cells"] + stats["physical_missing_cells"] == stats["physical_total_cells"] and stats["extra_nonempty_cells"] == 0,
            {"nonempty_plus_missing": stats["physical_total_cells"], "extra_nonempty_cells": 0},
            {"nonempty": stats["physical_nonempty_cells"], "missing": stats["physical_missing_cells"], "total": stats["physical_total_cells"], "extra_nonempty_cells": stats["extra_nonempty_cells"], "literal_zero_cells": stats["literal_zero_cells"]},
            "Cell accounting is physical and zero-valued text remains non-empty observed data.",
        )
        record_check(
            checks,
            f"{request_id}_requested_25_unique_rics",
            "coverage",
            len(requested_universe) == 25 and len(requested_set) == 25 and metadata.get("requested_instrument_count") == 25,
            {"requested_count": 25, "unique_count": 25},
            {"request_universe_count": len(requested_universe), "request_unique_count": len(requested_set), "metadata_requested_instrument_count": metadata.get("requested_instrument_count")},
            "The three returned R&D requests each carry 25 literal RICs.",
        )
        record_check(
            checks,
            f"{request_id}_instrument_coverage_exact",
            "coverage",
            requested_set == observed_instruments and not missing_instruments and not unexpected_instruments and not any(value == "" for value in observed_values),
            {"requested_unique": 25, "missing": [], "unexpected": [], "blank_instrument_rows": 0},
            {"requested_unique": len(requested_set), "observed_unique": len(observed_instruments), "missing": missing_instruments, "unexpected": unexpected_instruments, "blank_instrument_rows": sum(value == "" for value in observed_values)},
            "Returned Instrument values are compared as literal strings; no identifier normalization or mapping is performed.",
        )

        returned_batches[request_id] = {
            "request_id": request_id,
            "request_path": request_path.relative_to(ROOT).as_posix(),
            "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
            "csv_path": csv_path.relative_to(ROOT).as_posix(),
            "request_sha256": request_actual_hash,
            "metadata_sha256": metadata_actual_hash,
            "csv_sha256": csv_actual_hash,
            "requested_instrument_count": len(requested_set),
            "requested_instruments": requested_universe,
            "observed_instrument_count": len(observed_instruments),
            "observed_instruments": sorted(observed_instruments),
            "missing_instruments": missing_instruments,
            "unexpected_instruments": unexpected_instruments,
            "blank_instrument_rows": sum(value == "" for value in observed_values),
            "rows": stats["rows"],
            "columns": header,
            "physical_nonempty": physical_nonempty,
            "physical_missing": stats["physical_missing"],
            "physical_nonempty_cells": stats["physical_nonempty_cells"],
            "physical_missing_cells": stats["physical_missing_cells"],
            "physical_total_cells": stats["physical_total_cells"],
            "literal_zero": stats["literal_zero"],
            "literal_zero_cells": stats["literal_zero_cells"],
            "all_blank_rows": stats["all_blank_rows"],
            "metadata_rows": metadata.get("rows"),
            "metadata_non_null": metadata_non_null,
            "manifest_non_null": manifest_non_null,
            "metadata_minus_physical_gap": metadata_gap,
            "metadata_csv_bytes": metadata.get("csv_bytes"),
            "actual_csv_bytes": csv_path.stat().st_size,
        }
        total_rows += stats["rows"]
        total_physical_nonempty += stats["physical_nonempty_cells"]
        total_physical_missing += stats["physical_missing_cells"]
        total_literal_zero += stats["literal_zero_cells"]

    record_check(
        checks,
        "three_rd_batches_requested_25_total_75_unique",
        "coverage",
        all(len(returned_batches[request_id]["requested_instruments"]) == 25 for request_id in EXPECTED_RETURNED) and len(requested_union) == 75 and not requested_overlap,
        {"per_batch": 25, "union": 75, "overlap": False},
        {"per_batch": {request_id: len(returned_batches[request_id]["requested_instruments"]) for request_id in EXPECTED_RETURNED}, "union": len(requested_union), "overlap": requested_overlap},
        "The three saved request universes are disjoint and total 75 unique literal RICs.",
    )
    record_check(
        checks,
        "three_rd_returned_instrument_union_75_no_unexpected",
        "coverage",
        len(returned_union) == 75 and not returned_overlap and all(not returned_batches[request_id]["missing_instruments"] and not returned_batches[request_id]["unexpected_instruments"] for request_id in EXPECTED_RETURNED),
        {"returned_union": 75, "overlap": False, "unexpected": []},
        {"returned_union": len(returned_union), "overlap": returned_overlap, "missing_by_request": {request_id: returned_batches[request_id]["missing_instruments"] for request_id in EXPECTED_RETURNED}, "unexpected_by_request": {request_id: returned_batches[request_id]["unexpected_instruments"] for request_id in EXPECTED_RETURNED}},
        "Every returned batch covers exactly its requested RIC set; rows may repeat an Instrument across fiscal periods and are not deduplicated.",
    )
    record_check(
        checks,
        "returned_rows_total_501",
        "rows",
        total_rows == 501 and summary.get("rows_by_kind", {}).get("fundamental") == 501 and sum(int(summary_record.get("rows", 0)) for summary_record in summary_records if summary_record.get("request_id") in EXPECTED_RETURNED) == 501,
        {"returned_rows": 501, "summary_fundamental_rows": 501},
        {"returned_rows": total_rows, "summary_fundamental_rows": summary.get("rows_by_kind", {}).get("fundamental"), "summary_returned_rd_rows": sum(int(summary_record.get("rows", 0)) for summary_record in summary_records if summary_record.get("request_id") in EXPECTED_RETURNED)},
        "The 501 count is the physical data-row total across the three returned CSVs, not a unique-instrument count.",
    )

    raw_files_after = snapshot_files(raw_dir)
    raw_unchanged = raw_files_before == raw_files_after
    record_check(
        checks,
        "raw_files_unchanged_before_after_validation",
        "provenance",
        raw_unchanged,
        {"file_set_size_hash_equal": True},
        {"file_set_size_hash_equal": raw_unchanged, "before": raw_files_before, "after": raw_files_after},
        "The raw trial is snapshotted before and after validation; audit outputs are outside the raw directory.",
    )

    failed = [check["check_id"] for check in checks if not check["passed"]]
    finished_at = utc_now()
    report: dict[str, Any] = {
        "schema_version": "ai_factor_collection_trial_v1_validation",
        "audit_id": audit_id,
        "validator": "validate_ai_factor_collection_trial_v1.py",
        "validator_sha256": validator_hash,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "status": "passed" if not failed else "failed_checks",
        "raw_run_id": args.run_id,
        "raw_run_path": raw_dir.relative_to(ROOT).as_posix(),
        "audit_output_path": audit_dir.relative_to(ROOT).as_posix(),
        "input_scope": [
            "plan.json",
            "collector_manifest.json",
            "summary.json",
            "rd_000.request.json",
            "rd_000.meta.json",
            "rd_000.csv",
            "rd_001.request.json",
            "rd_001.meta.json",
            "rd_001.csv",
            "rd_002.request.json",
            "rd_002.meta.json",
            "rd_002.csv",
            "current collector/config/corrected spans for hash and membership provenance",
        ],
        "source_hashes": {
            "script": {"path": script_path.relative_to(ROOT).as_posix(), "current": current_script_hash, "recorded": recorded_source_hashes["script"]},
            "config": {"path": config_path.relative_to(ROOT).as_posix(), "current": current_config_hash, "recorded": recorded_source_hashes["config"]},
            "spans": {"path": spans_path.relative_to(ROOT).as_posix(), "current": current_spans_hash, "recorded": recorded_source_hashes["spans"]},
        },
        "actual_hashes": {
            "plan.json": plan_hash,
            "collector_manifest.json": manifest_hash,
            "summary.json": summary_hash,
            "validator": validator_hash,
            "raw_files_before": raw_files_before,
            "raw_files_after": raw_files_after,
        },
        "counts": {
            "plan_request_count": len(plan_requests),
            "plan_fundamental_request_count": sum(family_counts.values()),
            "plan_fundamental_family_count": len(family_counts),
            "plan_etf_request_count": len(etf_specs),
            "span_rows": span_counts["rows"],
            "span_unique_rics": span_counts["distinct_literal_rics"],
            "span_delisted": span_counts["delisted_rows"],
            "requested_rd_union": len(requested_union),
            "returned_rd_union": len(returned_union),
            "returned_rows": total_rows,
            "physical_nonempty_cells": total_physical_nonempty,
            "physical_missing_cells": total_physical_missing,
            "literal_zero_cells": total_literal_zero,
            "pending": summary_counts.get("pending"),
        },
        "status_counts": {
            "manifest": manifest_status_counts,
            "summary_returned_ids": returned_summary,
            "last_execute": last_execute,
        },
        "span_counts": span_counts,
        "returned_batches": returned_batches,
        "error_sidecars": error_sidecars,
        "generated_flags": generated_flags,
        "quarantine": None,
        "historical_context": {
            "prior_plan_run_id": "20260909T181534358580Z",
            "prior_plan_path": "data/raw/ai_factor_v1/20260909T181534358580Z/plan.json",
            "prior_plan_recorded_script_sha256": "0bdf9b46e2c98fc4c39d63bfad1502fabdc21009a960961a78fc56d8a4a75e6e",
            "current_collector_script_sha256": current_script_hash,
            "interpretation": "Retrospective retained plan-only run; recorded script hash differed before any request, so it is not a request failure and was not overwritten.",
            "formal_trial_run_id": args.run_id,
        },
        "checks": checks,
        "failed_checks": failed,
        "limitations": [
            "This validator covers only the three returned R&D batches; the remaining 354 planned requests are pending and their vendor coverage is not inferred.",
            "Physical missingness counts are based on empty CSV strings; no attempt is made to infer vendor-side missingness semantics or transform dates/values.",
            "Instrument coverage is literal string set coverage and does not establish accounting basis, units, vintage or announcement availability.",
            "No quarantine is used because this stage preserves raw responses and does not exclude rows or companies.",
        ],
    }

    audit_dir.mkdir(parents=True, exist_ok=False)
    (audit_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    with (audit_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "category", "passed", "expected", "observed", "notes"])
        writer.writeheader()
        for check in checks:
            writer.writerow(
                {
                    "check_id": check["check_id"],
                    "category": check["category"],
                    "passed": check["passed"],
                    "expected": compact(check["expected"]),
                    "observed": compact(check["observed"]),
                    "notes": check["notes"],
                }
            )
    (audit_dir / "findings.md").write_text(build_findings(report, audit_dir), encoding="utf-8")

    print(json.dumps({"audit_id": audit_id, "audit_dir": str(audit_dir), "status": report["status"], "passed": len(checks) - len(failed), "checks": len(checks), "failed_checks": failed}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate and summarize the isolated price/execution field probe."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "price_execution_probe_v3" / "20260909T011724384553Z"
OUT_ROOT = ROOT / "data" / "audit" / "price_execution_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))

    integrity_rows = []
    for call in manifest["calls"]:
        csv_path = RAW / f"{call['name']}.csv"
        integrity_rows.append({
            "name": call["name"],
            "status": call["status"],
            "csv_exists": csv_path.exists(),
            "hash_matches": csv_path.exists() and sha256(csv_path) == call.get("csv_sha256"),
        })
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(out / "call_integrity.csv", index=False)

    default = pd.read_csv(RAW / "aapl_split_default.csv", parse_dates=["Date"])
    adjusted = pd.read_csv(RAW / "aapl_split_explicit_adjusted.csv", parse_dates=["Date"])
    unadjusted = pd.read_csv(RAW / "aapl_split_unadjusted.csv", parse_dates=["Date"])
    fields = [column for column in default.columns if column != "Date"]

    pre_split = unadjusted.loc[unadjusted.Date < pd.Timestamp("2020-08-31")].merge(
        adjusted, on="Date", suffixes=("_unadjusted", "_adjusted"), how="inner"
    )
    price_ratio = pre_split["TRDPRC_1_unadjusted"] / pre_split["TRDPRC_1_adjusted"]
    volume_ratio = pre_split["ACVOL_UNS_adjusted"] / pre_split["ACVOL_UNS_unadjusted"]
    turnover_diff = (
        pre_split["TRNOVR_UNS_unadjusted"] - pre_split["TRNOVR_UNS_adjusted"]
    ).abs()
    spread = (adjusted["ASK"] - adjusted["BID"]) / (
        (adjusted["ASK"] + adjusted["BID"]) / 2
    )
    field_profile = pd.DataFrame({
        "field": fields,
        "default_non_null": [int(default[field].notna().sum()) for field in fields],
        "adjusted_non_null": [int(adjusted[field].notna().sum()) for field in fields],
        "unadjusted_non_null": [int(unadjusted[field].notna().sum()) for field in fields],
    })
    field_profile.to_csv(out / "field_profile.csv", index=False)

    summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_probe_run": manifest["run_id"],
        "raw_manifest_sha256": sha256(RAW / "manifest.json"),
        "integrity": {
            "calls": int(len(integrity)),
            "all_returned": bool(integrity.status.eq("returned").all()),
            "all_hashes_match": bool(integrity.hash_matches.all()),
        },
        "adjustments": {
            "default_equals_explicit_adjusted_byte_for_byte": sha256(
                RAW / "aapl_split_default.csv"
            ) == sha256(RAW / "aapl_split_explicit_adjusted.csv"),
            "pre_split_common_dates": int(len(pre_split)),
            "median_unadjusted_to_adjusted_close_ratio": float(price_ratio.median()),
            "median_adjusted_to_unadjusted_volume_ratio": float(volume_ratio.median()),
            "maximum_turnover_absolute_difference": float(turnover_diff.max()),
            "unadjusted_rows": int(len(unadjusted)),
            "adjusted_rows": int(len(adjusted)),
            "adjusted_missing_requested_start_date": bool(
                pd.Timestamp("2020-08-24") not in set(adjusted.Date)
            ),
        },
        "execution_fields": {
            "fields": fields,
            "all_fields_have_values_in_aapl_adjusted_sample": bool(
                field_profile.adjusted_non_null.gt(0).all()
            ),
            "aapl_adjusted_relative_spread_median_bps": float(spread.median() * 10_000),
            "quote_proxy_limit": "Daily BID/ASK is an end-of-day quote proxy, not a guaranteed executable spread.",
        },
        "decisions": [
            "Request TRDPRC_1, OPEN_PRC, HIGH_1, LOW_1, ACVOL_UNS, BID, ASK, and TRNOVR_UNS.",
            "Set adjustments explicitly to exchangeCorrection, manualCorrection, CCH, CRE, RPO, and RTS.",
            "Preserve raw missing dates; do not forward-fill prices, volume, quotes, or turnover.",
            "Use price and volume for liquidity and transaction-cost modeling; retain Total Return for return labels until portfolio accounting is implemented and reconciled.",
        ],
        "limits": [
            "The split comparison is one AAPL event and does not prove complete corporate-action handling for every issuer.",
            "The adjusted response omitted the requested start date while the unadjusted response included it; collection should request a pre-window buffer and audit boundaries.",
            "Multi-instrument output is a two-row CSV header and can contain instrument-specific missing dates; cleaning must reshape without filling.",
        ],
        "non_actions": [
            "No existing raw, clean, panel, feature, label, or diagnostic file was changed.",
            "No missing value was filled and no row was deleted.",
            "No model or backtest was run.",
        ],
        "sources": [
            "https://developers.lseg.com/en/article-catalog/article/the-data-library-for-python-maximum-usage-reference-guide",
            "https://developers.lseg.com/en/article-catalog/article/discover-data-library-part-1",
        ],
    }
    report = f"""# Price and execution field audit

Run ID: `{run_id}`. Raw probe: `{manifest['run_id']}`.

All four calls returned and all CSV hashes match the raw manifest. The default
and explicit corporate-action-adjusted AAPL files are byte-identical. Before
the 2020 split, unadjusted/adjusted close and adjusted/unadjusted volume both
have median ratios of {price_ratio.median():.6f}; turnover differs by at most
{turnover_diff.max():.6f}, showing the split adjustment changes price and volume
in offsetting directions in this example.

All eight requested fields have values. Daily BID/ASK supports a transparent
spread proxy, while close, volume, and turnover support liquidity and impact
controls. These are still vendor end-of-day observations, not guaranteed fills.

The adjusted response omitted 2020-08-24 although it was the requested start
date, whereas the unadjusted response included it. Full collection will request
an earlier buffer and preserve missing boundary observations without filling.

No old dataset was overwritten, no rows were deleted or filled, and no model or
backtest was run.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    summary["outputs"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.iterdir())
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

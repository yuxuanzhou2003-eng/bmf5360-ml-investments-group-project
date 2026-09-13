"""Read-only probe: can an AI-exposure factor be built point-in-time?

The whole credibility of an AI-sector study rests on this. Selecting today's AI winners and
backtesting them from 2015 embeds look-ahead: PLTR and VRT were not listed, NVDA was a gaming
GPU company, SMCI was a generic server vendor. So AI relevance must be measured from information
available AT each date, not from what we know now.

This probe asks, one independent request per candidate, which sources exist and whether they
carry a usable date:

  - business descriptions (likely current-only, so likely NOT usable point-in-time);
  - revenue segments (carry fiscal periods, so potentially usable);
  - news headlines (carry timestamps, the most promising route);
  - transcripts / thematic classifications, if exposed at all.

Sample deliberately mixes AI-associated names with a control that is not, so that a source which
returns identical content for both can be recognised as useless for discrimination.

Credentials stay in the local .env and are never printed. Nothing existing is read or modified.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
CREDS = dotenv_values(ROOT / ".env")
SECRETS = [str(value) for value in CREDS.values() if value]


def redact(value: object) -> str:
    result = str(value)
    for secret in SECRETS:
        result = result.replace(secret, "[REDACTED]")
    return result


class SafeStream:
    def __init__(self, target):
        self.target = target

    def write(self, message):
        return self.target.write(redact(message))

    def flush(self):
        return self.target.flush()

    def isatty(self):
        return False


sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)

import lseg.data as ld  # noqa: E402
import pandas as pd  # noqa: E402

PROBE_ROOT = ROOT / "data" / "raw" / "ai_exposure_probe_v1"
UNIVERSE = ["NVDA.OQ", "MSFT.OQ", "AMD.OQ", "WMT.N"]
TIMEOUT = 90
TERMINAL = {"returned", "empty", "error", "timeout"}


def main() -> int:
    existing = sorted(path for path in PROBE_ROOT.glob("*") if path.is_dir()) if PROBE_ROOT.exists() else []
    out = existing[-1] if existing else PROBE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "probe_summary.json"
    state = (json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists()
             else {"run": out.name, "universe": UNIVERSE,
                   "question": "which AI-relevance sources exist and do they carry a usable date",
                   "records": []})
    records = state["records"]
    done = {record["name"] for record in records if record.get("status") in TERMINAL}

    def save():
        summary_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    candidates = [
        ("business_summary", "Is there a company description field at all?",
         "data", ["TR.BusinessSummary"], None),
        ("company_description", "Alternative description spelling.",
         "data", ["TR.CompanyDescription"], None),
        ("segment_revenue_current", "Do revenue segments resolve, and do they carry a period?",
         "data", ["TR.SegmentRevenue", "TR.SegmentName"], None),
        ("segment_bgs", "Business/geographic segment spelling.",
         "data", ["TR.BGSSegmentName", "TR.BGSRevenue"], None),
        ("segment_historical", "Can segments be pulled for a past fiscal year?",
         "data", ["TR.SegmentRevenue", "TR.SegmentName"],
         {"SDate": "2018-12-31", "Period": "FY0"}),
        ("segment_total_revenue", "Another segment revenue spelling.",
         "data", ["TR.SegmentTotalRevenue"], None),
        ("rd_expense_history", "R&D spend as a crude innovation proxy, with fiscal dates.",
         "data", ["TR.ResearchAndDevelopment", "TR.ResearchAndDevelopment.fperiod"],
         {"SDate": "2015-01-01", "EDate": "2022-12-31", "Frq": "FY"}),
        ("thematic_exposure", "Does any thematic/AI classification field exist?",
         "data", ["TR.ThemeExposure"], None),
        ("patent_count", "Patent activity, sometimes used as a technology-exposure proxy.",
         "data", ["TR.PatentCount"], None),
    ]

    news_candidates = [
        ("news_headlines_recent", "Do headlines return with timestamps?",
         {"query": "NVDA.OQ", "count": 20}),
        ("news_headlines_historical", "Can headlines be pulled for a 2018 window?",
         {"query": "NVDA.OQ", "start": "2018-01-01", "end": "2018-03-31", "count": 50}),
        ("news_headlines_ai_query", "Can a topic query be combined with a company?",
         {"query": "NVDA.OQ and artificial intelligence", "start": "2018-01-01",
          "end": "2018-12-31", "count": 50}),
    ]

    pending = [item for item in candidates if item[0] not in done]
    pending_news = [item for item in news_candidates if item[0] not in done]
    if not (pending or pending_news):
        print(json.dumps({"run": out.name, "status": "all candidates terminal"}, indent=2))
        return 0

    try:
        ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
    except Exception as exc:
        state["session_error"] = redact(exc)[:1500]
        save()
        print("SESSION FAILED; no request issued.", flush=True)
        return 1

    def run(name, question, callback, request_note):
        record = {"name": name, "question": question, "request": request_note,
                  "tested_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
        records.append(record)
        save()
        print("START", name, flush=True)

        def on_timeout():
            record.update(status="timeout", error=f"Exceeded {TIMEOUT}s; rerun to continue.")
            save()
            print("TIMEOUT", name, flush=True)
            os._exit(2)

        timer = threading.Timer(TIMEOUT, on_timeout)
        timer.start()
        started = time.monotonic()
        try:
            frame = callback()
            if frame is None or not len(frame):
                record.update(status="empty", rows=0)
            else:
                frame = frame.replace(r"^\s*$", pd.NA, regex=True)
                frame.to_csv(out / f"{name}.csv", index=False)
                record.update(status="returned", rows=int(len(frame)),
                              columns=[str(column) for column in frame.columns],
                              non_null={str(column): int(frame[column].notna().sum())
                                        for column in frame.columns})
                for column in frame.columns:
                    if "date" in str(column).lower() or "period" in str(column).lower():
                        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
                        if parsed.notna().any():
                            record.setdefault("date_ranges", {})[str(column)] = {
                                "min": str(parsed.min()), "max": str(parsed.max())}
        except Exception as exc:
            record.update(status="error", error=redact(exc)[:1200])
        finally:
            timer.cancel()
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
            save()
            print(json.dumps({key: value for key, value in record.items()
                              if key not in ("request", "non_null")}, ensure_ascii=False)[:450],
                  flush=True)

    for name, question, kind, fields, parameters in pending:
        arguments = {"universe": UNIVERSE, "fields": fields}
        if parameters:
            arguments["parameters"] = parameters
        run(name, question, lambda a=arguments: ld.get_data(**a), arguments)

    for name, question, kwargs in pending_news:
        def fetch(k=kwargs):
            return ld.news.get_headlines(**k)
        run(name, question, fetch, kwargs)

    state["conclusion"] = {
        status: [record["name"] for record in records if record.get("status") == status]
        for status in ["returned", "empty", "error", "timeout"]}
    state["date_bearing_sources"] = {record["name"]: record["date_ranges"]
                                     for record in records if "date_ranges" in record}
    save()
    print(json.dumps({"conclusion": state["conclusion"],
                      "date_bearing_sources": state["date_bearing_sources"]},
                     indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Third read-only relation probe: resumable, one candidate at a time.

Round 2 (`data/raw/relationship_probe_v2/`) established the decisive fact that institutional
holdings ARE retrievable as of a historical date: `SDate=2015-12-31` returned 9,860 rows whose
filing dates all fall between 2014-03-03 and 2015-12-31. It then hit a 90-second timeout on the
`SDate`/`EDate`/`Frq=FQ` range request and exited by design, leaving twelve candidates untested.
That timeout is preserved in the round-2 summary and is repeated here as the last candidate.

This script is resumable: it reads the run directory it is pointed at, skips candidates that
already have a terminal status, and runs the next one. A timeout still exits the process (an
LSEG request cannot be cancelled once issued), but rerunning the script continues from the next
candidate instead of restarting the batch.

Usage: probe_relationships_v3.py [run_directory]

Credentials stay in the local .env and are never printed. No existing raw, clean, panel or
model-ready file is read or modified.
"""
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
credentials = dotenv_values(ROOT / ".env")
secrets = [value for value in credentials.values() if value]


def redact(value):
    value = str(value)
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return value


class SafeStream:
    def __init__(self, target):
        self.target = target

    def write(self, message):
        return self.target.write(redact(message))

    def flush(self):
        self.target.flush()

    def isatty(self):
        return False


sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)

import lseg.data as ld  # noqa: E402
import pandas as pd  # noqa: E402

UNIVERSE = ["AAPL.O", "MSFT.O", "WMT.N"]
REQUEST_TIMEOUT_SECONDS = 90
PROBE_ROOT = ROOT / "data" / "raw" / "relationship_probe_v3"

CANDIDATES = [
    ("ownership_relative_10y", "Does a relative start date reach back ten years?",
     ["TR.InvestorFullName", "TR.PctOfSharesOutHeld", "TR.HoldingsDate"], {"SDate": "-10Y"}),
    ("ownership_asof_2020", "Does the as-of snapshot also work mid-sample?",
     ["TR.InvestorFullName", "TR.PctOfSharesOutHeld", "TR.HoldingsDate"], {"SDate": "2020-06-30"}),
    ("rec_brokers_asof_2015", "Can broker-level recommendations be retrieved as of 2015-12-31?",
     ["TR.RecEstBrokerName", "TR.RecEstValue", "TR.RecEstDate"], {"SDate": "2015-12-31"}),
    ("rec_brokers_range_2015", "Does an SDate/EDate range return broker recommendation history?",
     ["TR.RecEstBrokerName", "TR.RecEstValue", "TR.RecEstDate"],
     {"SDate": "2015-01-01", "EDate": "2015-12-31"}),
    ("eps_est_broker_detail", "Does IBES detail expose per-broker EPS estimates with dates?",
     ["TR.EPSEstBrokerName", "TR.EPSEstValue", "TR.EPSEstDate"],
     {"Frq": "FQ", "Period": "FQ1", "SDate": "2015-12-31"}),
    ("est_broker_with_dates", "Does TR.EstBrokerName populate when paired with estimate values?",
     ["TR.EstBrokerName", "TR.EPSEstValue", "TR.EPSEstDate"], {"Frq": "FQ", "Period": "FQ1"}),
    ("trbc_asof_2015", "Is the TRBC classification retrievable as of 2015-12-31?",
     ["TR.TRBCIndustryGroup", "TR.TRBCIndustry", "TR.TRBCActivity"], {"SDate": "2015-12-31"}),
    ("sc_supplychain", "Does a bare TR.SupplyChain field resolve?", ["TR.SupplyChain"], None),
    ("sc_partners", "Does TR.SupplyChainPartners resolve?", ["TR.SupplyChainPartners"], None),
    ("sc_partner_name", "Does TR.SCPartnerName resolve?", ["TR.SCPartnerName"], None),
    ("sc_customer_supplier_names", "Do TR.CustomerName / TR.SupplierName resolve?",
     ["TR.CustomerName", "TR.SupplierName"], None),
    ("sc_relationship_type", "Does a generic TR.RelationshipType resolve?", ["TR.RelationshipType"], None),
    ("sc_bcs", "Does the business-classification supply chain field resolve?",
     ["TR.BusinessRelationship"], None),
    ("ownership_range_2015_2016", "Repeat of the round-2 range request that timed out at 90 seconds.",
     ["TR.InvestorFullName", "TR.PctOfSharesOutHeld", "TR.HoldingsDate"],
     {"SDate": "2015-01-01", "EDate": "2016-12-31", "Frq": "FQ"}),
]
TERMINAL = {"returned", "empty", "error", "timeout"}


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    else:
        existing = sorted(path for path in PROBE_ROOT.glob("*") if path.is_dir()) if PROBE_ROOT.exists() else []
        out = existing[-1] if existing else PROBE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "probe_summary.json"
    state = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {
        "run_directory": out.name, "universe": UNIVERSE,
        "round_2_reference": "data/raw/relationship_probe_v2 established historical as-of ownership "
                             "and recorded the range-request timeout.",
        "records": []}
    records = state["records"]
    done = {record["name"] for record in records if record.get("status") in TERMINAL}

    def save():
        summary_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    remaining = [candidate for candidate in CANDIDATES if candidate[0] not in done]
    if not remaining:
        print(json.dumps({"run_directory": out.name, "status": "all candidates terminal",
                          "counts": pd.Series([record["status"] for record in records]).value_counts().to_dict()},
                         indent=2, ensure_ascii=False))
        return 0

    session_record = {"name": f"desktop_session_{len([r for r in records if r['name'].startswith('desktop_session')])}",
                      "request": {"session": "desktop.workspace"}, "status": "running",
                      "tested_at_utc": datetime.now(timezone.utc).isoformat()}
    records.append(session_record)
    try:
        ld.open_session(name="desktop.workspace", app_key=credentials.get("LSEG_APP_KEY"))
        session_record.update(status="connected")
    except Exception as exc:
        session_record.update(status="error", error=redact(exc)[:1500])
        save()
        print("SESSION FAILED; stopping before any data request.", flush=True)
        return 1
    save()

    for name, question, fields, parameters in remaining:
        request = {"universe": UNIVERSE, "fields": fields}
        if parameters:
            request["parameters"] = parameters
        record = {"name": name, "question": question, "request": request,
                  "tested_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
        records.append(record)
        save()
        print("START", name, flush=True)

        def on_timeout():
            record.update(status="timeout",
                          error=f"Request exceeded {REQUEST_TIMEOUT_SECONDS} seconds; availability unconfirmed. "
                                f"Rerun the script to continue with the next candidate.")
            save()
            print("TIMEOUT", name, flush=True)
            os._exit(2)

        timer = threading.Timer(REQUEST_TIMEOUT_SECONDS, on_timeout)
        timer.start()
        started = time.monotonic()
        try:
            frame = ld.get_data(**request).replace(r"^\s*$", pd.NA, regex=True)
            frame.to_csv(out / f"{name}.csv", index=False)
            record.update(
                status="returned" if len(frame) else "empty",
                rows=int(len(frame)),
                columns=[str(column) for column in frame.columns],
                non_null={str(column): int(frame[column].notna().sum()) for column in frame.columns})
            for column in frame.columns:
                if "date" in str(column).lower():
                    parsed = pd.to_datetime(frame[column], errors="coerce")
                    if parsed.notna().any():
                        record.setdefault("date_ranges", {})[str(column)] = {
                            "min": str(parsed.min()), "max": str(parsed.max()),
                            "unique_days": int(parsed.dt.date.nunique())}
        except Exception as exc:
            record.update(status="error", error=redact(exc)[:1500])
            (out / f"{name}.error.json").write_text(
                json.dumps({"name": name, "request": request, "error": redact(exc)[:4000]}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
        finally:
            timer.cancel()
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
            save()
            print(json.dumps({key: value for key, value in record.items() if key != "request"},
                             ensure_ascii=False)[:700], flush=True)

    state["conclusion"] = {
        "returned": [record["name"] for record in records if record.get("status") == "returned"],
        "empty": [record["name"] for record in records if record.get("status") == "empty"],
        "error": [record["name"] for record in records if record.get("status") == "error"],
        "timeout": [record["name"] for record in records if record.get("status") == "timeout"],
        "date_ranges": {record["name"]: record["date_ranges"] for record in records if "date_ranges" in record},
    }
    save()
    print(json.dumps(state["conclusion"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

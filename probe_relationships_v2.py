"""Second read-only relation probe: historical depth and remaining supply-chain field spellings.

Round 1 (`data/raw/relationship_probe_v1/20260909T081614330938Z/`) established that TRBC,
analyst counts, broker-level recommendations and institutional ownership resolve, while every
tried supply-chain and peer spelling was rejected as an unresolvable field. Round 1 returned
only current snapshots: ownership filing dates started at 2024-12-31, which cannot support a
2015-2022 point-in-time graph. This round asks one question per request: can each candidate
relation be retrieved as of a historical date?

Each candidate is requested independently. Credentials stay in the local .env and are never
printed. No existing raw, clean, panel or model-ready file is read or modified.
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

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "relationship_probe_v2" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
UNIVERSE = ["AAPL.O", "MSFT.O", "WMT.N"]
REQUEST_TIMEOUT_SECONDS = 90
records = []


def save():
    (OUT / "probe_summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "universe": UNIVERSE, "records": records}, indent=2,
                   ensure_ascii=False), encoding="utf-8")


def probe(name, question, fields, parameters=None):
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
                      error=f"Request exceeded {REQUEST_TIMEOUT_SECONDS} seconds; availability unconfirmed.")
        save()
        print("TIMEOUT", name, flush=True)
        os._exit(2)

    timer = threading.Timer(REQUEST_TIMEOUT_SECONDS, on_timeout)
    timer.start()
    started = time.monotonic()
    try:
        frame = ld.get_data(**request)
        frame = frame.replace(r"^\s*$", pd.NA, regex=True)
        frame.to_csv(OUT / f"{name}.csv", index=False)
        record.update(
            status="returned" if len(frame) else "empty",
            rows=int(len(frame)),
            columns=[str(column) for column in frame.columns],
            non_null={str(column): int(frame[column].notna().sum()) for column in frame.columns},
        )
        for column in frame.columns:
            if "date" in str(column).lower():
                parsed = pd.to_datetime(frame[column], errors="coerce")
                if parsed.notna().any():
                    record.setdefault("date_ranges", {})[str(column)] = {
                        "min": str(parsed.min()), "max": str(parsed.max()),
                        "unique_days": int(parsed.dt.date.nunique())}
    except Exception as exc:
        record.update(status="error", error=redact(exc)[:1500])
        (OUT / f"{name}.error.json").write_text(
            json.dumps({"name": name, "request": request, "error": redact(exc)[:4000]}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
    finally:
        timer.cancel()
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        save()
        print(json.dumps({key: value for key, value in record.items() if key != "request"},
                         ensure_ascii=False)[:700], flush=True)
    return record


session = {"session": "desktop.workspace"}
record = {"name": "desktop_session", "request": session, "status": "running",
          "tested_at_utc": datetime.now(timezone.utc).isoformat()}
records.append(record)
try:
    ld.open_session(name="desktop.workspace", app_key=credentials.get("LSEG_APP_KEY"))
    record.update(status="connected")
except Exception as exc:
    record.update(status="error", error=redact(exc)[:1500])
    save()
    print("SESSION FAILED; stopping before any data request.", flush=True)
    sys.exit(1)
save()

CANDIDATES = [
    # Ownership as of a historical date: the decisive question for a point-in-time common-owner graph.
    ("ownership_asof_2015", "Can institutional holdings be retrieved as of 2015-12-31?",
     ["TR.InvestorFullName", "TR.PctOfSharesOutHeld", "TR.HoldingsDate"], {"SDate": "2015-12-31"}),
    ("ownership_range_2015_2016", "Does an SDate/EDate range return quarterly ownership history?",
     ["TR.InvestorFullName", "TR.PctOfSharesOutHeld", "TR.HoldingsDate"],
     {"SDate": "2015-01-01", "EDate": "2016-12-31", "Frq": "FQ"}),
    ("ownership_relative_10y", "Does a relative start date reach back ten years?",
     ["TR.InvestorFullName", "TR.PctOfSharesOutHeld", "TR.HoldingsDate"], {"SDate": "-10Y"}),
    # Broker-level analyst coverage as of a historical date.
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
    # TRBC classification as of a historical date.
    ("trbc_asof_2015", "Is the TRBC classification retrievable as of 2015-12-31?",
     ["TR.TRBCIndustryGroup", "TR.TRBCIndustry", "TR.TRBCActivity"], {"SDate": "2015-12-31"}),
    # Remaining supply-chain spellings, one family per request.
    ("sc_supplychain", "Does a bare TR.SupplyChain field resolve?", ["TR.SupplyChain"], None),
    ("sc_partners", "Does TR.SupplyChainPartners resolve?", ["TR.SupplyChainPartners"], None),
    ("sc_partner_name", "Does TR.SCPartnerName resolve?", ["TR.SCPartnerName"], None),
    ("sc_customer_supplier_names", "Do TR.CustomerName / TR.SupplierName resolve?",
     ["TR.CustomerName", "TR.SupplierName"], None),
    ("sc_relationship_type", "Does a generic TR.RelationshipType resolve?", ["TR.RelationshipType"], None),
    ("sc_bcs", "Does the business-classification supply chain field resolve?",
     ["TR.BusinessRelationship"], None),
]

for name, question, fields, parameters in CANDIDATES:
    probe(name, question, fields, parameters)

summary = {
    "run_id": RUN_ID,
    "status_counts": pd.Series([record["status"] for record in records]).value_counts().to_dict(),
    "returned": [record["name"] for record in records if record.get("status") == "returned"],
    "empty": [record["name"] for record in records if record.get("status") == "empty"],
    "error": [record["name"] for record in records if record.get("status") == "error"],
    "date_ranges": {record["name"]: record["date_ranges"] for record in records if "date_ranges" in record},
}
(OUT / "run_conclusion.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

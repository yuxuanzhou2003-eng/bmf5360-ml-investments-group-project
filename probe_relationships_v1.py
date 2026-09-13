"""Read-only probe for economic-relation data (supply chain, analyst coverage, ownership, peers, TRBC).

Each candidate is requested independently so that one rejected field cannot hide the
availability of the others. Credentials stay in the local .env and are never printed.
No existing raw, clean, panel or model-ready file is read or modified.
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
OUT = ROOT / "data" / "raw" / "relationship_probe_v1" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
UNIVERSE = ["AAPL.O", "MSFT.O", "WMT.N"]
REQUEST_TIMEOUT_SECONDS = 60
records = []


def save():
    (OUT / "probe_summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "universe": UNIVERSE, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")


def probe(name, request, callback):
    record = {"name": name, "request": request,
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
        frame = callback()
        if frame is None:
            record.update(status="connected")
        else:
            frame = frame.replace(r"^\s*$", pd.NA, regex=True)
            frame.to_csv(OUT / f"{name}.csv", index=False)
            record.update(
                status="returned" if len(frame) else "empty",
                rows=int(len(frame)),
                columns=[str(column) for column in frame.columns],
                non_null={str(column): int(frame[column].notna().sum()) for column in frame.columns},
            )
    except Exception as exc:
        record.update(status="error", error=redact(exc)[:1500])
        (OUT / f"{name}.error.json").write_text(
            json.dumps({"name": name, "request": request, "error": redact(exc)[:4000]}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
    finally:
        timer.cancel()
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        save()
        print(json.dumps({key: record[key] for key in record if key != "request"},
                         ensure_ascii=False)[:600], flush=True)
    return record


def get_data(fields, parameters=None, universe=None):
    arguments = {"universe": universe or UNIVERSE, "fields": fields}
    if parameters:
        arguments["parameters"] = parameters
    return arguments, (lambda: ld.get_data(**arguments))


probe("desktop_session", {"session": "desktop.workspace"},
      lambda: ld.open_session(name="desktop.workspace", app_key=credentials.get("LSEG_APP_KEY")) and None)
if records[-1]["status"] == "error":
    print("SESSION FAILED; stopping before any data request.", flush=True)
    sys.exit(1)

CANDIDATES = [
    # TRBC classification: the cheapest economically motivated edge, expected to be available.
    ("trbc_classification", ["TR.TRBCEconomicSector", "TR.TRBCBusinessSector", "TR.TRBCIndustryGroup",
                             "TR.TRBCIndustry", "TR.TRBCActivity"], None),
    # Supply chain relationships: candidate field spellings are probed one family at a time.
    ("supplychain_relationships", ["TR.SupplyChainRelationship"], None),
    ("supplychain_partner_names", ["TR.SCRelationshipPartnerName"], None),
    ("supplychain_customers", ["TR.SupplyChainCustomers"], None),
    ("supplychain_suppliers", ["TR.SupplyChainSuppliers"], None),
    ("supplychain_sc_type", ["TR.SCRelationshipType"], None),
    ("supplychain_revenue_share", ["TR.SCRelationshipRevenueShare"], None),
    # Analyst coverage: firm-level count first, then broker-level identity.
    ("analyst_count", ["TR.NumberOfAnalysts"], None),
    ("estimate_broker_names", ["TR.EstBrokerName"], {"Frq": "FQ", "Period": "FQ1"}),
    ("recommendation_broker_names", ["TR.RecEstBrokerName"], None),
    ("broker_id", ["TR.BrokerID", "TR.BrokerName"], None),
    # Institutional ownership: common-owner edges.
    ("investor_names", ["TR.InvestorFullName", "TR.InvestorType"], None),
    ("investor_holdings", ["TR.InvestorFullName", "TR.PctOfSharesOutHeld"], None),
    ("shareholders", ["TR.SharesHeld", "TR.HoldingsDate"], None),
    # Peer sets.
    ("peers", ["TR.Peers"], None),
    ("peer_rics", ["TR.PeerRIC"], None),
    ("company_peers", ["TR.CompanyPeers"], None),
]

for name, fields, parameters in CANDIDATES:
    request, callback = get_data(fields, parameters)
    probe(name, request, callback)

# Search-based discovery: does the platform expose a relationship view at all?
try:
    from lseg.data import discovery

    views = [view for view in dir(discovery.Views) if not view.startswith("_")]
    (OUT / "search_views.json").write_text(json.dumps(views, indent=2), encoding="utf-8")
    records.append({"name": "search_views_enumerated", "status": "returned", "rows": len(views),
                    "tested_at_utc": datetime.now(timezone.utc).isoformat()})
    save()
    print("search views:", len(views), flush=True)
except Exception as exc:
    records.append({"name": "search_views_enumerated", "status": "error", "error": redact(exc)[:1500],
                    "tested_at_utc": datetime.now(timezone.utc).isoformat()})
    save()

print(json.dumps({"run_id": RUN_ID,
                  "status_counts": pd.Series([record["status"] for record in records]).value_counts().to_dict(),
                  "returned": [record["name"] for record in records if record.get("status") == "returned"],
                  "empty": [record["name"] for record in records if record.get("status") == "empty"],
                  "error": [record["name"] for record in records if record.get("status") == "error"]},
                 indent=2, ensure_ascii=False), flush=True)

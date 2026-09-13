"""Read-only check: is the masked broker token a stable broker identifier?

The common-analyst edge depends entirely on being able to tell whether the SAME broker covers
two different companies. Round-1/2/3 probes showed that 33.1% of broker rows come back as
`Permission Denied <number>` instead of a name. This script tests whether that number is a
stable broker-level identifier (usable for matching) or a row-level token (which would make a
third of the edges unmatchable and invalidate the approach).

Decision rule, declared before running:
  - If masked tokens recur across companies with a distribution comparable to named brokers,
    the token is a broker identifier and the common-coverage edge is constructible.
  - If masked tokens are overwhelmingly unique to a single company, the token carries no
    cross-company identity and the approach fails on a third of its edges.
Cross-date recurrence is checked the same way.

Sample: 30 instruments drawn deterministically from the corrected universe spans, restricted to
those that are index members across the whole 2015-2021 development window so that all three
as-of dates are populated. Credentials stay in the local .env and are never printed. No existing
raw, clean, panel or model-ready file is modified.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "broker_identity_probe_v1" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
SPANS = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
AS_OF_DATES = ["2015-12-31", "2018-06-30", "2021-06-30"]
SAMPLE_SIZE = 30
REQUEST_TIMEOUT_SECONDS = 120
MASK_PREFIX = "Permission Denied"

spans = pd.read_csv(SPANS)
spans["member_from"] = pd.to_datetime(spans.member_from)
spans["member_to"] = pd.to_datetime(spans.member_to)
eligible = spans.loc[spans.member_from.le("2015-01-01") & spans.member_to.ge("2021-12-31")].copy()
eligible = eligible.sort_values("ric").reset_index(drop=True)
step = max(1, len(eligible) // SAMPLE_SIZE)
universe = eligible.ric.iloc[::step].head(SAMPLE_SIZE).tolist()

records = []
state = {"run_id": RUN_ID, "sample_size": len(universe), "universe": universe,
         "eligible_pool": int(len(eligible)), "as_of_dates": AS_OF_DATES,
         "selection_rule": "index member for the whole 2015-01-01..2021-12-31 window, "
                           "sorted by RIC, evenly spaced deterministic sample",
         "decision_rule": "masked tokens must recur across companies like named brokers do",
         "records": records}


def save():
    (OUT / "probe_summary.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


save()
try:
    ld.open_session(name="desktop.workspace", app_key=credentials.get("LSEG_APP_KEY"))
except Exception as exc:
    state["session_error"] = redact(exc)[:1500]
    save()
    print("SESSION FAILED; no data request issued.", flush=True)
    sys.exit(1)

frames = {}
for as_of in AS_OF_DATES:
    name = f"rec_brokers_asof_{as_of.replace('-', '')}"
    request = {"universe": universe,
               "fields": ["TR.RecEstBrokerName", "TR.RecEstValue", "TR.RecEstDate"],
               "parameters": {"SDate": as_of}}
    record = {"name": name, "as_of": as_of, "request": request, "status": "running",
              "tested_at_utc": datetime.now(timezone.utc).isoformat()}
    records.append(record)
    save()
    print("START", name, flush=True)

    def on_timeout():
        record.update(status="timeout", error=f"Exceeded {REQUEST_TIMEOUT_SECONDS}s; rerun to continue.")
        save()
        print("TIMEOUT", name, flush=True)
        os._exit(2)

    timer = threading.Timer(REQUEST_TIMEOUT_SECONDS, on_timeout)
    timer.start()
    started = time.monotonic()
    try:
        frame = ld.get_data(**request).replace(r"^\s*$", pd.NA, regex=True)
        frame.to_csv(OUT / f"{name}.csv", index=False)
        frames[as_of] = frame
        record.update(status="returned", rows=int(len(frame)),
                      instruments_returned=int(frame.Instrument.nunique()))
    except Exception as exc:
        record.update(status="error", error=redact(exc)[:1500])
    finally:
        timer.cancel()
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        save()
        print(json.dumps({key: value for key, value in record.items() if key != "request"},
                         ensure_ascii=False)[:400], flush=True)

if not frames:
    state["conclusion"] = {"verdict": "no data returned"}
    save()
    sys.exit(1)


def recurrence(frame):
    """How many distinct companies does each broker token cover?"""
    usable = frame.loc[frame["Broker Name"].notna(), ["Instrument", "Broker Name"]].drop_duplicates()
    masked = usable["Broker Name"].str.startswith(MASK_PREFIX, na=False)
    counts = usable.groupby("Broker Name").Instrument.nunique()
    masked_tokens = usable.loc[masked, "Broker Name"].unique()
    named_tokens = usable.loc[~masked, "Broker Name"].unique()
    return {
        "rows": int(len(frame)),
        "distinct_instruments": int(usable.Instrument.nunique()),
        "masked_row_share": float(masked.mean()) if len(usable) else float("nan"),
        "masked_tokens": int(len(masked_tokens)),
        "named_tokens": int(len(named_tokens)),
        "masked_companies_per_token_mean": float(counts.reindex(masked_tokens).mean()),
        "masked_companies_per_token_median": float(counts.reindex(masked_tokens).median()),
        "masked_tokens_covering_one_company_only": int((counts.reindex(masked_tokens) == 1).sum()),
        "masked_tokens_covering_multiple": int((counts.reindex(masked_tokens) > 1).sum()),
        "named_companies_per_token_mean": float(counts.reindex(named_tokens).mean()),
        "named_companies_per_token_median": float(counts.reindex(named_tokens).median()),
        "named_tokens_covering_one_company_only": int((counts.reindex(named_tokens) == 1).sum()),
        "named_tokens_covering_multiple": int((counts.reindex(named_tokens) > 1).sum()),
    }


per_date = {as_of: recurrence(frame) for as_of, frame in frames.items()}
token_sets = {}
for as_of, frame in frames.items():
    names = frame.loc[frame["Broker Name"].notna(), "Broker Name"]
    token_sets[as_of] = {
        "masked": set(names[names.str.startswith(MASK_PREFIX, na=False)].unique()),
        "named": set(names[~names.str.startswith(MASK_PREFIX, na=False)].unique()),
    }
dates = list(frames)
cross_date = {}
for i in range(len(dates) - 1):
    a, b = dates[i], dates[i + 1]
    for kind in ["masked", "named"]:
        first, second = token_sets[a][kind], token_sets[b][kind]
        overlap = first & second
        cross_date[f"{kind}_{a}_vs_{b}"] = {
            "tokens_first": len(first), "tokens_second": len(second), "overlap": len(overlap),
            "jaccard": round(len(overlap) / len(first | second), 4) if (first | second) else None,
        }

masked_multi = float(np.mean([value["masked_tokens_covering_multiple"] /
                              max(1, value["masked_tokens"]) for value in per_date.values()]))
named_multi = float(np.mean([value["named_tokens_covering_multiple"] /
                             max(1, value["named_tokens"]) for value in per_date.values()]))
verdict = ("masked token behaves like a stable broker identifier"
           if masked_multi >= 0.5 * named_multi and masked_multi > 0.2 else
           "masked token does NOT recur across companies like a broker identifier")
state["conclusion"] = {
    "per_date": per_date,
    "cross_date_token_overlap": cross_date,
    "masked_share_of_tokens_covering_multiple_companies": round(masked_multi, 4),
    "named_share_of_tokens_covering_multiple_companies": round(named_multi, 4),
    "verdict": verdict,
}
save()
print(json.dumps(state["conclusion"], indent=2, ensure_ascii=False), flush=True)

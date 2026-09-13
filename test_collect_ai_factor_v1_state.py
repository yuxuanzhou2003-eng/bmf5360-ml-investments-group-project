"""Local-only artifact state-machine checks for collect_ai_factor_v1.py.

The test uses temporary files and a small in-memory DataFrame.  It never imports,
opens, or calls the LSEG client and never touches an existing raw run.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import collect_ai_factor_v1 as collector


REQUEST = {
    "universe": ["TEST.OQ"],
    "fields": ["TR.TestValue"],
    "parameters": {"SDate": "2013-01-01", "EDate": "2026-06-30", "Frq": "FY"},
}
REQUEST_HASH = collector.request_sha256(REQUEST)
SPEC = {
    "request_id": "state_test",
    "kind": "fundamental",
    "family_id": "state_test_family",
    "group": "state_test",
    "method": "get_data",
    "timeout_seconds": 90,
    "request": REQUEST,
    "request_sha256": REQUEST_HASH,
    "request_file_sha256": REQUEST_HASH,
}
PLAN = {
    "schema_version": collector.SCHEMA_VERSION,
    "run_id": "local_state_test",
    "raw_response_policy": {},
}


def success_artifacts(root: Path) -> None:
    paths = collector.artifact_paths(root, SPEC)
    physical_request_hash = collector.atomic_write_request(paths["request"], REQUEST)
    if physical_request_hash != REQUEST_HASH:
        raise AssertionError("canonical request file hash unexpectedly differs")
    frame = pd.DataFrame({"Instrument": ["TEST.OQ"], "TR.TestValue": [1.0]})
    csv_hash = collector.atomic_write_frame(paths["csv"], frame, include_index=False)
    metadata = {
        "schema_version": collector.SCHEMA_VERSION,
        "run_id": PLAN["run_id"],
        "request_id": SPEC["request_id"],
        "request": REQUEST,
        "request_sha256": REQUEST_HASH,
        "request_file_sha256": physical_request_hash,
        "status": "returned",
        "no_frame_returned": False,
        "csv_path": paths["csv"].name,
        "csv_sha256": csv_hash,
        "rows": 1,
        "columns": ["Instrument", "TR.TestValue"],
        "dataframe_notna_counts": {"Instrument": 1, "TR.TestValue": 1},
        "serialized_nonempty_counts": {"Instrument": 1, "TR.TestValue": 1},
    }
    collector.atomic_write_json(paths["metadata"], metadata)


class ArtifactStateTests(unittest.TestCase):
    def test_pending_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ai_factor_state_pending_") as directory:
            state, detail = collector.existing_artifact_state(Path(directory), SPEC)
            self.assertEqual(state, "pending")
            self.assertIsNone(detail)

    def test_success_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ai_factor_state_success_") as directory:
            root = Path(directory)
            success_artifacts(root)
            state, detail = collector.existing_artifact_state(root, SPEC)
            self.assertEqual(state, "verified")
            self.assertEqual(detail["status"], "returned")

    def test_serialized_empty_differs_from_dataframe_notna(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ai_factor_state_counts_") as directory:
            root = Path(directory)
            path = root / "counts.csv"
            frame = pd.DataFrame({"blank": [""], "text_zero": ["0"], "missing": [pd.NA]})
            analysis = collector.frame_analysis(frame)
            collector.atomic_write_frame(path, frame, include_index=False)
            serialized = collector.serialized_nonempty_counts(
                path, analysis["columns"], analysis["rows"], include_index=False
            )
            self.assertEqual(analysis["dataframe_notna_counts"]["blank"], 1)
            self.assertEqual(serialized["blank"], 0)
            self.assertEqual(serialized["text_zero"], 1)
            self.assertEqual(serialized["missing"], 0)

    def test_error_and_timeout_are_retryable(self) -> None:
        for failure_status in ("error", "timeout"):
            with self.subTest(failure_status=failure_status):
                with tempfile.TemporaryDirectory(prefix=f"ai_factor_state_{failure_status}_") as directory:
                    root = Path(directory)
                    paths = collector.artifact_paths(root, SPEC)
                    request_file_hash = collector.atomic_write_request(paths["request"], REQUEST)
                    self.assertEqual(request_file_hash, REQUEST_HASH)
                    sidecar = collector.write_error_sidecar(
                        root,
                        PLAN,
                        SPEC,
                        failure_status,
                        f"synthetic {failure_status}",
                        0.1,
                        attempt=1,
                    )
                    state, detail = collector.existing_artifact_state(root, SPEC)
                    self.assertEqual(state, "failed")
                    self.assertEqual(detail["status"], failure_status)
                    self.assertEqual(sidecar["request_file_sha256"], request_file_hash)
                    self.assertTrue((root / "state_test.attempt_001.error.json").exists())

    def test_tampered_request_blocks_even_with_error_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ai_factor_state_tampered_request_") as directory:
            root = Path(directory)
            paths = collector.artifact_paths(root, SPEC)
            collector.atomic_write_request(paths["request"], REQUEST)
            collector.write_error_sidecar(root, PLAN, SPEC, "error", "synthetic", 0.1, attempt=1)
            paths["request"].write_bytes((json.dumps(REQUEST, indent=2) + "\n").encode("utf-8"))
            state, _detail = collector.existing_artifact_state(root, SPEC)
            self.assertEqual(state, "mismatch")

    def test_tampered_success_csv_blocks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ai_factor_state_tampered_csv_") as directory:
            root = Path(directory)
            success_artifacts(root)
            paths = collector.artifact_paths(root, SPEC)
            with paths["csv"].open("ab") as handle:
                handle.write(b"tampered\n")
            state, _detail = collector.existing_artifact_state(root, SPEC)
            self.assertEqual(state, "mismatch")

    def test_tampered_success_metadata_blocks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ai_factor_state_tampered_meta_") as directory:
            root = Path(directory)
            success_artifacts(root)
            paths = collector.artifact_paths(root, SPEC)
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            metadata["request_file_sha256"] = "0" * 64
            paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
            state, _detail = collector.existing_artifact_state(root, SPEC)
            self.assertEqual(state, "mismatch")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main(verbosity=2)

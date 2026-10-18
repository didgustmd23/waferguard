"""합성 객체로 범용 변환·구조 선택·검토 중단을 검사한다."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.pickle_converters import analyze_structure
from src.agents.pickle_import_cli import open_run, status
from src.agents.pickle_results import inspect_canonical
from src.preprocessing.canonical_data import load_table, validate_data
from tests.agents import test_wm_pickle_import as wm_tests


class PickleConversionTest(unittest.TestCase):
    def setUp(self):
        """실행기·worker는 재사용하고 OS 전송·pickle 로더만 모의한다."""
        self.harness = wm_tests.WmPickleImportTest()
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)

    def run_payload(self, payload):
        harness = self.harness
        harness.frame = payload
        approval = harness.approval()

        with patch("src.agents.pickle_import.run_import_worker", side_effect=harness.transport):
            result = harness.manager.run_next(approval_id=approval.approval_id)

        return result, status(harness.tool)

    def test_table_types_and_missing(self):
        """문자열 object·날짜·Nullable·인덱스를 복원한다."""
        frame = pd.DataFrame({
            "number": pd.Series([1, pd.NA, 3], dtype="Int64"),
            "text": pd.Series(["a", None, pd.NA], dtype=object),
            "flag": pd.Series([True, pd.NA, False], dtype="boolean"),
            "float": [float("nan"), float("inf"), 2.0],
            "time": pd.to_datetime(["2026-01-01", None, "2026-01-03"]),
        })
        frame.index = pd.Index(["a", "b", "c"], name="source")
        result, report = self.run_payload(frame)
        self.assertEqual(result.status, "completed")
        canonical = Path(report["local_canonical_directory"])
        pd.testing.assert_frame_equal(load_table(canonical), frame)

    def test_numeric_array_shape_dtype(self):
        """복소 숫자 배열도 object pickle 없이 NPY로 보존한다."""
        array = np.array([[1 + 2j, complex(float("nan"), 0)]], dtype=np.complex64)
        result, report = self.run_payload(array)
        self.assertEqual(result.status, "completed")
        canonical = Path(report["local_canonical_directory"])

        with patch("pickle.load", side_effect=AssertionError("부모 pickle 금지")):
            restored = np.load(canonical / "array.npy", allow_pickle=False)

        np.testing.assert_array_equal(restored, array)
        self.assertEqual(restored.dtype, array.dtype)

    def test_unique_nested_object_records_path(self):
        """단일 객체를 선택하고 나머지 원본은 snapshot에 보존한다."""
        result, report = self.run_payload({"description": "synthetic", "data": [np.arange(3)]})
        self.assertEqual(result.status, "completed")
        source = json.loads((Path(report["local_canonical_directory"]) / "source.json").read_text())
        self.assertEqual(source["selected_path"], ["data", 0])
        self.assertEqual(source["unselected_original"], "input_snapshot")

    def test_ambiguity_stops_without_completion(self):
        """여러 후보를 임의 선택하거나 완료로 등록하지 않는다."""
        result, report = self.run_payload({"first": np.arange(3), "second": np.arange(4)})
        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertEqual(result.status, "blocked_input")
        self.assertEqual(report["structure_status"], "review_required")
        structure = Path(report["local_structure_report"]).with_suffix(".json")
        self.assertEqual(json.loads(structure.read_text())["reason"], "MULTIPLE_CANDIDATES")
        self.assertNotIn("ingest", self.harness.tool.states.load().artifact_refs)

    def test_explicit_selection_and_converter(self):
        """dict 키·목록 인덱스로만 선택하며 변환기 불일치는 검토한다."""
        objects = {"first": np.arange(3), "second": np.arange(4)}
        selected, report = analyze_structure(objects, "numeric_array_v1", ("second",))
        self.assertIs(selected, objects["second"])
        self.assertEqual(report["selected_path"], ["second"])
        self.assertEqual(analyze_structure(objects, select_path=("missing",))[1]["reason"], "SELECT_PATH_INVALID")
        self.assertEqual(analyze_structure(objects["first"], "tabular_v1")[1]["reason"], "CONVERTER_MISMATCH")

    def test_selected_path_in_new_approved_run(self):
        """명시적 선택을 새 실행·승인에 결합해 다중 후보에서 한 객체만 변환한다."""
        from src.agents.pickle_import_cli import create, executor

        harness = self.harness
        original = harness.directory
        directory = original.parent / "selected"

        with patch("src.agents.pickle_import_cli._directory", return_value=directory):
            create("selected", harness.source, harness.tool.limits, 65536, 1024 * 1024,
                   confirmed=True, converter_id="numeric_array_v1", select_path=("second",))
            harness.directory = directory
            harness.tool = open_run("selected")
            harness.manager = executor(harness.tool)
            result, report = self.run_payload({"first": np.arange(3), "second": np.arange(4)})

        self.assertEqual(result.status, "completed")
        canonical = Path(report["local_canonical_directory"])
        np.testing.assert_array_equal(np.load(canonical / "array.npy", allow_pickle=False), np.arange(4))
        self.assertTrue((original / "selection.json").exists())

    def test_review_reply_mismatch_rejected(self):
        """다른 요청의 구조 보고서를 현재 검토 결과로 받아들이지 않는다."""
        harness = self.harness
        harness.frame = {"first": np.arange(3), "second": np.arange(4)}
        approval = harness.approval()

        def tampered(request, root, signal):
            outcome = harness.transport(request, root, signal)
            path = Path(request.output_directory) / "review/structure.json"
            report = json.loads(path.read_text())
            report["request_id"] = "f" * 64
            path.write_text(json.dumps(report), encoding="utf-8")
            return outcome

        with patch("src.agents.pickle_import.run_import_worker", side_effect=tampered):
            result = harness.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertNotIn("ingest", harness.tool.states.load().artifact_refs)

    def test_unsupported_and_cycles_need_review(self):
        """중첩 셀·사용자 객체·순환 구조를 추측 변환하지 않는다."""
        class Unknown:
            def __repr__(self):
                raise AssertionError("사용자 객체 repr 호출 금지")

        cycle = []
        cycle.append(cycle)

        for payload in (Unknown(), pd.DataFrame({"nested": [np.arange(2)]}), np.array([Unknown()], dtype=object), cycle):
            with self.subTest(kind=type(payload).__name__):
                value, report = analyze_structure(payload)
                self.assertIsNone(value)
                self.assertEqual(report["status"], "review_required")

    def test_selector_contract_and_approval_binding(self):
        """변환기·경로 변경에는 별도 계약·승인이 필요하다."""
        contract = self.harness.tool.contract

        for settings in ({"converter_id": "unknown"}, {"select_path": (True,)}, {"select_path": (-1,)},
                         {"select_path": ("a",) * 5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(contract, **settings)

        record_path = self.harness.directory / "selection.json"
        record = json.loads(record_path.read_text())
        record["contract"]["converter_id"] = "numeric_array_v1"
        record_path.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaises(ValueError):
            open_run("example")

    def test_schema_and_manifest_tampering(self):
        """해시와 실제 행의 타입을 각각 검증한다."""
        result, report = self.run_payload(pd.DataFrame({"number": [1, 2]}))
        self.assertEqual(result.status, "completed")
        canonical = Path(report["local_canonical_directory"])
        (canonical / "data.jsonl").write_text('{"index":0,"values":["wrong"]}\n{"index":1,"values":[2]}\n', encoding="utf-8")

        with self.assertRaises(ValueError):
            inspect_canonical(canonical, self.harness.tool.contract)

        with self.assertRaises(ValueError):
            validate_data(canonical)

    def test_output_bound_includes_metadata(self):
        """metadata 포함 출력 상한 초과는 완료로 처리하지 않는다."""
        from src.agents.pickle_contract import digest
        from src.agents.pickle_import_worker import execute_request

        harness = self.harness
        request = harness.tool._request("e" * 32)
        root = harness.tool._stage(request)
        settings = root / "input/import.json"
        packet = json.loads(settings.read_text())
        packet["max_output_bytes"] = 50
        request = replace(request, config_id=digest(packet))
        settings.write_text(json.dumps(packet), encoding="utf-8")

        with patch("pandas.read_pickle", return_value=np.arange(3)):
            reply = execute_request(request, root)

        self.assertEqual(reply.status, "blocked_input")
        self.assertFalse((Path(request.output_directory) / "canonical/SUCCESS.json").exists())


if __name__ == "__main__":
    unittest.main()

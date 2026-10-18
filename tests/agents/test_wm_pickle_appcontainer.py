"""명시적으로 활성화한 경우에만 실제 AppContainer로 고정 합성 pickle을 변환한다."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.pickle_import_cli import create, executor, open_run, status
from src.agents.worker_contract import WorkerLimits


@unittest.skipUnless(os.environ.get("WAFERGUARD_TEST_APPCONTAINER") == "1", "실제 Windows 격리 검사 선택 실행")
class WmPickleAppContainerTest(unittest.TestCase):
    def _generic_roundtrip(self, payload, converter):
        """고정 합성 표·배열을 실제 격리 런타임에 전달하고 부모 검증까지 확인한다."""
        from src.preprocessing.canonical_data import load_table

        with tempfile.TemporaryDirectory(prefix="pickle_native_") as temporary:
            root = Path(temporary)
            source = root / "synthetic.pkl"
            pd.to_pickle(payload, source)
            directory = root / "runs/native"
            directory.parent.mkdir()

            with patch("src.agents.pickle_import_cli._directory", return_value=directory):
                create("native", source, WorkerLimits(60, 2048, 2), 1024 * 1024, 1024 * 1024, confirmed=True)
                tool = open_run("native")
                manager = executor(tool)
                self.assertEqual(manager.run_next().code, "STEP_COMPLETED")
                self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")
                approval = manager.approvals.record_confirmed_approval(manager.scope_for("ingest"))
                result = manager.run_next(approval_id=approval.approval_id)
                diagnostics = [path.read_text(encoding="utf-8") for path in
                               (root / "workers").glob("*/scratch/task/diagnostic.json")]
                self.assertEqual(result.status, "completed", (result.code, diagnostics))
                self.assertTrue(tool.worker_started)
                report = status(open_run("native"))
                canonical = Path(report["local_canonical_directory"])

                if converter == "tabular_v1":
                    pd.testing.assert_frame_equal(load_table(canonical), payload)
                else:
                    np.testing.assert_array_equal(np.load(canonical / "array.npy", allow_pickle=False), payload)

    def test_generic_table_roundtrip(self):
        """문자열·결측값이 든 일반 표를 실제 AppContainer에서 변환한다."""
        self._generic_roundtrip(pd.DataFrame({"value": [1.0, float("nan")], "text": ["a", "b"]}), "tabular_v1")

    def test_generic_array_roundtrip(self):
        """숫자 배열을 실제 AppContainer에서 object 없는 NPY로 변환한다."""
        self._generic_roundtrip(np.arange(12, dtype=np.float32).reshape(3, 4), "numeric_array_v1")

    def test_synthetic_pickle_roundtrip(self):
        """원본 폴더 권한 없이 복사본을 읽고 숫자 canonical 결과를 승인·커밋한다."""
        with tempfile.TemporaryDirectory(prefix="wm_pickle_native_") as temporary:
            root = Path(temporary)
            source = root / "synthetic.pkl"
            array = np.array([[0, 1, 2], [1, 1, 2]], dtype=np.uint8)
            pd.DataFrame({"waferMap": [array], "failureType": ["Center"],
                          "lotName": ["synthetic_lot"]}).to_pickle(source)
            directory = root / "runs/native"
            directory.parent.mkdir()

            with patch("src.agents.pickle_import_cli._directory", return_value=directory):
                create("native", source, WorkerLimits(60, 2048, 2), 1024 * 1024, 1024 * 1024, confirmed=True)
                tool = open_run("native")
                manager = executor(tool)
                self.assertEqual(manager.run_next().code, "STEP_COMPLETED")
                self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")
                # 이 승인은 이 테스트에서 생성한 고정 합성 pickle 한 건에만 적용한다.
                approval = manager.approvals.record_confirmed_approval(manager.scope_for("ingest"))
                result = manager.run_next(approval_id=approval.approval_id)

                diagnostics = [path.read_text(encoding="utf-8") for path in
                               (root / "workers").glob("*/scratch/task/diagnostic.json")]
                self.assertEqual(result.status, "completed", (result.code, diagnostics))
                self.assertTrue(tool.worker_started)
                report = status(open_run("native"))
                canonical = Path(report["local_canonical_directory"])

                with np.load(canonical / "maps/shard_000000.npz", allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays["map_000000000"], array)

                self.assertEqual(report["canonical_integrity"], "passed")


if __name__ == "__main__":
    unittest.main()

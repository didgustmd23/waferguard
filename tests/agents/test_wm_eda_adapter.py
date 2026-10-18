"""합성 canonical metadata로 WM EDA 어댑터·실행기 연결을 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from scripts.wm_sandbox.export_canonical import export_frame
from src.agents.adapters.wm811k_metadata import WmMetadataEdaAdapter
from src.agents.approval_store import ApprovalStore
from src.agents.contracts import ToolResult
from src.agents.executor import ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.wafer_dl.validate_canonical import CLASS_NAMES


class WmEdaAdapterTest(unittest.TestCase):
    def setUp(self):
        """실제 데이터에서 파생되지 않은 9개 클래스의 작은 숫자 맵을 만든다."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.canonical = self.root / "canonical"
        rows = []

        for index, label in enumerate(CLASS_NAMES):
            array = np.ones((3, 4), dtype=np.uint8)
            array.flat[:index] = 2
            rows.append({
                "waferMap": array, "failureType": label,
                "lotName": f"independent_fixture_{index}",
            })

        export_frame(pd.DataFrame(rows), self.canonical, "wm811k_" + "b" * 64)
        (self.canonical / "SUCCESS.json").write_text(json.dumps({
            "status": "canonical_export_complete", "training_ready": False,
        }), encoding="utf-8")
        (self.canonical / "provenance.json").write_text(json.dumps({
            "source_sha256": "b" * 64, "source_bytes": 100,
        }), encoding="utf-8")

        self.adapter = WmMetadataEdaAdapter(self.canonical)

    def test_executor_waits_for_approval_then_generates_report(self):
        """공통 실행기의 승인·내용 검증·완료 표식을 실제 EDA 코어와 연결한다."""
        def preflight(output, arguments):
            (output / "synthetic_check.txt").write_text("passed", encoding="utf-8")

            return ToolResult(status="passed")

        states = LocalStateStore(self.root / "run")
        states.create("synthetic_operator", policy_version="test_policy")
        approvals = ApprovalStore(self.root / "approvals.sqlite3")
        plan = ExecutionPlan(
            self.adapter.source_id, self.adapter.config_id, "test_policy",
            (
                ExecutionStep("preflight", "run_preflight"),
                ExecutionStep("run_eda", "run_eda", json.dumps(self.adapter.arguments)),
            ),
        )
        executor = RulesExecutor(
            states, approvals, plan,
            (
                RegisteredTool(
                    "run_preflight", preflight,
                    lambda output, arguments: (output / "synthetic_check.txt").read_text() == "passed",
                ),
                self.adapter.registered_tool(),
            ),
        )

        self.assertEqual(executor.run_next().code, "STEP_COMPLETED")
        self.assertEqual(executor.run_next().code, "APPROVAL_REQUIRED")

        record = approvals.record_confirmed_approval(executor.scope_for("run_eda"))
        original_manifest = (self.canonical / "manifest.jsonl").read_bytes()

        # metadata 모드에서 예시 배열 로더가 호출되면 테스트를 즉시 실패시킨다.
        with patch("src.wafer_dl.eda.load_example_maps", side_effect=AssertionError("NPZ_READ_FORBIDDEN")):
            result = executor.run_next(approval_id=record.approval_id)

        self.assertEqual(result.status, "completed")
        reference = states.load().artifact_refs["run_eda"]
        output = states.directory / reference

        self.assertTrue((output / "eda/eda.md").is_file())
        self.assertTrue((output / "eda/correlations.png").is_file())
        self.assertFalse((output / "eda/examples.png").exists())
        self.assertTrue((output / "AGENT_COMPLETE.json").is_file())
        self.assertEqual((self.canonical / "manifest.jsonl").read_bytes(), original_manifest)

    def test_input_change_and_arguments_are_rejected(self):
        """고정 입력 변경과 예시 이미지 모드로의 우회를 거부한다."""
        with self.assertRaises(ValueError):
            self.adapter._check_input({"mode": "include_examples"})

        manifest = self.canonical / "manifest.jsonl"
        original = manifest.read_bytes()
        manifest.write_bytes(original + b"\n")

        with self.assertRaisesRegex(ValueError, "입력"):
            self.adapter._check_input(self.adapter.arguments)

        changed = WmMetadataEdaAdapter(self.canonical)

        self.assertNotEqual(changed.source_id, self.adapter.source_id)

    def test_constructor_does_not_read_npz_or_pickle(self):
        """metadata 식별에는 숫자 shard·pickle의 내용이 필요하지 않다."""
        for shard in self.canonical.rglob("*.npz"):
            shard.unlink()

        adapter = WmMetadataEdaAdapter(self.canonical)

        self.assertEqual(adapter.source_id, self.adapter.source_id)


if __name__ == "__main__":
    unittest.main()

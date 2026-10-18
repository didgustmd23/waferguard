"""개발 구간 변환 검사·Test 배열 금지·소실 검토 중단·출력 검증 검사."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.agents import test_wm_split_commit as fixtures
from src.agents import wm_transform_cli as check_cli
from src.agents.wm_canonical_contract import inventory
from src.agents.wm_transform_worker import execute_request
from src.agents.wm_transform_transport import ENTRY, WORKER_FILES
from src.agents.worker_contract import WorkerLimits
from src.wafer_dl.check_transform import read_checked_map


class WmTransformTest(unittest.TestCase):
    transport = fixtures.WmSplitCommitTest.transport
    proposal_transport = fixtures.WmSplitCommitTest.proposal_transport
    commit_transport = fixtures.WmSplitCommitTest.commit_transport

    def setUp(self):
        """이전 단계는 합성 fixture로 완료하고 새 검사만 별도 승인 대상으로 만든다."""
        fixtures.WmSplitCommitTest.setUp(self)
        approval = fixtures.WmSplitCommitTest.approval(self)

        with patch("src.agents.wm_split_commit.run_commit_worker", side_effect=self.commit_transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.split_tool = self.tool
        self.check_dir = self.root / "runs/check"
        mocking = patch("src.agents.wm_transform_cli._directory", return_value=self.check_dir)
        mocking.start()
        self.addCleanup(mocking.stop)
        check_cli.create("check", "commit", WorkerLimits(120, 2048, 2), 32 * 1024 * 1024, confirmed=True)
        self.tool = check_cli.open_run("check")
        self.manager = check_cli.executor(self.tool)

    def approval(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.wm_transform.run_check_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))

    def check_transport(self, request, root, signal):
        reply = execute_request(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes},
        }}
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
        return outcome

    def test_development_only_and_no_cache_or_source_change(self):
        """같은 shard에 있어도 Test 배열은 복원하지 않고 원본·캐시를 변경하지 않는다."""
        approval = self.approval()
        before = inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes)
        manifest = Path(self.tool.evidence["split_directory"]) / "split_manifest.jsonl"
        allowed = {row["sample_id"] for row in map(json.loads, manifest.read_text().splitlines())
                   if row["split"] in {"train", "validation"}}
        seen = set()

        def only_development(archive, row):
            self.assertIn(row["sample_id"], allowed)
            seen.add(row["sample_id"])
            return read_checked_map(archive, row)

        with patch("src.agents.wm_transform.run_check_worker", side_effect=self.check_transport), \
                patch("src.wafer_dl.check_transform.read_checked_map", side_effect=only_development):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "completed", result)
        self.assertEqual(seen, allowed)
        status = check_cli.status(check_cli.open_run("check"))
        self.assertEqual(status["check_status"], "transform_check_passed")
        self.assertFalse(status["test_maps_read"])
        self.assertTrue(status["visual_review_required"])
        self.assertEqual(inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes), before)
        self.assertEqual({path.name for path in Path(status["local_result_directory"]).iterdir()},
                         {"preservation.csv", "examples.png", "transform_check.json", "transform_check.md", "SUCCESS.json", "result.json"})

    def test_loss_requires_review_without_completion(self):
        """영역 소실이 있으면 보고서는 보존하되 완료 등록·자동 해상도 변경을 하지 않는다."""
        import numpy as np
        from src.wafer_dl.transform import transform_map

        def lost(original, config):
            transformed, geometry = transform_map(original, config)
            return np.zeros_like(transformed), geometry

        approval = self.approval()

        with patch("src.agents.wm_transform.run_check_worker", side_effect=self.check_transport), \
                patch("src.wafer_dl.isolated_transform_check.transform_map", side_effect=lost):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED", result)
        status = check_cli.status(check_cli.open_run("check"))
        self.assertEqual(status["check_status"], "transform_review_required")
        self.assertNotIn("derive_preparation", self.tool.states.load().artifact_refs)

    def test_changed_split_receipt_rejected(self):
        next(self.commit_dir.glob("attempts/*/receipt.json")).write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            check_cli.open_run("check")

    def test_tampered_csv_not_committed(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.check_transport(request, root, signal)
            (Path(request.output_directory) / "check/preservation.csv").write_bytes(b"changed")
            return outcome

        with patch("src.agents.wm_transform.run_check_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")

    def test_missing_cleanup_rejected(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.check_transport(request, root, signal)
            outcome["launch"]["resources"]["cleanup_confirmed"] = False
            return outcome

        with patch("src.agents.wm_transform.run_check_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")

    def test_existing_and_isolated_core_results_match(self):
        """기존 변환 CLI 코어와 격리 wrapper의 행별 지표·클래스 집계가 같다."""
        from src.wafer_dl.check_transform import run_transform_check
        from src.wafer_dl.isolated_transform_check import run_isolated_transform_check

        config_path = Path(__file__).resolve().parents[2] / "configs/wm811k/transform.json"
        split_dir = self.tool.evidence["split_directory"]
        canonical = self.tool.contract.snapshot_directory
        original_output, isolated_output = self.root / "original_check", self.root / "isolated_check"
        workspace = self.root / "isolated_workspace"
        workspace.mkdir()
        original_result = run_transform_check(split_dir, original_output, config_path, canonical)
        isolated_result = run_isolated_transform_check(split_dir, isolated_output, config_path, canonical,
                                                       workspace=workspace, max_output_bytes=32 * 1024 * 1024)
        self.assertEqual(original_result, isolated_result)
        self.assertEqual((original_output / "preservation.csv").read_bytes(),
                         (isolated_output / "preservation.csv").read_bytes())
        self.assertEqual(json.loads((original_output / "transform_check.json").read_text(encoding="utf-8")),
                         json.loads((isolated_output / "transform_check.json").read_text(encoding="utf-8")))

    def test_copied_worker_bootstraps(self):
        copied = self.root / "copied"
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        script = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
                  "from src.agents.wm_transform_worker import execute_request; "
                  "from src.agents.wm_transform_results import inspect_check; "
                  "from src.wafer_dl.isolated_transform_check import run_isolated_transform_check")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(copied),
                                 str(Path(sys.prefix) / "Lib/site-packages")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

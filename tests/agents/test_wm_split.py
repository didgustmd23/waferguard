"""완료 EDA에 연결한 WM 그룹 분할 제안의 승인·비파괴·출력 검증 검사."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.agents import test_wm_canonical as fixtures
from src.agents import wm_split_cli as proposal_cli
from src.agents.wm_canonical_contract import inventory
from src.agents.wm_split_transport import ENTRY, WORKER_FILES
from src.agents.wm_split_worker import execute_request
from src.agents.worker_contract import WorkerLimits


class WmSplitTest(unittest.TestCase):
    transport = fixtures.WmCanonicalTest.transport

    def setUp(self):
        """27개 합성 맵으로 기존 품질·EDA를 완료하고 새 제안 작업을 만든다."""
        fixtures.WmCanonicalTest.setUp(self)
        self.manager.run_next()
        self.manager.run_next()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda"))

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=self.transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.eda_tool = self.tool
        self.proposal_dir = self.root / "runs/proposal"
        mocking = patch("src.agents.wm_split_cli._directory", return_value=self.proposal_dir)
        mocking.start()
        self.addCleanup(mocking.stop)
        proposal_cli.create("proposal", "example", WorkerLimits(120, 2048, 2), confirmed=True)
        self.tool = proposal_cli.open_run("proposal")
        self.manager = proposal_cli.executor(self.tool)

    def approval(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.wm_split.run_proposal_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))

    def proposal_transport(self, request, root, signal):
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

    def test_proposal_preserves_source_and_never_finalizes(self):
        """배열을 읽거나 분할을 확정하지 않고 승인된 메타데이터 후보만 만든다."""
        approval = self.approval()
        before = inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes)
        eda_state = self.eda_tool.states.load()

        with patch("src.agents.wm_split.run_proposal_worker", side_effect=self.proposal_transport) as launch, \
                patch("numpy.load", side_effect=AssertionError("분할안에서 배열 파싱 금지")):
            result = self.manager.run_next(approval_id=approval.approval_id)
            self.manager.run_next()

        self.assertEqual(result.status, "completed", result)
        launch.assert_called_once()
        report = proposal_cli.status(proposal_cli.open_run("proposal"))
        self.assertEqual(report["proposal_status"], "proposal_ready")
        self.assertFalse(report["split_complete"])
        preview = json.loads(Path(report["local_report"]).with_suffix(".json").read_text(encoding="utf-8"))
        self.assertTrue(preview["overlap_check_passed"])
        self.assertEqual(preview["config"]["ratios"], [0.7, 0.15, 0.15])
        self.assertEqual(len(preview["candidates"]), 32)
        self.assertEqual(self.eda_tool.states.load(), eda_state)
        self.assertEqual(inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes), before)
        self.assertFalse(Path(report["local_report"]).with_name("split_manifest.jsonl").exists())

    def test_changed_eda_result_rejected(self):
        next(self.directory.glob("attempts/*/receipt.json")).write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            proposal_cli.open_run("proposal")

    def test_tampered_group_assignments_rejected(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.proposal_transport(request, root, signal)
            (Path(request.output_directory) / "proposal/groups.csv").write_bytes(b"changed")
            return outcome

        with patch("src.agents.wm_split.run_proposal_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("derive_preparation", self.tool.states.load().artifact_refs)

    def test_changed_snapshot_never_parses_manifest(self):
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        next((root / "input/canonical/maps").iterdir()).write_bytes(b"changed")

        with patch("src.wafer_dl.validate_canonical.validate_manifest", side_effect=AssertionError("변조 파싱 금지")):
            self.assertEqual(execute_request(request, root).status, "blocked_input")

    def test_missing_cleanup_never_commits(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.proposal_transport(request, root, signal)
            outcome["launch"]["resources"]["cleanup_confirmed"] = False
            return outcome

        with patch("src.agents.wm_split.run_proposal_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")

    def test_copied_worker_bootstraps(self):
        copied = self.root / "copied"
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        script = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
                  "from src.agents.wm_split_worker import execute_request; "
                  "from src.agents.wm_split_results import inspect_proposal; "
                  "from src.wafer_dl.split import choose_assignment")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(copied),
                                 str(Path(sys.prefix) / "Lib/site-packages")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

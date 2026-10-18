"""WM 고정 그룹 배정 확정·원본 행 대응·기존 변환기 호환성 검사."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.agents import test_wm_split as fixtures
from src.agents import wm_split_commit_cli as commit_cli
from src.agents.wm_canonical_contract import inventory
from src.agents.wm_split_commit_transport import WORKER_FILES, ENTRY
from src.agents.wm_split_commit_worker import execute_request
from src.agents.worker_contract import WorkerLimits
from src.wafer_dl.check_transform import index_split


class WmSplitCommitTest(unittest.TestCase):
    transport = fixtures.WmSplitTest.transport
    proposal_transport = fixtures.WmSplitTest.proposal_transport

    def setUp(self):
        """실제 사용자 기록 대신 합성 품질·EDA·제안을 완료한 뒤 새 확정을 준비한다."""
        fixtures.WmSplitTest.setUp(self)
        self.manager.run_next()
        self.manager.run_next()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))

        with patch("src.agents.wm_split.run_proposal_worker", side_effect=self.proposal_transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.proposal_tool = self.tool
        self.commit_dir = self.root / "runs/commit"
        mocking = patch("src.agents.wm_split_commit_cli._directory", return_value=self.commit_dir)
        mocking.start()
        self.addCleanup(mocking.stop)
        commit_cli.create("commit", "proposal", WorkerLimits(120, 2048, 2), 32 * 1024 * 1024, confirmed=True)
        self.tool = commit_cli.open_run("commit")
        self.manager = commit_cli.executor(self.tool)

    def approval(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")
        review = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("review_eda"))
        self.assertEqual(self.manager.run_next(approval_id=review.approval_id).code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.wm_split_commit.run_commit_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("commit_split"))

    def commit_transport(self, request, root, signal):
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

    def test_fixed_assignment_commits_without_search_or_arrays(self):
        """검토한 배정을 그대로 적용하고 기존 index_split으로 모든 원본 행을 재대조한다."""
        approval = self.approval()
        before = inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes)
        proposal_state = self.proposal_tool.states.load()

        with patch("src.agents.wm_split_commit.run_commit_worker", side_effect=self.commit_transport) as launch, \
                patch("src.wafer_dl.split.choose_assignment", side_effect=AssertionError("후보 재탐색 금지")), \
                patch("numpy.load", side_effect=AssertionError("배열 파싱 금지")):
            result = self.manager.run_next(approval_id=approval.approval_id)
            self.manager.run_next()

        self.assertEqual(result.status, "completed", result)
        launch.assert_called_once()
        status = commit_cli.status(commit_cli.open_run("commit"))
        self.assertTrue(status["split_complete"])
        self.assertFalse(status["training_ready"])
        directory = Path(status["local_split_directory"])
        database = sqlite3.connect(":memory:")

        try:
            report = index_split(Path(self.tool.contract.snapshot_directory), directory, database)
        finally:
            database.close()

        self.assertEqual(report["class_counts"], self.tool.evidence["proposal"]["class_counts"])
        self.assertEqual(report["selected_candidate"], self.tool.evidence["proposal"]["selected_candidate"])
        self.assertEqual(self.proposal_tool.states.load(), proposal_state)
        self.assertEqual(inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes), before)

    def test_changed_proposal_receipt_rejected(self):
        next(self.proposal_dir.glob("attempts/*/receipt.json")).write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            commit_cli.open_run("commit")

    def test_changed_group_copy_blocks_before_commit(self):
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        (root / "input/proposal/groups.csv").write_bytes(b"changed")

        with patch("src.wafer_dl.commit_split.commit_reviewed_split", side_effect=AssertionError("변조 적용 금지")):
            self.assertEqual(execute_request(request, root).status, "blocked_input")

    def test_changed_output_never_commits(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.commit_transport(request, root, signal)
            (Path(request.output_directory) / "split/split_manifest.jsonl").write_bytes(b"changed")
            return outcome

        with patch("src.agents.wm_split_commit.run_commit_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("commit_split", self.tool.states.load().artifact_refs)

    def test_missing_cleanup_never_commits(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.commit_transport(request, root, signal)
            outcome["launch"]["resources"]["cleanup_confirmed"] = False
            return outcome

        with patch("src.agents.wm_split_commit.run_commit_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")

    def test_cancel_before_worker(self):
        approval = self.approval()
        commit_cli.cancel(self.tool)

        with patch("src.agents.wm_split_commit.run_commit_worker") as launch:
            result = self.manager.run_next(approval_id=approval.approval_id)

        launch.assert_not_called()
        self.assertNotEqual(result.status, "completed")

    def test_copied_worker_bootstraps(self):
        copied = self.root / "copied"
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        script = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
                  "from src.agents.wm_split_commit_worker import execute_request; "
                  "from src.agents.wm_split_commit_results import inspect_split; "
                  "from src.wafer_dl.commit_split import commit_reviewed_split")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(copied),
                                 str(Path(sys.prefix) / "Lib/site-packages")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

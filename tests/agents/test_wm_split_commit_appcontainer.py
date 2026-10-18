"""선택 실행하는 WM 고정 그룹 분할 확정의 실제 AppContainer 합성 검사."""

import os
from pathlib import Path
import unittest

from tests.agents import test_wm_split_commit as fixtures
from src.agents import wm_split_commit_cli as commit_cli


@unittest.skipUnless(os.environ.get("WAFERGUARD_TEST_APPCONTAINER") == "1", "실제 Windows 격리 검사 선택 실행")
class WmSplitCommitAppContainerTest(unittest.TestCase):
    setUp = fixtures.WmSplitCommitTest.setUp
    transport = fixtures.WmSplitCommitTest.transport
    proposal_transport = fixtures.WmSplitCommitTest.proposal_transport
    approval = fixtures.WmSplitCommitTest.approval

    def test_native_fixed_assignment_commit(self):
        """이전 제안은 모의 OS로 준비하고 실제 확정 worker·Job 정리·커밋을 확인한다."""
        approval = self.approval()
        result = self.manager.run_next(approval_id=approval.approval_id)
        diagnostics = [path.read_text() for path in (self.root / "workers").glob("*/scratch/task/diagnostic.json")]
        self.assertEqual(result.status, "completed", (result, diagnostics))
        self.assertTrue(self.tool.worker_started)
        report = commit_cli.status(commit_cli.open_run("commit"))
        self.assertTrue(report["split_complete"])
        self.assertFalse(report["training_ready"])
        self.assertTrue((Path(report["local_split_directory"]) / "split_manifest.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()

"""선택 실행하는 합성 WM 그룹 분할 제안의 실제 AppContainer 검사."""

import os
from pathlib import Path
import unittest

from tests.agents import test_wm_split as fixtures
from src.agents import wm_split_cli as proposal_cli


@unittest.skipUnless(os.environ.get("WAFERGUARD_TEST_APPCONTAINER") == "1", "실제 Windows 격리 검사 선택 실행")
class WmSplitAppContainerTest(unittest.TestCase):
    setUp = fixtures.WmSplitTest.setUp
    transport = fixtures.WmSplitTest.transport
    approval = fixtures.WmSplitTest.approval

    def test_native_proposal_without_finalization(self):
        """이전 EDA만 모의 OS로 준비하고 새 제안의 실제 격리·정리·커밋을 확인한다."""
        approval = self.approval()
        result = self.manager.run_next(approval_id=approval.approval_id)
        diagnostics = [path.read_text() for path in (self.root / "workers").glob("*/scratch/task/diagnostic.json")]
        self.assertEqual(result.status, "completed", (result, diagnostics))
        self.assertTrue(self.tool.worker_started)
        report = proposal_cli.status(proposal_cli.open_run("proposal"))
        self.assertEqual(report["proposal_status"], "proposal_ready")
        self.assertFalse(report["split_complete"])
        self.assertTrue(Path(report["local_report"]).is_file())


if __name__ == "__main__":
    unittest.main()

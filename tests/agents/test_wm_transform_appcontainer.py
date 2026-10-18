"""선택 실행하는 합성 WM 개발 구간 변환 검사의 실제 AppContainer 검사."""

import os
from pathlib import Path
import unittest

from tests.agents import test_wm_transform as fixtures
from src.agents import wm_transform_cli as check_cli


@unittest.skipUnless(os.environ.get("WAFERGUARD_TEST_APPCONTAINER") == "1", "실제 Windows 격리 검사 선택 실행")
class WmTransformAppContainerTest(unittest.TestCase):
    setUp = fixtures.WmTransformTest.setUp
    transport = fixtures.WmTransformTest.transport
    proposal_transport = fixtures.WmTransformTest.proposal_transport
    commit_transport = fixtures.WmTransformTest.commit_transport
    approval = fixtures.WmTransformTest.approval

    def test_native_transform_check(self):
        """이전 확정 기록은 모의 OS로 준비하고 새 검사·그림·Job 정리는 실제 격리 실행한다."""
        approval = self.approval()
        result = self.manager.run_next(approval_id=approval.approval_id)
        diagnostics = [path.read_text() for path in (self.root / "workers").glob("*/scratch/task/diagnostic.json")]
        self.assertEqual(result.status, "completed", (result, diagnostics))
        self.assertTrue(self.tool.worker_started)
        report = check_cli.status(check_cli.open_run("check"))
        self.assertEqual(report["check_status"], "transform_check_passed")
        self.assertFalse(report["test_maps_read"])
        self.assertTrue(report["visual_review_required"])
        self.assertTrue((Path(report["local_result_directory"]) / "examples.png").is_file())


if __name__ == "__main__":
    unittest.main()

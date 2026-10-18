"""명시적으로 선택한 경우에만 합성 충돌 후보를 실제 Windows 격리 환경에서 검사한다."""

import os
from pathlib import Path
import unittest

from tests.agents import test_wm_conflict as fixtures
from src.agents import wm_conflict_cli as candidate_cli


@unittest.skipUnless(os.environ.get("WAFERGUARD_TEST_APPCONTAINER") == "1", "실제 Windows 격리 검사 선택 실행")
class WmConflictAppContainerTest(unittest.TestCase):
    # 이전 품질 기록만 모의 OS 전송으로 준비한다. 이번 후보 worker는 실제 AppContainer에서 실행한다.
    setUp = fixtures.WmConflictTest.setUp
    transport = staticmethod(fixtures.WmConflictTest.transport)
    quality_transport = fixtures.WmConflictTest.quality_transport
    approval = fixtures.WmConflictTest.approval

    def test_candidate_native_roundtrip(self):
        """합성 입력의 manifest 진단·후보 검증·Job 정리·커밋을 실제 격리 환경에서 확인한다."""
        approval = self.approval()
        result = self.manager.run_next(approval_id=approval.approval_id)
        diagnostics = [path.read_text() for path in (self.root / "workers").glob("*/scratch/task/diagnostic.json")]
        self.assertEqual(result.status, "completed", (result, diagnostics))
        self.assertTrue(self.tool.worker_started)
        report = candidate_cli.status(candidate_cli.open_run("candidate"))
        self.assertEqual(report["candidate_status"], "candidate_ready")
        self.assertFalse(report["data_modified"])
        self.assertFalse(report["exclusion_approved"])
        self.assertTrue(Path(report["local_report"]).is_file())


if __name__ == "__main__":
    unittest.main()

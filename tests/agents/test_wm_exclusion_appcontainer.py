"""선택 실행하는 합성 WM 충돌 제외 적용의 실제 AppContainer 검사."""

import os
from pathlib import Path
import unittest

from tests.agents import test_wm_exclusion as fixtures
from src.agents import wm_exclusion_cli as apply_cli


@unittest.skipUnless(os.environ.get("WAFERGUARD_TEST_APPCONTAINER") == "1", "실제 Windows 격리 검사 선택 실행")
class WmExclusionAppContainerTest(unittest.TestCase):
    native_runtime = True
    # 이전 품질·후보 기록만 모의 OS 전송으로 준비한다. 새 제외 worker는 실제 격리 실행한다.
    setUp = fixtures.WmExclusionTest.setUp
    transport = staticmethod(fixtures.WmExclusionTest.transport)
    quality_transport = fixtures.WmExclusionTest.quality_transport
    candidate_transport = fixtures.WmExclusionTest.candidate_transport
    approval = fixtures.WmExclusionTest.approval

    def test_native_derived_canonical(self):
        """새 canonical 생성·숫자 배열 재검증·Job 정리·부모 검증·커밋을 확인한다."""
        approval = self.approval()
        result = self.manager.run_next(approval_id=approval.approval_id)
        diagnostics = [path.read_text() for path in (self.root / "workers").glob("*/scratch/task/diagnostic.json")]
        self.assertEqual(result.status, "completed", (result, diagnostics))
        self.assertTrue(self.tool.worker_started)
        report = apply_cli.status(apply_cli.open_run("apply"))
        self.assertEqual(report["derived_status"], "derived_ready")
        self.assertFalse(report["original_modified"])
        self.assertFalse(report["training_ready"])
        self.assertTrue(Path(report["local_canonical"]).is_dir())


if __name__ == "__main__":
    unittest.main()

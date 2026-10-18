"""명시적으로 선택한 경우에만 합성 WM canonical을 실제 AppContainer에서 처리한다."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.wm_canonical_cli import create_demo, executor, open_run, status
from src.agents.worker_contract import WorkerLimits


@unittest.skipUnless(os.environ.get("WAFERGUARD_TEST_APPCONTAINER") == "1", "실제 Windows 격리 검사 선택 실행")
class WmCanonicalAppContainerTest(unittest.TestCase):
    def _run_demo(self, review):
        """승인은 이 테스트가 만든 합성 canonical 한 건의 정확한 범위에만 발급한다."""
        with tempfile.TemporaryDirectory(prefix="wm_eda_native_") as temporary:
            root = Path(temporary)
            directory = root / "runs/native"
            directory.parent.mkdir()

            with patch("src.agents.wm_canonical_cli._directory", return_value=directory):
                create_demo("native", WorkerLimits(120, 2048, 2), review=review)
                tool = open_run("native")
                manager = executor(tool)
                self.assertEqual(manager.run_next().code, "STEP_COMPLETED")
                self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")
                approval = manager.approvals.record_confirmed_approval(manager.scope_for("run_eda"))
                result = manager.run_next(approval_id=approval.approval_id)
                diagnostics = [path.read_text() for path in (root / "workers").glob("*/scratch/task/diagnostic.json")]
                self.assertEqual(result.status, "blocked_input" if review else "completed", (result, diagnostics))
                self.assertTrue(tool.worker_started)
                report = status(open_run("native"))
                self.assertEqual(report["quality_status"], "review_required" if review else "passed")
                self.assertEqual(report["eda_generated"], not review)

                if review:
                    self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
                    self.assertNotIn("run_eda", tool.states.load().artifact_refs)
                else:
                    self.assertTrue(Path(report["local_eda_report"]).is_file())

    def test_quality_and_eda_roundtrip(self):
        """정상 합성 묶음의 검증·EDA·부모 검증·커밋·프로필 정리를 확인한다."""
        self._run_demo(False)

    def test_review_stops_before_eda(self):
        """실제 격리 환경에서도 검토 필요를 자동 해제하거나 EDA를 실행하지 않는다."""
        self._run_demo(True)


if __name__ == "__main__":
    unittest.main()

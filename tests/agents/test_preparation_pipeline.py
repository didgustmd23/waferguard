"""통합 실행 관리자의 합성 연결 검사. 네이티브 운송만 모의한다."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.agents import test_preparation_candidate as fixtures
from src.agents import preparation_pipeline_cli as cli
from src.agents import preparation_candidate_cli as candidate
from src.agents.preparation_apply_worker import execute_request as apply_worker
from src.agents.worker_contract import WorkerLimits


class PreparationPipelineTest(unittest.TestCase):
    fixture_commit = fixtures.PreparationCandidateTest.fixture_commit
    complete = fixtures.PreparationCandidateTest.complete
    transport = fixtures.PreparationCandidateTest.transport

    def setUp(self):
        fixtures.PreparationCandidateTest.setUp(self)
        self.pipeline_root = self.root / "pipeline_run"
        self.apply_root = self.root / "apply_run"
        self.config_path = self.root / "pipeline.json"
        example = Path(__file__).resolve().parents[2] / "configs/agents/pipelines/tabular_preparation_example.json"
        config = json.loads(example.read_text(encoding="utf-8"))
        # 합성 범주가 행마다 다르므로 미지 범주 정책을 명시한다.
        config["steps"][0]["settings"]["unknown_category"] = "ignore"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        for target, value in (
            ("src.agents.preparation_pipeline_cli._directory", self.pipeline_root),
            ("src.agents.preparation_pipeline_cli._current_actor", "synthetic_actor"),
            ("src.agents.preparation_apply_cli._directory", self.apply_root),
            ("src.agents.preparation_apply_cli._current_actor", "synthetic_actor"),
        ):
            context = patch(target, return_value=value)
            context.start()
            self.addCleanup(context.stop)

    def create_run(self):
        return cli.create("example", self.config_path, self.split_run, self.scope_path, self.export_run,
                          WorkerLimits(60, 2048, 2), 65536, 65536, confirmed=True)

    def approve(self, prefix):
        with patch("os.isatty", return_value=True), patch("secrets.token_hex", return_value="abcd"), patch("builtins.input", return_value=prefix + "-ABCD"):
            return cli.operate("example", "approve")

    def finish_candidate(self):
        self.assertEqual(cli.operate("example", "next")["code"], "STEP_COMPLETED")
        self.assertEqual(cli.operate("example", "next")["code"], "APPROVAL_REQUIRED")
        approval = self.approve("CANDIDATE")

        with patch("src.agents.preparation_candidate.run_candidate_worker", side_effect=self.transport):
            self.assertEqual(cli.operate("example", "next", approval["approval_id"])["status"], "completed")

        return approval["approval_id"]

    def review(self):
        with patch("os.isatty", return_value=True), patch("secrets.token_hex", return_value="abcd"), patch("builtins.input", return_value="REVIEW-ABCD"):
            return cli.operate("example", "review")

    def test_full_flow_separate_approvals_and_resume(self):
        """기존 실행기를 연결하되 후보 승인으로 적용하지 않고 완료 후 재실행하지 않는다."""
        self.assertFalse(self.create_run()["worker_started"])
        self.finish_candidate()
        self.assertEqual(cli.operate("example", "next")["status"], "awaiting_candidate_review")
        self.assertFalse(self.apply_root.exists())
        self.assertEqual(self.review()["status"], "application_created")
        self.assertEqual(cli.operate("example", "next")["code"], "STEP_COMPLETED")
        self.assertEqual(cli.operate("example", "next")["code"], "APPROVAL_REQUIRED")

        approval = self.approve("APPLY")

        with patch("src.agents.preparation_apply.run_apply_worker", side_effect=lambda request, root, signal: self.transport(request, root, signal, apply_worker)):
            self.assertEqual(cli.operate("example", "next", approval["approval_id"])["status"], "completed")

        result = cli.operate("example", "status")
        self.assertFalse(result["refit"])
        self.assertFalse(result["test_used"])

        with patch("src.agents.preparation_apply.run_apply_worker") as launch:
            cli.operate("example", "next")

        launch.assert_not_called()

    def test_candidate_approval_rejected_for_application(self):
        """다른 단계의 승인은 거부하고 실패 상태를 자동 초기화하지 않는다."""
        self.create_run()
        old_approval = self.finish_candidate()
        self.review()
        cli.operate("example", "next")
        cli.operate("example", "next")

        with patch("src.agents.preparation_apply.run_apply_worker") as launch:
            result = cli.operate("example", "next", old_approval)

        launch.assert_not_called()
        self.assertNotEqual(result["status"], "completed")

        with self.assertRaises(ValueError):
            self.approve("APPLY")

    def test_missing_approval(self):
        """통합 CLI도 승인 없이 worker를 시작하지 않는다."""
        self.create_run()
        cli.operate("example", "next")

        with patch("src.agents.preparation_candidate.run_candidate_worker") as launch:
            self.assertEqual(cli.operate("example", "next")["code"], "APPROVAL_REQUIRED")

        launch.assert_not_called()

    def test_review_declined_and_early_review(self):
        """후보 완료 전·확인 문구 불일치 시 적용 작업을 생성하지 않는다."""
        self.create_run()

        with self.assertRaises(ValueError):
            cli.operate("example", "review")

        self.finish_candidate()

        with patch("os.isatty", return_value=True), patch("builtins.input", return_value="아니요"):
            self.assertEqual(cli.operate("example", "review")["code"], "REVIEW_DECLINED")

        self.assertFalse(self.apply_root.exists())

    def test_cancel_before_execution(self):
        """취소 신호를 하위 실행에 전달하고 이후 실행을 막는다."""
        self.create_run()
        self.assertEqual(cli.operate("example", "cancel")["status"], "cancel_requested")

        with patch("src.agents.preparation_candidate.run_candidate_worker") as launch:
            self.assertEqual(cli.operate("example", "next")["status"], "cancel_requested")

        launch.assert_not_called()

    def test_cancel_at_review_preserves_candidate(self):
        """완료 후보는 보존하고 후속 적용을 차단한다."""
        self.create_run()
        self.finish_candidate()
        cli.operate("example", "cancel")
        self.assertEqual(candidate.open_run("example-candidate").states.load().status, "completed")
        self.assertEqual(self.review()["status"], "cancel_requested")
        self.assertFalse(self.apply_root.exists())

    def test_plan_tampering_and_duplicate_create(self):
        """계획 변경·기존 실행 덮어쓰기를 거부한다."""
        self.create_run()

        with self.assertRaises(FileExistsError):
            self.create_run()

        path = self.pipeline_root / "pipeline.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["steps"][0]["settings"]["numeric_imputation"] = "mean"
        path.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaises(ValueError):
            cli.operate("example", "status")

    def test_review_binding_tampering(self):
        """후보 검토 후 연결 식별값이 바뀌면 적용을 시작하지 않는다."""
        self.create_run()
        self.finish_candidate()
        self.review()
        path = self.pipeline_root / "review.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["candidate_id"] = "0" * 64
        path.write_text(json.dumps(record), encoding="utf-8")

        with patch("src.agents.preparation_apply.run_apply_worker") as launch, self.assertRaises(ValueError):
            cli.operate("example", "next")

        launch.assert_not_called()

    def test_lock_and_code_change(self):
        """동시 명령과 통합 제어 코드 변경은 재개 전에 거부한다."""
        self.create_run()

        with cli._run_lock(self.pipeline_root) as acquired:
            self.assertTrue(acquired)

            with self.assertRaises(ValueError):
                cli.operate("example", "next")

        with patch.object(cli, "_code_id", return_value="changed"), self.assertRaises(ValueError):
            cli.operate("example", "status")


if __name__ == "__main__":
    unittest.main()

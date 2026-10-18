"""내보내기의 승인·worker·부모 검증·커밋 연결. 네이티브 OS 운송만 모의한다."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.agents import test_partition_export as fixtures
from src.agents.partition_export import PartitionExportTool
from src.agents.partition_export_cli import executor, create, open_run, approve, status, cancel
from src.agents.partition_export_contract import POLICY_VERSION
from src.agents.partition_export_worker import execute_request
from src.agents.partition_export_transport import WORKER_FILES, ENTRY
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits


class PartitionExportExecutionTest(unittest.TestCase):
    # 기존 합성 원본·확정 분할 fixture만 재사용한다. 테스트 케이스는 상속하지 않는다.
    fixture_commit = fixtures.PartitionExportTest.fixture_commit
    complete = fixtures.PartitionExportTest.complete

    def setUp(self):
        fixtures.PartitionExportTest.setUp(self)
        self.states = LocalStateStore(self.root / "runs/example")
        self.states.create("synthetic_actor", policy_version=POLICY_VERSION)
        self.tool = PartitionExportTool(self.states, self.contract)
        self.manager = executor(self.tool)
        self.calls = 0

    def waiting(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

    def approval(self):
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("ingest")).approval_id

    def transport(self, request, root, signal):
        """추출·파일 저장은 본 worker로 수행하되 AppContainer를 실제 실행하지 않는다."""
        self.calls += 1
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

    def run_export(self, transport=None):
        self.waiting()

        with patch("src.agents.partition_export.run_export_worker", side_effect=transport or self.transport):
            result = self.manager.run_next(approval_id=self.approval())

        task = self.manager.approvals.get_task(self.manager.scope_for("ingest"))

        return result, self.manager._output(task), task

    def test_approval_commit_contracts_and_no_rerun(self):
        """승인 전 실행하지 않으며 승인 1회로 커밋하고 두 구간 입력 계약을 확인한다."""
        with patch("src.agents.partition_export.run_export_worker") as launch:
            self.waiting()

        launch.assert_not_called()

        with patch("src.agents.partition_export.run_export_worker", side_effect=self.transport):
            result = self.manager.run_next(approval_id=self.approval())

        self.assertEqual(result.status, "completed")
        self.assertEqual(status(self.tool)["partition_contracts"], "passed")
        self.assertEqual(self.calls, 1)

        with patch("src.agents.partition_export.run_export_worker") as launch:
            self.manager.run_next()

        launch.assert_not_called()

    def test_resource_failure_never_commits(self):
        """부분 결과가 있어도 시간 초과를 성공으로 바꾸지 않는다."""
        def timed_out(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["status"] = "timed_out"
            return outcome

        result, output, task = self.run_export(timed_out)
        self.assertEqual(result.status, "failed_runtime")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())
        self.assertNotEqual(task["status"], "committed")

    def test_missing_job_cleanup_never_commits(self):
        """종료 코드 0이어도 Job 정리 증거가 없으면 거부한다."""
        def missing(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["resources"]["cleanup_confirmed"] = False
            return outcome

        result, output, _ = self.run_export(missing)
        self.assertEqual(result.status, "failed_runtime")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_cancel_during_worker_never_commits(self):
        """worker 성공 이후라도 취소 요청이 있으면 등록하지 않는다."""
        def cancelled(request, root, signal):
            outcome = self.transport(request, root, signal)
            cancel(self.tool)
            return outcome

        result, output, _ = self.run_export(cancelled)
        self.assertNotEqual(result.status, "completed")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_output_overflow_never_commits(self):
        """출력 상한은 파일을 자르지 않고 입력 차단으로 종료한다."""
        self.tool = PartitionExportTool(self.states, replace(self.contract, max_partition_bytes=16))
        self.manager = executor(self.tool)
        result, output, _ = self.run_export()
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_changed_limits_reject_previous_approval(self):
        """출력 크기 변경은 기존 계획·승인을 사용하지 않는다."""
        self.waiting()
        approval = self.approval()
        changed = PartitionExportTool(self.states, replace(self.contract, max_partition_bytes=32768))

        with patch("src.agents.partition_export.run_export_worker") as launch:
            result = executor(changed).run_next(approval_id=approval)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        launch.assert_not_called()

    def test_parent_rejects_tampered_payload(self):
        """worker 응답이 성공이어도 구간 바이트 변조는 거부한다."""
        def tampered(request, root, signal):
            outcome = self.transport(request, root, signal)
            (Path(request.output_directory) / "exports/train/input.bin").write_bytes(b"changed")
            return outcome

        result, output, _ = self.run_export(tampered)
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_cli_create_reopen_and_explicit_confirmation(self):
        """생성은 worker를 시작하지 않으며 선택 변경·비대화형 승인을 거부한다."""
        scope_path = self.root / "scope.json"
        scope_path.write_text(self.split.artifact.scope_json, encoding="utf-8")
        profile_path = self.root / "profile.json"
        profile_path.write_text(json.dumps(self.profile.to_dict()), encoding="utf-8")
        args = ("example", self.split.artifact.run_directory, scope_path, self.root / "snapshot",
                profile_path, WorkerLimits(60, 2048, 2), 65536)

        with patch("src.agents.partition_export_cli._directory", return_value=self.root / "cli_run"), patch("src.agents.partition_export_cli._current_actor", return_value="synthetic_actor"):
            with self.assertRaises(ValueError):
                create(*args)

            self.assertFalse(create(*args, confirmed=True)["worker_started"])
            tool = open_run("example")
            self.assertEqual(tool.contract, self.contract)

            with patch("os.isatty", return_value=False), self.assertRaises(ValueError):
                approve(tool)

            selection = self.root / "cli_run/selection.json"
            record = json.loads(selection.read_text(encoding="utf-8"))
            record["max_partition_bytes"] = 32768
            selection.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaises(ValueError):
                open_run("example")

    def test_copied_worker_bootstraps_without_project(self):
        """복사 허용 목록만으로 독립 프로세스가 worker를 불러와 추출한다."""
        request = self.tool._request("c" * 32)
        root = self.tool._stage(request)
        project = Path(__file__).resolve().parents[2]

        for name in WORKER_FILES:
            target = root / "tools" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        shutil.copyfile(project / ENTRY, root / "tools/worker.py")
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
            cwd=root, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertFalse((Path(request.output_directory) / "exports/test").exists())


    def test_mismatched_reply_is_rejected(self):
        """다른 작업의 성공 응답을 현재 내보내기 결과로 인정하지 않는다."""
        def mismatch(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / "worker_result.json"
            reply = json.loads(path.read_text(encoding="utf-8"))
            reply["request_id"] = "d" * 64
            path.write_text(json.dumps(reply), encoding="utf-8")
            return outcome

        result, output, _ = self.run_export(mismatch)
        self.assertNotEqual(result.status, "completed")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_saved_transport_is_rechecked(self):
        """완료 뒤에도 운송 증거가 바뀌면 상태 조회에서 계약 통과를 출력하지 않는다."""
        result, output, _ = self.run_export()
        self.assertEqual(result.status, "completed")
        path = self.tool._work_root(output.name) / "transport.json"
        outcome = json.loads(path.read_text(encoding="utf-8"))
        outcome["profile_cleanup"] = False
        path.write_text(json.dumps(outcome), encoding="utf-8")

        with self.assertRaises(ValueError):
            status(self.tool)

    def test_interactive_approval_accepts_only_exact_token(self):
        """대화형 승인도 범위 재확인을 거치며 다른 확인 문자열은 발급하지 않는다."""
        self.waiting()

        with patch("os.isatty", return_value=True), patch("secrets.token_hex", return_value="abcd1234"), patch("builtins.input", return_value="wrong"), patch("builtins.print"):
            self.assertEqual(approve(self.tool), {"code": "APPROVAL_DECLINED"})

        with patch("os.isatty", return_value=True), patch("secrets.token_hex", return_value="abcd1234"), patch("builtins.input", return_value="EXPORT-ABCD1234"), patch("builtins.print"):
            self.assertIn("approval_id", approve(self.tool))


if __name__ == "__main__":
    unittest.main()

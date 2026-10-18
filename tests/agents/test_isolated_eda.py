"""승인·요청 결합·부모 검증을 합성 EDA로 검사한다. OS 격리 transport는 모의 처리한다."""

import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.agents.eda_worker import execute_request
from src.agents.isolated_eda import SyntheticEdaTool, validate_report
from src.agents.isolated_eda_cli import _executor, cancel
from src.agents.state import LocalStateStore
from src.agents.synthetic_eda_data import CSV_BYTES, VERSION
from src.agents.worker_contract import WorkerLimits
from src.agents.isolated_eda_transport import WORKER_FILES


class IsolatedEdaTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.states = LocalStateStore(self.root / "demos" / "demo")
        self.states.create("synthetic_operator", policy_version=VERSION)
        self.tool = SyntheticEdaTool(self.states, WorkerLimits(60, 2048, 2))
        self.executor = _executor(self.tool)
        self.calls = 0

    def waiting(self):
        """환경 검사 후 EDA 승인 대기 상태를 만든다."""
        self.assertEqual(self.executor.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.executor.run_next().code, "APPROVAL_REQUIRED")

    def approve(self):
        return self.executor.approvals.record_confirmed_approval(self.executor.scope_for("run_eda")).approval_id

    def transport(self, request, root, signal):
        """실제 공통 EDA·요청 해석을 실행하되 AppContainer/ACL/Job 실행은 대체한다."""
        self.calls += 1
        reply = execute_request(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        return {
            "request_id": request.request_id, "profile_cleanup": True,
            "launch": {"status": "exited", "exit_code": 0, "resources": {
                "status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                "configured_process_limit": request.limits.processes,
            }},
        }

    def test_approval_then_completion_and_no_rerun(self):
        """승인 전에는 실행하지 않고 승인 1회 소비 후 검증·커밋한다."""
        with patch("src.agents.isolated_eda.run_isolated_eda", side_effect=self.transport):
            self.waiting()
            self.assertEqual(self.calls, 0)
            approval = self.approve()
            result = self.executor.run_next(approval_id=approval)
            self.assertEqual(result.status, "completed")
            self.assertEqual(self.calls, 1)
            self.assertEqual(self.executor.approvals.get(approval).status, "consumed")
            state = self.states.load()
            output = self.states.directory / state.artifact_refs["run_eda"]
            self.assertTrue((output / "AGENT_COMPLETE.json").is_file())
            self.assertTrue(validate_report(output / "eda"))
            self.executor.run_next()
            self.assertEqual(self.calls, 1)

    def test_limit_change_invalidates_approval(self):
        """승인 후 메모리·프로세스 수·시간 요청이 바뀌면 원래 승인으로 실행하지 않는다."""
        self.waiting()
        approval = self.approve()
        changed = SyntheticEdaTool(self.states, WorkerLimits(60, 2048, 3))

        with patch("src.agents.isolated_eda.run_isolated_eda") as transport:
            result = _executor(changed).run_next(approval_id=approval)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        transport.assert_not_called()

    def test_worker_rejects_non_fixture_input(self):
        """같은 이름의 CSV라도 내용이 다르면 실제 데이터로 처리하지 않는다."""
        request = self.tool._request("a" * 32)
        root = self.tool._stage(request)
        Path(request.input_files[0]).write_bytes(CSV_BYTES.replace(b"1,2,A,0", b"9,9,A,0"))
        reply = execute_request(request, root)
        self.assertEqual(reply.status, "blocked_input")
        self.assertFalse((Path(request.output_directory) / "eda").exists())

    def test_mismatched_worker_reply_rejected(self):
        """다른 작업 결과와 현재 요청의 성공 보고를 혼합하지 않는다."""
        self.waiting()

        def wrong_reply(request, root, signal):
            transport = self.transport(request, root, signal)
            path = Path(request.output_directory) / "worker_result.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["task_id"] = "f" * 32
            path.write_text(json.dumps(value), encoding="utf-8")
            return transport

        with patch("src.agents.isolated_eda.run_isolated_eda", side_effect=wrong_reply):
            result = self.executor.run_next(approval_id=self.approve())

        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        self.assertFalse(self.states.load().artifact_refs.get("run_eda"))

    def test_resource_failure_never_commits_partial_output(self):
        """시간/자원 초과는 실패로 남고 부분 결과에 완료 표식을 붙이지 않는다."""
        self.waiting()

        def failed(request, root, signal):
            (Path(request.output_directory) / "partial.txt").write_text("synthetic")
            return {"request_id": request.request_id, "profile_cleanup": True, "launch": {"status": "resource_exceeded"}}

        with patch("src.agents.isolated_eda.run_isolated_eda", side_effect=failed):
            result = self.executor.run_next(approval_id=self.approve())

        self.assertEqual(result.code, "WORKER_RESOURCE_EXCEEDED")
        self.assertEqual(result.status, "failed_runtime")
        # 정상 preflight 표식과 실패한 EDA 예약의 출력을 구분한다.
        task = self.executor.approvals.get_task(self.executor.scope_for("run_eda"))
        self.assertEqual(task["status"], "failed")
        self.assertFalse((self.executor._output(task) / "AGENT_COMPLETE.json").exists())

    def test_cancel_before_execution(self):
        """실행 전 취소는 worker 시작 없이 종료하고 미사용 승인을 폐기한다."""
        self.waiting()
        approval = self.approve()
        cancel(self.tool)

        with patch("src.agents.isolated_eda.run_isolated_eda") as transport:
            self.executor.run_next(approval_id=approval)

        self.assertEqual(self.states.load().status, "cancelled")
        self.assertEqual(self.executor.approvals.get(approval).status, "revoked")
        transport.assert_not_called()

    def test_cancel_during_execution_never_commits(self):
        """worker 실행 중 취소 신호가 도착하면 성공 보고보다 취소를 우선한다."""
        self.waiting()

        def interrupted(request, root, signal):
            result = self.transport(request, root, signal)
            (self.states.directory / "cancel.requested").write_bytes(b"cancel_synthetic_eda_v1\n")
            self.assertTrue(signal.is_set())
            return result

        with patch("src.agents.isolated_eda.run_isolated_eda", side_effect=interrupted):
            result = self.executor.run_next(approval_id=self.approve())

        self.assertEqual(result.status, "cancelled")
        self.assertFalse(self.states.load().artifact_refs.get("run_eda"))

    def test_missing_resource_evidence_rejected(self):
        """종료 코드가 0이어도 Job 정리·한도 확인이 없으면 성공으로 등록하지 않는다."""
        self.waiting()

        def incomplete(request, root, signal):
            return {
                "request_id": request.request_id, "profile_cleanup": True,
                "launch": {"status": "exited", "exit_code": 0},
            }

        with patch("src.agents.isolated_eda.run_isolated_eda", side_effect=incomplete):
            result = self.executor.run_next(approval_id=self.approve())

        self.assertEqual(result.code, "WORKER_FAILED")
        self.assertFalse(self.states.load().artifact_refs.get("run_eda"))

    def test_copied_package_bootstraps_without_project(self):
        """프로젝트 import 캐시 없이 복사 허용 목록만으로 요청·EDA 코어를 불러온다."""
        project = Path(__file__).resolve().parents[2]
        tools = self.root / "copied_tools"

        for relative in WORKER_FILES:
            target = tools / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        # 패키지 의존성 검사이며 AppContainer/ACL/Job 검증은 아니다.
        script = (
            "import sys; sys.path[:0] = sys.argv[1:]; "
            "from src.agents.worker_contract import WorkerRequest; "
            "from src.agents.synthetic_eda_data import fixture_input; "
            "from src.preprocessing.eda import collect_eda; "
            "collect_eda(fixture_input())"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script,
             str(tools), str(Path(sys.prefix) / "Lib" / "site-packages")],
            cwd=self.root, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_parent_detects_csv_tampering(self):
        """SUCCESS.json이 있어도 통계 CSV 변경은 부모 재계산으로 거부한다."""
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        self.transport(request, root, self.tool.cancel)
        report = Path(request.output_directory) / "eda"
        self.assertTrue(validate_report(report))
        (report / "missing.csv").write_text("feature,missing_count\nx,999\n", encoding="utf-8")
        self.assertFalse(validate_report(report))


if __name__ == "__main__":
    unittest.main()

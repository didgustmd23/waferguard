"""참조 모듈의 승인·격리 파일 계약·부모 검증을 합성 입력으로 확인한다."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.agents.eda_input_contract import bind_eda_input
from src.agents.input_snapshot import create_snapshot
from src.agents.reviewed_module import ReviewedModuleTool
from src.agents.reviewed_module_cli import executor, approve, cancel, create_demo, open_run, status
from src.agents.reviewed_module_contract import POLICY_VERSION, demo_profile
from src.agents.reviewed_module_results import read_result
from src.agents.reviewed_module_transport import WORKER_FILES, ENTRY
from src.agents.reviewed_module_worker import execute_request
from src.agents.state import LocalStateStore
from src.agents.synthetic_eda_data import CSV_BYTES
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.quality import QualityFinding, QualityReport


class ReviewedModuleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "source.csv"
        source.write_bytes(CSV_BYTES)
        snapshot = create_snapshot(source, self.root / "snapshot", max_input_bytes=4096)
        self.contract = bind_eda_input(self.root / "snapshot", snapshot, demo_profile(), WorkerLimits(60, 2048, 2))
        self.states = LocalStateStore(self.root / "runs/example")
        self.states.create("synthetic_operator", policy_version=POLICY_VERSION)
        self.tool = ReviewedModuleTool(self.states, self.contract)
        self.manager = executor(self.tool)
        self.calls = 0

    def waiting(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

    def transport(self, request, root, signal):
        """네이티브 운송만 모의하며 worker·결과·기존 커밋 로직은 그대로 실행한다."""
        self.calls += 1
        reply = execute_request(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes}}}
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
        return outcome

    def run_module(self, transport=None):
        self.waiting()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda"))

        with patch("src.agents.reviewed_module.run_reviewed_worker", side_effect=transport or self.transport):
            result = self.manager.run_next(approval_id=approval.approval_id)

        task = self.manager.approvals.get_task(self.manager.scope_for("run_eda"))
        return result, self.manager._output(task)

    def test_approval_completion_and_no_rerun(self):
        """승인 전 시작 금지·완료 커밋·재실행 방지·변경 없는 합성 입력을 확인한다."""
        with patch("src.agents.reviewed_module.run_reviewed_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        # 이미 사전 검사를 통과했으므로 승인만 발급해 실행한다.
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda"))

        with patch("src.agents.reviewed_module.run_reviewed_worker", side_effect=self.transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.assertEqual(status(self.tool)["module_status"], "passed")
        self.assertEqual((self.root / "snapshot/input.bin").read_bytes(), CSV_BYTES)

        with patch("src.agents.reviewed_module.run_reviewed_worker") as launch:
            self.manager.run_next()

        launch.assert_not_called()

    def test_settings_change_invalidates_approval(self):
        """승인 후 설정을 바꾸면 이전 승인으로 실행할 수 없다."""
        self.waiting()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda"))
        changed = ReviewedModuleTool(self.states, self.contract, False)

        with patch("src.agents.reviewed_module.run_reviewed_worker") as launch:
            result = executor(changed).run_next(approval_id=approval.approval_id)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        launch.assert_not_called()
        self.assertNotEqual(changed.plan.plan_id, self.tool.plan.plan_id)

    def test_exclude_complete_option(self):
        """완전 관측 행 제외 설정도 고정 합성 정답과 일치한다."""
        self.tool = ReviewedModuleTool(self.states, self.contract, False)
        self.manager = executor(self.tool)
        result, output = self.run_module()
        self.assertEqual(result.status, "completed")
        self.assertEqual((output / "pipeline/row_missing/missing_row_histogram.csv").read_text(encoding="utf-8").splitlines(),
                         ["missing_feature_count,row_count", "1,1"])

    def test_quality_review_stops_before_module_import(self):
        """품질 검토 필요 시 확장 import·계산·완료 커밋이 없다."""
        review = QualityReport("review_required", (QualityFinding("HIGH_MISSING_RATIO", "review", "y", 5),))

        with patch("src.preprocessing.quality.inspect_quality", return_value=review), patch("src.agents.module_catalogue.build_reviewed_registry", side_effect=AssertionError("검토 뒤 실행")):
            result, output = self.run_module()

        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())
        self.assertFalse((output / "pipeline/row_missing").exists())

    def test_nonfixture_and_staged_changes_rejected(self):
        """다른 CSV 계약과 준비 후 변경된 입력을 거부한다."""
        source = self.root / "other.csv"
        source.write_bytes(CSV_BYTES.replace(b"1,2,A", b"9,2,A"))
        snapshot = create_snapshot(source, self.root / "other", max_input_bytes=4096)
        other = bind_eda_input(self.root / "other", snapshot, demo_profile(), self.contract.limits)

        with self.assertRaises(ValueError):
            ReviewedModuleTool(self.states, other)

        request = self.tool._request("a" * 32)
        root = self.tool._stage(request)
        (root / "input/snapshot/input.bin").write_bytes(b"changed")
        self.assertEqual(execute_request(request, root).status, "blocked_input")

    def test_parent_rejects_tampering_without_calculation(self):
        """부모는 원본 재파싱·확장 계산 없이 정답 범위와 파일 추가를 거부한다."""
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        self.transport(request, root, self.tool.cancel)
        directory = Path(request.output_directory) / "pipeline"

        with patch("pandas.read_csv", side_effect=AssertionError("부모 원본 파싱")), patch("src.agents.module_catalogue.build_reviewed_registry", side_effect=AssertionError("부모 확장 import")):
            self.assertIsNotNone(read_result(directory, self.tool.config_id, True))

        (directory / "row_missing/missing_row_histogram.csv").write_text("missing_feature_count,row_count\n0,999\n", encoding="utf-8")
        self.assertIsNone(read_result(directory, self.tool.config_id, True))

    def test_missing_resource_evidence_rejected(self):
        """worker 성공에도 Job 정리 증거가 없으면 완료하지 않는다."""
        def missing(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["resources"] = {}
            return outcome

        result, output = self.run_module(missing)
        self.assertEqual(result.code, "WORKER_FAILED")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_resource_exceeded_and_cancel_never_commit(self):
        """자원 초과·실행 중 취소는 부분 결과보다 우선한다."""
        def exceeded(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["status"] = "resource_exceeded"
            return outcome

        result, output = self.run_module(exceeded)
        self.assertEqual(result.code, "WORKER_RESOURCE_EXCEEDED")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_cancel_during_execution(self):
        """성공 파일이 있어도 실행 중 취소는 완료로 등록하지 않는다."""
        def cancelled(request, root, signal):
            outcome = self.transport(request, root, signal)
            cancel(self.tool)
            return outcome

        result, output = self.run_module(cancelled)
        self.assertNotEqual(result.status, "completed")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_cli_create_and_approval(self):
        """CLI 생성은 worker 시작 없이 계약을 보존하고 대화형 확인만 승인한다."""
        directory = self.root / "runs/cli_demo"

        with patch("src.agents.reviewed_module_cli._directory", return_value=directory):
            self.assertFalse(create_demo("cli_demo", self.contract.limits)["worker_started"])
            tool = open_run("cli_demo")

        manager = executor(tool)
        manager.run_next()
        manager.run_next()

        with patch("os.isatty", return_value=False), self.assertRaises(ValueError):
            approve(tool)

        with patch("os.isatty", return_value=True), patch("secrets.token_hex", return_value="abcd"), patch("builtins.input", return_value="MODULE-ABCD"), patch("builtins.print"):
            approval = approve(tool)

        self.assertEqual(manager.approvals.get(approval["approval_id"]).status, "pending")

    def test_copied_package_without_project_imports(self):
        """복사 목록·명세만으로 새 프로세스에서 worker를 실행할 수 있다."""
        request = self.tool._request("c" * 32)
        root = self.tool._stage(request)
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = root / "tools" / ("worker.py" if relative == ENTRY else relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        code = (
            "import sys, os; from pathlib import Path; prefix = Path(sys.prefix); "
            "handles = [os.add_dll_directory(str(p)) for p in (prefix, prefix/'DLLs', prefix/'Library/bin') if p.is_dir()] if os.name == 'nt' else []; "
            f"sys.path[:0] = [{str(root / 'tools')!r}, str(prefix/'Lib/site-packages')]; "
            "from worker import execute_request; from src.agents.worker_contract import WorkerRequest; "
            f"request = WorkerRequest.from_json(Path({str(root / 'input/request.json')!r}).read_text(encoding='utf-8')); "
            f"result = execute_request(request, Path({str(root)!r})); assert result.status == 'succeeded', result"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-c", code], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

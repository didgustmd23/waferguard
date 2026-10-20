"""snapshot EDA 연결을 합성 입력으로 검증한다. OS transport는 모의 처리한다."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.agents.runtime_fixtures import run_supported_preflight

from src.agents.eda_input_contract import bind_eda_input
from src.agents.eda_result_validation import validate_eda_report
from src.agents.input_snapshot import create_snapshot
from src.agents.snapshot_eda import POLICY_VERSION, SnapshotEdaTool
from src.agents.snapshot_eda_cli import executor, cancel
from src.agents import snapshot_eda_cli as cli
from src.agents.snapshot_eda_transport import WORKER_FILES
from src.agents.snapshot_eda_worker import execute_request, _verify_staged_snapshot
from src.agents.state import LocalStateStore
from src.agents.synthetic_eda_data import CSV_BYTES
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile


class SnapshotEdaTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "source.csv"
        source.write_bytes(CSV_BYTES)
        self.snapshot_dir = self.root / "snapshot"
        snapshot = create_snapshot(source, self.snapshot_dir, max_input_bytes=1024)
        self.profile = DatasetProfile(
            "example", "tabular_csv_v1", "eda_review_v1",
            ColumnRoles(numeric=("x", "y"), categorical=("category",), target="target"),
            max_plot_features=3,
        )
        self.contract = bind_eda_input(self.snapshot_dir, snapshot, self.profile, WorkerLimits(60, 2048, 2))
        self.states = LocalStateStore(self.root / "runs" / "example")
        self.states.create("synthetic_operator", policy_version=POLICY_VERSION)
        self.tool = SnapshotEdaTool(self.states, self.contract)
        self.manager = executor(self.tool)
        self.calls = 0

    def waiting(self):
        self.assertEqual(run_supported_preflight(self.manager, cli).code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

    def approval(self):
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda")).approval_id

    def transport(self, request, root, signal):
        """실제 worker 코어는 실행하되 AppContainer·Job·ACL은 모의 처리한다."""
        self.calls += 1
        result = execute_request(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(result.to_json(), encoding="utf-8")
        return {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if result.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                "configured_process_limit": request.limits.processes},
        }}

    def test_approval_worker_validation_commit_no_rerun(self):
        """승인 전 실행 금지·1회 소비·실제 EDA·부모 검증·커밋·중복 방지를 확인한다."""
        with patch("src.agents.snapshot_eda.run_snapshot_eda", side_effect=self.transport):
            self.waiting()
            self.assertEqual(self.calls, 0)
            approval = self.approval()
            result = self.manager.run_next(approval_id=approval)
            self.assertEqual(result.status, "completed")
            self.assertEqual(self.manager.approvals.get(approval).status, "consumed")
            self.manager.run_next()
            self.assertEqual(self.calls, 1)

    def test_profile_change_cannot_use_previous_plan(self):
        """프로파일 변경 후 기존 계획의 승인을 재사용할 수 없다."""
        self.waiting()
        approval = self.approval()
        changed = replace(self.contract, profile_json=json.dumps(replace(self.profile, max_plot_features=2).to_dict()))
        manager = executor(SnapshotEdaTool(self.states, changed))

        with patch("src.agents.snapshot_eda.run_snapshot_eda") as transport:
            result = manager.run_next(approval_id=approval)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        transport.assert_not_called()

    def test_worker_blocks_staged_input_and_settings_tamper(self):
        """작업용 복사본 변경·요청과 다른 설정은 CSV 파싱 전에 거부한다."""
        for index, target in enumerate(("snapshot/input.bin", "eda.json")):
            request = self.tool._request(str(index) * 32)
            root = self.tool._stage(request)
            (root / "input" / target).write_bytes(b"changed")

            with patch("pandas.read_csv", side_effect=AssertionError("읽기 금지")):
                result = execute_request(request, root)

            self.assertEqual(result.status, "blocked_input")

    def test_access_denial_diagnostic_has_no_raw_error(self):
        """파일 접근 거부는 고정 단계 코드만 남기고 경로·원문 예외를 기록하지 않는다."""
        request = self.tool._request("d" * 32)
        root = self.tool._stage(request)

        with patch("src.agents.snapshot_eda_worker._verify_staged_snapshot", side_effect=PermissionError("sensitive_source_path")):
            reply = execute_request(request, root)

        self.assertEqual(reply.status, "blocked_input")
        text = (root / "scratch" / "task" / "diagnostic.json").read_text()
        self.assertEqual(json.loads(text), {"stage": "SNAPSHOT", "code": "ACCESS_DENIED"})
        self.assertNotIn("sensitive_source_path", text)

    def test_worker_never_stats_ungranted_ancestors(self):
        """입력 밖의 조상 조회가 거부돼도 worker는 허용된 파일의 내용 검증을 수행한다."""
        request = self.tool._request("e" * 32)
        root = self.tool._stage(request)
        original = Path.lstat
        denied = {root, *root.parents}

        def restricted(path, *args, **kwargs):
            if path in denied:
                raise PermissionError("합성 조상 접근 거부")

            return original(path, *args, **kwargs)

        with patch.object(Path, "lstat", restricted):
            self.assertTrue(_verify_staged_snapshot(root, self.contract.snapshot))

            # 부모용 전체 경로 검사는 생략되거나 완화되지 않는다.
            with self.assertRaises(PermissionError):
                from src.agents.input_snapshot import verify_snapshot
                verify_snapshot(root / "input" / "snapshot", self.contract.snapshot)

    def test_worker_scope_still_rejects_hardlinks(self):
        """검사 범위를 줄여도 입력 하드 링크는 계속 거부한다."""
        import os

        request = self.tool._request("f" * 32)
        root = self.tool._stage(request)
        directory = root / "input" / "snapshot"
        self.assertTrue(_verify_staged_snapshot(root, self.contract.snapshot))
        os.link(directory / "input.bin", self.root / "hardlink.bin")
        self.assertFalse(_verify_staged_snapshot(root, self.contract.snapshot))

    def test_clean_exit_two_preserves_input_block_reason(self):
        """입력 차단 종료를 일반 실행 실패로 뭉뚱그리거나 성공으로 승격하지 않는다."""
        self.waiting()

        def blocked(request, root, signal):
            (root / "input" / "snapshot" / "input.bin").write_bytes(b"changed")
            return self.transport(request, root, signal)

        with patch("src.agents.snapshot_eda.run_snapshot_eda", side_effect=blocked):
            result = self.manager.run_next(approval_id=self.approval())

        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        self.assertNotIn("run_eda", self.states.load().artifact_refs)

    def test_failed_job_does_not_commit(self):
        """자원 초과는 부분 결과를 완료로 등록하지 않는다."""
        self.waiting()

        def failed(request, root, signal):
            return {"request_id": request.request_id, "profile_cleanup": True, "launch": {"status": "resource_exceeded"}}

        with patch("src.agents.snapshot_eda.run_snapshot_eda", side_effect=failed):
            result = self.manager.run_next(approval_id=self.approval())

        self.assertEqual(result.code, "WORKER_RESOURCE_EXCEEDED")
        self.assertNotIn("run_eda", self.states.load().artifact_refs)

    def test_cancel_before_launch_revokes_approval(self):
        """실행 전 취소는 승인 폐기 후 프로세스 생성 없이 종료한다."""
        self.waiting()
        approval = self.approval()
        cancel(self.tool)

        with patch("src.agents.snapshot_eda.run_snapshot_eda") as transport:
            self.manager.run_next(approval_id=approval)

        transport.assert_not_called()
        self.assertEqual(self.manager.approvals.get(approval).status, "revoked")

    def test_parent_checks_results_without_loading_source(self):
        """부모는 원본 CSV를 재파싱하지 않으며 결과의 범위·파일 목록을 검사한다."""
        request = self.tool._request("a" * 32)
        root = self.tool._stage(request)
        self.transport(request, root, self.tool.cancel)
        report = Path(request.output_directory) / "eda"

        with patch("pandas.read_csv", side_effect=AssertionError("부모 원본 파싱 금지")):
            self.assertTrue(validate_eda_report(report, self.profile))

        summary = json.loads((report / "summary.json").read_text())
        summary["row_count"] = -1
        (report / "summary.json").write_text(json.dumps(summary))
        self.assertFalse(validate_eda_report(report, self.profile))

    def test_extra_pickle_result_is_rejected(self):
        """worker 출력에 예상하지 않은 pickle을 추가하면 열지 않고 거부한다."""
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        self.transport(request, root, self.tool.cancel)
        report = Path(request.output_directory) / "eda"
        (report / "extra.pkl").write_bytes(b"not executed")

        with patch("pickle.load", side_effect=AssertionError("역직렬화 금지")):
            self.assertFalse(validate_eda_report(report, self.profile))

    def test_cli_selection_and_interactive_approval(self):
        """CLI 선택 저장·동일 계정 복원·명시 승인 발급을 검사한다. OS 실행은 하지 않는다."""
        profile_path = self.root / "profile.json"
        profile_path.write_text(json.dumps(self.profile.to_dict()), encoding="utf-8")
        directory = self.root / "runs" / "cli_example"

        with patch.object(cli, "_directory", return_value=directory), patch.object(cli, "_current_actor", return_value="synthetic_operator"):
            with self.assertRaises(ValueError):
                cli.create("cli_example", self.snapshot_dir, profile_path, self.contract.limits)

            cli.create("cli_example", self.snapshot_dir, profile_path, self.contract.limits, confirmed=True)
            tool = cli.open_run("cli_example")
            manager = cli.executor(tool)
            self.assertEqual(run_supported_preflight(manager, cli).code, "STEP_COMPLETED")
            self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")

            with patch.object(cli.os, "isatty", return_value=False), self.assertRaises(ValueError):
                cli.approve(tool)

            with patch.object(cli.os, "isatty", return_value=True), patch.object(cli.secrets, "token_hex", return_value="11223344"), patch("builtins.input", return_value="EDA-11223344"), patch("builtins.print"):
                result = cli.approve(tool)

            self.assertEqual(manager.approvals.get(result["approval_id"]).status, "pending")

    def test_cleanup_failure_never_commits(self):
        """worker 결과가 있어도 프로필 정리를 확인하지 못하면 완료로 등록하지 않는다."""
        self.waiting()

        def incomplete(request, root, signal):
            result = self.transport(request, root, signal)
            result["profile_cleanup"] = False
            return result

        with patch("src.agents.snapshot_eda.run_snapshot_eda", side_effect=incomplete):
            result = self.manager.run_next(approval_id=self.approval())

        self.assertEqual(result.code, "WORKER_FAILED")
        self.assertNotIn("run_eda", self.states.load().artifact_refs)

    def test_different_schema_and_categorical_only_input(self):
        """고정 fixture 이외의 컬럼 이름과 범주형 전용 입력도 같은 코어로 처리한다."""
        for index, (payload, roles) in enumerate((
            (b"measurement,grade\n1,ok\n2,reject\n3,ok\n", ColumnRoles(numeric=("measurement",), target="grade")),
            (b"kind\nA\nB\nA\n", ColumnRoles(categorical=("kind",))),
        )):
            source = self.root / f"other_{index}.csv"
            source.write_bytes(payload)
            directory = self.root / f"other_snapshot_{index}"
            snapshot = create_snapshot(source, directory, max_input_bytes=1024)
            profile = replace(self.profile, roles=roles)
            contract = bind_eda_input(directory, snapshot, profile, self.contract.limits)
            tool = SnapshotEdaTool(self.states, contract)
            request = tool._request(str(index + 5) * 32)
            root = tool._stage(request)
            reply = execute_request(request, root)
            self.assertEqual(reply.status, "succeeded")
            self.assertTrue(validate_eda_report(Path(request.output_directory) / "eda", profile))

    def test_cancel_during_worker_does_not_commit(self):
        """worker 완료와 취소가 겹치면 취소 신호를 우선하고 결과를 등록하지 않는다."""
        self.waiting()

        def cancelled(request, root, signal):
            result = self.transport(request, root, signal)
            (self.states.directory / "cancel.requested").write_bytes(b"cancel_synthetic_eda_v1\n")
            return result

        with patch("src.agents.snapshot_eda.run_snapshot_eda", side_effect=cancelled):
            result = self.manager.run_next(approval_id=self.approval())

        self.assertEqual(result.status, "cancelled")
        self.assertNotIn("run_eda", self.states.load().artifact_refs)

    def test_copied_dependencies_import_without_project(self):
        """새 worker의 명시 복사 목록만으로 프로파일·snapshot 코어를 import한다."""
        project = Path(__file__).resolve().parents[2]
        tools = self.root / "copied_tools"

        for name in WORKER_FILES:
            target = tools / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        script = (
            "import sys; sys.path[:0]=sys.argv[1:]; "
            "from src.preprocessing.profiles import DatasetProfile; "
            "from src.agents.input_snapshot import verify_snapshot; "
            "from src.agents.worker_contract import WorkerRequest"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(tools), str(Path(sys.prefix) / "Lib" / "site-packages")],
            cwd=self.root, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()

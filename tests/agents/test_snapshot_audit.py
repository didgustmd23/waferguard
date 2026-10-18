"""품질 검사 격리 연결의 합성 검증. OS transport만 모의 처리한다."""

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
from src.agents.quality_result_validation import read_quality_report
from src.agents.snapshot_audit import POLICY_VERSION, SnapshotAuditTool
from src.agents.snapshot_audit_cli import executor, cancel
from src.agents.snapshot_audit_worker import execute_request
from src.agents.snapshot_audit_transport import WORKER_FILES
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules


class SnapshotAuditTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = DatasetProfile(
            "audit_example", "tabular_csv_v1", "eda_review_v1",
            ColumnRoles(numeric=("x",), target="target"),
            quality_rules=QualityRules(allowed_labels=(0, 1), target_missing="block"),
        )
        self.calls = 0

    def setup_run(self, payload=b"x,target\n1,0\n2,1\n3,0\n", profile=None):
        source = self.root / "source.csv"
        source.write_bytes(payload)
        self.directory = self.root / "snapshot"
        snapshot = create_snapshot(source, self.directory, max_input_bytes=4096)
        contract = bind_eda_input(self.directory, snapshot, profile or self.profile, WorkerLimits(60, 2048, 2))
        self.states = LocalStateStore(self.root / "runs" / "example")
        self.states.create("synthetic_operator", policy_version=POLICY_VERSION)
        self.tool = SnapshotAuditTool(self.states, contract)
        self.manager = executor(self.tool)

    def waiting(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

    def approval(self):
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("audit")).approval_id

    def transport(self, request, root, signal):
        """실제 품질 검사 코어·보고서 쓰기는 실행하되 OS 격리를 대체한다."""
        self.calls += 1
        reply = execute_request(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        return {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                "configured_process_limit": request.limits.processes},
        }}

    def run_audit(self):
        self.waiting()

        with patch("src.agents.snapshot_audit.run_snapshot_audit", side_effect=self.transport):
            result = self.manager.run_next(approval_id=self.approval())

        task = self.manager.approvals.get_task(self.manager.scope_for("audit"))
        return result, self.manager._output(task), task

    def test_passed_approval_commit_and_no_rerun(self):
        """승인 전 실행 금지·통과 후 커밋·중복 실행 금지를 확인한다."""
        self.setup_run()
        result, output, task = self.run_audit()
        self.assertEqual(result.status, "completed")
        self.assertEqual(task["status"], "committed")
        self.assertEqual(self.calls, 1)
        self.assertTrue((output / "AGENT_COMPLETE.json").is_file())
        self.assertFalse((output / "eda").exists())

        with patch("src.agents.snapshot_audit.run_snapshot_audit") as transport:
            self.manager.run_next()

        transport.assert_not_called()

    def test_review_keeps_report_without_completion(self):
        """상수 특징은 검토 필요로 중단하며 데이터를 고치거나 완료 커밋하지 않는다."""
        self.setup_run(b"x,target\n1,0\n1,1\n1,0\n")
        before = (self.directory / "input.bin").read_bytes()
        result, output, task = self.run_audit()
        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertEqual(task["status"], "failed")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())
        self.assertEqual(read_quality_report(output / "quality", self.profile)["status"], "review_required")
        self.assertEqual((self.directory / "input.bin").read_bytes(), before)

    def test_unknown_label_blocks_without_commit(self):
        """미지 라벨은 차단 보고서를 남기며 자동 재라벨링하지 않는다."""
        self.setup_run(b"x,target\n1,99\n2,1\n3,0\n")
        result, output, task = self.run_audit()
        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())
        report = read_quality_report(output / "quality", self.profile)
        self.assertEqual(report["status"], "blocked_input")
        self.assertEqual(report["findings"][0]["code"], "UNKNOWN_TARGET_LABEL")

    def test_bad_numeric_is_quality_finding(self):
        """로더 단계에서 수치 역할을 미리 강제하지 않아 품질 오류 근거를 보존한다."""
        self.setup_run(b"x,target\nbad,0\n2,1\n3,0\n")
        result, output, task = self.run_audit()
        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        report = read_quality_report(output / "quality", self.profile)
        self.assertIn("INVALID_NUMERIC_TYPE", [item["code"] for item in report["findings"]])

    def test_missing_column_is_reported(self):
        """필수 컬럼 누락을 품질 보고서에 남긴다."""
        self.setup_run(b"other,target\n1,0\n2,1\n")
        result, output, task = self.run_audit()
        report = read_quality_report(output / "quality", self.profile)
        self.assertEqual(report["findings"][0]["code"], "REQUIRED_COLUMN_MISSING")

    def test_tampered_input_and_request_rejected(self):
        """작업용 입력 변경은 품질 판정이 아닌 입력 계약 오류로 차단한다."""
        self.setup_run()
        request = self.tool._request("a" * 32)
        root = self.tool._stage(request)
        (root / "input" / "snapshot" / "input.bin").write_bytes(b"changed")
        self.assertEqual(execute_request(request, root).status, "blocked_input")
        self.assertFalse((Path(request.output_directory) / "quality").exists())

    def test_forged_pass_is_rejected_without_source_parse(self):
        """검토 결과를 passed로 바꾼 보고서와 추가 파일은 거부한다."""
        self.setup_run(b"x,target\n1,0\n1,1\n")
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        self.transport(request, root, self.tool.cancel)
        directory = Path(request.output_directory) / "quality"

        with patch("pandas.read_csv", side_effect=AssertionError("부모 원본 파싱 금지")):
            self.assertIsNotNone(read_quality_report(directory, self.profile))

        path = directory / "quality.json"
        report = json.loads(path.read_text())
        report["status"] = "passed"
        path.write_text(json.dumps(report))
        self.assertIsNone(read_quality_report(directory, self.profile))
        (directory / "extra.pkl").write_bytes(b"not executed")
        self.assertIsNone(read_quality_report(directory, self.profile))

    def test_policy_change_needs_new_plan(self):
        """품질 통지 수준 변경도 기존 승인의 재사용을 차단한다."""
        self.setup_run()
        self.waiting()
        approval = self.approval()
        profile = replace(self.profile, quality_rules=replace(self.profile.quality_rules, constant_feature="block"))
        contract = replace(self.tool.contract, profile_json=json.dumps(profile.to_dict()))
        manager = executor(SnapshotAuditTool(self.states, contract))

        with patch("src.agents.snapshot_audit.run_snapshot_audit") as transport:
            result = manager.run_next(approval_id=approval)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        transport.assert_not_called()

    def test_identifier_and_timestamp_findings(self):
        """ID 중복과 명시한 날짜 형식 위반을 비파괴적으로 검사한다."""
        profile = replace(
            self.profile, roles=ColumnRoles(numeric=("x",), target="target", identifiers=("id",), timestamp="date"),
            quality_rules=replace(self.profile.quality_rules, duplicate_id="block", timestamp_format="%Y-%m-%d"),
        )
        self.setup_run(b"x,target,id,date\n1,0,same,2026-01-01\n2,1,same,bad\n", profile)
        result, output, task = self.run_audit()
        report = read_quality_report(output / "quality", profile)
        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        self.assertEqual({item["code"] for item in report["findings"]}, {"DUPLICATE_IDENTIFIER", "INVALID_TIMESTAMP"})

    def test_all_missing_features_require_review(self):
        """전체 결측과 높은 결측률을 진단하되 대치·제거하지 않는다."""
        self.setup_run(b"x,target\n,0\n,1\n,0\n")
        result, output, task = self.run_audit()
        report = read_quality_report(output / "quality", self.profile)
        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertEqual({item["code"] for item in report["findings"]}, {"HIGH_MISSING_RATIO", "ALL_MISSING_FEATURE"})

    def test_resource_failure_never_commits(self):
        """자원 초과는 작업 결과가 존재해도 완료로 등록하지 않는다."""
        self.setup_run()
        self.waiting()

        def exceeded(request, root, signal):
            value = self.transport(request, root, signal)
            value["launch"]["status"] = "resource_exceeded"
            value["launch"]["resources"]["status"] = "resource_exceeded"
            return value

        with patch("src.agents.snapshot_audit.run_snapshot_audit", side_effect=exceeded):
            result = self.manager.run_next(approval_id=self.approval())

        self.assertEqual(result.code, "WORKER_RESOURCE_EXCEEDED")
        self.assertNotIn("audit", self.states.load().artifact_refs)

    def test_copied_worker_runs_without_project_imports(self):
        """고정 복사 목록만으로 별도 프로세스에서 실제 품질 코어를 실행한다."""
        self.setup_run()
        request = self.tool._request("c" * 32)
        root = self.tool._stage(request)
        project = Path(__file__).resolve().parents[2]
        tools = root / "tools"

        for relative in WORKER_FILES:
            target = tools / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        shutil.copyfile(project / "src/agents/snapshot_audit_worker.py", tools / "worker.py")
        script = (
            "import sys;sys.path[:0]=sys.argv[1:3];from pathlib import Path;"
            "from src.agents.worker_contract import WorkerRequest;import worker;"
            "root=Path(sys.argv[3]);request=WorkerRequest.from_json((root/'input/request.json').read_text());"
            "assert worker.execute_request(request,root).status=='succeeded'"
        )
        # 패키지 독립성 검사이며 AppContainer·ACL·Job 검증은 아니다.
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(tools), str(Path(sys.prefix) / "Lib/site-packages"), str(root)],
            cwd=self.root, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_cancel_prevents_worker(self):
        """취소는 미사용 승인을 폐기하고 worker 실행을 막는다."""
        self.setup_run()
        self.waiting()
        approval = self.approval()
        cancel(self.tool)

        with patch("src.agents.snapshot_audit.run_snapshot_audit") as transport:
            self.manager.run_next(approval_id=approval)

        transport.assert_not_called()
        self.assertEqual(self.manager.approvals.get(approval).status, "revoked")


if __name__ == "__main__":
    unittest.main()

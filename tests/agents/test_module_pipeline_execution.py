"""승인·모듈 worker·부모 검증 연결을 합성 입력으로 검사한다. OS transport만 모의 처리한다."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.agents import module_pipeline_cli as cli
from tests.agents.runtime_fixtures import run_supported_preflight

from src.agents.eda_input_contract import bind_eda_input
from src.agents.input_snapshot import create_snapshot
from src.agents.module_pipeline import ModulePipelineTool
from src.agents.module_pipeline_cli import executor, cancel, approve, create_demo, open_run
from src.agents.module_pipeline_contract import POLICY_VERSION
from src.agents.module_pipeline_results import read_pipeline_result
from src.agents.module_pipeline_transport import WORKER_FILES, ENTRY
from src.agents.module_pipeline_worker import execute_request
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules


class ModulePipelineExecutionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = 0
        self.profile = DatasetProfile(
            "pipeline_example", "tabular_csv_v1", "eda_review_v1",
            ColumnRoles(numeric=("x", "y"), target="target"),
            quality_rules=QualityRules(allowed_labels=(0, 1)),
        )
        self.config = {
            "schema_version": "module_pipeline_v1", "pipeline_id": "example",
            "input": {"contract_id": "tabular_data", "contract_version": 1, "partition": "unsplit"},
            "steps": [
                {"step_id": "quality", "module_id": "tabular_quality", "version": "1.0.0", "inputs": {"data": "source.data"}},
                {"step_id": "missing", "module_id": "eda_missing", "version": "1.0.0", "inputs": {"data": "source.data"}, "depends_on": ["quality"]},
                {"step_id": "correlations", "module_id": "eda_correlations", "version": "1.0.0", "inputs": {"data": "source.data"}},
            ],
        }

    def setup_run(self, payload=b"x,y,target\n1,2,0\n2,4,1\n3,6,0\n"):
        source = self.root / "source.csv"
        source.write_bytes(payload)
        self.snapshot = self.root / "snapshot"
        metadata = create_snapshot(source, self.snapshot, max_input_bytes=4096)
        contract = bind_eda_input(self.snapshot, metadata, self.profile, WorkerLimits(60, 2048, 2))
        self.states = LocalStateStore(self.root / "runs" / "example")
        self.states.create("synthetic_operator", policy_version=POLICY_VERSION)
        self.tool = ModulePipelineTool(self.states, contract, self.config)
        self.manager = executor(self.tool)

    def waiting(self):
        self.assertEqual(run_supported_preflight(self.manager, cli).code, "STEP_COMPLETED")
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

    def approval(self):
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda")).approval_id

    def transport(self, request, root, signal):
        """실제 worker·계산·파일 저장은 실행하지만 AppContainer 실행은 대체한다."""
        self.calls += 1
        reply = execute_request(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")

        return {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes},
        }}

    def run_pipeline(self, transport=None):
        self.waiting()

        with patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=transport or self.transport):
            result = self.manager.run_next(approval_id=self.approval())

        task = self.manager.approvals.get_task(self.manager.scope_for("run_eda"))

        return result, self.manager._output(task), task

    def test_approval_execution_commit_and_no_rerun(self):
        """승인 전에는 worker를 시작하지 않고 전체 성공만 한 번 커밋한다."""
        self.setup_run()
        before = (self.snapshot / "input.bin").read_bytes()
        result, output, task = self.run_pipeline()
        self.assertEqual(result.status, "completed")
        self.assertEqual(task["status"], "committed")
        self.assertTrue((output / "AGENT_COMPLETE.json").exists())
        self.assertEqual(self.calls, 1)
        self.assertEqual((self.snapshot / "input.bin").read_bytes(), before)
        self.assertFalse((output / "pipeline/missing/pearson.csv").exists())
        self.assertFalse((output / "pipeline/missing/missing.png").exists())

        with patch("src.agents.module_pipeline.run_pipeline_worker") as launch:
            self.manager.run_next()

        launch.assert_not_called()

    def test_review_stops_before_following_analysis(self):
        """상수 특징은 품질 보고서만 보존하고 후속 계산·완료 기록을 하지 않는다."""
        self.setup_run(b"x,y,target\n1,2,0\n1,4,1\n1,6,0\n")

        with patch("src.preprocessing.eda.analyze_missing", side_effect=AssertionError("검토 뒤 분석 실행")):
            result, output, task = self.run_pipeline()

        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertEqual(task["status"], "failed")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())
        self.assertFalse((output / "pipeline/missing").exists())
        report = read_pipeline_result(output / "pipeline", self.tool.module_plan, self.profile)
        self.assertEqual(report["completed_steps"], ["quality"])

    def test_profile_label_policy_is_inherited(self):
        """모듈에 별도 설정이 없어도 프로파일 허용 라벨 규칙을 누락하지 않는다."""
        self.setup_run(b"x,y,target\n1,2,99\n2,4,1\n3,6,0\n")
        result, output, _ = self.run_pipeline()
        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        self.assertFalse((output / "pipeline/missing").exists())
        self.assertIn("UNKNOWN_TARGET_LABEL", (output / "pipeline/quality/quality.json").read_text())

    def test_selected_module_only(self):
        """선택하지 않은 상관분석·전체 보고서·그림을 만들지 않는다."""
        self.config["steps"] = [self.config["steps"][1]]
        self.config["steps"][0]["depends_on"] = []
        self.setup_run()

        with patch("src.preprocessing.eda.analyze_correlations", side_effect=AssertionError("미선택 분석")):
            result, output, _ = self.run_pipeline()

        self.assertEqual(result.status, "completed")
        self.assertEqual({path.name for path in (output / "pipeline").iterdir()}, {"missing", "pipeline_result.json"})

    def test_plan_and_limits_change_invalidates_approval(self):
        """승인 후 순서·자원 변경은 원래 승인으로 실행하지 않는다."""
        self.setup_run()
        self.waiting()
        approval = self.approval()
        changed = deepcopy(self.config)
        changed["steps"][-1]["settings"] = {}
        changed["steps"] = changed["steps"][:-1]
        tool = ModulePipelineTool(self.states, self.tool.contract, changed)

        with patch("src.agents.module_pipeline.run_pipeline_worker") as launch:
            result = executor(tool).run_next(approval_id=approval)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        launch.assert_not_called()
        self.assertNotEqual(self.tool.plan.plan_id, ModulePipelineTool(
            self.states, replace(self.tool.contract, limits=WorkerLimits(60, 1024, 2)), self.config,
        ).plan.plan_id)

    def test_staged_tampering_is_rejected(self):
        """복사본·작업 설정·요청 식별값이 바뀌면 결과를 만들지 않는다."""
        self.setup_run()
        request = self.tool._request("a" * 32)
        root = self.tool._stage(request)
        (root / "input/snapshot/input.bin").write_bytes(b"changed")
        self.assertEqual(execute_request(request, root).status, "blocked_input")
        self.assertFalse((Path(request.output_directory) / "pipeline").exists())

    def test_parent_rejects_tampered_result_without_source_parse(self):
        """부모는 원본을 파싱하지 않으며 범위를 벗어난 집계와 추가 파일을 거부한다."""
        self.setup_run()
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        self.transport(request, root, self.tool.cancel)
        directory = Path(request.output_directory) / "pipeline"

        with patch("pandas.read_csv", side_effect=AssertionError("부모 원본 파싱")):
            self.assertIsNotNone(read_pipeline_result(directory, self.tool.module_plan, self.profile))

        path = directory / "missing/missing.csv"
        path.write_text("feature,missing_count,missing_ratio,observed_unique\nx,99,33,0\ny,0,0,3\n")
        self.assertIsNone(read_pipeline_result(directory, self.tool.module_plan, self.profile))
        (directory / "extra.pkl").write_bytes(b"not executed")
        self.assertIsNone(read_pipeline_result(directory, self.tool.module_plan, self.profile))

    def test_missing_job_evidence_never_commits(self):
        """종료 코드 0이어도 Job 정리·한도 증거가 없으면 실패 처리한다."""
        self.setup_run()

        def missing(request, root, signal):
            result = self.transport(request, root, signal)
            result["launch"]["resources"] = {}
            return result

        result, output, _ = self.run_pipeline(missing)
        self.assertEqual(result.code, "WORKER_FAILED")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_resource_exceeded_never_commits(self):
        """자원 초과 시 부분 산출물을 완료로 등록하지 않는다."""
        self.setup_run()

        def exceeded(request, root, signal):
            result = self.transport(request, root, signal)
            result["launch"]["status"] = "resource_exceeded"
            return result

        result, output, _ = self.run_pipeline(exceeded)
        self.assertEqual(result.code, "WORKER_RESOURCE_EXCEEDED")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_cancel_before_execution(self):
        """실행 전 취소는 승인 폐기 후 worker 시작을 막는다."""
        self.setup_run()
        self.waiting()
        approval = self.approval()
        cancel(self.tool)

        with patch("src.agents.module_pipeline.run_pipeline_worker") as launch:
            self.manager.run_next(approval_id=approval)

        launch.assert_not_called()
        self.assertEqual(self.manager.approvals.get(approval).status, "revoked")

    def test_noninteractive_approval_is_rejected(self):
        """파이프 입력으로 실행 승인을 자동 발급하지 않는다."""
        self.setup_run()
        self.waiting()

        with patch("src.agents.module_pipeline_cli.os.isatty", return_value=False), self.assertRaises(ValueError):
            approve(self.tool)

        self.assertEqual(self.calls, 0)

    def test_cli_selection_roundtrip_and_interactive_approval(self):
        """CLI 저장·복원·토큰 확인 승인까지 합성 입력으로 연결한다. worker는 시작하지 않는다."""
        pipeline = self.root / "pipeline.json"
        pipeline.write_text(json.dumps(self.config), encoding="utf-8")
        directory = self.root / "runs" / "cli_demo"
        directory.parent.mkdir()

        with patch("src.agents.module_pipeline_cli._directory", return_value=directory):
            created = create_demo("cli_demo", pipeline, WorkerLimits(60, 2048, 2))
            self.assertFalse(created["worker_started"])
            tool = open_run("cli_demo")

        manager = executor(tool)
        self.assertEqual(run_supported_preflight(manager, cli).code, "STEP_COMPLETED")
        self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")

        with patch("src.agents.module_pipeline_cli.os.isatty", return_value=True), \
             patch("src.agents.module_pipeline_cli.secrets.token_hex", return_value="aaaaaaaa"), \
             patch("builtins.input", return_value="PIPELINE-AAAAAAAA"), patch("builtins.print"):
            record = approve(tool)

        self.assertEqual(manager.approvals.get(record["approval_id"]).status, "pending")

    def test_mismatched_worker_reply_is_rejected(self):
        """다른 작업의 성공 응답을 현재 작업의 완료로 등록하지 않는다."""
        self.setup_run()

        def forged(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / "worker_result.json"
            reply = json.loads(path.read_text())
            reply["request_id"] = "0" * 64
            path.write_text(json.dumps(reply), encoding="utf-8")
            return outcome

        result, output, _ = self.run_pipeline(forged)
        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_quality_review_cannot_be_relabelled_passed(self):
        """정상 생성된 검토 보고서의 상태만 통과로 변경하면 부모 검증이 거부한다."""
        self.setup_run(b"x,y,target\n1,2,0\n1,4,1\n1,6,0\n")
        request = self.tool._request("d" * 32)
        root = self.tool._stage(request)
        self.transport(request, root, self.tool.cancel)
        directory = Path(request.output_directory) / "pipeline"
        path = directory / "pipeline_result.json"
        result = json.loads(path.read_text())
        result["status"] = "passed"
        path.write_text(json.dumps(result), encoding="utf-8")

        self.assertIsNone(read_pipeline_result(directory, self.tool.module_plan, self.profile))

    def test_cancel_during_execution(self):
        """worker 성공 보고가 있어도 실행 중 취소가 도착하면 커밋하지 않는다."""
        self.setup_run()

        def cancelled(request, root, signal):
            result = self.transport(request, root, signal)
            cancel(self.tool)
            return result

        result, output, _ = self.run_pipeline(cancelled)
        self.assertEqual(result.status, "cancelled")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_copied_worker_imports_without_project(self):
        """복사 허용 목록만으로 별도 프로세스에서 모듈 코어를 실행할 수 있다."""
        self.setup_run()
        request = self.tool._request("c" * 32)
        root = self.tool._stage(request)
        project = Path(__file__).resolve().parents[2]
        tools = root / "tools"

        for name in WORKER_FILES:
            target = tools / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        shutil.copyfile(project / ENTRY, tools / "worker.py")
        script = (
            "import sys;sys.path[:0]=sys.argv[1:3];from pathlib import Path;"
            "from src.agents.worker_contract import WorkerRequest;import worker;"
            "root=Path(sys.argv[3]);r=WorkerRequest.from_json((root/'input/request.json').read_text());"
            "assert worker.execute_request(r,root).status=='succeeded'"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(tools), str(Path(sys.prefix) / "Lib/site-packages"), str(root)],
            cwd=self.root, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()

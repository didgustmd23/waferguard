"""분할 승인 CLI와 격리 worker 연결을 합성 입력으로 검사한다. OS 운송은 모의한다."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents.snapshot_split_cli import create, open_run, executor, approve, export_scope
from src.agents.snapshot_split_worker import execute_request
from src.agents.snapshot_split_transport import WORKER_FILES, ENTRY
from src.agents.partition_export_cli import create as create_export, open_run as open_export
from src.agents.partition_export_cli import executor as export_executor, status as export_status
from src.agents.partition_export_worker import execute_request as execute_export
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules
from src.preprocessing.split import SplitRules


class SnapshotSplitExecutionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.csv"
        pd.DataFrame({"x": range(120), "label": [0, 1] * 60,
                      "id": [f"{index:04d}" for index in range(120)],
                      "time": pd.date_range("2026-01-01", periods=120, freq="h").strftime("%Y-%m-%d %H:%M:%S")}).to_csv(self.source, index=False)
        self.profile = DatasetProfile(
            "split_example", "tabular_csv_v1", "split_review_v1",
            ColumnRoles(numeric=("x",), target="label", timestamp="time", identifiers=("id",)),
            max_plot_features=1,
            quality_rules=QualityRules(allowed_labels=(0, 1), target_missing="block", duplicate_id="block",
                                       identifier_missing="block", timestamp_missing="block", timestamp_format="%Y-%m-%d %H:%M:%S"),
            split_rules=SplitRules(strategy="time", timestamp_format="%Y-%m-%d %H:%M:%S"),
        )
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(self.profile.to_dict()), encoding="utf-8")
        self.directory = self.root / "runs/example"
        self.directory.parent.mkdir()
        directory_patch = patch("src.agents.snapshot_split_cli._directory", return_value=self.directory)
        actor_patch = patch("src.agents.snapshot_split_cli._current_actor", return_value="synthetic_actor")
        directory_patch.start()
        actor_patch.start()
        self.addCleanup(directory_patch.stop)
        self.addCleanup(actor_patch.stop)
        self.calls = []

    def setup_run(self, **options):
        result = create("example", self.source, self.profile_path, WorkerLimits(60, 2048, 2), 65536,
                        confirmed=True, **options)
        self.assertFalse(result["worker_started"])
        self.tool = open_run("example")
        self.manager = executor(self.tool)

    def transport(self, request, root, signal, worker=execute_request):
        """본 worker 계산을 실행하며 네이티브 Job 종료 증거만 합성한다."""
        self.calls.append(request.stage)
        reply = worker(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes},
        }}
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
        return outcome

    def advance(self, stage):
        """대화형 UI 대신 테스트 승인만 발급한다. 선행 단계는 실제 실행기로 진행한다."""
        self.assertEqual(self.tool.states.load().current_stage, stage)
        self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")
        record = self.manager.approvals.record_confirmed_approval(self.manager.scope_for(stage))

        with patch("src.agents.snapshot_split.run_split_worker", side_effect=self.transport):
            return self.manager.run_next(approval_id=record.approval_id)

    def prepare_preview(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        for stage in ("audit", "run_eda", "review_eda", "propose_split"):
            self.assertEqual(self.advance(stage).code, "STEP_COMPLETED")

    def test_full_workflow_scope_and_export_connection(self):
        """진짜 승인·커밋한 분할 범위를 저장하고 후속 내보내기·구간 계약까지 연결한다."""
        self.setup_run()
        before = self.source.read_bytes()
        self.prepare_preview()
        self.assertEqual(self.advance("commit_split").status, "completed")
        paths = export_scope(self.tool)
        self.assertEqual(self.calls, ["audit", "run_eda", "propose_split", "commit_split"])
        self.assertEqual(self.source.read_bytes(), before)
        self.assertTrue(Path(paths["split_scope"]).is_file())

        with patch("src.agents.partition_export_cli._directory", return_value=self.root / "export_runs/example"), patch("src.agents.partition_export_cli._current_actor", return_value="synthetic_actor"):
            (self.root / "export_runs").mkdir()
            create_export("example", paths["split_run_dir"], paths["split_scope"], paths["snapshot_dir"], paths["profile"],
                          WorkerLimits(60, 2048, 2), 65536, confirmed=True)
            tool = open_export("example")
            manager = export_executor(tool)
            manager.run_next()
            self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")
            approval = manager.approvals.record_confirmed_approval(manager.scope_for("ingest"))

            with patch("src.agents.partition_export.run_export_worker", side_effect=lambda request, root, signal: self.transport(request, root, signal, execute_export)):
                self.assertEqual(manager.run_next(approval_id=approval.approval_id).status, "completed")

            self.assertEqual(export_status(tool)["partition_contracts"], "passed")

    def test_review_is_not_automatically_approved(self):
        """EDA 실행 뒤 사람 검토 승인 없이는 분할 제안을 실행하지 않는다."""
        self.setup_run()
        self.manager.run_next()
        self.advance("audit")
        self.advance("run_eda")

        with patch("src.agents.snapshot_split.run_split_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        self.assertNotIn("propose_split", self.calls)

    def test_quality_review_stops_without_deleting(self):
        """상수 센서 검토 상태는 후속 EDA·분할을 자동 실행하거나 센서를 삭제하지 않는다."""
        self.source.write_text("x,label,id,time\n1,0,0001,2026-01-01 00:00:00\n1,1,0002,2026-01-01 01:00:00\n", encoding="utf-8")
        self.setup_run()
        self.manager.run_next()
        result = self.advance("audit")
        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertEqual(self.calls, ["audit"])
        self.assertNotIn("audit", self.tool.states.load().artifact_refs)

    def test_feature_notice_policy_is_explicit(self):
        """분할 전용 통지 옵션은 구조·라벨·ID 차단을 바꾸지 않고 프로파일에 보존한다."""
        self.setup_run(allow_feature_findings=True)
        rules = self.tool.contract.profile.quality_rules
        self.assertEqual(rules.constant_feature, "ignore")
        self.assertEqual(rules.target_missing, "block")
        self.assertNotEqual(self.tool.contract.profile.signature, self.profile.signature)

    def test_tampered_preview_blocks_commit(self):
        """확정 승인 뒤 제안 변경은 선행 산출물 검사에서 차단한다."""
        self.setup_run()
        self.prepare_preview()
        self.manager.run_next()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("commit_split"))
        state = self.tool.states.load()
        path = self.directory / state.artifact_refs["propose_split"] / "proposal.json"
        path.write_text("{}", encoding="utf-8")

        with patch("src.agents.snapshot_split.run_split_worker") as launch:
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.code, "PREREQUISITE_ARTIFACT_INVALID")
        launch.assert_not_called()

    def test_changed_snapshot_and_original_independence(self):
        """원본 이후 변경은 복사본을 바꾸지 않으며, 선택한 snapshot 변경은 재개를 막는다."""
        self.setup_run()
        self.source.write_text("changed", encoding="utf-8")
        self.assertEqual(open_run("example").plan.plan_id, self.tool.plan.plan_id)
        (Path(self.tool.contract.snapshot_directory) / "input.bin").write_text("changed", encoding="utf-8")

        with self.assertRaises(ValueError):
            open_run("example")

    def test_no_scope_before_commit_and_no_automatic_approval(self):
        """미완료 실행에서 승인 범위 증거를 내보내지 않으며 비대화형 승인을 거부한다."""
        self.setup_run()

        with self.assertRaises(ValueError):
            export_scope(self.tool)

        with patch("os.isatty", return_value=False), self.assertRaises(ValueError):
            approve(self.tool)

    def test_copied_worker_imports_without_project(self):
        """고정 복사 파일만으로 별도 프로세스에서 품질 검사를 실행한다."""
        self.setup_run()
        request = self.tool._request_for("a" * 32, "audit")
        root = self.tool._stage_for(request, None)
        project = Path(__file__).resolve().parents[2]

        for name in WORKER_FILES:
            target = root / "tools" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        shutil.copyfile(project / ENTRY, root / "tools/worker.py")
        result = subprocess.run([sys.executable, "-I", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                                cwd=root, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))

    def test_missing_cleanup_never_commits(self):
        """성공 응답이 있어도 Job 정리 증거가 없으면 audit을 완료 등록하지 않는다."""
        self.setup_run()
        self.manager.run_next()
        self.manager.run_next()
        record = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("audit"))

        def missing(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["resources"]["cleanup_confirmed"] = False
            return outcome

        with patch("src.agents.snapshot_split.run_split_worker", side_effect=missing):
            result = self.manager.run_next(approval_id=record.approval_id)

        self.assertEqual(result.status, "failed_runtime")
        self.assertNotIn("audit", self.tool.states.load().artifact_refs)

    def test_resource_failure_never_commits(self):
        """worker가 결과를 남겼더라도 자원 초과 종료를 성공으로 변경하지 않는다."""
        self.setup_run()
        self.manager.run_next()
        self.manager.run_next()
        record = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("audit"))

        def exceeded(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["status"] = "resource_exceeded"
            return outcome

        with patch("src.agents.snapshot_split.run_split_worker", side_effect=exceeded):
            result = self.manager.run_next(approval_id=record.approval_id)

        self.assertEqual(result.status, "failed_runtime")
        self.assertNotIn("audit", self.tool.states.load().artifact_refs)


    def test_eda_selects_headless_backend_before_report(self):
        """환경 기본값과 관계없이 보고서 생성 전에 Agg를 지정한다."""
        self.setup_run()
        request = self.tool._request_for("b" * 32, "run_eda")
        root = self.tool._stage_for(request, None)
        calls = []

        def select_backend(name, **options):
            calls.append((name, options))

        def render(*args, **kwargs):
            self.assertEqual(calls, [("Agg", {"force": True})])
            raise RuntimeError("합성 렌더링 실패")

        with patch("matplotlib.use", side_effect=select_backend), patch("src.preprocessing.eda_report.generate_eda_report", side_effect=render):
            reply = execute_request(request, root)

        self.assertEqual(reply.status, "failed_runtime")
        diagnosis = json.loads((root / "scratch/task/diagnostic.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnosis, {"stage": "EDA", "code": "RUNTIME_FAILED"})
        self.assertEqual(reply.artifact_refs, ())


if __name__ == "__main__":
    unittest.main()

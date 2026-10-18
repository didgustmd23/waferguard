"""합성 canonical로 승인·WM 코어 재사용·검토 중단·결과 변조 차단을 검사한다."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.agents.wm_canonical_cli import create_demo, executor, open_run, status
from src.agents.wm_canonical_contract import inventory
from src.agents.wm_canonical_worker import execute_request
from src.agents.wm_canonical_transport import WORKER_FILES, ENTRY
from src.agents.worker_contract import WorkerLimits


class WmCanonicalTest(unittest.TestCase):
    def setUp(self):
        """실제 pickle·WM 데이터 대신 27개의 독립 합성 맵으로 입력을 준비한다."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory = self.root / "runs/example"
        self.directory.parent.mkdir()
        mocking = patch("src.agents.wm_canonical_cli._directory", return_value=self.directory)
        mocking.start()
        self.addCleanup(mocking.stop)

        with patch("pandas.read_pickle", side_effect=AssertionError("pickle 읽기 금지")), \
                patch("numpy.load", side_effect=AssertionError("생성 시 NPZ 파싱 금지")):
            result = create_demo("example", WorkerLimits(60, 2048, 2))

        self.assertFalse(result["worker_started"])
        self.tool = open_run("example")
        self.manager = executor(self.tool)

    def approval(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.wm_canonical.run_wm_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("run_eda"))

    def transport(self, request, root, signal):
        """OS 전송만 모의하며 기존 WM 검증·EDA 코어는 실제로 수행한다."""
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

    def test_approval_then_completion_no_rerun(self):
        """승인 후 기존 WM 보고서를 만들고 재실행·원본 변경 없이 완료한다."""
        approval = self.approval()
        before = inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes)

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=self.transport) as launch, \
                patch("pandas.read_pickle", side_effect=AssertionError("pickle 재읽기 금지")):
            result = self.manager.run_next(approval_id=approval.approval_id)
            self.manager.run_next()

        self.assertEqual(result.status, "completed", result)
        launch.assert_called_once()
        report = status(open_run("example"))
        self.assertEqual(report["quality_status"], "passed")
        self.assertTrue(report["eda_generated"])
        self.assertTrue(Path(report["local_eda_report"]).is_file())
        self.assertEqual(inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes), before)

    def test_quality_review_never_runs_eda(self):
        """독립 그룹 부족을 자동 해제하지 않고 품질 보고서만 남긴다."""
        from src.agents.wm_canonical_cli import create

        # 첫 클래스별 표본만 가진 별도 합성 묶음으로 새 실행을 만든다.
        other = self.root / "runs/review"

        with patch("src.agents.wm_canonical_cli._directory", return_value=other):
            create_demo("review", self.tool.limits, review=True)
            self.tool = open_run("review")
            self.directory = other
            self.manager = executor(self.tool)
            approval = self.approval()

            with patch("src.agents.wm_canonical.run_wm_worker", side_effect=self.transport), \
                    patch("src.wafer_dl.eda.run_eda", side_effect=AssertionError("검토 필요 시 EDA 금지")):
                result = self.manager.run_next(approval_id=approval.approval_id)

            report = status(self.tool)

        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED")
        self.assertEqual(report["quality_status"], "review_required")
        self.assertFalse(report["eda_generated"])
        self.assertNotIn("run_eda", self.tool.states.load().artifact_refs)

    def test_changed_snapshot_is_blocked_before_parse(self):
        """맵 바이트가 달라지면 WM 검증·배열 파싱 전에 차단한다."""
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        shard = next((root / "input/canonical/maps").iterdir())
        shard.write_bytes(b"changed")

        with patch("src.wafer_dl.validate_canonical.validate_bundle", side_effect=AssertionError("변조 파싱 금지")):
            reply = execute_request(request, root)

        self.assertEqual(reply.status, "blocked_input")

    def test_resource_failure_does_not_commit(self):
        """부분 보고서가 있어도 자원 초과 종료는 완료하지 않는다."""
        approval = self.approval()

        def failed(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["status"] = "resource_exceeded"
            return outcome

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=failed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")
        self.assertNotIn("run_eda", self.tool.states.load().artifact_refs)

    def test_changed_output_is_rejected(self):
        """worker 성공 주장과 별개로 그림·표의 해시 변경은 부모가 거부한다."""
        approval = self.approval()

        def tampered(request, root, signal):
            outcome = self.transport(request, root, signal)
            (Path(request.output_directory) / "analysis/eda/classes.csv").write_bytes(b"changed")
            return outcome

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=tampered):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("run_eda", self.tool.states.load().artifact_refs)

    def test_cancel_never_starts_worker(self):
        """실행 전 취소는 원본 파싱·worker 실행 없이 종료한다."""
        from src.agents.isolated_eda_cli import cancel

        approval = self.approval()
        cancel(self.tool)

        with patch("src.agents.wm_canonical.run_wm_worker") as launch:
            result = self.manager.run_next(approval_id=approval.approval_id)

        launch.assert_not_called()
        self.assertNotEqual(result.status, "completed")

    def test_missing_job_cleanup_rejected(self):
        """유효한 분석 파일이 있어도 전체 Job 정리 증거가 빠지면 완료하지 않는다."""
        approval = self.approval()

        def no_cleanup(request, root, signal):
            outcome = self.transport(request, root, signal)
            del outcome["launch"]["resources"]["cleanup_confirmed"]
            return outcome

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=no_cleanup):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("run_eda", self.tool.states.load().artifact_refs)

    def test_output_limit_blocks_partial_reports(self):
        """출력 상한을 초과한 부분 EDA를 완료 산출물로 보고하지 않는다."""
        from src.agents.wm_canonical_contract import digest

        request = self.tool._request("d" * 32)
        root = self.tool._stage(request)
        path = root / "input/analysis.json"
        packet = json.loads(path.read_text())
        packet["max_output_bytes"] = 1
        request = replace(request, config_id=digest(packet))
        path.write_text(json.dumps(packet), encoding="utf-8")
        reply = execute_request(request, root)
        self.assertEqual(reply.status, "blocked_input")
        self.assertFalse((Path(request.output_directory) / "analysis/result.json").exists())

    def test_request_mismatch_does_not_parse(self):
        """다른 source ID의 요청은 기존 WM 검증기에 전달하지 않는다."""
        request = self.tool._request("c" * 32)
        root = self.tool._stage(request)
        changed = replace(request, source_id="other_source")

        with patch("src.wafer_dl.validate_canonical.validate_bundle", side_effect=AssertionError("불일치 입력 금지")):
            reply = execute_request(changed, root)

        self.assertEqual(reply.status, "blocked_input")

    def test_settings_change_requires_new_run(self):
        """예시 맵·품질 상한·자원 변경에는 새 계획과 승인이 필요하다."""
        selection = self.directory / "selection.json"
        record = json.loads(selection.read_text())
        record["contract"]["include_examples"] = True
        selection.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaises(ValueError):
            open_run("example")

    def test_unknown_paths_and_hardlinks_rejected(self):
        """임의 추가 파일·디렉터리 탈출 경로는 허용 파일 명세로 처리하지 않는다."""
        from src.agents.wm_canonical_contract import check_files

        files = self.tool.contract.files
        files["../outside"] = {"size": 0, "sha256": "0" * 64}

        with self.assertRaises(ValueError):
            check_files(files, self.tool.contract.max_input_bytes)

        (Path(self.tool.contract.snapshot_directory) / "extra.pkl").write_bytes(b"not allowed")

        with self.assertRaises(ValueError):
            self.tool.contract.check()

    def test_host_entry_refuses_data_access(self):
        """일반 프로세스에서 worker 진입점을 호출하면 입력을 읽기 전에 거부한다."""
        from src.agents.wm_canonical_worker import main

        with patch("sys.argv", ["worker", str(self.root), "a" * 64]), \
                patch("src.agents.pickle_import_worker._isolated_process", return_value=False), \
                patch("src.agents.wm_canonical_worker.execute_request", side_effect=AssertionError("host 실행 금지")):
            self.assertEqual(main(), 2)

    def test_copied_worker_bootstraps(self):
        """원본 프로젝트 import 캐시 없이 고정 복사 목록만으로 worker를 로딩한다."""
        copied = self.root / "copied"
        copied.mkdir()
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        script = (
            "import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
            "from src.agents.wm_canonical_worker import execute_request; "
            "from src.agents.wm_canonical_contract import inventory; "
            "from src.agents.wm_canonical_results import result_files; "
            "from src.wafer_dl.eda import run_eda"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(copied),
                                 str(Path(sys.prefix) / "Lib/site-packages")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

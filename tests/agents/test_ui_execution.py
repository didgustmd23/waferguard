"""승인·격리 실행·결과 화면 연결 검사. 합성 데이터와 모의 worker만 사용한다."""

from contextlib import closing
from importlib.util import find_spec
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from src.agents import pipeline_approval_cli as backend
from src.agents.contracts import ToolResult
from src.agents.pipeline_csv_execution import CsvPipelineExecution
from src.agents.module_pipeline_results import read_pipeline_result
from src.agents.reviewed_module_contract import demo_profile
from src.preprocessing.builtin_modules import build_builtin_registry
from src.preprocessing.contracts import TabularInput
from src.preprocessing.quality import write_quality_report
from src.ui.dialogue_service import fingerprint
from src.ui.execution_service import UiExecution
from src.ui.result_service import load_results
from src.ui.state import SCREENS, resolve_screen
from src.ui import navigation  # 모의 함수가 최초 import의 전역 참조로 남지 않게 한다.
from tests.agents import test_pipeline_approval as fixtures

PIPELINE_PATH = fixtures.PIPELINE_PATH
PROFILE_PATH = fixtures.PROFILE_PATH


class UiExecutionTest(unittest.TestCase):
    setUp = fixtures.PipelineApprovalTest.setUp
    run_directory = fixtures.PipelineApprovalTest.run_directory
    cli_call = fixtures.PipelineApprovalTest.cli_call
    create_run = fixtures.PipelineApprovalTest.create_run

    def prepare(self, pipeline=PIPELINE_PATH):
        self.calls = 0
        patcher = patch("src.agents.pipeline_approval_cli._directory", side_effect=self.run_directory)
        patcher.start()
        self.addCleanup(patcher.stop)
        profile = self.root / "demo_profile.json"
        profile.write_text(json.dumps(demo_profile().to_dict()), encoding="utf-8")
        backend.create("ui-test01", self.snapshot_dir, profile, pipeline,
                       backend.WorkerLimits(60, 2048, 2), confirmed=True)
        selection = backend._read_json(self.run_directory("ui-test01") / "selection.json", 256 * 1024)
        receipt = {"name": "ui-test01", "run_dir": str(self.run_directory("ui-test01")),
                   "binding_id": selection["execution_plan"]["config_id"]}
        self.execution = UiExecution(receipt, fingerprint(selection))
        patcher = patch("src.agents.pipeline_approval.build_pipeline_execution", side_effect=self.fake_factory)
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.execution

    def fake_factory(self, adapter_id, states, binding):
        """고정 합성 CSV의 코어만 실행한다. native worker 시작 함수는 호출하지 않는다."""
        import pandas as pd

        actual = CsvPipelineExecution(states, binding)
        tool = actual.worker_tool
        test = self

        class Fake:
            worker_tool = tool
            worker_started = False

            def execute(self, output, arguments):
                test.calls += 1
                self.worker_started = True
                data = TabularInput(pd.read_csv(test.snapshot_dir / "input.bin"), tool.contract.profile.roles)
                target = output / "pipeline"
                target.mkdir()
                completed = []
                status = "passed"

                for step in tool.module_plan.steps:
                    module = build_builtin_registry().resolve(step.module_id, step.version)
                    result = module.calculate(data, **module.validate_settings(json.loads(step.settings_json)))
                    child = target / step.step_id

                    if step.module_id == "tabular_quality":
                        write_quality_report(result, child)
                        status = result.status
                    else:
                        child.mkdir()
                        if step.module_id == "eda_missing":
                            result.to_csv(child / "missing.csv", encoding="utf-8-sig")
                        else:
                            for name, table in result.items():
                                table.to_csv(child / f"{name}.csv", encoding="utf-8-sig")

                    completed.append(step.step_id)
                    if status != "passed":
                        break

                (target / "pipeline_result.json").write_text(json.dumps({
                    "version": "module_pipeline_result_v1", "pipeline_signature": tool.module_plan.signature,
                    "row_count": len(data.frame), "status": status, "completed_steps": completed,
                }), encoding="utf-8")
                return ToolResult(status)

            def validate(self, output, arguments):
                result = read_pipeline_result(output / "pipeline", tool.module_plan, tool.contract.profile)
                return result is not None and result["status"] == "passed"

        return Fake()

    def ready(self, execution):
        execution.command("preflight")
        execution.command("preflight")
        self.assertEqual(execution.read_state().status, "awaiting_approval")
        execution.command("preview")

    def test_complete_and_duplicate_execution_rejected(self):
        execution = self.prepare()
        self.ready(execution)
        self.assertEqual(self.calls, 0)

        with self.assertRaises(ValueError):
            execution.command("approve")

        execution.command("approve", confirmed=True)
        self.assertEqual(self.calls, 0)
        self.assertEqual(backend.status(execution.name)["approval_status"], "pending")
        execution.command("execute", confirmed=True)
        self.assertEqual(execution.read_state().status, "completed")
        self.assertTrue(execution.worker_started)
        self.assertEqual(backend.status(execution.name)["approval_status"], "consumed")
        result = load_results(execution)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(any(item["suffix"] == ".csv" for item in result["files"]))

        with self.assertRaises(ValueError):
            execution.command("execute", confirmed=True)
        self.assertEqual(self.calls, 1)

    def test_review_stops_analysis_and_can_open_report(self):
        execution = self.prepare(PIPELINE_PATH.with_name("tabular_review_example.json"))
        self.ready(execution)
        execution.command("approve", confirmed=True)
        execution.command("execute", confirmed=True)
        self.assertEqual(execution.read_state().status, "blocked_input")
        result = load_results(execution)
        self.assertEqual(result["status"], "review_required")
        self.assertFalse(any("missing/" in item["name"] for item in result["files"]))
        self.assertTrue(any(item["suffix"] == ".md" for item in result["files"]))

    def test_changed_preview_or_state_does_not_issue_approval(self):
        execution = self.prepare()
        self.ready(execution)
        preview = execution.preview
        state = execution.read_state()
        from src.agents.state import LocalStateStore
        LocalStateStore(execution.directory).save(state, expected_revision=state.revision)

        with self.assertRaises(ValueError):
            backend.record_approval(execution.name, preview, confirmed=True)

        with closing(sqlite3.connect(execution.directory / "approvals.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 0)

    def test_changed_snapshot_after_approval_never_starts_worker(self):
        execution = self.prepare()
        self.ready(execution)
        execution.command("approve", confirmed=True)
        (self.snapshot_dir / "input.bin").write_bytes(b"changed")

        with self.assertRaises(ValueError):
            execution.command("execute", confirmed=True)
        self.assertEqual(self.calls, 0)

    def test_no_retry_after_interrupted_execution(self):
        execution = self.prepare()
        self.ready(execution)
        execution.command("approve", confirmed=True)

        with patch.object(backend, "next_step", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                execution.command("execute", confirmed=True)

        self.assertTrue(execution.error)
        with self.assertRaises(ValueError):
            execution.command("execute", confirmed=True)
        self.assertEqual(self.calls, 0)

    def test_result_tampering_is_rejected(self):
        execution = self.prepare()
        self.ready(execution)
        execution.command("approve", confirmed=True)
        execution.command("execute", confirmed=True)
        result = load_results(execution)
        report = next(item for item in result["files"] if item["suffix"] == ".md")
        (Path(result["directory"]) / report["name"]).write_text("modified", encoding="utf-8")

        with self.assertRaises(ValueError):
            load_results(execution)

    def test_approval_scope_tampering_and_duplicate_issuance_rejected(self):
        execution = self.prepare()
        self.ready(execution)
        from copy import deepcopy
        original = deepcopy(execution.preview)
        modified = deepcopy(original)
        modified["summary"]["limits"]["processes"] = 8

        with self.assertRaises(ValueError):
            backend.record_approval(execution.name, modified, confirmed=True)

        backend.record_approval(execution.name, original, confirmed=True)
        with self.assertRaises(ValueError):
            backend.record_approval(execution.name, original, confirmed=True)

        with closing(sqlite3.connect(execution.directory / "approvals.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 1)
        self.assertEqual(self.calls, 0)

    def test_expired_approval_never_starts_worker(self):
        execution = self.prepare()
        self.ready(execution)
        execution.command("approve", confirmed=True)

        with closing(sqlite3.connect(execution.directory / "approvals.sqlite3")) as db:
            db.execute("UPDATE approvals SET expires_at = ?", ("2000-01-01T00:00:00+00:00",))
            db.commit()

        with self.assertRaises(ValueError):
            execution.command("execute", confirmed=True)
        self.assertEqual(self.calls, 0)

    def test_state_refresh_is_read_only_and_result_gate_independent_of_new_input(self):
        execution = self.prepare()
        with patch.object(backend, "_open", side_effect=AssertionError("실행기 생성 금지")), patch(
                "sqlite3.connect", side_effect=AssertionError("DB 변경 금지")), patch(
                "subprocess.Popen", side_effect=AssertionError("worker 금지")):
            for _ in range(3):
                self.assertEqual(execution.read_state().status, "created")

        self.assertEqual(resolve_screen(SCREENS[4], execution_available=True), SCREENS[4])
        self.assertEqual(resolve_screen(SCREENS[3], execution_available=True), SCREENS[0])

    @unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 필요")
    def test_streamlit_approval_execution_results_and_rerender(self):
        from streamlit.testing.v1 import AppTest

        execution = self.prepare()
        receipt = {"name": execution.name, "run_dir": str(execution.directory)}

        with patch("src.ui.candidate_view.current_candidate_receipt", return_value=receipt), patch(
                "src.ui.navigation.current_candidate_receipt", return_value=receipt), patch(
                "src.ui.navigation.review_available", return_value=True), patch(
                "src.ui.navigation.input_available", return_value=True):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/ui/app.py"))
            app.session_state["ui_active_execution"] = execution
            app.session_state["ui_execution_candidate"] = None
            app.session_state["ui_screen"] = SCREENS[3]
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any(item.key == "ui_preflight" for item in app.button),
                            [(item.key, item.label) for item in app.button])
            app.button(key="ui_preflight").click().run()
            self.assertFalse(execution.error, [item.value for item in app.error])
            app.button(key="ui_preflight").click().run()
            app.button(key="ui_approval_preview").click().run()
            next(item for item in app.button if item.label == "이번 실행 승인").click().run()
            self.assertIsNone(execution.approval_id)
            app.checkbox[0].check()
            next(item for item in app.button if item.label == "이번 실행 승인").click().run()
            self.assertEqual(self.calls, 0)
            app.run()
            self.assertEqual(self.calls, 0)
            app.checkbox[0].check()
            next(item for item in app.button if item.label == "승인한 작업 실행").click().run()
            app.run()
            self.assertEqual(self.calls, 1)
            app.button(key="ui_step_5").click().run()
            app.button(key="ui_load_results").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.table)
            app.button(key="ui_refresh_status").click().run()
            self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()

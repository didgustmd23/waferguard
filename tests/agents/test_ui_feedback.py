"""사용자 안내·권한 사전 검사·경로 진단. 실제 승격·worker·파일 수정은 하지 않는다."""

from importlib.util import find_spec
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.ui import runtime_notice
from src.ui.workflow_feedback import diagnose_paths, show_error
from src.ui.workflow_service import UiWorkflow


class FeedbackTest(unittest.TestCase):
    def test_actual_server_permission_query_success_denial_and_unknown(self):
        for value in (0, 1):
            check = Mock(return_value=value)
            windll = SimpleNamespace(shell32=SimpleNamespace(IsUserAnAdmin=check))
            with patch.object(runtime_notice.os, "name", "nt"), \
                    patch.object(runtime_notice.ctypes, "windll", windll, create=True):
                result = runtime_notice.server_permissions()
            self.assertEqual(result["elevated"], bool(value))
            self.assertEqual(result["pid"], os.getpid())
            check.assert_called_once_with()

        with patch.object(runtime_notice.os, "name", "nt"), patch.object(
                runtime_notice.ctypes, "windll", SimpleNamespace(), create=True):
            self.assertIsNone(runtime_notice.server_permissions()["elevated"])

    def test_denied_or_unknown_ui_permission_does_not_submit_or_consume_approval(self):
        from src.ui.workflow_view import _command
        from src.ui.execution_view import _submit

        for elevated in (False, None):
            for submit in (_command, _submit):
                for action in ("approve", "execute"):
                    st, controller = MagicMock(), Mock()
                    controller.approval_id = "unchanged"
                    with patch.object(runtime_notice, "server_permissions", return_value={"elevated": elevated}):
                        submit(st, controller, action, confirmed=True)
                    controller.command.assert_not_called()
                    self.assertEqual(controller.approval_id, "unchanged")
                    st.rerun.assert_not_called()

    def test_relative_and_absolute_diagnosis_match_without_mutating_selection(self):
        schema = {"fields": [{"name": "pipeline", "title": "계획", "kind": "json_file"},
                             {"name": "directory", "title": "폴더", "kind": "directory"}]}
        values = {"pipeline": "configs/agents/pipelines/wm_development_example.json", "directory": "configs/agents"}
        original = dict(values)
        relative = diagnose_paths(schema, values)
        absolute = diagnose_paths(schema, {key: str(Path(value).absolute()) for key, value in values.items()})
        self.assertTrue(all(row["valid"] for row in relative + absolute))
        self.assertEqual([r["resolved"] for r in relative], [r["resolved"] for r in absolute])
        self.assertEqual(values, original)

    def test_bad_path_type_missing_control_and_wrong_kind_are_diagnosed(self):
        schema = {"fields": [{"name": "path", "title": "계획 파일", "kind": "json_file"}]}
        for value in ("", "\x00bad", "missing/file.json", "configs/agents", 123,
                      "configs/../config.json", "C:/file.json:stream"):
            with self.subTest(value=value):
                row = diagnose_paths(schema, {"path": value})[0]
                self.assertFalse(row["valid"])
                self.assertIsNone(row["resolved"])

    def test_safe_failure_messages_do_not_leak_raw_exception_or_read_inputs(self):
        st = MagicMock()
        show_error(st, RuntimeError("PRIVATE_PATH_AND_SECRET"), "조건을 확인하세요.")
        st.error.assert_called_once_with("조건을 확인하세요.")
        self.assertNotIn("PRIVATE_PATH", str(st.mock_calls))
        show_error(st, ValueError("작업 배정이 현재 모듈 계획의 버전·서명과 다릅니다."), "기본 안내")
        self.assertIn("PIPELINE_BINDING_MISMATCH", str(st.caption.call_args))


@unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 필요")
class FeedbackAppTest(unittest.TestCase):
    def test_failed_run_explains_failure_without_loading_result_artifacts(self):
        from streamlit.testing.v1 import AppTest

        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        workflow.name = "failed-example"
        workflow.state = {"status": "failed_runtime", "stage": "run_eda"}
        service = Mock()
        with patch.object(UiWorkflow, "service", return_value=service), patch.object(
                runtime_notice, "server_permissions", return_value={"elevated": False, "pid": 1, "python": "synthetic"}):
            app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import results\nresults(st)")
            app.session_state["ui_workflow"] = workflow
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("정상 완료되지" in item.value for item in app.error))
            self.assertFalse(any(button.key == "workflow_results" for button in app.button))
            app.run()
        service.results.assert_not_called()
        service.next.assert_not_called()
        service.status.assert_not_called()

    def test_non_admin_approval_is_disabled_and_review_has_collapsed_details(self):
        from streamlit.testing.v1 import AppTest
        from tests.agents.test_ui_workflow import FakeService

        service = FakeService()
        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        with patch.object(UiWorkflow, "service", return_value=service), patch.object(
                runtime_notice, "server_permissions", return_value={"elevated": False, "pid": 1, "python": "synthetic"}):
            workflow.connect("synthetic")
            workflow.command("next")
            workflow.command("preview")
            app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import approval\napproval(st)")
            app.session_state["ui_workflow"] = workflow
            app.run()
            app.button(key="approval_flow_start").click().run()
            button = app.button(key="FormSubmitter:approval_flow_approve-승인하고 실행")
            self.assertTrue(button.disabled)
            self.assertEqual(service.approval_calls, 0)

            review = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import review\nreview(st)")
            review.session_state["ui_workflow"] = workflow
            review.run()
            self.assertFalse(review.exception)
            self.assertTrue(any("아직 데이터를 분석" in item.value for item in review.success))
            details = next(item for item in review.expander if item.label == "모듈 기술 상세")
            self.assertFalse(details.proto.expanded)

    def test_invalid_selected_path_names_field_and_discards_previous_preview(self):
        from streamlit.testing.v1 import AppTest
        from tests.agents.test_ui_workflow import FakeService

        service = FakeService()
        service.preview_form = Mock()
        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        workflow.reviewed = True
        workflow.creation_preview = SimpleNamespace(packet={"run_name": "old-preview"})
        with patch.object(UiWorkflow, "service", return_value=service):
            app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import inputs\ninputs(st)")
            app.session_state["ui_workflow"] = workflow
            app.session_state["ui_settings"] = {"limits": {"wall_seconds": 300, "memory_mib": 2048, "processes": 2}}
            app.run()
            prefix = "workflow_wm_canonical_eda_"
            app.text_input(key="_widget_" + prefix + "bindings").input("missing/bindings.json")
            app.text_input(key="_widget_" + prefix + "canonical_dir").input("configs/agents")
            app.text_input(key=prefix + "name").input("new-preview")
            app.button(key="FormSubmitter:workflow_creation_wm_canonical_eda-생성 범위 검사·미리보기").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("현재 작업 배정 JSON: 확인 필요" in item.value for item in app.text))
        self.assertIsNone(workflow.creation_preview)
        service.preview_form.assert_not_called()
        self.assertFalse(any(button.label == "새 작업 준비" for button in app.button))


if __name__ == "__main__":
    unittest.main()

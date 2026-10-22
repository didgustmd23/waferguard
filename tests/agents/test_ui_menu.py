"""메뉴 전환만으로 실행하지 않고, 기존 도구는 명시적으로 여는지 검사한다."""

from importlib.util import find_spec
from pathlib import Path
import unittest
from unittest.mock import patch


@unittest.skipUnless(find_spec("streamlit"), "Streamlit이 필요합니다.")
class MenuTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        for target in ("src.agents.task_recommend.request_local_proposal", "src.ui.settings.list_local_models",
                       "src.agents.task_orchestrator.start_task", "src.agents.task_orchestrator._run_step",
                       "src.agents.task_consent.grant"):
            mocker = patch(target, side_effect=AssertionError("메뉴 전환 중 호출 금지"))
            self.calls.append(mocker.start())
            self.addCleanup(mocker.stop)

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/ui/app.py"), default_timeout=30).run()

    def assert_no_calls(self, app):
        self.assertFalse(app.exception, [item.value for item in app.exception])
        for called in self.calls:
            called.assert_not_called()

    def test_default_and_settings_hide_legacy_tools(self):
        app = self.app()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        self.assertFalse(app.checkbox(key="ui_show_advanced_tools").value)
        self.assertFalse(any(item.key in ("ui_workflow_mode", "ui_proposal_scope") for item in app.selectbox))
        self.assertTrue(any(item.key == "task_goal" for item in app.text_area))
        app.radio(key="ui_task_screen").set_value("설정").run()
        self.assertTrue(any(item.label == "Ollama 모델명" for item in app.text_input))
        self.assertFalse(any("동작 모드" in item.label for item in app.selectbox))
        app.radio(key="ui_task_screen").set_value("작업").run()
        self.assert_no_calls(app)

    def test_advanced_tools_are_explicit_and_can_be_hidden(self):
        app = self.app()
        app.session_state["ui_draft"] = "보존할 기존 초안"
        app.checkbox(key="ui_show_advanced_tools").check().run()
        self.assertEqual(app.selectbox(key="ui_workflow_mode").options, ["통합 작업", "LLM 제안", "등록 모듈"])
        app.selectbox(key="ui_workflow_mode").select("LLM 제안").run()
        self.assertTrue(any(item.key == "ui_proposal_scope" for item in app.selectbox))
        app.selectbox(key="ui_workflow_mode").select("등록 모듈").run()
        self.assertTrue(any(item.key == "workflow_module_select" for item in app.selectbox))
        app.checkbox(key="ui_show_advanced_tools").uncheck().run()
        self.assertEqual(app.session_state["ui_workflow_mode"], "통합 작업")
        self.assertEqual(app.session_state["ui_draft"], "보존할 기존 초안")
        self.assertFalse(any(item.key == "workflow_module_select" for item in app.selectbox))
        self.assert_no_calls(app)


if __name__ == "__main__":
    unittest.main()

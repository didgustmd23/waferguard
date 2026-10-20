"""화면 뼈대의 상태·검증·미연결 경계를 검사한다. 모델·worker는 실행하지 않는다."""

from copy import deepcopy
import importlib
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.ui.state import SCREENS, default_settings, validate_draft, validate_settings
from src.ui.state import SCREEN_LOCKS, resolve_screen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_AVAILABLE = find_spec("streamlit") is not None


class UiSettingsTest(unittest.TestCase):
    def test_only_request_and_settings_are_available_before_service_connection(self):
        for screen in SCREENS:
            expected = screen if screen in (SCREENS[0], "설정") else SCREENS[0]
            self.assertEqual(resolve_screen(screen), expected)

        self.assertEqual(resolve_screen("unknown"), SCREENS[0])
        self.assertEqual(len(SCREEN_LOCKS), 4)

    def test_defaults_reuse_existing_config_and_do_not_mutate_it(self):
        first = default_settings()
        second = default_settings()
        first["model"]["model"] = "changed"
        self.assertNotEqual(first, second)
        self.assertEqual(validate_settings(second), second)

    def test_remote_cloud_and_invalid_limits_are_rejected(self):
        for field, value in (("host", "remote.invalid"), ("model", "test:cloud"), ("port", 0)):
            candidate = default_settings()
            candidate["model"][field] = value

            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_settings(candidate)

        for field, value in (("processes", 9), ("wall_seconds", 0), ("memory_mib", 32769)):
            candidate = default_settings()
            candidate["limits"][field] = value

            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_settings(candidate)

    def test_extra_fields_and_unsupported_modes_are_rejected(self):
        candidate = default_settings()
        candidate["api_key"] = "synthetic"

        with self.assertRaises(ValueError):
            validate_settings(candidate)

        candidate = default_settings()
        candidate["mode"] = "external"

        with self.assertRaises(ValueError):
            validate_settings(candidate)

    def test_draft_limits_and_controls(self):
        self.assertEqual(validate_draft("품질 검사만 요청합니다."), "품질 검사만 요청합니다.")

        for text in ("", " ", "x" * 4001, "가" * 3000, "x\x1by"):
            with self.subTest(length=len(text)), self.assertRaises(ValueError):
                validate_draft(text)


class UiShellRenderingTest(unittest.TestCase):
    """Streamlit 미설치 환경에서도 화면 함수의 서비스 미호출 경계를 검사한다."""

    def setUp(self):
        # 모델 설정 폼 검사가 실제 해설 정책 파일을 변경하지 않게 한다.
        policy_patcher = patch("src.ui.commentary_view.save_policy")
        policy_patcher.start()
        self.addCleanup(policy_patcher.stop)
        self.st = MagicMock()
        self.st.button.return_value = False
        self.st.session_state = {}
        self.submitted = False
        self.st.selectbox.side_effect = lambda label, options, **kwargs: options[kwargs.get("index", 0)]
        self.st.text_input.side_effect = lambda *args, **kwargs: kwargs.get("value", "")
        self.st.text_area.side_effect = lambda *args, **kwargs: kwargs.get("value", "")
        self.st.checkbox.side_effect = lambda *args, **kwargs: kwargs.get("value", False)
        self.st.number_input.side_effect = lambda *args, **kwargs: kwargs["value"]
        self.st.form_submit_button.side_effect = lambda *args, **kwargs: self.submitted

        # UI 의존성이 설치되지 않아도 테스트용 렌더러로만 import한다.
        with patch.dict(sys.modules, {"streamlit": self.st}):
            self.app = importlib.import_module("src.ui.app")

        patcher = patch.object(self.app, "st", self.st)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_all_screens_render_and_execution_controls_are_disabled(self):
        for screen in SCREENS:
            self.st.session_state["ui_screen"] = screen
            self.app.main()

        for call in self.st.button.call_args_list:
            key = call.kwargs.get("key", "")
            if key not in ("ui_step_1", "ui_settings_navigation"):
                self.assertTrue(call.kwargs.get("disabled"))
        self.assertEqual(self.st.session_state["ui_settings_revision"], 0)
        self.assertEqual(self.st.session_state["ui_draft"], "")

    def test_form_submission_only_stores_memory_draft(self):
        self.submitted = True
        self.st.text_area.side_effect = lambda *args, **kwargs: "품질 검사를 요청합니다."
        self.app.main()
        self.assertEqual(self.st.session_state["ui_draft"], "품질 검사를 요청합니다.")
        self.submitted = False
        self.st.session_state["ui_screen"] = SCREENS[1]
        self.app.main()
        self.assertEqual(self.st.session_state["ui_screen"], SCREENS[0])
        self.assertEqual(self.st.session_state["ui_draft"], "품질 검사를 요청합니다.")

    def test_settings_submit_validates_before_applying(self):
        self.app.main()
        before = deepcopy(self.st.session_state["ui_settings"])
        self.st.session_state["ui_screen"] = "설정"
        self.submitted = True
        self.st.text_input.side_effect = lambda label, **kwargs: "bad:cloud"
        self.app.main()
        self.assertEqual(self.st.session_state["ui_settings"], before)
        self.assertEqual(self.st.session_state["ui_settings_revision"], 0)
        self.st.error.assert_called()

        self.st.text_input.side_effect = lambda label, **kwargs: "synthetic:local"
        self.app.main()
        self.assertEqual(self.st.session_state["ui_settings"]["model"]["model"], "synthetic:local")
        self.assertEqual(self.st.session_state["ui_settings_revision"], 1)
        self.app.main()
        self.assertEqual(self.st.session_state["ui_settings_revision"], 1)

    def test_rerender_preserves_state_without_network_or_file_writes(self):
        self.app.main()
        before = deepcopy(self.st.session_state)

        with patch("http.client.HTTPConnection", side_effect=AssertionError("네트워크 금지")):
            with patch("subprocess.Popen", side_effect=AssertionError("프로세스 금지")):
                with patch("pathlib.Path.write_text", side_effect=AssertionError("파일 저장 금지")):
                    with patch("sqlite3.connect", side_effect=AssertionError("승인 DB 접근 금지")):
                        for screen in SCREENS:
                            self.st.session_state["ui_screen"] = screen
                            self.app.main()

        # 직접 지정한 잠긴 화면도 정규화되고 그 외 작업 상태는 보존된다.
        self.st.session_state["ui_screen"] = before["ui_screen"]
        self.assertEqual(self.st.session_state, before)

    def test_every_settings_field_has_help_including_disabled_policy(self):
        self.st.session_state["ui_screen"] = "설정"
        self.app.main()

        for widget in (self.st.selectbox, self.st.text_input, self.st.number_input, self.st.checkbox,
                       self.st.form_submit_button, self.st.button):
            self.assertTrue(widget.call_args_list)

            for call in widget.call_args_list:
                with self.subTest(label=call.args[0]):
                    self.assertTrue(call.kwargs.get("help", "").strip())

        self.assertTrue(all(call.kwargs.get("disabled") is True for call in self.st.button.call_args_list
                            if not call.kwargs.get("key", "").startswith("ui_")))


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit 설치 후 실제 AppTest를 실행하세요.")
class StreamlitAppTest(unittest.TestCase):
    def test_actual_settings_tooltips(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py")).run()
        app.button(key="ui_settings_navigation").click().run()
        self.assertEqual(len(app.exception), 0)

        for widget in (*app.selectbox, *app.text_input, *app.number_input, *app.checkbox):
            with self.subTest(label=widget.label):
                self.assertTrue(widget.proto.help.strip())

    def test_actual_streamlit_pages_and_rerun(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py"))
        app.run()
        self.assertEqual(len(app.exception), 0)

        for screen in SCREENS:
            app.session_state["ui_screen"] = screen
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(all(item.disabled for item in app.button
                                if "FormSubmitter" not in (item.key or "")
                                and item.key not in ("ui_step_1", "ui_settings_navigation")))
            self.assertEqual(app.session_state["ui_screen"], resolve_screen(screen))

    def test_actual_sidebar_navigation_and_draft_do_not_unlock_review(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "src/ui/app.py")).run()
        self.assertEqual(len(app.radio), 0)

        for number in range(2, 6):
            self.assertTrue(app.button(key=f"ui_step_{number}").disabled)

        app.text_area[0].set_value("결측률을 확인해 주세요.")
        next(item for item in app.button if "FormSubmitter" in (item.key or "")).click().run()
        self.assertTrue(app.button(key="ui_step_2").disabled)
        app.button(key="ui_settings_navigation").click().run()
        self.assertEqual(app.session_state["ui_screen"], "설정")
        app.button(key="ui_step_1").click().run()
        self.assertEqual(app.session_state["ui_screen"], SCREENS[0])
        self.assertEqual(app.text_area[0].value, "결측률을 확인해 주세요.")
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()

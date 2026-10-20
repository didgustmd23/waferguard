"""모듈 검토·미리보기·생성 버튼 상태를 합성 서비스로 검사한다."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.ui.creation_submit import creation_submission, mark_creation_pending
from src.ui.workflow_service import UiWorkflow
from tests.agents.test_ui_workflow import FakeService


class CreationSubmitTest(unittest.TestCase):
    def test_confirmation_pending_consumption_and_failure_unlock(self):
        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        st = SimpleNamespace(session_state={"confirm": False})
        mark_creation_pending(st, workflow, "create", "confirm")
        self.assertFalse(workflow.creation_busy)
        st.session_state["confirm"] = True
        mark_creation_pending(st, workflow, "create", "confirm")
        mark_creation_pending(st, workflow, "preview")
        self.assertEqual(workflow.creation_requested, "create")

        with self.assertRaises(RuntimeError):
            with creation_submission(workflow, "create"):
                self.assertIsNone(workflow.creation_requested)
                self.assertTrue(workflow.creation_busy)
                raise RuntimeError("synthetic")
        self.assertFalse(workflow.creation_busy)
        self.assertTrue(workflow._command_lock.acquire(blocking=False))
        workflow._command_lock.release()

    def test_parallel_command_lock_rejects_creation(self):
        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        mark_creation_pending(SimpleNamespace(session_state={}), workflow, "preview")
        workflow._command_lock.acquire()
        try:
            with self.assertRaises(ValueError):
                with creation_submission(workflow, "preview"):
                    self.fail("다른 요청 처리 중에는 생성하지 않습니다.")
            self.assertIsNone(workflow.creation_requested)
        finally:
            workflow._command_lock.release()


class CreationSubmitAppTest(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        for target, kwargs in (
            ("src.ui.workflow_service.UiWorkflow.service", {"return_value": self.service}),
            ("src.ui.runtime_notice.server_permissions", {"return_value": {"elevated": True, "pid": 1, "python": "synthetic"}}),
            ("src.ui.workflow_view.diagnose_paths", {"return_value": []}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def app(self, view):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import " + view + "\n" + view + "(st)")
        app.session_state["ui_workflow"] = self.workflow
        app.session_state["ui_settings"] = {"limits": {"wall_seconds": 300, "memory_mib": 2048, "processes": 2}}
        return app.run()

    def test_review_disables_and_guides_to_step_three(self):
        app = self.app("review")
        self.assertFalse(app.button(key="workflow_review").disabled)
        app.button(key="workflow_review").click().run()
        self.assertTrue(app.button(key="workflow_review").disabled)
        self.assertTrue(any("3 입력 선택" in item.value for item in app.success))

    def test_preview_and_create_are_disabled_before_backend_and_do_not_repeat(self):
        import streamlit as st

        self.workflow.reviewed = True
        disabled = {}
        original_button = st.form_submit_button
        original_preview, original_create = self.service.preview_form, self.service.create

        def button(label, **kwargs):
            disabled[label] = kwargs.get("disabled", False)
            return original_button(label, **kwargs)

        def preview(*args, **kwargs):
            self.assertTrue(disabled["생성 범위 검사·미리보기"])
            self.assertTrue(self.workflow.creation_busy)
            return original_preview(*args, **kwargs)

        def create(*args, **kwargs):
            self.assertTrue(disabled["생성 범위 검사·미리보기"])
            self.assertTrue(disabled["새 작업 준비"])
            self.assertTrue(self.workflow.creation_busy)
            return original_create(*args, **kwargs)

        with patch("streamlit.form_submit_button", side_effect=button), patch.object(
                self.service, "preview_form", side_effect=preview) as check, patch.object(
                self.service, "create", side_effect=create) as prepare:
            app = self.app("inputs")
            app.button(key="FormSubmitter:workflow_creation_wm_canonical_eda-생성 범위 검사·미리보기").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(self.workflow.creation_busy)
            app.run()
            check.assert_called_once()
            next(item for item in app.checkbox if item.label == "위 설정으로 새 작업과 배정 기록만 생성합니다").check()
            app.button(key="FormSubmitter:workflow_create_confirm-새 작업 준비").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(self.workflow.creation_busy)
            app.run()
            prepare.assert_called_once()

    def test_preview_failure_unlocks_and_does_not_leak_raw_error(self):
        self.workflow.reviewed = True
        with patch.object(self.service, "preview_form", side_effect=ValueError("PRIVATE_VALUE")) as check:
            app = self.app("inputs")
            app.button(key="FormSubmitter:workflow_creation_wm_canonical_eda-생성 범위 검사·미리보기").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="FormSubmitter:workflow_creation_wm_canonical_eda-생성 범위 검사·미리보기").disabled)
            self.assertTrue(app.error)
            self.assertFalse(any("PRIVATE_VALUE" in item.value for item in app.error))
            app.run()
            check.assert_called_once()

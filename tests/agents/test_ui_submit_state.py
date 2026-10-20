"""실행 클릭 직후 비활성화·명령 잠금·중복 제출 차단. 실제 worker는 실행하지 않는다."""

from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.ui.submit_state import mark_execution_pending, clear_unsubmitted_execution
from src.ui.workflow_service import UiWorkflow


class SubmitStateTest(unittest.TestCase):
    def test_callback_marks_only_confirmed_click_and_does_not_submit(self):
        controller = SimpleNamespace(busy=False, execution_requested=False, approval_id="keep", command=Mock())
        st = SimpleNamespace(session_state={"confirm": False})
        mark_execution_pending(st, controller, "confirm")
        self.assertFalse(controller.busy)
        st.session_state["confirm"] = True
        mark_execution_pending(st, controller, "confirm")
        self.assertTrue(controller.busy)
        self.assertTrue(controller.execution_requested)
        self.assertEqual(controller.approval_id, "keep")
        controller.command.assert_not_called()
        clear_unsubmitted_execution(controller)
        self.assertFalse(controller.busy)

        controller.busy = True
        clear_unsubmitted_execution(controller)
        self.assertTrue(controller.busy)  # 실제 실행 중 상태는 UI 예약 해제로 풀지 않는다.

    def test_wm_concurrent_execute_is_not_forwarded_twice(self):
        from tests.agents.test_ui_workflow import FakeService

        service = FakeService()
        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        started, release = Event(), Event()
        original = service.next

        def execute(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("synthetic wait")
            return original(*args, **kwargs)

        with patch.object(UiWorkflow, "service", return_value=service):
            workflow.connect("synthetic")
            workflow.command("next")
            workflow.command("preview")
            workflow.command("approve", confirmed=True)
            with patch.object(service, "next", side_effect=execute) as backend, ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(workflow.command, "execute", confirmed=True)
                try:
                    self.assertTrue(started.wait(5))
                    self.assertTrue(workflow.busy)
                    self.assertIsNone(workflow.approval_id)
                    self.assertEqual(workflow.command("execute", confirmed=True), {"status": "busy"})
                    self.assertEqual(backend.call_count, 1)
                finally:
                    release.set()
                future.result(timeout=5)

        self.assertFalse(workflow.busy)
        self.assertFalse(workflow.error)

    def test_csv_existing_command_lock_preserves_active_request(self):
        from src.ui.execution_service import UiExecution

        execution = UiExecution.__new__(UiExecution)
        execution._lock = Lock()
        execution.busy = True
        execution.approval_id = "keep"
        execution._lock.acquire()
        try:
            with patch("src.ui.execution_service.backend.next_step") as backend:
                self.assertEqual(execution.command("execute", confirmed=True), {"status": "busy"})
                backend.assert_not_called()
            self.assertTrue(execution.busy)
            self.assertEqual(execution.approval_id, "keep")
        finally:
            execution._lock.release()

    @unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 필요")
    def test_button_is_rendered_disabled_before_backend_call_and_rerun_does_not_execute(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from tests.agents.test_ui_workflow import FakeService

        service = FakeService()
        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        buttons, seen_busy = [], []
        real_button, real_next = st.form_submit_button, service.next

        def render_button(label, *args, **kwargs):
            if label == "승인하고 실행":
                buttons.append(kwargs.get("disabled"))
            return real_button(label, *args, **kwargs)

        def execute(*args, **kwargs):
            seen_busy.append(workflow.busy)
            self.assertFalse(workflow.execution_requested)
            self.assertTrue(buttons[-1])
            return real_next(*args, **kwargs)

        with patch.object(UiWorkflow, "service", return_value=service), patch(
                "src.ui.runtime_notice.server_permissions",
                return_value={"elevated": True, "pid": 1, "python": "synthetic"}), patch(
                "streamlit.form_submit_button", side_effect=render_button):
            workflow.connect("synthetic")
            app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import approval\napproval(st)")
            app.session_state["ui_workflow"] = workflow
            app.run()
            app.button(key="approval_flow_start").click().run()
            self.assertFalse(buttons[-1])
            with patch.object(service, "next", side_effect=execute) as backend:
                key = f"approval_flow_confirm_{workflow.name}_{workflow.state['stage']}_{workflow.state['revision']}"
                app.checkbox(key=key).check()
                app.button(key="FormSubmitter:approval_flow_approve-승인하고 실행").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(seen_busy, [True])
                backend.assert_called_once()
                self.assertEqual(workflow.state["status"], "completed")
                app.run()
                backend.assert_called_once()

        self.assertFalse(workflow.busy)


if __name__ == "__main__":
    unittest.main()

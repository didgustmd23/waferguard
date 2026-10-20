"""승인 경계 자동 진행. 실제 데이터·승인·worker는 사용하지 않는다."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.ui.approval_flow import ApprovalFlow, mark_flow_pending
from src.ui.workflow_service import UiWorkflow
from tests.agents.test_ui_workflow import FakeService


class ApprovalFlowTest(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        patcher = patch.object(UiWorkflow, "service", return_value=self.service)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.controller = UiWorkflow("wm_canonical_eda", "1.0.0")
        self.controller.name = "synthetic"
        self.flow = ApprovalFlow(self.controller, workflow=True)

    def test_start_stops_at_approval_and_single_approval_completes(self):
        self.flow.run("start")
        self.assertTrue(self.flow.ready)
        self.assertEqual(self.service.approval_calls, 0)
        self.assertEqual(self.service.next_calls, 1)
        self.flow.run("approve")
        self.assertEqual(self.flow.state["status"], "completed")
        self.assertEqual(self.service.approval_calls, 1)
        self.assertEqual(self.service.next_calls, 2)

    def test_next_approval_is_not_approved_automatically(self):
        original = self.service.next

        def advance(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs["approval_id"]:
                self.service.snapshot.update(status="running", stage="commit_split", revision=2,
                                             actions=[{"id": "next"}])
            elif kwargs["expected_stage"] == "commit_split":
                self.service.snapshot.update(status="awaiting_approval", stage="commit_split", revision=3,
                                             actions=[{"id": "approve"}])
            return result

        self.service.next = advance
        self.flow.run("start")
        self.flow.run("approve")
        self.assertTrue(self.flow.ready)
        self.assertEqual(self.flow.state["stage"], "commit_split")
        self.assertEqual(self.service.approval_calls, 1)

    def test_execution_failure_is_not_retried(self):
        self.flow.run("start")
        with patch.object(self.service, "next", side_effect=RuntimeError("synthetic")) as worker:
            with self.assertRaises(RuntimeError):
                self.flow.run("approve")
            with self.assertRaises(ValueError):
                self.flow.run("approve")
            worker.assert_called_once()
        self.assertFalse(self.flow.busy)
        self.assertIsNone(self.flow.pending)

    def test_blocked_input_stops_and_no_progress_is_not_retried(self):
        self.service.snapshot.update(status="blocked_input", actions=[])
        self.flow.run("start")
        self.assertEqual(self.service.next_calls, 0)

        self.service.snapshot.update(status="created", actions=[{"id": "next"}])
        flow = ApprovalFlow(self.controller, workflow=True)
        self.controller.state = None
        with patch.object(self.service, "next", return_value={"status": "running"}) as step:
            with self.assertRaises(ValueError):
                flow.run("start")
        step.assert_called_once()

    def test_changed_scope_existing_approval_and_lock_are_rejected(self):
        self.flow.run("start")
        self.controller.state["revision"] += 1
        with self.assertRaises(ValueError):
            self.flow.run("approve")
        self.assertEqual(self.service.approval_calls, 0)

        flow = ApprovalFlow(self.controller, workflow=True)
        self.controller.state["actions"] = [{"id": "execute"}]
        with self.assertRaises(ValueError):
            flow.run("start")

        flow._lock.acquire()
        try:
            self.assertEqual(flow.run("start"), {"status": "busy"})
        finally:
            flow._lock.release()

    def test_callback_marks_only_explicit_confirmed_click(self):
        st = SimpleNamespace(session_state={"confirm": False})
        mark_flow_pending(st, self.flow, "approve", "confirm")
        self.assertIsNone(self.flow.pending)
        st.session_state["confirm"] = True
        mark_flow_pending(st, self.flow, "approve", "confirm")
        self.assertEqual(self.flow.pending, "approve")
        self.assertTrue(self.flow.busy)
        self.assertEqual(self.service.approval_calls, 0)

    def test_csv_reuses_existing_commands(self):
        controller = Mock()
        controller.name, controller.error = "csv-demo", False
        snapshot = {"status": "created", "current_stage": "preflight", "revision": 0}
        calls = []

        def command(action, **kwargs):
            calls.append(action)
            if action == "preflight":
                snapshot.update(status="awaiting_approval", current_stage="run_pipeline", revision=1)
            elif action == "preview":
                controller.preview = {"summary": {"synthetic_only": True}}
            elif action == "execute":
                snapshot.update(status="completed", revision=3)
            return {"status": snapshot["status"]}

        controller.read_state.side_effect = lambda: SimpleNamespace(**snapshot)
        controller.command.side_effect = command
        flow = ApprovalFlow(controller, workflow=False)
        flow.run("start")
        self.assertEqual(calls, ["preflight", "preview"])
        flow.run("approve")
        self.assertEqual(calls, ["preflight", "preview", "approve", "execute"])


    def test_approval_contract_failure_never_calls_worker(self):
        self.flow.run("start")
        with patch.object(self.service, "record_approval", side_effect=ValueError("scope changed")), patch.object(
                self.service, "next") as worker:
            with self.assertRaises(ValueError):
                self.flow.run("approve")
            worker.assert_not_called()
        self.assertTrue(self.flow.error)

    def test_executor_busy_is_not_automatically_retried(self):
        with patch.object(self.service, "next", return_value={"status": "running", "code": "EXECUTOR_BUSY"}) as step:
            with self.assertRaises(ValueError):
                self.flow.run("start")
            with self.assertRaises(ValueError):
                self.flow.run("start")
            step.assert_called_once()


class ApprovalFlowAppTest(unittest.TestCase):
    def test_render_has_no_calls_and_approval_disables_before_execution(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st

        controller = UiWorkflow("wm_canonical_eda", "1.0.0")
        controller.name = "synthetic"
        service = FakeService()
        disabled = []
        original = st.form_submit_button

        def button(label, **kwargs):
            disabled.append(kwargs.get("disabled", False))
            return original(label, **kwargs)

        def execute(*args, **kwargs):
            self.assertTrue(disabled[-1])
            return real_next(*args, **kwargs)

        with patch.object(UiWorkflow, "service", return_value=service), patch(
                "src.ui.runtime_notice.server_permissions", return_value={"elevated": True, "pid": 1, "python": "synthetic"}), patch(
                "streamlit.form_submit_button", side_effect=button):
            app = AppTest.from_string("import streamlit as st\nfrom src.ui.approval_flow_view import render_approval_flow\nrender_approval_flow(st, st.session_state['controller'], workflow=True)")
            app.session_state["controller"] = controller
            app.run()
            self.assertEqual(service.next_calls, 0)
            app.button(key="approval_flow_start").click().run()
            self.assertFalse(app.exception)
            app.run()
            self.assertEqual(service.next_calls, 1)
            real_next = service.next
            with patch.object(service, "next", side_effect=execute):
                app.checkbox[0].check()
                app.button(key="FormSubmitter:approval_flow_approve-승인하고 실행").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(service.next_calls, 2)
            app.run()
            self.assertEqual(service.next_calls, 2)

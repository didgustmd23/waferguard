"""등록 모듈 공통 UI 검사. 합성 상태·모의 서비스만 사용하며 native worker는 실행하지 않는다."""

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from src.agents.wm_creation_schema import creation_arguments, creation_schema
from src.agents.wm_run_creation import CREATION_MODULES
from src.agents.workflow_approval import WorkflowApprovalPreview
from src.ui.workflow_service import UiWorkflow


class CreationSchemaTest(unittest.TestCase):
    def values(self, module_id):
        values = {field["name"]: field["default"] for field in creation_schema(module_id)["fields"]}
        values.update(bindings="synthetic-bindings.json", name="synthetic01")
        if "canonical_dir" in values:
            values["canonical_dir"] = "synthetic-canonical"
        return values

    def test_six_forms_convert_only_supported_arguments(self):
        for module_id in CREATION_MODULES:
            with self.subTest(module_id=module_id):
                result = creation_arguments(module_id, self.values(module_id))
                self.assertEqual(result["name"], "synthetic01")
                self.assertNotIn("batch_size", result)

        self.assertIn("canonical_options", creation_arguments("wm_canonical_eda", self.values("wm_canonical_eda")))
        self.assertIn("dataset_options", creation_arguments("wm_dataset_check", self.values("wm_dataset_check")))
        self.assertFalse(creation_schema("wm_conflict_candidate")["supported"])

    def test_unknown_fields_boolean_integer_nonfinite_and_empty_rejected(self):
        import math

        for changes in ({"injected": "code"}, {"max_output_bytes": True}, {"name": ""},
                        {"max_group_ratio": math.nan}, {"max_group_ratio": math.inf}):
            values = self.values("wm_canonical_eda") | changes
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                creation_arguments("wm_canonical_eda", values)

    def test_service_delegates_form_to_existing_creation(self):
        from src.agents.wm_workflow_service import WmWorkflowService
        from src.agents.worker_contract import WorkerLimits

        service = WmWorkflowService("wm_canonical_eda")
        with patch.object(service, "preview_creation") as preview:
            service.preview_form(self.values(service.module_id), WorkerLimits())
            self.assertIn("canonical_options", preview.call_args.kwargs)
            self.assertTrue(service.capabilities()["creation_form_ready"])


class FakeService:
    def __init__(self):
        self.snapshot = {"status": "created", "stage": "preflight", "revision": 0,
                         "actions": [{"id": "next"}], "runtime": {"supported": True}}
        self.status_calls = 0
        self.next_calls = 0
        self.approval_calls = 0
        self.create_calls = 0

    def status(self, name):
        self.status_calls += 1
        return deepcopy(self.snapshot)

    def capabilities(self):
        return {"creation_supported": True, "runtime": {"supported": True}}

    def creation_schema(self):
        return creation_schema("wm_canonical_eda")

    def preview_form(self, values, limits):
        preview = Mock()
        preview.packet = {"module_id": "wm_canonical_eda", "run_name": values["name"], "worker_started": False}
        return preview

    def create(self, preview, confirmed):
        self.create_calls += 1
        return {"run_name": preview.packet["run_name"], "bindings": "new-bindings.json", "worker_started": False}

    def preview_approval(self, name):
        return WorkflowApprovalPreview('{"summary":{"synthetic_only":true}}')

    def record_approval(self, name, preview, confirmed):
        self.approval_calls += 1
        self.snapshot["actions"] = [{"id": "execute"}]
        return {"approval_id": "synthetic-approval"}

    def next(self, name, expected_stage, expected_revision, approval_id):
        self.next_calls += 1
        if approval_id is None:
            self.snapshot.update(status="awaiting_approval", stage="run_eda", revision=1, actions=[{"id": "approve"}])
        else:
            self.snapshot.update(status="completed", revision=2, actions=[])
        return {"status": self.snapshot["status"], "worker_started": approval_id is not None}


class WorkflowControllerTest(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        patcher = patch.object(UiWorkflow, "service", return_value=self.service)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.workflow = UiWorkflow("wm_canonical_eda", "1.0.0")

    def test_no_implicit_calls_then_separate_approval_and_execute(self):
        self.assertEqual(self.service.status_calls, 0)
        self.workflow.connect("synthetic01")
        self.workflow.command("next")
        self.workflow.command("preview")
        with self.assertRaises(ValueError):
            self.workflow.command("approve")
        self.workflow.command("approve", confirmed=True)
        self.assertEqual(self.service.next_calls, 1)
        self.workflow.command("execute", confirmed=True)
        self.assertEqual(self.service.next_calls, 2)
        with self.assertRaises(ValueError):
            self.workflow.command("execute", confirmed=True)
        self.assertEqual(self.service.approval_calls, 1)

    def test_changed_creation_draft_and_repeated_creation_rejected(self):
        values = CreationSchemaTest().values("wm_canonical_eda")
        limits = {"wall_seconds": 60, "memory_mib": 2048, "processes": 2}
        self.workflow.preview_creation(values, limits)
        with self.assertRaises(ValueError):
            self.workflow.create(values | {"name": "changed"}, limits, confirmed=True)
        self.workflow.create(values, limits, confirmed=True)
        self.assertEqual(self.service.create_calls, 1)
        with self.assertRaises(ValueError):
            self.workflow.create(values, limits, confirmed=True)

    def test_failed_execution_does_not_retry_and_consumes_local_id(self):
        self.workflow.connect("synthetic01")
        self.workflow.command("next")
        self.workflow.command("preview")
        self.workflow.command("approve", confirmed=True)
        with patch.object(self.service, "next", side_effect=RuntimeError("synthetic")) as execute:
            with self.assertRaises(RuntimeError):
                self.workflow.command("execute", confirmed=True)
            with self.assertRaises(ValueError):
                self.workflow.command("execute", confirmed=True)
            self.assertEqual(execute.call_count, 1)
        self.assertIsNone(self.workflow.approval_id)

    def test_review_required_has_no_followup(self):
        self.service.snapshot.update(status="blocked_input", actions=[])
        self.workflow.connect("synthetic01")
        with self.assertRaises(ValueError):
            self.workflow.command("next")
        self.assertEqual(self.service.next_calls, 0)

    def test_navigation_flags_do_not_query_service(self):
        from src.ui.workflow_view import flags
        st = Mock(session_state={"ui_workflow": self.workflow})
        self.assertFalse(flags(st)["saved_available"])
        self.assertEqual(self.service.status_calls, 0)


class WorkflowResultsTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.csv"
        self.path.write_text("feature,value\nx,1\n", encoding="utf-8")
        self.workflow = Mock(name="workflow")
        self.workflow.name = "synthetic01"
        self.service = self.workflow.service.return_value
        self.result = {"status": "completed", "items": [
            {"kind": "table", "path": str(self.path), "label": "synthetic.csv"}]}
        self.service.results.return_value = self.result

    def test_verified_result_is_local_bounded_copy(self):
        from src.ui.workflow_results import load_workflow_results

        result = load_workflow_results(self.workflow)
        self.assertEqual(result["files"][0]["data"], self.path.read_bytes())
        self.assertEqual(self.service.results.call_count, 2)

    def test_changed_result_between_checks_rejected(self):
        from src.ui.workflow_results import load_workflow_results

        self.service.results.side_effect = [self.result, self.result | {"status": "blocked_input"}]
        with self.assertRaises(ValueError):
            load_workflow_results(self.workflow)

    def test_size_limit_rejects_before_display_read(self):
        from src.ui.workflow_results import load_workflow_results

        with patch("src.ui.workflow_results._fingerprint", return_value={"size": 9 * 1024 ** 2}), \
                patch("src.ui.workflow_results._read_file") as read:
            with self.assertRaises(ValueError):
                load_workflow_results(self.workflow)
            read.assert_not_called()


@unittest.skipUnless(find_spec("streamlit"), "Streamlit 미설치")
class WorkflowAppTest(unittest.TestCase):
    def test_same_five_screens_and_rerender_never_executes(self):
        from streamlit.testing.v1 import AppTest
        from src.ui.state import SCREENS

        service = FakeService()
        with patch.object(UiWorkflow, "service", return_value=service):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/ui/app.py"), default_timeout=30).run()
            app.selectbox(key="ui_workflow_mode").select("등록 모듈").run()
            self.assertFalse(app.exception)
            app.selectbox(key="workflow_module_select").select("wm_canonical_eda")
            app.button(key="FormSubmitter:workflow_select-이 모듈 선택").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(service.status_calls, 0)
            app.button(key="ui_step_2").click().run()
            app.button(key="workflow_review").click().run()
            app.button(key="ui_step_3").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len([b for b in app.button if b.key and b.key.startswith("ui_step_")]), 5)
            self.assertEqual(app.session_state["ui_screen"], SCREENS[2])
            self.assertTrue(any(item.label == "canonical 폴더" for item in app.text_input))
            app.run()
            self.assertEqual(service.next_calls, 0)
            self.assertEqual(service.approval_calls, 0)
            self.assertEqual(service.create_calls, 0)
            self.assertEqual(service.status_calls, 0)

    def test_approval_and_execution_are_separate_submissions(self):
        from streamlit.testing.v1 import AppTest
        from src.ui.state import SCREENS

        service = FakeService()
        workflow = UiWorkflow("wm_canonical_eda", "1.0.0")
        with patch.object(UiWorkflow, "service", return_value=service):
            workflow.connect("synthetic01")
            app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src/ui/app.py"), default_timeout=30).run()
            app.session_state["ui_workflow_mode"] = "등록 모듈"
            app.session_state["ui_workflow"] = workflow
            app.session_state["ui_screen"] = SCREENS[3]
            app.run()
            self.assertFalse(app.exception)
            calls = service.status_calls
            app.run()
            self.assertEqual(service.status_calls, calls)
            app.button(key="workflow_next").click().run()
            app.button(key="workflow_preview").click().run()
            app.checkbox[0].check()
            app.button(key="FormSubmitter:workflow_approve-이번 단계 승인").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(service.approval_calls, 1, [e.value for e in app.error])
            self.assertEqual(service.next_calls, 1)
            app.run()
            self.assertEqual(service.approval_calls, 1)
            app.checkbox[0].check()
            app.button(key="FormSubmitter:workflow_execute-승인 단계 실행").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(service.next_calls, 2)
            app.run()
            self.assertEqual(service.next_calls, 2)


if __name__ == "__main__":
    unittest.main()

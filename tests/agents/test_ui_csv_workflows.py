"""별도 화면 없이 등록 명세로 CSV 다섯 작업을 표시한다."""

import unittest
from unittest.mock import patch

from src.agents.workflow_service_registry import CSV_IDS
from src.ui.workflow_service import UiWorkflow


class CsvWorkflowUiTest(unittest.TestCase):
    def test_all_creation_forms_render_without_original_data_access(self):
        from streamlit.testing.v1 import AppTest

        with patch("src.ui.runtime_notice.server_permissions", return_value={"elevated": True, "pid": 1, "python": "synthetic"}):
            for module in CSV_IDS:
                with self.subTest(module=module):
                    workflow = UiWorkflow(module, "1.0.0")
                    workflow.reviewed = True
                    app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import inputs\ninputs(st)")
                    app.session_state["ui_workflow"] = workflow
                    app.session_state["ui_settings"] = {"limits": {"wall_seconds": 300, "memory_mib": 2048, "processes": 2}}
                    with patch("src.agents.csv_workflow_service.CsvWorkflowService.preview_form") as preview:
                        app.run()
                        self.assertFalse(app.exception)
                        self.assertTrue(any(button.label == "생성 범위 검사·미리보기" for button in app.button))
                        preview.assert_not_called()
                        if module == "csv_local_handoff":
                            self.assertFalse(any(item.label == "메모리 (MiB)" for item in app.number_input))

    def test_unsupported_creation_receipt_disables_execution_start(self):
        from streamlit.testing.v1 import AppTest

        workflow = UiWorkflow("csv_split_review", "1.0.0")
        workflow.name = "synthetic"
        workflow.creation_receipt = {"runtime": {"supported": False, "notice": "지원되지 않는 환경"}}
        app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import approval\napproval(st)")
        app.session_state["ui_workflow"] = workflow
        with patch("src.ui.runtime_notice.server_permissions", return_value={"elevated": True, "pid": 1, "python": "synthetic"}), patch(
                "src.agents.csv_workflow_service.CsvWorkflowService.next") as execute:
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="approval_flow_start").disabled)
            execute.assert_not_called()

    def test_supported_creation_receipt_enables_start_without_execution(self):
        from streamlit.testing.v1 import AppTest

        workflow = UiWorkflow("csv_split_review", "1.0.0")
        workflow.name = "synthetic"
        workflow.creation_receipt = {"runtime": {"supported": True, "notice": "Windows Python 3.12 지원"}}
        app = AppTest.from_string("import streamlit as st\nfrom src.ui.workflow_view import approval\napproval(st)")
        app.session_state["ui_workflow"] = workflow

        with patch("src.ui.runtime_notice.server_permissions", return_value={"elevated": True, "pid": 1, "python": "synthetic"}), patch(
                "src.agents.csv_workflow_service.CsvWorkflowService.next") as execute:
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="approval_flow_start").disabled)
            execute.assert_not_called()

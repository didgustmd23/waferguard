"""CSV 실행안의 저장·재연결·공통 UI. 모델·네이티브 OS 운송은 모의한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.agents import llm_workflow_store as store
from src.agents.llm_workflow_binding import bind_workflow_step
from src.ui.csv_plan_service import CsvPlanDialogue, CsvPlanSession
from src.ui.state import default_settings
from src.ui.model_commands import ModelCommandLedger
from src.ui.approval_flow import ApprovalFlow
from tests.agents import test_llm_workflow_binding as fixtures


ROOT = Path(__file__).resolve().parents[2]


class CsvPlanStoreTest(unittest.TestCase):
    setUp = fixtures.WorkflowBindingTest.setUp
    start_patch = fixtures.WorkflowBindingTest.start_patch
    transport = fixtures.WorkflowBindingTest.transport
    finish = fixtures.WorkflowBindingTest.finish
    selection = fixtures.WorkflowBindingTest.selection

    def saved(self):
        self.selection()
        self.store_root = self.root / "plans"
        return store.save_workflow("plan01", self.proposal, self.context, root=self.store_root)

    def test_save_restore_has_no_model_worker_or_approval_authority(self):
        saved = self.saved()
        plan = CsvPlanSession(saved)

        with patch("http.client.HTTPConnection") as model:
            plan.restore()

        model.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertEqual(plan.index, 0)
        self.assertIsNone(plan.active.approval_id)
        self.assertIsNone(plan.active.name)

        with self.assertRaises(FileExistsError):
            store.save_workflow("plan01", self.proposal, self.context, root=self.store_root)

    def test_created_work_reconnects_without_consuming_approval_or_running(self):
        plan = CsvPlanSession(self.saved())
        plan.restore()
        plan.active.preview_creation(self.values[0], self.limits.__dict__)
        result = plan.active.create(self.values[0], self.limits.__dict__, confirmed=True)
        self.assertFalse(result["worker_started"])
        restored = CsvPlanSession(store.load_workflow("plan01", root=self.store_root))
        restored.restore()
        self.assertEqual(restored.active.name, "split01")
        self.assertEqual(restored.active.state["stage"], "preflight")
        self.assertIsNone(restored.active.approval_id)
        self.assertFalse(restored.active.execution_requested)
        self.assertEqual(self.calls, [])

        with self.assertRaises(ValueError):
            restored.advance()

    def test_unknown_creation_intent_never_retries(self):
        saved = self.saved()
        binding = bind_workflow_step(self.proposal, **self.context, step_id=self.steps[0]["step_id"],
                                    values=self.values[0], limits=self.limits)
        store.begin_creation(saved, binding, root=self.store_root)

        with self.assertRaises(FileExistsError):
            store.begin_creation(saved, binding, root=self.store_root)

        with self.assertRaises(ValueError):
            CsvPlanSession(saved).restore()

        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "split/split01").exists())

    def test_modified_saved_plan_code_rejected(self):
        saved = self.saved()

        with patch.object(store, "binding_code_id", return_value="different"):
            with self.assertRaises(ValueError):
                store.load_workflow("plan01", root=self.store_root)

        path = Path(saved["directory"]) / "plan.json"
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet["selection"]["proposal"]["plan_signature"] = "0" * 64
        path.write_text(json.dumps(packet), encoding="utf-8")

        with self.assertRaises(ValueError):
            store.load_workflow("plan01", root=self.store_root)

    def test_terminal_or_forged_completion_never_advances_on_restore(self):
        saved = self.saved()
        plan = CsvPlanSession(saved)
        plan.restore()
        plan.active.preview_creation(self.values[0], self.limits.__dict__)
        plan.active.create(self.values[0], self.limits.__dict__, confirmed=True)
        _, tool, _ = plan.active.service()._open("split01")

        for status, blockers in (("blocked_input", []), ("failed_runtime", []), ("cancelled", []),
                                 ("blocked_input", ["SECURITY_LOCKED"])):
            state = tool.states.load()
            state.status = status
            state.blockers = blockers
            tool.states.save(state, expected_revision=state.revision)
            restored = CsvPlanSession(saved)
            restored.restore()
            self.assertEqual(restored.active.state["status"], status)

            with self.assertRaises(ValueError):
                restored.advance()

        state = tool.states.load()
        state.status = "completed"
        state.blockers = []
        tool.states.save(state, expected_revision=state.revision)

        with self.assertRaises(ValueError):
            CsvPlanSession(saved).restore()

        self.assertEqual(self.calls, [])

    def test_changed_settings_reject_pending_model_send(self):
        self.saved()
        settings = default_settings()
        dialogue = CsvPlanDialogue(settings, self.context, ledger=ModelCommandLedger(self.root / "ledger.db"))
        self.addCleanup(dialogue.close)
        pending = dialogue.prepare("선택 규칙으로 분할부터 적용까지 제안하세요.", settings)
        changed = json.loads(json.dumps(settings))
        changed["model"]["num_ctx"] += 512

        with patch("src.agents.langchain_local.request_local_proposal") as model:
            with self.assertRaises(ValueError):
                dialogue.send(pending["command_id"], changed, confirmed=True)

        model.assert_not_called()

    def test_receipt_run_identity_tamper_rejected(self):
        saved = self.saved()
        plan = CsvPlanSession(saved)
        plan.restore()
        plan.active.preview_creation(self.values[0], self.limits.__dict__)
        plan.active.create(self.values[0], self.limits.__dict__, confirmed=True)
        path = Path(saved["directory"]) / (self.steps[0]["step_id"] + ".created.json")
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet["identity"]["run_id"] = "0" * 32
        path.write_text(json.dumps(packet), encoding="utf-8")

        with self.assertRaises(ValueError):
            CsvPlanSession(saved).restore()

    def test_full_plan_common_approval_and_verified_restore(self):
        saved = self.saved()
        plan = CsvPlanSession(saved)
        plan.restore()

        for index in range(5):
            active = plan.active
            active.preview_creation(self.values[index], self.limits.__dict__)

            if index == 3:
                self.assertIsNotNone(active.candidate_report)
                self.assertIn("후보 보고서", active.creation_confirmation_label)

            active.create(self.values[index], self.limits.__dict__, confirmed=True)
            self.finish(ApprovalFlow(active, workflow=True))

            if index < 4:
                plan.advance()
                self.assertIsNone(plan.active.name)

                with self.assertRaises(ValueError):
                    plan.advance()

        before = list(self.calls)
        restored = CsvPlanSession(store.load_workflow("plan01", root=self.store_root))
        restored.restore()
        self.assertEqual(restored.index, 4)
        self.assertEqual(restored.active.state["status"], "completed")
        self.assertIsNone(restored.active.approval_id)
        self.assertEqual(before, self.calls)


class CsvPlanScreenTest(unittest.TestCase):
    def button(self, app, label):
        return next(item for item in app.button if item.label == label)

    def test_scope_preview_send_save_and_rerun(self):
        example = json.loads((ROOT / "configs/agents/llm/workflow_proposal_example.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_save = store.save_workflow
            original_load = store.load_workflow

            def save(name, proposal, context):
                return original_save(name, proposal, context, root=root / "plans")

            def load(name, **kwargs):
                return original_load(name, root=kwargs.get("root", root / "plans"))

            with patch("src.agents.langchain_local.request_local_proposal", return_value=json.dumps(example)) as model:
                with patch.object(store, "save_workflow", side_effect=save), patch.object(store, "load_workflow", side_effect=load):
                    with patch("src.ui.dialogue_service.ModelCommandLedger", side_effect=lambda: ModelCommandLedger(root / "commands.db")):
                        app = AppTest.from_file(str(ROOT / "src/ui/app.py"), default_timeout=30).run()
                        next(item for item in app.selectbox if item.label == "제안 범위").select("CSV 분할·전처리").run()
                        self.assertFalse(app.exception)
                        self.assertTrue(self.button(app, "2 제안 검토 · 잠김").disabled)
                        self.button(app, "이 규칙으로 새 대화 준비").click().run()
                        next(item for item in app.text_area if item.label == "CSV 요청 / 확인 질문 답변").set_value("선택 규칙으로 분할부터 적용까지 제안하세요.")
                        self.button(app, "CSV 전송 내용 미리보기").click().run()
                        model.assert_not_called()
                        next(item for item in app.checkbox if item.label == "위 공개 규칙과 문장을 로컬 모델에 전달합니다").check()
                        self.button(app, "CSV 확인한 내용 전송").click().run()
                        self.assertFalse(app.exception)
                        self.assertEqual(model.call_count, 1)
                        app.run()
                        self.assertEqual(model.call_count, 1)
                        self.button(app, "2 제안 검토").click().run()
                        next(item for item in app.checkbox if item.label == "제안 순서·규칙을 검토했으며 로컬 저장에 동의합니다").check()
                        self.button(app, "검토한 CSV 실행안 저장").click().run()
                        self.assertFalse(app.exception)
                        self.button(app, "3 입력 선택").click().run()
                        self.assertFalse(app.exception)
                        self.assertTrue(any(item.label == "원본 CSV" for item in app.text_input))
                        app.run()
                        self.assertEqual(model.call_count, 1)
                        self.button(app, "1 요청·대화").click().run()
                        next(item for item in app.text_input if item.label == "저장된 CSV 실행안 이름").set_value("csv_plan01")
                        self.button(app, "CSV 실행안 불러오기").click().run()
                        self.assertFalse(app.exception)
                        self.assertEqual(model.call_count, 1)
                        self.assertIsNone(app.session_state["ui_csv_plan"].active.approval_id)


if __name__ == "__main__":
    unittest.main()

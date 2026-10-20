"""모델·worker를 실행하지 않고 UI의 전송 확인·단일 소비·제안 잠금을 검사한다."""

from copy import deepcopy
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.agents.llm_config import LocalModelConfig
from src.agents.llm_proposal import ALLOWED, SOURCE, build_request_payload
from src.agents.llm_proposal_contract import validate_proposal
from src.agents.llm_proposal_store import list_saved_proposals, save_proposal
from src.ui.dialogue_service import UiDialogue
from src.ui.model_commands import ModelCommandLedger
from src.ui.state import SCREENS, default_settings
from tests.agents.test_llm_proposal import valid_content


ROOT = Path(__file__).resolve().parents[2]


def checked_result():
    result = validate_proposal(valid_content(), expected_input=dict(SOURCE), allowed_modules=ALLOWED).to_dict()
    return dict(result, status="proposal_validated")


def mock_dialogue(config):
    dialogue = Mock()
    dialogue.preview.side_effect = lambda text: {
        "endpoint": config.endpoint, "payload": build_request_payload(text, config),
        "model_called": False, "worker_started": False, "execution_approved": False,
    }
    dialogue.send.return_value = checked_result()
    return dialogue


class UiDialogueTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "commands.sqlite3"
        self.settings = default_settings()
        self.backend = mock_dialogue(LocalModelConfig(**self.settings["model"]))
        self.dialogue = UiDialogue(self.settings, factory=lambda config: self.backend,
                                   ledger=ModelCommandLedger(self.path))
        self.addCleanup(self.dialogue.close)

    def prepare(self):
        return self.dialogue.prepare("품질 검사 후 결측률을 확인해 주세요.", self.settings)

    def send(self, pending):
        return self.dialogue.send(pending["command_id"], self.settings, confirmed=True)

    def test_preview_and_unconfirmed_submission_never_call_model_or_write_ledger(self):
        pending = self.prepare()
        self.assertIn("품질 검사", pending["preview"]["payload"]["messages"][1]["content"])
        self.assertFalse(self.path.exists())

        with self.assertRaises(ValueError):
            self.dialogue.send(pending["command_id"], self.settings)

        self.backend.send.assert_not_called()

    def test_duplicate_command_is_never_sent_twice_and_gate_requires_validated_result(self):
        self.assertIsNone(self.dialogue.validated_proposal(self.settings))
        pending = self.prepare()
        self.assertEqual(self.send(pending)["status"], "proposal_validated")

        with self.assertRaises(ValueError):
            self.send(pending)

        self.backend.send.assert_called_once()
        checked = self.dialogue.validated_proposal(self.settings)
        self.assertIsNotNone(checked)
        self.assertFalse(checked["execution_approved"])
        self.assertFalse(checked["worker_started"])
        checked["plan_signature"] = "changed"
        self.assertIsNotNone(self.dialogue.validated_proposal(self.settings))

    def test_changed_settings_and_changed_preview_invalidate_confirmation(self):
        pending = self.prepare()
        changed = deepcopy(self.settings)
        changed["model"]["temperature"] = 0.7

        with self.assertRaises(ValueError):
            self.dialogue.send(pending["command_id"], changed, confirmed=True)

        self.dialogue.pending["preview"]["payload"]["model"] = "tampered"

        with self.assertRaises(ValueError):
            self.send(pending)

        self.backend.send.assert_not_called()

    def test_invalid_response_never_unlocks_review_or_retries(self):
        self.backend.send.return_value = {"proposal": {"approved": True}}
        pending = self.prepare()

        with self.assertRaises(ValueError):
            self.send(pending)

        self.assertEqual(self.dialogue.phase, "failed")
        self.assertIsNone(self.dialogue.validated_proposal(self.settings))

        with self.assertRaises(ValueError):
            self.send(pending)

        self.backend.send.assert_called_once()

    def test_failed_or_interrupted_call_stays_consumed(self):
        for failure in (RuntimeError("sensitive raw exception"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                # 실패한 대화 자체는 다시 준비할 수 없으므로 새 세션을 만든다.
                dialogue = UiDialogue(self.settings, factory=lambda config: self.backend,
                                      ledger=ModelCommandLedger(self.path))
                self.addCleanup(dialogue.close)
                pending = dialogue.prepare("결측률만 확인해 주세요.", self.settings)
                self.backend.send.side_effect = failure

                with self.assertRaises(type(failure)):
                    dialogue.send(pending["command_id"], self.settings, confirmed=True)

                self.assertIsNone(dialogue.validated_proposal(self.settings))
                self.assertFalse(ModelCommandLedger(self.path).claim(pending["command_id"], pending["binding"]))
                self.assertNotIn("sensitive", dialogue.error)

    def test_ledger_failure_prevents_network_and_busy_does_not_send(self):
        pending = self.prepare()
        self.dialogue._lock.acquire()

        try:
            self.assertEqual(self.send(pending)["status"], "busy")
        finally:
            self.dialogue._lock.release()

        with patch.object(self.dialogue._ledger, "claim", side_effect=sqlite3.OperationalError):
            with self.assertRaises(sqlite3.OperationalError):
                self.send(pending)

        self.backend.send.assert_not_called()

    def test_existing_pending_command_is_not_recovered_or_sent_again(self):
        pending = self.prepare()
        self.assertTrue(ModelCommandLedger(self.path).claim(pending["command_id"], pending["binding"]))
        self.assertEqual(self.send(pending)["status"], "already_consumed")
        self.backend.send.assert_not_called()

    def test_ledger_contains_no_prompt_or_response(self):
        self.send(self.prepare())

        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("SELECT * FROM model_commands").fetchone()

        self.assertEqual(row[2], "completed")
        self.assertNotIn("품질", str(row))
        self.assertNotIn("pipeline", str(row))

    def test_independent_ledger_connections_claim_same_command_only_once(self):
        # 테이블을 먼저 준비한 뒤 서로 다른 DB 연결의 동시 INSERT를 비교한다.
        ModelCommandLedger(self.path).claim("setup", "setup")

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: ModelCommandLedger(self.path).claim("same", "digest"), range(2)))

        self.assertEqual(sorted(results), [False, True])

    def test_tampered_result_and_changed_model_cannot_unlock_review(self):
        self.send(self.prepare())
        self.dialogue.result["plan_signature"] = "tampered"
        self.assertIsNone(self.dialogue.validated_proposal(self.settings))
        self.dialogue.result = checked_result()
        changed = deepcopy(self.settings)
        changed["model"]["model"] = "another:local"
        self.assertIsNone(self.dialogue.validated_proposal(changed))

    def test_clarification_has_no_review_access_and_requires_another_confirmation(self):
        proposal = {"schema_version": "llm_proposal_v1", "status": "needs_clarification",
                    "pipeline": None, "questions": ["어떤 검사가 필요하신가요?"]}
        result = validate_proposal(json.dumps(proposal), expected_input=dict(SOURCE), allowed_modules=ALLOWED).to_dict()
        self.backend.send.return_value = dict(result, status="awaiting_answer")
        self.send(self.prepare())
        self.assertIsNone(self.dialogue.validated_proposal(self.settings))
        self.assertIsNone(self.dialogue.pending)
        self.backend.send.return_value = checked_result()
        pending = self.dialogue.prepare("결측 검사입니다.", self.settings)
        self.assertEqual(self.backend.send.call_count, 1)
        self.send(pending)
        self.assertEqual(self.backend.send.call_count, 2)


class UiSavedRestoreTest(unittest.TestCase):
    """저장 제안 불러오기가 모델·명령 기록 없이 기존 복원 검사만 거치는지 확인한다."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "proposals"
        self.ledger_path = Path(temporary.name) / "commands.sqlite3"
        self.settings = default_settings()
        self.backend = mock_dialogue(LocalModelConfig(**self.settings["model"]))
        save_proposal("saved_one", checked_result(), root=self.root)

    def dialogue(self):
        dialogue = UiDialogue(self.settings, factory=lambda config: self.backend,
                              ledger=ModelCommandLedger(self.ledger_path), proposal_root=self.root)
        self.addCleanup(dialogue.close)
        return dialogue

    def test_restore_unlocks_review_without_model_or_ledger(self):
        dialogue = self.dialogue()
        receipt = dialogue.restore_saved("saved_one", self.settings)
        self.assertEqual(receipt["status"], "proposal_restored")
        self.assertFalse(receipt["execution_approved"])
        self.assertFalse(receipt["worker_started"])
        self.assertEqual(dialogue.phase, "proposal_validated")
        self.assertIsNotNone(dialogue.validated_proposal(self.settings))
        self.assertEqual(dialogue.saved_proposal(self.settings)["name"], "saved_one")
        self.assertIsNone(dialogue.pending)
        self.backend.preview.assert_not_called()
        self.backend.send.assert_not_called()

    def test_tampered_missing_or_changed_settings_keep_new_state(self):
        path = self.root / "saved_one.json"
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["proposal"]["pipeline"]["steps"][0]["module_id"] = "unknown_module"
        (self.root / "tampered.json").write_text(json.dumps(envelope), encoding="utf-8")
        changed = deepcopy(self.settings)
        changed["model"]["model"] = "another:local"

        for name, settings in (("tampered", self.settings), ("missing_one", self.settings),
                               ("saved_one", changed)):
            dialogue = self.dialogue()

            with self.subTest(name=name), self.assertRaises(ValueError):
                dialogue.restore_saved(name, settings)

            self.assertEqual(dialogue.phase, "new")
            self.assertIsNone(dialogue.validated_proposal(self.settings))
            self.assertIsNone(dialogue.saved)

        self.backend.send.assert_not_called()

    def test_restore_only_into_fresh_dialogue(self):
        dialogue = self.dialogue()
        dialogue.prepare("품질 검사 후 결측률을 확인해 주세요.", self.settings)

        with self.assertRaises(ValueError):
            dialogue.restore_saved("saved_one", self.settings)

        restored = self.dialogue()
        restored.restore_saved("saved_one", self.settings)

        with self.assertRaises(ValueError):
            restored.restore_saved("saved_one", self.settings)


@unittest.skipUnless(find_spec("langgraph"), "LangGraph 설치 환경에서 실제 대화 그래프를 검사하세요.")
class UiRealDialogueTest(unittest.TestCase):
    def test_real_graph_clarification_and_proposal_use_existing_unobserved_transport(self):
        from src.agents.llm_dialogue import ProposalDialogue

        question = json.dumps({"schema_version": "llm_proposal_v1", "status": "needs_clarification",
                               "pipeline": None, "questions": ["어떤 검사를 원하시나요?"]})
        settings = default_settings()

        with TemporaryDirectory() as directory:
            dialogue = UiDialogue(settings, factory=ProposalDialogue,
                                  ledger=ModelCommandLedger(Path(directory) / "commands.sqlite3"))
            self.addCleanup(dialogue.close)

            with patch("src.agents.langchain_local.request_local_proposal", side_effect=[question, valid_content()]) as model:
                first = dialogue.prepare("검사 종류를 정하지 않았습니다.", settings)
                dialogue.send(first["command_id"], settings, confirmed=True)
                self.assertEqual(dialogue.phase, "awaiting_answer")
                second = dialogue.prepare("품질 검사 후 결측률을 확인해 주세요.", settings)
                self.assertIn("어떤 검사를 원하시나요?", second["preview"]["payload"]["messages"][1]["content"])
                dialogue.send(second["command_id"], settings, confirmed=True)
                self.assertIsNotNone(dialogue.validated_proposal(settings))
                self.assertEqual(model.call_count, 2)


@unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 후 실제 UI 연결 테스트를 실행하세요.")
class UiDialogueAppTest(unittest.TestCase):
    def test_failed_call_is_sanitized_and_rerender_does_not_retry(self):
        from streamlit.testing.v1 import AppTest

        settings = default_settings()
        backend = mock_dialogue(LocalModelConfig(**settings["model"]))
        backend.send.side_effect = RuntimeError("PRIVATE_EXCEPTION_CONTENT")

        with TemporaryDirectory() as directory:
            ledger = ModelCommandLedger(Path(directory) / "commands.sqlite3")

            with patch("src.ui.dialogue_view.UiDialogue", side_effect=lambda settings: UiDialogue(
                    settings, factory=lambda config: backend, ledger=ledger)):
                app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                app.selectbox(key="ui_source_mode_widget").set_value("데이터 없이 제안만 검토").run()
                app.text_area[0].set_value("품질 검사를 요청합니다.")
                next(item for item in app.button if item.label == "요청 초안 보관").click().run()
                app.button(key="ui_preview_request").click().run()
                app.checkbox[0].check()
                next(item for item in app.button if item.label == "확인한 내용 전송").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertNotIn("PRIVATE_EXCEPTION_CONTENT", " ".join(error.value for error in app.error))
                app.run()
                self.assertTrue(app.button(key="ui_step_2").disabled)
                backend.send.assert_called_once()

    def test_explicit_preview_consent_and_rerender_never_repeat_model_call(self):
        from streamlit.testing.v1 import AppTest

        settings = default_settings()
        backend = mock_dialogue(LocalModelConfig(**settings["model"]))
        with TemporaryDirectory() as directory:
            ledger = ModelCommandLedger(Path(directory) / "commands.sqlite3")

            def factory(settings):
                return UiDialogue(settings, factory=lambda config: backend, ledger=ledger)

            with patch("src.ui.dialogue_view.UiDialogue", side_effect=factory):
                app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                app.selectbox(key="ui_source_mode_widget").set_value("데이터 없이 제안만 검토").run()
                app.text_area[0].set_value("품질 검사 후 결측률을 확인해 주세요.")
                next(item for item in app.button if item.label == "요청 초안 보관").click().run()
                app.button(key="ui_preview_request").click().run()
                backend.send.assert_not_called()
                next(item for item in app.button if item.label == "확인한 내용 전송").click().run()
                backend.send.assert_not_called()
                app.checkbox[0].check()
                next(item for item in app.button if item.label == "확인한 내용 전송").click().run()
                backend.send.assert_called_once()
                self.assertFalse(app.button(key="ui_step_2").disabled)
                self.assertTrue(app.button(key="ui_step_3").disabled)
                app.button(key="ui_step_2").click().run()
                self.assertEqual(app.session_state["ui_screen"], SCREENS[1])
                app.run()
                self.assertEqual(len(app.exception), 0)
                backend.send.assert_called_once()

    def test_saved_list_empty_refresh_and_explicit_restore_never_call_model(self):
        from streamlit.testing.v1 import AppTest

        settings = default_settings()
        backend = mock_dialogue(LocalModelConfig(**settings["model"]))

        with TemporaryDirectory() as directory:
            root = Path(directory) / "proposals"
            ledger = ModelCommandLedger(Path(directory) / "commands.sqlite3")

            def factory(settings):
                return UiDialogue(settings, factory=lambda config: backend, ledger=ledger, proposal_root=root)

            with patch("src.ui.dialogue_view.UiDialogue", side_effect=factory), \
                    patch("src.ui.dialogue_view.list_saved_proposals", side_effect=lambda: list_saved_proposals(root=root)):
                app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                app.selectbox(key="ui_source_mode_widget").set_value("데이터 없이 제안만 검토").run()
                self.assertTrue(any("저장된 제안이 없습니다" in item.value for item in app.info))
                self.assertTrue(next(item for item in app.button if item.label == "불러오기").disabled)

                # 화면 갱신만으로는 새 파일을 읽지 않고, 새로고침을 눌러야 목록에 나온다.
                save_proposal("saved_one", checked_result(), root=root)
                app.run()
                self.assertTrue(next(item for item in app.button if item.label == "불러오기").disabled)
                next(item for item in app.button if item.label == "목록 새로고침").click().run()
                choice = next(item for item in app.selectbox if item.label == "저장된 제안")
                self.assertEqual(list(choice.options)[0].split(" · ")[0], "saved_one")
                self.assertTrue(app.button(key="ui_step_2").disabled)
                choice.set_value("saved_one").run()
                self.assertTrue(app.button(key="ui_step_2").disabled)

                next(item for item in app.button if item.label == "불러오기").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertFalse(app.button(key="ui_step_2").disabled)
                app.run()
                backend.preview.assert_not_called()
                backend.send.assert_not_called()

    def test_tampered_saved_proposal_shows_fixed_error_and_stays_locked(self):
        from streamlit.testing.v1 import AppTest

        settings = default_settings()
        backend = mock_dialogue(LocalModelConfig(**settings["model"]))

        with TemporaryDirectory() as directory:
            root = Path(directory) / "proposals"
            save_proposal("saved_one", checked_result(), root=root)
            envelope = json.loads((root / "saved_one.json").read_text(encoding="utf-8"))
            envelope["proposal"]["questions"] = ["PRIVATE_TAMPERED_TEXT"]
            (root / "saved_one.json").write_text(json.dumps(envelope), encoding="utf-8")
            ledger = ModelCommandLedger(Path(directory) / "commands.sqlite3")

            def factory(settings):
                return UiDialogue(settings, factory=lambda config: backend, ledger=ledger, proposal_root=root)

            with patch("src.ui.dialogue_view.UiDialogue", side_effect=factory), \
                    patch("src.ui.dialogue_view.list_saved_proposals", side_effect=lambda: list_saved_proposals(root=root)):
                app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                app.selectbox(key="ui_source_mode_widget").set_value("데이터 없이 제안만 검토").run()
                next(item for item in app.button if item.label == "불러오기").click().run()
                self.assertEqual(len(app.exception), 0)
                errors = " ".join(error.value for error in app.error)
                self.assertIn("검사를 통과하지 못했습니다", errors)
                self.assertNotIn("PRIVATE_TAMPERED_TEXT", errors)
                self.assertTrue(app.button(key="ui_step_2").disabled)
                backend.send.assert_not_called()

    def test_missing_optional_dependency_shows_guidance_without_sending(self):
        from streamlit.testing.v1 import AppTest

        with patch("src.ui.dialogue_view.UiDialogue", side_effect=ImportError):
            app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
            app.selectbox(key="ui_source_mode_widget").set_value("데이터 없이 제안만 검토").run()
            app.text_area[0].set_value("품질 검사만 요청합니다.")
            next(item for item in app.button if item.label == "요청 초안 보관").click().run()
            app.button(key="ui_preview_request").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("의존성" in error.value for error in app.error))
            self.assertTrue(app.button(key="ui_step_2").disabled)


if __name__ == "__main__":
    unittest.main()

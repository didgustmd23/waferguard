"""확인한 UI 입력을 기존 생성기로 연결한다. 합성 파일·모의 모델만 사용한다."""

from contextlib import closing
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.agents import llm_execution_candidate as existing
from src.agents import llm_proposal_store as store
from src.agents.input_snapshot import create_snapshot
from src.agents.llm_config import LocalModelConfig
from src.agents.synthetic_eda_data import CSV_BYTES
from src.ui.candidate_service import UiCandidate
from src.ui.dialogue_service import UiDialogue
from src.ui.input_view import preview_input
from src.ui.model_commands import ModelCommandLedger
from src.ui.state import SCREENS, default_settings, resolve_screen
from tests.agents.test_ui_dialogue import mock_dialogue


ROOT = Path(__file__).resolve().parents[2]


class UiCandidateTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.saved = self.root / "proposals"
        self.settings = default_settings()
        backend = mock_dialogue(LocalModelConfig(**self.settings["model"]))
        self.dialogue = UiDialogue(self.settings, factory=lambda config: backend,
                                   ledger=ModelCommandLedger(self.root / "commands.sqlite3"),
                                   proposal_root=self.saved)
        self.addCleanup(self.dialogue.close)
        pending = self.dialogue.prepare("품질 검사 후 결측률을 확인해 주세요.", self.settings)
        self.dialogue.send(pending["command_id"], self.settings, confirmed=True)
        self.dialogue.save("saved01", self.settings, confirmed=True)
        source = self.root / "source.csv"
        source.write_bytes(CSV_BYTES)
        self.snapshot = self.root / "snapshot"
        create_snapshot(source, self.snapshot, max_input_bytes=4096)
        self.profile = self.root / "profile.json"
        self.profile.write_bytes((ROOT / "configs/agents/profiles/tabular_example.json").read_bytes())

        for target, replacement in (
            ("src.agents.llm_execution_candidate.load_proposal", lambda name: store.load_proposal(name, root=self.saved)),
            ("src.agents.pipeline_approval_cli._directory", lambda name, **kwargs: self.root / "runs" / name),
        ):
            patcher = patch(target, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

        (self.root / "runs").mkdir()
        result = preview_input(self.dialogue, self.settings, self.snapshot, self.profile)
        self.preview = {"settings_id": self.dialogue.settings_id,
                        "plan_signature": self.dialogue.saved_proposal(self.settings)["plan_signature"],
                        "proposal_name": "saved01", "result": result}
        self.candidate = UiCandidate(self.preview, self.dialogue, self.settings)
        patcher = patch("src.ui.candidate_service.create_candidate", side_effect=lambda *args, **kwargs:
                        existing.create_candidate(*args, **kwargs, root=self.root / "candidates"))
        self.creator = patcher.start()
        self.addCleanup(patcher.stop)

    def create(self, name="ui_run01"):
        return self.candidate.create(name, self.dialogue, self.settings, confirmed=True)

    def test_creation_requires_confirmation_and_preserves_no_approval_or_worker(self):
        with self.assertRaises(ValueError):
            self.candidate.create("ui_run01", self.dialogue, self.settings)
        self.creator.assert_not_called()

        with patch("subprocess.Popen", side_effect=AssertionError("worker 금지")):
            with patch("http.client.HTTPConnection", side_effect=AssertionError("모델 호출 금지")):
                receipt = self.create()

        self.assertEqual(receipt["status"], "execution_candidate_created")
        self.assertEqual(receipt["recorded_status"], "created")
        self.assertFalse(receipt["worker_started"])
        self.assertFalse(receipt["execution_approval_issued"])
        self.assertFalse(receipt["model_called"])
        with closing(sqlite3.connect(Path(receipt["run_dir"]) / "approvals.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 0)

    def test_duplicate_submit_and_reread_never_create_again(self):
        receipt = self.create()
        self.assertEqual(self.create(), receipt)
        self.assertEqual(self.candidate.verified_receipt(self.dialogue, self.settings), receipt)
        self.creator.assert_called_once()

    def test_changed_snapshot_profile_settings_or_saved_proposal_block_creation(self):
        changed = deepcopy(self.settings)
        changed["limits"]["wall_seconds"] += 1
        with self.assertRaises(ValueError):
            self.candidate.create("ui_run01", self.dialogue, changed, confirmed=True)
        self.creator.assert_not_called()
        self.profile.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.create()
        self.assertEqual(self.candidate.phase, "failed")
        self.assertFalse((self.root / "candidates").exists())

    def test_changed_snapshot_after_preview_is_rejected(self):
        (self.snapshot / "input.bin").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse((self.root / "candidates").exists())

    def test_failed_or_interrupted_creation_is_never_automatically_retried(self):
        self.creator.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.create()
        with self.assertRaises(ValueError):
            self.create()
        self.creator.assert_called_once()
        self.assertIsNone(self.candidate.verified_receipt(self.dialogue, self.settings))

    def test_changed_saved_proposal_and_source_do_not_reuse_confirmation(self):
        self.dialogue.ui_source_id = "different_source"
        with self.assertRaises(ValueError):
            self.create()
        self.creator.assert_not_called()
        self.dialogue.ui_source_id = None
        (self.saved / "saved01.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.create()
        self.creator.assert_not_called()

    def test_modified_preview_or_invalid_creator_response_never_unlocks_approval(self):
        self.candidate.preview["result"]["selection"]["limits"]["processes"] = 3
        with self.assertRaises(ValueError):
            self.create()
        self.creator.assert_not_called()
        candidate = UiCandidate(self.preview, self.dialogue, self.settings)
        self.creator.return_value = {"status": "untrusted"}
        self.creator.side_effect = None
        with self.assertRaises((ValueError, KeyError)):
            candidate.create("ui_run01", self.dialogue, self.settings, confirmed=True)
        self.assertIsNone(candidate.verified_receipt(self.dialogue, self.settings))

    def test_partial_records_are_preserved_without_retry(self):
        def fail_after_partial(*args, **kwargs):
            partial = self.root / "partial"
            partial.mkdir()
            (partial / "candidate.json").write_text("{}", encoding="utf-8")
            raise RuntimeError("PRIVATE_ERROR")

        self.creator.side_effect = fail_after_partial
        with self.assertRaises(RuntimeError):
            self.create()
        self.assertTrue((self.root / "partial/candidate.json").exists())
        self.assertEqual(self.candidate.phase, "failed")
        with self.assertRaises(ValueError):
            self.create()
        self.creator.assert_called_once()

    def test_receipt_tampering_or_missing_records_relocks_approval(self):
        receipt = self.create()
        marker = Path(receipt["candidate_directory"]) / "CREATED.json"
        marker.write_text("{}", encoding="utf-8")
        self.assertIsNone(self.candidate.verified_receipt(self.dialogue, self.settings))
        marker.unlink()
        self.assertIsNone(self.candidate.verified_receipt(self.dialogue, self.settings))
        with self.assertRaises(ValueError):
            self.create()
        self.creator.assert_called_once()

    def test_busy_and_invalid_names_do_not_call_creator(self):
        self.candidate._lock.acquire()
        try:
            self.assertEqual(self.create()["status"], "busy")
        finally:
            self.candidate._lock.release()

        for name in ("../escape", "UPPER", "x" * 49):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.create(name)
        self.creator.assert_not_called()

    def test_screen_four_requires_all_predecessors_and_five_stays_locked(self):
        self.assertEqual(resolve_screen(SCREENS[3], candidate_available=True), SCREENS[0])
        flags = dict(proposal_available=True, saved_available=True, candidate_available=True)
        self.assertEqual(resolve_screen(SCREENS[3], **flags), SCREENS[3])
        self.assertEqual(resolve_screen(SCREENS[4], **flags), SCREENS[0])


@unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 후 실제 UI 흐름을 검사하세요.")
class UiCandidateAppTest(unittest.TestCase):
    def test_create_opens_four_without_approval_and_rerun_is_read_only(self):
        from streamlit.testing.v1 import AppTest

        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = default_settings()
            backend = mock_dialogue(LocalModelConfig(**settings["model"]))
            source = root / "source.csv"
            source.write_bytes(CSV_BYTES)
            snapshot = root / "snapshot"
            create_snapshot(source, snapshot, max_input_bytes=4096)
            profile = root / "profile.json"
            profile.write_bytes((ROOT / "configs/agents/profiles/tabular_example.json").read_bytes())
            (root / "runs").mkdir()

            def factory(settings):
                return UiDialogue(settings, factory=lambda config: backend,
                                  ledger=ModelCommandLedger(root / "commands.sqlite3"),
                                  proposal_root=root / "proposals")

            with patch("src.ui.dialogue_view.UiDialogue", side_effect=factory), patch(
                    "src.agents.llm_execution_candidate.load_proposal",
                    side_effect=lambda name: store.load_proposal(name, root=root / "proposals")), patch(
                    "src.agents.pipeline_approval_cli._directory", side_effect=lambda name, **kwargs: root / "runs" / name):
                with patch("src.ui.candidate_service.create_candidate", side_effect=lambda *args, **kwargs:
                           existing.create_candidate(*args, **kwargs, root=root / "candidates")) as creator:
                    app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                    app.selectbox(key="ui_source_mode_widget").set_value("데이터 없이 제안만 검토").run()
                    app.text_area[0].set_value("품질 검사 후 결측률을 확인해 주세요.")
                    next(item for item in app.button if item.label == "요청 초안 보관").click().run()
                    app.button(key="ui_preview_request").click().run()
                    app.checkbox[0].check()
                    next(item for item in app.button if item.label == "확인한 내용 전송").click().run()
                    app.button(key="ui_step_2").click().run()
                    app.checkbox[0].check()
                    next(item for item in app.button if item.label == "검토한 제안 저장").click().run()
                    app.button(key="ui_step_3").click().run()
                    app.text_input[0].set_value(str(snapshot))
                    app.text_input[1].set_value(str(profile))
                    app.checkbox[0].check()
                    next(item for item in app.button if item.label == "입력 결합 미리보기").click().run()
                    app.text_input[1].set_value(str(root / "changed_profile.json"))
                    with patch("src.ui.path_picker.pick_local_path", return_value={"status": "cancelled"}):
                        next(item for item in app.button if item.label == "파일 찾기").click().run()
                    self.assertFalse(any(item.label == "실행 후보 생성" for item in app.button))
                    creator.assert_not_called()
                    app.text_input[1].set_value(str(profile))
                    next(item for item in app.button if item.label == "입력 결합 미리보기").click().run()
                    next(item for item in app.button if item.label == "실행 후보 생성").click().run()
                    creator.assert_not_called()
                    app.checkbox[1].check()
                    next(item for item in app.button if item.label == "실행 후보 생성").click().run()
                    self.assertFalse(app.button(key="ui_step_4").disabled)
                    app.run()
                    creator.assert_called_once()
                    app.button(key="ui_step_4").click().run()
                    self.assertEqual(app.session_state["ui_screen"], SCREENS[3])
                    self.assertFalse(app.button(key="approval_flow_start").disabled)
                    self.assertFalse(any(item.label == "승인하고 실행" for item in app.button))
                    self.assertTrue(app.button(key="ui_step_5").disabled)
                    self.assertEqual(len(app.exception), 0)
                    backend.send.assert_called_once()
                    with closing(sqlite3.connect(root / "runs/ui_run01/approvals.sqlite3")) as db:
                        self.assertEqual(db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0], 0)

                    # 생성 영수증이 변경되면 완료 플래그와 무관하게 4번을 다시 잠근다.
                    (root / "candidates/ui_run01/CREATED.json").write_text("{}", encoding="utf-8")
                    app.run()
                    self.assertTrue(app.button(key="ui_step_4").disabled)
                    self.assertEqual(app.session_state["ui_screen"], SCREENS[0])
                    creator.assert_called_once()


if __name__ == "__main__":
    unittest.main()

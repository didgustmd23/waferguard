"""제안 저장·복원·입력 메뉴와 기존 바인더 연결을 합성 파일로만 검사한다."""

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.agents import llm_proposal_store as store
from src.agents.input_snapshot import create_snapshot
from src.agents.llm_config import LocalModelConfig
from src.agents.synthetic_eda_data import CSV_BYTES
from src.ui.dialogue_service import UiDialogue
from src.ui.input_view import preview_input
from src.ui.model_commands import ModelCommandLedger
from src.ui.state import SCREENS, default_settings, resolve_screen
from tests.agents.test_ui_dialogue import mock_dialogue


ROOT = Path(__file__).resolve().parents[2]


class UiHandoffTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.saved = self.root / "proposals"
        self.settings = default_settings()
        self.backend = mock_dialogue(LocalModelConfig(**self.settings["model"]))
        self.dialogue = UiDialogue(self.settings, factory=lambda config: self.backend,
                                   ledger=ModelCommandLedger(self.root / "commands.sqlite3"),
                                   proposal_root=self.saved)
        self.addCleanup(self.dialogue.close)
        pending = self.dialogue.prepare("품질 검사 후 결측률을 확인해 주세요.", self.settings)
        self.dialogue.send(pending["command_id"], self.settings, confirmed=True)

    def save(self, name="ui_saved01"):
        return self.dialogue.save(name, self.settings, confirmed=True)

    def fixtures(self):
        source = self.root / "source.csv"
        source.write_bytes(CSV_BYTES)
        snapshot = self.root / "snapshot"
        create_snapshot(source, snapshot, max_input_bytes=4096)
        profile = self.root / "profile.json"
        profile.write_bytes((ROOT / "configs/agents/profiles/tabular_example.json").read_bytes())
        return snapshot, profile

    def test_save_requires_confirmation_and_does_not_store_dialogue(self):
        with self.assertRaises(ValueError):
            self.dialogue.save("ui_saved01", self.settings)

        self.assertFalse(self.saved.exists())
        receipt = self.save()
        payload = Path(receipt["local_file"]).read_text(encoding="utf-8")
        self.assertNotIn("품질 검사 후", payload)
        self.assertFalse(receipt["execution_approved"])
        self.assertFalse(receipt["worker_started"])
        self.assertIsNotNone(self.dialogue.saved_proposal(self.settings))
        self.backend.send.assert_called_once()

    def test_duplicate_save_is_read_only_and_existing_file_is_not_overwritten(self):
        receipt = self.save()
        path = Path(receipt["local_file"])
        before = (path.stat().st_mtime_ns, path.read_bytes())

        with patch("src.ui.dialogue_service.save_proposal", side_effect=AssertionError("재저장 금지")):
            self.assertEqual(self.save(), receipt)

        self.assertEqual(before, (path.stat().st_mtime_ns, path.read_bytes()))
        self.dialogue.saved = None

        with self.assertRaises(FileExistsError):
            self.save()

        self.assertEqual(before, (path.stat().st_mtime_ns, path.read_bytes()))

    def test_invalid_name_and_changed_settings_never_save(self):
        for name in ("../escape", "UPPER", "", "x" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.save(name)

        changed = deepcopy(self.settings)
        changed["limits"]["wall_seconds"] += 1

        with self.assertRaises(ValueError):
            self.dialogue.save("ui_saved01", changed, confirmed=True)

        self.assertFalse(self.saved.exists())

    def test_missing_or_tampered_saved_file_keeps_input_locked(self):
        path = Path(self.save()["local_file"])
        path.write_text("{}", encoding="utf-8")
        self.assertIsNone(self.dialogue.saved_proposal(self.settings))
        path.unlink()
        self.assertIsNone(self.dialogue.saved_proposal(self.settings))

    def test_navigation_requires_both_validated_and_saved_proposal(self):
        self.assertEqual(resolve_screen(SCREENS[2], proposal_available=True), SCREENS[0])
        self.assertEqual(resolve_screen(SCREENS[2], saved_available=True), SCREENS[0])
        self.assertEqual(resolve_screen(SCREENS[2], proposal_available=True, saved_available=True), SCREENS[2])

        for screen in SCREENS[3:5]:
            self.assertEqual(resolve_screen(screen, proposal_available=True, saved_available=True), SCREENS[0])

    def test_real_binder_preview_does_not_call_model_create_approval_or_worker(self):
        self.save()
        snapshot, profile = self.fixtures()
        original = (snapshot / "input.bin").read_bytes()

        with patch("src.agents.llm_execution_candidate.load_proposal",
                   side_effect=lambda name: store.load_proposal(name, root=self.saved)):
            with patch("src.agents.pipeline_approval_cli.create", side_effect=AssertionError("작업 생성 금지")):
                with patch("subprocess.Popen", side_effect=AssertionError("프로세스 금지")):
                    result = preview_input(self.dialogue, self.settings, snapshot, profile)

        self.assertEqual(result["status"], "candidate_preview")
        self.assertFalse(result["worker_started"])
        self.assertFalse(result["model_called"])
        self.assertFalse(result["execution_approval_issued"])
        self.assertEqual((snapshot / "input.bin").read_bytes(), original)
        self.backend.send.assert_called_once()

    def test_snapshot_tampering_is_rejected(self):
        self.save()
        snapshot, profile = self.fixtures()
        (snapshot / "input.bin").write_bytes(b"changed")

        with patch("src.agents.llm_execution_candidate.load_proposal",
                   side_effect=lambda name: store.load_proposal(name, root=self.saved)):
            with self.assertRaises(ValueError):
                preview_input(self.dialogue, self.settings, snapshot, profile)


@unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 후 실제 저장 화면을 검사하세요.")
class UiSaveAppTest(unittest.TestCase):
    def test_review_save_unlocks_input_and_rerender_does_not_save_again(self):
        from streamlit.testing.v1 import AppTest

        settings = default_settings()
        backend = mock_dialogue(LocalModelConfig(**settings["model"]))

        with TemporaryDirectory() as directory:
            root = Path(directory)

            def factory(settings):
                return UiDialogue(settings, factory=lambda config: backend,
                                  ledger=ModelCommandLedger(root / "commands.sqlite3"),
                                  proposal_root=root / "proposals")

            with patch("src.ui.dialogue_view.UiDialogue", side_effect=factory):
                app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                app.selectbox(key="ui_source_mode_widget").set_value("데이터 없이 제안만 검토").run()
                app.text_area[0].set_value("품질 검사 후 결측률을 확인해 주세요.")
                next(item for item in app.button if item.label == "요청 초안 보관").click().run()
                app.button(key="ui_preview_request").click().run()
                app.checkbox[0].check()
                next(item for item in app.button if item.label == "확인한 내용 전송").click().run()
                app.button(key="ui_step_2").click().run()
                self.assertTrue(app.button(key="ui_step_3").disabled)
                next(item for item in app.button if item.label == "검토한 제안 저장").click().run()
                self.assertFalse((root / "proposals").exists())
                app.checkbox[0].check()
                next(item for item in app.button if item.label == "검토한 제안 저장").click().run()
                self.assertFalse(app.button(key="ui_step_3").disabled)
                saved = root / "proposals/ui_proposal01.json"
                before = saved.stat().st_mtime_ns
                app.run()
                self.assertEqual(saved.stat().st_mtime_ns, before)
                app.button(key="ui_step_3").click().run()
                self.assertEqual(app.session_state["ui_screen"], SCREENS[2])
                self.assertTrue(app.button(key="ui_step_4").disabled)

                source = root / "source.csv"
                source.write_bytes(CSV_BYTES)
                snapshot = root / "snapshot"
                create_snapshot(source, snapshot, max_input_bytes=4096)
                profile = root / "profile.json"
                profile.write_bytes((ROOT / "configs/agents/profiles/tabular_example.json").read_bytes())
                app.text_input[0].set_value(str(snapshot))
                app.text_input[1].set_value(str(profile))
                app.checkbox[0].check()

                with patch("src.agents.llm_execution_candidate.load_proposal",
                           side_effect=lambda name: store.load_proposal(name, root=root / "proposals")):
                    with patch("src.ui.input_view.preview_input", wraps=preview_input) as preview:
                        next(item for item in app.button if item.label == "입력 결합 미리보기").click().run()
                        self.assertTrue(any("입력 결합 검사를 통과" in item.value for item in app.success))
                        app.run()
                        preview.assert_called_once()

                self.assertEqual(len(app.exception), 0)
                backend.send.assert_called_once()

                # 저장 파일이 훼손되면 UI 완료 플래그가 남아 있어도 3번을 다시 잠근다.
                saved.write_text("{}", encoding="utf-8")
                app.run()
                self.assertEqual(app.session_state["ui_screen"], SCREENS[0])
                self.assertTrue(app.button(key="ui_step_3").disabled)


if __name__ == "__main__":
    unittest.main()

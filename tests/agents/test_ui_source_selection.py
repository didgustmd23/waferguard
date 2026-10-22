"""선행 데이터 선택·로컬 복사·전송 정보 분리를 합성 파일로 확인한다."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.agents.input_snapshot import InputTooLarge, Snapshot, verify_snapshot
from src.ui.source_selection import (CSV_SOURCE, CSV_SNAPSHOT, PLAN_ONLY, UNSUPPORTED,
                                     prepare_source_snapshot, select_source, selected_source)
from src.ui.dialogue_view import current_dialogue
from src.ui.state import default_settings
from src.ui.dialogue_service import UiDialogue
from src.ui.model_commands import ModelCommandLedger
from src.agents.llm_config import LocalModelConfig
from tests.agents.test_ui_dialogue import mock_dialogue


ROOT = Path(__file__).resolve().parents[2]


class UiSourceSelectionTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "synthetic.csv"
        self.source.write_bytes(b"x,y\n1,2\n3,4\n")

    def test_selection_checks_metadata_without_reading_data_or_copying(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("내용 읽기 금지")):
            selection = select_source(CSV_SOURCE, str(self.source))

        self.assertEqual(selection["path"], str(self.source.resolve()))
        self.assertEqual(selection["size_bytes"], 12)
        self.assertEqual(list(self.root.iterdir()), [self.source])
        self.assertIsNone(selected_source({}))
        self.assertIsNone(selected_source({"ui_source_mode": CSV_SOURCE}))
        self.assertEqual(selected_source({"ui_source_mode": PLAN_ONLY})["selection_id"], "proposal_only")
        self.assertIsNone(selected_source({"ui_source_mode": UNSUPPORTED}))

    def test_wrong_kind_path_or_extension_is_rejected(self):
        for mode, path in ((UNSUPPORTED, str(self.source)), (CSV_SOURCE, ""),
                           (CSV_SOURCE, str(self.root)), (CSV_SOURCE, str(self.root / "missing.csv"))):
            with self.subTest(mode=mode, path=path), self.assertRaises(ValueError):
                select_source(mode, path)

        pickle = self.root / "unsafe.pkl"
        pickle.write_bytes(b"not-a-pickle")

        with self.assertRaises(ValueError):
            select_source(CSV_SOURCE, str(pickle))

    def test_copy_preserves_source_and_existing_snapshot_is_selectable(self):
        selection = select_source(CSV_SOURCE, str(self.source))
        before = self.source.read_bytes()
        result = prepare_source_snapshot(selection, 1, root=self.root / "snapshots")
        copied = Path(result["directory"])
        self.assertTrue(verify_snapshot(copied, Snapshot(**result["snapshot"])))
        self.assertEqual((copied / "input.bin").read_bytes(), before)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(select_source(CSV_SNAPSHOT, str(copied))["path"], str(copied))

    def test_changed_source_and_oversize_are_blocked_before_snapshot_creation(self):
        selection = select_source(CSV_SOURCE, str(self.source))
        self.source.write_bytes(b"changed")

        with self.assertRaises(ValueError):
            prepare_source_snapshot(selection, 1, root=self.root / "changed_output")

        self.assertFalse((self.root / "changed_output").exists())
        self.source.write_bytes(b"x" * (1024 * 1024 + 1))
        selection = select_source(CSV_SOURCE, str(self.source))

        with self.assertRaises(InputTooLarge):
            prepare_source_snapshot(selection, 1, root=self.root / "oversize_output")

        self.assertFalse(any((self.root / "oversize_output").iterdir()))

    def test_source_change_invalidates_dialogue_and_input_preview(self):
        dialogue = Mock()
        dialogue.matches.return_value = True
        dialogue.ui_source_id = "proposal_only"
        state = {"ui_source_mode": PLAN_ONLY, "ui_settings": default_settings(),
                 "ui_dialogue": dialogue, "ui_input_preview": {"old": True}}
        st = SimpleNamespace(session_state=state)
        self.assertIs(current_dialogue(st), dialogue)
        state["ui_source_mode"] = CSV_SOURCE
        self.assertIsNone(current_dialogue(st))
        dialogue.close.assert_called_once()
        self.assertNotIn("ui_input_preview", state)


@unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 후 초기 데이터 선택 UI를 검사하세요.")
class UiSourceAppTest(unittest.TestCase):
    def test_unselected_or_unsupported_data_cannot_call_model(self):
        from streamlit.testing.v1 import AppTest

        with patch("src.ui.dialogue_view.UiDialogue") as factory:
            app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
            app.checkbox(key="ui_show_advanced_tools").check().run()
            app.selectbox(key="ui_workflow_mode").select("LLM 제안").run()
            app.text_area[0].set_value("결측률을 확인해 주세요.")
            next(item for item in app.button if item.label == "요청 초안 보관").click().run()
            self.assertTrue(app.button(key="ui_preview_request").disabled)
            app.selectbox(key="ui_source_mode_widget").set_value(UNSUPPORTED).run()
            self.assertTrue(app.button(key="ui_preview_request").disabled)
            factory.assert_not_called()

    def test_csv_selection_is_local_and_path_is_not_sent_to_model(self):
        from streamlit.testing.v1 import AppTest

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "private_source.csv"
            source.write_bytes(b"x,y\n1,2\n")
            settings = default_settings()
            backend = mock_dialogue(LocalModelConfig(**settings["model"]))

            def factory(settings):
                return UiDialogue(settings, factory=lambda config: backend,
                                  ledger=ModelCommandLedger(root / "commands.sqlite3"),
                                  proposal_root=root / "proposals")

            with patch("src.ui.dialogue_view.UiDialogue", side_effect=factory):
                app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                app.checkbox(key="ui_show_advanced_tools").check().run()
                app.selectbox(key="ui_workflow_mode").select("LLM 제안").run()
                app.selectbox(key="ui_source_mode_widget").set_value(CSV_SOURCE).run()
                self.assertTrue(any(item.label == "데이터 경로" for item in app.text_input))
                app.text_input[0].set_value(str(source))
                next(item for item in app.button if item.label == "로컬 데이터 선택 확인").click().run()
                app.text_area[0].set_value("품질 검사 후 결측률을 확인해 주세요.")
                next(item for item in app.button if item.label == "요청 초안 보관").click().run()
                app.button(key="ui_preview_request").click().run()
                dialogue = app.session_state["ui_dialogue"]
                payload = str(dialogue.pending["preview"]["payload"])
                self.assertNotIn(str(source), payload)
                self.assertNotIn(source.name, payload)
                self.assertEqual(list(root.iterdir()), [source])
                backend.send.assert_not_called()
                app.checkbox[0].check()
                next(item for item in app.button if item.label == "확인한 내용 전송").click().run()
                app.button(key="ui_step_2").click().run()
                app.checkbox[0].check()
                next(item for item in app.button if item.label == "검토한 제안 저장").click().run()
                app.button(key="ui_step_3").click().run()

                with patch("src.ui.input_view.prepare_source_snapshot", side_effect=lambda selection, maximum:
                           prepare_source_snapshot(selection, maximum, root=root / "snapshots")) as copy:
                    next(item for item in app.button if item.label == "선택한 CSV의 snapshot 준비").click().run()
                    copy.assert_not_called()
                    self.assertFalse((root / "snapshots").exists())
                    app.checkbox[0].check()
                    next(item for item in app.button if item.label == "선택한 CSV의 snapshot 준비").click().run()
                    copy.assert_called_once()
                    self.assertTrue(app.text_input[0].disabled)
                    snapshot_path = Path(app.text_input[0].value)
                    self.assertEqual((snapshot_path / "input.bin").read_bytes(), source.read_bytes())
                    app.run()
                    copy.assert_called_once()

                app.button(key="ui_step_1").click().run()
                app.selectbox(key="ui_source_mode_widget").set_value(PLAN_ONLY).run()
                self.assertNotIn("ui_dialogue", app.session_state)
                self.assertEqual(len(app.exception), 0)
                backend.send.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""선택 창을 실제로 띄우지 않고 경로 입력·취소·재실행 방지 경계를 검사한다."""

from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.ui import path_dialog, path_picker
from src.ui.source_selection import CSV_SOURCE, CSV_SNAPSHOT


ROOT = Path(__file__).resolve().parents[2]


class NativePathTransportTest(unittest.TestCase):
    def test_path_field_and_browse_button_use_bottom_alignment(self):
        st = MagicMock()
        st.session_state = {}
        st.text_input.return_value = "D:/sample.csv"

        result = path_picker.path_input(st, "데이터 경로", key="sample", kind="csv_file", help="경로 안내")

        self.assertEqual(result, "D:/sample.csv")
        st.columns.assert_called_once_with([4, 1], vertical_alignment="bottom")

    def test_only_fixed_script_and_kind_are_launched_without_shell(self):
        process = Mock(returncode=0)
        process.communicate.return_value = (json.dumps({"status": "selected", "path": "D:/자료/실험 파일.csv"}).encode(), None)
        process.poll.return_value = 0

        with patch.object(path_picker.os, "name", "nt"):
            with patch.object(path_picker.subprocess, "Popen", return_value=process) as launch:
                result = path_picker.pick_local_path("csv_file")

        self.assertEqual(result["path"], "D:/자료/실험 파일.csv")
        self.assertEqual(launch.call_args.args[0], [sys.executable, "-I", str(path_picker.DIALOG_SCRIPT), "csv_file"])
        self.assertFalse(launch.call_args.kwargs["shell"])
        self.assertEqual(launch.call_args.kwargs["creationflags"], subprocess.CREATE_NO_WINDOW)
        process.communicate.assert_called_once_with(timeout=180)

    def test_cancelled_malformed_and_oversized_replies(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0

        for raw, status in ((b'{"status":"cancelled"}', "cancelled"),
                            (b'not json', "unavailable"), (b'x' * 32769, "unavailable"),
                            (b'{"status":"selected","path":"x","extra":true}', "unavailable")):
            process.communicate.return_value = (raw, None)

            with self.subTest(raw=raw[:30]), patch.object(path_picker.os, "name", "nt"):
                with patch.object(path_picker.subprocess, "Popen", return_value=process):
                    self.assertEqual(path_picker.pick_local_path("directory")["status"], status)

    def test_timeout_cleans_up_child_and_releases_lock(self):
        process = Mock()
        process.poll.return_value = None
        process.communicate.side_effect = [subprocess.TimeoutExpired("fixed_helper", 180), (b"", None)]

        with patch.object(path_picker.os, "name", "nt"):
            with patch.object(path_picker.subprocess, "Popen", return_value=process):
                self.assertEqual(path_picker.pick_local_path("directory"), {"status": "unavailable"})

        process.kill.assert_called_once()
        self.assertTrue(path_picker._DIALOG_LOCK.acquire(blocking=False))
        path_picker._DIALOG_LOCK.release()

    def test_invalid_kind_or_busy_dialog_never_launches_process(self):
        with patch.object(path_picker.subprocess, "Popen") as launch:
            self.assertEqual(path_picker.pick_local_path("arbitrary_command"), {"status": "unavailable"})
            path_picker._DIALOG_LOCK.acquire()

            try:
                self.assertEqual(path_picker.pick_local_path("directory"), {"status": "unavailable"})
            finally:
                path_picker._DIALOG_LOCK.release()

            launch.assert_not_called()

    def test_cancel_or_failure_preserves_entered_path(self):
        state = {"field": "D:/existing.csv", "_widget_field": "D:/existing.csv"}
        st = SimpleNamespace(session_state=state)

        for status in ("cancelled", "unavailable"):
            with patch.object(path_picker, "pick_local_path", return_value={"status": status}):
                path_picker.fill_from_dialog(st, "field", "csv_file")

            self.assertEqual(state["field"], "D:/existing.csv")
            self.assertEqual(state["_widget_field"], "D:/existing.csv")


class NativeDialogHelperTest(unittest.TestCase):
    def test_file_filters_and_folder_mode_with_mocked_tk(self):
        tk = Mock()
        tk.filedialog.askopenfilename.return_value = "D:/자료/선택.csv"
        tk.filedialog.askdirectory.return_value = "D:/자료"

        with patch.dict(sys.modules, {"tkinter": tk}):
            self.assertEqual(path_dialog.choose_path("csv_file")["path"], "D:/자료/선택.csv")
            self.assertEqual(tk.filedialog.askopenfilename.call_args.kwargs["filetypes"][0][1], "*.csv")
            path_dialog.choose_path("json_file")
            self.assertEqual(tk.filedialog.askopenfilename.call_args.kwargs["filetypes"][0][1], "*.json")
            self.assertEqual(path_dialog.choose_path("directory")["path"], "D:/자료")

        self.assertEqual(tk.Tk.return_value.destroy.call_count, 3)

    def test_cancel_and_tk_failure_do_not_expose_raw_exception(self):
        tk = Mock()
        tk.filedialog.askdirectory.return_value = ""

        with patch.dict(sys.modules, {"tkinter": tk}):
            self.assertEqual(path_dialog.choose_path("directory"), {"status": "cancelled"})
            tk.Tk.side_effect = RuntimeError("PRIVATE_EXCEPTION")
            self.assertEqual(path_dialog.choose_path("directory"), {"status": "unavailable"})


@unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 후 폼 연결 테스트를 실행하세요.")
class PathPickerAppTest(unittest.TestCase):
    def test_browse_fills_form_without_confirming_data_or_reopening_on_rerun(self):
        from streamlit.testing.v1 import AppTest

        with TemporaryDirectory() as directory:
            source = Path(directory) / "실험 파일.csv"
            source.write_bytes(b"x,y\n1,2\n")
            with patch.object(path_picker, "pick_local_path", return_value={"status": "selected", "path": str(source)}) as pick:
                app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
                app.selectbox(key="ui_source_mode_widget").set_value(CSV_SOURCE).run()
                app.button(key="source_csv_path_browse").click().run()
                self.assertEqual(app.text_input[0].value, str(source))
                self.assertNotIn("ui_source_selection", app.session_state)
                self.assertFalse((Path(directory) / "snapshot").exists())
                app.run()
                pick.assert_called_once_with("csv_file")
                next(item for item in app.button if item.label == "로컬 데이터 선택 확인").click().run()
                self.assertEqual(app.session_state["ui_source_selection"]["path"], str(source))
                pick.assert_called_once()
                self.assertEqual(len(app.exception), 0)

    def test_folder_picker_and_cancel_keep_manual_entry(self):
        from streamlit.testing.v1 import AppTest

        with patch.object(path_picker, "pick_local_path", return_value={"status": "cancelled"}) as pick:
            app = AppTest.from_file(str(ROOT / "src/ui/app.py")).run()
            app.selectbox(key="ui_source_mode_widget").set_value(CSV_SNAPSHOT).run()
            app.text_input[0].set_value("D:/manually_entered/snapshot")
            app.button(key="source_snapshot_path_browse").click().run()
            self.assertEqual(app.text_input[0].value, "D:/manually_entered/snapshot")
            pick.assert_called_once_with("directory")
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()

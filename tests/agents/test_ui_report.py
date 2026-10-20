"""검증된 결과 묶음의 상대 PNG 표시와 외부 참조·파일 열기 차단을 검사한다."""

from importlib.util import find_spec
import struct
import unittest
from unittest.mock import MagicMock, patch
import zlib

from src.ui.report_view import render_report, referenced_png
from src.ui.execution_view import render_result_files


def png(value=127):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes([0, value]))) + chunk(b"IEND", b""))


class UiReportTest(unittest.TestCase):
    def setUp(self):
        self.image = {"name": "analysis/eda/distributions.png", "suffix": ".png", "data": png()}
        self.report = {"name": "analysis/eda/eda.md", "suffix": ".md",
                       "data": "# 구조·품질 EDA\n\n![분포](distributions.png)\n\n설명입니다.".encode()}
        self.files = [self.report, self.image]

    def test_relative_reference_uses_exact_report_directory_and_in_memory_bytes(self):
        other = self.image | {"name": "other/distributions.png", "data": png(10)}
        with patch("pathlib.Path.open", side_effect=AssertionError("파일 추가 읽기 금지")):
            self.assertEqual(referenced_png(self.report["name"], "distributions.png", self.files + [other]),
                             self.image["data"])

        self.assertIsNone(referenced_png(self.report["name"], "distributions.png", self.files + [self.image]))

    def test_external_absolute_traversal_and_stream_references_are_not_resolved(self):
        for reference in ("https://example.invalid/image.png", "//server/image.png", "/image.png",
                          "../eda/distributions.png", "C:/image.png", "file:///image.png",
                          "data:image/png;base64,xx", "distributions.png:stream", "folder\\image.png",
                          "distributions.png?x", "distributions.png#x"):
            with self.subTest(reference=reference):
                self.assertIsNone(referenced_png(self.report["name"], reference, self.files))

    def test_report_body_contains_image_and_explanation_without_markdown_fetch(self):
        st = MagicMock()
        render_report(st, self.report, self.files)
        st.image.assert_called_once_with(self.image["data"])
        texts = [call.args[0] for call in st.text.call_args_list]
        self.assertIn("분포", texts)
        self.assertIn("설명입니다.", texts)
        self.assertFalse(any("![" in text for text in texts))
        st.markdown.assert_not_called()

    def test_windows_crlf_report_resolves_same_image(self):
        st = MagicMock()
        report = self.report | {"data": self.report["data"].replace(b"\n", b"\r\n")}
        render_report(st, report, self.files)
        st.image.assert_called_once_with(self.image["data"])

    def test_missing_corrupt_or_external_image_is_not_loaded(self):
        for reference, files in (("missing.png", self.files),
                                  ("distributions.png", [self.image | {"data": b"invalid"}]),
                                  ("https://example.invalid/a.png", self.files)):
            st = MagicMock()
            report = self.report | {"data": f"![그림]({reference})".encode()}
            render_report(st, report, files)
            st.image.assert_not_called()
            st.warning.assert_called_once()

    def test_image_reference_count_is_bounded(self):
        st = MagicMock()
        report = self.report | {"data": ("![그림](distributions.png)\n" * 40).encode()}
        render_report(st, report, self.files)
        self.assertEqual(st.image.call_count, 32)
        st.caption.assert_called_once()

    def test_shared_result_view_routes_markdown_to_inline_renderer(self):
        st = MagicMock()
        render_result_files(st, self.files)
        self.assertEqual(st.image.call_count, 2)  # 본문 참조와 개별 PNG 결과
        st.markdown.assert_not_called()

    @unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 필요")
    def test_streamlit_report_renders_image_without_execution(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("""import streamlit as st
from src.ui.report_view import render_report
from tests.agents.test_ui_report import png
files = [{'name': 'eda.md', 'suffix': '.md', 'data': b'![plot](plot.png)'},
         {'name': 'plot.png', 'suffix': '.png', 'data': png()}]
render_report(st, files[0], files)
""").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("image")), 1)


class UiReportTableTest(unittest.TestCase):
    def report(self, text):
        return {"name": "proposal/proposal.md", "suffix": ".md", "data": text.encode()}

    def test_class_split_table_has_named_columns_and_nine_rows_for_lf_and_crlf(self):
        header = "| 클래스 번호 | Train | Validation | Test |\n| --- | ---: | ---: | ---: |\n"
        rows = "".join(f"| {number} | 1 | 1 | 1 |\n" for number in range(9))
        for newline in ("\n", "\r\n"):
            st = MagicMock()
            render_report(st, self.report(("설명\n\n" + header + rows + "\n후속 설명").replace("\n", newline)), [])
            st.dataframe.assert_called_once()
            frame = st.dataframe.call_args.args[0]
            self.assertEqual(list(frame.columns), ["클래스 번호", "Train", "Validation", "Test"])
            self.assertEqual(frame.shape, (9, 4))
            self.assertEqual(frame.iloc[0].tolist(), ["0", "1", "1", "1"])
            self.assertTrue(any("후속 설명" in call.args[0] for call in st.text.call_args_list))

    def test_incomplete_duplicate_header_and_code_block_stay_plain_text(self):
        for text in ("| A | B |\n| 1 | 2 |", "| A | A |\n| --- | --- |\n| 1 | 2 |",
                     "```text\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```"):
            st = MagicMock()
            render_report(st, self.report(text), [])
            st.dataframe.assert_not_called()
            st.text.assert_called_once()

    def test_cell_links_and_html_remain_plain_dataframe_strings(self):
        st = MagicMock()
        cell = '![external](https://example.invalid/a.png)'
        render_report(st, self.report(f'| A | B |\n| --- | --- |\n| {cell} | <img src=x> |'), [])
        frame = st.dataframe.call_args.args[0]
        self.assertEqual(frame.iloc[0].tolist(), [cell, '<img src=x>'])
        st.markdown.assert_not_called()
        st.image.assert_not_called()

    def test_escaped_pipe_is_preserved_in_cell(self):
        st = MagicMock()
        render_report(st, self.report('| A | B |\n| --- | --- |\n| x\\|y | z |'), [])
        self.assertEqual(st.dataframe.call_args.args[0].iloc[0, 0], 'x|y')

    def test_candidate_escaped_identifiers_display_without_markdown_backslashes(self):
        st = MagicMock()
        text = '| 항목 | 설정 |\n| --- | --- |\n| missing\\_ratio\\_threshold | 0.5 |\n\n'
        text += '| 특징 | 사유 |\n| --- | --- |\n| sensor\\_5 | CONSTANT\\_OR\\_ALL\\_MISSING |'
        render_report(st, self.report(text), [])
        frames = [call.args[0] for call in st.dataframe.call_args_list]
        self.assertEqual(frames[0].iloc[0, 0], 'missing_ratio_threshold')
        self.assertEqual(frames[1].iloc[0].tolist(), ['sensor_5', 'CONSTANT_OR_ALL_MISSING'])
        st.markdown.assert_not_called()
        st.image.assert_not_called()

    def test_csv_candidate_preview_uses_common_tables_without_creation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.ui.csv_plan_view import inputs

        report = '| 항목 | 설정 |\n| --- | --- |\n| scaling | none |\n\n'
        report += '| 특징 | 사유 |\n| --- | --- |\n'
        report += ''.join(f'| sensor\\_{index} | HIGH\\_MISSING\\_RATIO |\n' for index in range(146))
        plan = SimpleNamespace(active=SimpleNamespace(candidate_report=report))
        st = MagicMock()

        with patch('src.ui.csv_plan_view._progress', return_value=plan):
            with patch('src.ui.csv_plan_view.workflow_view.inputs') as form:
                inputs(st)

        self.assertEqual([call.args[0].shape for call in st.dataframe.call_args_list], [(1, 2), (146, 2)])
        self.assertEqual(st.dataframe.call_args_list[1].args[0].iloc[-1, 0], 'sensor_145')
        form.assert_called_once_with(st)
        st.markdown.assert_not_called()

    def test_rows_and_table_count_are_bounded(self):
        st = MagicMock()
        header = '| A | B |\n| --- | --- |\n'
        render_report(st, self.report(header + '| x | y |\n' * 205), [])
        self.assertEqual(st.dataframe.call_args.args[0].shape, (200, 2))
        st.caption.assert_called_once()

        st = MagicMock()
        render_report(st, self.report(('\n' + header + '| x | y |\n') * 20), [])
        self.assertEqual(st.dataframe.call_count, 16)

    @unittest.skipUnless(find_spec("streamlit"), "Streamlit 설치 필요")
    def test_streamlit_displays_real_table(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("""import streamlit as st
from src.ui.report_view import render_report
report = {'name': 'proposal.md', 'suffix': '.md', 'data': b'| Train | Test |\\n| --- | --- |\\n| 1 | 1 |'}
render_report(st, report, [])
""").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(app.dataframe[0].value.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()

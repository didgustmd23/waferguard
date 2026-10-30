"""보고서 그림 저장·상대 경로 연결·모델 의견 분리를 검사한다. 원본·worker·모델 호출은 없다."""

import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
from PIL import Image

from src.agents import report_figures, tabular_report_evidence as evidence, task_report
from tests.agents import test_tabular_report_scope as scope


matplotlib.use("Agg")

PNG = b"\x89PNG\r\n\x1a\n" + b"eda-bytes"


def chart(title="분할 그래프", section="split", name="split_ratio.png", rows=None):
    rows = [{"구간": "train", "요청 비율": 0.7, "확정 비율": 0.68}, {"구간": "$x_1$", "요청 비율": 0.3, "확정 비율": 0.32}] if rows is None else rows
    return {"title": title, "notes": [], "category": "구간", "values": ["요청 비율", "확정 비율"], "unit": "비율", "horizontal": False,
            "rows": rows, "section": section, "file": name}


def facts(**changes):
    return dict(scope.facts(), goal="목표", technical={}, cautions=[], images=[{"label": "missing.png"}], charts=[chart()],
                eda_tables=[{"title": "표", "notes": [], "rows": [{"항목": "a"}]}], **changes)


class DrawTest(unittest.TestCase):
    def test_png_has_no_metadata_and_is_repeatable(self):
        data = report_figures.draw_chart(chart())
        self.assertEqual(data, report_figures.draw_chart(chart()))
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(set(image.info) - {"dpi"}, set())

    def test_horizontal_chart_with_markup_name_draws(self):
        rows = [{"클래스": "01. $\\frac{", "건수": 3}, {"클래스": "02. 정상", "건수": 1}]
        data = report_figures.draw_chart(dict(chart(rows=rows), category="클래스", values=["건수"], horizontal=True))
        self.assertTrue(data.startswith(b"\x89PNG"))


class SaveTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.images = [{"label": "missing.png", "path": "ignored", "data": PNG}]

    def save(self, value=None, images=None):
        return report_figures.save_figures(value or facts(), self.images if images is None else images, self.folder, 1)

    def test_saves_chart_and_same_eda_bytes_with_relative_links(self):
        result = self.save()
        figures = result["figures"]
        self.assertTrue(re.fullmatch(r"figures_1_[0-9a-f]{16}", figures["directory"]))
        self.assertEqual({item["status"] for item in figures["items"]}, {"saved"})
        target = self.folder / figures["directory"]
        self.assertEqual((target / "missing.png").read_bytes(), PNG)
        self.assertTrue((target / "split_ratio.png").is_file())
        text = task_report.render_markdown(result)
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        self.assertEqual(sorted(links), sorted(f"{figures['directory']}/{name}" for name in ("missing.png", "split_ratio.png")))
        self.assertNotIn(str(self.folder), text)
        # 보고서 폴더를 옮겨도 상대 경로가 그대로 이어진다.
        moved = self.folder.parent / (self.folder.name + "_moved")
        shutil.copytree(self.folder, moved)
        self.addCleanup(shutil.rmtree, moved, True)
        self.assertTrue(all((moved / link).is_file() for link in links))

    def test_eda_image_is_separate_from_its_caption_in_markdown(self):
        text = task_report.render_markdown(self.save())
        image_pos = text.index("![missing.png](")
        self.assertEqual(text[image_pos - 2:image_pos], "\n\n")

    def test_same_content_reuses_folder_and_different_file_is_kept(self):
        first = self.save()
        target = self.folder / first["figures"]["directory"]
        stamp = (target / "split_ratio.png").stat().st_mtime_ns
        self.assertEqual(self.save()["figures"], first["figures"])
        self.assertEqual((target / "split_ratio.png").stat().st_mtime_ns, stamp)
        (target / "missing.png").write_bytes(b"other")
        result = self.save()
        item = next(item for item in result["figures"]["items"] if item["label"] == "missing.png")
        self.assertEqual((item["status"], item["reason"]), ("failed", report_figures.NOT_OVERWRITTEN))
        self.assertEqual((target / "missing.png").read_bytes(), b"other")

    def test_failures_are_reported_without_path_and_rest_is_saved(self):
        with patch.object(report_figures, "draw_chart", side_effect=RuntimeError(str(self.folder))):
            result = self.save(images=[])
        reasons = [item["reason"] for item in result["figures"]["items"]]
        self.assertEqual(reasons, [report_figures.DRAW_FAILED, report_figures.SOURCE_MISSING])
        self.assertIsNone(result["figures"]["directory"])
        self.assertEqual(sum(1 for item in result["warnings"] if "그림 저장 실패" in item["내용"]), 2)
        text = task_report.render_markdown(result)
        self.assertEqual(text.count("- 그림 저장 실패: "), 2)
        self.assertNotIn("![", text)
        self.assertNotIn(str(self.folder), text)

    def test_missing_font_and_write_failure_are_named(self):
        with patch("pathlib.Path.open", side_effect=OSError(str(self.folder))):
            items = self.save()["figures"]["items"]
        self.assertEqual({item["reason"] for item in items}, {report_figures.SAVE_FAILED})
        self.assertEqual([path for path in self.folder.rglob("*") if path.is_file()], [])
        with patch.object(report_figures, "_font", side_effect=LookupError):
            item = self.save()["figures"]["items"][0]
        self.assertEqual(item["reason"], report_figures.FONT_MISSING)

    def test_read_failure_keeps_existing_file(self):
        first = self.save()
        path = self.folder / first["figures"]["directory"] / "missing.png"
        with patch("pathlib.Path.read_bytes", side_effect=PermissionError(str(self.folder))):
            item = next(item for item in self.save()["figures"]["items"] if item["label"] == "missing.png")
        self.assertEqual((item["status"], item["reason"]), ("failed", report_figures.SAVE_FAILED))
        self.assertEqual(path.read_bytes(), PNG)

    def test_junction_folder_is_not_written(self):
        name = self.save()["figures"]["directory"]
        shutil.rmtree(self.folder / name)
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(self.folder / name), outside.name], capture_output=True)
        if created.returncode:
            self.skipTest("junction을 만들 수 없는 환경")
        self.addCleanup(os.rmdir, self.folder / name)
        result = self.save()
        self.assertEqual({item["reason"] for item in result["figures"]["items"]}, {report_figures.SAVE_FAILED})
        self.assertIsNone(result["figures"]["directory"])
        self.assertEqual(list(Path(outside.name).iterdir()), [])

    def test_attempt_must_be_integer(self):
        result = report_figures.save_figures(facts(), self.images, self.folder, "../../escape")
        self.assertEqual({item["reason"] for item in result["figures"]["items"]}, {report_figures.SAVE_FAILED})
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_user_settings_cannot_turn_on_tex(self):
        with matplotlib.rc_context(rc={"text.usetex": True}), patch("subprocess.check_output", side_effect=AssertionError):
            self.assertTrue(report_figures.draw_chart(chart()).startswith(b"\x89PNG"))

    def test_oversized_image_and_unsafe_label(self):
        images = [{"label": "../결측 그림.png", "data": PNG}, {"label": "big.png", "data": b"x" * (report_figures.FIGURE_LIMIT + 1)}]
        value = dict(facts(), images=[{"label": item["label"]} for item in images])
        items = self.save(value, images)["figures"]["items"]
        self.assertEqual([item["file"] for item in items], ["split_ratio.png", "eda_1.png", "big.png"])
        self.assertEqual(items[2]["reason"], report_figures.TOO_LARGE)

    def test_empty_chart_makes_no_file(self):
        value = dict(facts(), charts=[chart(rows=[])], images=[])
        self.assertEqual(self.save(value, [])["figures"], {"directory": None, "items": []})


class LayoutTest(unittest.TestCase):
    def test_sections_keep_requested_order_and_no_graph_section(self):
        opinions = {"notes": ["모델 문장입니다."], "columns": ["항목", "최종 값", "모델 의견"],
                    "rows": [{"항목": "preparation.outlier", "최종 값": "clip", "모델 의견": "<b>극단값</b>"}]}
        table = {"title": "표", "notes": [], "rows": [{"항목": "a"}]}
        target = chart("target 그래프", "eda", "target_distribution.png")
        text = task_report.render_markdown(dict(facts(), charts=[chart(), target], eda_tables=[table], decision_tables=[table],
                                                split_table={"title": "분할", "rows": [{"구간": "train"}]}, model_opinions=opinions))
        order = ["## 핵심 결과와 주요 문제", "## 처리 전 데이터와 기본 EDA", "### target 그래프", "### 분할 EDA 그래프 (검증된 PNG)",
                 "## 계획과 결정", "## 분할 결과", "### 분할 그래프", "## 컬럼별 처리와 제외 사유", "## 실제 전후 비교", "## 경고",
                 "## 참고 사항", "## 모델 의견과 한계 (계산 사실 아님)", "## 적용 구간과 결과 파일 활용"]
        positions = [text.index(heading) for heading in order]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("## 그래프", text)
        section = text[text.index("## 모델 의견과 한계"):text.index("## 적용 구간과")]
        self.assertIn("현재 로컬 모델의 한계", section)
        self.assertIn("&lt;b&gt;극단값&lt;/b&gt;", section)
        self.assertNotIn("극단값", text[:text.index("## 모델 의견과 한계")])


class OpinionTest(unittest.TestCase):
    def test_only_final_model_items_are_collected(self):
        plan = {"roles": {"a": "제외", "b": "수치형"}, "split": {"strategy": "random"}, "preparation": {"outlier": "clip", "scaling": "none"}}
        origins = {"roles.a": ("모델 권장", "<긴 문장>"), "preparation.outlier": ("모델 권장", "극단값"), "preparation.scaling": ("사용자 변경", "버린 문장")}
        result = evidence.model_opinions({}, plan, lambda _, path: origins.get(path, ("기본 계획", None)))
        self.assertEqual(result["rows"], [{"항목": "roles.a", "최종 값": "제외", "모델 의견": "<긴 문장>"},
                                          {"항목": "preparation.outlier", "최종 값": "clip", "모델 의견": "극단값"}])
        empty = evidence.model_opinions({}, plan, lambda *_: ("기본 계획", None))
        self.assertEqual(empty["rows"], [])
        self.assertIn("확정 계획에 남은 모델 의견이 없습니다.", empty["notes"])

    def test_processing_row_loses_model_sentence_only(self):
        rows = [{"컬럼": "a", "제외 사유": "모델 권장으로 제외(모델 의견: 비밀 문장)"}, {"컬럼": "b", "제외 사유": "사용자 제외"}, {"컬럼": "c"}]
        result = evidence.without_model_reason(rows)
        self.assertNotIn("비밀 문장", str(result))
        self.assertEqual(result[1:], rows[1:])
        self.assertEqual(rows[0]["제외 사유"], "모델 권장으로 제외(모델 의견: 비밀 문장)")
        # 모델이 이유를 쓰지 않은 항목은 모델 의견 절에 행이 없으므로 그 절을 가리키지 않는다.
        self.assertEqual(evidence.without_model_reason([{"제외 사유": evidence.NO_MODEL_REASON}]), [{"제외 사유": evidence.NO_MODEL_REASON}])


if __name__ == "__main__":
    unittest.main()

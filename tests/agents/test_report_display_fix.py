"""보고서 표시 보완 4건(질문 답변·표 열 순서·결측률 그래프·구간별 클래스 순서). 합성 사실만 쓰며 worker·모델은 쓰지 않는다."""

import io
import json
import unittest

import pandas as pd

from src.agents import tabular_report_evidence, task_report
from src.agents.task_plan import apply_answers, question
from tests.agents.test_tabular_report_scope import facts, record, sources


ROLE_OPTIONS = ("수치형", "범주형", "target", "식별자", "제외")
FIELDS = [{"path": "roles." + name, "type": "enum", "options": list(ROLE_OPTIONS)} for name in ("Passengerid", "2urvived", "a|b*")]
PLAN = {"roles": {"Passengerid": "수치형", "2urvived": "수치형", "a|b*": "범주형"}, "split": {}, "preparation": {}}


def _questions():
    # 실제 질문 형식: 역할 선택(path 있음, 응답값=역할)과 컬럼 선택(path 없음, 선택지가 roles.<컬럼>=target을 정함).
    role = question("identifier:0000000000000000", "advisory", "식별 번호라면 식별자로 지정하세요.", path="roles.Passengerid",
                    options=[{"value": value, "label": value} for value in ("식별자", "수치형", "제외")])
    target = question("target", "required", "target 컬럼을 고르세요.", options=[
        *[{"value": name, "label": name, "path": "roles." + name, "set": "target"} for name in ("2urvived", "a|b*")],
        {"value": "__no_target__", "label": "예측 대상 없음"}])
    return [role, target]


def _confirmed(answers):
    final, applied = apply_answers(PLAN, _questions(), answers, FIELDS)
    return {"confirmed": {"plan": final, "answers": applied}}


class AnswerDisplayTests(unittest.TestCase):
    def test_column_choice_is_not_shown_as_role(self):
        lines = task_report.answer_notes(_confirmed({"target": "2urvived", "identifier:0000000000000000": "식별자"}))
        text = "\n".join(lines)
        self.assertNotIn("roles.2urvived = 2urvived", text)
        self.assertIn("roles.2urvived = target", text)
        self.assertIn("고른 선택지: 2urvived", text)
        # 역할 선택 질문은 응답값이 곧 역할이므로 기존 표기를 유지한다.
        self.assertIn("질문 답변으로 정한 항목: roles.Passengerid = 식별자", lines)

    def test_choice_without_plan_change_and_unknown_legacy_value(self):
        lines = task_report.answer_notes(_confirmed({"target": "__no_target__"}))
        self.assertEqual(len(lines), 1)
        self.assertIn("바꾼 계획 항목 없음", lines[0])
        self.assertNotIn("roles.", lines[0])
        # 확정 계획에 없는 경로의 이전 기록은 값을 추정하지 않는다.
        legacy = {"confirmed": {"plan": PLAN, "answers": [{"id": "target", "path": "roles.gone", "value": "gone"}]}}
        self.assertIn("확인되지 않음", task_report.answer_notes(legacy)[0])
        self.assertEqual(task_report.answer_notes({"confirmed": None}), [])

    def test_special_characters_stay_plain_in_markdown(self):
        lines = task_report.answer_notes(_confirmed({"target": "a|b*"}))
        self.assertIn("roles.a|b* = target", lines[0])
        rendered = task_report._plain(lines[0])
        self.assertNotIn("|", rendered)
        self.assertIn("\\*", rendered)


# v1·v2·배열 대상 테스트(113건)와 기존 저장 보고서에서 수집한 표 머리글(조립 순서). 새 표가 생기면 여기에 더한다.
HEADERS = (
    "구분|컬럼|내용|처리", "구간|요청 비율|확정 비율|행 수|그룹 수|클래스|변환", "컬럼|역할|결정 출처|처리|제외 사유", "컬럼|역할|처리",
    "항목|대상 구간|컬럼 범위|단위|처리 전|처리 후|근거", "구간|요청 비율|확정 비율|행 수|막대 요청 비율|막대 확정 비율",
    "클래스|건수|비율|막대 건수", "항목|값|대상·분모|근거",
    "컬럼|역할|자료형|결측(건)|결측 비율|관측 고유값|상수·전체 결측|공백 문자열|+무한대|-무한대",
    "특징|관측 수|평균|표준편차|최솟값|25%|중앙값|75%|최댓값|비고", "컬럼|역할|유한 관측 수|하한|상한|후보 수|후보 비율",
    "특징|범주 수|빈 표기(건)|최빈 범주 점유율|희소 범주 수|희소 범주 행 비율", "순위|클래스|건수|비율",
    "특징 A|특징 B|Pearson|Spearman|쌍별 관측 수", "항목|내용",
    "구분|항목|설정 이름|최종 값|결정 출처|모델 의견|실행 기록 대조|실제 변경(근거)", "파일|용도|크기(byte)", "파일|크기(byte)",
    "구분|내용", "구분|대상|내용", "항목|값", "항목|값|결정 출처", "항목|전|후", "항목|전|후 (실측)|예상 (후보)|근거", "파일",
)


class ColumnOrderTests(unittest.TestCase):
    def _roundtrip(self, value):
        # 작업 기록 저장과 같은 키 정렬 후 다시 읽은 사실.
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))

    def test_every_known_header_survives_key_sorting(self):
        for header in HEADERS:
            keys = header.split("|")
            self.assertEqual(task_report.table_columns([dict(sorted((key, 0) for key in keys))]), keys)

    def _with_chart(self):
        split = tabular_report_evidence.split_chart(record()["confirmed"]["plan"]["split"], sources()["proposal"])
        return dict(facts(), goal="목표", technical={}, images=[], cautions=[], charts=[split])

    def test_saved_facts_render_with_the_original_column_order(self):
        result = self._with_chart()
        stored = self._roundtrip(result)
        self.assertEqual(task_report.render_markdown(stored), task_report.render_markdown(result))

        tables = [result["processing"], result["comparison"], result["split_table"]["rows"], *[table["rows"] for table in
                  result.get("eda_tables", []) + result.get("decision_tables", [])], *[chart["rows"] for chart in result["charts"]]]
        for rows in tables:
            if rows:
                self.assertEqual(task_report.table_columns(self._roundtrip(rows)), list(rows[0]))

    def test_identifier_and_split_columns_come_first(self):
        result = self._roundtrip(self._with_chart())
        self.assertEqual(task_report.table_columns(result["processing"])[0], "컬럼")
        self.assertEqual(task_report.table_columns(result["split_table"]["rows"])[:4], ["구간", "요청 비율", "확정 비율", "행 수"])
        chart = next(chart for chart in result["charts"] if chart["category"] == "구간")
        self.assertEqual(task_report.table_columns(chart["rows"])[:4], ["구간", "요청 비율", "확정 비율", "행 수"])

    def test_rows_with_different_keys_keep_every_column(self):
        rows = [{"처리": "a", "컬럼": "x"}, {"컬럼": "y", "역할": "target", "새 열": 1}]
        self.assertEqual(task_report.table_columns(rows), ["컬럼", "역할", "처리", "새 열"])
        table = task_report._table(rows)
        self.assertEqual(table[0], "| 컬럼 | 역할 | 처리 | 새 열 |")
        self.assertEqual(table[2], "| x |  | a |  |")


class MissingPlotTests(unittest.TestCase):
    def _plot(self, ratios):
        from src.preprocessing.eda_report import plot_missing
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frame = pd.DataFrame({"missing_count": [round(r * 1309) for r in ratios], "missing_ratio": ratios},
                             index=[f"f{index}" for index in range(len(ratios))])
        figure = plot_missing({"missing": frame}, 12)
        axis = figure.axes[0]
        texts = [text.get_text() for text in axis.texts]
        heights = sorted(patch.get_height() for patch in axis.patches)
        bottom, top = axis.get_ylim()
        png = io.BytesIO()
        figure.savefig(png, format="png")
        plt.close(figure)
        return {"texts": texts, "heights": heights, "bottom": bottom, "top": top, "label": axis.get_ylabel(), "png": png.getvalue()}

    def test_small_ratio_is_labelled_without_inflating_the_bar(self):
        result = self._plot([2 / 1309, 0.0, 0.0])
        self.assertEqual(result["bottom"], 0)
        self.assertLess(result["top"], 1)
        self.assertGreater(result["top"], 2 / 1309)
        self.assertAlmostEqual(result["heights"][-1], 2 / 1309)
        self.assertIn("0.15%", result["texts"])
        self.assertEqual(result["texts"].count("0%"), 2)
        self.assertIn("%", result["label"])

    def test_mixed_and_high_ratios_keep_true_scale(self):
        mixed = self._plot([0.8, 2 / 1309, 0.0])
        self.assertEqual(mixed["bottom"], 0)
        self.assertGreaterEqual(mixed["top"], 0.8)
        self.assertLessEqual(mixed["top"], 1)
        self.assertEqual(mixed["heights"], sorted([0.8, 2 / 1309, 0.0]))
        self.assertIn("80%", mixed["texts"])
        self.assertIn("0.15%", mixed["texts"])
        self.assertIn("0%", mixed["texts"])
        self.assertEqual(self._plot([1.0])["top"], 1)

    def test_all_zero_says_no_missing(self):
        result = self._plot([0.0, 0.0])
        self.assertEqual(result["bottom"], 0)
        self.assertTrue(any("No missing values" in text for text in result["texts"]))
        self.assertTrue(result["png"].startswith(b"\x89PNG"))


class ClassOrderTests(unittest.TestCase):
    def test_every_split_uses_the_same_class_order(self):
        value = sources()
        value["proposal"]["summary"][1]["classes"] = [{"label": "False", "count": 1}, {"label": "True", "count": 1}]
        # 표기가 같아 보여도 타입이 다른 클래스는 합치지 않는다.
        value["proposal"]["summary"][2]["classes"] = [{"label": 1, "count": 1}, {"label": "False", "count": 2}, {"label": "True", "count": 1}]
        split = {row["구간"]: row["클래스"] for row in facts(record(), value)["split_table"]["rows"]}
        self.assertEqual(split["train"], "True: 3, False: 3")
        self.assertEqual(split["validation"], "True: 1, False: 1")
        self.assertEqual(split["test"], "True: 1, False: 2, 1: 1")


if __name__ == "__main__":
    unittest.main()

"""보고서 표시 보완 4건(질문 답변·표 열 순서·결측률 그래프·구간별 클래스 순서). 합성 사실만 쓰며 worker·모델은 쓰지 않는다."""

import io
import json
import unittest
from unittest.mock import patch

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
    "구분|항목|설정 이름|최종 값|결정 출처|실행 기록 대조|실제 변경(근거)", "항목|최종 값|모델 의견", "파일|용도|크기(byte)", "파일|크기(byte)",
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


IMPACT_TITLE = "예상 처리 영향 (Train 후보 계산, 영향이 있는 특징만)"
# UI_DISPLAY_ACCEPTANCE 3절의 고정 정답. 제품 코드의 순서(COLUMN_ORDER·조립 결과)에서 가져오지 않는다.
IMPACT_HEADER = ["특징", "변환 전 고유값", "clipping 변경 값", "구간 수", "출력 열", "변환 후 상수", "정보 손실",
                 # 값 변화 근거 3개 열(기존 7개 열 순서 뒤에 추가, test_preparation_scope_guidance 참고).
                 "변환 후 고유값", "값 변화", "관련 설정"]
# 아래 _review_facts 행 조합의 기대 셀(화면 표시 글자). 열끼리 값 벡터가 달라 열 교환을 판별한다.
# x: clip 2/(Train 6행 - 결측 1)=40.0%, c: 원핫 2열, b: 구간 3개·변환 후 상수(변환 전 3종이라 정보 손실), k: 원래 상수(정보 손실 아님).
IMPACT_CELLS = [
    ["x", "5", "2 (40.0%)", "-", "1", "아니오", "-", "미집계", "-", "-"],
    ["c", "2", "미집계", "-", "2", "아니오", "-", "미집계", "-", "-"],
    ["b", "3", "1 (16.7%)", "3", "1", "예", "예", "1", "상수화(여러 값 → 1)", "구간화 · 극단값 자르기"],
    ["k", "1", "미집계", "-", "1", "예", "-", "1", "원래 상수(변환 전 1종)", "-"],
]
# 같은 화면 인접 표의 고정 머리글(제목 → 머리글).
ADJACENT_HEADERS = {
    "특징별 처리": ["컬럼", "역할", "결정 출처", "처리", "제외 사유"],
    "제거 후보": ["특징", "사유"],
    "출력 폭 (구간화·원핫)": ["구분", "입력 특징", "출력 열"],
    "실제 실행 설정 (후보 계산에 쓴 값)": ["항목", "설정 이름", "값"],
}


def _impact(column, kind, width, distinct, *, clipped=None, constant=False, missing=0):
    return {"column": column, "kind": kind, "width": width, "missing": missing, "imputed": 0, "clipped": clipped,
            "distinct_before": distinct, "constant_after": constant, "unknown": 0, "missing_after": 0}


def _numeric(column, **extra):
    return dict({"column": column, "kind": "numeric", "impute": 1.0, "clip": None, "center": 0.0, "scale": 1.0, "edges": None,
                 "width": 1}, **extra)


def _review_facts(*, affected=True):
    """실제 review_facts로 결정 D 화면 사실을 만든다(키 정렬 전). 후보·전처리 근거는 문서 3절의 표기 규칙과 맞춘다.

    affected=False면 영향 집계는 있으나 표시 조건(clip·상수·구간화·출력 폭>1)에 해당하는 특징이 없는 정상 후보다.
    """
    from src.agents import tabular_task

    task_record = record()
    value = sources() if affected else sources(categories=('["str", "a"]',))
    candidate, preprocessor = value["candidate"], value["preprocessor"]

    if affected:
        # 합성 집계다. 실제 clipping·구간화 계산을 검증하지 않는다.
        task_record["confirmed"]["plan"]["roles"].update(b="수치형", k="수치형")
        task_record["diagnostic"]["columns"] += [{"column": "b", "missing_count": 0, "missing_ratio": 0.0, "observed_unique": 3},
                                                 {"column": "k", "missing_count": 0, "missing_ratio": 0.0, "observed_unique": 1}]
        candidate["retained"] += ["b", "k"]
        candidate["impact"][0].update(clipped=2)
        candidate["impact"] += [_impact("b", "numeric", 1, 3, clipped=1, constant=True), _impact("k", "numeric", 1, 1, constant=True)]
        preprocessor["state"]["records"][0].update(clip=[0.0, 1.0])
        preprocessor["state"]["records"] += [_numeric("b", clip=[0.0, 2.0], edges=[1.0, 2.0]), _numeric("k")]

    # 결정 출처 계산은 이 검사 대상이 아니다. 기존 보고서 범위 fixture와 같이 고정한다.
    with patch.object(tabular_task, "_origin", return_value=("기본 계획", None)):
        return tabular_task.review_facts(task_record, candidate, preprocessor, None)


def _task(facts_value, *, stored=True):
    """결정 D 작업 기록. stored면 작업 기록 저장 후 다시 읽은 것처럼 키를 정렬한다."""
    candidate = {"signature": "s", "facts": facts_value}

    if stored:
        candidate = json.loads(json.dumps(candidate, ensure_ascii=False, sort_keys=True))

    return {"task_id": "0" * 32, "kind": "tabular_v2", "attempt": 1, "phase": "awaiting_apply", "candidate": candidate}


def _built(facts_value, title):
    return next(table["rows"] for table in facts_value["tables"] if table["title"] == title)


class ApplyScreenImpactTableTests(unittest.TestCase):
    """UI_DISPLAY_ACCEPTANCE 4절 A01~A06. 실제 review_facts → 저장 후보 facts → _render_apply 경로를 쓴다."""

    def _render(self, task):
        """적용 화면을 그려 제목별 바로 뒤 요소(표면 DataFrame, 빈 표면 ('text', 글자))를 돌려준다."""
        from streamlit.testing.v1 import AppTest

        def app():
            from unittest.mock import patch
            import streamlit as st
            from src.agents import task_orchestrator
            from src.ui import task_view

            # 계획 수정 버튼 노출 판단은 작업 저장소를 읽는다. 표 표시와 무관하므로 고정한다.
            with patch.object(task_orchestrator, "apply_revisable", return_value=False):
                task_view._render_apply(st, st.session_state["review_task"])

        result = AppTest.from_function(app)
        result.session_state["review_task"] = task
        result.run(timeout=60)
        self.assertFalse(result.exception, result.exception)

        def walk(node):
            for child in getattr(node, "children", {}).values():
                if getattr(child, "children", None):
                    yield from walk(child)
                else:
                    yield child

        # 빈 표는 '(없음)' 글자로 그려져 제목 순번과 DataFrame 순번이 다르다. 화면 순서에서 제목 다음 요소를 짝짓는다.
        elements = list(walk(result.main))
        sections = {}

        for index, element in enumerate(elements[:-1]):
            if element.type == "markdown" and element.value.startswith("#### "):
                following = elements[index + 1]
                sections[element.value[5:]] = following.value if following.type == "dataframe" else ("text", following.value)

        return sections

    def _cells(self, frame):
        return [list(row) for row in frame.values.tolist()]

    def test_a01_a02_fixed_header_cells_and_row_order(self):
        built_facts = _review_facts()
        frame = self._render(_task(built_facts))[IMPACT_TITLE]
        self.assertEqual(list(frame.columns), IMPACT_HEADER)
        self.assertEqual(self._cells(frame), IMPACT_CELLS)
        # 조립된 사실과의 대응도 유지한다(화면은 글자로 표시).
        self.assertEqual(self._cells(frame), [[str(row[key]) for key in IMPACT_HEADER] for row in _built(built_facts, IMPACT_TITLE)])

    def test_a03_missing_keys_stay_blank_in_the_fixed_order(self):
        task = _task(_review_facts())
        rows = _built(task["candidate"]["facts"], IMPACT_TITLE)
        del rows[1]["구간 수"]
        del rows[2]["정보 손실"]
        frame = self._render(task)[IMPACT_TITLE]
        self.assertEqual(list(frame.columns), IMPACT_HEADER)
        expected = [list(row) for row in IMPACT_CELLS]
        expected[1][3], expected[2][6] = "", ""
        self.assertEqual(self._cells(frame), expected)

    def test_a04_no_affected_row_shows_title_and_empty_notice(self):
        facts_value = _review_facts(affected=False)
        self.assertEqual(_built(facts_value, IMPACT_TITLE), [])
        self.assertEqual(self._render(_task(facts_value))[IMPACT_TITLE], ("text", "(없음)"))

        # 영향 집계가 없는 이전 후보는 표 자체를 생략한다. A04의 입력·정답과 다른 경우다.
        from src.agents import tabular_task

        legacy = sources()
        legacy["candidate"].pop("impact")

        with patch.object(tabular_task, "_origin", return_value=("기본 계획", None)):
            old = tabular_task.review_facts(record(), legacy["candidate"], legacy["preprocessor"], None)

        self.assertNotIn(IMPACT_TITLE, self._render(_task(old)))

    def test_a05_adjacent_tables_keep_headers_values_and_rows(self):
        built_facts = _review_facts()
        sections = self._render(_task(built_facts))

        for title, header in ADJACENT_HEADERS.items():
            frame = sections[title]
            rows = _built(built_facts, title)
            self.assertTrue(rows, title)
            self.assertEqual(list(frame.columns), header, title)
            self.assertEqual(self._cells(frame), [[str(row[key]) for key in header] for row in rows], title)

        self.assertEqual([row[0] for row in self._cells(sections["특징별 처리"])], ["x", "c", "d", "label", "id", "b", "k"])
        self.assertEqual(self._cells(sections["제거 후보"])[0][0], "d")

    def test_a06_built_and_stored_facts_render_the_same(self):
        built_facts = _review_facts()
        fresh = self._render(_task(built_facts, stored=False))
        stored = self._render(_task(built_facts))
        self.assertEqual(list(fresh), list(stored))
        self.assertIn(IMPACT_TITLE, fresh)

        for title, shown in fresh.items():
            if isinstance(shown, tuple):
                self.assertEqual(shown, stored[title], title)
            else:
                self.assertEqual(list(shown.columns), list(stored[title].columns), title)
                self.assertEqual(self._cells(shown), self._cells(stored[title]), title)

def _screen(render):
    """render(st)를 AppTest로 그려 화면 순서의 (요소 종류, 값) 목록을 돌려준다. 표는 DataFrame 그대로다."""
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st
        st.session_state["render"](st)

    result = AppTest.from_function(app)
    result.session_state["render"] = render
    result.run(timeout=60)
    assert not result.exception, result.exception

    def walk(node):
        for child in getattr(node, "children", {}).values():
            if getattr(child, "children", None):
                yield from walk(child)
            else:
                yield child

    return [(element.type, element.value) for element in walk(result.main)]


def _frames(elements):
    return [value for kind, value in elements if kind == "dataframe"]


class ExplicitColumnOrderTests(unittest.TestCase):
    """표를 만드는 쪽이 정한 열 순서. 고정 기대값은 요청한 순서를 그대로 적고 제품 상수에서 가져오지 않는다."""

    def test_explicit_order_keeps_blank_cells_and_extra_columns(self):
        rows = [{"b": 1, "a": 2}, {"a": 3, "extra": 4}]
        self.assertEqual(task_report.table_columns(rows, ["a", "b"]), ["a", "b", "extra"])
        self.assertEqual(task_report._table(rows, ["a", "b"]),
                         ["| a | b | extra |", "| --- | --- | --- |", "| 2 | 1 |  |", "| 3 |  | 4 |"])
        # 순서 정보가 없으면 기존 방식(COLUMN_ORDER·첫 출현) 그대로다.
        self.assertEqual(task_report.table_columns(rows), task_report.table_columns(rows, None))
        frame = _frames(_screen(lambda st: __import__("src.ui.task_view", fromlist=["_rows"])._rows(st, rows, ["a", "b"])))[0]
        self.assertEqual(list(frame.columns), ["a", "b", "extra"])
        self.assertEqual(frame.values.tolist(), [["2", "1", ""], ["3", "", "4"]])

    def test_quality_findings(self):
        from types import SimpleNamespace
        from src.ui import task_view

        problem = {"findings": [{"code": "CODE_A", "subject": "col|*1"}, {"code": "CODE_B", "subject": None}]}
        frame = _frames(_screen(lambda st: task_view._render_findings(st, SimpleNamespace(FINDING_LABELS={"CODE_A": "지적 A"}), problem)))[0]
        self.assertEqual(list(frame.columns), ["지적", "대상"])
        self.assertEqual(frame.values.tolist(), [["지적 A", "col|*1"], ["CODE_B", "전체"]])

    def test_recommended_changes(self):
        from src.ui import task_view

        state = {"recommended": [{"path": "preparation.scaling", "value": "standard", "reason": "r1", "check": "확인 1"},
                                 {"path": "roles.a", "value": "식별자", "reason": "r2"}],
                 "changes": [{"path": "preparation.scaling", "before": "none"}, {"path": "roles.a", "before": "수치형"}]}
        frame = _frames(_screen(lambda st: task_view._render_recommended(st, state)))[0]
        self.assertEqual(list(frame.columns), ["항목", "기본값", "권장값", "근거", "확인"])
        self.assertEqual(frame.values.tolist(), [["preparation.scaling", "none", "standard", "모델 의견: r1", "확인 1"],
                                                 ["roles.a", "수치형", "식별자", "모델 의견: r2", "-"]])

    def test_option_guide(self):
        from src.ui import task_view

        guide = [{"label": "스케일링", "name": "scaling", "value_text": "평균 0", "value": "standard", "source": "사용자 변경",
                  "applies": "수치형 특징", "basic": True, "what": "w", "warnings": [], "facts": [], "opinion": None,
                  "caution": "c", "off": "o"}]
        frame = _frames(_screen(lambda st: task_view._render_guide(st, guide, "옵션 설명")))[0]
        self.assertEqual(list(frame.columns), ["옵션", "설정 이름", "현재 값", "출처", "적용 대상"])
        self.assertEqual(frame.values.tolist(), [["스케일링", "scaling", "평균 0", "사용자 변경", "수치형 특징"]])

    def test_final_plan_preview_tables(self):
        from src.agents import task_adapters, task_orchestrator
        from src.ui import task_view

        plan = {"roles": {"a": "수치형"}, "split": {"stratify": True}, "preparation": {"scaling": "standard"}}
        preview = {"plan": plan, "guide": [], "digest": "d",
                   "origins": {"roles.a": {"source": "base"}, "split.stratify": {"source": "answer"},
                               "preparation.scaling": {"source": "model", "reason": "모델 이유"}},
                   "changes": [{"path": "preparation.scaling", "before": "none", "after": "standard"}]}
        task = {"task_id": "0" * 32, "attempt": 1, "kind": "tabular_v2"}

        def render(st):
            with patch.object(task_orchestrator, "preview_plan", return_value=preview):
                task_view._render_final_plan(st, task, task_adapters.adapter("tabular_v2"), {}, "k",
                                             {"plan": plan, "answers": {}})

        frames = _frames(_screen(render))
        changed = next(frame for frame in frames if "모델 의견" in frame.columns)
        self.assertEqual(list(changed.columns), ["항목", "값", "출처", "모델 의견"])
        self.assertEqual(changed.values.tolist(), [["preparation.scaling", "standard", "모델 권장", "모델 이유"],
                                                   ["split.stratify", "True", "질문 답변", ""]])
        edited = next(frame for frame in frames if "표시한 값" in frame.columns)
        self.assertEqual(list(edited.columns), ["항목", "표시한 값", "최종 값"])
        self.assertEqual(edited.values.tolist(), [["preparation.scaling", "none", "standard"]])

    def test_array_channel_table_from_stored_facts(self):
        from src.agents import array_task, task_orchestrator
        from src.ui import task_view

        state = {"channels": [{"channel": "ch0", "impute": 1.5, "clip": [0.0, 2.0], "center": 0.5, "scale": 2.0},
                              {"channel": "ch1", "impute": None, "clip": None, "center": 1.0, "scale": 1.0}]}
        impact = [{"clipped": 3, "values": 30, "imputed": 2, "constant_after": False},
                  {"clipped": None, "values": 30, "imputed": 0, "constant_after": True}]
        table = array_task._channel_table(state, impact, 10)
        header = ["채널", "대치값(Train 중앙값)", "clip 경계", "center", "scale", "clipping 변경 값", "대치한 값", "변환 후 상수"]
        self.assertEqual(table["columns"], header)
        built = table["rows"]
        facts_value = {"metrics": [], "tables": [table], "text": None, "text_truncated": False}
        stored = json.loads(json.dumps({"signature": "s", "facts": facts_value}, ensure_ascii=False, sort_keys=True))
        task = {"task_id": "0" * 32, "kind": array_task.KIND, "attempt": 1, "phase": "awaiting_apply", "candidate": stored}

        def render(st):
            with patch.object(task_orchestrator, "apply_revisable", return_value=False):
                task_view._render_apply(st, task)

        frame = _frames(_screen(render))[0]
        self.assertEqual(list(frame.columns), header)
        self.assertEqual(frame.values.tolist(), [[str(row[key]) for key in header] for row in built])

    def test_stored_facts_without_column_order_keep_current_display(self):
        from src.agents import task_orchestrator
        from src.ui import task_view

        # 열 순서 정보가 없는 이전 facts: 저장 정렬 + COLUMN_ORDER의 현재 표시 그대로다.
        rows = [{"항목": "x", "근거": "r", "기본값": 1}]
        facts_value = {"metrics": [], "tables": [{"title": "이전 표", "rows": rows}], "text": None, "text_truncated": False}
        task = {"task_id": "0" * 32, "kind": "tabular_v2", "attempt": 1, "phase": "awaiting_apply",
                "candidate": json.loads(json.dumps({"signature": "s", "facts": facts_value}, ensure_ascii=False, sort_keys=True))}

        def render(st):
            with patch.object(task_orchestrator, "apply_revisable", return_value=False):
                task_view._render_apply(st, task)

        frame = _frames(_screen(render))[0]
        self.assertEqual(list(frame.columns), task_report.table_columns([dict(sorted(rows[0].items()))]))


# 배열 표의 고정 정답(요청한 순서). 제품 상수나 조립 결과에서 가져오지 않는다.
ARRAY_DIAGNOSIS_HEADER = ["배열", "형상", "자료형", "구분", "비유한 값", "값 종류", "평균", "최솟값(로컬)", "최댓값(로컬)",
                          "상수 표본", "동일 표본"]
ARRAY_SPLIT_HEADER = ["구간", "표본", "클래스", "그룹"]
ARRAY_SPLIT_CELLS = [["train", "44", "0: 20, 1: 24", "7"], ["validation", "11", "0: 6, 1: 5", "2"], ["test", "9", "0: 4, 1: 5", "-"]]


def _array_summary():
    """배열 진단 요약 입력. 이름순으로 1차원 라벨 후보(a)가 표본 배열(x)보다 먼저 오고, 사용하지 않는 배열(z)도 있다."""
    return {"headers": {"a": {"shape": [64], "dtype": "<i8"}, "x": {"shape": [64, 4, 4, 2], "dtype": "<f4"},
                        "z": {"shape": [3, 3], "dtype": "<f8"}},
            "samples": {"x": {"overall": {"nonfinite": 1, "distinct": 300, "mean": 29.5432, "min": 0.058581, "max": 99.8612},
                              "constant_samples": 2, "duplicate_samples": 3}},
            "meta": {"a": {"missing": 0, "unique": 2}}}


def _array_assignment():
    return {"summary": [{"split": "train", "samples": 44, "classes": [{"value": 0, "count": 20}, {"value": 1, "count": 24}], "groups": 7},
                        {"split": "validation", "samples": 11, "classes": [{"value": 0, "count": 6}, {"value": 1, "count": 5}], "groups": 2},
                        {"split": "test", "samples": 9, "classes": [{"value": 0, "count": 4}, {"value": 1, "count": 5}], "groups": None}]}


def _array_apply_task(tables, *, stored=True):
    from src.agents import array_task

    return dict(_task({"metrics": [], "tables": tables, "text": None, "text_truncated": False}, stored=stored), kind=array_task.KIND)


def _render_apply_frames(task):
    from src.agents import task_orchestrator
    from src.ui import task_view

    def render(st):
        with patch.object(task_orchestrator, "apply_revisable", return_value=False):
            task_view._render_apply(st, task)

    return _frames(_screen(render))


class ArrayTableOrderTests(unittest.TestCase):
    """배열 진단 요약·결정 D 구간별 표본. 실제 어댑터 조립 함수와 화면 렌더 함수를 쓴다."""

    def test_diagnosis_summary_on_review_screen(self):
        from src.agents import array_task
        from src.ui import task_view

        task = {"task_id": "0" * 32, "kind": array_task.KIND, "attempt": 1}

        def render(st):
            with patch.object(array_task, "_summary", return_value=_array_summary()), \
                    patch.object(array_task, "input_notices", return_value=[]):
                task_view._render_diagnosis(st, task, array_task)

        frame = _frames(_screen(render))[0]
        self.assertEqual(list(frame.columns), ARRAY_DIAGNOSIS_HEADER)
        self.assertEqual(frame.values.tolist(), [
            ["a", "64", "<i8", "1차원(라벨·그룹·ID 후보)", "0", "2", "", "", "", "", ""],
            ["x", "64×4×4×2", "<f4", "표본 배열 후보", "1", ">256", "29.54", "0.05858", "99.86", "2", "3"],
            ["z", "3×3", "<f8", "사용 안 함", "", "", "", "", "", "", ""]])

    def test_split_table_on_apply_screen_built_and_stored(self):
        from src.agents import array_task

        table = array_task._split_table(_array_assignment())
        self.assertEqual(table["title"], "구간별 표본")

        for stored in (False, True):
            frame = _render_apply_frames(_array_apply_task([table], stored=stored))[0]
            self.assertEqual(list(frame.columns), ARRAY_SPLIT_HEADER, stored)
            self.assertEqual(frame.values.tolist(), ARRAY_SPLIT_CELLS, stored)

    def test_split_table_missing_cell_and_extra_column(self):
        from src.agents import array_task

        table = json.loads(json.dumps(array_task._split_table(_array_assignment()), ensure_ascii=False))
        del table["rows"][2]["그룹"]
        table["rows"][0]["추가 열"] = "보존"
        frame = _render_apply_frames(_array_apply_task([table]))[0]
        self.assertEqual(list(frame.columns), ARRAY_SPLIT_HEADER + ["추가 열"])
        self.assertEqual(frame.values.tolist(), [ARRAY_SPLIT_CELLS[0] + ["보존"], ARRAY_SPLIT_CELLS[1] + [""],
                                                 ARRAY_SPLIT_CELLS[2][:3] + ["", ""]])

    def test_array_apply_screen_keeps_both_tables(self):
        from src.agents import array_task

        state = {"channels": [{"channel": "ch0", "impute": None, "clip": [0.0, 2.0], "center": 0.5, "scale": 2.0}]}
        impact = [{"clipped": 3, "values": 30, "imputed": 0, "constant_after": False}]
        frames = _render_apply_frames(_array_apply_task([array_task._split_table(_array_assignment()),
                                                          array_task._channel_table(state, impact, 10)]))
        self.assertEqual([list(frame.columns) for frame in frames],
                         [ARRAY_SPLIT_HEADER, ["채널", "대치값(Train 중앙값)", "clip 경계", "center", "scale", "clipping 변경 값",
                                               "대치한 값", "변환 후 상수"]])
        self.assertEqual(frames[1].values.tolist(), [["ch0", "-", "0.0 ~ 2.0", "0.5", "2.0", "3 (10.0%)", "0", "아니오"]])


if __name__ == "__main__":
    unittest.main()

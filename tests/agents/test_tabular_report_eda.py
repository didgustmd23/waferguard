"""처리 전 EDA·계획과 결정 절과 분할 EDA 근거 읽기를 검사한다.

실제 EDA 생성 함수로 만든 합성 결과 바이트를 쓴다. 원본 데이터 재읽기·worker·모델 호출은 없다.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

from src.agents import eda_result_validation, tabular_report_evidence as evidence, tabular_task, task_report
from src.agents import tabular_workflow_service as service
from src.agents.artifacts import ArtifactContext
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.eda import collect_eda
from src.preprocessing.eda_report import write_eda_report
from src.preprocessing.input_profile import profile_input
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules
from tests.agents import test_tabular_report_scope as scope


# build()가 실제 EDA 그림을 이 프로세스에서 그린다. 기본 Tk 백엔드 객체가 뒤의 화면 테스트 스레드에서 정리되면 Tcl이 종료된다.
matplotlib.use("Agg")

ROLE_NAMES = {"numeric": "수치형", "categorical": "범주형"}


def build(frame, roles, directory):
    """실제 EDA 생성기로 결과 폴더를 만들고 (EDA 바이트, 진단, 확정 계획 역할)을 돌려준다."""
    data = TabularInput(frame, roles)
    write_eda_report(data, collect_eda(data), directory, max_plot_features=12)
    payloads = {path.name: path.read_bytes() for path in sorted(directory.iterdir())}
    profiled = profile_input(frame)
    diagnostic = {"summary": profiled["summary"], "columns": json.loads(profiled["columns"].to_json(orient="records"))}
    plan_roles = {name: "제외" for name in frame.columns}
    plan_roles.update({name: "수치형" for name in roles.numeric})
    plan_roles.update({name: "범주형" for name in roles.categorical})
    plan_roles.update({name: "식별자" for name in roles.identifiers})
    if roles.target:
        plan_roles[roles.target] = "target"
    plan = {"roles": plan_roles, "split": dict(asdict(SplitRules()), strategy="random"), "preparation": asdict(PreparationRules())}
    return payloads, diagnostic, plan


def mixed_frame():
    # x는 결측 3칸, z는 상수(상관 계산 불가), c는 '0012'·'12'·'NA' 문자열과 결측, label은 문자열 '0012'·'1'.
    return pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, np.nan, 10.0, 11.0, 100.0],
        "y": [2.0, 4.1, 6.0, 8.2, 9.9, 12.0, 14.1, 16.0, 18.2, 20.0, 22.1, 24.0],
        "z": [1.0] * 12,
        "c": ["0012", "12", "NA", "0012", None, "12", "0012", "NA", "0012", "12", "0012", None],
        "label": ["0012", "1"] * 6,
        "id": [f"r{index}" for index in range(12)],
    })


MIXED_ROLES = ColumnRoles(numeric=("x", "y", "z"), categorical=("c",), target="label", identifiers=("id",))


def table(tables, title):
    return next(item for item in tables if item["title"] == title)


class EdaFactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        root = Path(cls.scratch.name)
        cls.mixed = build(mixed_frame(), MIXED_ROLES, root / "mixed")
        categorical = pd.DataFrame({"c": ["a", "b", "a", "b", "a", "c"], "d": ["p", "p", "q", "q", "p", "p"]})
        cls.categorical_only = build(categorical, ColumnRoles(categorical=("c", "d")), root / "categorical")
        single = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "c": list("aabbcc"), "label": [0, 1] * 3})
        cls.single = build(single, ColumnRoles(numeric=("x",), categorical=("c",), target="label"), root / "single")
        names = [f"feature_{index:02d}_" + "w" * 100 for index in range(25)]
        rng = np.random.default_rng(0)
        wide = pd.DataFrame(rng.normal(size=(30, 25)), columns=names)
        wide.iloc[:3, 0] = np.nan
        cls.wide = build(wide, ColumnRoles(numeric=tuple(names)), root / "wide")

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def facts(self, case):
        return evidence.eda_tables(*case)

    def test_mixed_input_keeps_scopes_and_preserves_strings(self):
        tables = self.facts(self.mixed)
        self.assertEqual([item["title"] for item in tables],
                         ["입력 개요", "컬럼별 결측·고유값·비유한 값", "수치 특징 기술 통계", "IQR 이상치 후보 (진단용)",
                          "범주 특징 요약", "target 클래스 분포", "수치 특징 상관 (쌍별)"])
        for item in tables:
            # 표마다 대상·검증 수준을 적는다. 진단(모든 컬럼)과 분할 EDA(특징 역할) 집계를 한 표에 섞지 않는다.
            self.assertTrue(any(note.startswith("대상:") for note in item["notes"]) or item["title"] == "입력 개요", item["title"])
            self.assertTrue(any(note.startswith("검증 수준:") for note in item["notes"]), item["title"])
        overview = {row["항목"]: row for row in table(tables, "입력 개요")["rows"]}
        self.assertEqual(overview["행 수"]["값"], 12)
        self.assertIn("모든 컬럼", overview["행 수"]["대상·분모"])
        self.assertIn("특징 역할 컬럼 4개", overview["특징 값이 같은 행"]["대상·분모"])
        self.assertEqual(overview["target"]["값"], "있음")

        columns = table(tables, "컬럼별 결측·고유값·비유한 값")
        self.assertEqual(columns["rows"][0]["컬럼"], "x")
        self.assertEqual(columns["rows"][0]["결측(건)"], 3)
        self.assertIn("모든 컬럼 6개 · 분모 12행", columns["notes"][0])
        self.assertEqual({row["컬럼"]: row["상수·전체 결측"] for row in columns["rows"]}["z"], "예")

        numeric = {row["특징"]: row for row in table(tables, "수치 특징 기술 통계")["rows"]}
        self.assertEqual(numeric["x"]["관측 수"], 9)
        self.assertEqual(numeric["z"]["비고"], "상수(표준편차 0)")

        outliers = table(tables, "IQR 이상치 후보 (진단용)")
        self.assertEqual([row["컬럼"] for row in outliers["rows"]], ["x"])
        self.assertTrue(any("오류 판정이 아닙니다" in note for note in outliers["notes"]))

        categories = table(tables, "범주 특징 요약")
        row = categories["rows"][0]
        # '0012'·'12'·'NA'는 표기 그대로 서로 다른 범주이며 'NA' 문자열을 결측으로 바꾸지 않는다. 결측 2칸은 빈 표기다.
        self.assertEqual((row["범주 수"], row["빈 표기(건)"]), (3, 2))
        self.assertEqual(row["최빈 범주 점유율"], "41.7%")
        self.assertNotIn("0012", json.dumps(categories["rows"], ensure_ascii=False))
        self.assertNotIn("NA", json.dumps(categories["rows"], ensure_ascii=False))

        target = table(tables, "target 클래스 분포")
        self.assertEqual({row["클래스"] for row in target["rows"]}, {"0012", "1"})
        self.assertEqual([row["건수"] for row in target["rows"]], [6, 6])

        correlations = table(tables, "수치 특징 상관 (쌍별)")
        pairs = {(row["특징 A"], row["특징 B"]): row for row in correlations["rows"]}
        self.assertEqual(set(pairs), {("x", "y"), ("x", "z"), ("y", "z")})
        self.assertEqual(pairs[("x", "z")]["Pearson"], evidence.NOT_COMPUTED)
        self.assertNotEqual(pairs[("x", "y")]["Pearson"], evidence.NOT_COMPUTED)
        self.assertEqual(pairs[("x", "y")]["쌍별 관측 수"], 9)
        # 계산 가능한 쌍을 먼저, 계산 불가 쌍은 뒤에 둔다. 0으로 바꾸지 않는다.
        self.assertEqual(list(pairs)[0], ("x", "y"))
        self.assertNotIn("0", [row["Pearson"] for row in correlations["rows"]])
        self.assertTrue(any("계산 불가 2쌍" in note for note in correlations["notes"]))

    def test_no_target_no_numeric_and_single_numeric(self):
        tables = self.facts(self.categorical_only)
        self.assertEqual(table(tables, "target 클래스 분포")["rows"], [])
        self.assertIn("해당 없음", table(tables, "target 클래스 분포")["notes"][0])
        self.assertEqual(table(tables, "수치 특징 기술 통계")["rows"], [])
        self.assertTrue(any("수치형 특징이 없어" in note for note in table(tables, "수치 특징 기술 통계")["notes"]))
        self.assertIn("수치형 특징이 없어", table(tables, "수치 특징 상관 (쌍별)")["notes"][0])
        self.assertEqual(table(tables, "IQR 이상치 후보 (진단용)")["rows"], [])
        self.assertEqual({row["항목"]: row["값"] for row in table(tables, "입력 개요")["rows"]}["target"], "없음(해당 없음)")
        categories = table(tables, "범주 특징 요약")["rows"]
        self.assertEqual([row["특징"] for row in categories], ["c", "d"])

        single = self.facts(self.single)
        self.assertEqual(table(single, "수치 특징 상관 (쌍별)")["rows"], [])
        self.assertIn("1개라", table(single, "수치 특징 상관 (쌍별)")["notes"][0])
        # 정수 target 라벨도 저장 표기 그대로 보인다.
        self.assertEqual({row["클래스"] for row in table(single, "target 클래스 분포")["rows"]}, {"0", "1"})

    def test_wide_table_and_long_names_show_selection_and_omitted_counts(self):
        tables = self.facts(self.wide)
        numeric = table(tables, "수치 특징 기술 통계")
        self.assertEqual(len(numeric["rows"]), evidence.TOP)
        self.assertTrue(any("표시 20개 / 전체 25개 / 생략 5개" in note for note in numeric["notes"]))
        self.assertTrue(all(len(row["특징"]) <= evidence.NAME_LIMIT for row in numeric["rows"]))
        correlations = table(tables, "수치 특징 상관 (쌍별)")
        self.assertTrue(any("표시 20개 / 전체 300개 / 생략 280개" in note for note in correlations["notes"]))
        self.assertTrue(all(row["특징 A"] != row["특징 B"] for row in correlations["rows"]))
        shown = [abs(float(row["Pearson"])) for row in correlations["rows"]]
        self.assertEqual(shown, sorted(shown, reverse=True))
        columns = table(tables, "컬럼별 결측·고유값·비유한 값")
        self.assertTrue(any("표시 20개 / 전체 25개 / 생략 5개" in note for note in columns["notes"]))
        self.assertTrue(columns["rows"][0]["컬럼"].startswith("feature_00_"))
        self.assertTrue(any(note.startswith("생략한 컬럼의 진단 행은") for note in columns["notes"]))


class DecisionFactsTest(unittest.TestCase):
    def setUp(self):
        self.sources = scope.sources()
        self.record = scope.record()
        self.plan = self.record["confirmed"]["plan"]
        self.expected = tabular_task._expected_impact(self.sources["candidate"], self.sources["preprocessor"])

    def tables(self, *, origin=None, candidate=None, split_rules=None, application="keep"):
        def default(record, path):
            if path == "preparation.outlier":
                return "모델 권장", "모델 문장: 극단값이 많음"
            return ("질문 답변", None) if path == "split.stratify" else ("기본 계획", None)

        return evidence.decision_tables(
            self.record, self.plan, origin or default, "사용 못 함: 기본 계획에서 시작(테스트)",
            self.sources["candidate"] if candidate is None else candidate,
            self.sources["application"] if application == "keep" else application,
            self.expected, asdict(SplitRules()) if split_rules is None else split_rules)

    def test_model_opinion_execution_check_and_changes_are_separate(self):
        summary, settings = self.tables()
        self.assertEqual(summary["rows"][0], {"항목": "모델 권장안", "내용": "사용 못 함: 기본 계획에서 시작(테스트)"})
        self.assertIn("모델 권장 1", summary["rows"][1]["내용"])
        self.assertIn("질문 답변 1", summary["rows"][1]["내용"])
        rows = {row["설정 이름"]: row for row in settings["rows"]}
        self.assertEqual(rows["outlier"]["결정 출처"], "모델 권장")
        # 모델 문장은 계산 사실 표에 두지 않고 model_opinions의 별도 절에 있다.
        self.assertNotIn("모델 의견", rows["outlier"])
        self.assertNotIn("모델 문장: 극단값이 많음", str(settings))
        self.assertEqual(rows["outlier"]["실제 변경(근거)"], "Train 값 0개 clip(후보 worker 집계) · 적용 Train 0·Validation 0개(적용 worker 집계)")
        self.assertEqual(rows["numeric_imputation"]["실제 변경(근거)"], "Train 대치 1칸(후보 worker 집계)")
        self.assertEqual(rows["drop_constant"]["실제 변경(근거)"], "Train 기준 제거 1개(후보)")
        self.assertIn("Validation 미등록 범주 발생 0건(특징별 발생 합계", rows["unknown_category"]["실제 변경(근거)"])
        # 후보 계산 설정(binding.rules)에 numeric_imputation만 있다. 나머지 항목은 실행 기록이 없다고 적는다.
        self.assertEqual(rows["numeric_imputation"]["실행 기록 대조"], "실행 값과 같음")
        self.assertEqual(rows["outlier"]["실행 기록 대조"], "실행 기록 없음")
        # 분할 확정 계획(group·0.5/0.25/0.25)과 실행 규칙 기본값의 차이는 숨기지 않는다.
        self.assertEqual(rows["strategy"]["실행 기록 대조"], "다름(실행 값: random)")
        self.assertEqual(rows["strategy"]["실제 변경(근거)"], "분할 결과 표 참조")
        self.assertEqual(rows["random_state"]["실행 기록 대조"], "다름(실행 값: 42)" if SplitRules().random_state == 42
                         else "다름(실행 값: " + str(SplitRules().random_state) + ")")

    def test_old_or_missing_records_are_not_shown_as_zero(self):
        old = dict(self.sources["candidate"], impact=None)
        self.expected = tabular_task._expected_impact(old, self.sources["preprocessor"])
        rows = {row["설정 이름"]: row for row in self.tables(candidate=old, application=None)[1]["rows"]}
        self.assertEqual(rows["numeric_imputation"]["실제 변경(근거)"], tabular_task.OLD_CANDIDATE)
        self.assertEqual(rows["unknown_category"]["실제 변경(근거)"], evidence.MISSING)
        self.expected = tabular_task._expected_impact(self.sources["candidate"], self.sources["preprocessor"])
        rows = {row["설정 이름"]: row for row in self.tables(split_rules={})[1]["rows"]}
        self.assertEqual(rows["strategy"]["실행 기록 대조"], evidence.MISSING)


class RenderTest(unittest.TestCase):
    def test_new_sections_follow_order_and_legacy_facts_are_unchanged(self):
        base = dict(scope.facts(), goal="목표", technical={})
        legacy = task_report.render_markdown(base)
        self.assertNotIn("## 처리 전 데이터와 기본 EDA", legacy)
        self.assertNotIn("## 계획과 결정", legacy)
        with tempfile.TemporaryDirectory() as folder:
            payloads, diagnostic, plan = build(mixed_frame(), MIXED_ROLES, Path(folder) / "eda")
        tables = evidence.eda_tables(payloads, diagnostic, plan)
        tables[1]["rows"][0]["컬럼"] = "x | <b>*y*</b>"
        decision = evidence.decision_tables(scope.record(), scope.record()["confirmed"]["plan"], lambda record, path: ("기본 계획", None),
                                            "사용함", None, None, {"missing": tabular_task.MISSING}, {})
        text = task_report.render_markdown(dict(base, eda_tables=tables, eda_note=evidence.EDA_SCOPE_NOTE, decision_tables=decision))
        positions = [text.index(heading) for heading in ("### 주요 문제", "## 처리 전 데이터와 기본 EDA", "## 계획과 결정", "## 분할 결과",
                                                          "## 컬럼별 처리와 제외 사유")]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Test 구간은 이 보고서에서 새로 읽거나 집계하지 않았습니다", text)
        self.assertIn("### 수치 특징 상관 (쌍별)", text)
        # 입력 유래 문자열은 표 칸·태그로 해석되지 않는다.
        self.assertNotIn("<b>", text)
        self.assertIn("x / &lt;b&gt;\\*y\\*&lt;/b&gt;", text)
        # 새 절을 뺀 나머지는 이전 facts의 출력과 같다.
        start, end = text.index("## 처리 전 데이터와 기본 EDA"), text.index("## 분할 결과")
        self.assertEqual(text[:start] + text[end:], legacy)


class ChartFactsTest(unittest.TestCase):
    """보고서 그래프 2종: 분할 결과 표·target 표와 같은 검증된 값만 쓰고, 근거가 없으면 그리지 않는다."""

    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        root = Path(cls.scratch.name)
        cls.mixed = build(mixed_frame(), MIXED_ROLES, root / "mixed")
        categorical = pd.DataFrame({"c": ["a", "b", "a", "b", "a", "c"], "d": ["p", "p", "q", "q", "p", "p"]})
        cls.no_target = build(categorical, ColumnRoles(categorical=("c",)), root / "no_target")
        one = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "label": ["양품"] * 6})
        cls.one_class = build(one, ColumnRoles(numeric=("x",), target="label"), root / "one")
        # 앞 30자가 같은 긴 한글 클래스 25개(건수 25..1). 줄인 이름이 같아도 순위로 구분해야 한다.
        names = ["불량_유형_가나다라마바사아자차카타파하_공통_접두어_" + f"{index:02d}_" + "끝" * 60 for index in range(25)]
        labels = [name for index, name in enumerate(names) for _ in range(25 - index)]
        many = pd.DataFrame({"x": np.arange(len(labels), dtype=float), "label": labels})
        cls.many = build(many, ColumnRoles(numeric=("x",), target="label"), root / "many")
        cls.names = names

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_split_chart_uses_the_same_values_as_the_split_table(self):
        record, sources = scope.record(), scope.sources()
        chart = evidence.split_chart(record["confirmed"]["plan"]["split"], sources["proposal"])
        table_rows = scope.facts()["split_table"]["rows"]
        self.assertEqual([row["구간"] for row in chart["rows"]], ["train", "validation", "test"])
        for row, shown in zip(chart["rows"], table_rows):
            self.assertEqual((row["구간"], row["요청 비율"], f"{row['확정 비율']:.3f}", row["행 수"]),
                             (shown["구간"], shown["요청 비율"], shown["확정 비율"], shown["행 수"]))
        # 요청 0.5/0.25/0.25와 확정 0.5/0.167/0.333이 다르다. 비율과 행 수를 따로 두고 반올림·차이 이유를 적는다.
        self.assertEqual([row["확정 비율"] for row in chart["rows"]], [0.5, 0.167, 0.333])
        self.assertEqual(chart["values"], ["요청 비율", "확정 비율"])
        notes = " ".join(chart["notes"])
        self.assertIn("분모 12행", notes)
        self.assertIn("반올림", notes)
        self.assertIn("validation (0.25 → 0.167)", notes)
        self.assertIn("전처리 전", notes)

    @staticmethod
    def split_case(requested, counts):
        total = sum(counts)
        proposal = {"summary": [{"split": name, "rows": count, "ratio": count / total}
                                for name, count in zip(evidence.SPLITS, counts)]}
        return evidence.split_chart({name + "_ratio": value for name, value in zip(evidence.SPLITS, requested)}, proposal)

    def test_split_gap_is_chosen_at_display_precision(self):
        # 7,043행 실제 배정 4930/1056/1057: 셋째 자리까지는 요청과 차이가 드러나지 않는다(실제 비율은 같지 않다).
        chart = self.split_case((0.7, 0.15, 0.15), (4930, 1056, 1057))
        notes = " ".join(chart["notes"])
        self.assertEqual([(row["확정 비율"], row["행 수"]) for row in chart["rows"]], [(0.7, 4930), (0.15, 1056), (0.15, 1057)])
        self.assertNotIn("차이가 가장 큰 구간", notes)
        self.assertIn("행 수를 정수로 배정해 요청 비율과 소폭 차이가 있습니다", notes)
        self.assertNotRegex(notes, "요청[^.]*(같습니다|일치합니다)")
        # 표시상 차이가 있으면 반올림 전 실제 차이로 가장 큰 구간을 고른다. 마지막 사례는 셋째 자리에서는 validation과
        # test가 같아 보이지만(0.033) 실제 차이는 test(0.03349)가 validation(0.0334)보다 크다.
        for requested, counts, expected in (((0.7, 0.15, 0.15), (8, 2, 2), "train (0.7 → 0.667)"),
                                            ((0.6, 0.2, 0.2), (7, 3, 2), "validation (0.2 → 0.250)"),
                                            ((0.4, 0.3, 0.3), (40009, 33340, 26651), "test (0.3 → 0.267)")):
            with self.subTest(counts=counts):
                notes = " ".join(self.split_case(requested, counts)["notes"])
                self.assertIn("요청과 확정의 차이가 가장 큰 구간: " + expected, notes)
                self.assertNotIn("소폭 차이", notes)
        # 실제 비율이 정확히 같으면 차이 안내를 두지 않는다.
        notes = " ".join(self.split_case((0.5, 0.25, 0.25), (6, 3, 3))["notes"])
        self.assertNotIn("차이가 가장 큰 구간", notes)
        self.assertNotIn("소폭 차이", notes)

    def test_split_chart_ratio_format_matches_the_split_table(self):
        chart = self.split_case((0.7, 0.15, 0.15), (4930, 1056, 1057))
        self.assertEqual(task_report.chart_rows(chart),
                         [{"구간": "train", "요청 비율": 0.7, "확정 비율": "0.700", "행 수": 4930},
                          {"구간": "validation", "요청 비율": 0.15, "확정 비율": "0.150", "행 수": 1056},
                          {"구간": "test", "요청 비율": 0.15, "확정 비율": "0.150", "행 수": 1057}])
        section = task_report.render_markdown(dict(scope.facts(), goal="목표", technical={}, images=[], charts=[chart]))
        self.assertIn("| train | 0.7 | 0.700 | 4930 | " + "█" * 20 + " | " + "█" * 20 + " |", section)
        # 자릿수를 정하지 않은 그래프(target)는 값 그대로다.
        target = evidence.target_chart(self.mixed[0])
        self.assertEqual(task_report.chart_rows(target), target["rows"])

    def test_split_chart_is_not_drawn_without_evidence(self):
        split = scope.record()["confirmed"]["plan"]["split"]
        proposal = scope.sources()["proposal"]
        partial = {"summary": [item for item in proposal["summary"] if item["split"] != "test"]}
        broken = {"summary": [dict(item, ratio=None) if item["split"] == "train" else item for item in proposal["summary"]]}
        for case, missing in ((None, "train, validation, test"), (partial, "test"), (broken, "train")):
            chart = evidence.split_chart(split, case)
            self.assertEqual(chart["rows"], [])
            self.assertEqual(chart["notes"], ["근거 없음: " + missing + " 구간의 요청 비율·확정 비율·행 수 중 일부가 검증된 결과에 없어 "
                                              "그래프를 그리지 않았습니다(추정하지 않음)."])
        self.assertEqual(evidence.split_chart({}, proposal)["rows"], [])

    def test_target_chart_matches_the_target_table(self):
        payloads, diagnostic, plan = self.mixed
        chart = evidence.target_chart(payloads)
        shown = table(evidence.eda_tables(payloads, diagnostic, plan), "target 클래스 분포")["rows"]
        self.assertEqual([(row["클래스"], row["건수"], row["비율"]) for row in chart["rows"]],
                         [(f"{row['순위']:02d}. " + row["클래스"], row["건수"], row["비율"]) for row in shown])
        self.assertEqual([row["클래스"] for row in chart["rows"]], ["01. 0012", "02. 1"])
        self.assertTrue(chart["horizontal"])
        self.assertIn("대상: 전체 입력(분할 전·전처리 전) · target · 분모 12행(결측 포함). 구간별·처리 후 분포가 아닙니다.", chart["notes"])
        self.assertIn(evidence.EDA_BASIS, chart["notes"])

    def test_no_target_and_single_class(self):
        chart = evidence.target_chart(self.no_target[0])
        self.assertEqual(chart["rows"], [])
        self.assertTrue(chart["notes"][0].startswith("해당 없음: target 역할이 없어"))
        chart = evidence.target_chart(self.one_class[0])
        self.assertEqual(chart["rows"], [{"클래스": "01. 양품", "건수": 6, "비율": "100.0%"}])
        self.assertIn("클래스 1개뿐이라 클래스 간 불균형은 비교할 수 없습니다.", chart["notes"])

    def test_many_classes_show_limit_omitted_count_and_distinct_long_korean_names(self):
        payloads, diagnostic, plan = self.many
        chart = evidence.target_chart(payloads)
        labels = [row["클래스"] for row in chart["rows"]]
        self.assertEqual(len(labels), evidence.TOP)
        self.assertEqual(len(set(labels)), len(labels))
        # 줄인 이름은 서로 같지만 앞의 순위로 구분된다.
        self.assertEqual(len({label.split(". ", 1)[1] for label in labels}), 1)
        self.assertTrue(all(len(label) <= 4 + evidence.CHART_NAME_LIMIT for label in labels))
        self.assertEqual([row["건수"] for row in chart["rows"]], list(range(25, 5, -1)))
        # 생략 5개 클래스의 행 5+4+3+2+1=15개.
        self.assertIn("표시 20개 / 전체 25개 / 생략 5개 · 생략한 클래스의 행 15개", " ".join(chart["notes"]))
        self.assertIn(f"그 표는 {evidence.NAME_LIMIT}자까지 보이며, 줄이지 않은 이름은 기술 상세 분할 EDA 폴더의 target_distribution.csv",
                      " ".join(chart["notes"]))
        shown = table(evidence.eda_tables(payloads, diagnostic, plan), "target 클래스 분포")
        self.assertEqual([row["순위"] for row in shown["rows"]], list(range(1, 21)))
        self.assertIn(f"이름이 {evidence.NAME_LIMIT}자를 넘는 클래스 20개는 줄여 표시했습니다(순위로 구분).", shown["notes"])
        self.assertTrue(shown["rows"][0]["클래스"].startswith(self.names[0][:evidence.NAME_LIMIT - 1]))

    def test_markdown_has_the_chart_values_bars_and_legacy_output_is_unchanged(self):
        base = dict(scope.facts(), goal="목표", technical={}, images=[{"label": "missing.png"}])
        legacy = task_report.render_markdown(base)
        split = evidence.split_chart(scope.record()["confirmed"]["plan"]["split"], scope.sources()["proposal"])
        target = evidence.target_chart(self.mixed[0])
        target["rows"][0]["클래스"] = "01. <b>|*x*</b>"
        empty = evidence.target_chart(self.no_target[0])
        text = task_report.render_markdown(dict(base, charts=[split, target, empty]))
        section = text[text.index("## 그래프"):text.index("## 경고")]
        positions = [section.index(heading) for heading in ("### 요청·확정 분할 비율 (구간별)", "### target 클래스 분포 (전체 입력, 분할 전)",
                                                             "### 분할 EDA 그래프 (검증된 PNG, 화면에서 표시)", "- missing.png")]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("| 구간 | 요청 비율 | 확정 비율 | 행 수 | 막대 요청 비율 | 막대 확정 비율 |", section)
        # 같은 그래프의 최댓값(0.5) 기준 20칸. 0.167은 7칸, 0.333은 13칸이다.
        self.assertIn("| train | 0.5 | 0.500 | 6 | " + "█" * 20 + " | " + "█" * 20 + " |", section)
        self.assertIn("| validation | 0.25 | 0.167 | 2 | " + "█" * 10 + " | " + "█" * 7 + " |", section)
        self.assertIn("| test | 0.25 | 0.333 | 4 | " + "█" * 10 + " | " + "█" * 13 + " |", section)
        self.assertIn("01. &lt;b&gt;/\\*x\\*&lt;/b&gt;", section)
        self.assertNotIn("<b>", section)
        self.assertIn("- 해당 없음: target 역할이 없어", section)
        self.assertEqual(section.count("- 막대:"), 2)
        # 그래프 절을 뺀 나머지는 같고, 그래프가 없는 이전 facts의 출력은 바뀌지 않는다.
        start, end = text.index("## 그래프"), text.index("## 경고")
        old_start, old_end = legacy.index("## 그래프"), legacy.index("## 경고")
        self.assertEqual(text[:start] + text[end:], legacy[:old_start] + legacy[old_end:])
        self.assertTrue(legacy[old_start:old_end].startswith("## 그래프\n- missing.png — "))
        self.assertNotIn("###", legacy[old_start:old_end])

    def test_screen_draws_the_same_rows_in_fact_order(self):
        from src.ui import task_view

        class Screen:
            def __init__(self):
                self.calls = []

            def __getattr__(self, name):
                return lambda *args, **kwargs: self.calls.append((name, args, kwargs))

        split = evidence.split_chart(scope.record()["confirmed"]["plan"]["split"], scope.sources()["proposal"])
        target = evidence.target_chart(self.many[0])
        for chart, offset in ((split, "xOffset"), (target, None)):
            screen = Screen()
            with patch.object(task_view, "_rows") as rows:
                task_view._chart(screen, chart)
            # 화면 표도 Markdown 표와 같은 표시 형식(분할 결과 표의 확정 비율 셋째 자리)이다.
            rows.assert_called_once_with(screen, task_report.chart_rows(chart))
            spec = next(args[0] for name, args, _ in screen.calls if name == "altair_chart").to_dict()
            data = next(iter(spec["datasets"].values()))
            category = chart["category"]
            self.assertEqual(data, [{category: row[category], "계열": key, "값": row[key]}
                                    for row in chart["rows"] for key in chart["values"]])
            axis = spec["encoding"]["y" if chart["horizontal"] else "x"]
            self.assertEqual(axis["sort"], [row[category] for row in chart["rows"]])
            self.assertEqual(axis["title"], category)
            self.assertEqual(spec["encoding"]["x" if chart["horizontal"] else "y"]["title"], chart["unit"])
            self.assertEqual(offset in spec["encoding"], offset is not None)
            # 높이는 막대 영역 높이다. 전체 높이에 맞추면(fit) 축·범례가 자리를 차지해 막대가 0px로 눌린다.
            self.assertEqual(spec["autosize"], {"type": "fit-x", "contains": "padding"})
            bands = len(chart["rows"]) * (len(chart["values"]) if chart["horizontal"] else 1)
            self.assertGreaterEqual(spec["height"], 24 * bands if chart["horizontal"] else 200)
            # 긴 한글 라벨과 겹치지 않도록 가로 막대의 이름 축 제목은 축 위에 가로로 둔다.
            self.assertEqual(axis["axis"].get("titleAngle"), 0 if chart["horizontal"] else None)
            self.assertIn(("text", ("■ " + chart["title"],), {}), screen.calls)
        screen = Screen()
        task_view._chart(screen, evidence.target_chart(self.no_target[0]))
        self.assertEqual([name for name, _, _ in screen.calls], ["text", "text"])


def sealed(directory, payloads, context, *, files=None):
    """테스트용 커밋 폴더: eda/ 파일과 완료 표식(크기·해시·실행 문맥)."""
    (directory / "eda").mkdir(parents=True)
    for name, data in payloads.items():
        (directory / "eda" / name).write_bytes(data)
    listed = files if files is not None else {
        "eda/" + name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()}
    marker = {"version": 1, "context": asdict(context), "files": dict(listed, **{"worker_receipt.json": {"size": 2, "sha256": "0" * 64}})}
    (directory / "AGENT_COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")


class CommittedEdaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.payloads = build(mixed_frame(), MIXED_ROLES, Path(cls.scratch.name) / "source")[0]

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.output = Path(self.folder.name) / "run_eda"
        self.context = ArtifactContext("tabular_source:" + "1" * 64, "2" * 64, "3" * 64, "split_policy", "run_eda")
        self.profile = SimpleNamespace(roles=MIXED_ROLES)

    def tearDown(self):
        self.folder.cleanup()

    def read(self, rows=12):
        return service._committed_eda(self.output, self.context, self.profile, rows)

    def test_valid_result_returns_the_validated_bytes(self):
        sealed(self.output, self.payloads, self.context)
        self.assertEqual(self.read(), self.payloads)

    def test_tampered_missing_extra_and_mixed_results_are_rejected(self):
        listed = {"eda/" + name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in self.payloads.items()}
        cases = {
            "보고 근거 파일이 완료 기록과 다릅니다": ("tamper", listed),
            "파일 구성이 고정 보고서와 다릅니다": ("missing", {key: value for key, value in listed.items() if key != "eda/pearson.csv"}),
            "파일 구성이 고정 보고서와 다릅니다 ": ("extra", dict(listed, **{"eda/extra.csv": listed["eda/pearson.csv"]})),
            "이 분할 실행의 결과가 아닙니다": ("context", listed),
            "크기가 상한을 넘습니다": ("size", dict(listed, **{"eda/pearson.csv": {"size": 8 * 1024 * 1024 + 1, "sha256": "0" * 64}})),
        }
        for message, (kind, files) in cases.items():
            with self.subTest(kind=kind):
                self.tearDown()
                self.setUp()
                sealed(self.output, self.payloads, self.context, files=files)
                if kind == "tamper":
                    path = self.output / "eda" / "missing.csv"
                    data = path.read_bytes()
                    path.write_bytes(data[:-1] + (b"9" if data[-1:] != b"9" else b"8"))
                if kind == "context":
                    marker = json.loads((self.output / "AGENT_COMPLETE.json").read_text(encoding="utf-8"))
                    marker["context"]["plan_id"] = "4" * 64
                    (self.output / "AGENT_COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message.strip()):
                    self.read()

    def test_total_size_limit_row_count_and_contract_violation_are_rejected(self):
        sealed(self.output, self.payloads, self.context)
        with patch.object(eda_result_validation, "MAX_TOTAL_BYTES", 1024):
            with self.assertRaisesRegex(ValueError, "크기가 상한을 넘습니다"):
                self.read()
        with self.assertRaisesRegex(ValueError, "행 수가 승인된 입력과 다릅니다"):
            self.read(rows=13)
        # 표식까지 다시 쓴 계약 위반(헤더 변경)은 기존 EDA 결과 검증기가 거부한다.
        self.tearDown()
        self.setUp()
        forged = dict(self.payloads, **{"missing.csv": self.payloads["missing.csv"].replace(b"missing_count", b"missing_total")})
        sealed(self.output, forged, self.context)
        with self.assertRaisesRegex(ValueError, "결과 계약 검사를 통과하지 못했습니다"):
            self.read()

    def test_marker_format_and_scratch_failure_are_explained_separately(self):
        sealed(self.output, self.payloads, self.context)
        path = self.output / "AGENT_COMPLETE.json"
        marker = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(dict(marker, version=2)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "완료 기록 형식이 올바르지 않습니다"):
            self.read()
        path.write_text(json.dumps(marker), encoding="utf-8")
        with patch("tempfile.TemporaryDirectory", side_effect=PermissionError("locked")):
            with self.assertRaisesRegex(ValueError, "임시 사본을 만들거나 정리하지 못했습니다"):
                self.read()

    def test_copy_change_during_validation_and_source_change_after_read(self):
        sealed(self.output, self.payloads, self.context)
        real = eda_result_validation.validate_eda_report

        def changed_copy(directory, profile):
            result = real(directory, profile)
            (Path(directory) / "pearson.csv").write_bytes(b"changed")
            return result

        with patch.object(eda_result_validation, "validate_eda_report", changed_copy):
            with self.assertRaisesRegex(ValueError, "결과 계약 검사를 통과하지 못했습니다"):
                self.read()

        replaced = []

        def replace_source(directory, profile):
            # 검증 중 원본 폴더 파일을 바꿔도 해석에는 검증한 사본과 같은 바이트만 쓴다(원본 재열기 없음).
            target = self.output / "eda" / "target_distribution.csv"
            target.write_bytes(target.read_bytes().replace(b"0012", b"9999"))
            replaced.append(directory)
            return real(directory, profile)

        with patch.object(eda_result_validation, "validate_eda_report", replace_source):
            result = self.read()
        self.assertEqual(len(replaced), 1)
        self.assertEqual(result, self.payloads)
        self.assertIn(b"0012", result["target_distribution.csv"])


if __name__ == "__main__":
    unittest.main()

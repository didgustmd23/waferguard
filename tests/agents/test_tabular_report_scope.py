"""보고서 비교 범위·출처·표기만 검사한다. 합성 사실 묶음을 쓰며 실제 데이터·모델·worker는 사용하지 않는다."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents import tabular_task, task_report
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules, TabularPreprocessor, impact_totals


ROLES = {"x": "수치형", "c": "범주형", "d": "수치형", "label": "target", "id": "식별자"}


def column(name, missing, unique):
    return {"column": name, "missing_count": missing, "missing_ratio": missing / 12, "observed_unique": unique}


def record(**preparation):
    split = dict(asdict(SplitRules()), strategy="group", train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25, random_state=0)
    return {
        "confirmed": {"plan": {"roles": dict(ROLES), "split": split,
                               "preparation": dict(asdict(PreparationRules()), **preparation)}, "answers": []},
        # 전체 입력 기준 결측 4칸: x 3칸(이 중 Train 1칸, 나머지는 Test에 있다고 가정), c 1칸(Test).
        "diagnostic": {"summary": {"row_count": 12, "column_count": 5},
                       "columns": [column("x", 3, 8), column("c", 1, 2), column("d", 0, 1), column("label", 0, 2), column("id", 0, 12)]},
        "source": {},
    }


def sources(*, impute=5.0, fill="missing", categories=("missing", '["str", "a"]'), x_missing=1, application=True, plot_limit=12):
    width = len(categories)
    candidate = {
        "binding": {"train_rows": 6, "rules": {"numeric_imputation": "median"}},
        "removed": [{"column": "d", "reason": "CONSTANT_OR_ALL_MISSING"}], "retained": ["x", "c"],
        "output_feature_count": 1 + width, "preprocessor_sha256": "0" * 64, "schema_version": "preparation_candidate_report_v3",
        "impact": [
            {"column": "x", "kind": "numeric", "width": 1, "missing": x_missing, "imputed": x_missing if impute is not None else 0,
             "clipped": None, "distinct_before": 5, "constant_after": False, "unknown": 0,
             "missing_after": 0 if impute is not None else x_missing},
            {"column": "c", "kind": "categorical", "width": width, "missing": 0, "imputed": 0, "clipped": None,
             "distinct_before": 2, "constant_after": False, "unknown": 0, "missing_after": 0},
        ],
    }
    preprocessor = {"state": {"records": [
        {"column": "x", "kind": "numeric", "impute": impute, "clip": None, "center": 0.0, "scale": 1.0, "edges": None, "width": 1},
        {"column": "c", "kind": "categorical", "fill": fill, "categories": list(categories), "width": width}]}}
    totals = {"clipped_values": 0, "clipped_features": 0, "constant_features": 0, "unknown_categories": 0,
              "missing_values": 0 if impute is not None else x_missing}
    applied = {"schema_version": "preparation_application_v3", "binding": {"output_feature_count": 1 + width},
               "partitions": {"train": {"rows": 6}, "validation": {"rows": 2}},
               "impact": {"train": dict(totals), "validation": dict(totals, missing_values=0)}, "test_used": False}
    proposal = {"summary": [
        {"split": "train", "rows": 6, "ratio": 0.5, "classes": [{"label": "True", "count": 3}, {"label": "False", "count": 3}], "group_count": 3},
        {"split": "validation", "rows": 2, "ratio": 2 / 12, "classes": [{"label": "True", "count": 1}, {"label": "False", "count": 1}], "group_count": 1},
        {"split": "test", "rows": 4, "ratio": 4 / 12, "classes": [{"label": "True", "count": 2}, {"label": "False", "count": 2}], "group_count": 2}]}
    return {"quality": {"status": "passed", "findings": []}, "proposal": proposal, "candidate": candidate,
            "preprocessor": preprocessor, "application": applied if application else None, "files": {}, "bundle": "bundle",
            "reports": [], "image_labels": ["missing.png", "distributions.png", "correlations.png"], "renamed": [],
            "notices": [], "input": {}, "plot_limit": plot_limit}


def facts(record_value=None, source_value=None):
    with patch.object(tabular_task, "_origin", return_value=("기본 계획", None)), \
            patch.object(tabular_task.rules, "identifier_like", return_value=[]):
        return tabular_task.report_from_sources(record_value or record(), source_value or sources())


def rows(result, item):
    return [row for row in result["comparison"] if row["항목"] == item]


class ComparisonScopeTests(unittest.TestCase):
    def test_missing_scopes_are_separate_and_never_subtracted(self):
        result = facts()
        diagnosed = rows(result, "결측 칸 (진단)")[0]
        self.assertEqual(diagnosed["대상 구간"], "전체 입력(분할 전)")
        self.assertEqual(diagnosed["처리 전"], 4)
        # 전체 입력 범위는 변환하지 않았으므로 처리 후 값을 두지 않는다(Test에 남은 결측을 '0'으로 보이게 하지 않음).
        self.assertNotIn("0", str(diagnosed["처리 후"]))
        self.assertIn("비교 대상 없음", diagnosed["처리 후"])

        train_input = rows(result, "입력 결측")
        self.assertEqual([(row["대상 구간"], row["처리 전"]) for row in train_input][0], ("train", 1))
        self.assertIn("worker 집계", train_input[0]["근거"])
        self.assertIn("원본 건수를 다시 센 것이 아님", train_input[0]["근거"])
        validation = [row for row in train_input if row["대상 구간"] == "validation"][0]
        self.assertTrue(str(validation["처리 전"]).startswith("미집계("))
        self.assertEqual(rows(result, "대치 건수")[0]["처리 후"], 1)

        output = {row["대상 구간"]: row for row in rows(result, "출력 결측")}
        self.assertEqual(set(output), {"train", "validation"})
        self.assertEqual(output["train"]["처리 전"], "-")
        self.assertIn("출력 열 3개", output["train"]["컬럼 범위"])
        self.assertIn("출력 파일로 대조", output["train"]["근거"])
        for row in result["comparison"]:
            self.assertEqual(set(row), {"항목", "대상 구간", "컬럼 범위", "단위", "처리 전", "처리 후", "근거"})
            self.assertNotIn("모두 처리", str(row))

    def test_test_partition_is_not_transformed_or_counted(self):
        result = facts()
        test = [row for row in rows(result, "구간 행 수") if row["대상 구간"] == "test"][0]
        self.assertEqual(test["처리 전"], "4행 (분할 확정)")
        self.assertEqual(test["처리 후"], "변환하지 않음")
        self.assertIn("새로 집계하지 않", test["근거"])
        self.assertIn("Test", result["comparison_note"])
        self.assertIn("차감하지 않", result["comparison_note"])
        # 근거 없이 Test에 결측이 남았다고 단정하지 않는다.
        self.assertFalse(any("Test" in str(row) and "결측이 남" in str(row) for row in result["comparison"]))

    def test_split_rows_are_not_candidate_estimates(self):
        result = facts()
        for row in result["comparison"]:
            self.assertNotIn("예상 (후보)", row)
        split = {row["구간"]: row for row in result["split_table"]["rows"]}
        self.assertEqual(split["validation"]["요청 비율"], 0.25)
        self.assertEqual(split["validation"]["확정 비율"], "0.167")
        self.assertEqual(split["test"]["행 수"], 4)
        self.assertEqual(split["train"]["그룹 수"], 3)
        self.assertIn("True: 3", split["train"]["클래스"])
        self.assertTrue(any("요청" in line and "확정" in line for line in result["lines"]))

    def test_candidate_estimate_and_applied_measure_keep_sources(self):
        result = facts()
        train = [row for row in rows(result, "출력 결측") if row["대상 구간"] == "train"][0]
        self.assertIn("후보 예상 0", train["근거"])
        clip = {row["대상 구간"]: row for row in rows(result, "clipping으로 바뀐 값")}
        self.assertIn("worker 집계", clip["validation"]["근거"])
        old = sources(application=False)
        result = facts(source_value=old)
        self.assertEqual(rows(result, "출력 결측")[0]["처리 후"], tabular_task.MISSING)

    def test_unknown_categories_are_feature_occurrences_not_rows(self):
        # Validation 1행의 범주형 특징 2개가 모두 Train에 없던 값이면 기존 집계는 특징별 발생 합계 2건이다.
        train = pd.DataFrame({"a": ["p", "q", "p", "q"], "b": ["r", "s", "r", "s"]})
        validation = pd.DataFrame({"a": ["new"], "b": ["other"]})
        preprocessor = TabularPreprocessor(ColumnRoles(categorical=("a", "b")), PreparationRules(unknown_category="ignore")).fit(train)
        unknown = impact_totals(preprocessor.impact(validation))["unknown_categories"]
        self.assertEqual(unknown, 2)

        value = sources()
        value["application"]["impact"]["validation"]["unknown_categories"] = unknown
        result = dict(facts(source_value=value), goal="목표", technical={})
        row = {item["대상 구간"]: item for item in rows(result, "미등록 범주 발생")}["validation"]
        self.assertEqual(row["처리 후"], 2)
        self.assertEqual(row["단위"], "건(특징별 발생 합계)")
        self.assertNotIn("행", row["단위"])
        # 화면과 Markdown은 같은 비교 표 사실을 쓴다.
        markdown = task_report.render_markdown(result)
        line = next(text for text in markdown.splitlines() if "미등록 범주 발생" in text and "validation" in text)
        self.assertIn("건(특징별 발생 합계)", line)
        self.assertNotIn("2행", line)

    def test_one_to_one_output_without_imputation_keeps_missing(self):
        result = facts(record(numeric_imputation="none"), sources(impute=None, x_missing=2))
        self.assertEqual(rows(result, "대치 건수")[0]["처리 후"], 0)
        self.assertEqual([row for row in rows(result, "출력 결측") if row["대상 구간"] == "train"][0]["처리 후"], 2)


class ProcessingWordingTests(unittest.TestCase):
    def processing(self, **options):
        return {row["컬럼"]: row["처리"] for row in facts(source_value=sources(**options))["processing"]}

    def test_numeric_value_method_and_count_are_distinct(self):
        action = self.processing()["x"]
        self.assertIn("대치값 5 (Train 중앙값)", action)
        self.assertIn("대치 건수 Train 1칸", action)
        self.assertNotIn("결측 대치 5", action)

    def test_numeric_imputation_without_missing_is_described_as_no_target(self):
        self.assertIn("대치 대상 없음", self.processing(x_missing=0)["x"])

    def test_categorical_fill_is_not_numeric_median(self):
        action = self.processing(fill='["str", "a"]', categories=('["str", "a"]', '["str", "b"]'))["c"]
        self.assertIn("Train 최빈값 'a'로 대치", action)
        self.assertNotIn("중앙값", action)
        reserved = self.processing()["c"]
        self.assertIn("별도 결측 범주", reserved)
        self.assertNotIn("중앙값", reserved)


class CaptionTests(unittest.TestCase):
    def test_captions_use_confirmed_limit_and_counts(self):
        notes = facts(source_value=sources(plot_limit=1))["image_notes"]
        self.assertIn("결측률 내림차순", notes["missing.png"])
        self.assertIn("상위 1개", notes["missing.png"])
        # EDA는 확정 계획의 역할로 계산하므로 후보가 나중에 제거한 d도 특징 수에 들어간다.
        self.assertIn("특징 3개", notes["missing.png"])
        self.assertIn("입력 순서 앞쪽 1개", notes["distributions.png"])
        self.assertIn("분할·대치·스케일 전", notes["distributions.png"])

    def test_unknown_limit_is_not_guessed(self):
        notes = facts(source_value=sources(plot_limit=None))["image_notes"]
        self.assertNotIn("12", notes["missing.png"])
        self.assertIn("미확인", notes["missing.png"])

    def test_no_wrong_position_hint(self):
        self.assertFalse(any("아래 그래프" in text for text in facts()["cautions"]))


class MarkdownEscapeTests(unittest.TestCase):
    def test_quotes_are_plain_and_injection_still_blocked(self):
        text = task_report._plain("결측은 'a' \"b\" <script>x</script> & # [l](u) `c` *e* _f_ !g | h")
        self.assertNotIn("&#x27;", text)
        self.assertNotIn("&\\#", text)
        self.assertIn("'a'", text)
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("&amp;", text)
        for character in "#[]`*_!":
            self.assertNotIn(" " + character, text.replace("\\" + character, ""))
        self.assertNotIn("|", text)

    def test_rendered_report_keeps_old_and_new_facts(self):
        new = dict(facts(), goal="목표", technical={})
        markdown = task_report.render_markdown(new)
        self.assertIn("## 분할 결과", markdown)
        self.assertIn("대상: 전체 입력", markdown)
        self.assertNotIn("&#x27;", markdown)
        self.assertNotIn("&\\#x27;", markdown)
        old = {key: value for key, value in new.items() if key not in ("split_table", "image_notes")}
        legacy = task_report.render_markdown(old)
        self.assertNotIn("## 분할 결과", legacy)
        self.assertIn("- missing.png", legacy)


if __name__ == "__main__":
    unittest.main()

"""보고서 표시만 검사한다. 실제 데이터·모델·worker는 사용하지 않는다."""

import unittest
from unittest.mock import patch

from src.agents import tabular_task
from src.preprocessing.transform import PreparationRules
from dataclasses import asdict


class ReportWordingTests(unittest.TestCase):
    def record(self, role, **column):
        return {
            "confirmed": {"plan": {"roles": {"flag": role}, "preparation": asdict(PreparationRules())}},
            "diagnostic": {"columns": [dict(column="flag", missing_count=0,
                observed_unique=2, **column)], "summary": {}},
            "source": {},
        }

    def action(self, fill, categories):
        state = {"state": {"records": [{"column": "flag", "kind": "categorical",
            "fill": fill, "categories": categories, "width": len(categories)}]}}
        with patch.object(tabular_task, "_origin", return_value=("기본 계획", None)):
            return tabular_task._processing(self.record("범주형"), state, None)[0]["처리"]

    def test_most_frequent_has_no_extra_missing_category(self):
        action = self.action('["str", "Male"]', ['["str", "Male"]', '["str", "Female"]'])
        self.assertIn("범주 2개 → 출력 2열", action)
        self.assertIn("결측 시 'Male'로 채움", action)
        self.assertNotIn("결측 범주 포함", action)
        self.assertNotIn('["str"', action)

    def test_reserved_missing_category_is_distinct_from_literal_string(self):
        action = self.action("missing", ["missing", '["str", "missing"]'])
        self.assertIn("별도 결측 범주 포함", action)
        self.assertIn("별도 결측 범주", action)
        self.assertEqual(tabular_task._category_display('["str", "missing"]'), "missing")

    def test_typed_values_and_invalid_token(self):
        self.assertEqual(tabular_task._category_display('["bool", false]'), "false")
        self.assertEqual(tabular_task._category_display('["number", 2.5]'), "2.5")
        self.assertEqual(tabular_task._category_display("invalid"), tabular_task.MISSING)

    def test_iqr_candidates_do_not_imply_invalid_binary_values(self):
        record = self.record("수치형", outlier_candidate_count=3)
        record["confirmed"]["plan"]["preparation"]["outlier"] = "none"
        with patch.object(tabular_task.rules, "identifier_like", return_value=[]):
            rows = tabular_task._problems(record, None)
        self.assertIn("정상 0/1 값도 포함", rows[0]["내용"])
        self.assertIn("오류 판정이 아닙니다", rows[0]["내용"])
        self.assertEqual(rows[0]["처리"], "변경하지 않음(이상치 처리 미선택)")


if __name__ == "__main__":
    unittest.main()

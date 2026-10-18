"""기능별 EDA 계산과 보고서 계층의 책임 분리를 합성 입력으로 검증한다."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.eda import (
    analyze_categories, analyze_correlations, analyze_missing, analyze_numeric,
    analyze_summary, analyze_target, collect_eda,
)
from src.preprocessing.eda_report import write_eda_report


class EdaModulesTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
            "category": ["a", "b", "a", None],
            "label": [0, 1, 0, 1],
        })
        self.data = TabularInput(
            self.frame, ColumnRoles(numeric=("x", "y"), categorical=("category",), target="label"),
        )

    def test_missing_does_not_run_other_analyses(self):
        """결측 분석만 요청하면 상관·기술 통계·범주·라벨 분석을 호출하지 않는다."""
        names = ("analyze_correlations", "analyze_numeric", "analyze_categories", "analyze_target")
        mocks = []

        # 불필요한 함수가 한 번이라도 호출되면 즉시 실패하도록 경계를 고정한다.
        for name in names:
            mock = patch(f"src.preprocessing.eda.{name}", side_effect=AssertionError("선택하지 않은 분석 호출"))
            mocks.append(mock.start())
            self.addCleanup(mock.stop)

        result = collect_eda(self.data, analyses=("missing",))

        self.assertEqual(set(result), {"summary", "missing"})
        self.assertEqual(result["missing"].loc["x", "missing_count"], 1)

        for mock in mocks:
            mock.assert_not_called()

    def test_full_collection_matches_independent_functions(self):
        """전체 묶음의 각 결과가 기능별 독립 호출과 동일하다."""
        tables = collect_eda(self.data)
        self.assertEqual(tables["summary"], analyze_summary(self.data))
        expected = {
            "missing": analyze_missing(self.data),
            "numeric_statistics": analyze_numeric(self.data),
            "categorical_distribution": analyze_categories(self.data),
            "target_distribution": analyze_target(self.data),
            **analyze_correlations(self.data),
        }

        for name, table in expected.items():
            pd.testing.assert_frame_equal(tables[name], table)

    def test_computation_preserves_source_and_does_not_write(self):
        """계산 계층은 입력 값을 바꾸거나 보고서·그림을 저장하지 않는다."""
        before = self.frame.copy(deep=True)

        with patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("계산 중 파일 저장")):
            collect_eda(self.data)

        pd.testing.assert_frame_equal(self.frame, before)

    def test_selection_is_strict_and_summary_only_is_supported(self):
        """미지원·중복·잘못된 타입은 거부하며 빈 선택은 요약만 반환한다."""
        for invalid in (("unknown",), ("missing", "missing"), "missing", (1,), {"missing"}):
            with self.subTest(selection=invalid), self.assertRaises(ValueError):
                collect_eda(self.data, analyses=invalid)

        self.assertEqual(set(collect_eda(self.data, analyses=())), {"summary"})

    def test_partial_result_cannot_claim_full_report(self):
        """부분 분석을 전체 보고서로 저장하거나 완료 표식을 남기지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"

            with self.assertRaises(ValueError):
                write_eda_report(self.data, collect_eda(self.data, analyses=("missing",)), output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

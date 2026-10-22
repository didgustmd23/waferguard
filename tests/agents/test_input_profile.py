"""역할 지정 전 입력 진단 계산을 고정 합성 CSV와 손계산 상수 정답으로 검사한다."""

from io import BytesIO
import math
import unittest

import pandas as pd

from src.preprocessing.builtin_modules import build_builtin_registry
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.input_profile import PROFILE_COLUMNS, build_diagnostic_registry, profile_input


# 8행 10열. 정답은 아래 주석의 손계산 값이며 테스트 안에서 pandas로 다시 계산하지 않는다.
CSV_BYTES = (
    b"int_col,float_col,pos_inf,neg_inf,both_inf,all_inf,text_col,bool_col,mixed,edge\n"
    b"1,1.5,inf,-inf,inf,inf,CANARY_ALPHA,True,1,1\n"
    b"2,2.5,1,-inf,-inf,-inf,CANARY_BETA,False,2,2\n"
    b"3,,2,10,3,inf,  ,True,CANARY_MIX,3\n"
    b"4,3.5,3,10,,inf,,True,4,4\n"
    b"5,4.5,4,10,,-inf,CANARY_ALPHA,False,5,5\n"
    b"6,,5,10,,inf,CANARY_GAMMA,True,6,6\n"
    b"7,5.5,6,,,inf, ,True,7,7\n"
    b"50,6.5,7,20,,inf,CANARY_BETA,False,8,11.5\n"
)


def _frame():
    return pd.read_csv(BytesIO(CSV_BYTES), encoding="utf-8-sig")


class InputProfileTest(unittest.TestCase):
    def setUp(self):
        result = profile_input(_frame())
        self.summary = result["summary"]
        self.table = result["columns"].set_index("column")

    def row(self, name):
        return {key: (None if pd.isna(value) else value) for key, value in self.table.loc[name].items()}

    def assert_stats(self, name, expected):
        row = self.row(name)

        for key, value in expected.items():
            if isinstance(value, float) and not value.is_integer():
                self.assertAlmostEqual(row[key], value, places=12, msg=f"{name}.{key}")
            else:
                self.assertEqual(row[key], value, msg=f"{name}.{key}")

    def test_summary_and_columns(self):
        self.assertEqual(self.summary, {"version": "input_profile_v1", "row_count": 8, "column_count": 10, "repeated_rows": 0})
        self.assertEqual(tuple(self.table.reset_index().columns), PROFILE_COLUMNS)
        self.assertEqual(list(self.table.index), ["int_col", "float_col", "pos_inf", "neg_inf", "both_inf", "all_inf", "text_col", "bool_col", "mixed", "edge"])
        self.assertEqual(
            self.table["dtype_group"].to_dict(),
            {"int_col": "numeric", "float_col": "numeric", "pos_inf": "numeric", "neg_inf": "numeric", "both_inf": "numeric",
             "all_inf": "numeric", "text_col": "text", "bool_col": "boolean", "mixed": "text", "edge": "numeric"},
        )

    def test_integer_column_statistics_and_iqr(self):
        # 정렬 1..7,50: q25=2.75, q50=4.5, q75=6.25, IQR=3.5, 경계 -2.5·11.5, 표본분산 1879.5/7
        self.assert_stats("int_col", {
            "missing_count": 0, "missing_ratio": 0.0, "observed_unique": 8, "blank_string_count": None,
            "pos_inf_count": 0, "neg_inf_count": 0, "finite_count": 8, "mean": 9.75, "std": math.sqrt(268.5),
            "min": 1.0, "q25": 2.75, "median": 4.5, "q75": 6.25, "max": 50.0,
            "iqr_lower": -2.5, "iqr_upper": 11.5, "outlier_candidate_count": 1,
        })

    def test_missing_float_column(self):
        # 결측 2개, 유한값 1.5..6.5: q25=2.75, q75=5.25, 경계 -1.0·9.0, 표본분산 17.5/5
        self.assert_stats("float_col", {
            "missing_count": 2, "missing_ratio": 0.25, "observed_unique": 6, "finite_count": 6,
            "mean": 4.0, "std": math.sqrt(3.5), "min": 1.5, "q25": 2.75, "median": 4.0, "q75": 5.25, "max": 6.5,
            "iqr_lower": -1.0, "iqr_upper": 9.0, "outlier_candidate_count": 0,
        })

    def test_infinity_counts_are_separate_from_finite_statistics(self):
        # +inf 1개: 유한값 1..7만 통계에 사용, 고유값은 inf 포함 8
        self.assert_stats("pos_inf", {
            "missing_count": 0, "observed_unique": 8, "pos_inf_count": 1, "neg_inf_count": 0, "finite_count": 7,
            "mean": 4.0, "std": math.sqrt(28 / 6), "min": 1.0, "q25": 2.5, "median": 4.0, "q75": 5.5, "max": 7.0,
            "iqr_lower": -2.0, "iqr_upper": 10.0, "outlier_candidate_count": 0,
        })
        # -inf 2개·결측 1개: 유한값 10,10,10,10,20, IQR 0이라 20만 후보
        self.assert_stats("neg_inf", {
            "missing_count": 1, "missing_ratio": 0.125, "observed_unique": 3, "pos_inf_count": 0, "neg_inf_count": 2,
            "finite_count": 5, "mean": 12.0, "std": math.sqrt(20.0), "min": 10.0, "q25": 10.0, "median": 10.0,
            "q75": 10.0, "max": 20.0, "iqr_lower": 10.0, "iqr_upper": 10.0, "outlier_candidate_count": 1,
        })
        # +inf·-inf 함께, 유한값 1개: 표준편차와 IQR은 빈칸
        self.assert_stats("both_inf", {
            "missing_count": 5, "observed_unique": 3, "pos_inf_count": 1, "neg_inf_count": 1, "finite_count": 1,
            "mean": 3.0, "std": None, "min": 3.0, "q25": 3.0, "median": 3.0, "q75": 3.0, "max": 3.0,
            "iqr_lower": None, "iqr_upper": None, "outlier_candidate_count": None,
        })
        # 전부 무한대: 유한값 0개, 통계 전부 빈칸
        self.assert_stats("all_inf", {
            "missing_count": 0, "observed_unique": 2, "pos_inf_count": 6, "neg_inf_count": 2, "finite_count": 0,
            "mean": None, "std": None, "min": None, "q25": None, "median": None, "q75": None, "max": None,
            "iqr_lower": None, "iqr_upper": None, "outlier_candidate_count": None,
        })

        for name in ("int_col", "float_col", "pos_inf", "neg_inf", "both_inf", "all_inf", "edge"):
            row = self.row(name)
            self.assertEqual(row["finite_count"] + row["pos_inf_count"] + row["neg_inf_count"], 8 - row["missing_count"])

    def test_iqr_boundary_value_is_not_candidate(self):
        # 1..7,11.5: 상한 6.25+5.25=11.5와 같은 값은 후보가 아니다. 표본분산 77.21875/7
        self.assert_stats("edge", {
            "finite_count": 8, "mean": 4.9375, "std": math.sqrt(11.03125), "q25": 2.75, "q75": 6.25,
            "iqr_lower": -2.5, "iqr_upper": 11.5, "outlier_candidate_count": 0,
        })

    def test_text_boolean_and_mixed_have_no_statistics_or_values(self):
        # 공백만 있는 문자열 2개("  ", " ")는 결측이 아니며 따로 센다. 빈 칸 1개만 결측이다.
        self.assert_stats("text_col", {"missing_count": 1, "observed_unique": 5, "blank_string_count": 2})
        self.assert_stats("bool_col", {"missing_count": 0, "observed_unique": 2, "blank_string_count": None})
        self.assert_stats("mixed", {"missing_count": 0, "observed_unique": 8, "blank_string_count": 0})

        for name in ("text_col", "bool_col", "mixed"):
            row = self.row(name)
            self.assertTrue(all(row[key] is None for key in PROFILE_COLUMNS[6:]), name)

        rendered = self.table.to_csv()
        self.assertNotIn("CANARY", rendered)

    def test_repeated_rows_and_input_unchanged(self):
        frame = pd.read_csv(BytesIO(b"a,b\n1,x\n1,x\n2,y\n"))
        before = frame.copy()
        self.assertEqual(profile_input(frame)["summary"]["repeated_rows"], 2)
        pd.testing.assert_frame_equal(frame, before)

        frame = _frame()
        before = frame.copy()
        profile_input(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_general_numeric_validation_is_not_relaxed(self):
        """진단이 무한대를 세더라도 일반 수치 역할 검사는 그대로 무한대를 거부한다."""
        with self.assertRaises(ValueError):
            TabularInput(_frame(), ColumnRoles(numeric=("pos_inf",))).validate()

    def test_diagnostic_registry_is_separate(self):
        module = build_diagnostic_registry().resolve("input_profile", "1.0.0")
        self.assertEqual(module.contract.preconditions, ("snapshot_verified", "safe_loader_completed"))
        self.assertEqual(module.contract.kind, "analysis")
        self.assertEqual({item.module_id for item in build_diagnostic_registry().contracts()}, {"input_profile", "header_dedupe"})

        for module_id in ("input_profile", "header_dedupe"):
            with self.assertRaises(ValueError):
                build_builtin_registry().resolve(module_id, "1.0.0")

        with self.assertRaises(ValueError):
            module.validate_settings({"iqr_multiplier": 3.0})


if __name__ == "__main__":
    unittest.main()

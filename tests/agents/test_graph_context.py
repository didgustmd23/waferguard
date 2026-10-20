"""검증된 집계 기반 그래프 문맥. 실제 모델·원본 데이터·worker를 실행하지 않는다."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.ui.commentary_policy import default_policy
from src.ui.commentary_service import build_packet
from src.ui.graph_context import context_for_graph


def evidence(name, csv, *, limit=2, prefix="run_eda/eda/"):
    return {"name": prefix + name, "data": csv.encode("utf-8"), "plot_limit": limit}


class GraphContextTest(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            evidence("missing.csv", "feature,missing_count,missing_ratio,observed_unique\na,1,0.1,5\nb,9,0.9,2\nc,8,0.8,3\n"),
            evidence("numeric_statistics.csv", ",count,mean,std,min,25%,50%,75%,max\na,10,100,0,100,100,100,100,100\nb,10,0.12,0.01,0.1,0.11,0.12,0.13,0.14\n"),
            evidence("pearson.csv", ",a,b,c\na,1,-0.2,0.3\nb,-0.2,1,\nc,0.3,,1\n"),
            evidence("correlation_support.csv", ",a,b,c\na,10,8,7\nb,8,10,6\nc,7,6,10\n")]

    def test_missing_reports_only_top_features_with_total_scope(self):
        value = context_for_graph("run_eda/eda/missing.png", self.evidence)
        self.assertEqual(value["total_feature_count"], 3)
        self.assertEqual(value["displayed_feature_count"], 2)
        self.assertEqual([row["feature"] for row in value["values"]], ["b", "c"])
        self.assertEqual(value["values"][0]["missing_ratio"], 0.9)

    def test_correlation_preserves_sign_pair_support_and_null(self):
        value = context_for_graph("run_eda/eda/correlations.png", self.evidence)
        self.assertEqual(value["pairs"], [{"feature_a": "a", "feature_b": "b", "pearson": -0.2, "observed_count": 8}])
        self.assertEqual(value["features"], ["a", "b"])
        large = [item | {"plot_limit": 3} for item in self.evidence]
        expanded = context_for_graph("run_eda/eda/correlations.png", large)
        pair = next(pair for pair in expanded["pairs"] if (pair["feature_a"], pair["feature_b"]) == ("b", "c"))
        self.assertIsNone(pair["pearson"])
        self.assertEqual(expanded["pair_count"], 3)
        self.assertEqual(expanded["pairs"][0]["pearson"], 0.3)
        self.assertTrue(all(pair["feature_a"] != pair["feature_b"] for pair in expanded["pairs"]))
        self.assertEqual(len({frozenset((pair["feature_a"], pair["feature_b"])) for pair in expanded["pairs"]}), 3)

    def test_numeric_statistics_preserve_constant_column_and_units(self):
        value = context_for_graph("run_eda/eda/distributions.png", self.evidence)
        self.assertEqual(value["values"][0]["std"], 0)
        self.assertEqual(value["values"][0]["mean"], 100)
        self.assertIn("결측을 제외", value["statistics_basis"])

    def test_sensor_names_and_coefficients_are_bound_in_the_same_record(self):
        source = [evidence("pearson.csv", ",sensor_3,sensor_6,sensor_7\nsensor_3,1,-0.64,0.3\nsensor_6,-0.64,1,-0.15\nsensor_7,0.3,-0.15,1\n", limit=3),
                  evidence("correlation_support.csv", ",sensor_3,sensor_6,sensor_7\nsensor_3,10,8,7\nsensor_6,8,10,6\nsensor_7,7,6,10\n", limit=3)]
        pairs = context_for_graph("run_eda/eda/correlations.png", source)["pairs"]
        values = {(pair["feature_a"], pair["feature_b"]): pair for pair in pairs}
        self.assertEqual(values[("sensor_3", "sensor_6")]["pearson"], -0.64)
        self.assertEqual(values[("sensor_6", "sensor_7")]["pearson"], -0.15)
        self.assertEqual(values[("sensor_6", "sensor_7")]["observed_count"], 6)

    def test_mismatched_matrix_axes_are_rejected_instead_of_relabelled(self):
        source = deepcopy(self.evidence)
        source[2]["data"] = b",b,a,c\na,1,-0.2,0.3\nb,-0.2,1,0\nc,0.3,0,1\n"
        with self.assertRaises(ValueError):
            context_for_graph("run_eda/eda/correlations.png", source)

    def test_other_run_tables_are_not_mixed_and_unknown_scope_is_not_guessed(self):
        value = context_for_graph("other/eda/missing.png", self.evidence)
        self.assertEqual(value["status"], "unavailable")
        broken = deepcopy(self.evidence)
        broken[0]["plot_limit"] = 99
        with self.assertRaises(ValueError):
            context_for_graph("run_eda/eda/missing.png", broken)

    def test_graph_payload_includes_evidence_only_when_tables_allowed(self):
        result = {"status": "completed", "files": [{"name": "run_eda/eda/missing.png", "data": b"fake-png"}],
                  "graph_evidence": self.evidence}
        with patch("src.ui.commentary_service.valid_report_png", return_value=True):
            packet = build_packet(result, default_policy() | {"graphs": True})
            self.assertEqual(packet["assets"][0]["graph_context"]["total_feature_count"], 3)
            with patch("src.ui.commentary_service.context_for_graph", side_effect=AssertionError("수치 접근 금지")):
                packet = build_packet(result, default_policy() | {"graphs": True, "tables": False})
                self.assertEqual(packet["assets"][0]["graph_context"]["status"], "not_permitted")
                self.assertNotIn("missing_ratio", str(packet))

    def test_graph_permission_off_does_not_read_evidence(self):
        result = {"status": "completed", "files": [{"name": "missing.png", "data": b"fake-png"}],
                  "graph_evidence": self.evidence}
        with patch("src.ui.commentary_service.context_for_graph", side_effect=AssertionError("접근 금지")):
            with self.assertRaises(ValueError):
                build_packet(result, default_policy())


if __name__ == "__main__":
    unittest.main()

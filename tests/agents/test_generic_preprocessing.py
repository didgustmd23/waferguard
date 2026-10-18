"""독립 표형 데이터로 범용 계약·EDA·프로파일 실행 계획을 검증한다."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.agents.adapters.registry import create_eda_adapter
from src.agents.approval_store import ApprovalStore
from src.agents.contracts import ToolResult
from src.agents.executor import RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.agents.workflow import build_profile_plan
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.eda import collect_eda
from src.preprocessing.eda_report import generate_eda_report
from src.preprocessing.profiles import DatasetProfile, load_profile, profile_from_dict
from src.preprocessing.quality import QualityRules
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules


class GenericPreprocessingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.roles = ColumnRoles(
            numeric=("measurement_a", "measurement_b"),
            categorical=("process_type",), target="quality_label",
            group="batch_id", timestamp="captured_at", identifiers=("sample_id",),
        )
        self.frame = pd.DataFrame({
            "measurement_a": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
            "measurement_b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "process_type": ["a", "a", "b", "b", "a", "b"],
            "quality_label": ["normal", "review", "reject"] * 2,
            "batch_id": ["batch_1"] * 3 + ["batch_2"] * 3,
            "sample_id": list(range(6)),
            "captured_at": pd.date_range("2026-01-01", periods=6),
        })
        self.profile = DatasetProfile(
            "independent_example", "tabular_csv_v1", "eda_review_v1", self.roles,
            quality_rules=QualityRules(timestamp_format="%Y-%m-%d"),
        )

    def test_numeric_roles_and_multiclass_summary(self):
        """클래스 수를 고정하지 않으며 ID·target·group은 상관분석에서 제외한다."""
        data = TabularInput(self.frame, self.roles)
        tables = collect_eda(data)

        self.assertEqual(tables["summary"]["row_count"], 6)
        self.assertEqual(set(tables["pearson"].columns), set(self.roles.numeric))
        self.assertEqual(tables["correlation_support"].loc["measurement_a", "measurement_b"], 5)
        self.assertAlmostEqual(tables["pearson"].iloc[0, 1], 1.0)
        self.assertEqual(len(tables["target_distribution"]), 3)
        self.assertEqual(tables["missing"].loc["measurement_a", "missing_count"], 1)
        self.assertNotIn("captured_at", tables["pearson"].columns)

    def test_unlabeled_and_categorical_only_data(self):
        """라벨과 수치 특징이 없는 데이터도 명시적 계약으로 분석한다."""
        roles = ColumnRoles(categorical=("process_type",))
        tables = collect_eda(TabularInput(self.frame, roles))

        self.assertTrue(tables["target_distribution"].empty)
        self.assertTrue(tables["pearson"].empty)

        result = generate_eda_report(TabularInput(self.frame, roles), self.root / "categorical_report")

        self.assertEqual(result, {"status": "eda_complete"})

    def test_invalid_roles_and_inputs(self):
        """역할 중복·필수 컬럼 누락·비수치·무한대를 거부한다."""
        with self.assertRaises(ValueError):
            ColumnRoles(numeric=("measurement_a",), target="measurement_a")

        with self.assertRaises(ValueError):
            TabularInput(self.frame.drop(columns="measurement_a"), self.roles).validate()

        bad = self.frame.copy()
        bad["measurement_a"] = "not_numeric"

        with self.assertRaises(ValueError):
            TabularInput(bad, self.roles).validate()

        bad = self.frame.copy()
        bad.loc[0, "measurement_a"] = np.inf

        with self.assertRaises(ValueError):
            TabularInput(bad, self.roles).validate()

    def test_profile_roundtrip_and_unknown_fields(self):
        """프로파일 변경을 식별하고 임의 실행 필드·알 수 없는 템플릿을 거부한다."""
        raw = self.profile.to_dict()

        self.assertEqual(profile_from_dict(raw), self.profile)
        self.assertNotEqual(replace(self.profile, max_plot_features=5).signature, self.profile.signature)

        raw["python_module"] = "untrusted"

        with self.assertRaises(ValueError):
            profile_from_dict(raw)

        with self.assertRaises(ValueError):
            replace(self.profile, workflow_id="arbitrary_workflow")

    def test_profile_workflow_and_changed_configuration(self):
        """EDA 계획에 학습 입력 생성을 강제하지 않고 프로파일을 계획 ID에 연결한다."""
        plan = build_profile_plan(self.profile, "synthetic_source")
        stages = tuple(step.stage for step in plan.steps)

        self.assertNotIn("prepare_cache", stages)
        self.assertNotIn("commit_split", stages)
        self.assertIn("review_eda", stages)
        self.assertEqual(plan.profile_id, self.profile.profile_id)

        changed = build_profile_plan(replace(self.profile, max_plot_features=4), "synthetic_source")

        self.assertNotEqual(plan.plan_id, changed.plan_id)

        with self.assertRaises(ValueError):
            replace(plan, steps=plan.steps[:-1])

        training = build_profile_plan(
            replace(self.profile, workflow_id="training_preparation_v1", split_rules=SplitRules(), preparation_rules=PreparationRules()), "synthetic_source",
        )

        self.assertIn("prepare_data", [step.stage for step in training.steps])

    def test_csv_profile_runs_with_existing_executor(self):
        """WM과 무관한 CSV도 같은 실행기·승인·공통 EDA로 완료한다."""
        path = self.root / "synthetic.csv"
        self.frame.to_csv(path, index=False)
        adapter = create_eda_adapter(self.profile, path)
        plan = build_profile_plan(self.profile, adapter.source_id)
        states = LocalStateStore(self.root / "run")
        states.create("synthetic_operator", policy_version=plan.policy_version)
        approvals = ApprovalStore(self.root / "approvals.sqlite3")

        def check_tool(output, arguments):
            (output / "check.txt").write_text("synthetic_passed", encoding="utf-8")

            return ToolResult(status="passed")

        tools = [adapter.registered_tool(), adapter.audit_tool()]

        for step in plan.steps:
            if step.action_id not in {"run_eda", "run_audit"}:
                tools.append(RegisteredTool(
                    step.action_id, check_tool,
                    lambda output, arguments: (output / "check.txt").read_text() == "synthetic_passed",
                ))

        executor = RulesExecutor(states, approvals, plan, tools)

        for step in plan.steps:
            result = executor.run_next()

            if result.code == "APPROVAL_REQUIRED":
                approval = approvals.record_confirmed_approval(executor.scope_for(step.stage))
                result = executor.run_next(approval_id=approval.approval_id)

            self.assertEqual(result.code, "STEP_COMPLETED")

        self.assertEqual(states.load().status, "completed")

    def test_source_change_and_unknown_adapter(self):
        """CSV 변경과 미등록 어댑터를 거부한다."""
        path = self.root / "synthetic.csv"
        self.frame.to_csv(path, index=False)
        adapter = create_eda_adapter(self.profile, path)
        path.write_text("changed", encoding="utf-8")

        with self.assertRaises(ValueError):
            adapter._check_input(adapter.arguments)

        with self.assertRaises(ValueError):
            create_eda_adapter(replace(self.profile, adapter_id="unregistered"), path)

        path.write_text("measurement_a,measurement_a\n1,2\n", encoding="utf-8")
        duplicate_adapter = create_eda_adapter(self.profile, path)

        with self.assertRaisesRegex(ValueError, "중복"):
            duplicate_adapter.load_input()

    def test_sample_profiles_and_dependency_direction(self):
        """공통 코어가 agents·wafer_dl·sensor_ml을 import하지 않는지 확인한다."""
        root = Path(__file__).resolve().parents[2]

        for path in (root / "configs/agents/profiles").glob("*.json"):
            self.assertIsInstance(load_profile(path), DatasetProfile)

        for path in (root / "src/preprocessing").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(any(part in module.split(".") for part in ("agents", "wafer_dl", "sensor_ml")))


if __name__ == "__main__":
    unittest.main()

"""합성 입력으로 품질 기준의 경계·예외·실행기 차단을 검증한다."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.agents.adapters.registry import create_eda_adapter
from src.agents.approval_store import ApprovalStore
from src.agents.contracts import ToolResult
from src.agents.executor import RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.agents.workflow import build_profile_plan
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.profiles import DatasetProfile, profile_from_dict
from src.preprocessing.quality import QualityRules, inspect_quality, write_quality_report


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.roles = ColumnRoles(
            numeric=("x", "y"), categorical=("kind",), target="label",
            group="group_id", timestamp="time", identifiers=("id",),
        )
        self.frame = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0],
            "kind": ["a", "b", "a", "b"], "label": ["ok", "bad", "ok", "bad"],
            "group_id": ["g1", "g1", "g2", "g2"], "id": [1, 2, 3, 4],
            "time": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        })
        self.rules = QualityRules(allowed_labels=("ok", "bad"), timestamp_format="%Y-%m-%d")

    def inspect(self, frame=None, *, rules=None, roles=None):
        return inspect_quality(
            TabularInput(self.frame if frame is None else frame, roles or self.roles),
            rules or self.rules,
        )

    def test_pass_and_no_mutation(self):
        """정상 입력을 통과시키고 원본 DataFrame을 변경하지 않는다."""
        before = self.frame.copy(deep=True)
        report = self.inspect()

        self.assertEqual(report.status, "passed")
        self.assertEqual(report.findings, ())
        pd.testing.assert_frame_equal(before, self.frame)

    def test_missing_boundary_and_constant_candidates(self):
        """50%는 초과가 아니며 75%·상수·전체 결측은 검토 대상으로 남긴다."""
        frame = self.frame.copy()
        frame.loc[:1, "x"] = np.nan

        self.assertEqual(self.inspect(frame).status, "passed")

        frame.loc[2, "x"] = np.nan
        report = self.inspect(frame)
        codes = {finding.code for finding in report.findings}

        self.assertEqual(report.status, "review_required")
        self.assertIn("HIGH_MISSING_RATIO", codes)
        self.assertIn("CONSTANT_FEATURE", codes)

        frame["x"] = np.nan
        frame["kind"] = pd.Categorical([None] * 4)
        codes = {finding.code for finding in self.inspect(frame).findings}

        self.assertIn("ALL_MISSING_FEATURE", codes)

    def test_structural_and_numeric_errors_are_blocked(self):
        """필수 컬럼·중복 컬럼·비수치·무한대는 통지 수준과 무관하게 차단한다."""
        cases = [
            self.frame.drop(columns="x"),
            self.frame.iloc[:0],
            self.frame.assign(x="wrong_type"),
            self.frame.assign(x=np.inf),
            pd.concat([self.frame, self.frame[["x"]]], axis=1),
        ]

        for index, frame in enumerate(cases):
            with self.subTest(case=index):
                self.assertEqual(self.inspect(frame).status, "blocked_input")

    def test_target_values_and_missing_policy(self):
        """미지 label은 차단하고 라벨 결측은 설정한 정책으로 처리한다."""
        frame = self.frame.copy()
        frame.loc[0, "label"] = "PRIVATE_LABEL_CANARY"
        report = self.inspect(frame)

        self.assertEqual(report.status, "blocked_input")
        self.assertNotIn("PRIVATE_LABEL_CANARY", json.dumps(report.to_dict()))

        frame.loc[0, "label"] = " "

        self.assertEqual(self.inspect(frame).status, "review_required")
        self.assertEqual(self.inspect(frame, rules=replace(self.rules, target_missing="block")).status, "blocked_input")
        self.assertEqual(self.inspect(frame, rules=replace(self.rules, target_missing="ignore")).status, "passed")

    def test_numeric_labels_and_bool_are_distinct(self):
        """숫자 label의 CSV float 변환은 허용하지만 bool을 숫자로 혼용하지 않는다."""
        frame = self.frame.copy()
        frame["label"] = [1.0, 0.0, 1.0, 0.0]
        rules = replace(self.rules, allowed_labels=(0, 1))

        self.assertEqual(self.inspect(frame, rules=rules).status, "passed")

        frame["label"] = [True, False, True, False]

        self.assertEqual(self.inspect(frame, rules=rules).status, "blocked_input")

    def test_unlabeled_data_and_identifier_groups(self):
        """target 없는 입력과 복합 ID를 지원하고 중복·결측을 구분한다."""
        roles = replace(self.roles, target=None)
        rules = replace(self.rules, allowed_labels=None)

        self.assertEqual(self.inspect(roles=roles, rules=rules).status, "passed")

        frame = self.frame.copy()
        frame.loc[1, "id"] = 1
        rules = replace(self.rules, duplicate_id="block")

        self.assertEqual(self.inspect(frame, rules=rules).status, "blocked_input")
        frame["id_part"] = frame["kind"]
        composite_roles = replace(self.roles, identifiers=("id", "id_part"))

        self.assertEqual(self.inspect(frame, rules=rules, roles=composite_roles).status, "passed")

        frame.loc[0, "group_id"] = None
        frame.loc[0, "id"] = np.nan
        codes = {finding.code for finding in self.inspect(frame).findings}

        self.assertIn("GROUP_MISSING", codes)
        self.assertIn("IDENTIFIER_MISSING", codes)

    def test_timestamp_requires_format_and_rejects_invalid_value(self):
        """형식 미지정은 검토로 처리하고 지정한 형식의 오류는 차단한다."""
        rules = replace(self.rules, timestamp_format=None)

        self.assertEqual(self.inspect(rules=rules).status, "review_required")

        frame = self.frame.copy()
        frame.loc[0, "time"] = "invalid_time"

        self.assertEqual(self.inspect(frame).status, "blocked_input")

        frame["time"] = pd.to_datetime(self.frame["time"])

        self.assertEqual(self.inspect(frame, rules=rules).status, "passed")

    def test_rules_and_profile_serialization(self):
        """잘못된 기준을 거부하고 설정 변경을 프로파일 식별값에 반영한다."""
        for value in (-0.1, 1.1, True, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    QualityRules(missing_ratio_threshold=value)

        with self.assertRaises(ValueError):
            QualityRules(allowed_labels=(1, 1.0))

        profile = DatasetProfile("quality_example", "tabular_csv_v1", "eda_review_v1", self.roles)
        protected = replace(profile, quality_rules=self.rules)

        self.assertEqual(profile_from_dict(protected.to_dict()), protected)
        self.assertNotEqual(profile.signature, protected.signature)

    def test_local_report_preserves_status(self):
        """검토 보고서는 완료·승인으로 바꾸지 않고 상세 근거를 로컬에 보존한다."""
        frame = self.frame.assign(x=1.0)
        report = self.inspect(frame)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "quality"
            write_quality_report(report, output)

            self.assertEqual(json.loads((output / "quality.json").read_text()), report.to_dict())
            self.assertFalse((output / "SUCCESS.json").exists())

    def test_bad_csv_retains_blocked_report(self):
        """CSV 헤더 오류도 원문 예외 대신 로컬 차단 보고서로 남긴다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "invalid.csv"
            source.write_text("x,x\n1,2\n", encoding="utf-8")
            profile = DatasetProfile("invalid_csv", "tabular_csv_v1", "eda_review_v1", self.roles)
            adapter = create_eda_adapter(profile, source)
            output = root / "attempt"
            output.mkdir()
            result = adapter.audit_tool().execute(output, adapter.arguments)
            report = json.loads((output / "quality/quality.json").read_text())

            self.assertEqual(result.status, "blocked_input")
            self.assertEqual(report["findings"][0]["code"], "INPUT_LOAD_CONTRACT_INVALID")
            self.assertEqual(result.metrics, {})

    def test_executor_blocks_eda_after_audit_review(self):
        """품질 검토 필요 시 후속 EDA를 실행하지 않고 부분 보고서를 보존한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "input.csv"
            self.frame.assign(x=1.0).to_csv(path, index=False)
            profile = DatasetProfile(
                "quality_gate", "tabular_csv_v1", "eda_review_v1", self.roles,
                quality_rules=self.rules,
            )
            adapter = create_eda_adapter(profile, path)
            plan = build_profile_plan(profile, adapter.source_id)
            states = LocalStateStore(root / "run")
            states.create("synthetic_operator", policy_version=plan.policy_version)

            def check(output, arguments):
                (output / "check.txt").write_text("passed")

                return ToolResult(status="passed")

            tools = [adapter.registered_tool(), adapter.audit_tool()]

            for step in plan.steps:
                if step.action_id not in {"run_audit", "run_eda"}:
                    tools.append(RegisteredTool(
                        step.action_id, check,
                        lambda output, arguments: (output / "check.txt").read_text() == "passed",
                    ))

            executor = RulesExecutor(states, ApprovalStore(root / "approvals.sqlite3"), plan, tools)
            self.assertEqual(executor.run_next().code, "STEP_COMPLETED")
            self.assertEqual(executor.run_next().code, "TOOL_REVIEW_REQUIRED")
            self.assertEqual(executor.run_next().code, "RUN_NOT_EXECUTABLE")
            self.assertEqual(list(states.directory.rglob("eda.md")), [])
            self.assertEqual(len(list(states.directory.rglob("quality.json"))), 1)


if __name__ == "__main__":
    unittest.main()

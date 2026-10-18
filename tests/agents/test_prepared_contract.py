"""독립 최종 계약 검사·로컬 보고서·실행 단계 연결을 합성 입력으로 검증한다."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents.adapters.tabular import csv_eda_adapter
from src.agents.approval_store import ApprovalStore
from src.agents.artifacts import ArtifactContext, complete_artifacts
from src.agents.contracts import ToolResult
from src.agents.executor import RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.agents.workflow import build_profile_plan
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.prepared_contract import inspect_prepared
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules
from src.preprocessing.report import render_preparation_report
from src.preprocessing.split import SplitRules, propose_split
from src.preprocessing.transform import PreparationRules, TabularPreprocessor


class PreparedContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "synthetic.csv"
        frame = pd.DataFrame({
            "value": np.arange(120, dtype=float), "constant": 1.0,
            "kind": ["A", "B"] * 60, "label": ["normal", "fail"] * 60,
        })
        frame.loc[:5, "value"] = np.nan
        frame.to_csv(self.source, index=False)
        self.profile = DatasetProfile(
            "synthetic", "tabular_csv_v1", "tabular_preparation_v1",
            ColumnRoles(numeric=("value", "constant"), categorical=("kind",), target="label"),
            quality_rules=QualityRules(constant_feature="ignore"),
            split_rules=SplitRules(stratify=True),
            preparation_rules=PreparationRules(drop_constant=True, scaling="standard"),
        )
        self.adapter, self.prepared, self.commit = self.fixture(self.profile, "base")

    def fixture(self, profile, name):
        """실제 데이터 없이 등록된 준비 도구로 검사 대상 산출물을 만든다."""
        adapter = csv_eda_adapter(self.source, profile)
        proposal = propose_split(
            adapter.load_input(), profile.split_rules,
            source_id=adapter.source_id, config_id=adapter.config_id,
        )
        commit = self.root / f"{name}_commit"
        commit.mkdir()
        (commit / "split.json").write_text(json.dumps(proposal), encoding="utf-8")
        output = self.root / f"{name}_prepared"
        output.mkdir()
        result = adapter.preparation_tool().execute(output, adapter.arguments, commit)
        self.assertEqual(result.status, "passed")
        complete_artifacts(output, ArtifactContext(adapter.source_id, adapter.config_id, "fixture_plan", "preprocess_policy_v1", "prepare_data"))

        return adapter, output, commit

    def inspect(self, *, directory=None, adapter=None):
        adapter = adapter or self.adapter

        return inspect_prepared(
            directory or self.prepared, adapter.load_input(), adapter.profile.preparation_rules,
            adapter.profile.split_rules, source_id=adapter.source_id, config_id=adapter.config_id,
        )

    def test_success_without_refitting(self):
        """독립 검사에서 fit 없이 행·컬럼·정답 계약을 확인한다."""
        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("재학습 금지")):
            result = self.inspect()

        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(check["passed"] for check in result["checks"]))
        self.assertEqual(result["summary"]["retained_input_features"], 2)
        self.assertEqual([row["rows"] for row in result["summary"]["splits"]], [84, 18])
        self.assertEqual(sum(row["output_missing_count"] for row in result["summary"]["splits"]), 0)

    def test_row_order_and_target_mismatch(self):
        """행 위치와 정답을 각각 변경하면 행 연결 검사에서 중단한다."""
        path = self.prepared / "row_contract.json"
        original = path.read_text(encoding="utf-8")

        for field in ("row_positions", "targets"):
            value = json.loads(original)

            if field == "row_positions":
                value["train"][field].reverse()
            else:
                value["train"][field][0] = "changed"

            path.write_text(json.dumps(value), encoding="utf-8")
            result = self.inspect()
            self.assertEqual(result["status"], "blocked_input")
            self.assertEqual(result["checks"][-1]["code"], "ROW_ALIGNMENT")

        path.write_text(original, encoding="utf-8")

    def test_output_columns_order_and_dtype(self):
        """출력 열 순서와 수치 타입을 저장 변환기 계약에 대조한다."""
        path = self.prepared / "train_features.csv"
        original = path.read_text(encoding="utf-8")
        frame = pd.read_csv(path)
        frame.iloc[:, ::-1].to_csv(path, index=False)
        self.assertEqual(self.inspect()["checks"][-1]["code"], "TRAIN_OUTPUT_SCHEMA")
        path.write_text(original, encoding="utf-8")
        frame.iloc[0, 0] = np.nan
        frame.to_csv(path, index=False)
        self.assertEqual(self.inspect()["checks"][-1]["code"], "TRAIN_VALUE_POLICY")

    def test_nan_and_infinite_policy(self):
        """NaN은 none 대치의 수치 출력에만 허용하고 원핫·무한값은 거부한다."""
        profile = replace(self.profile, preparation_rules=PreparationRules(drop_constant=True, numeric_imputation="none"))
        adapter, output, _ = self.fixture(profile, "nan")
        self.assertEqual(self.inspect(directory=output, adapter=adapter)["status"], "passed")
        path = output / "validation_features.csv"
        frame = pd.read_csv(path)
        frame.iloc[0, 1] = np.nan
        frame.to_csv(path, index=False)
        self.assertEqual(self.inspect(directory=output, adapter=adapter)["checks"][-1]["code"], "VALIDATION_VALUE_POLICY")
        frame.iloc[0, 1] = np.inf
        frame.to_csv(path, index=False)
        self.assertEqual(self.inspect(directory=output, adapter=adapter)["status"], "blocked_input")

    def test_value_mismatch(self):
        """형태가 맞아도 저장 변환기의 실제 transform 결과가 다르면 실패한다."""
        path = self.prepared / "validation_features.csv"
        frame = pd.read_csv(path)
        frame.iloc[0, 0] += 1
        frame.to_csv(path, index=False)
        result = self.inspect()

        self.assertEqual(result["checks"][-1]["code"], "VALIDATION_TRANSFORM_MATCH")
        self.assertNotIn("summary", result)

    def test_split_and_source_binding(self):
        """확정 분할·원본 식별값 변경을 거부한다."""
        path = self.prepared / "split_contract.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_id"] = "changed"
        path.write_text(json.dumps(value), encoding="utf-8")
        self.assertEqual(self.inspect()["checks"][-1]["code"], "SPLIT_BINDING")

    def test_file_set_and_malformed_metadata(self):
        """추가 파일·손상 JSON은 고정 실패 코드만 반환한다."""
        extra = self.prepared / "unexpected.txt"
        extra.write_text("synthetic", encoding="utf-8")
        self.assertEqual(self.inspect()["checks"][-1]["code"], "PREPARED_FILES")
        extra.unlink()
        (self.prepared / "preparation.json").write_text('{"status": NaN}', encoding="utf-8")
        result = self.inspect()

        self.assertEqual(result["checks"][-1]["code"], "PREPARATION_BINDING")
        self.assertNotIn("summary", result)
        self.assertNotIn(str(self.source), json.dumps(result))

    def test_report_determinism_and_escaping(self):
        """보고서는 결정적 한국어 템플릿이며 특수 컬럼명이 표·HTML로 삽입되지 않는다."""
        result = self.inspect()
        report = render_preparation_report(result)
        self.assertEqual(report, render_preparation_report(result))
        self.assertIn("Train에서만", report)
        self.assertIn("Test는 변환하지", report)
        modified = deepcopy(result)
        modified["summary"]["removed"][0]["column"] = "<script>|\n`"
        escaped = render_preparation_report(modified)
        self.assertNotIn("<script>", escaped)
        self.assertIn("&lt;script&gt;&#124;", escaped)

        with self.assertRaises(ValueError):
            render_preparation_report({**result, "status": "blocked_input"})

    def test_registered_finalization_and_report_tampering(self):
        """검사 도구와 보고서 도구가 재학습 없이 연결되고 보고서 변경을 탐지한다."""
        inspect_tool, report_tool = self.adapter.finalization_tools()
        checked = self.root / "checked"
        checked.mkdir()
        report = self.root / "report"
        report.mkdir()

        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("재학습 금지")):
            self.assertEqual(inspect_tool.execute(checked, self.adapter.arguments, self.prepared).status, "passed")
            self.assertTrue(inspect_tool.validate(checked, self.adapter.arguments, self.prepared))
            self.assertEqual(report_tool.execute(report, self.adapter.arguments, checked).status, "passed")
            self.assertTrue(report_tool.validate(report, self.adapter.arguments, checked))

        (report / "report.md").write_text("changed", encoding="utf-8")
        self.assertFalse(report_tool.validate(report, self.adapter.arguments, checked))

    def test_failed_check_has_no_completion_or_report(self):
        """독립 검사 실패에는 완료 요약·보고서 컨텍스트를 생성하지 않는다."""
        (self.prepared / "row_contract.json").write_text("{}", encoding="utf-8")
        checked = self.root / "failed_check"
        checked.mkdir()
        inspect_tool, _ = self.adapter.finalization_tools()
        result = inspect_tool.execute(checked, self.adapter.arguments, self.prepared)

        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((checked / "AGENT_COMPLETE.json").exists())
        self.assertFalse((checked / "report_context.json").exists())

    def executor(self):
        """사전 점검·사람 검토만 합성 대체하고 나머지는 실제 등록 도구를 연결한다."""
        plan = build_profile_plan(self.profile, self.adapter.source_id)
        states = LocalStateStore(self.root / "run")
        states.create("synthetic_actor", policy_version=plan.policy_version)
        approvals = ApprovalStore(self.root / "approval.sqlite3")

        def preflight(output, arguments):
            (output / "preflight.txt").write_text("synthetic", encoding="utf-8")

            return ToolResult(status="passed")

        tools = [
            self.adapter.audit_tool(), self.adapter.registered_tool(), *self.adapter.split_tools(),
            self.adapter.preparation_tool(), *self.adapter.finalization_tools(),
            RegisteredTool("run_preflight", preflight, lambda output, args: (output / "preflight.txt").is_file()),
            RegisteredTool("review_eda", preflight, lambda output, args: (output / "preflight.txt").is_file()),
        ]
        executor = RulesExecutor(states, approvals, plan, tools)

        return executor, states, approvals, plan

    def advance(self, executor, approvals, step):
        """테스트용 확인을 마친 뒤에만 합성 승인을 발급한다."""
        result = executor.run_next()

        if result.code == "APPROVAL_REQUIRED":
            record = approvals.record_confirmed_approval(executor.scope_for(step.stage))
            result = executor.run_next(approval_id=record.approval_id)

        self.assertEqual(result.code, "STEP_COMPLETED", (step.stage, result))

    def test_executor_through_final_report(self):
        """품질·EDA·분할·준비·최종 검사·보고서를 합성 입력으로 연결 실행한다."""
        executor, states, approvals, plan = self.executor()

        for step in plan.steps:
            self.advance(executor, approvals, step)

        state = states.load()
        self.assertEqual(state.status, "completed")
        report = states.directory / state.artifact_refs["render_report"]
        self.assertTrue((report / "report.md").is_file())
        self.assertEqual(json.loads((report / "report.json").read_text())["status"], "passed")

    def test_prepared_change_blocks_report_stage(self):
        """최종 검사 후라도 준비 산출물이 변경되면 보고서 단계 전에 차단한다."""
        executor, states, approvals, plan = self.executor()

        for step in plan.steps[:-1]:
            self.advance(executor, approvals, step)

        state = states.load()
        output = states.directory / state.artifact_refs["prepare_data"]
        (output / "train_features.csv").write_text("changed", encoding="utf-8")
        result = executor.run_next()

        self.assertEqual(result.code, "PREREQUISITE_ARTIFACT_INVALID")
        self.assertNotIn("render_report", states.load().artifact_refs)

    def test_unknown_ignored_category_is_reported(self):
        """ignore로 진행한 새 범주를 정상 범주로 숨기지 않고 건수로 기록한다."""
        frame = pd.read_csv(self.source)
        frame["kind"] = [f"kind_{index}" for index in range(len(frame))]
        frame.to_csv(self.source, index=False)
        profile = replace(self.profile, preparation_rules=PreparationRules(drop_constant=True, unknown_category="ignore"))
        adapter, output, _ = self.fixture(profile, "unknown")
        result = self.inspect(directory=output, adapter=adapter)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["summary"]["splits"][1]["unknown_category_count"], 18)
        self.assertIn("새 범주 발생", render_preparation_report(result))


if __name__ == "__main__":
    unittest.main()

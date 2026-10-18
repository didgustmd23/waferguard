"""합성 데이터로 Train-only 학습·범주·저장·에이전트 연결을 검증한다."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.agents.adapters.tabular import csv_eda_adapter
from src.agents.approval_store import ApprovalStore
from src.agents.contracts import ToolResult
from src.agents.executor import RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.agents.workflow import build_profile_plan
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.profiles import DatasetProfile, profile_from_dict
from src.preprocessing.split import SplitRules, propose_split
from src.preprocessing.transform import (
    PreparationRules, TabularPreprocessor, build_preparation,
    load_preparation, prepare_splits, save_preparation,
)


class GenericTransformTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.roles = ColumnRoles(numeric=("value",), categorical=("kind",), target="label", identifiers=("id",))
        self.train = pd.DataFrame({
            "value": [1.0, 2.0, np.nan, 4.0], "kind": ["A", "B", None, "A"],
            "label": [0, 1, 0, 1], "id": range(4),
        })

    def test_train_statistics_and_feature_only_inference(self):
        """평균·대치값은 Train 기준이고 추론에는 라벨·ID가 필요하지 않다."""
        pipeline = build_preparation(self.roles, PreparationRules(scaling="standard", unknown_category="ignore"))
        before = self.train.copy(deep=True)
        pipeline.fit(self.train)
        transformer = pipeline.named_steps["preprocess"]
        record = transformer.state_["records"][0]
        self.assertEqual(record["impute"], 2.0)
        self.assertEqual(record["center"], 2.25)
        saved_state = json.dumps(transformer.state_)
        validation = pd.DataFrame({"kind": ["new", "A"], "value": [1_000_000.0, np.nan]})
        output = pipeline.transform(validation)

        self.assertEqual(json.dumps(transformer.state_), saved_state)
        self.assertEqual(output.iloc[0, 1:].sum(), 0.0)
        self.assertEqual(transformer.diagnose(validation)[1]["unknown_category_count"], 1)
        self.assertAlmostEqual(output.iloc[1, 0], (2.0 - 2.25) / record["scale"])
        self.assertFalse(any("label" in name or "id__" in name for name in output.columns))
        pd.testing.assert_frame_equal(before, self.train)

    def test_unknown_category_error_and_reserved_missing(self):
        """새 범주는 기본 오류이며 실제 missing 문자열과 결측 전용 범주를 구분한다."""
        pipeline = build_preparation(self.roles, PreparationRules()).fit(self.train)

        with self.assertRaisesRegex(ValueError, "범주"):
            pipeline.transform(pd.DataFrame({"value": [1.0], "kind": ["new"]}))

        train = self.train.copy()
        train["kind"] = ["missing", None, "missing", None]
        transformed = build_preparation(self.roles, PreparationRules()).fit_transform(train)
        self.assertFalse(np.array_equal(transformed.iloc[0, 1:], transformed.iloc[1, 1:]))

    def test_missing_filter_strict_boundary_and_constant(self):
        """50% 초과만 제거하며 상수 제거도 학습 구간에서만 결정한다."""
        roles = ColumnRoles(numeric=("half", "high", "constant"))
        frame = pd.DataFrame({"half": [1.0, np.nan, 2.0, np.nan], "high": [1.0, np.nan, np.nan, np.nan], "constant": [3.0] * 4})
        rules = PreparationRules(missing_ratio_threshold=0.5, drop_constant=True)
        pipeline = build_preparation(roles, rules).fit(frame)
        removed = pipeline.named_steps["preprocess"].state_["removed"]

        self.assertEqual({row["column"] for row in removed}, {"high", "constant"})
        self.assertEqual(pipeline.transform(frame).shape, (4, 1))

    def test_scaling_and_no_imputation(self):
        """각 스케일 방식의 고정 기준과 NaN 보존 옵션을 확인한다."""
        roles = ColumnRoles(numeric=("value",))
        frame = pd.DataFrame({"value": [0.0, 2.0, 4.0]})

        for method, expected in (("standard", [-1.224744871391589, 0.0, 1.224744871391589]), ("minmax", [0.0, 0.5, 1.0]), ("robust", [-1.0, 0.0, 1.0])):
            with self.subTest(method=method):
                actual = build_preparation(roles, PreparationRules(scaling=method)).fit_transform(frame)
                np.testing.assert_allclose(actual.iloc[:, 0], expected)

        pipeline = build_preparation(roles, PreparationRules(numeric_imputation="none")).fit(frame)
        self.assertTrue(np.isnan(pipeline.transform(pd.DataFrame({"value": [np.nan]})).iloc[0, 0]))
        minmax = build_preparation(roles, PreparationRules(scaling="minmax")).fit(frame)
        self.assertEqual(minmax.transform(pd.DataFrame({"value": [8.0]})).iloc[0, 0], 2.0)

    def test_iqr_clip_preserves_rows_and_train_bounds(self):
        """이상치 행을 삭제하지 않고 저장한 Train 경계로만 clip한다."""
        roles = ColumnRoles(numeric=("value",))
        pipeline = build_preparation(roles, PreparationRules(outlier="iqr_clip")).fit(pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}))
        actual = pipeline.transform(pd.DataFrame({"value": [-1000.0, 1000.0]}))

        self.assertEqual(actual.shape, (2, 1))
        np.testing.assert_allclose(actual.iloc[:, 0], [-1.5, 4.5])

    def test_binning_fixed_edges_and_onehot(self):
        """범주화 경계는 Train에서 고정하며 범위를 벗어난 값도 끝 구간으로 배치한다."""
        roles = ColumnRoles(numeric=("value",))

        for method in ("quantile", "uniform"):
            pipeline = build_preparation(roles, PreparationRules(binning=method, n_bins=2, bin_encoding="onehot"))
            pipeline.fit(pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}))
            actual = pipeline.transform(pd.DataFrame({"value": [-100.0, 100.0]}))
            np.testing.assert_array_equal(actual, [[1.0, 0.0], [0.0, 1.0]])

        constant = build_preparation(roles, PreparationRules(binning="quantile", bin_encoding="onehot"))
        self.assertEqual(constant.fit_transform(pd.DataFrame({"value": [1.0] * 4})).shape, (4, 1))

    def test_most_frequent_and_category_types(self):
        """결측을 Train 최빈값으로 대치하고 숫자·문자열·bool 범주 충돌을 막는다."""
        rules = PreparationRules(categorical_imputation="most_frequent")
        transformed = build_preparation(self.roles, rules).fit_transform(self.train)
        np.testing.assert_array_equal(transformed.iloc[0, 1:], transformed.iloc[2, 1:])
        frame = pd.DataFrame({"kind": pd.Series([1, "1", True], dtype=object)})
        roles = ColumnRoles(categorical=("kind",))
        pipeline = build_preparation(roles, PreparationRules()).fit(frame)
        self.assertEqual(len(pipeline.named_steps["preprocess"].state_["records"][0]["categories"]), 4)

    def test_invalid_inputs_and_limits(self):
        """타입·컬럼·전체 결측·출력 크기 위반을 조용히 보정하지 않는다."""
        pipeline = build_preparation(self.roles, PreparationRules()).fit(self.train)

        for frame in (self.train.drop(columns="value"), self.train.assign(value=np.inf), self.train.assign(value="text")):
            with self.assertRaises(ValueError):
                pipeline.transform(frame)

        with self.assertRaises(ValueError):
            build_preparation(self.roles, PreparationRules()).fit(self.train.assign(value=np.nan))

        with self.assertRaises(ValueError):
            build_preparation(self.roles, PreparationRules(max_categories=1)).fit(self.train)

        with self.assertRaises(ValueError):
            build_preparation(self.roles, PreparationRules(max_output_features=1)).fit(self.train)

        small = build_preparation(self.roles, PreparationRules(max_output_cells=1)).fit(self.train)

        with self.assertRaises(ValueError):
            small.transform(self.train)

        for changes in ({"scaling": "bad"}, {"binning": "uniform", "scaling": "standard"}, {"n_bins": True}, {"missing_ratio_threshold": float("nan")}):
            with self.assertRaises(ValueError):
                PreparationRules(**changes)

    def test_pipeline_clone_and_fold_independence(self):
        """교차검증용 clone이 서로 다른 Fold 통계를 독립적으로 학습한다."""
        pipeline = build_preparation(self.roles, PreparationRules())
        first = clone(pipeline).fit(self.train)
        second = clone(pipeline).fit(self.train.assign(value=[10.0, 20.0, 30.0, 40.0]))

        self.assertEqual(first.named_steps["preprocess"].state_["records"][0]["impute"], 2.0)
        self.assertEqual(second.named_steps["preprocess"].state_["records"][0]["impute"], 25.0)
        self.assertFalse(hasattr(pipeline.named_steps["preprocess"], "state_"))

    def test_json_roundtrip_without_refit(self):
        """저장·복원에는 pickle과 fit이 없고 열 순서·배치가 달라도 결과가 일치한다."""
        pipeline = build_preparation(self.roles, PreparationRules(scaling="robust")).fit(self.train)
        path = self.root / "preprocessor.json"
        save_preparation(pipeline, path)

        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("다시 학습하면 안 됩니다.")):
            restored = load_preparation(path)
            actual = restored.transform(self.train.loc[:, ["kind", "value"]])
            pd.testing.assert_frame_equal(actual, pipeline.transform(self.train))
            pd.testing.assert_frame_equal(restored.transform(self.train.iloc[[0]]), actual.iloc[[0]])

        with self.assertRaises(FileExistsError):
            save_preparation(pipeline, path)

    def test_invalid_saved_contract(self):
        """무한 통계·잘못된 출력 폭·컬럼 순서·임의 필드는 복원하지 않는다."""
        path = self.root / "original.json"
        save_preparation(build_preparation(self.roles, PreparationRules()).fit(self.train), path)
        original = json.loads(path.read_text(encoding="utf-8"))

        for mode in ("scale", "width", "columns", "field"):
            payload = json.loads(json.dumps(original))

            if mode == "scale":
                payload["state"]["records"][0]["scale"] = 0
            elif mode == "width":
                payload["state"]["records"][0]["width"] = 100
            elif mode == "columns":
                payload["output_columns"].reverse()
            else:
                payload["module"] = "unknown"

            bad = self.root / f"bad_{mode}.json"
            bad.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_preparation(bad)

    def test_roundtrip_other_branches_and_categorical_only(self):
        """범주화·clip·NaN 보존과 범주형 전용 경로도 JSON 복원 결과가 일치한다."""
        cases = (
            (self.roles, PreparationRules(binning="quantile", bin_encoding="onehot"), self.train),
            (self.roles, PreparationRules(outlier="iqr_clip", categorical_imputation="most_frequent"), self.train),
            (ColumnRoles(numeric=("value",)), PreparationRules(numeric_imputation="none"), self.train.assign(value=np.nan)),
            (ColumnRoles(categorical=("kind",)), PreparationRules(), self.train),
        )

        for index, (roles, rules, frame) in enumerate(cases):
            with self.subTest(index=index):
                pipeline = build_preparation(roles, rules).fit(frame)
                path = self.root / f"branch_{index}.json"
                save_preparation(pipeline, path)
                restored = load_preparation(path)
                pd.testing.assert_frame_equal(pipeline.transform(frame), restored.transform(frame))

    def test_prepare_split_fits_train_only_and_skips_test(self):
        """Test의 극단값은 대치값에 영향을 주지 않으며 Test 출력도 만들지 않는다."""
        roles = ColumnRoles(numeric=("value",), timestamp="time")
        frame = pd.DataFrame({"value": [1.0] * 7 + [100.0] + [1_000_000.0] * 2, "time": pd.date_range("2026-01-01", periods=10)})
        data = TabularInput(frame, roles)
        split_rules = SplitRules(strategy="time", train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2)
        proposal = propose_split(data, split_rules, source_id="synthetic", config_id="config")
        pipeline, outputs = prepare_splits(data, proposal, split_rules, PreparationRules(), source_id="synthetic", config_id="config")

        self.assertEqual(pipeline.named_steps["preprocess"].state_["train_rows"], 7)
        self.assertEqual(pipeline.named_steps["preprocess"].state_["records"][0]["impute"], 1.0)
        self.assertEqual(set(outputs), {"train", "validation"})

    def test_profile_rules_roundtrip_and_binding(self):
        """전처리 설정도 프로파일·계획 식별값에 포함한다."""
        profile = DatasetProfile(
            "synthetic", "tabular_csv_v1", "prepare_review_v1", self.roles,
            split_rules=SplitRules(stratify=True), preparation_rules=PreparationRules(),
        )
        self.assertEqual(profile_from_dict(profile.to_dict()), profile)
        self.assertNotEqual(replace(profile, preparation_rules=PreparationRules(scaling="standard")).signature, profile.signature)

        with self.assertRaises(ValueError):
            replace(profile, preparation_rules=None)


class PreparationExecutorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "synthetic.csv"
        pd.DataFrame({"value": range(120), "kind": ["A", "B"] * 60, "label": [0, 1] * 60}).to_csv(self.source, index=False)
        self.profile = DatasetProfile(
            "synthetic", "tabular_csv_v1", "prepare_review_v1",
            ColumnRoles(numeric=("value",), categorical=("kind",), target="label"),
            split_rules=SplitRules(stratify=True), preparation_rules=PreparationRules(scaling="standard"),
        )
        self.adapter = csv_eda_adapter(self.source, self.profile)
        self.plan = build_profile_plan(self.profile, self.adapter.source_id)
        self.states = LocalStateStore(self.root / "run")
        self.states.create("synthetic_actor", policy_version=self.plan.policy_version)
        self.approvals = ApprovalStore(self.root / "approval.sqlite3")

        def synthetic_check(output, arguments):
            (output / "checked.txt").write_text("synthetic", encoding="utf-8")

            return ToolResult(status="passed")

        tools = [*self.adapter.split_tools(), self.adapter.preparation_tool()]

        for step in self.plan.steps:
            if step.stage not in {"propose_split", "commit_split", "prepare_data"}:
                tools.append(RegisteredTool(step.action_id, synthetic_check, lambda output, arguments: (output / "checked.txt").is_file()))

        self.executor = RulesExecutor(self.states, self.approvals, self.plan, tools)

    def ready(self):
        """준비 단계 이전까지만 합성 승인을 사용해 진행한다."""
        for step in self.plan.steps[:-1]:
            result = self.executor.run_next()

            if result.code == "APPROVAL_REQUIRED":
                approval = self.approvals.record_confirmed_approval(self.executor.scope_for(step.stage))
                result = self.executor.run_next(approval_id=approval.approval_id)

            self.assertEqual(result.code, "STEP_COMPLETED")

    def test_approval_preparation_and_saved_contract(self):
        """확정 분할과 추가 승인을 거친 후 입력 특징·행 계약을 검증한다."""
        self.ready()
        self.assertEqual(self.executor.run_next().code, "APPROVAL_REQUIRED")
        approval = self.approvals.record_confirmed_approval(self.executor.scope_for("prepare_data"))
        result = self.executor.run_next(approval_id=approval.approval_id)
        self.assertEqual(result.code, "STEP_COMPLETED")
        self.assertEqual(result.status, "completed")
        state = self.states.load()
        output = self.states.directory / state.artifact_refs["prepare_data"]
        report = json.loads((output / "preparation.json").read_text())

        self.assertEqual(report["train_rows"], 84)
        self.assertFalse(report["test_transformed"])
        self.assertFalse((output / "test_features.csv").exists())
        contract = json.loads((output / "row_contract.json").read_text())
        self.assertEqual(len(contract["train"]["targets"]), 84)
        self.assertEqual(len(contract["validation"]["row_positions"]), 18)
        commit = self.states.directory / state.artifact_refs["commit_split"]

        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("재학습 금지")):
            self.assertTrue(self.adapter.preparation_tool().validate(output, self.adapter.arguments, commit))

        (output / "validation_features.csv").write_text("changed", encoding="utf-8")
        self.assertFalse(self.adapter.preparation_tool().validate(output, self.adapter.arguments, commit))

    def test_source_change_blocks_preparation(self):
        """승인 후 원본이 바뀌면 새 원본으로 전처리를 시작하지 않는다."""
        self.ready()
        approval = self.approvals.record_confirmed_approval(self.executor.scope_for("prepare_data"))
        self.source.write_text("changed", encoding="utf-8")
        result = self.executor.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.code, "TOOL_RUNTIME_FAILED")
        self.assertNotIn("prepare_data", self.states.load().artifact_refs)

    def test_unknown_validation_category_blocks_without_completion(self):
        """새 범주 오류를 입력 차단으로 기록하며 ignore로 자동 변경하지 않는다."""
        frame = pd.read_csv(self.source)
        frame["kind"] = [f"kind_{index}" for index in range(len(frame))]
        frame.to_csv(self.source, index=False)
        adapter = csv_eda_adapter(self.source, self.profile)
        proposal = propose_split(
            adapter.load_input(), self.profile.split_rules,
            source_id=adapter.source_id, config_id=adapter.config_id,
        )
        dependency = self.root / "synthetic_commit"
        dependency.mkdir()
        (dependency / "split.json").write_text(json.dumps(proposal), encoding="utf-8")
        output = self.root / "blocked_preparation"
        output.mkdir()
        result = adapter.preparation_tool().execute(output, adapter.arguments, dependency)

        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())
        self.assertFalse((output / "preprocessor.json").exists())
        self.assertEqual(self.profile.preparation_rules.unknown_category, "error")


if __name__ == "__main__":
    unittest.main()

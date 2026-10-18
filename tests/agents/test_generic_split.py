"""실제 데이터 없이 공통 분할 경계와 승인 대상의 정합성을 검증한다."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.agents.adapters.tabular import csv_eda_adapter
from src.agents.approval_store import ApprovalStore
from src.agents.artifacts import complete_artifacts
from src.agents.contracts import ToolResult
from src.agents.executor import RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.agents.workflow import build_profile_plan
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.profiles import DatasetProfile, profile_from_dict
from src.preprocessing.split import SplitRules, propose_split, validate_proposal


class GenericSplitTest(unittest.TestCase):
    def setUp(self):
        # 위치와 index를 일부러 다르게 두어 위치 계약을 확인한다.
        self.frame = pd.DataFrame({
            "value": range(120), "label": ["normal", "fail"] * 60,
            "id": range(120), "lot": [f"lot_{index // 4}" for index in range(120)],
            "time": pd.date_range("2026-01-01", periods=60).repeat(2),
        }, index=[0] * 120)
        self.roles = ColumnRoles(numeric=("value",), target="label", identifiers=("id",))

    def propose(self, rules, roles=None, frame=None):
        return propose_split(
            TabularInput(self.frame if frame is None else frame, roles or self.roles),
            rules, source_id="synthetic_source", config_id="synthetic_config",
        )

    def test_random_stratification_and_no_mutation(self):
        """다중 역할 입력을 변경하지 않고 클래스 비율과 전체 행을 보존한다."""
        before = self.frame.copy(deep=True)
        rules = SplitRules(stratify=True)
        result = self.propose(rules)

        self.assertEqual(result, self.propose(rules))
        self.assertEqual(result["status"], "passed")
        self.assertEqual([row["rows"] for row in result["summary"]], [84, 18, 18])

        for row in result["summary"]:
            self.assertEqual(row["classes"][0]["count"], row["classes"][1]["count"])

        pd.testing.assert_frame_equal(before, self.frame)

    def test_group_disjointness(self):
        """같은 그룹이 Train·Validation·Test에 걸쳐 나뉘지 않는다."""
        roles = replace(self.roles, group="lot")
        result = self.propose(SplitRules(strategy="group"), roles)
        sets = [set(self.frame.iloc[positions]["lot"]) for positions in result["partitions"].values()]

        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

        with self.assertRaises(ValueError):
            self.propose(SplitRules(), roles)

    def test_time_ties_and_group_boundaries(self):
        """동일 시각을 나누지 않으며 시간 순서와 함께 선언한 그룹 경계도 검사한다."""
        roles = replace(self.roles, timestamp="time")
        rules = SplitRules(strategy="time")
        result = self.propose(rules, roles)
        bounds = [(self.frame.iloc[positions].time.min(), self.frame.iloc[positions].time.max()) for positions in result["partitions"].values()]

        self.assertLess(bounds[0][1], bounds[1][0])
        self.assertLess(bounds[1][1], bounds[2][0])

        bad = self.frame.copy()
        bad["lot"] = "crossing_lot"

        with self.assertRaisesRegex(ValueError, "그룹"):
            self.propose(rules, replace(roles, group="lot"), bad)

    def test_explicit_timestamp_format(self):
        """문자열 시간의 모호한 자동 파싱을 허용하지 않는다."""
        frame = self.frame.copy()
        frame["time"] = frame.time.dt.strftime("%Y-%m-%d")
        roles = replace(self.roles, timestamp="time")

        with self.assertRaises(ValueError):
            self.propose(SplitRules(strategy="time"), roles, frame)

        self.assertEqual(self.propose(SplitRules(strategy="time", timestamp_format="%Y-%m-%d"), roles, frame)["status"], "passed")

    def test_invalid_configuration_and_infeasible_data(self):
        """잘못된 비율·계층 분할·시드·빈 구간은 자동 우회하지 않는다."""
        for changes in (
            {"train_ratio": 0.9}, {"strategy": "time", "stratify": True},
            {"random_state": True}, {"test_ratio": float("nan")}, {"strategy": "unknown"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SplitRules(**changes)

        with self.assertRaises(ValueError):
            self.propose(SplitRules(stratify=True), frame=self.frame.iloc[:3])

        frame = self.frame.copy()
        frame["time"] = pd.Timestamp("2026-01-01")

        with self.assertRaises(ValueError):
            self.propose(SplitRules(strategy="time"), replace(self.roles, timestamp="time"), frame)

    def test_missing_labels_and_duplicate_identifiers(self):
        """결측 label·중복 ID를 임의 보정하거나 제거하지 않는다."""
        frame = self.frame.copy()
        frame.iloc[0, frame.columns.get_loc("label")] = None

        with self.assertRaises(ValueError):
            self.propose(SplitRules(), frame=frame)

        frame = self.frame.copy()
        frame["id"] = 1

        with self.assertRaises(ValueError):
            self.propose(SplitRules(), frame=frame)

    def test_missing_class_requires_review(self):
        """필수 클래스가 한 구간에 없으면 검토 상태이며 저장 검증을 통과하지 못한다."""
        frame = self.frame.copy()
        frame["label"] = ["fail"] * 5 + ["normal"] * 115
        roles = replace(self.roles, timestamp="time")
        rules = SplitRules(strategy="time")
        proposal = self.propose(rules, roles, frame)

        self.assertEqual(proposal["status"], "review_required")

        with self.assertRaises(ValueError):
            validate_proposal(proposal, TabularInput(frame, roles), rules, source_id="synthetic_source", config_id="synthetic_config")

    def test_modified_proposal_and_source_are_rejected(self):
        """위치·분포·원본·설정 변경이 있으면 동일 승인 대상으로 취급하지 않는다."""
        rules = SplitRules()
        proposal = self.propose(rules)
        data = TabularInput(self.frame, self.roles)
        self.assertTrue(validate_proposal(proposal, data, rules, source_id="synthetic_source", config_id="synthetic_config"))

        for key, value in (("source_id", "changed"), ("config_id", "changed"), ("proposal_id", "changed")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_proposal({**proposal, key: value}, data, rules, source_id="synthetic_source", config_id="synthetic_config")

    def test_profile_roundtrip_and_signature(self):
        """분할 설정을 프로파일 서명에 포함하고 기존 EDA 프로파일은 유지한다."""
        profile = DatasetProfile("synthetic", "tabular_csv_v1", "split_review_v1", self.roles, split_rules=SplitRules(stratify=True))
        self.assertEqual(profile_from_dict(profile.to_dict()), profile)
        self.assertNotEqual(replace(profile, split_rules=SplitRules(random_state=43)).signature, profile.signature)

        with self.assertRaises(ValueError):
            replace(profile, split_rules=None)


class SplitExecutorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "synthetic.csv"
        pd.DataFrame({"value": range(120), "label": [0, 1] * 60}).to_csv(self.source, index=False)
        self.profile = DatasetProfile(
            "synthetic", "tabular_csv_v1", "split_review_v1",
            ColumnRoles(numeric=("value",), target="label"), split_rules=SplitRules(stratify=True),
        )
        self.adapter = csv_eda_adapter(self.source, self.profile)
        self.plan = build_profile_plan(self.profile, self.adapter.source_id)
        self.states = LocalStateStore(self.root / "run")
        self.states.create("synthetic_actor", policy_version=self.plan.policy_version)
        self.approvals = ApprovalStore(self.root / "approvals.sqlite3")

        def synthetic_check(output, arguments):
            # EDA·검토 내용은 기존 전용 테스트에서 다룬다. 여기서는 분할 연결만 확인한다.
            (output / "checked.txt").write_text("synthetic", encoding="utf-8")

            return ToolResult(status="passed")

        tools = list(self.adapter.split_tools())

        for step in self.plan.steps:
            if step.stage not in {"propose_split", "commit_split"}:
                tools.append(RegisteredTool(step.action_id, synthetic_check, lambda output, arguments: (output / "checked.txt").is_file()))

        self.executor = RulesExecutor(self.states, self.approvals, self.plan, tools)

    def preview(self):
        """실제 사용자의 승인 UI를 대신하는 합성 승인만 발급한다."""
        for step in self.plan.steps[:-1]:
            result = self.executor.run_next()

            if result.code == "APPROVAL_REQUIRED":
                record = self.approvals.record_confirmed_approval(self.executor.scope_for(step.stage))
                result = self.executor.run_next(approval_id=record.approval_id)

            self.assertEqual(result.code, "STEP_COMPLETED")

    def test_preview_approval_commit_and_resume(self):
        """제안만으로 확정하지 않고 승인 뒤 정확히 같은 위치 목록을 저장한다."""
        with self.assertRaises(ValueError):
            self.executor.scope_for("commit_split")

        self.preview()
        self.assertEqual(self.executor.run_next().code, "APPROVAL_REQUIRED")
        scope = self.executor.scope_for("commit_split")
        record = self.approvals.record_confirmed_approval(scope)
        result = self.executor.run_next(approval_id=record.approval_id)
        self.assertEqual(result.status, "completed")
        state = self.states.load()
        preview = self.states.directory / state.artifact_refs["propose_split"]
        committed = self.states.directory / state.artifact_refs["commit_split"]
        self.assertEqual(json.loads((preview / "proposal.json").read_text()), json.loads((committed / "split.json").read_text()))
        assignment = pd.read_csv(committed / "assignments.csv")
        self.assertEqual(len(assignment), 120)
        self.assertEqual(assignment.row_position.nunique(), 120)
        self.assertEqual(self.executor.run_next().code, "RUN_NOT_EXECUTABLE")

    def test_modified_preview_is_blocked_before_commit(self):
        """승인 발급 뒤 분할 제안 파일을 바꾸면 저장 전에 중단한다."""
        self.preview()
        record = self.approvals.record_confirmed_approval(self.executor.scope_for("commit_split"))
        state = self.states.load()
        path = self.states.directory / state.artifact_refs["propose_split"] / "proposal.json"
        path.write_text("{}", encoding="utf-8")
        result = self.executor.run_next(approval_id=record.approval_id)

        self.assertEqual(result.code, "PREREQUISITE_ARTIFACT_INVALID")
        self.assertNotIn("commit_split", self.states.load().artifact_refs)

    def test_modified_source_is_blocked(self):
        """승인 후 CSV가 바뀌면 새 원본으로 분할을 덮어 저장하지 않는다."""
        self.preview()
        record = self.approvals.record_confirmed_approval(self.executor.scope_for("commit_split"))
        self.source.write_text("changed", encoding="utf-8")
        result = self.executor.run_next(approval_id=record.approval_id)

        self.assertEqual(result.code, "TOOL_RUNTIME_FAILED")
        self.assertNotIn("commit_split", self.states.load().artifact_refs)

    def test_changed_completion_invalidates_existing_approval(self):
        """파일 정합성 표식이 새로 생성되어도 이전 제안 승인을 재사용하지 못한다."""
        self.preview()
        record = self.approvals.record_confirmed_approval(self.executor.scope_for("commit_split"))
        state = self.states.load()
        output = self.states.directory / state.artifact_refs["propose_split"]
        (output / "proposal.md").write_text("changed synthetic report", encoding="utf-8")
        (output / "AGENT_COMPLETE.json").unlink()
        step = next(step for step in self.plan.steps if step.stage == "propose_split")
        complete_artifacts(output, self.executor._context(step))
        result = self.executor.run_next(approval_id=record.approval_id)

        self.assertEqual(result.code, "APPROVAL_SCOPE_MISMATCH")
        self.assertNotIn("commit_split", self.states.load().artifact_refs)

    def test_infeasible_proposal_returns_input_block(self):
        """표본 부족은 고정 진단 파일을 남기고 완료 표식 없이 입력 차단한다."""
        self.source.write_text("value,label\n1,0\n2,1\n3,0\n", encoding="utf-8")
        adapter = csv_eda_adapter(self.source, self.profile)
        output = self.root / "blocked_proposal"
        output.mkdir()
        tool = adapter.split_tools()[0]
        result = tool.execute(output, adapter.arguments)

        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())
        self.assertEqual(json.loads((output / "split_check.json").read_text())["code"], "SPLIT_CONSTRAINTS_UNMET")


if __name__ == "__main__":
    unittest.main()

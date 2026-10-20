"""승인·커밋 증거와 구간 복사본 결합을 합성 산출물로 검사한다. 내보내기 worker 테스트는 아니다."""

from contextlib import closing
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents.approval_store import ApprovalStore
from src.agents.approvals import ApprovalScope
from src.agents.artifacts import ArtifactContext, complete_artifacts, _fingerprint
from src.agents.input_snapshot import create_snapshot
from src.agents.partition_input_contract import bind_partition_input
from src.agents.split_evidence import read_confirmed_split, read_committed_artifact
from src.agents.state import LocalStateStore
from src.agents.adapters.tabular import csv_eda_adapter
from src.agents.contracts import ToolResult
from src.agents.executor import RegisteredTool, RulesExecutor
from src.agents.workflow import build_profile_plan
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.split import SplitRules, propose_split, _digest


class PartitionInputContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.frame = pd.DataFrame({"x": range(60), "label": [0, 1] * 30, "id": range(60)})
        self.roles = ColumnRoles(numeric=("x",), target="label", identifiers=("id",))
        self.proposal = propose_split(
            TabularInput(self.frame, self.roles), SplitRules(stratify=True),
            source_id="synthetic_source", config_id="synthetic_config",
        )
        self.split_run, self.split_scope, self.split_output = self.fixture_run("split_run", "commit_split", "commit_split", "synthetic_config")
        (self.split_output / "split.json").write_text(json.dumps(self.proposal), encoding="utf-8")
        assignments = [{"row_position": position, "split": name} for name, positions in self.proposal["partitions"].items() for position in positions]
        pd.DataFrame(assignments).sort_values("row_position").to_csv(self.split_output / "assignments.csv", index=False)
        self.mark(self.split_output, self.split_scope)
        self.split = read_confirmed_split(self.split_run, self.split_scope)

    def fixture_run(self, name, stage, tool_id, config_id):
        """검토된 생산자의 커밋 이력을 합성한다. 실제 승인 UI·worker 실행을 대신하지 않는다."""
        run = self.root / name
        states = LocalStateStore(run)
        state = states.create("synthetic_actor")
        state.status = "completed"
        state.current_stage = stage
        states.save(state, expected_revision=state.revision)
        scope = ApprovalScope(
            state.actor_id, state.run_id, stage, tool_id, f"{state.run_id}:{stage}",
            "synthetic_arguments", "synthetic_source", config_id, "synthetic_plan", "preprocess_policy_v1",
        )
        store = ApprovalStore(run / "approvals.sqlite3")
        approval = store.record_confirmed_approval(scope)
        task_id = store.consume_and_reserve(approval.approval_id, scope).reservation_id

        for before, after in (("reserved", "running"), ("running", "verified"), ("verified", "committed")):
            store.transition_task(scope, task_id, expected=before, status=after)

        output = run / "attempts" / task_id
        output.mkdir(parents=True)

        return run, scope, output

    def mark(self, directory, scope):
        """테스트 데이터의 완료 표식을 구성한다. 실제 사용자 산출물에는 사용하지 않는다."""
        (directory / "AGENT_COMPLETE.json").unlink(missing_ok=True)
        complete_artifacts(directory, ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id))

    def fixture_export(self):
        """후속 내보내기 worker가 제공해야 할 파일 계약의 합성 fixture."""
        run, scope, output = self.fixture_run("export_run", "ingest", "run_ingest", "export_config")
        receipt = {
            "schema_version": "partition_exports_v1", "split_evidence_id": self.split.signature,
            "source_id": self.split_scope.source_id, "proposal_id": self.proposal["proposal_id"],
            "roles": json.loads(json.dumps(asdict(self.roles))), "partitions": {},
        }

        for name in ("train", "validation"):
            positions = self.proposal["partitions"][name]
            source = self.root / f"{name}.csv"
            self.frame.iloc[positions].to_csv(source, index=False)
            snapshot = create_snapshot(source, output / name, max_input_bytes=4096)
            receipt["partitions"][name] = {
                "row_count": len(positions), "row_positions_sha256": _digest(positions), "snapshot": asdict(snapshot),
            }

        (output / "partition_exports.json").write_text(json.dumps(receipt), encoding="utf-8")
        self.mark(output, scope)
        return run, scope, output, receipt

    def test_confirmed_split_and_readonly_database(self):
        """승인·커밋·분할 메타데이터를 확인하며 DB와 원본 CSV를 변경·파싱하지 않는다."""
        database = self.split_run / "approvals.sqlite3"
        before = _fingerprint(database)

        with patch("pandas.read_csv", side_effect=AssertionError("부모 CSV 파싱")):
            split = read_confirmed_split(self.split_run, self.split_scope)
            self.assertTrue(split.check())

        self.assertEqual(_fingerprint(database), before)
        self.assertEqual(split.positions("train"), tuple(self.proposal["partitions"]["train"]))

    def test_train_validation_binding_and_no_test(self):
        """두 구간은 서로 다른 계약으로 연결하며 Test를 자동 전달하지 않는다."""
        run, scope, output, _ = self.fixture_export()
        export = read_committed_artifact(run, scope)

        with patch("pandas.read_csv", side_effect=AssertionError("부모 CSV 파싱")):
            train = bind_partition_input(self.split, export, "train", self.roles)
            validation = bind_partition_input(self.split, export, "validation", self.roles)
            self.assertTrue(train.check())

        self.assertNotEqual(train.contract_id, validation.contract_id)
        self.assertEqual(train.roles, self.roles)
        self.assertFalse(set(train.row_positions) & set(validation.row_positions))
        self.assertFalse((output / "test").exists())

        with self.assertRaises(ValueError):
            bind_partition_input(self.split, export, "test", self.roles)

    def test_scope_and_commit_requirements(self):
        """미커밋·미소비 승인·잘못된 기대 범위는 증거로 인정하지 않는다."""
        with self.assertRaises(ValueError):
            read_confirmed_split(self.split_run, replace(self.split_scope, config_id="different"))

        for sql in ("UPDATE task_runs SET status='failed'", "UPDATE approvals SET status='revoked'"):
            with closing(sqlite3.connect(self.split_run / "approvals.sqlite3")) as connection, connection:
                connection.execute(sql)

            with self.assertRaises(ValueError):
                read_confirmed_split(self.split_run, self.split_scope)

            with closing(sqlite3.connect(self.split_run / "approvals.sqlite3")) as connection, connection:
                connection.execute("UPDATE task_runs SET status='committed'")
                connection.execute("UPDATE approvals SET status='consumed'")

    def test_tampered_split_and_assignments(self):
        """파일 변조·배정표 불일치·위치 중복을 거부한다."""
        path = self.split_output / "assignments.csv"
        path.write_text("row_position,split\n0,train\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.split.check()

        self.mark(self.split_output, self.split_scope)

        with self.assertRaises(ValueError):
            read_confirmed_split(self.split_run, self.split_scope)

    def test_uncommitted_csv_cannot_be_bound(self):
        """임의 CSV·분할 확정 폴더만으로 내보내기 출처를 선언할 수 없다."""
        with self.assertRaises(ValueError):
            bind_partition_input(self.split, self.split.artifact, "train", self.roles)

        run, scope, output = self.fixture_run("no_receipt", "ingest", "run_ingest", "export_config")
        (output / "train.csv").write_text("x\n1\n", encoding="utf-8")
        self.mark(output, scope)

        with self.assertRaises(ValueError):
            bind_partition_input(self.split, read_committed_artifact(run, scope), "train", self.roles)

    def test_export_metadata_mismatch(self):
        """완료 표식이 있어도 잘못된 분할·행 순서·역할 연결은 거부한다."""
        run, scope, output, receipt = self.fixture_export()
        original = json.dumps(receipt)

        for change in ("split", "rows", "roles", "extra"):
            value = json.loads(original)

            if change == "split":
                value["split_evidence_id"] = "different"
            elif change == "rows":
                value["partitions"]["train"]["row_positions_sha256"] = "0" * 64
            elif change == "roles":
                value["roles"]["numeric"] = ["other"]
            else:
                value["command"] = "run"

            (output / "partition_exports.json").write_text(json.dumps(value), encoding="utf-8")
            self.mark(output, scope)

            with self.subTest(change=change), self.assertRaises(ValueError):
                bind_partition_input(self.split, read_committed_artifact(run, scope), "train", self.roles)

    def test_changed_snapshot_invalidates_existing_contract(self):
        """계약 생성 이후 복사본이 바뀌면 다시 사용하지 않는다."""
        run, scope, output, _ = self.fixture_export()
        contract = bind_partition_input(self.split, read_committed_artifact(run, scope), "train", self.roles)
        (output / "train/input.bin").write_bytes(b"changed")

        with self.assertRaises(ValueError):
            contract.check()

    def test_cancelled_source_is_not_used(self):
        """취소 또는 보안 잠금 상태의 원본 실행에서 자동으로 이어 가지 않는다."""
        states = LocalStateStore(self.split_run)
        state = states.load()
        state.cancel_requested = True
        states.save(state, expected_revision=state.revision)

        with self.assertRaises(ValueError):
            self.split.check()

    def test_existing_split_executor_output_is_accepted(self):
        """기존 분할 코어·승인·커밋·배정표 형식을 변경하지 않고 읽을 수 있다."""
        source = self.root / "existing.csv"
        self.frame.to_csv(source, index=False)
        profile = DatasetProfile(
            "existing_example", "tabular_csv_v1", "split_review_v1", self.roles,
            split_rules=SplitRules(stratify=True),
        )
        adapter = csv_eda_adapter(source, profile)
        plan = build_profile_plan(profile, adapter.source_id)
        states = LocalStateStore(self.root / "existing_run")
        states.create("synthetic_actor", policy_version=plan.policy_version)
        approvals = ApprovalStore(states.directory / "approvals.sqlite3")

        def synthetic_check(output, arguments):
            (output / "checked.txt").write_text("synthetic", encoding="utf-8")
            return ToolResult("passed")

        tools = list(adapter.split_tools())

        for step in plan.steps:
            if step.stage not in ("propose_split", "commit_split"):
                tools.append(RegisteredTool(
                    step.action_id, synthetic_check,
                    lambda output, arguments: (output / "checked.txt").is_file(),
                ))

        manager = RulesExecutor(states, approvals, plan, tools)

        for step in plan.steps:
            result = manager.run_next()

            if result.code == "APPROVAL_REQUIRED":
                approval = approvals.record_confirmed_approval(manager.scope_for(step.stage))
                result = manager.run_next(approval_id=approval.approval_id)

            self.assertEqual(result.code, "STEP_COMPLETED")

        evidence = read_confirmed_split(states.directory, manager.scope_for("commit_split"))
        self.assertEqual(evidence.proposal["source_id"], adapter.source_id)
        self.assertTrue(evidence.check())


if __name__ == "__main__":
    unittest.main()

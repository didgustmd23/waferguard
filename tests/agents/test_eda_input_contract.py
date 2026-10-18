"""합성 파일로 EDA 입력 계약과 승인 범위의 변경 결합을 검증한다."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.agents.eda_input_contract import EdaInputContract, bind_eda_input
from src.agents.input_snapshot import PAYLOAD_NAME, create_snapshot
from src.agents.worker_contract import WorkerLimits
from src.agents.approval_store import ApprovalStore
from src.agents.executor import RegisteredTool, RulesExecutor
from src.agents.state import LocalStateStore
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile


class EdaInputContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "source.csv"
        source.write_bytes(b"value,target\n1,0\n2,1\n")
        self.directory = self.root / "snapshot"
        self.snapshot = create_snapshot(source, self.directory, max_input_bytes=100)
        self.profile = DatasetProfile(
            "example", "tabular_csv_v1", "eda_review_v1",
            ColumnRoles(numeric=("value",), target="target"),
        )
        self.limits = WorkerLimits(60, 2048, 2)
        self.contract = bind_eda_input(self.directory, self.snapshot, self.profile, self.limits)

    def test_roundtrip_and_plan_binding(self):
        """복원 후 입력·역할·그래프 설정·한도·계획 식별값이 유지된다."""
        restored = EdaInputContract.from_json(self.contract.to_json())
        self.assertEqual(restored, self.contract)
        self.assertTrue(restored.check_input())
        plan = restored.execution_plan()
        self.assertEqual(plan.source_id, self.snapshot.source_id)
        self.assertEqual(plan.config_id, restored.contract_id)
        self.assertEqual(json.loads(plan.steps[1].arguments_json), json.loads(restored.to_json()))

    def test_profile_and_limits_change_approval_scope(self):
        """역할·설정·자원 변경은 기존 승인과 다른 범위를 만든다."""
        states = LocalStateStore(self.root / "state")
        states.create("synthetic_operator")
        approvals = ApprovalStore(self.root / "approvals.sqlite3")

        def scope(contract):
            # scope 생성만 검사하며 도구 함수는 실행하지 않는다.
            tools = tuple(
                RegisteredTool(action, lambda *_: None, lambda *_: False, requires_approval=True)
                for action in ("run_preflight", "run_eda")
            )
            return RulesExecutor(states, approvals, contract.execution_plan(), tools).scope_for("run_eda")

        original = scope(self.contract)

        for profile in (
            replace(self.profile, max_plot_features=2),
            replace(self.profile, roles=ColumnRoles(numeric=("value",))),
        ):
            changed = bind_eda_input(self.directory, self.snapshot, profile, self.limits)
            self.assertNotEqual(scope(changed), original)

        changed = bind_eda_input(self.directory, self.snapshot, self.profile, WorkerLimits(61, 2048, 2))
        self.assertNotEqual(scope(changed), original)

    def test_tamper_and_unknown_adapter_rejected(self):
        """복사본 변경과 미등록 형식의 분석 요청을 거부한다."""
        with self.assertRaises(ValueError):
            bind_eda_input(self.directory, self.snapshot, replace(self.profile, adapter_id="pickle_v1"), self.limits)

        (self.directory / PAYLOAD_NAME).write_bytes(b"changed")

        with self.assertRaises(ValueError):
            self.contract.execution_plan()

    def test_strict_json_and_manifest_not_authority(self):
        """추가 필드·중복 키·잘못된 버전·manifest 변경을 거부한다."""
        payload = self.contract.to_dict()
        payload["command"] = "not_allowed"

        with self.assertRaises(ValueError):
            EdaInputContract.from_json(json.dumps(payload))

        with self.assertRaises(ValueError):
            EdaInputContract.from_json('{"profile":{},"profile":{}}')

        with self.assertRaises(ValueError):
            replace(self.contract, schema_version="unknown")

        marker = self.directory / "snapshot.json"
        metadata = json.loads(marker.read_text())
        metadata["max_input_bytes"] = 200
        marker.write_text(json.dumps(metadata))

        with self.assertRaises(ValueError):
            self.contract.check_input()


if __name__ == "__main__":
    unittest.main()

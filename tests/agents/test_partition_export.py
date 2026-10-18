"""구간 추출·부모 파일 검증을 합성 데이터로 확인한다. 실제 OS 격리·실데이터는 사용하지 않는다."""

import csv
from dataclasses import asdict, replace
import hashlib
import json
from io import StringIO
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents.approval_store import ApprovalStore
from src.agents.approvals import ApprovalScope
from src.agents.artifacts import ArtifactContext, complete_artifacts
from src.agents.eda_input_contract import bind_eda_input
from src.agents.input_snapshot import create_snapshot
from src.agents.partition_export_contract import PartitionExportContract, POLICY_VERSION
from src.agents.partition_export_results import read_partition_export
from src.agents.partition_export_worker import execute_request
from src.agents.partition_csv import iter_csv_rows, MAX_RECORD_BYTES, MAX_COLUMNS
from src.agents.partition_input_contract import bind_partition_input
from src.agents.split_evidence import read_confirmed_split, read_committed_artifact
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits, WorkerRequest
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.split import SplitRules, propose_split, _digest


class PartitionExportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.csv"
        self.rows = [[f"{index}.00", f"{index:04d}", f'한글,{index}\n"인용"', str(index % 2)] for index in range(60)]

        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["x", "id", "text", "label"])
            writer.writerows(self.rows)

        self.roles = ColumnRoles(numeric=("x",), categorical=("text",), target="label", identifiers=("id",))
        self.profile = DatasetProfile("export_example", "tabular_csv_v1", "split_review_v1", self.roles, split_rules=SplitRules(stratify=True))
        frame = pd.DataFrame({"x": range(60), "id": [row[1] for row in self.rows], "text": [row[2] for row in self.rows], "label": [index % 2 for index in range(60)]})
        source_id = "csv:" + hashlib.sha256(self.source.read_bytes()).hexdigest()
        proposal = propose_split(TabularInput(frame, self.roles), self.profile.split_rules, source_id=source_id, config_id=self.profile.signature)
        run, scope, output = self.fixture_commit("split_run", "commit_split", "commit_split", source_id, self.profile.signature)
        (output / "split.json").write_text(json.dumps(proposal), encoding="utf-8")
        pd.DataFrame([{"row_position": position, "split": name} for name, positions in proposal["partitions"].items() for position in positions]).sort_values("row_position").to_csv(output / "assignments.csv", index=False)
        self.complete(output, scope)
        self.split = read_confirmed_split(run, scope)
        snapshot = create_snapshot(self.source, self.root / "snapshot", max_input_bytes=65536)
        source_contract = bind_eda_input(self.root / "snapshot", snapshot, self.profile, WorkerLimits(60, 2048, 2))
        self.contract = PartitionExportContract(self.split, source_contract, 65536)

    def fixture_commit(self, name, stage, tool, source, config):
        """신뢰된 생산자의 승인·커밋 이력을 합성한다. 실제 실행 UI 검증은 아니다."""
        run = self.root / name
        states = LocalStateStore(run)
        state = states.create("synthetic_actor", policy_version=POLICY_VERSION)
        state.status = "completed"
        state.current_stage = stage
        states.save(state, expected_revision=state.revision)
        scope = ApprovalScope(state.actor_id, state.run_id, stage, tool, f"{state.run_id}:{stage}", "synthetic_arguments", source, config, "synthetic_plan", POLICY_VERSION)
        approvals = ApprovalStore(run / "approvals.sqlite3")
        approval = approvals.record_confirmed_approval(scope)
        task = approvals.consume_and_reserve(approval.approval_id, scope).reservation_id

        for before, after in (("reserved", "running"), ("running", "verified"), ("verified", "committed")):
            approvals.transition_task(scope, task, expected=before, status=after)

        output = run / "attempts" / task
        output.mkdir(parents=True)

        return run, scope, output

    def complete(self, output, scope):
        complete_artifacts(output, ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id))

    def stage(self, contract=None):
        """승인 후 부모가 수행해야 할 고정 복사를 테스트에서 재현한다. native launch는 하지 않는다."""
        contract = contract or self.contract
        root = self.root / "workers" / ("a" * 32)

        for name in ("input", "output/result", "scratch/task"):
            (root / name).mkdir(parents=True)

        create_snapshot(contract.source.snapshot_directory + "/input.bin", root / "input/snapshot", max_input_bytes=contract.source.snapshot.max_input_bytes)
        packet = contract.packet("a" * 64)
        (root / "input/export.json").write_text(json.dumps(packet), encoding="utf-8")
        shutil.copyfile(contract.split.artifact.directory / "split.json", root / "input/split.json")
        request = WorkerRequest(
            task_id=root.name, run_id="b" * 32, actor_id="synthetic_actor", stage="ingest", action_id="run_ingest",
            source_id=self.split.artifact.scope.source_id, config_id=_digest(packet), plan_id="synthetic_plan",
            policy_version=POLICY_VERSION, limits=contract.source.limits,
            input_files=tuple(str(root / name) for name in ("input/snapshot/input.bin", "input/snapshot/snapshot.json", "input/export.json", "input/split.json")),
            dependency_directories=(), output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"),
        )

        return root, request

    def test_exports_exact_selected_rows_without_value_conversion(self):
        """앞자리 0·소수 문자열·한글·인용·줄바꿈과 확정 행 순서를 보존한다."""
        root, request = self.stage()
        before = (self.root / "snapshot/input.bin").read_bytes()

        with patch("pandas.read_csv", side_effect=AssertionError("불필요한 수치 파싱")):
            result = execute_request(request, root)
            report = read_partition_export(root / "output/result/exports", self.contract)

        self.assertEqual(result.status, "succeeded")
        self.assertIsNotNone(report)

        for name in ("train", "validation"):
            with (root / f"output/result/exports/{name}/input.bin").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))

            self.assertEqual(rows[0], ["x", "id", "text", "label"])
            self.assertEqual(rows[1:], [self.rows[position] for position in self.split.positions(name)])

        self.assertFalse((root / "output/result/exports/test").exists())
        self.assertEqual((self.root / "snapshot/input.bin").read_bytes(), before)

    def test_output_bound_blocks_partial_completion(self):
        """출력 상한을 넘으면 잘라서 성공 처리하지 않고 완료 영수증을 만들지 않는다."""
        contract = replace(self.contract, max_partition_bytes=16)
        root, request = self.stage(contract)
        result = execute_request(request, root)
        self.assertEqual(result.status, "blocked_input")
        self.assertEqual(result.artifact_refs, ())
        self.assertFalse((root / "output/result/exports/partition_exports.json").exists())
        self.assertIsNone(read_partition_export(root / "output/result/exports", contract))

    def test_changed_split_is_rejected(self):
        """분할 파일이 승인된 해시와 다르면 새로운 배정을 만들지 않는다."""
        root, request = self.stage()
        (root / "input/split.json").write_text("{}", encoding="utf-8")
        self.assertEqual(execute_request(request, root).status, "blocked_input")

    def test_changed_source_is_rejected(self):
        """복사본 변조는 내보내기 전에 차단한다."""
        root, request = self.stage()
        (root / "input/snapshot/input.bin").write_bytes(b"changed")
        self.assertEqual(execute_request(request, root).status, "blocked_input")
        self.assertFalse((root / "output/result/exports").exists())

    def test_wrong_request_action_is_rejected(self):
        """다른 도구의 요청을 내보내기 작업으로 실행하지 않는다."""
        root, request = self.stage()
        wrong = replace(request, stage="run_eda", action_id="run_eda")
        self.assertEqual(execute_request(wrong, root).status, "blocked_input")

    def test_profile_and_source_binding(self):
        """다른 프로파일·원본을 확정 분할에 붙일 수 없다."""
        profile = replace(self.profile, split_rules=replace(self.profile.split_rules, random_state=43))
        source = replace(self.contract.source, profile_json=json.dumps(profile.to_dict()))

        with self.assertRaises(ValueError):
            PartitionExportContract(self.split, source, 65536)

        with self.assertRaises(ValueError):
            replace(self.contract, max_partition_bytes=True)

    def test_parent_rejects_receipt_and_payload_tampering(self):
        """출처·행 연결 변경과 실제 파일 변조를 부모가 거부한다."""
        root, request = self.stage()
        execute_request(request, root)
        output = root / "output/result/exports"
        path = output / "partition_exports.json"
        original = path.read_text(encoding="utf-8")
        receipt = json.loads(original)
        receipt["partitions"]["train"]["row_count"] += 1
        path.write_text(json.dumps(receipt), encoding="utf-8")
        self.assertIsNone(read_partition_export(output, self.contract))
        path.write_text(original, encoding="utf-8")
        (output / "train/input.bin").write_bytes(b"changed")
        self.assertIsNone(read_partition_export(output, self.contract))

    def test_parent_rejects_extra_test_output(self):
        """Test나 미선언 파일을 추가하면 전체 내보내기를 거부한다."""
        root, request = self.stage()
        execute_request(request, root)
        output = root / "output/result/exports"
        (output / "test.csv").write_text("not allowed", encoding="utf-8")
        self.assertIsNone(read_partition_export(output, self.contract))

    def test_csv_record_and_column_limits(self):
        """큰 레코드·과다 컬럼이 메모리에 무제한 쌓이지 않도록 거부한다."""
        for payload in ("x" * (MAX_RECORD_BYTES + 1), ",".join(["x"] * (MAX_COLUMNS + 1))):
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                list(iter_csv_rows(StringIO(payload)))

        self.assertEqual(list(iter_csv_rows(StringIO('x,y\n"a\nb",01\n'))), [["x", "y"], ["a\nb", "01"]])

    def test_export_output_can_bind_after_committed_producer(self):
        """실제 worker 출력 형식이 앞 단계의 구간 입력 계약과 연결되는지 확인한다."""
        root, request = self.stage()
        self.assertEqual(execute_request(request, root).status, "succeeded")
        run, scope, output = self.fixture_commit("export_run", "ingest", "run_ingest", request.source_id, request.config_id)

        for name in ("train", "validation"):
            shutil.copytree(root / "output/result/exports" / name, output / name)

        shutil.copyfile(root / "output/result/exports/partition_exports.json", output / "partition_exports.json")
        self.complete(output, scope)
        export = read_committed_artifact(run, scope)
        contract = bind_partition_input(self.split, export, "train", self.roles)
        self.assertTrue(contract.check())


if __name__ == "__main__":
    unittest.main()

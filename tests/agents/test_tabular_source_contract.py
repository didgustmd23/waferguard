"""v2 출처 계약을 실제 합성 승인 DB·커밋 증거와 대조한다."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.agents.approvals import ApprovalScope
from src.agents.approval_store import ApprovalStore
from src.agents.artifacts import ArtifactContext, complete_artifacts
from src.agents.input_snapshot import create_snapshot
from src.agents.split_evidence import read_committed_artifact
from src.agents.split_evidence import read_confirmed_split
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits
from src.agents.tabular_source_contract import (
    TabularSourceContract, TabularEdaInputContract, TabularPartitionInputContract, materialization_config_id,
)
from src.preprocessing.contracts import ColumnRoles, TabularInput
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.split import SplitRules, propose_split, _digest
from src.preprocessing.tabular_bundle import BundleRef, write_bundle, write_subset
from tests.agents import test_partition_export as export_tests


class TabularSourceContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        raw = self.root / "original.csv"
        self.frame = pd.DataFrame({"x": list(range(30)), "label": [index % 2 == 0 for index in range(30)]},
                                  index=pd.Index([index // 2 for index in range(30)], name="original"))
        raw.write_bytes(self.frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        self.snapshot = create_snapshot(raw, self.root / "snapshot", max_input_bytes=65536)
        self.spec = {"loader": "delimited_v1", "encoding": "utf-8-sig", "delimiter": ",", "header": "first_row",
                     "empty_field": "missing", "header_repair": "reject"}
        self.limits = WorkerLimits(60, 2048, 2)
        self.inspect_id, self.code_id = "a" * 64, "b" * 64
        spec_json = json.dumps(self.spec)
        self.inspection_sha256 = "d" * 64
        self.config_id = materialization_config_id(self.snapshot, spec_json, self.inspect_id, self.limits, self.code_id,
                                                   inspection_sha256=self.inspection_sha256, max_output_bytes=65536)
        run = self.root / "run"
        states = LocalStateStore(run)
        state = states.create("synthetic_actor", policy_version="tabular_input_v2")
        state.status, state.current_stage = "completed", "ingest"
        states.save(state, expected_revision=state.revision)
        scope = ApprovalScope(state.actor_id, state.run_id, "ingest", "run_ingest", state.run_id + ":ingest",
                              "synthetic_arguments", self.snapshot.source_id, self.config_id, "synthetic_plan", "tabular_input_v2")
        approvals = ApprovalStore(run / "approvals.sqlite3")
        approval = approvals.record_confirmed_approval(scope)
        task = approvals.consume_and_reserve(approval.approval_id, scope).reservation_id
        for before, after in (("reserved", "running"), ("running", "verified"), ("verified", "committed")):
            approvals.transition_task(scope, task, expected=before, status=after)
        output = run / "attempts" / task
        output.mkdir(parents=True)
        provenance = {"source_snapshot_sha256": self.snapshot.sha256, "source_snapshot_size_bytes": self.snapshot.size_bytes,
                      "loader_id": "delimited_v1", "loader_version": "1.0.0", "read_spec": self.spec,
                      "inspection_id": self.inspect_id, "inspection_sha256": self.inspection_sha256, "policy_version": "tabular_input_v2"}
        self.bundle = write_bundle(self.frame, output / "table",
                                   provenance=provenance, max_output_bytes=65536)
        complete_artifacts(output, ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id))
        evidence = read_committed_artifact(run, scope)
        self.source = TabularSourceContract(str(self.root / "snapshot"), self.snapshot, str(output / "table"), self.bundle,
                                           spec_json, self.inspect_id, evidence, self.limits, self.code_id, self.inspection_sha256, 65536)

    def test_source_roundtrip_and_packet_does_not_expose_local_paths(self):
        self.assertTrue(self.source.check_input())
        self.assertEqual(TabularSourceContract.from_dict(self.source.to_dict()), self.source)
        self.assertEqual(TabularSourceContract.from_json(self.source.to_json()), self.source)
        self.assertTrue(self.source.source_id.startswith("tabular_source:"))
        self.assertNotIn(str(self.root), json.dumps(self.source.packet()))

    def test_source_id_tracks_interpretation_but_not_local_paths(self):
        altered = dict(self.spec, empty_field="preserve")
        self.assertNotEqual(replace(self.source, read_spec_json=json.dumps(altered)).source_id, self.source.source_id)
        self.assertEqual(replace(self.source, bundle_directory=str(self.root / "moved")).source_id, self.source.source_id)

    def test_rejects_snapshot_bundle_and_approval_config_changes(self):
        for modified in (replace(self.source, inspection_id="c" * 64),
                         replace(self.source, inspection_sha256="c" * 64),
                         replace(self.source, max_output_bytes=65537),
                         replace(self.source, bundle=replace(self.bundle, manifest_sha256="c" * 64))):
            with self.assertRaises(ValueError):
                modified.check_input()
        (Path(self.source.snapshot_directory) / "input.bin").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.source.check_input()

    def test_rejects_uncommitted_or_wrong_directory_even_with_identical_bundle(self):
        with self.assertRaises(ValueError):
            replace(self.source, bundle_directory=str(self.root)).check_input()
        state_file = Path(self.source.materialization.run_directory) / "state.json"
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state["cancel_requested"] = True
        state_file.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.source.check_input()

    def test_rejects_unknown_spec_duplicate_json_and_version_downgrade(self):
        for payload in (dict(self.spec, converter="os.system"), dict(self.spec, encoding="cp949")):
            with self.assertRaises(ValueError):
                replace(self.source, read_spec_json=json.dumps(payload))
        with self.assertRaises(ValueError):
            replace(self.source, read_spec_json='{"loader":"a","loader":"b"}')
        with self.assertRaises(ValueError):
            TabularSourceContract.from_dict(dict(self.source.to_dict(), schema_version="tabular_source_v1"))

    def test_eda_requires_v2_profile_and_existing_columns(self):
        profile = DatasetProfile("synthetic", "tabular_canonical_v2", "eda_review_v1", ColumnRoles(numeric=("x",), target="label"))
        contract = TabularEdaInputContract(self.source, json.dumps(profile.to_dict()), self.limits)
        self.assertTrue(contract.check_input())
        self.assertEqual(TabularEdaInputContract.from_dict(contract.to_dict()), contract)
        for modified in (replace(profile, adapter_id="tabular_csv_v1"),
                         replace(profile, roles=ColumnRoles(numeric=("absent",)))):
            with self.assertRaises(ValueError):
                TabularEdaInputContract(self.source, json.dumps(modified.to_dict()), self.limits).check_input()

    def test_limits_and_source_serialization_are_strict(self):
        with self.assertRaises((TypeError, ValueError)):
            TabularSourceContract.from_dict(dict(self.source.to_dict(), extra=True))
        with self.assertRaises((TypeError, ValueError)):
            replace(self.source, limits={})

    def partition_fixture(self):
        harness = export_tests.PartitionExportTest()
        harness.root = self.root
        roles = ColumnRoles(numeric=("x",), target="label")
        frame = self.frame
        rules = SplitRules(stratify=True)
        proposal = propose_split(TabularInput(frame, roles), rules, source_id=self.source.source_id, config_id="profile")
        run, scope, output = harness.fixture_commit("split", "commit_split", "commit_split", self.source.source_id, "profile")
        (output / "split.json").write_text(json.dumps(proposal), encoding="utf-8")
        pd.DataFrame([{"row_position": position, "split": name} for name, positions in proposal["partitions"].items()
                      for position in positions]).sort_values("row_position").to_csv(output / "assignments.csv", index=False)
        harness.complete(output, scope)
        split = read_confirmed_split(run, scope)
        run, scope, output = harness.fixture_commit("export", "ingest", "run_ingest", self.source.source_id, "export")
        source_directory = self.root / "full"
        reference = write_bundle(frame, source_directory, provenance={}, max_output_bytes=65536)
        partitions = {}
        for name in ("train", "validation"):
            positions = list(split.positions(name))
            provenance = {"parent_source_id": self.source.source_id, "split_signature": split.signature,
                          "row_positions_sha256": _digest(positions)}
            selected = write_subset(source_directory, positions, output / name, expected=reference,
                                    provenance=provenance, max_output_bytes=65536)
            partitions[name] = {"bundle": asdict(selected), "row_positions_sha256": _digest(positions)}
        (output / "partition_exports.json").write_text(json.dumps({
            "schema_version": "partition_exports_v2", "split_evidence_id": split.signature,
            "source_id": self.source.source_id, "roles": asdict(roles), "partitions": partitions,
        }), encoding="utf-8")
        harness.complete(output, scope)
        export = read_committed_artifact(run, scope)
        return TabularPartitionInputContract(split, export, "train", BundleRef.from_dict(partitions["train"]["bundle"]),
                                             json.dumps(asdict(roles)), self.source.source_id)

    def test_partition_requires_committed_matching_split_and_preserves_positions(self):
        contract = self.partition_fixture()
        self.assertTrue(contract.check())
        self.assertEqual(TabularPartitionInputContract.from_dict(contract.to_dict()), contract)
        self.assertEqual(contract.row_positions, contract.split.positions("train"))
        for modified in (replace(contract, parent_source_id="tabular_source:" + "c" * 64),
                         replace(contract, partition="validation"), replace(contract, bundle=replace(contract.bundle, rows=1))):
            with self.assertRaises(ValueError):
                modified.check()
        with self.assertRaises(ValueError):
            replace(contract, partition="test")

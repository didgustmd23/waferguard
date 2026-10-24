"""v2 typed 진단·분할·구간 내보내기를 실제 계산·승인 DB와 모의 OS 운송으로 검사한다."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from src.agents import snapshot_split_cli as split_cli, partition_export_cli as export_cli
from src.agents import snapshot_split_worker, partition_export_worker, module_pipeline_worker
from src.agents.approval_store import ApprovalStore
from src.agents.executor import RulesExecutor, RegisteredTool
from src.agents.input_profile_pipeline import bind_tabular_diagnostic
from src.agents.pipeline_binding_registry import build_pipeline_execution
from src.agents.split_evidence import read_confirmed_split
from src.agents.state import LocalStateStore
from src.agents.tabular_source_contract import TabularPartitionInputContract
from src.agents import tabular_pipeline
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.quality import QualityRules
from src.preprocessing.split import SplitRules
from src.preprocessing.tabular_bundle import load_bundle, read_bundle_schema
from tests.agents import test_tabular_source_contract as source_fixture


class TabularSplitExportTest(unittest.TestCase):
    def setUp(self):
        source_fixture.TabularSourceContractTest.setUp(self)
        self.profile = DatasetProfile("typed_example", "tabular_canonical_v2", "split_review_v1",
                                      ColumnRoles(numeric=("x",), target="label"), max_plot_features=1,
                                      split_rules=SplitRules(stratify=True))
        (self.root / "splits").mkdir()
        (self.root / "exports").mkdir()
        self.start_patch(patch.object(split_cli, "_directory", side_effect=lambda name, **_: self.root / "splits" / name))
        self.start_patch(patch.object(export_cli, "_directory", side_effect=lambda name, **_: self.root / "exports" / name))
        self.start_patch(patch.object(split_cli, "_current_actor", return_value="synthetic_actor"))
        self.start_patch(patch.object(export_cli, "_current_actor", return_value="synthetic_actor"))
        self.calls = []
        self.start_patch(patch("src.agents.snapshot_split.run_split_worker", side_effect=lambda *a: self.transport(*a, worker=snapshot_split_worker.execute_request)))
        self.start_patch(patch("src.agents.partition_export.run_export_worker", side_effect=lambda *a: self.transport(*a, worker=partition_export_worker.execute_request)))
        self.start_patch(patch("src.agents.module_pipeline.run_pipeline_worker", side_effect=lambda *a: self.transport(*a, worker=module_pipeline_worker.execute_request)))

    def start_patch(self, item):
        result = item.start()
        self.addCleanup(item.stop)
        return result

    def transport(self, request, root, cancel, *, worker):
        self.calls.append(request.stage)
        reply = worker(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes}}}
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
        return outcome

    def drive(self, tool, manager):
        for _ in range(15):
            state = tool.states.load()
            if state.status in {"completed", "blocked_input", "review_required", "failed_runtime", "security_locked"}:
                return state
            result = manager.run_next()
            if result.code == "APPROVAL_REQUIRED":
                approval = manager.approvals.record_confirmed_approval(manager.scope_for(state.current_stage))
                manager.run_next(approval_id=approval.approval_id)
        self.fail("단계 진행이 끝나지 않았습니다.")

    def split(self, name="typed"):
        split_cli.create_from_tabular(name, self.source, self.profile, self.limits, confirmed=True)
        tool = split_cli.open_run(name)
        manager = split_cli.executor(tool)
        state = self.drive(tool, manager)
        self.assertEqual(state.status, "completed", state)
        evidence = read_confirmed_split(tool.states.directory, manager.scope_for("commit_split"))
        return tool, manager, evidence

    def make_source(self, frame, name):
        """고정 typed 표를 실제 합성 생성 승인·커밋 증거에 결합한다."""
        from src.agents.approvals import ApprovalScope
        from src.agents.artifacts import complete_artifacts, ArtifactContext
        from src.agents.input_snapshot import create_snapshot
        from src.agents.split_evidence import read_committed_artifact
        from src.agents.tabular_source_contract import materialization_config_id, TabularSourceContract
        from src.preprocessing.tabular_bundle import write_bundle
        directory = self.root / name
        directory.mkdir()
        raw = directory / "source.csv"
        raw.write_text(frame.to_csv(index=False), encoding="utf-8")
        snapshot = create_snapshot(raw, directory / "snapshot", max_input_bytes=65536)
        config = materialization_config_id(snapshot, self.source.read_spec_json, self.inspect_id, self.limits, self.code_id,
                                           inspection_sha256=self.inspection_sha256, max_output_bytes=65536)
        states = LocalStateStore(directory / "run")
        state = states.create("synthetic_actor", policy_version="tabular_input_v2")
        state.status, state.current_stage = "completed", "ingest"
        states.save(state, expected_revision=state.revision)
        scope = ApprovalScope(state.actor_id, state.run_id, "ingest", "run_ingest", state.run_id + ":ingest", "fixture", snapshot.source_id,
                              config, "fixture", "tabular_input_v2")
        approvals = ApprovalStore(states.directory / "approvals.sqlite3")
        approval = approvals.record_confirmed_approval(scope)
        task = approvals.consume_and_reserve(approval.approval_id, scope).reservation_id
        for before, after in (("reserved", "running"), ("running", "verified"), ("verified", "committed")):
            approvals.transition_task(scope, task, expected=before, status=after)
        output = states.directory / "attempts" / task
        output.mkdir(parents=True)
        provenance = {"source_snapshot_sha256": snapshot.sha256, "source_snapshot_size_bytes": snapshot.size_bytes,
                      "loader_id": "delimited_v1", "loader_version": "1.0.0", "read_spec": self.spec, "inspection_id": self.inspect_id,
                      "inspection_sha256": self.inspection_sha256, "policy_version": "tabular_input_v2"}
        reference = write_bundle(frame, output / "table", provenance=provenance, max_output_bytes=65536)
        complete_artifacts(output, ArtifactContext(scope.source_id, scope.config_id, scope.plan_id, scope.policy_version, scope.tool_id))
        return TabularSourceContract(str(directory / "snapshot"), snapshot, str(output / "table"), reference, self.source.read_spec_json,
                                     self.inspect_id, read_committed_artifact(states.directory, scope), self.limits, self.code_id,
                                     self.inspection_sha256, 65536)

    def test_typed_split_and_export_preserve_positions_index_and_bool_target(self):
        original = Path(self.source.snapshot_directory, "input.bin").read_bytes()
        tool, manager, split = self.split()
        self.assertEqual(split.artifact.scope.source_id, self.source.source_id)
        export_cli.create_from_tabular("typed_export", "typed", self.source, self.limits, 65536, confirmed=True)
        exported = export_cli.open_run("typed_export")
        export_manager = export_cli.executor(exported)
        self.assertEqual(self.drive(exported, export_manager).status, "completed")
        task = export_manager.approvals.get_task(export_manager.scope_for("ingest"))
        output = export_manager._output(task)
        self.assertEqual({p.name for p in output.iterdir()}, {"train", "validation", "partition_exports.json", "AGENT_COMPLETE.json"})
        receipt = json.loads((output / "partition_exports.json").read_text(encoding="utf-8"))
        from src.agents.split_evidence import read_committed_artifact
        from src.preprocessing.tabular_bundle import BundleRef
        for name in ("train", "validation"):
            reference = BundleRef.from_dict(receipt["partitions"][name]["bundle"])
            restored = load_bundle(output / name, expected=reference)
            pd.testing.assert_frame_equal(restored, self.frame.iloc[list(split.positions(name))])
            contract = TabularPartitionInputContract(split, read_committed_artifact(exported.states.directory, export_manager.scope_for("ingest")),
                                                     name, reference, json.dumps(asdict(self.profile.roles)), self.source.source_id)
            self.assertTrue(contract.check())
        self.assertEqual(Path(self.source.snapshot_directory, "input.bin").read_bytes(), original)
        self.assertFalse((output / "test").exists())
        self.assertEqual(export_cli.status(exported)["partition_contracts"], "passed")

    def test_diagnostic_uses_fixed_plan_and_schema_bound_parent_check(self):
        binding = bind_tabular_diagnostic(self.source, self.limits)
        states = LocalStateStore(self.root / "diagnostics/runs/example")
        states.create("synthetic_actor", policy_version="tabular_diagnostic_worker_v2")
        execution = build_pipeline_execution(binding.adapter_id, states, binding)
        tool = execution.worker_tool
        from src.agents.snapshot_split_cli import _preflight, _validate_preflight
        manager = RulesExecutor(states, ApprovalStore(states.directory / "approvals.sqlite3"), tool.plan,
                                (RegisteredTool("run_preflight", _preflight, _validate_preflight),
                                 RegisteredTool("run_eda", tool.execute, tool.validate, requires_approval=True)))
        self.assertEqual(self.drive(tool, manager).status, "completed")
        task = manager.approvals.get_task(manager.scope_for("run_eda"))
        result = tool._read_result(manager._output(task) / "pipeline")
        self.assertEqual(result["row_count"], len(self.frame))
        import csv
        from io import StringIO
        folder = manager._output(task) / "pipeline/input_profile"
        path = folder / "input_profile.csv"
        rows = list(csv.reader(StringIO(path.read_text(encoding="utf-8-sig"))))
        rows[1], rows[2] = rows[2], rows[1]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
        self.assertIsNone(tool._read_result(manager._output(task) / "pipeline"))

    def test_invalid_profile_and_source_are_rejected_before_run_creation(self):
        with self.assertRaises(ValueError):
            split_cli.create_from_tabular("bad", self.source, replace(self.profile, adapter_id="tabular_csv_v1"), self.limits, confirmed=True)
        self.assertFalse((self.root / "splits/bad").exists())

    def test_diagnostic_binding_runs_through_existing_pipeline_approval(self):
        from src.agents.pipeline_approval import PipelineApprovalRun
        binding = bind_tabular_diagnostic(self.source, self.limits)
        states = LocalStateStore(self.root / "diagnostics/runs/approval")
        states.create("synthetic_actor", policy_version=binding.policy_version)
        run = PipelineApprovalRun(states, binding)
        self.assertEqual(self.drive(run.pipeline_execution.worker_tool, run.executor).status, "completed")
        self.assertEqual(self.calls, ["run_eda"])
        changed = replace(self.source, inspection_id="c" * 64)
        with self.assertRaises(ValueError):
            split_cli.create_from_tabular("bad", changed, self.profile, self.limits, confirmed=True)
        self.assertFalse((self.root / "splits/bad").exists())

    def test_quality_review_stops_before_eda_and_partitioning(self):
        profile = replace(self.profile, roles=ColumnRoles(numeric=("x", "label")),
                          quality_rules=QualityRules(high_missing="review"))
        # bool은 수치 역할의 유효 입력이 아니다. 기존 품질 차단을 우회하지 않는다.
        split_cli.create_from_tabular("review", self.source, profile, self.limits, confirmed=True)
        tool = split_cli.open_run("review")
        state = self.drive(tool, split_cli.executor(tool))
        self.assertIn(state.status, {"blocked_input", "review_required"})
        self.assertNotIn("run_eda", self.calls)
        self.assertNotIn("commit_split", self.calls)

    def test_group_split_preserves_ids_large_integers_and_arbitrary_index(self):
        self.frame = pd.DataFrame({"x": [float(i) for i in range(60)], "label": [True, False] * 30,
                                   "g": [f"group_{i // 3}" for i in range(60)], "id": [f"{i:04d}" for i in range(60)],
                                   "large_id": pd.Series([2**63 + i for i in range(60)], dtype="uint64")})
        self.frame.index = pd.Index([8, 8, 3, 2] + list(range(56)), name="original")
        self.source = self.make_source(self.frame, "group_source")
        self.profile = replace(self.profile, roles=ColumnRoles(numeric=("x",), target="label", group="g", identifiers=("id", "large_id")),
                               split_rules=SplitRules(strategy="group", stratify=False))
        self.test_typed_split_and_export_preserve_positions_index_and_bool_target()
        tool = split_cli.open_run("typed")
        split = read_confirmed_split(tool.states.directory, split_cli.executor(tool).scope_for("commit_split"))
        sets = [set(self.frame.iloc[list(split.positions(p))]["g"]) for p in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_high_missing_review_stops_without_automatic_exclusion(self):
        self.frame["missing"] = [None] * 25 + list(range(5))
        self.source = self.make_source(self.frame, "missing_source")
        self.profile = replace(self.profile, roles=ColumnRoles(numeric=("x", "missing"), target="label"))
        split_cli.create_from_tabular("review", self.source, self.profile, self.limits, confirmed=True)
        tool = split_cli.open_run("review")
        state = self.drive(tool, split_cli.executor(tool))
        self.assertNotEqual(state.status, "completed")
        self.assertEqual(self.calls, ["audit"])
        self.assertIn("missing", tool.contract.profile.roles.numeric)

    def test_large_row_count_and_resource_escalation_fail_before_run_creation(self):
        for source, limits in ((replace(self.source, bundle=replace(self.bundle, rows=1_000_001)), self.limits),
                                (self.source, replace(self.limits, memory_mib=4096))):
            with self.subTest(rows=source.bundle.rows), self.assertRaises(ValueError):
                split_cli.create_from_tabular("oversized", source, self.profile, limits, confirmed=True)
            self.assertFalse((self.root / "splits/oversized").exists())

    def test_diagnostic_worker_rejects_forged_plan_and_profile_after_config_rehash(self):
        binding = bind_tabular_diagnostic(self.source, self.limits)
        for attack in ("step", "id", "signature", "profile", "settings"):
            states = LocalStateStore(self.root / "diagnostics/runs" / attack)
            states.create("synthetic_actor", policy_version="tabular_diagnostic_worker_v2")
            tool = build_pipeline_execution(binding.adapter_id, states, binding).worker_tool
            request = tool._request((str(len(attack)) * 32)[:32])
            root = tool._stage(request)
            path = root / "input/eda.json"
            packet = json.loads(path.read_text(encoding="utf-8"))
            if attack == "step":
                packet["pipeline"]["steps"].append(dict(packet["pipeline"]["steps"][0], step_id="extra"))
            elif attack == "id":
                packet["pipeline"]["pipeline_id"] = "other"
            elif attack == "signature":
                packet["pipeline_signature"] = "f" * 64
            elif attack == "profile":
                packet["profile"] = self.profile.to_dict()
            else:
                packet["pipeline"]["steps"][0]["settings"] = {"anything": True}
            path.write_text(json.dumps(packet), encoding="utf-8")
            from src.agents.snapshot_eda import _digest
            reply = module_pipeline_worker.execute_request(replace(request, config_id=_digest(packet)), root)
            with self.subTest(attack=attack):
                self.assertEqual(reply.status, "blocked_input")
                self.assertFalse((root / "output/result/pipeline").exists())

    def test_export_rejects_another_committed_source_before_creation(self):
        self.split()
        other = self.make_source(self.frame.assign(x=self.frame["x"] + 100), "other_source")
        with self.assertRaises(ValueError):
            export_cli.create_from_tabular("bad_export", "typed", other, self.limits, 65536, confirmed=True)
        self.assertFalse((self.root / "exports/bad_export").exists())

    def test_split_worker_rejects_csv_disguised_as_canonical_and_profile_null(self):
        split_cli.create_from_tabular("typed", self.source, self.profile, self.limits, confirmed=True)
        tool = split_cli.open_run("typed")
        from src.agents.snapshot_eda import _digest
        for attack in ("csv", "null_profile", "interpretation"):
            request = tool._request_for((str(len(attack)) * 32)[:32], "audit")
            root = tool._stage_for(request, None)
            packet = json.loads((root / "input/split_task.json").read_text(encoding="utf-8"))
            if attack == "csv":
                (root / "input/table/data.jsonl").write_text("x,label\n1,true\n", encoding="utf-8")
            elif attack == "null_profile":
                packet["profile"] = None
            else:
                packet["source"]["read_spec"]["empty_field"] = "preserve"
            (root / "input/split_task.json").write_text(json.dumps(packet), encoding="utf-8")
            reply = snapshot_split_worker.execute_request(replace(request, config_id=_digest(packet)), root)
            with self.subTest(attack=attack):
                self.assertEqual(reply.status, "blocked_input")

    def test_parent_rejects_export_position_tampering(self):
        self.split()
        export_cli.create_from_tabular("typed_export", "typed", self.source, self.limits, 65536, confirmed=True)
        tool = export_cli.open_run("typed_export")
        manager = export_cli.executor(tool)
        self.assertEqual(self.drive(tool, manager).status, "completed")
        output = manager._output(manager.approvals.get_task(manager.scope_for("ingest")))
        from src.agents.partition_export_results import read_partition_export
        receipt_path = output / "partition_exports.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["partitions"]["train"]["row_positions_sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.assertIsNone(read_partition_export(output, tool.contract, committed=True))

    def test_worker_copy_lists_include_typed_reader_but_not_parent_approval_modules(self):
        from src.agents import module_pipeline_transport, snapshot_split_transport, partition_export_transport
        for item in (module_pipeline_transport, snapshot_split_transport, partition_export_transport):
            with self.subTest(module=item.__name__):
                self.assertIn("src/preprocessing/tabular_bundle.py", item.WORKER_FILES)
                self.assertIn("src/agents/tabular_worker_input.py", item.WORKER_FILES)
                self.assertNotIn("src/agents/tabular_pipeline.py", item.WORKER_FILES)
                self.assertNotIn("src/agents/tabular_source_contract.py", item.WORKER_FILES)
                self.assertNotIn("src/agents/approval_store.py", item.WORKER_FILES)
                self.assertIn("src/agents/tabular_pipeline.py", item.CONTROL_FILES)

    def test_parent_rejects_self_consistent_split_missing_source_rows(self):
        from src.preprocessing.contracts import TabularInput
        from src.preprocessing.split import propose_split
        import csv
        split_cli.create_from_tabular("typed", self.source, self.profile, self.limits, confirmed=True)
        tool = split_cli.open_run("typed")
        proposal = propose_split(TabularInput(self.frame.iloc[:15], self.profile.roles), self.profile.split_rules,
                                 source_id=tool.plan.source_id, config_id=tool.plan.config_id)
        self.assertEqual(proposal["status"], "passed")
        preview, committed = self.root / "short_preview", self.root / "short_commit"
        preview.mkdir()
        committed.mkdir()
        (preview / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
        (preview / "proposal.md").write_text("short fixture", encoding="utf-8")
        (committed / "split.json").write_text(json.dumps(proposal), encoding="utf-8")
        assignments = {p: name for name, positions in proposal["partitions"].items() for p in positions}
        with (committed / "assignments.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["row_position", "split"])
            writer.writerows(sorted(assignments.items()))
        self.assertIsNone(tool._report_status(preview, "propose_split"))
        self.assertIsNone(tool._report_status(committed, "commit_split"))

    def test_parent_rejects_eda_from_incomplete_input(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from src.preprocessing.eda_report import generate_eda_report
        from src.preprocessing.contracts import TabularInput
        split_cli.create_from_tabular("typed", self.source, self.profile, self.limits, confirmed=True)
        tool = split_cli.open_run("typed")
        directory = self.root / "short_eda"
        generate_eda_report(TabularInput(self.frame.iloc[:15], self.profile.roles), directory, max_plot_features=1)
        self.assertIsNone(tool._report_status(directory, "run_eda"))

    def test_parent_rejects_wrong_diagnostic_type_for_known_text_schema(self):
        import csv
        from io import StringIO
        self.source = self.make_source(self.frame.assign(code=[f"id_{i}" for i in range(len(self.frame))]), "text_source")
        binding = bind_tabular_diagnostic(self.source, self.limits)
        states = LocalStateStore(self.root / "diagnostics/runs/text")
        states.create("synthetic_actor", policy_version="tabular_diagnostic_worker_v2")
        tool = build_pipeline_execution(binding.adapter_id, states, binding).worker_tool
        from src.agents.snapshot_split_cli import _preflight, _validate_preflight
        manager = RulesExecutor(states, ApprovalStore(states.directory / "approvals.sqlite3"), tool.plan,
            (RegisteredTool("run_preflight", _preflight, _validate_preflight), RegisteredTool("run_eda", tool.execute, tool.validate, requires_approval=True)))
        self.assertEqual(self.drive(tool, manager).status, "completed")
        output = manager._output(manager.approvals.get_task(manager.scope_for("run_eda"))) / "pipeline"
        path = output / "input_profile/input_profile.csv"
        rows = list(csv.reader(StringIO(path.read_text(encoding="utf-8-sig"))))
        row = next(item for item in rows[1:] if item[0] == "code")
        row[1], row[5] = "other", ""
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
        self.assertIsNone(tool._read_result(output))

    def test_copied_v2_workers_run_without_parent_modules_or_repository(self):
        from src.agents import module_pipeline_transport, snapshot_split_transport, partition_export_transport
        binding = bind_tabular_diagnostic(self.source, self.limits)
        states = LocalStateStore(self.root / "diagnostics/runs/copied")
        states.create("synthetic_actor", policy_version="tabular_diagnostic_worker_v2")
        diagnostic = build_pipeline_execution(binding.adapter_id, states, binding).worker_tool
        request = diagnostic._request("a" * 32)
        roots = [(module_pipeline_transport, request, diagnostic._stage(request))]
        self.split()
        split_tool = split_cli.open_run("typed")
        request = split_tool._request_for("b" * 32, "audit")
        roots.append((snapshot_split_transport, request, split_tool._stage_for(request, None)))
        export_cli.create_from_tabular("export", "typed", self.source, self.limits, 65536, confirmed=True)
        export_tool = export_cli.open_run("export")
        request = export_tool._request("c" * 32)
        roots.append((partition_export_transport, request, export_tool._stage(request)))
        project = Path(__file__).resolve().parents[2]
        for transport, request, root in roots:
            for name in (*transport.WORKER_FILES, transport.ENTRY):
                target = root / "tools" / ("worker.py" if name == transport.ENTRY else name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project / name, target)
            script = ("import sys,runpy,pathlib; sys.path[:0]=[sys.argv[1],sys.argv[2]]; "
                      "ns=runpy.run_path(sys.argv[1]+'/worker.py'); from src.agents.worker_contract import WorkerRequest; "
                      "root=pathlib.Path(sys.argv[3]); req=WorkerRequest.from_json((root/'input/request.json').read_text(encoding='utf-8')); "
                      "reply=ns['execute_request'](req,root); assert reply.status=='succeeded',reply; "
                      "assert 'src.agents.tabular_source_contract' not in sys.modules; "
                      "assert 'src.agents.approval_store' not in sys.modules; print('COPIED_V2_OK')")
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(root / "tools"),
                                     str(Path(sys.prefix) / "Lib/site-packages"), str(root)], cwd=root,
                                    capture_output=True, text=True, encoding="utf-8", timeout=60)
            with self.subTest(module=transport.__name__):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("COPIED_V2_OK", result.stdout)

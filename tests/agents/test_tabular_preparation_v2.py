"""typed Train 기준·Validation 적용과 메타데이터 보존을 합성 승인 흐름으로 검사한다."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import PropertyMock, patch

import pandas as pd

from src.agents import snapshot_split_cli as split_cli, partition_export_cli as export_cli
from src.agents import preparation_candidate_cli as candidate_cli, preparation_apply_cli as apply_cli
from src.agents import snapshot_split_worker, partition_export_worker, preparation_candidate_worker, preparation_apply_worker
from src.agents.preparation_apply_contract import read_candidate_evidence
from src.agents.preparation_candidate_contract import PreparationCandidateContract
from src.agents.preparation_candidate_results import read_candidate
from src.agents.preparation_apply_results import read_application
from src.agents.partition_input_contract import bind_typed_partition_input
from src.agents.tabular_preparation_worker import load_row_metadata
from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.profiles import DatasetProfile
from src.preprocessing.split import SplitRules
from src.preprocessing.transform import PreparationRules
from src.preprocessing.tabular_bundle import read_bundle_schema
from tests.agents import test_tabular_source_contract as source_fixture, test_tabular_split_export as split_fixture


class TabularPreparationV2Test(unittest.TestCase):
    make_source = split_fixture.TabularSplitExportTest.make_source
    drive = split_fixture.TabularSplitExportTest.drive
    transport = split_fixture.TabularSplitExportTest.transport
    start_patch = split_fixture.TabularSplitExportTest.start_patch

    def setUp(self):
        source_fixture.TabularSourceContractTest.setUp(self)
        self.frame = pd.DataFrame({"x": pd.array([1, None, 9, 100, None, 700], dtype="Float64"),
                                   "label": pd.array([True, False, True, False, True, False], dtype="boolean"),
                                   "g": ["a", "a", "a", "b", "b", "c"], "id": [f"{i:04d}" for i in range(6)],
                                   "large_id": pd.array([2**63 + i for i in range(6)], dtype="UInt64")})
        self.frame.index = pd.Index([8, 8, 3, 2, 7, 1], name="original")
        self.source = self.make_source(self.frame, "typed_source")
        self.profile = DatasetProfile("typed_prepare", "tabular_canonical_v2", "split_review_v1",
            ColumnRoles(numeric=("x",), target="label", group="g", identifiers=("id", "large_id")), max_plot_features=1,
            split_rules=SplitRules(strategy="group", random_state=0, train_ratio=1/3, validation_ratio=1/3, test_ratio=1/3, require_all_classes=False))
        self.rules = PreparationRules(numeric_imputation="median")
        self.calls = []
        for module, name in ((split_cli, "splits"), (export_cli, "exports"), (candidate_cli, "candidates"), (apply_cli, "applications")):
            (self.root / name).mkdir()
            self.start_patch(patch.object(module, "_directory", side_effect=lambda run, folder=name, **_: self.root / folder / run))
            self.start_patch(patch.object(module, "_current_actor", return_value="synthetic_actor"))
        for target, worker in (("src.agents.snapshot_split.run_split_worker", snapshot_split_worker.execute_request),
                               ("src.agents.partition_export.run_export_worker", partition_export_worker.execute_request),
                               ("src.agents.preparation_candidate.run_candidate_worker", preparation_candidate_worker.execute_request),
                               ("src.agents.preparation_apply.run_apply_worker", preparation_apply_worker.execute_request)):
            self.start_patch(patch(target, side_effect=lambda *args, selected=worker: self.transport(*args, worker=selected)))

    def train(self):
        split_cli.create_from_tabular("split", self.source, self.profile, self.limits, confirmed=True)
        tool = split_cli.open_run("split")
        manager = split_cli.executor(tool)
        self.assertEqual(self.drive(tool, manager).status, "completed")
        from src.agents.split_evidence import read_confirmed_split, read_committed_artifact
        split = read_confirmed_split(tool.states.directory, manager.scope_for("commit_split"))
        self.assertEqual(split.positions("train"), (0, 1, 2))
        self.assertEqual(split.positions("validation"), (3, 4))
        export_cli.create_from_tabular("export", "split", self.source, self.limits, 65536, confirmed=True)
        tool = export_cli.open_run("export")
        manager = export_cli.executor(tool)
        self.assertEqual(self.drive(tool, manager).status, "completed")
        export = read_committed_artifact(tool.states.directory, manager.scope_for("ingest"))
        return bind_typed_partition_input(split, export, "train", self.profile.roles)

    def candidate(self):
        train = self.train()
        candidate_cli.create_from_tabular("candidate", train, self.rules, self.limits, 1024 * 1024, confirmed=True)
        tool = candidate_cli.open_run("candidate")
        manager = candidate_cli.executor(tool)
        self.assertEqual(self.drive(tool, manager).status, "completed")
        return tool, manager, read_candidate_evidence(tool.states.directory, "synthetic_actor")

    def test_train_median_is_five_validation_is_not_refitted_and_metadata_is_typed(self):
        tool, manager, evidence = self.candidate()
        state = json.loads((evidence.directory / "preprocessor.json").read_text(encoding="utf-8"))
        self.assertEqual(state["state"]["records"][0]["impute"], 5.0)
        self.assertNotIn("validation", " ".join(tool._request("a" * 32).input_files))
        apply_cli.create("apply", evidence.artifact.run_directory, self.limits, 1024 * 1024, confirmed=True)
        applied = apply_cli.open_run("apply")
        manager = apply_cli.executor(applied)
        with patch("src.preprocessing.transform.TabularPreprocessor.fit", side_effect=AssertionError("refit 금지")):
            self.assertEqual(self.drive(applied, manager).status, "completed")
        output = manager._output(manager.approvals.get_task(manager.scope_for("apply_preparation"))) / "prepared"
        result = read_application(output, applied.contract, applied.code_id)
        self.assertIsNotNone(result)
        self.assertIs(result["refit"], False)
        features = pd.read_csv(output / "validation_features.csv")
        self.assertEqual(features.iloc[:, 0].tolist(), [100.0, 5.0])
        for name, positions in (("train", [0, 1, 2]), ("validation", [3, 4])):
            restored = load_row_metadata(output / f"{name}_rows.json")
            expected = self.frame.iloc[positions][["label", "g", "id", "large_id"]]
            pd.testing.assert_frame_equal(restored, expected)
        self.assertFalse(any("test" in p.name for p in output.iterdir()))

    def test_repeated_checks_in_one_public_call_share_only_matching_results(self):
        from src.preprocessing import tabular_bundle as bundle
        from src.agents.tabular_preparation import checked_feature_types
        train = self.train()
        consumed = []
        original = bundle._rows

        def counted(root, schema, *, verified_stream=None):
            if verified_stream is not None:
                consumed.append(root)
            return original(root, schema, verified_stream=verified_stream)

        def with_roles(**changes):
            return replace(train, roles_json=json.dumps(asdict(replace(train.roles, **changes))))

        categorical = with_roles(identifiers=("large_id",), categorical=("id",))
        # 같은 범주 컬럼이라도 역할이 다르면 검사 조건이 달라 공유하지 않는다.
        other_roles = with_roles(identifiers=(), categorical=("id",), numeric=("x", "large_id"))
        with patch.object(bundle, "_rows", counted):
            with bundle.verification_scope():
                for _ in range(3):
                    self.assertTrue(train.check())
                self.assertEqual(len(consumed), 1)
                self.assertEqual(checked_feature_types(categorical), checked_feature_types(categorical))
                self.assertEqual(len(consumed), 2)
                checked_feature_types(other_roles)
                self.assertEqual(len(consumed), 3)
                # 다른 출처·기대 지문은 재사용하지 않고 거부한다.
                for forged in (replace(train, parent_source_id="tabular_source:" + "0" * 64),
                               replace(train, bundle=replace(train.bundle, completion_sha256="0" * 64))):
                    with self.subTest(forged=forged.parent_source_id[-8:]), self.assertRaises(ValueError):
                        forged.check()
                # 공유 결과가 기록된 상태에서도 행 위치가 다른 계약은 거부한다. 공유값은 계약이 아니라 바이트에서 읽은 위치 해시다.
                from src.agents.tabular_source_contract import TabularPartitionInputContract, _digest
                shared = bundle.scoped_result(train.bundle_directory, expected=train.bundle, condition=("row_positions",),
                                              compute=lambda: self.fail("기록된 결과를 써야 합니다"))
                self.assertEqual(shared, _digest(list(train.row_positions)))
                with patch.object(TabularPartitionInputContract, "row_positions", new_callable=PropertyMock,
                                  return_value=(0, 1, 5)), self.assertRaises(ValueError):
                    train.check()
                self.assertEqual(len(consumed), 3)
            # 공개 호출이 끝나면 다음 검사는 다시 행을 소비한다.
            train.check()
            self.assertEqual(len(consumed), 4)
            # 같은 범위 안에서도 구간 바이트가 바뀌면 재사용하지 않고 거부한다.
            with bundle.verification_scope():
                train.check()
                path = train.bundle_directory / "data.jsonl"
                path.write_bytes(path.read_bytes()[:-2] + b" \n")
                with self.assertRaises(ValueError):
                    train.check()

    def test_v1_downgrade_and_nontrain_input_are_rejected(self):
        train = self.train()
        with self.assertRaises((TypeError, ValueError)):
            PreparationCandidateContract(train, self.rules, self.limits)
        with self.assertRaises(ValueError):
            PreparationCandidateContract(replace(train, partition="validation"), self.rules, self.limits, schema_version="preparation_candidate_v2")
        valid = PreparationCandidateContract(train, self.rules, self.limits, schema_version="preparation_candidate_v2")
        self.assertEqual(valid.packet("a" * 64)["version"], "preparation_candidate_v2")
        self.assertNotIn("snapshot", valid.packet("a" * 64))

    def test_candidate_state_or_input_change_is_rejected_before_apply_creation(self):
        _, _, evidence = self.candidate()
        (evidence.directory / "preprocessor.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            apply_cli.create("apply", evidence.artifact.run_directory, self.limits, 1024 * 1024, confirmed=True)
        self.assertFalse((self.root / "applications/apply").exists())

    def test_typed_input_changed_after_apply_approval_never_starts_worker(self):
        _, _, evidence = self.candidate()
        apply_cli.create("apply", evidence.artifact.run_directory, self.limits, 1024 * 1024, confirmed=True)
        tool = apply_cli.open_run("apply")
        manager = apply_cli.executor(tool)
        self.assertEqual(manager.run_next().code, "STEP_COMPLETED")
        self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")
        scope = manager.scope_for("apply_preparation")
        approval = manager.approvals.record_confirmed_approval(scope)
        path = tool.contract.train.bundle_directory / "data.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with patch("src.agents.preparation_apply.run_apply_worker") as worker:
            result = manager.run_next(approval_id=approval.approval_id)
        worker.assert_not_called()
        self.assertNotEqual(result.status, "completed")
        task = manager.approvals.get_task(scope)
        self.assertFalse((manager._output(task) / "AGENT_COMPLETE.json").exists())

    def test_numeric_bool_or_timestamp_feature_is_rejected_before_candidate_creation(self):
        train = self.train()
        roles = replace(train.roles, numeric=("label",), target=None)
        altered = replace(train, roles_json=json.dumps(asdict(roles)))
        with self.assertRaises(ValueError):
            candidate_cli.create_from_tabular("bad", altered, self.rules, self.limits, 1024 * 1024, confirmed=True)
        self.assertFalse((self.root / "candidates/bad").exists())
        from src.agents.tabular_preparation import check_feature_types
        from src.agents.tabular_source_contract import TabularPartitionInputContract
        with patch.object(TabularPartitionInputContract, "check", return_value=True):
            with self.assertRaisesRegex(ValueError, "수치 특징"):
                check_feature_types(altered)
        # 타입 검사만 분리해 시각 특징도 같은 사전 검사에서 거부하는지 확인한다.
        self.frame["date"] = pd.date_range("2020-01-01", periods=6)
        source = self.make_source(self.frame, "timestamp_source")
        schema = read_bundle_schema(source.bundle_directory, expected=source.bundle)
        with patch.object(TabularPartitionInputContract, "check", return_value=True), patch("src.agents.tabular_preparation.read_bundle_schema", return_value=schema):
            invalid = replace(train, roles_json=json.dumps(asdict(replace(train.roles, numeric=("date",)))))
            with self.assertRaisesRegex(ValueError, "수치 특징"):
                check_feature_types(invalid)

    def test_parent_dependencies_are_fingerprinted_but_not_copied_to_workers(self):
        from src.agents import preparation_candidate_transport as ct, preparation_apply_transport as at
        for transport in (ct, at):
            for name in ("src/agents/tabular_input_worker.py", "src/preprocessing/tabular_loaders.py", "src/agents/tabular_source_contract.py", "src/agents/tabular_preparation.py"):
                with self.subTest(transport=transport.__name__, path=name):
                    self.assertIn(name, transport.CONTROL_FILES)
                    self.assertNotIn(name, transport.WORKER_FILES)

    def test_different_actor_is_rejected_before_typed_candidate_creation(self):
        train = self.train()
        with patch.object(candidate_cli, "_current_actor", return_value="other_actor"):
            with self.assertRaises(ValueError):
                candidate_cli.create_from_tabular("bad", train, self.rules, self.limits, 1024 * 1024, confirmed=True)
        self.assertFalse((self.root / "candidates/bad").exists())

    def test_typed_pipeline_uses_existing_candidate_review_and_apply_backends(self):
        from src.agents import preparation_pipeline_cli as pipeline
        train = self.train()
        (self.root / "pipelines").mkdir()
        self.start_patch(patch.object(pipeline, "_directory", side_effect=lambda name, **_: self.root / "pipelines" / name))
        self.start_patch(patch.object(pipeline, "_current_actor", return_value="synthetic_actor"))
        config = Path(__file__).resolve().parents[2] / "configs/agents/pipelines/tabular_preparation_example.json"
        result = pipeline.create_from_tabular("typed", config, train, self.limits, 1024 * 1024, 1024 * 1024, confirmed=True)
        self.assertFalse(result["worker_started"])
        directory, record, tool = pipeline.open_run("typed")
        self.assertEqual(tool.contract.schema_version, "preparation_candidate_v2")
        self.assertEqual(tool.contract.train, train)
        self.assertFalse((directory / "review.json").exists())

    def test_nullable_category_and_large_integer_metadata_survive_copied_workers(self):
        from src.agents import preparation_candidate_transport as ct, preparation_apply_transport as at
        self.frame["cat"] = pd.Categorical(["v", None, "w", "v", "w", "v"], categories=["v", "w", "unused"], ordered=True)
        self.frame["label"] = pd.array([2**63 + i for i in range(6)], dtype="UInt64")
        self.source = self.make_source(self.frame, "category_source")
        self.profile = replace(self.profile, roles=replace(self.profile.roles, categorical=("cat",)))
        tool, _, evidence = self.candidate()
        request = tool._request("c" * 32)
        roots = [(ct, request, tool._stage(request))]
        apply_cli.create("apply", evidence.artifact.run_directory, self.limits, 1024 * 1024, confirmed=True)
        applied = apply_cli.open_run("apply")
        request = applied._request("d" * 32)
        roots.append((at, request, applied._stage(request)))
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
                      "assert 'src.agents.approval_store' not in sys.modules; print('COPIED_TYPED_PREPARATION_OK')")
            result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(root / "tools"),
                                     str(Path(sys.prefix) / "Lib/site-packages"), str(root)], cwd=root,
                                    capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report_root = roots[1][2] / "output/result/prepared"
        expected = self.frame.iloc[[3, 4]][["label", "g", "id", "large_id"]]
        pd.testing.assert_frame_equal(load_row_metadata(report_root / "validation_rows.json"), expected)
        self.assertIsNotNone(read_application(report_root, applied.contract, applied.code_id))
        rows_path = report_root / "validation_rows.json"
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        rows["metadata"][0][0] = 1
        rows_path.write_text(json.dumps(rows), encoding="utf-8")
        report_path = report_root / "application.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        from src.agents.artifacts import _fingerprint
        digest = _fingerprint(rows_path)
        report["partitions"]["validation"]["row_contract"] = {"size_bytes": digest["size"], "sha256": digest["sha256"]}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        self.assertIsNone(read_application(report_root, applied.contract, applied.code_id))

    def test_large_integer_feature_warning_and_version_downgrade(self):
        self.profile = replace(self.profile, roles=replace(self.profile.roles, numeric=("x", "large_id"), identifiers=("id",)))
        train = self.train()
        result = candidate_cli.create_from_tabular("candidate", train, self.rules, self.limits, 1024 * 1024, confirmed=True)
        self.assertEqual(result["input_warnings"], [{"code": "FLOAT64_PRECISION_POSSIBLE", "column": "large_id"}])
        tool = candidate_cli.open_run("candidate")
        manager = candidate_cli.executor(tool)
        self.assertEqual(self.drive(tool, manager).status, "completed")
        evidence = read_candidate_evidence(tool.states.directory, "synthetic_actor")
        path = evidence.directory / "candidate.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["schema_version"] = "preparation_candidate_report_v2"
        path.write_text(json.dumps(report), encoding="utf-8")
        self.assertIsNone(read_candidate(evidence.directory, tool.contract, tool.code_id))

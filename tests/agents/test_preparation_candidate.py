"""Train 후보 산출·승인·격리 파일 계약을 합성 입력으로 확인한다. OS 운송은 모의한다."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.agents import test_partition_export as fixtures
from src.agents.partition_export_cli import create as create_export, open_run as open_export, executor as export_executor
from src.agents.partition_export_worker import execute_request as export_worker
from src.agents.preparation_candidate import PreparationCandidateTool
from src.agents.preparation_candidate_cli import create, open_run, executor, status
from src.agents.preparation_candidate_contract import PreparationCandidateContract, saved_export_scope
from src.agents.preparation_candidate_results import read_candidate
from src.agents.preparation_candidate_transport import WORKER_FILES, ENTRY
from src.agents.preparation_candidate_worker import execute_request
from src.agents.partition_input_contract import bind_partition_input
from src.agents.split_evidence import read_committed_artifact
from src.agents.state import LocalStateStore
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.transform import PreparationRules, TabularPreprocessor, build_preparation


class PreparationCandidateTest(unittest.TestCase):
    fixture_commit = fixtures.PartitionExportTest.fixture_commit
    complete = fixtures.PartitionExportTest.complete

    def setUp(self):
        fixtures.PartitionExportTest.setUp(self)
        self.split_run = Path(self.split.artifact.run_directory)
        (self.split_run / "profile.json").write_text(json.dumps(self.profile.to_dict()), encoding="utf-8")
        self.scope_path = self.root / "split_scope.json"
        self.scope_path.write_text(self.split.artifact.scope_json, encoding="utf-8")
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(self.profile.to_dict()), encoding="utf-8")
        self.export_run = self.root / "export_runs/example"
        self.export_run.parent.mkdir()

        with patch("src.agents.partition_export_cli._directory", return_value=self.export_run), patch("src.agents.partition_export_cli._current_actor", return_value="synthetic_actor"):
            create_export("example", self.split_run, self.scope_path, self.root / "snapshot", self.profile_path,
                          WorkerLimits(60, 2048, 2), 65536, confirmed=True)
            tool = open_export("example")
            manager = export_executor(tool)
            manager.run_next()
            manager.run_next()
            approval = manager.approvals.record_confirmed_approval(manager.scope_for("ingest"))

            with patch("src.agents.partition_export.run_export_worker", side_effect=lambda request, root, signal: self.transport(request, root, signal, export_worker)):
                self.assertEqual(manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.export = read_committed_artifact(self.export_run, saved_export_scope(self.export_run))
        self.train = bind_partition_input(self.split, self.export, "train", self.roles)
        self.rules = PreparationRules(missing_ratio_threshold=0.5, drop_constant=True)
        self.rules_path = self.root / "rules.json"
        self.rules_path.write_text(json.dumps(asdict(self.rules)), encoding="utf-8")
        self.directory = self.root / "candidate_runs/example"
        self.directory.parent.mkdir()
        directory_patch = patch("src.agents.preparation_candidate_cli._directory", return_value=self.directory)
        actor_patch = patch("src.agents.preparation_candidate_cli._current_actor", return_value="synthetic_actor")
        directory_patch.start()
        actor_patch.start()
        self.addCleanup(directory_patch.stop)
        self.addCleanup(actor_patch.stop)
        self.calls = 0

    def setup_run(self, maximum=65536):
        result = create("example", self.split_run, self.scope_path, self.export_run, self.rules_path,
                        WorkerLimits(60, 2048, 2), maximum, confirmed=True)
        self.assertFalse(result["worker_started"])
        self.tool = open_run("example")
        self.manager = executor(self.tool)

    def transport(self, request, root, signal, worker=execute_request):
        reply = worker(request, root)
        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes},
        }}
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
        return outcome

    def run_candidate(self, transport=None):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.preparation_candidate.run_candidate_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))

        with patch("src.agents.preparation_candidate.run_candidate_worker", side_effect=transport or self.transport):
            result = self.manager.run_next(approval_id=approval.approval_id)

        task = self.manager.approvals.get_task(self.manager.scope_for("derive_preparation"))
        return result, self.manager._output(task)

    def test_train_only_candidate_without_transform(self):
        """Train에서만 fit하고 transform·모델 학습·Validation 전달 없이 후보를 커밋한다."""
        self.setup_run()
        original = TabularPreprocessor.fit
        fitted = []

        def capture(model, frame, y=None):
            fitted.append(frame.copy())
            self.assertIsNone(y)
            return original(model, frame, y)

        with patch.object(TabularPreprocessor, "fit", capture), patch.object(TabularPreprocessor, "transform", side_effect=AssertionError("적용 금지")):
            result, output = self.run_candidate()

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(fitted), 1)
        self.assertEqual(len(fitted[0]), len(self.train.row_positions))
        self.assertEqual(set(fitted[0]), set(self.roles.feature_columns))
        payload = json.loads((output / "candidate/preprocessor.json").read_text(encoding="utf-8"))
        numeric = next(record for record in payload["state"]["records"] if record["column"] == "x")
        self.assertEqual(numeric["impute"], float(np.median(self.train.row_positions)))
        self.assertFalse(status(self.tool)["transformed"])
        self.assertEqual({path.name for path in (output / "candidate").iterdir()}, {"preprocessor.json", "candidate.json", "candidate.md"})
        request = self.tool._request(output.name)
        root = self.tool._work_root(output.name)
        self.assertEqual({path.name for path in (root / "input").iterdir()}, {"snapshot", "candidate_task.json", "request.json"})
        self.assertFalse(any("validation" in value or "test" in value for value in request.input_files))

    def test_validation_input_is_rejected(self):
        validation = bind_partition_input(self.split, self.export, "validation", self.roles)

        with self.assertRaises(ValueError):
            PreparationCandidateContract(validation, self.rules, WorkerLimits())

    def test_changed_rules_reject_previous_approval(self):
        self.setup_run()
        self.manager.run_next()
        self.manager.run_next()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))
        changed = PreparationCandidateTool(self.tool.states, replace(self.tool.contract, rules=replace(self.rules, numeric_imputation="mean")))

        with patch("src.agents.preparation_candidate.run_candidate_worker") as launch:
            result = executor(changed).run_next(approval_id=approval.approval_id)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        launch.assert_not_called()

    def test_size_limit_blocks_partial_candidate(self):
        self.setup_run(maximum=32)
        result, output = self.run_candidate()
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_parent_rejects_tampered_candidate(self):
        self.setup_run()

        def tampered(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / "candidate/candidate.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["transformed"] = True
            path.write_text(json.dumps(report), encoding="utf-8")
            return outcome

        result, output = self.run_candidate(tampered)
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_missing_resource_evidence_is_rejected(self):
        self.setup_run()

        def missing(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["profile_cleanup"] = False
            return outcome

        result, output = self.run_candidate(missing)
        self.assertEqual(result.status, "failed_runtime")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_parent_never_refits(self):
        self.setup_run()
        result, output = self.run_candidate()
        self.assertEqual(result.status, "completed")

        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("부모 재산출 금지")), patch("pandas.read_csv", side_effect=AssertionError("부모 원본 파싱 금지")):
            self.assertIsNotNone(read_candidate(output / "candidate", self.tool.contract, self.tool.code_id))

    def test_exact_threshold_and_constant_policy(self):
        """결측률 50%는 유지하며 초과·상수만 후보 제거하고 중앙값을 검증한다."""
        frame = pd.DataFrame({"half": [1, 3, np.nan, np.nan], "over": [2, np.nan, np.nan, np.nan], "constant": [5] * 4})
        roles = type(self.roles)(numeric=("half", "over", "constant"))
        pipeline = build_preparation(roles, self.rules).fit(frame)
        state = pipeline.named_steps["preprocess"].state_
        self.assertEqual(state["records"][0]["column"], "half")
        self.assertEqual(state["records"][0]["impute"], 2)
        self.assertEqual({item["column"] for item in state["removed"]}, {"over", "constant"})

    def test_copied_worker_bootstraps(self):
        self.setup_run()
        request = self.tool._request("a" * 32)
        root = self.tool._stage(request)
        project = Path(__file__).resolve().parents[2]

        for name in WORKER_FILES:
            target = root / "tools" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        shutil.copyfile(project / ENTRY, root / "tools/worker.py")
        result = subprocess.run([sys.executable, "-I", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                                cwd=root, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))


    def test_previous_approval_database_is_read_only(self):
        """후보 생성이 선행 분할·내보내기 승인 DB를 갱신하지 않는다."""
        databases = [self.split_run / "approvals.sqlite3", self.export_run / "approvals.sqlite3"]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in databases]
        self.setup_run()
        after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in databases]
        self.assertEqual(before, after)

    def test_cancel_during_derivation_never_commits(self):
        """후보 파일이 만들어졌더라도 취소가 도착하면 완료로 등록하지 않는다."""
        from src.agents.isolated_eda_cli import cancel

        self.setup_run()

        def cancelled(request, root, signal):
            outcome = self.transport(request, root, signal)
            cancel(self.tool)
            return outcome

        result, output = self.run_candidate(cancelled)
        self.assertNotEqual(result.status, "completed")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_standard_scaling_statistics_use_train_only(self):
        """기존 스케일링 기준 산출을 재사용하고 적용 파일은 만들지 않는다."""
        rules = replace(self.rules, numeric_imputation="mean", scaling="standard")
        self.rules_path.write_text(json.dumps(asdict(rules)), encoding="utf-8")
        self.setup_run()
        result, output = self.run_candidate()
        self.assertEqual(result.status, "completed")
        payload = json.loads((output / "candidate/preprocessor.json").read_text(encoding="utf-8"))
        numeric = next(record for record in payload["state"]["records"] if record["column"] == "x")
        self.assertAlmostEqual(numeric["center"], float(np.mean(self.train.row_positions)))
        self.assertAlmostEqual(numeric["scale"], float(np.std(self.train.row_positions)))


if __name__ == "__main__":
    unittest.main()

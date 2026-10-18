"""저장 기준의 별도 승인·두 구간 적용·행 연결 검증. OS 운송만 모의한다."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.agents import test_preparation_candidate as fixtures
from src.agents.preparation_candidate_worker import execute_request as candidate_worker
from src.agents.preparation_apply_cli import create, open_run, executor, status, approve
from src.agents.preparation_apply_contract import read_candidate_evidence
from src.agents.preparation_apply import PreparationApplyTool
from src.agents.preparation_apply_results import read_application
from src.agents.preparation_apply_transport import WORKER_FILES, ENTRY
from src.agents.preparation_apply_worker import execute_request
from src.agents.worker_contract import WorkerLimits
from src.preprocessing.transform import load_preparation, TabularPreprocessor


class PreparationApplyTest(unittest.TestCase):
    fixture_commit = fixtures.PreparationCandidateTest.fixture_commit
    complete = fixtures.PreparationCandidateTest.complete

    def setUp(self):
        fixtures.PreparationCandidateTest.setUp(self)
        # 합성 text가 행마다 다르므로 정상 적용은 명시적인 unknown ignore 후보로 준비한다.
        self.rules = replace(self.rules, unknown_category="ignore")
        self.rules_path.write_text(json.dumps(asdict(self.rules)), encoding="utf-8")
        fixtures.PreparationCandidateTest.setup_run(self)
        result, _ = fixtures.PreparationCandidateTest.run_candidate(
            self, transport=lambda request, root, signal: self.transport(request, root, signal, candidate_worker))
        self.assertEqual(result.status, "completed")
        self.candidate_run = self.directory
        self.evidence = read_candidate_evidence(self.candidate_run, "synthetic_actor")
        self.directory = self.root / "apply_runs/example"
        self.directory.parent.mkdir()

        for target, value in (("_directory", self.directory), ("_current_actor", "synthetic_actor")):
            patcher = patch("src.agents.preparation_apply_cli." + target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def setup_apply(self, maximum=65536):
        self.assertFalse(create("example", self.candidate_run, WorkerLimits(60, 2048, 2), maximum, confirmed=True)["worker_started"])
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

    def run_apply(self, transport=None):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.preparation_apply.run_apply_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("apply_preparation"))

        with patch("src.agents.preparation_apply.run_apply_worker", side_effect=transport or self.transport):
            result = self.manager.run_next(approval_id=approval.approval_id)

        task = self.manager.approvals.get_task(self.manager.scope_for("apply_preparation"))
        return result, self.manager._output(task)

    def test_apply_without_refit_and_preserve_rows(self):
        """fit 없이 같은 상태로 변환하고 원본 행 위치·라벨·ID 문자열을 보존한다."""
        self.setup_apply()
        before = (self.evidence.directory / "preprocessor.json").read_bytes()

        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("재산출 금지")):
            result, output = self.run_apply()

        self.assertEqual(result.status, "completed")
        pipeline = load_preparation(self.evidence.directory / "preprocessor.json")

        for name, part in (("train", self.train), ("validation", self.tool.contract.validation)):
            frame = pd.read_csv(part.snapshot_directory / "input.bin", dtype={column: "string" for column in self.roles.categorical})
            expected = pipeline.transform(frame)
            actual = pd.read_csv(output / f"prepared/{name}_features.csv")
            self.assertEqual(list(actual), list(expected))
            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-14, atol=1e-14)
            rows = json.loads((output / f"prepared/{name}_rows.json").read_text(encoding="utf-8"))
            self.assertEqual(rows["row_positions"], list(part.row_positions))
            self.assertEqual(rows["metadata_columns"], ["label", "id"])
            self.assertEqual(rows["metadata"], [[self.rows[index][3], self.rows[index][1]] for index in part.row_positions])

        self.assertEqual((self.evidence.directory / "preprocessor.json").read_bytes(), before)
        self.assertFalse(status(self.tool)["refit"])
        self.assertFalse(status(self.tool)["test_used"])

    def test_parent_never_transforms_or_parses_inputs(self):
        self.setup_apply()
        result, output = self.run_apply()
        self.assertEqual(result.status, "completed")

        with patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("부모 재산출 금지")), patch.object(TabularPreprocessor, "transform", side_effect=AssertionError("부모 적용 금지")), patch("pandas.read_csv", side_effect=AssertionError("부모 입력 파싱 금지")):
            self.assertIsNotNone(read_application(output / "prepared", self.tool.contract, self.tool.code_id))

    def test_candidate_tampering_blocks_reopen(self):
        self.setup_apply()
        (self.evidence.directory / "preprocessor.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            open_run("example")

    def test_missing_approval_cannot_apply(self):
        self.setup_apply()
        self.manager.run_next()
        self.manager.run_next()

        with patch("src.agents.preparation_apply.run_apply_worker") as launch:
            result = self.manager.run_next(approval_id="0" * 32)

        self.assertNotEqual(result.status, "completed")
        launch.assert_not_called()

    def test_output_limit_never_commits(self):
        self.setup_apply(maximum=16)
        result, output = self.run_apply()
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_changed_limit_invalidates_approval(self):
        self.setup_apply()
        self.manager.run_next()
        self.manager.run_next()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("apply_preparation"))
        changed = PreparationApplyTool(self.tool.states, replace(self.tool.contract, max_partition_bytes=32768))

        with patch("src.agents.preparation_apply.run_apply_worker") as launch:
            result = executor(changed).run_next(approval_id=approval.approval_id)

        self.assertEqual(result.code, "PLAN_MISMATCH")
        launch.assert_not_called()

    def test_parent_rejects_result_tampering(self):
        self.setup_apply()

        def tampered(request, root, signal):
            outcome = self.transport(request, root, signal)
            (Path(request.output_directory) / "prepared/validation_features.csv").write_text("changed", encoding="utf-8")
            return outcome

        result, output = self.run_apply(tampered)
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_cancel_after_worker_never_commits(self):
        from src.agents.isolated_eda_cli import cancel

        self.setup_apply()

        def cancelled(request, root, signal):
            outcome = self.transport(request, root, signal)
            cancel(self.tool)
            return outcome

        result, output = self.run_apply(cancelled)
        self.assertNotEqual(result.status, "completed")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_missing_resource_evidence_is_rejected(self):
        self.setup_apply()

        def missing(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["profile_cleanup"] = False
            return outcome

        result, output = self.run_apply(missing)
        self.assertEqual(result.status, "failed_runtime")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_approval_is_interactive_and_explicit(self):
        self.setup_apply()
        self.manager.run_next()
        self.manager.run_next()

        with patch("os.isatty", return_value=False), self.assertRaises(ValueError):
            approve(self.tool)

        with patch("os.isatty", return_value=True), patch("secrets.token_hex", return_value="abcd1234"), patch("builtins.print"), patch("builtins.input", return_value="wrong"):
            self.assertEqual(approve(self.tool), {"code": "APPROVAL_DECLINED"})

    def test_copied_worker_bootstraps(self):
        self.setup_apply()
        request = self.tool._request("c" * 32)
        root = self.tool._stage(request)
        self.assertFalse(any("test" in value for value in request.input_files))
        project = Path(__file__).resolve().parents[2]

        for name in WORKER_FILES:
            target = root / "tools" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        shutil.copyfile(project / ENTRY, root / "tools/worker.py")
        result = subprocess.run([sys.executable, "-I", "-B", str(root / "tools/worker.py"), str(root), request.request_id],
                                cwd=root, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))


    def test_unknown_category_error_never_commits_partial_output(self):
        """새 범주를 거부하는 후보에서는 Validation 실패 시 Train 부분 결과도 커밋하지 않는다."""
        error_candidate = self.root / "candidate_runs/error_case"
        self.rules_path.write_text(json.dumps(asdict(replace(self.rules, unknown_category="error"))), encoding="utf-8")

        with patch("src.agents.preparation_candidate_cli._directory", return_value=error_candidate):
            fixtures.PreparationCandidateTest.setup_run(self)
            result, _ = fixtures.PreparationCandidateTest.run_candidate(
                self, transport=lambda request, root, signal: self.transport(request, root, signal, candidate_worker))
            self.assertEqual(result.status, "completed")

        self.candidate_run = error_candidate
        self.setup_apply()
        result, output = self.run_apply()
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())

    def test_parent_rejects_infinite_value_with_matching_hash(self):
        """해시를 함께 바꾸더라도 무한대 출력은 형식 검증에서 거부한다."""
        from src.agents.artifacts import _fingerprint

        self.setup_apply()

        def invalid_number(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / "prepared/validation_features.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            values = lines[1].split(",")
            values[0] = "inf"
            lines[1] = ",".join(values)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
            report_path = path.parent / "application.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            fingerprint = _fingerprint(path)
            report["partitions"]["validation"]["features"] = {"size_bytes": fingerprint["size"], "sha256": fingerprint["sha256"]}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            return outcome

        result, output = self.run_apply(invalid_number)
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse((output / "AGENT_COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()

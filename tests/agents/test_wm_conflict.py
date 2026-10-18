"""합성 충돌 맵으로 이전 검토 기록 결합·별도 승인·비파괴 후보 진단을 검사한다."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.wm_sandbox.export_canonical import CLASS_NAMES, export_bundle
from src.agents import wm_canonical_cli as quality_cli
from src.agents import wm_conflict_cli as candidate_cli
from src.agents.wm_canonical_contract import digest, inventory
from src.agents.wm_canonical_worker import execute_request as quality_worker
from src.agents.wm_conflict_worker import execute_request as candidate_worker
from src.agents.wm_conflict_transport import ENTRY, WORKER_FILES
from src.agents.worker_contract import WorkerLimits


class WmConflictTest(unittest.TestCase):
    def setUp(self):
        """각 클래스 4개와 라벨 충돌·미라벨 복사본만 사용한다. 실데이터는 읽지 않는다."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.quality_dir = self.root / "runs/quality"
        self.candidate_dir = self.root / "runs/candidate"
        self.quality_dir.parent.mkdir()

        for target, directory in (("wm_canonical_cli", self.quality_dir), ("wm_conflict_cli", self.candidate_dir)):
            mocking = patch(f"src.agents.{target}._directory", return_value=directory)
            mocking.start()
            self.addCleanup(mocking.stop)

        rows = []

        for label_index, label in enumerate(CLASS_NAMES):
            for repeat in range(4):
                number = label_index * 4 + repeat + 1
                array = np.ones((8, 8), dtype=np.uint8)

                for bit in range(6):
                    array.flat[bit] = 2 if number & (1 << bit) else 1

                rows.append({"waferMap": array, "failureType": label, "lotName": f"fixture_{number}"})

        rows.extend([
            {"waferMap": rows[0]["waferMap"].copy(), "failureType": CLASS_NAMES[1], "lotName": "conflict"},
            {"waferMap": rows[0]["waferMap"].copy(), "failureType": [], "lotName": "unlabeled"},
        ])
        export_bundle(pd.DataFrame(rows), self.root / "canonical", "a" * 64, 100,
                      execution_mode="synthetic_demo", max_output_bytes=1024 * 1024)
        quality_cli.create("quality", self.root / "canonical", WorkerLimits(120, 2048, 2),
                           1024 * 1024, 32 * 1024 * 1024, confirmed=True)
        self.quality = quality_cli.open_run("quality")
        manager = quality_cli.executor(self.quality)
        manager.run_next()
        self.assertEqual(manager.run_next().code, "APPROVAL_REQUIRED")
        approval = manager.approvals.record_confirmed_approval(manager.scope_for("run_eda"))

        with patch("src.agents.wm_canonical.run_wm_worker", side_effect=self.quality_transport):
            result = manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.code, "TOOL_REVIEW_REQUIRED", result)
        candidate_cli.create("candidate", "quality", WorkerLimits(120, 2048, 2), confirmed=True)
        self.tool = candidate_cli.open_run("candidate")
        self.manager = candidate_cli.executor(self.tool)

    @staticmethod
    def transport(worker, request, root, signal):
        """모의 OS 증거만 제공한다. manifest·SQL·출력 검증 코드는 그대로 실행한다."""
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

    def quality_transport(self, request, root, signal):
        return self.transport(quality_worker, request, root, signal)

    def candidate_transport(self, request, root, signal):
        return self.transport(candidate_worker, request, root, signal)

    def approval(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.wm_conflict.run_conflict_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))

    def test_candidate_preserves_original_and_review(self):
        """충돌 해시의 미라벨 행도 후보에 포함하되 원본·기존 검토 중단은 유지한다."""
        approval = self.approval()
        before = inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes)
        previous_state = self.quality.states.load()

        with patch("src.agents.wm_conflict.run_conflict_worker", side_effect=self.candidate_transport) as launch, \
                patch("numpy.load", side_effect=AssertionError("후보 진단에서 NPZ 파싱 금지")), \
                patch("pandas.read_pickle", side_effect=AssertionError("pickle 파싱 금지")):
            result = self.manager.run_next(approval_id=approval.approval_id)
            self.manager.run_next()

        self.assertEqual(result.status, "completed", result)
        launch.assert_called_once()
        report = candidate_cli.status(candidate_cli.open_run("candidate"))
        diagnosis = json.loads(Path(report["local_report"]).with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(diagnosis["candidate_conflict_hash_groups"], 1)
        self.assertEqual(diagnosis["candidate_excluded_manifest_rows"], 3)
        self.assertEqual(diagnosis["hypothetical_exclusion"]["split_assessment"]["status"], "preconditions_passed")
        rows_file = Path(report["local_report"]).parents[1] / "excluded_rows.csv"
        self.assertEqual(rows_file.read_text().splitlines(), ["source_row_position", "0", "36", "37"])
        self.assertFalse(report["data_modified"])
        self.assertFalse(report["exclusion_approved"])
        self.assertEqual(self.quality.states.load(), previous_state)
        self.assertEqual(inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes), before)

    def test_changed_prior_receipt_rejected(self):
        """새 후보가 기존 품질 증거를 건너뛰거나 변경된 결과를 사용하지 않는다."""
        receipt = next(self.quality_dir.glob("attempts/*/receipt.json"))
        receipt.write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            candidate_cli.open_run("candidate")

    def test_tampered_output_never_commits(self):
        """worker 성공과 별개로 후보 행 위치가 바뀌면 완료 처리하지 않는다."""
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.candidate_transport(request, root, signal)
            (Path(request.output_directory) / "candidate/excluded_rows.csv").write_bytes(b"source_row_position\n1\n")
            return outcome

        with patch("src.agents.wm_conflict.run_conflict_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("derive_preparation", self.tool.states.load().artifact_refs)

    def test_missing_cleanup_never_commits(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.candidate_transport(request, root, signal)
            del outcome["launch"]["resources"]["cleanup_confirmed"]
            return outcome

        with patch("src.agents.wm_conflict.run_conflict_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")
        self.assertNotIn("derive_preparation", self.tool.states.load().artifact_refs)

    def test_changed_snapshot_blocks_before_manifest(self):
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        next((root / "input/canonical/maps").iterdir()).write_bytes(b"changed")

        with patch("src.wafer_dl.validate_canonical.validate_manifest", side_effect=AssertionError("변조 파싱 금지")):
            reply = candidate_worker(request, root)

        self.assertEqual(reply.status, "blocked_input")

    def test_baseline_mismatch_and_output_limit_rejected(self):
        """입력이 같아도 기준 집계 불일치·출력 상한 초과는 부분 후보로 종료한다."""
        for index, key in enumerate(("quality_evidence", "max_output_bytes")):
            request = self.tool._request(str(index + 1) * 32)
            root = self.tool._stage(request)
            path = root / "input/analysis.json"
            packet = json.loads(path.read_text())

            if key == "quality_evidence":
                packet[key]["row_count"] += 1
            else:
                packet[key] = 1

            request = replace(request, config_id=digest(packet))
            path.write_text(json.dumps(packet), encoding="utf-8")
            reply = candidate_worker(request, root)
            self.assertEqual(reply.status, "blocked_input")
            self.assertFalse((Path(request.output_directory) / "candidate/candidate.json").exists())

    def test_cancel_before_worker(self):
        approval = self.approval()
        candidate_cli.cancel(self.tool)

        with patch("src.agents.wm_conflict.run_conflict_worker") as launch:
            result = self.manager.run_next(approval_id=approval.approval_id)

        launch.assert_not_called()
        self.assertNotEqual(result.status, "completed")

    def test_copied_worker_bootstraps(self):
        """프로젝트가 없는 격리 복사본에서도 허용 목록만으로 진단 코어를 로딩한다."""
        copied = self.root / "copied"
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        script = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
                  "from src.agents.wm_conflict_worker import execute_request; "
                  "from src.agents.wm_conflict_results import inspect_candidate; "
                  "from src.wafer_dl.diagnose_split import compare_conflict_exclusion")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(copied),
                                 str(Path(sys.prefix) / "Lib/site-packages")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

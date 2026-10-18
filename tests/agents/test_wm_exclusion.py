"""합성 충돌 후보를 새 canonical로 적용하는 승인·재검증·원본 보존 검사."""

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

from tests.agents import test_wm_conflict as fixtures
from src.agents import wm_conflict_cli as candidate_cli
from src.agents import wm_exclusion_cli as apply_cli
from src.agents.wm_canonical_contract import digest, inventory
from src.agents.wm_exclusion_worker import execute_request
from src.agents.wm_exclusion_transport import WORKER_FILES, ENTRY
from src.agents.worker_contract import WorkerLimits
from src.wafer_dl.validate_canonical import validate_bundle


class WmExclusionTest(unittest.TestCase):
    transport = staticmethod(fixtures.WmConflictTest.transport)
    quality_transport = fixtures.WmConflictTest.quality_transport
    candidate_transport = fixtures.WmConflictTest.candidate_transport

    def setUp(self):
        """직전 품질·후보 기록은 합성 입력만으로 만들고 실제 사용자 기록은 사용하지 않는다."""
        fixtures.WmConflictTest.setUp(self)
        self.manager.run_next()
        self.manager.run_next()
        approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))

        with patch("src.agents.wm_conflict.run_conflict_worker", side_effect=self.candidate_transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")

        self.source_candidate = self.tool
        self.apply_dir = self.root / "runs/apply"
        mocking = patch("src.agents.wm_exclusion_cli._directory", return_value=self.apply_dir)
        mocking.start()
        self.addCleanup(mocking.stop)
        apply_cli.create("apply", "candidate", WorkerLimits(120, 2048, 2), 32 * 1024 * 1024, confirmed=True)
        self.tool = apply_cli.open_run("apply")
        self.manager = apply_cli.executor(self.tool)

    def approval(self):
        self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.wm_exclusion.run_exclusion_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("derive_preparation"))

    def apply_transport(self, request, root, signal):
        return self.transport(execute_request, request, root, signal)

    def test_apply_revalidates_and_preserves_maps_and_source(self):
        """3개 충돌 행만 제외하고 남은 배열·라벨·원본 대응과 품질을 재검증한다."""
        approval = self.approval()
        before = inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes)
        candidate_state = self.source_candidate.states.load()

        with patch("src.agents.wm_exclusion.run_exclusion_worker", side_effect=self.apply_transport) as launch, \
                patch("pandas.read_pickle", side_effect=AssertionError("pickle 재읽기 금지")):
            result = self.manager.run_next(approval_id=approval.approval_id)
            self.manager.run_next()

        self.assertEqual(result.status, "completed", result)
        launch.assert_called_once()
        status = apply_cli.status(apply_cli.open_run("apply"))
        canonical = Path(status["local_canonical"])
        report = validate_bundle(canonical)
        self.assertEqual(report["row_count"], 35)
        self.assertEqual(report["split_assessment"]["status"], "preconditions_passed")
        self.assertEqual(report["duplicate_audit"]["map_label_conflicts"], 0)
        self.assertFalse(status["original_modified"])
        self.assertFalse(status["training_ready"])
        self.assertEqual(self.source_candidate.states.load(), candidate_state)
        self.assertEqual(inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes), before)
        parent = Path(self.tool.contract.snapshot_directory)
        original_rows = [json.loads(line) for line in (parent / "manifest.jsonl").read_text().splitlines()]
        rows = [json.loads(line) for line in (canonical / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual([row["parent_source_row_position"] for row in rows], list(range(1, 36)))

        for position, row in enumerate(rows):
            original = original_rows[row["parent_source_row_position"]]
            self.assertEqual(row["source_row_position"], position)
            self.assertEqual(row["parent_sample_id"], original["sample_id"])
            self.assertEqual(row["pattern_label"], original["pattern_label"])
            self.assertEqual(row["map_hash"], original["map_hash"])

            with np.load(parent / original["map_ref"]["shard"], allow_pickle=False) as old, \
                    np.load(canonical / row["map_ref"]["shard"], allow_pickle=False) as new:
                np.testing.assert_array_equal(old[original["map_ref"]["key"]], new[row["map_ref"]["key"]])

    def test_changed_candidate_receipt_rejected(self):
        next(self.candidate_dir.glob("attempts/*/receipt.json")).write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            apply_cli.open_run("apply")

    def test_wrong_exclusion_and_output_cap_rejected(self):
        """검토한 후보 행 지문과 일치하지 않거나 출력이 초과하면 완료하지 않는다."""
        for number, change in enumerate(("selection", "limit"), 1):
            request = self.tool._request(str(number) * 32)
            root = self.tool._stage(request)
            path = root / "input/analysis.json"
            packet = json.loads(path.read_text())

            if change == "selection":
                evidence = packet["candidate_evidence"]
                evidence["selection_files"]["excluded_rows.csv"]["sha256"] = "0" * 64
                evidence["candidate_evidence_id"] = digest({key: value for key, value in evidence.items()
                                                            if key != "candidate_evidence_id"})
            else:
                packet["max_output_bytes"] = 1

            request = replace(request, config_id=digest(packet))
            path.write_text(json.dumps(packet), encoding="utf-8")
            reply = execute_request(request, root)
            self.assertEqual(reply.status, "blocked_input")
            self.assertFalse((Path(request.output_directory) / "derived/result.json").exists())

    def test_tampered_derived_never_commits(self):
        approval = self.approval()

        def tampered(request, root, signal):
            outcome = self.apply_transport(request, root, signal)
            (Path(request.output_directory) / "derived/canonical/manifest.jsonl").write_bytes(b"changed")
            return outcome

        with patch("src.agents.wm_exclusion.run_exclusion_worker", side_effect=tampered):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("derive_preparation", self.tool.states.load().artifact_refs)

    def test_missing_cleanup_rejected(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.apply_transport(request, root, signal)
            outcome["launch"]["resources"]["cleanup_confirmed"] = False
            return outcome

        with patch("src.agents.wm_exclusion.run_exclusion_worker", side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")
        self.assertNotIn("derive_preparation", self.tool.states.load().artifact_refs)

    def test_cancel_before_worker(self):
        approval = self.approval()
        apply_cli.cancel(self.tool)

        with patch("src.agents.wm_exclusion.run_exclusion_worker") as launch:
            result = self.manager.run_next(approval_id=approval.approval_id)

        launch.assert_not_called()
        self.assertNotEqual(result.status, "completed")

    def test_array_revalidation_failure_never_completes(self):
        """새 배열 바이트가 손상되면 결과 완료 표식과 커밋을 만들지 않는다."""
        from src.wafer_dl.exclude_conflicts import derive_canonical

        request = self.tool._request("e" * 32)
        root = self.tool._stage(request)

        def corrupted(source, output, workspace, evidence, maximum):
            result = derive_canonical(source, output, workspace, evidence, maximum)
            next((output / "maps").iterdir()).write_bytes(b"invalid zip")
            return result

        with patch("src.wafer_dl.exclude_conflicts.derive_canonical", side_effect=corrupted):
            reply = execute_request(request, root)

        self.assertEqual(reply.status, "blocked_input")
        self.assertFalse((Path(request.output_directory) / "derived/result.json").exists())

    def test_existing_output_not_overwritten(self):
        """코어를 직접 호출해도 기존 폴더나 원본 하위 경로를 수정하지 않는다."""
        from src.wafer_dl.exclude_conflicts import derive_canonical

        parent = Path(self.tool.contract.snapshot_directory)
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_bytes(b"preserved")

        for target in (output, parent / "nested"):
            with self.assertRaises(ValueError):
                derive_canonical(parent, target, self.root, self.tool.evidence, 1024 * 1024)

        self.assertEqual(sentinel.read_bytes(), b"preserved")

    def test_copied_worker_bootstraps(self):
        copied = self.root / "copied"
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        script = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
                  "from src.agents.wm_exclusion_worker import execute_request; "
                  "from src.agents.wm_exclusion_results import inspect_derived; "
                  "from src.wafer_dl.exclude_conflicts import derive_canonical")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(copied),
                                 str(Path(sys.prefix) / "Lib/site-packages")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class WmExclusionManyShardsTest(unittest.TestCase):
    """작은 맵을 여러 shard에 저장해 실데이터의 큰 파일 지문 목록을 재현한다."""

    transport = staticmethod(WmExclusionTest.transport)
    quality_transport = WmExclusionTest.quality_transport
    candidate_transport = WmExclusionTest.candidate_transport
    apply_transport = WmExclusionTest.apply_transport
    approval = WmExclusionTest.approval

    def setUp(self):
        import pandas as pd
        from scripts.wm_sandbox.export_canonical import export_bundle

        def many_shards(frame, *args, **kwargs):
            # Lot·라벨은 그대로 반복하므로 새 충돌을 만들지 않고 파일 개수만 늘린다.
            kwargs["shard_rows"] = 1
            return export_bundle(pd.concat([frame] * 18, ignore_index=True), *args, **kwargs)

        with patch("tests.agents.test_wm_conflict.export_bundle", side_effect=many_shards):
            WmExclusionTest.setUp(self)

    def test_large_manifest_and_receipt_complete_and_reopen(self):
        """64KiB 초과 결과 목록·영수증도 승인·복사·내용 검증·커밋·재조회가 가능하다."""
        approval = self.approval()

        with patch("src.agents.wm_exclusion.run_exclusion_worker", side_effect=self.apply_transport):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "completed", result)
        status = apply_cli.status(apply_cli.open_run("apply"))
        derived = Path(status["local_canonical"]).parent
        self.assertGreater((derived / "result.json").stat().st_size, 64 * 1024)
        self.assertGreater((derived.parent / "receipt.json").stat().st_size, 64 * 1024)
        self.assertEqual(status["derived_status"], "derived_ready")
        self.assertTrue((derived.parent / "AGENT_COMPLETE.json").is_file())


class WmExclusionRecordBoundsTest(unittest.TestCase):
    def test_record_limit_and_strict_json_and_ipc_limit(self):
        """큰 파일 목록만 별도 상한을 적용하고 중복 키·NaN·과대 JSON·IPC 완화는 거부한다."""
        from src.agents.wm_exclusion_results import MAX_RESULT_RECORD_BYTES, read_result_record
        from src.agents.worker_contract import _decode

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            large = json.dumps({"payload": "x" * (70 * 1024)})
            path.write_text(large, encoding="utf-8")
            self.assertEqual(len(read_result_record(path)["payload"]), 70 * 1024)

            with self.assertRaises(ValueError):
                _decode(large)

            for invalid in ('{"x":1,"x":2}', '{"x":NaN}', '[]', ' ' * (MAX_RESULT_RECORD_BYTES + 1)):
                path.write_text(invalid, encoding="utf-8")

                with self.assertRaises(ValueError):
                    read_result_record(path)


if __name__ == "__main__":
    unittest.main()

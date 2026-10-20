"""고정 CSV 검사기의 worker·부모 판정. 실제 AppContainer는 모의한다."""

import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import uuid

from src.agents import csv_runtime_probe as probe
from src.agents.csv_runtime_probe_tasks import STAGES, fixture, prepare_task
from src.agents.csv_runtime_probe_checks import validate_reply, launch_passed, verify_outputs
from src.agents.worker_contract import WorkerLimits, WorkerRequest


def execute_fixture(task, family):
    from src.agents.snapshot_split_worker import execute_request as split
    from src.agents.partition_export_worker import execute_request as export
    from src.agents.preparation_candidate_worker import execute_request as candidate
    from src.agents.preparation_apply_worker import execute_request as apply

    request = WorkerRequest.from_json((task / "input/request.json").read_text(encoding="utf-8"))
    result = {"split": split, "export": export, "candidate": candidate, "apply": apply}[family](request, task)
    (task / "output/result/worker_result.json").write_text(result.to_json(), encoding="utf-8")
    (task / "output/probe_receipt.json").write_text(json.dumps({"schema_version": "csv_probe_child_v1",
        "probe_id": task.parent.parent.name, "request_id": request.request_id, "family": family,
        "isolated_token_and_job": True, "python": [3, 12, 15]}), encoding="utf-8")
    return result


def successful_launch(limits):
    return {"status": "exited", "exit_code": 0, "resources": {"status": "succeeded",
        "cleanup_confirmed": True, "active_processes": 0, "configured_job_memory_bytes": limits.memory_mib * 1024 ** 2,
        "configured_process_limit": limits.processes}}


class CsvProbeCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name) / uuid.uuid4().hex
        cls.root.mkdir()
        (cls.root / "tasks").mkdir()
        cls.limits = WorkerLimits(60, 2048, 2)
        fixture(cls.root)
        cls.outputs, cls.tasks = {}, {}
        for stage in STAGES:
            task, family, request = prepare_task(cls.root, stage, cls.outputs, cls.limits)
            result = execute_fixture(task, family)
            if result.status != "succeeded":
                diagnostic = (task / "scratch/task/diagnostic.json").read_text(encoding="utf-8")
                raise AssertionError(stage + ": " + diagnostic)
            cls.outputs[stage] = task / "output/result"
            cls.tasks[stage] = (task, family, request)

    def test_worker_values_match_fixed_expectations(self):
        checks = verify_outputs(self.root, self.outputs)
        self.assertTrue(all(checks.values()), checks)

    def test_reply_requires_current_job_and_resource_cleanup(self):
        for stage, (task, family, request) in self.tasks.items():
            with self.subTest(stage=stage):
                launch = successful_launch(self.limits)
                self.assertTrue(validate_reply(self.root, task, family, request, launch))
                del launch["resources"]["cleanup_confirmed"]
                self.assertFalse(validate_reply(self.root, task, family, request, launch))

    def test_failed_or_unlimited_jobs_cannot_pass(self):
        for key, value in (("configured_process_limit", 99), ("active_processes", 1), ("configured_job_memory_bytes", 1)):
            launch = successful_launch(self.limits)
            launch["resources"][key] = value
            self.assertFalse(launch_passed(launch, self.limits))
        launch = successful_launch(self.limits)
        launch["status"] = "timed_out"
        self.assertFalse(launch_passed(launch, self.limits))

    def test_stage_skip_and_duplicate_fixture_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_task(self.root, "derive_preparation", {}, self.limits)
        with self.assertRaises(FileExistsError):
            fixture(self.root)

    def test_reply_from_other_request_is_rejected(self):
        task, family, request = self.tasks["audit"]
        receipt_path = task / "output/probe_receipt.json"
        original = receipt_path.read_bytes()
        receipt = json.loads(original)
        receipt["request_id"] = "different-request"
        try:
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertFalse(validate_reply(self.root, task, family, request, successful_launch(self.limits)))
        finally:
            receipt_path.write_bytes(original)

    def test_driver_cannot_run_without_appcontainer_token(self):
        from src.agents.csv_runtime_probe_worker import main

        task, family, request = self.tasks["audit"]
        before = list(sys.path)
        try:
            with patch("sys.argv", ["driver", str(task), request.request_id, family]), patch(
                    "src.agents.pickle_import_worker._isolated_process", return_value=False):
                self.assertEqual(main(), 2)
        finally:
            sys.path[:] = before


class CsvProbeParentTest(unittest.TestCase):
    def test_individual_copied_packages_bootstrap_without_project_import_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / uuid.uuid4().hex
            root.mkdir()
            (root / "tasks").mkdir()
            probe.copy_families(root, Path(__file__).resolve().parents[2])
            fixture(root)
            outputs = {}
            script = (
                "import sys,importlib; from pathlib import Path; "
                "tools=Path(sys.argv[1]); sys.path.insert(0,str(tools)); "
                "m=importlib.import_module('src.agents.'+sys.argv[2]+'_worker'); "
                "assert tools in Path(m.__file__).parents; "
                "from src.agents.worker_contract import WorkerRequest; "
                "task=Path(sys.argv[3]); "
                "q=WorkerRequest.from_json((task/'input/request.json').read_text(encoding='utf-8')); "
                "r=m.execute_request(q,task); print(r.status); "
                "sys.exit(0 if r.status=='succeeded' else 2)"
            )
            for stage in STAGES:
                task, family, request = prepare_task(root, stage, outputs, WorkerLimits(60, 2048, 2))
                result = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                    str(root / "tools" / family), probe.FAMILIES[family], str(task)],
                    capture_output=True, timeout=60, check=False)
                self.assertEqual(result.returncode, 0, stage + ": " + result.stderr.decode("utf-8", errors="replace"))
                outputs[stage] = task / "output/result"
            self.assertTrue(all(verify_outputs(root, outputs).values()))

    def test_parent_chain_and_cleanup_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / uuid.uuid4().hex
            root.mkdir()
            limits = WorkerLimits(60, 2048, 2)

            def launch(executable, arguments, environment, scratch, **kwargs):
                result = execute_fixture(Path(arguments[4]), arguments[6])
                self.assertEqual(result.status, "succeeded")
                return successful_launch(limits)

            with patch.object(probe, "check_environment"), patch.object(probe, "_new_root", return_value=root), patch.object(
                    probe, "runtime_files", return_value=[Path(sys.prefix) / "python.exe"]), patch.object(
                    probe, "_acl"), patch.object(probe, "ProbeProgress"), patch.object(probe, "AppContainer") as container:
                container.return_value.launch.side_effect = launch
                container.return_value.close.return_value = True
                _, result = probe.run_probe(limits)
                self.assertEqual(result["status"], "csv_compatibility_passed", result.get("failure_stage"))
                self.assertEqual(container.call_count, 7)
                self.assertTrue(all(job["profile_cleanup"] for job in result["jobs"]))
                self.assertFalse(result["production_worker_enabled"])

    def test_admin_denial_never_creates_probe_files(self):
        with patch.object(probe, "check_environment", side_effect=ValueError("admin required")), patch.object(probe, "_new_root") as create:
            with self.assertRaises(ValueError):
                probe.run_probe(WorkerLimits())
            create.assert_not_called()

    def test_cleanup_failure_keeps_review_and_stops_next_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / uuid.uuid4().hex
            root.mkdir()
            limits = WorkerLimits(60, 2048, 2)

            def launch(executable, arguments, environment, scratch, **kwargs):
                self.assertEqual(execute_fixture(Path(arguments[4]), arguments[6]).status, "succeeded")
                return successful_launch(limits)

            with patch.object(probe, "check_environment"), patch.object(probe, "_new_root", return_value=root), patch.object(
                    probe, "runtime_files", return_value=[Path(sys.prefix) / "python.exe"]), patch.object(
                    probe, "_acl"), patch.object(probe, "ProbeProgress"), patch.object(probe, "AppContainer") as container:
                container.return_value.launch.side_effect = launch
                container.return_value.close.return_value = False
                _, result = probe.run_probe(limits)
                self.assertEqual(result["status"], "review_required")
                self.assertFalse(result["profile_cleanup"])
                self.assertEqual(len(result["jobs"]), 1)
                self.assertEqual(container.call_count, 1)

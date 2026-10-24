"""입력 도구의 실제 실행기·승인 DB·worker와 부모 검증을 합성 입력으로 검사한다."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.agents import tabular_input_service as service
from src.agents import tabular_input_transport as native
from src.agents.artifacts import _fingerprint
from src.agents.input_snapshot import create_snapshot
from src.agents.tabular_input_worker import execute_request
from src.agents.tabular_input_transport import WORKER_FILES, CONTROL_FILES, ENTRY
from src.agents.tabular_source_contract import TabularSourceContract
from src.agents.worker_contract import WorkerLimits, WorkerResult
from src.preprocessing.tabular_loaders import CsvReadSpec


def transport(request, root, cancel):
    """OS 운송만 모의하고 고정 worker 계산·부모 검증·승인 소비는 실제로 수행한다."""
    reply = execute_request(request, root)
    (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
    outcome = {"request_id": request.request_id, "profile_cleanup": True,
               "launch": {"status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
                          "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                                        "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                                        "configured_process_limit": request.limits.processes}}}
    (Path(root) / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
    return outcome


def drive(item, name, authorize=None):
    """서비스의 공개 상태/승인/진행 API로만 완료 지점까지 간다."""
    for _ in range(5):
        state = item.status(name)
        if state["status"] in {"completed", "blocked_input", "failed_runtime", "security_locked"}:
            return state
        if state["status"] == "awaiting_approval":
            result = item.record_delegated_approval(name, item.preview_approval(name), authorize=authorize or (lambda _: None))
            state = item.status(name)
            item.next(name, expected_stage=state["stage"], expected_revision=state["revision"], approval_id=result["approval_id"])
        else:
            item.next(name, expected_stage=state["stage"], expected_revision=state["revision"])
    raise AssertionError("입력 도구의 단계가 끝나지 않았습니다.")


class TabularInputTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.csv"
        self.source.write_bytes(b"x,label\n1,a\n2,b\n")
        self.snapshot = create_snapshot(self.source, self.root / "snapshot", max_input_bytes=65536)
        self.limits = WorkerLimits(60, 2048, 2)
        self.spec = CsvReadSpec()
        self.inspect = service.TabularInputService("inspect", root=self.root / "runs")
        self.materialize = service.TabularInputService("materialize", root=self.root / "runs")
        self.mock = patch.object(service, "run_input_worker", side_effect=transport)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def create(self, item, name, inspection_run=None):
        return item.create(name, snapshot_directory=str(self.root / "snapshot"), snapshot=asdict(self.snapshot),
                           spec=self.spec.to_dict(), limits=asdict(self.limits), inspection_run=inspection_run,
                           max_output_bytes=8 * 1024 * 1024)

    def test_approval_preview_code_is_fingerprinted_but_not_copied(self):
        name = "src/agents/workflow_approval.py"
        self.assertIn(name, CONTROL_FILES)
        self.assertNotIn(name, WORKER_FILES)
        self.create(self.inspect, "inspect1")
        real = native._fingerprint

        def changed(path):
            result = real(path)
            return dict(result, sha256="0" * 64) if path.name == "workflow_approval.py" else result

        with patch.object(native, "_fingerprint", side_effect=changed), self.assertRaises(ValueError):
            self.inspect._open("inspect1")

    def ready_inspection(self):
        self.create(self.inspect, "inspect1")
        self.assertEqual(drive(self.inspect, "inspect1")["status"], "completed")
        return self.inspect.verified("inspect1")

    def test_inspect_and_materialize_consume_separate_approvals(self):
        self.ready_inspection()
        self.create(self.materialize, "materialize1", "inspect1")
        self.assertEqual(drive(self.materialize, "materialize1")["status"], "completed")
        source = self.materialize.source_contract("materialize1")
        self.assertTrue(source.check_input())
        self.assertEqual(source, TabularSourceContract.from_dict(source.to_dict()))
        self.assertEqual(source.snapshot.sha256, self.snapshot.sha256)
        for item, name in ((self.inspect, "inspect1"), (self.materialize, "materialize1")):
            _, tool, manager = item._open(name)
            task = manager.approvals.get_task(manager.scope_for("ingest"))
            self.assertEqual(task["status"], "committed")
            self.assertEqual(item.status(name)["actions"], [])
            self.assertTrue((tool.states.directory / "events.jsonl").read_text().find("DELEGATED_APPROVAL_RECORDED") >= 0)

    def test_worker_cannot_start_before_delegation_or_reuse_consumed_approval(self):
        self.create(self.inspect, "inspect1")
        state = self.inspect.status("inspect1")
        self.inspect.next("inspect1", expected_stage=state["stage"], expected_revision=state["revision"])
        state = self.inspect.status("inspect1")
        self.inspect.next("inspect1", expected_stage=state["stage"], expected_revision=state["revision"])
        state = self.inspect.status("inspect1")
        self.assertEqual(state["status"], "awaiting_approval")
        with patch.object(service, "run_input_worker", side_effect=AssertionError("미승인 실행")):
            self.inspect.next("inspect1", expected_stage=state["stage"], expected_revision=state["revision"])
        self.assertEqual(drive(self.inspect, "inspect1")["status"], "completed")
        with self.assertRaises(ValueError):
            self.inspect.next("inspect1", expected_stage="ingest", expected_revision=state["revision"])

    def test_materialize_rejects_noncommitted_or_changed_inspection(self):
        self.create(self.inspect, "inspect1")
        with self.assertRaises(ValueError):
            self.create(self.materialize, "materialize1", "inspect1")
        self.assertFalse((self.root / "runs/materialize/materialize1").exists())
        drive(self.inspect, "inspect1")
        folder, result = self.inspect.verified("inspect1")
        (folder / "inspection.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.create(self.materialize, "materialize1", "inspect1")

    def test_parent_rejects_tampered_worker_output_and_cleanup_claims(self):
        def altered(request, root, cancel):
            result = transport(request, root, cancel)
            result["launch"]["resources"]["active_processes"] = 1
            return result

        self.create(self.inspect, "inspect1")
        with patch.object(service, "run_input_worker", side_effect=altered):
            self.assertEqual(drive(self.inspect, "inspect1")["status"], "failed_runtime")
        with self.assertRaises(ValueError):
            self.inspect.verified("inspect1")

    def test_review_result_is_committed_but_not_materializable(self):
        self.source.write_bytes(b"x,x\n1,2\n")
        self.snapshot = create_snapshot(self.source, self.root / "duplicate", max_input_bytes=65536)
        item = service.TabularInputService("inspect", root=self.root / "runs")
        item.create("duplicate1", snapshot_directory=str(self.root / "duplicate"), snapshot=asdict(self.snapshot),
                    spec=self.spec.to_dict(), limits=asdict(self.limits))
        self.assertEqual(drive(item, "duplicate1")["status"], "completed")
        self.assertEqual(item.verified("duplicate1")[1]["inspection"]["status"], "review_required")
        with self.assertRaises(ValueError):
            self.materialize.create("materialize1", snapshot_directory=str(self.root / "duplicate"), snapshot=asdict(self.snapshot),
                                    spec=self.spec.to_dict(), limits=asdict(self.limits), inspection_run="duplicate1")

    def test_large_inspection_is_sidecar_not_worker_ipc(self):
        payload = ",".join(f"column_{i}" for i in range(600)) + "\n" + ",".join("1" for _ in range(600)) + "\n"
        self.source.write_text(payload, encoding="utf-8")
        self.snapshot = create_snapshot(self.source, self.root / "wide", max_input_bytes=65536)
        self.inspect.create("wide1", snapshot_directory=str(self.root / "wide"), snapshot=asdict(self.snapshot),
                            spec=self.spec.to_dict(), limits=asdict(self.limits))
        drive(self.inspect, "wide1")
        folder, result = self.inspect.verified("wide1")
        self.assertGreater((folder / "inspection.json").stat().st_size, 65536)
        self.materialize.create("wide2", snapshot_directory=str(self.root / "wide"), snapshot=asdict(self.snapshot),
                                spec=self.spec.to_dict(), limits=asdict(self.limits), inspection_run="wide1")
        _, tool, _ = self.materialize._open("wide2")
        request = tool._request("a" * 32)
        self.assertTrue(any(name.endswith("inspection.json") for name in request.input_files))
        self.assertLess(len(json.dumps(tool.arguments).encode("utf-8")), 65536)

    def test_worker_rejects_changed_paths_and_packet(self):
        self.create(self.inspect, "inspect1")
        _, tool, _ = self.inspect._open("inspect1")
        request = tool._request("a" * 32)
        root = tool._stage(request)
        reply = execute_request(replace(request, stage="run_eda", action_id="run_eda"), root)
        self.assertEqual(reply.status, "blocked_input")
        self.assertFalse((Path(request.output_directory) / "operation").exists())

    def test_worker_does_not_stat_ancestors_outside_granted_root(self):
        self.create(self.inspect, "inspect1")
        _, tool, _ = self.inspect._open("inspect1")
        request = tool._request("a" * 32)
        root = tool._stage(request)
        real_lstat = Path.lstat
        forbidden = {root, *root.parents}

        def restricted(path, *args, **kwargs):
            if path in forbidden:
                raise PermissionError("모의 허용 범위 밖")
            return real_lstat(path, *args, **kwargs)

        with patch.object(Path, "lstat", restricted):
            reply = execute_request(request, root)
        self.assertEqual(reply.status, "succeeded")

    def test_worker_output_changes_are_never_committed(self):
        for attack in ("extra", "binding", "counts", "completion", "hardlink"):
            name = "attack_" + attack
            self.create(self.inspect, name)

            def altered(request, root, cancel):
                result = transport(request, root, cancel)
                folder = Path(request.output_directory) / "operation"
                if attack == "extra":
                    (folder / "extra.txt").write_text("x", encoding="utf-8")
                elif attack == "completion":
                    (folder / "AGENT_COMPLETE.json").write_text("{}", encoding="utf-8")
                elif attack == "hardlink":
                    original = folder / "inspection.json"
                    backup = root / "scratch/linked.json"
                    backup.hardlink_to(original)
                else:
                    target = folder / ("binding.json" if attack == "binding" else "inspection.json")
                    data = json.loads(target.read_text(encoding="utf-8"))
                    if attack == "binding":
                        data["packet"]["max_output_bytes"] += 1
                    else:
                        data["columns"][0]["missing_count"] += 1
                    target.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                return result

            with self.subTest(attack=attack), patch.object(service, "run_input_worker", side_effect=altered):
                state = drive(self.inspect, name)
                self.assertNotEqual(state["status"], "completed")
                with self.assertRaises(ValueError):
                    self.inspect.verified(name)

    def test_cancelled_execution_never_runs_worker(self):
        self.create(self.inspect, "cancel1")
        _, tool, _ = self.inspect._open("cancel1")
        output = self.root / ("b" * 32)
        output.mkdir()
        with patch.object(tool.cancel, "is_set", return_value=True), patch.object(service, "run_input_worker") as run:
            self.assertEqual(tool.execute(output, tool.arguments).status, "blocked_input")
            run.assert_not_called()

    def test_worker_rejects_changed_settings_even_with_matching_config_hash(self):
        self.create(self.inspect, "inspect1")
        _, tool, _ = self.inspect._open("inspect1")
        request = tool._request("a" * 32)
        root = tool._stage(request)
        path = root / "input/settings.json"
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet["read_spec"]["converter"] = "os.system"
        path.write_text(json.dumps(packet), encoding="utf-8")
        from src.agents.snapshot_eda import _digest
        changed = replace(request, config_id=_digest(packet))
        self.assertEqual(execute_request(changed, root).status, "blocked_input")
        self.assertFalse((root / "output/result/operation").exists())

    def test_materialize_requires_exact_snapshot_spec_and_limits(self):
        self.ready_inspection()
        for key, value in (("spec", dict(self.spec.to_dict(), empty_field="preserve")),
                           ("limits", dict(asdict(self.limits), memory_mib=1024))):
            args = {"snapshot_directory": str(self.root / "snapshot"), "snapshot": asdict(self.snapshot),
                    "spec": self.spec.to_dict(), "limits": asdict(self.limits), "inspection_run": "inspect1"}
            args[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.materialize.create("wrong_" + key, **args)
            self.assertFalse((self.root / "runs/materialize" / ("wrong_" + key)).exists())

    def test_copied_worker_imports_without_parent_service_or_repository(self):
        self.create(self.inspect, "inspect1")
        _, tool, _ = self.inspect._open("inspect1")
        request = tool._request("a" * 32)
        root = tool._stage(request)
        project = Path(__file__).resolve().parents[2]
        for name in WORKER_FILES:
            target = root / "tools" / ("worker.py" if name == ENTRY else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)
        script = ("import sys,runpy,pathlib; "
                  "sys.path[:0]=[sys.argv[1],sys.argv[2]]; "
                  "ns=runpy.run_path(sys.argv[1]+'/worker.py'); "
                  "from src.agents.worker_contract import WorkerRequest; "
                  "root=pathlib.Path(sys.argv[3]); "
                  "r=WorkerRequest.from_json((root/'input/request.json').read_text(encoding='utf-8')); "
                  "reply=ns['execute_request'](r,root); "
                  "assert reply.status=='succeeded',reply; "
                  "assert 'src.agents.tabular_input_service' not in sys.modules; "
                  "assert 'src.agents.approval_store' not in sys.modules; print('COPIED_WORKER_OK')")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script, str(root / "tools"),
                                 str(Path(sys.prefix) / "Lib/site-packages"), str(root)], cwd=root,
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("COPIED_WORKER_OK", result.stdout)

"""합성 WM 입력으로 승인·격리 호출·canonical 검증·실패 차단을 검사한다."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents import pickle_import_cli as import_cli
from src.agents.pickle_import_cli import create, executor, open_run, status
from src.agents.pickle_import_worker import execute_request
from src.agents.pickle_import_transport import WORKER_FILES, ENTRY
from src.agents.pickle_results import inspect_canonical
from src.agents.worker_contract import WorkerLimits


class WmPickleImportTest(unittest.TestCase):
    def setUp(self):
        """원본 pickle 대신 바이트 placeholder와 모의 로더의 합성 DataFrame을 사용한다."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "runs/example"
        self.directory.parent.mkdir()
        self.source = self.root / "synthetic.pkl"
        self.source.write_bytes(b"synthetic placeholder: never deserialize in unit tests")
        self.frame = pd.DataFrame({
            "waferMap": [np.array([[1, 2]], dtype=np.uint8)],
            "failureType": ["Center"], "lotName": ["fixture_lot"],
        })

        for target, value in (
            ("src.agents.pickle_import_cli._directory", self.directory),
            ("src.agents.pickle_import_cli._current_actor", "synthetic_actor"),
        ):
            mocking = patch(target, return_value=value)
            mocking.start()
            self.addCleanup(mocking.stop)

        # 작업 생성 시 부모가 원본을 역직렬화하는 회귀를 즉시 탐지한다.
        with patch("pandas.read_pickle", side_effect=AssertionError("부모 역직렬화 금지")):
            result = create("example", self.source, WorkerLimits(60, 2048, 2), 65536, 1024 * 1024, confirmed=True)

        self.assertFalse(result["worker_started"])
        self.tool = open_run("example")
        self.manager = executor(self.tool)

    def transport(self, request, root, signal):
        """OS 전송만 모의하고 worker 계산·숫자 출력·부모 검증은 실제 구현을 사용한다."""
        with patch("pandas.read_pickle", return_value=self.frame) as reader:
            reply = execute_request(request, root)

        if reply.status == "succeeded":
            reader.assert_called_once_with(root / "input/snapshot/input.bin", compression=None)

        (Path(request.output_directory) / "worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        outcome = {"request_id": request.request_id, "profile_cleanup": True, "launch": {
            "status": "exited", "exit_code": 0 if reply.status == "succeeded" else 2,
            "resources": {"status": "succeeded", "cleanup_confirmed": True, "active_processes": 0,
                          "configured_job_memory_bytes": request.limits.memory_mib * 1024 * 1024,
                          "configured_process_limit": request.limits.processes},
        }}
        (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

        return outcome

    def approval(self):
        # 계산·승인 흐름 검사에서만 지원 OS 조건을 모의한다.
        # 전역 sys.version_info는 바꾸지 않고 worker 실행 전 모의를 해제한다.
        with patch.object(import_cli, "os", SimpleNamespace(name="nt")), patch.object(
            import_cli, "sys", SimpleNamespace(version_info=(3, 10, 0))
        ):
            self.assertEqual(self.manager.run_next().code, "STEP_COMPLETED")

        with patch("src.agents.pickle_import.run_import_worker") as launch:
            self.assertEqual(self.manager.run_next().code, "APPROVAL_REQUIRED")

        launch.assert_not_called()

        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for("ingest"))

    def test_unsupported_runtime_blocks_before_approval_or_worker(self):
        """3.12 실제 지원은 허용하지 않으며 사전 검사에서 중단한다."""
        with patch.object(import_cli, "os", SimpleNamespace(name="nt")), patch.object(
            import_cli, "sys", SimpleNamespace(version_info=(3, 12, 15))
        ), patch("src.agents.pickle_import.run_import_worker") as launch, patch.object(
            self.manager.approvals, "record_confirmed_approval"
        ) as approve:
            result = self.manager.run_next()

        self.assertEqual(result.code, "TOOL_INPUT_BLOCKED")
        self.assertEqual(result.status, "blocked_input")
        self.assertFalse(self.tool.states.load().artifact_refs)
        launch.assert_not_called()
        approve.assert_not_called()

    def test_approval_completion_and_reuse(self):
        """승인 전 미실행·한 번의 승인 소비·검증·완료 등록·재실행 방지를 확인한다."""
        approval = self.approval()

        with patch("src.agents.pickle_import.run_import_worker", side_effect=self.transport) as launch:
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, "completed")
            self.manager.run_next()

        launch.assert_called_once()
        report = status(open_run("example"))
        self.assertEqual(report["canonical_integrity"], "passed")
        self.assertFalse(report["training_ready"])
        canonical = Path(report["local_canonical_directory"])
        self.assertEqual(inspect_canonical(canonical, self.tool.contract)["row_count"], 1)

        with np.load(canonical / "maps/shard_000000.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["map_000000000"], self.frame.iloc[0]["waferMap"])

    def test_worker_rejects_changed_snapshot_before_loading(self):
        """고정 복사본이 변경되면 pickle 로더를 호출하기 전에 차단한다."""
        request = self.tool._request("b" * 32)
        root = self.tool._stage(request)
        (root / "input/snapshot/input.bin").write_bytes(b"modified")

        with patch("pandas.read_pickle", side_effect=AssertionError("변조 입력 읽기 금지")):
            reply = execute_request(request, root)

        self.assertEqual(reply.status, "blocked_input")

    def test_resource_failure_does_not_commit(self):
        """유효한 파일이 남아 있어도 자원 초과 종료는 완료 결과로 채택하지 않는다."""
        approval = self.approval()

        def failed(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome["launch"]["status"] = "resource_exceeded"
            (root / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")

            return outcome

        with patch("src.agents.pickle_import.run_import_worker", side_effect=failed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")
        self.assertNotIn("ingest", self.tool.states.load().artifact_refs)

    def test_tampered_canonical_is_not_committed(self):
        """worker 성공 응답이 있어도 출처 정보 변경은 부모가 거부한다."""
        approval = self.approval()

        def tampered(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / "canonical/provenance.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["source_sha256"] = "f" * 64
            path.write_text(json.dumps(value), encoding="utf-8")

            return outcome

        with patch("src.agents.pickle_import.run_import_worker", side_effect=tampered):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("ingest", self.tool.states.load().artifact_refs)

    def test_changed_limits_require_a_new_run(self):
        """저장된 출력·자원 제한이 바뀌면 기존 계획으로 재개하지 않는다."""
        selection = self.directory / "selection.json"
        value = json.loads(selection.read_text(encoding="utf-8"))
        value["contract"]["max_output_bytes"] += 1
        selection.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaises(ValueError):
            open_run("example")

    def test_missing_cleanup_evidence_is_rejected(self):
        """변환 파일이 정상이어도 Job 정리 확인이 빠지면 커밋하지 않는다."""
        approval = self.approval()

        def no_evidence(request, root, signal):
            outcome = self.transport(request, root, signal)
            del outcome["launch"]["resources"]["cleanup_confirmed"]

            return outcome

        with patch("src.agents.pickle_import.run_import_worker", side_effect=no_evidence):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertEqual(result.status, "failed_runtime")
        self.assertNotIn("ingest", self.tool.states.load().artifact_refs)

    def test_object_array_is_rejected(self):
        """성공 보고를 위조한 object NPZ도 부모에서 역직렬화하지 않는다."""
        approval = self.approval()

        def object_output(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / "canonical/maps/shard_000000.npz"
            np.savez(path, map_000000000=np.array([[object()]], dtype=object))

            return outcome

        with patch("src.agents.pickle_import.run_import_worker", side_effect=object_output), \
                patch("pickle.load", side_effect=AssertionError("부모 pickle 로더 금지")):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, "completed")
        self.assertNotIn("ingest", self.tool.states.load().artifact_refs)

    def test_request_mismatch_never_loads_pickle(self):
        """다른 정책의 요청은 pickle을 읽지 않고 거부한다."""
        request = self.tool._request("c" * 32)
        root = self.tool._stage(request)
        changed = replace(request, policy_version="other_policy")

        with patch("pandas.read_pickle", side_effect=AssertionError("요청 불일치 로더 금지")):
            reply = execute_request(changed, root)

        self.assertEqual(reply.status, "blocked_input")

    def test_worker_allowlist_bootstraps_independently(self):
        """프로젝트 import 캐시 없이 복사 허용 소스로 worker·변환기를 import한다."""
        copied = self.root / "copied"
        copied.mkdir()
        project = Path(__file__).resolve().parents[2]

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        code = (
            "import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
            "from src.agents.pickle_import_worker import execute_request; "
            "from src.agents.snapshot_eda_worker import _verify_staged_snapshot; "
            "from scripts.wm_sandbox.export_canonical import export_bundle; "
            "from src.agents.pickle_contract import PickleInput; print('bootstrap_passed')"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", code, str(copied), str(Path(sys.prefix) / "Lib/site-packages")],
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bootstrap_passed", result.stdout)


    def test_host_entrypoint_refuses_to_read_pickle(self):
        """일반 프로세스에서 worker 진입점을 직접 실행하면 파일을 읽기 전에 종료한다."""
        from src.agents.pickle_import_worker import main

        with patch("sys.argv", ["worker", str(self.root), "a" * 64]), \
                patch("src.agents.pickle_import_worker._isolated_process", return_value=False), \
                patch("pandas.read_pickle", side_effect=AssertionError("host worker 역직렬화 금지")):
            self.assertEqual(main(), 2)


class PicklePreflightRuntimeTest(unittest.TestCase):
    """운영 사전 검사의 지원 범위를 계산·승인 흐름 모의와 분리한다."""

    def test_platform_version_and_arguments_gate(self):
        cases = (
            ("nt", (3, 10, 0), {}, True),
            ("nt", (3, 12, 15), {}, False),
            ("nt", (3, 11, 0), {}, False),
            ("nt", (3, 13, 0), {}, False),
            ("posix", (3, 10, 0), {}, False),
            ("nt", (3, 10, 0), {"unexpected": True}, False),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            for index, (system, version, arguments, supported) in enumerate(cases):
                output = root / str(index)
                output.mkdir()

                with self.subTest(system=system, version=version, arguments=arguments), patch.object(
                    import_cli, "os", SimpleNamespace(name=system)
                ), patch.object(import_cli, "sys", SimpleNamespace(version_info=version)):
                    result = import_cli._preflight(output, arguments)
                    self.assertEqual(result.status, "passed" if supported else "blocked_input")
                    self.assertEqual((output / "preflight.json").exists(), supported)

                    if supported:
                        self.assertTrue(import_cli._validate_preflight(output, arguments))


if __name__ == "__main__":
    unittest.main()

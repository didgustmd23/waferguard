"""실제 worker·방화벽 변경 없이 요청·결과·경로 사전 계약만 검증한다."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.worker_contract import WorkerLimits, WorkerRequest, WorkerResult
from src.agents.worker_paths import WorkerPathPolicy


class WorkerContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.outputs = self.root / "outputs"
        self.scratch = self.root / "scratch"
        self.protected = self.inputs / "protected"

        for directory in (self.inputs, self.outputs, self.scratch, self.protected):
            directory.mkdir()

        self.source = self.inputs / "synthetic.csv"
        self.source.write_text("value\n1\n", encoding="utf-8")
        self.dependency = self.inputs / "previous"
        self.dependency.mkdir()
        self.output = self.outputs / "task"
        self.output.mkdir()
        self.temporary = self.scratch / "task"
        self.temporary.mkdir()
        self.paths = WorkerPathPolicy((self.inputs,), self.outputs, self.scratch, (self.protected,))
        self.request = WorkerRequest(
            task_id="a" * 32, run_id="b" * 32, actor_id="synthetic_user", stage="run_eda",
            action_id="run_eda", source_id="synthetic_source", config_id="config", plan_id="plan",
            policy_version="preprocess_policy_v1", input_files=(str(self.source),),
            dependency_directories=(str(self.dependency),), output_directory=str(self.output),
            scratch_directory=str(self.temporary), limits=WorkerLimits(),
        )

    def test_request_roundtrip_and_readonly_check(self):
        """형식·정책 검사만 수행하며 파일 생성이나 권한 변경은 하지 않는다."""
        before = {path.relative_to(self.root) for path in self.root.rglob("*")}
        restored = WorkerRequest.from_json(self.request.to_json())
        self.assertEqual(restored, self.request)
        self.assertTrue(restored.check_policy(self.paths, WorkerLimits()))
        self.assertEqual(before, {path.relative_to(self.root) for path in self.root.rglob("*")})

    def test_action_stage_and_network_rejected(self):
        """임의 코드·잘못된 단계·외부 통신 허용을 요청할 수 없다."""
        for changes in (
            {"action_id": "shell"}, {"stage": "prepare_data"}, {"network_policy": "allow_all"},
            {"task_id": "../task"}, {"actor_id": "bad\nuser"}, {"input_files": (["invalid"],)},
        ):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(self.request, **changes)

    def test_strict_json_fields_and_duplicate_keys(self):
        """자유 인자·셸·중복 키·잘못된 숫자·과도한 IPC를 거부한다."""
        for extra in ("command", "api_key", "arguments", "read_roots"):
            payload = json.loads(self.request.to_json())
            payload[extra] = "not_allowed"

            with self.assertRaises(ValueError):
                WorkerRequest.from_json(json.dumps(payload))

        for payload in ('{"task_id": "a", "task_id": "b"}', '{"limits": NaN}', '[]', ' ' * 65537):
            with self.assertRaises(ValueError):
                WorkerRequest.from_json(payload)

    def test_limits_and_request_identity(self):
        """자원·경로·정책 변경은 요청 ID를 바꾸며 관리자 상한을 넘길 수 없다."""
        changed = replace(self.request, limits=WorkerLimits(wall_seconds=301))
        self.assertNotEqual(changed.request_id, self.request.request_id)

        with self.assertRaises(ValueError):
            changed.check_policy(self.paths, WorkerLimits())

        for changes in ({"memory_mib": True}, {"wall_seconds": 0}, {"processes": 9}):
            with self.assertRaises(ValueError):
                WorkerLimits(**changes)

    def test_escape_relative_unc_and_streams(self):
        """문자열 접두사 우회·상위 이동·UNC·ADS·상대 경로를 거부한다."""
        outside = self.root / "inputs_other"
        outside.mkdir()
        source = outside / "outside.csv"
        source.write_text("synthetic", encoding="utf-8")

        for value in (
            str(source), "relative.csv", str(self.inputs) + "/../inputs/synthetic.csv",
            "\\\\server\\share\\source.csv", str(self.source) + ":secret",
            str(self.inputs / "NUL.csv"), str(self.inputs / "bad. "),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.paths.check_read(value)

    def test_protected_and_parent_directory_rejected(self):
        """보호 파일과 보호 하위 폴더를 포함한 상위 폴더 읽기를 모두 거부한다."""
        private = self.protected / "approval.sqlite3"
        private.write_text("synthetic", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.paths.check_read(private)

        with self.assertRaises(ValueError):
            self.paths.check_read(self.inputs, directory=True)

    def test_output_empty_and_roots_separated(self):
        """출력 루트 자체·기존 결과 덮어쓰기·읽기와 쓰기 영역 겹침을 거부한다."""
        with self.assertRaises(ValueError):
            self.paths.check_write(self.outputs)

        (self.output / "partial.txt").write_text("synthetic", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.request.check_policy(self.paths, WorkerLimits())

        for output, scratch in ((self.inputs, self.scratch), (self.outputs, self.outputs)):
            with self.assertRaises(ValueError):
                WorkerPathPolicy((self.inputs,), output, scratch)

    def test_missing_and_wrong_path_types(self):
        """존재하지 않는 경로·폴더를 파일로 지정한 요청·파일을 폴더로 지정한 요청을 거부한다."""
        for value, directory in ((self.inputs / "absent.csv", False), (self.inputs, False), (self.source, True)):
            with self.assertRaises(ValueError):
                self.paths.check_read(value, directory=directory)

    def test_reparse_guard(self):
        """OS 링크 생성 권한과 무관하게 재분석 지점 탐지 시 차단되는 분기를 확인한다."""
        from src.agents import worker_paths

        original = worker_paths._is_link

        with patch.object(worker_paths, "_is_link", side_effect=lambda path: path == self.source or original(path)):
            with self.assertRaises(ValueError):
                self.paths.check_read(self.source)

    def test_hardlinked_input_rejected(self):
        """동일 내용을 가리키는 별도 파일 경로로 읽기 경계를 우회하지 못하게 한다."""
        alias = self.inputs / "alias.csv"
        os.link(self.source, alias)

        with self.assertRaises(ValueError):
            self.paths.check_read(alias)

    def test_result_roundtrip_and_request_binding(self):
        """성공 종료 결과도 정확한 요청 ID에 결합하며 완료 검증을 대신하지 않는다."""
        result = WorkerResult(self.request.task_id, self.request.request_id, "succeeded", "WORKER_EXITED_OK", ("report.json",))
        self.assertEqual(WorkerResult.from_json(result.to_json()), result)
        self.assertTrue(result.matches(self.request))
        self.assertFalse(result.matches(replace(self.request, limits=WorkerLimits(wall_seconds=20))))

    def test_invalid_result_and_raw_errors_rejected(self):
        """실패를 성공 코드로 표현하거나 상세 오류·절대 산출물 경로를 반환할 수 없다."""
        for status, code, refs in (
            ("timed_out", "WORKER_EXITED_OK", ()),
            ("failed_runtime", "WORKER_FAILED", ("result.csv",)),
            ("succeeded", "WORKER_EXITED_OK", ("../result.csv",)),
            ("succeeded", "WORKER_EXITED_OK", ("C:/private/result.csv",)),
        ):
            with self.assertRaises(ValueError):
                WorkerResult(self.request.task_id, self.request.request_id, status, code, refs)

        value = json.loads(WorkerResult(self.request.task_id, self.request.request_id, "timed_out", "WORKER_TIMED_OUT").to_json())
        value["stderr"] = "private error"

        with self.assertRaises(ValueError):
            WorkerResult.from_json(json.dumps(value))


if __name__ == "__main__":
    unittest.main()

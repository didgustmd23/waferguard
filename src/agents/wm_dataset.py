"""기존 승인·AppContainer 실행 흐름을 재사용하는 원본 보존형 WM 개발 구간 Dataset 검사 도구."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from src.preprocessing.canonical_data import decode_json
from .isolated_eda import _read_file
from .wm_canonical import WmCanonicalTool
from .wm_canonical_contract import digest
from .wm_dataset_contract import POLICY_VERSION, read_cache_evidence
from .wm_dataset_runtime import decode_manifest, MAX_MANIFEST_BYTES, copy_runtime_file
from .wm_cache_results import cache_files
from .isolated_eda import CancelSignal
from .wm_dataset_results import inspect_dataset
from .wm_dataset_transport import code_manifest, run_dataset_worker
from .worker_contract import WorkerRequest, WorkerResult, _decode


class WmDatasetTool(WmCanonicalTool):
    """완료된 개발 캐시·런타임을 고정해 CPU 첫 배치 검사만 실행한다."""

    def __init__(self, states, contract, cache_run, evidence):
        self.states, self.contract, self.limits = states, contract, contract.limits
        self.cancel = CancelSignal(states)
        self.worker_started = False
        self.cache_run = cache_run
        self.evidence = evidence
        self.code_id = digest(code_manifest())
        self.arguments = contract.packet(self.code_id, evidence)
        self.config_id = digest(self.arguments)
        self.plan = ExecutionPlan(contract.source_id, self.config_id, POLICY_VERSION, (
            ExecutionStep("preflight", "run_preflight"),
            ExecutionStep("validate_contract", "validate_contract", json.dumps(self.arguments)),
        ))

    def _runtime(self):
        records = decode_manifest(_read_file(self.states.directory / "runtime_files.json", MAX_MANIFEST_BYTES),
                                  self.contract.max_runtime_bytes)

        if digest(records) != self.contract.runtime_sha256:
            raise ValueError("Dataset 런타임 명세가 승인 후 변경되었습니다.")

        return records

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != digest(code_manifest()):
            raise ValueError("Dataset 검사 코드·설정이 승인 범위와 다릅니다.")

        if read_cache_evidence(self.cache_run) != self.evidence:
            raise ValueError("완료된 캐시 증거가 변경되었습니다.")

        self.contract.check()
        self._runtime()

    def _request(self, task_id):
        request = super()._request(task_id)
        root = self._work_root(task_id)
        return replace(request, stage="validate_contract", action_id="validate_contract", policy_version=POLICY_VERSION,
                       input_files=tuple(str(root / ("input/" + name)) for name in
                                         ("analysis.json", "files.json", "runtime_files.json")),
                       dependency_directories=(str(root / "input/cache"),))

    def _stage(self, request):
        """승인 후 캐시만 제한 복사한다. canonical·Test·승인 DB는 전달하지 않는다."""
        from .worker_paths import WorkerPathPolicy

        root = self._work_root(request.task_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        _root(root.parent)
        root.mkdir()

        for name in ("input", "tools", "runtime", "output", "scratch"):
            (root / name).mkdir()

        source = _root(self.contract.cache_directory)
        target = root / "input/cache"
        target.mkdir()

        for name, record in self.contract.files.items():
            copy_runtime_file(source / name, target / name, record)

        if cache_files(target, self.contract.max_input_bytes) != self.contract.files:
            raise ValueError("Dataset 캐시 복사본이 승인 범위와 다릅니다.")

        (root / "input/analysis.json").write_text(json.dumps(self.arguments), encoding="utf-8")
        (root / "input/files.json").write_text(self.contract.files_json, encoding="utf-8")
        (root / "input/runtime_files.json").write_text(json.dumps(self._runtime()), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, source),
        ), self.limits)
        return root

    def execute(self, output, arguments):
        """성공 응답 외에도 종료·입력·파일 계약을 확인한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_dataset_worker(request, self._stage(request), self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})
        self.worker_started = launch.get("status") in {"exited", "timed_out", "resource_exceeded"}

        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        if not self._evidence(request, outcome):
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))

        reply_path = Path(request.output_directory) / "worker_result.json"

        if launch.get("status") == "exited" and launch.get("exit_code") == 2 and reply_path.is_file():
            reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))

            if reply.matches(request) and reply.status == "blocked_input" and not reply.artifact_refs:
                return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if not launch_succeeded(launch):
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(
                launch.get("status"), "WORKER_FAILED")
            return ToolResult("failed_runtime", errors=(code,))

        staged = _root(request.output_directory)
        reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))

        if (not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("dataset",)
            or {path.name for path in staged.iterdir()} != {"dataset", "worker_result.json"}):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        report = inspect_dataset(staged / "dataset", request, self.contract, self.evidence)
        shutil.copytree(staged / "dataset", output / "dataset")

        if inspect_dataset(output / "dataset", request, self.contract, self.evidence) != report:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        (output / "receipt.json").write_text(json.dumps({"request_id": request.request_id, "report": report}), encoding="utf-8")

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        return ToolResult("passed",
                          artifact_refs=("dataset", "receipt.json"))

    def inspect_attempt(self, output):
        """완료·검토 보고서 조회 시에도 요청·종료 증거·해시를 다시 대조한다."""
        self._check_arguments(self.arguments)
        output = _root(output)

        if {path.name for path in output.iterdir()} not in (
            {"dataset", "receipt.json"}, {"dataset", "receipt.json", "AGENT_COMPLETE.json"},
        ):
            raise ValueError("WM 개발 구간 Dataset 검사 시도 결과 파일 목록이 올바르지 않습니다.")

        request = self._request(output.name)
        work = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(work / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(work / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {}))
            or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("dataset",)):
            raise ValueError("WM 개발 구간 Dataset 검사 worker 종료 증거가 현재 작업과 다릅니다.")

        report = inspect_dataset(output / "dataset", request, self.contract, self.evidence)
        receipt = decode_json(_read_file(output / "receipt.json", 1024 * 1024))

        if receipt != {"request_id": request.request_id, "report": report}:
            raise ValueError("WM 개발 구간 Dataset 검사 결과 영수증이 검증된 내용과 다릅니다.")

        return report


    def validate(self, output, arguments):
        self._check_arguments(arguments)
        return not self.cancel.is_set() and self.inspect_attempt(output)["status"] == "dataset_batch_passed"

    def registered_tool(self):
        return RegisteredTool("validate_contract", self.execute, self.validate, requires_approval=True)

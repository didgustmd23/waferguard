"""고정 합성 참조 모듈의 승인·격리 실행·부모 검증·커밋 연결."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root, _fingerprint
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep
from .isolated_eda import _read_file
from .reviewed_module_contract import POLICY_VERSION, check_demo, module_identity
from .reviewed_module_results import read_result
from .reviewed_module_transport import code_manifest, run_reviewed_worker
from .snapshot_eda import SnapshotEdaTool, _digest
from .worker_contract import WorkerRequest, WorkerResult, _decode


class ReviewedModuleTool(SnapshotEdaTool):
    """입력 복사·승인 예약을 재사용하되 임의 CSV·모듈·계획은 받지 않는다."""

    def __init__(self, states, contract, include_complete=True):
        check_demo(contract.snapshot, contract.profile, include_complete)
        super().__init__(states, contract)
        self.include_complete = include_complete
        self.code_id = _digest(code_manifest())
        self.arguments.update(code_id=self.code_id, module=module_identity(include_complete))
        _decode(json.dumps(self.arguments))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(contract.snapshot.source_id, self.config_id, POLICY_VERSION,
            (ExecutionStep("preflight", "run_preflight"), ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments))))

    def _check_arguments(self, arguments):
        check_demo(self.contract.snapshot, self.contract.profile, self.include_complete)

        if arguments != self.arguments or self.code_id != _digest(code_manifest()) or _digest(arguments) != self.config_id:
            raise ValueError("참조 모듈·코드·설정·입력 계약이 승인 범위와 다릅니다.")

        self.contract.check_input()

    def _request(self, task_id):
        return replace(super()._request(task_id), policy_version=POLICY_VERSION)

    def _receipt(self, request):
        receipt = super()._receipt(request)
        receipt.update(version=POLICY_VERSION, module=module_identity(self.include_complete))
        return receipt

    def _evidence(self, request, outcome):
        """종료 코드뿐 아니라 요청 일치·Job 정리·설정 한도·프로필 삭제를 확인한다."""
        launch = outcome.get("launch", {})
        resources = launch.get("resources", {})
        return (
            outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
            and isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
            and resources.get("active_processes") == 0
            and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
            and resources.get("configured_process_limit") == self.limits.processes
        )

    def execute(self, output, arguments):
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        root = self._stage(request)
        outcome = run_reviewed_worker(request, root, self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})

        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        if not self._evidence(request, outcome):
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))

        staged = _root(request.output_directory)
        reply_path = staged / "worker_result.json"

        if launch.get("status") == "exited" and launch.get("exit_code") == 2 and reply_path.exists():
            reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))

            if reply.matches(request) and reply.status == "blocked_input" and not reply.artifact_refs:
                return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if not launch_succeeded(launch):
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(launch.get("status"), "WORKER_FAILED")
            return ToolResult("failed_runtime", errors=(code,))

        if {path.name for path in staged.iterdir()} != {"pipeline", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))
        result = read_result(staged / "pipeline", self.config_id, self.include_complete)

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("pipeline",) or result is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / "pipeline", output / "pipeline")
        (output / "worker_receipt.json").write_text(json.dumps(self._receipt(request)), encoding="utf-8")
        return ToolResult(result["status"], artifact_refs=("pipeline",),
            warnings=("QUALITY_REVIEW_REQUIRED",) if result["status"] == "review_required" else (),
            errors=("QUALITY_INPUT_BLOCKED",) if result["status"] == "blocked_input" else ())

    def validate(self, output, arguments):
        self._check_arguments(arguments)
        names = {path.name for path in _root(output).iterdir()}

        if self.cancel.is_set() or not {"pipeline", "worker_receipt.json"}.issubset(names) or names - {"pipeline", "worker_receipt.json", "AGENT_COMPLETE.json"}:
            return False

        request = self._request(output.name)
        root = self._work_root(output.name)
        saved_request = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (saved_request != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {}))
                or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("pipeline",)):
            return False

        result = read_result(output / "pipeline", self.config_id, self.include_complete)

        if result is None or result["status"] != "passed" or result != read_result(Path(request.output_directory) / "pipeline", self.config_id, self.include_complete):
            return False

        for relative in ("quality/quality.json", "quality/quality.md", "row_missing/missing_row_histogram.csv", "pipeline_result.json"):
            if _fingerprint(output / "pipeline" / relative) != _fingerprint(Path(request.output_directory) / "pipeline" / relative):
                return False

        return _decode(_read_file(output / "worker_receipt.json", 65536).decode("utf-8")) == self._receipt(request)

"""사용자 모듈 계획을 기존 승인·작업 예약·취소·결과 커밋에 연결한다."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep
from .isolated_eda import _read_file
from .module_pipeline_contract import POLICY_VERSION, bind_pipeline
from .module_pipeline_results import read_pipeline_result
from .module_pipeline_transport import code_manifest, run_pipeline_worker
from .snapshot_eda import SnapshotEdaTool, _digest
from .worker_contract import WorkerResult, _decode


class ModulePipelineTool(SnapshotEdaTool):
    """하나의 승인 작업 안에서 선택한 비파괴 모듈을 순차 실행한다."""

    def __init__(self, states, contract, config):
        super().__init__(states, contract)
        bound, self.module_plan = bind_pipeline(config, contract.profile)
        self.code_id = _digest(code_manifest())
        self.worker_started = False
        self.arguments.update(
            code_id=self.code_id, pipeline=bound, pipeline_signature=self.module_plan.signature,
        )
        # 전체 IPC도 기존 한도 안에 들어야 한다. 원본·프로파일을 조용히 잘라내지 않는다.
        _decode(json.dumps(self.arguments, ensure_ascii=False))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.snapshot.source_id, config_id=self.config_id, policy_version=POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments))),
        )

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != _digest(code_manifest()):
            raise ValueError("모듈 코드·설정이 승인 대상과 다릅니다.")

        _, plan = bind_pipeline(arguments["pipeline"], self.contract.profile)

        if plan.signature != self.module_plan.signature or _digest(arguments) != self.config_id:
            raise ValueError("모듈 계획 또는 입력 계약이 변경되었습니다.")

        self.contract.check_input()

    def _read_result(self, directory):
        """부모 결과 검증 입구. 진단 도구만 이 메서드를 별도 검증기로 바꾼다."""
        return read_pipeline_result(directory, self.module_plan, self.contract.profile)

    def _request(self, task_id):
        return replace(super()._request(task_id), policy_version=POLICY_VERSION)

    def _receipt(self, request):
        receipt = super()._receipt(request)
        receipt.update(version=POLICY_VERSION, pipeline_signature=self.module_plan.signature)

        return receipt

    def execute(self, output, arguments):
        """Job 종료 증거와 부모 파일 검증을 모두 통과한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        root = self._stage(request)
        outcome = run_pipeline_worker(request, root, self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})
        self.worker_started = launch.get("status") in {
            "exited", "timed_out", "resource_exceeded",
        }

        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        resources = launch.get("resources", {})
        evidence = (
            outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
            and isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
            and resources.get("active_processes") == 0
            and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
            and resources.get("configured_process_limit") == self.limits.processes
        )

        if not evidence:
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
        result = self._read_result(staged / "pipeline")

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("pipeline",) or result is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / "pipeline", output / "pipeline")
        (output / "worker_receipt.json").write_text(json.dumps(self._receipt(request)), encoding="utf-8")

        if self._read_result(output / "pipeline") != result:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        status = result["status"]

        return ToolResult(
            status, artifact_refs=("pipeline",),
            warnings=("QUALITY_REVIEW_REQUIRED",) if status == "review_required" else (),
            errors=("QUALITY_INPUT_BLOCKED",) if status == "blocked_input" else (),
        )

    def validate(self, output, arguments):
        self._check_arguments(arguments)
        names = {path.name for path in _root(output).iterdir()}

        if self.cancel.is_set() or not {"pipeline", "worker_receipt.json"}.issubset(names) or names - {"pipeline", "worker_receipt.json", "AGENT_COMPLETE.json"}:
            return False

        result = self._read_result(output / "pipeline")
        receipt = _decode(_read_file(output / "worker_receipt.json", 65536).decode("utf-8"))

        return result is not None and result["status"] == "passed" and receipt == self._receipt(self._request(output.name))

"""기존 승인·AppContainer 실행 흐름을 재사용하는 원본 보존형 충돌 제외 적용 도구."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .isolated_eda import _read_file
from .wm_canonical import WmCanonicalTool
from .wm_canonical_contract import digest
from .wm_exclusion_evidence import check_candidate_evidence
from .wm_exclusion_results import POLICY_VERSION, inspect_derived, read_result_record
from .wm_exclusion_transport import code_manifest, run_exclusion_worker
from .worker_contract import WorkerRequest, WorkerResult, _decode


class WmExclusionTool(WmCanonicalTool):
    """파일 복사·Job 증거 확인을 재사용하고 새 canonical의 결과 계약을 검사한다."""

    def __init__(self, states, contract, candidate_run, evidence):
        super().__init__(states, contract)
        self.candidate_run = candidate_run
        self.evidence = evidence
        self.code_id = digest(code_manifest())
        self.arguments = {**contract.packet(self.code_id), "version": POLICY_VERSION, "candidate_evidence": evidence}
        self.config_id = digest(self.arguments)
        self.plan = ExecutionPlan(contract.source_id, self.config_id, POLICY_VERSION, (
            ExecutionStep("preflight", "run_preflight"),
            ExecutionStep("derive_preparation", "derive_preparation", json.dumps(self.arguments)),
        ))

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != digest(code_manifest()):
            raise ValueError("충돌 제외 적용의 코드·설정이 승인 범위와 다릅니다.")

        previous = check_candidate_evidence(self.candidate_run, self.evidence)

        if previous.source_id != self.contract.source_id or previous.max_group_ratio != self.contract.max_group_ratio:
            raise ValueError("충돌 제외 적용과 후보 진단의 입력·검토 상한이 다릅니다.")

        self.contract.check()

    def _request(self, task_id):
        return replace(super()._request(task_id), stage="derive_preparation", action_id="derive_preparation",
                       policy_version=POLICY_VERSION)

    def execute(self, output, arguments):
        """성공 응답 외에도 종료·입력·파일 계약을 확인한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_exclusion_worker(request, self._stage(request), self.cancel)
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

        if (not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("derived",)
            or {path.name for path in staged.iterdir()} != {"derived", "worker_result.json"}):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        report = inspect_derived(staged / "derived", request, self.contract, digest(self.evidence), self.evidence)
        shutil.copytree(staged / "derived", output / "derived")

        if inspect_derived(output / "derived", request, self.contract, digest(self.evidence), self.evidence) != report:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        (output / "receipt.json").write_text(json.dumps({"request_id": request.request_id, "report": report}), encoding="utf-8")

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        return ToolResult("passed", artifact_refs=("derived", "receipt.json"))

    def inspect_attempt(self, output):
        """완료·검토 보고서 조회 시에도 요청·종료 증거·해시를 다시 대조한다."""
        self._check_arguments(self.arguments)
        output = _root(output)

        if {path.name for path in output.iterdir()} not in (
            {"derived", "receipt.json"}, {"derived", "receipt.json", "AGENT_COMPLETE.json"},
        ):
            raise ValueError("WM 충돌 제외 적용 시도 결과 파일 목록이 올바르지 않습니다.")

        request = self._request(output.name)
        work = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(work / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(work / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {}))
            or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("derived",)):
            raise ValueError("WM 충돌 제외 적용 worker 종료 증거가 현재 작업과 다릅니다.")

        report = inspect_derived(output / "derived", request, self.contract, digest(self.evidence), self.evidence)
        receipt = read_result_record(output / "receipt.json")

        if receipt != {"request_id": request.request_id, "report": report}:
            raise ValueError("WM 충돌 제외 적용 결과 영수증이 검증된 내용과 다릅니다.")

        return report


    def validate(self, output, arguments):
        self._check_arguments(arguments)
        return not self.cancel.is_set() and self.inspect_attempt(output)["status"] == "derived_ready"

    def registered_tool(self):
        return RegisteredTool("derive_preparation", self.execute, self.validate, requires_approval=True)

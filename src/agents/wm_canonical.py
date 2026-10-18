"""WM canonical 품질·EDA를 승인 실행기와 기존 AppContainer+Job에 연결한다."""

import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .isolated_eda import CancelSignal, _read_file
from .wm_canonical_contract import POLICY_VERSION, WmCanonicalInput, copy_snapshot, digest
from .wm_canonical_results import inspect_analysis
from .wm_canonical_transport import code_manifest, run_wm_worker
from .worker_contract import WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy


class WmCanonicalTool:
    """원본·승인 DB는 전달하지 않고 검토 결과를 자동 완료로 승격하지 않는다."""

    def __init__(self, states, contract):
        if not isinstance(contract, WmCanonicalInput):
            raise TypeError("WM canonical 입력 계약이 필요합니다.")

        contract.check()
        self.states = states
        self.contract = contract
        self.limits = contract.limits
        self.cancel = CancelSignal(states)
        self.worker_started = False
        self.code_id = digest(code_manifest())
        self.arguments = contract.packet(self.code_id)
        self.config_id = digest(self.arguments)
        self.plan = ExecutionPlan(contract.source_id, self.config_id, POLICY_VERSION, (
            ExecutionStep("preflight", "run_preflight"),
            ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments)),
        ))

    def _check_arguments(self, arguments):
        if arguments != self.arguments or digest(code_manifest()) != self.code_id:
            raise ValueError("WM 분석 설정·코드가 승인 범위와 다릅니다.")

        self.contract.check()

    def _work_root(self, task_id):
        if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
            raise ValueError("올바른 worker 작업 ID가 필요합니다.")

        return _root(self.states.directory.parent.parent) / "workers" / task_id

    def _request(self, task_id):
        root = self._work_root(task_id)
        state = self.states.load()
        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
            stage="run_eda", action_id="run_eda", source_id=self.plan.source_id,
            config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=POLICY_VERSION,
            input_files=(str(root / "input/analysis.json"), str(root / "input/files.json")),
            dependency_directories=(str(root / "input/canonical"),),
            output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"),
            limits=self.limits,
        )

    def _stage(self, request):
        """승인을 소비한 후에만 원본 대신 작업 전용 canonical 복사본을 전달한다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        _root(root.parent)
        root.mkdir(exist_ok=False)

        for name in ("input", "tools", "runtime", "output", "scratch"):
            (root / name).mkdir()

        copy_snapshot(self.contract.snapshot_directory, root / "input/canonical",
                      self.contract.max_input_bytes, self.contract.files)
        (root / "input/analysis.json").write_text(json.dumps(self.arguments), encoding="utf-8")
        (root / "input/files.json").write_text(self.contract.files_json, encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(self.contract.snapshot_directory)),
        ), self.limits)
        return root

    def _evidence(self, request, outcome):
        """worker 종료와 Job 전체 정리·요청한 한도의 OS 적용 증거를 확인한다."""
        resources = outcome.get("launch", {}).get("resources", {})
        return (outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
                and type(resources) is dict and resources.get("cleanup_confirmed") is True
                and resources.get("active_processes") == 0
                and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
                and resources.get("configured_process_limit") == self.limits.processes)

    def execute(self, output, arguments):
        """성공 응답 외에도 종료·입력·파일 계약을 확인한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_wm_worker(request, self._stage(request), self.cancel)
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

        if (not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("analysis",)
            or {path.name for path in staged.iterdir()} != {"analysis", "worker_result.json"}):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        report = inspect_analysis(staged / "analysis", request, self.contract)
        shutil.copytree(staged / "analysis", output / "analysis")

        if inspect_analysis(output / "analysis", request, self.contract) != report:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        (output / "receipt.json").write_text(json.dumps({"request_id": request.request_id, "report": report}), encoding="utf-8")

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        if report["quality_status"] == "review_required":
            return ToolResult("review_required", artifact_refs=("analysis", "receipt.json"),
                              warnings=("WM_QUALITY_REVIEW_REQUIRED",))

        return ToolResult("passed", artifact_refs=("analysis", "receipt.json"))

    def inspect_attempt(self, output):
        """완료·검토 보고서 조회 시에도 요청·종료 증거·해시를 다시 대조한다."""
        self._check_arguments(self.arguments)
        output = _root(output)

        if {path.name for path in output.iterdir()} not in (
            {"analysis", "receipt.json"}, {"analysis", "receipt.json", "AGENT_COMPLETE.json"},
        ):
            raise ValueError("WM 시도 결과 파일 목록이 올바르지 않습니다.")

        request = self._request(output.name)
        work = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(work / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(work / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {}))
            or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("analysis",)):
            raise ValueError("WM worker 종료 증거가 현재 작업과 다릅니다.")

        report = inspect_analysis(output / "analysis", request, self.contract)
        receipt = _decode(_read_file(output / "receipt.json", 65536).decode("utf-8"))

        if receipt != {"request_id": request.request_id, "report": report}:
            raise ValueError("WM 결과 영수증이 검증된 내용과 다릅니다.")

        return report

    def validate(self, output, arguments):
        self._check_arguments(arguments)
        return not self.cancel.is_set() and self.inspect_attempt(output)["quality_status"] == "passed"

    def registered_tool(self):
        return RegisteredTool("run_eda", self.execute, self.validate, requires_approval=True)

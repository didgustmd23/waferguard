"""Train 전용 후보 산출의 승인·격리 실행·부모 검증·커밋 연결."""

import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .input_snapshot import create_snapshot
from .isolated_eda import CancelSignal, _read_file
from .preparation_candidate_contract import PreparationCandidateContract, POLICY_VERSION
from .preparation_candidate_results import read_candidate
from .preparation_candidate_transport import code_manifest, run_candidate_worker
from .snapshot_eda import SnapshotEdaTool, _digest
from .worker_contract import WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy


class PreparationCandidateTool(SnapshotEdaTool):
    """기준 산출만 등록한다. 기준 적용 도구는 이번 실행 계획에 없다."""

    def __init__(self, states, contract):
        if not isinstance(contract, PreparationCandidateContract):
            raise TypeError("검증된 Train 후보 산출 계약이 필요합니다.")

        self.states = states
        self.contract = contract
        self.limits = contract.limits
        self.cancel = CancelSignal(states)
        self.code_id = _digest(code_manifest())
        self.arguments = json.loads(json.dumps(contract.packet(self.code_id)))
        _decode(json.dumps(self.arguments))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.train.snapshot.source_id, config_id=self.config_id, policy_version=POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"),
                   ExecutionStep("derive_preparation", "derive_preparation", json.dumps(self.arguments))),
        )

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != _digest(code_manifest()) or self.arguments != json.loads(json.dumps(self.contract.packet(self.code_id))):
            raise ValueError("Train 출처·후보 규칙·코드·한도가 승인 범위와 다릅니다.")

    def _request(self, task_id):
        root = self._work_root(task_id)
        state = self.states.load()
        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
            stage="derive_preparation", action_id="derive_preparation", source_id=self.plan.source_id,
            config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=POLICY_VERSION, limits=self.limits,
            input_files=tuple(str(root / "input" / name) for name in ("snapshot/input.bin", "snapshot/snapshot.json", "candidate_task.json")),
            dependency_directories=(), output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"),
        )

    def _stage(self, request):
        """승인 소비·예약 후 Train 파일만 복사한다. 다른 구간은 전달하지 않는다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir()

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        train = self.contract.train
        staged = create_snapshot(train.snapshot_directory / "input.bin", root / "input/snapshot",
                                 max_input_bytes=train.snapshot.max_input_bytes)

        if staged != train.snapshot:
            raise ValueError("작업용 Train 복사본이 승인 대상과 다릅니다.")

        (root / "input/candidate_task.json").write_text(json.dumps(self.arguments, ensure_ascii=False), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(train.export.run_directory), Path(train.split.artifact.run_directory)),
        ), self.limits)
        return root

    def _evidence(self, request, outcome):
        resources = outcome.get("launch", {}).get("resources", {})
        return (outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
                and isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
                and resources.get("active_processes") == 0
                and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
                and resources.get("configured_process_limit") == self.limits.processes)

    def execute(self, output, arguments):
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_candidate_worker(request, self._stage(request), self.cancel)
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

        if {path.name for path in staged.iterdir()} != {"candidate", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))
        report = read_candidate(staged / "candidate", self.contract, self.code_id)

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("candidate",) or report is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / "candidate", output / "candidate")
        return ToolResult("passed", artifact_refs=("candidate",))

    def validate(self, output, arguments):
        self._check_arguments(arguments)

        if self.cancel.is_set() or {path.name for path in _root(output).iterdir()} - {"candidate", "AGENT_COMPLETE.json"}:
            return False

        request = self._request(output.name)
        root = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {})) or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("candidate",):
            return False

        report = read_candidate(output / "candidate", self.contract, self.code_id)
        return report is not None and report == read_candidate(Path(request.output_directory) / "candidate", self.contract, self.code_id)

    def registered_tool(self):
        return RegisteredTool("derive_preparation", self.execute, self.validate, requires_approval=True)

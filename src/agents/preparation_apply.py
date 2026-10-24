"""저장 후보를 두 구간에 적용하는 별도 승인·격리 실행·검증·커밋 연결."""

import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root, _fingerprint
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .input_snapshot import create_snapshot
from .isolated_eda import CancelSignal, _read_file
from .preparation_apply_contract import PreparationApplyContract, POLICY_VERSION
from .preparation_apply_results import read_application
from .preparation_apply_transport import code_manifest, run_apply_worker
from .snapshot_eda import SnapshotEdaTool, _digest
from .worker_contract import WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy


class PreparationApplyTool(SnapshotEdaTool):
    """승인된 상태를 적용만 한다. 후보 산출 도구는 이 계획에 등록하지 않는다."""

    def __init__(self, states, contract):
        if not isinstance(contract, PreparationApplyContract):
            raise TypeError("완료 후보와 두 구간의 적용 계약이 필요합니다.")

        self.states = states
        self.contract = contract
        self.limits = contract.limits
        self.cancel = CancelSignal(states)
        self.code_id = _digest(code_manifest())
        self.arguments = json.loads(json.dumps(contract.packet(self.code_id)))
        _decode(json.dumps(self.arguments))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.source_id, config_id=self.config_id, policy_version=contract.schema_version,
            steps=(ExecutionStep("preflight", "run_preflight"),
                   ExecutionStep("apply_preparation", "apply_preparation", json.dumps(self.arguments))),
        )

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != _digest(code_manifest()) or self.arguments != json.loads(json.dumps(self.contract.packet(self.code_id))):
            raise ValueError("Train 출처·후보 규칙·코드·한도가 승인 범위와 다릅니다.")

    def _request(self, task_id):
        from .tabular_preparation_worker import input_files
        root = self._work_root(task_id)
        state = self.states.load()
        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
            stage="apply_preparation", action_id="apply_preparation", source_id=self.plan.source_id,
            config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=self.contract.schema_version, limits=self.limits,
            input_files=input_files(root, applying=True) if self.contract.schema_version == "preparation_apply_v2" else tuple(str(root / "input" / name) for name in (
                "train/input.bin", "train/snapshot.json", "validation/input.bin", "validation/snapshot.json",
                "preprocessor.json", "positions.json", "apply_task.json",
            )),
            dependency_directories=(), output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"),
        )

    def _stage(self, request):
        """별도 적용 승인 소비·예약 뒤 두 구간과 정확한 후보 상태만 복사한다."""
        if self.contract.schema_version == "preparation_apply_v2":
            from .tabular_preparation import stage_parts
            return stage_parts(self, request, applying=True)
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir()

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        for name, part in (("train", self.contract.train), ("validation", self.contract.validation)):
            staged = create_snapshot(part.snapshot_directory / "input.bin", root / "input" / name,
                                     max_input_bytes=part.snapshot.max_input_bytes)

            if staged != part.snapshot:
                raise ValueError("적용 입력 복사본이 승인 대상과 다릅니다.")

        shutil.copyfile(self.contract.candidate.directory / "preprocessor.json", root / "input/preprocessor.json")

        if _fingerprint(root / "input/preprocessor.json")["sha256"] != self.contract.candidate.state_sha256:
            raise ValueError("복사한 저장 기준의 해시가 승인 후보와 다릅니다.")

        positions = json.dumps(self.contract.positions(), ensure_ascii=False).encode("utf-8")

        if len(positions) > 8 * 1024 * 1024:
            raise ValueError("행 위치 전달 파일 크기 상한을 초과했습니다.")

        (root / "input/positions.json").write_bytes(positions)
        (root / "input/apply_task.json").write_text(json.dumps(self.arguments, ensure_ascii=False), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(self.contract.candidate.artifact.run_directory),
                             Path(self.contract.train.export.run_directory), Path(self.contract.train.split.artifact.run_directory)),
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
        outcome = run_apply_worker(request, self._stage(request), self.cancel)
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

        if {path.name for path in staged.iterdir()} != {"prepared", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))
        report = read_application(staged / "prepared", self.contract, self.code_id)

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("prepared",) or report is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / "prepared", output / "prepared")
        return ToolResult("passed", artifact_refs=("prepared",))

    def validate(self, output, arguments):
        self._check_arguments(arguments)

        if self.cancel.is_set() or {path.name for path in _root(output).iterdir()} - {"prepared", "AGENT_COMPLETE.json"}:
            return False

        request = self._request(output.name)
        root = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {})) or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("prepared",):
            return False

        report = read_application(output / "prepared", self.contract, self.code_id)
        return report is not None and report == read_application(Path(request.output_directory) / "prepared", self.contract, self.code_id)

    def registered_tool(self):
        return RegisteredTool("apply_preparation", self.execute, self.validate, requires_approval=True)

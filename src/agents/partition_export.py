"""승인된 Train·Validation 내보내기와 부모 검증·기존 실행기 커밋을 연결한다."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root, _fingerprint
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .input_snapshot import create_snapshot
from .isolated_eda import CancelSignal, _read_file
from .partition_export_contract import PartitionExportContract, POLICY_VERSION
from .partition_export_results import read_partition_export
from .partition_export_transport import code_manifest, run_export_worker
from .snapshot_eda import SnapshotEdaTool, _digest
from .worker_contract import WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy


class PartitionExportTool(SnapshotEdaTool):
    """고정 작업 경로만 재사용한다. EDA 실행·검증·승인 범위는 재사용하지 않는다."""

    def __init__(self, states, contract):
        if not isinstance(contract, PartitionExportContract):
            raise TypeError("확정 분할 내보내기 계약이 필요합니다.")

        self.states = states
        self.contract = contract
        self.limits = contract.source.limits
        self.cancel = CancelSignal(states)
        self.code_id = _digest(code_manifest())
        self.arguments = contract.packet(self.code_id)
        _decode(json.dumps(self.arguments, ensure_ascii=False))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.split.artifact.scope.source_id, config_id=self.config_id,
            policy_version=POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"),
                   ExecutionStep("ingest", "run_ingest", json.dumps(self.arguments))),
        )

    def _check_arguments(self, arguments):
        """원본·분할 이력·프로파일·코드·한도 변경은 이전 승인으로 실행하지 않는다."""
        if self.code_id != _digest(code_manifest()) or arguments != self.arguments:
            raise ValueError("내보내기 코드 또는 승인 설정이 변경되었습니다.")

        if self.contract.packet(self.code_id) != arguments or _digest(arguments) != self.config_id:
            raise ValueError("분할·입력 또는 출력 상한이 승인 범위와 다릅니다.")

    def _request(self, task_id):
        root = self._work_root(task_id)
        state = self.states.load()

        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
            stage="ingest", action_id="run_ingest", source_id=self.plan.source_id,
            config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=POLICY_VERSION,
            limits=self.limits, dependency_directories=(),
            input_files=tuple(str(root / "input" / name) for name in (
                "snapshot/input.bin", "snapshot/snapshot.json", "export.json", "split.json",
            )),
            output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"),
        )

    def _stage(self, request):
        """승인 소비·예약 이후에만 원본 복사본과 확정 분할을 작업 폴더로 전달한다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir(exist_ok=False)

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        source = self.contract.source
        staged = create_snapshot(
            Path(source.snapshot_directory) / "input.bin", root / "input/snapshot",
            max_input_bytes=source.snapshot.max_input_bytes,
        )

        if staged != source.snapshot:
            raise ValueError("작업용 원본 복사본이 승인 대상과 다릅니다.")

        shutil.copyfile(self.contract.split.artifact.directory / "split.json", root / "input/split.json")

        if _fingerprint(root / "input/split.json")["sha256"] != self.arguments["split_file_sha256"]:
            raise ValueError("복사한 확정 분할이 승인 대상과 다릅니다.")

        (root / "input/export.json").write_text(json.dumps(self.arguments, ensure_ascii=False), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(source.snapshot_directory),
                             Path(self.contract.split.artifact.run_directory)),
        ), self.limits)

        return root

    def execute(self, output, arguments):
        """종료 응답만 믿지 않고 Job 정리·요청 결합·파일을 모두 확인한다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_export_worker(request, self._stage(request), self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})

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

        if {path.name for path in staged.iterdir()} != {"exports", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))
        report = read_partition_export(staged / "exports", self.contract)

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("exports",) or report is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        # 완료 폴더는 기존 구간 입력 계약을 유지한다. 운송 증거는 별도 worker 폴더에 보존한다.
        for name in ("train", "validation"):
            shutil.copytree(staged / "exports" / name, output / name)

        shutil.copyfile(staged / "exports/partition_exports.json", output / "partition_exports.json")

        if read_partition_export(output, self.contract) != report or self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        return ToolResult("passed", artifact_refs=("train", "validation", "partition_exports.json"))

    def validate(self, output, arguments):
        """검증·커밋 직전 입력을 재확인한다. 완료 표식은 실행 관리자만 생성한다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return False

        # 완료 폴더의 형식을 늘리지 않고 작업 전용 운송 기록과 요청을 재대조한다.
        request = self._request(output.name)
        root = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        launch = outcome.get("launch", {})
        resources = launch.get("resources", {})
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (
            saved != request or outcome.get("request_id") != request.request_id
            or outcome.get("profile_cleanup") is not True or not launch_succeeded(launch)
            or not isinstance(resources, dict) or resources.get("cleanup_confirmed") is not True
            or resources.get("active_processes") != 0
            or resources.get("configured_job_memory_bytes") != self.limits.memory_mib * 1024 * 1024
            or resources.get("configured_process_limit") != self.limits.processes
            or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("exports",)
        ):
            return False

        report = read_partition_export(output, self.contract, committed=(output / "AGENT_COMPLETE.json").exists())

        return report is not None and report == read_partition_export(Path(request.output_directory) / "exports", self.contract)

    def registered_tool(self):
        return RegisteredTool("run_ingest", self.execute, self.validate, requires_approval=True)

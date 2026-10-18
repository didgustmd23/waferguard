"""기존 승인 실행기에 구조 판별형 pickle 변환·검토 중단을 연결한다."""

import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .input_snapshot import create_snapshot
from .isolated_eda import CancelSignal, _read_file
from .pickle_contract import POLICY_VERSION, PickleInput, digest
from .pickle_import_transport import code_manifest, run_import_worker
from .pickle_results import inspect_canonical, read_structure_review
from .worker_contract import WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy


class PickleImportTool:
    """승인된 복사본만 격리 worker에 전달하고 숫자 결과만 로컬에 등록한다."""

    def __init__(self, states, contract):
        if not isinstance(contract, PickleInput):
            raise TypeError("pickle 입력 계약이 필요합니다.")

        contract.check()
        self.states = states
        self.contract = contract
        self.limits = contract.limits
        self.cancel = CancelSignal(states)
        self.worker_started = False
        self.code_id = digest(code_manifest())
        self.arguments = contract.packet(self.code_id)
        self.config_id = digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.snapshot.source_id, config_id=self.config_id, policy_version=POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"),
                   ExecutionStep("ingest", "run_ingest", json.dumps(self.arguments))),
        )

    def _check_arguments(self, arguments):
        """입력·설정·변환 코드 변경은 실행과 기존 승인 재사용 전에 거부한다."""
        if arguments != self.arguments or self.code_id != digest(code_manifest()):
            raise ValueError("pickle 변환 코드 또는 설정이 승인 범위와 다릅니다.")

        self.contract.check()

    def _work_root(self, task_id):
        if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
            raise ValueError("유효한 worker 작업 ID가 필요합니다.")

        return _root(self.states.directory.parent.parent) / "workers" / task_id

    def _request(self, task_id):
        root = self._work_root(task_id)
        state = self.states.load()

        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
            stage="ingest", action_id="run_ingest", source_id=self.plan.source_id,
            config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=POLICY_VERSION,
            input_files=tuple(str(root / "input" / name) for name in (
                "snapshot/input.bin", "snapshot/snapshot.json", "import.json",
            )), dependency_directories=(), output_directory=str(root / "output/result"),
            scratch_directory=str(root / "scratch/task"), limits=self.limits,
        )

    def _stage(self, request):
        """승인 소비·예약 후에만 작업 폴더와 읽기 전용으로 전달할 복사본을 준비한다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        _root(root.parent)
        root.mkdir(exist_ok=False)

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        staged = create_snapshot(
            Path(self.contract.snapshot_directory) / "input.bin", root / "input/snapshot",
            max_input_bytes=self.contract.snapshot.max_input_bytes,
        )

        if staged != self.contract.snapshot:
            raise ValueError("worker에 복사한 pickle이 승인 대상과 다릅니다.")

        (root / "input/import.json").write_text(json.dumps(self.arguments), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(self.contract.snapshot_directory)),
        ), self.limits)

        return root

    def _evidence(self, request, outcome):
        """요청 결합·AppContainer 정리·Job 전체 프로세스 종료·자원 한도를 확인한다."""
        resources = outcome.get("launch", {}).get("resources", {})

        return (
            outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
            and isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
            and resources.get("active_processes") == 0
            and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
            and resources.get("configured_process_limit") == self.limits.processes
        )

    def execute(self, output, arguments):
        """격리 작업 종료와 부모의 canonical 재검증을 통과한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_import_worker(request, self._stage(request), self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})
        self.worker_started = launch.get("status") in {"exited", "timed_out", "resource_exceeded"}

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
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(
                launch.get("status"), "WORKER_FAILED",
            )
            return ToolResult("failed_runtime", errors=(code,))

        reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))

        if (
            reply.matches(request) and reply.status == "succeeded" and reply.artifact_refs == ("review",)
            and {path.name for path in staged.iterdir()} == {"review", "worker_result.json"}
        ):
            read_structure_review(staged / "review", request, self.contract.max_output_bytes)
            shutil.copytree(staged / "review", output / "review")
            read_structure_review(output / "review", request, self.contract.max_output_bytes)
            return ToolResult("review_required", artifact_refs=("review",), warnings=("PICKLE_STRUCTURE_REVIEW_REQUIRED",))

        if {path.name for path in staged.iterdir()} != {"canonical", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("canonical",):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        report = inspect_canonical(staged / "canonical", self.contract, request.request_id)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / "canonical", output / "canonical")

        if inspect_canonical(output / "canonical", self.contract, request.request_id) != report:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        receipt = {"version": POLICY_VERSION, "request_id": request.request_id, "report": report}
        (output / "import_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

        return ToolResult("passed", artifact_refs=("canonical", "import_receipt.json"))

    def validate(self, output, arguments):
        """커밋 직전과 재사용 시 종료 증거·파일 내용·출처를 다시 대조한다."""
        self._check_arguments(arguments)

        if self.cancel.is_set() or {path.name for path in _root(output).iterdir()} not in (
            {"canonical", "import_receipt.json"}, {"canonical", "import_receipt.json", "AGENT_COMPLETE.json"},
        ):
            return False

        request = self._request(output.name)
        root = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (
            saved != request or not self._evidence(request, outcome)
            or not launch_succeeded(outcome.get("launch", {})) or not reply.matches(request)
            or reply.status != "succeeded" or reply.artifact_refs != ("canonical",)
        ):
            return False

        receipt = _decode(_read_file(output / "import_receipt.json", 65536).decode("utf-8"))

        return receipt == {
            "version": POLICY_VERSION, "request_id": request.request_id,
            "report": inspect_canonical(output / "canonical", self.contract, request.request_id),
        }

    def registered_tool(self):
        return RegisteredTool("run_ingest", self.execute, self.validate, requires_approval=True)

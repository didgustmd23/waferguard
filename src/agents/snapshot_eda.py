"""snapshot EDA의 승인·예약 후 전달과 부모 결과 검증을 연결한다."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from .appcontainer_probe import launch_succeeded
from .artifacts import _root
from .contracts import ToolResult
from .eda_input_contract import EdaInputContract
from .eda_result_validation import validate_eda_report
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool
from .input_snapshot import PAYLOAD_NAME, create_snapshot
from .isolated_eda import CancelSignal, _read_file
from .snapshot_eda_transport import code_manifest, run_snapshot_eda
from .worker_contract import WorkerRequest, WorkerResult
from .worker_paths import WorkerPathPolicy


POLICY_VERSION = "snapshot_eda_execution_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class SnapshotEdaTool:
    """등록 CSV 분석 도구. 원본 폴더 권한·pickle 로더·비격리 fallback은 제공하지 않는다."""

    def __init__(self, states, contract):
        if not isinstance(contract, EdaInputContract):
            raise TypeError("검증된 EDA 입력 계약이 필요합니다.")

        self.states = states
        self.contract = contract
        self.limits = contract.limits
        self.code_id = _digest(code_manifest())
        self.arguments = {
            "contract_id": contract.contract_id, "snapshot": asdict(contract.snapshot),
            "profile": contract.profile.to_dict(), "limits": asdict(contract.limits), "code_id": self.code_id,
        }
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=contract.snapshot.source_id, config_id=self.config_id, policy_version=POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep("run_eda", "run_eda", json.dumps(self.arguments))),
        )
        self.cancel = CancelSignal(states)

    def _check_arguments(self, arguments):
        """코드·설정·선택한 복사본이 승인 범위에서 바뀌지 않았는지 재검사한다."""
        if arguments != self.arguments or self.code_id != _digest(code_manifest()):
            raise ValueError("EDA 코드·설정이 승인 대상과 다릅니다.")

        self.contract.check_input()

    def _work_root(self, task_id):
        if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
            raise ValueError("내부 작업 ID가 필요합니다.")

        return _root(self.states.directory.parent.parent) / "workers" / task_id

    def _request(self, task_id):
        root = self._work_root(task_id)
        state = self.states.load()
        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id, stage="run_eda", action_id="run_eda",
            source_id=self.plan.source_id, config_id=self.config_id, plan_id=self.plan.plan_id,
            policy_version=POLICY_VERSION, limits=self.limits,
            input_files=tuple(str(root / "input" / name) for name in ("snapshot/input.bin", "snapshot/snapshot.json", "eda.json")),
            dependency_directories=(), output_directory=str(root / "output" / "result"),
            scratch_directory=str(root / "scratch" / "task"),
        )

    def _stage(self, request):
        """승인 소비·예약 이후 입력을 새 영역에 복사하고 기대 해시를 대조한다."""
        root = self._work_root(request.task_id)
        # 새 실행기 namespace에서는 workers 디렉터리 자체가 아직 없을 수 있다.
        # 신뢰된 로그 루트 안에서만 필요한 부모 경로를 만들고, 작업별 root는
        # 아래에서 exist_ok=False로 생성해 기존 작업 덮어쓰기를 막는다.
        root.parent.mkdir(parents=True, exist_ok=True)
        _root(root.parent)
        root.mkdir(exist_ok=False)

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        staged = create_snapshot(
            Path(self.contract.snapshot_directory) / PAYLOAD_NAME, root / "input" / "snapshot",
            max_input_bytes=self.contract.snapshot.max_input_bytes,
        )

        if staged != self.contract.snapshot:
            raise ValueError("작업용 복사본이 승인된 내용과 다릅니다.")

        (root / "input" / "eda.json").write_text(json.dumps(self.arguments, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        (root / "input" / "request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        paths = WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, Path(self.contract.snapshot_directory)),
        )
        request.check_policy(paths, self.limits)
        return root

    def _receipt(self, request):
        return {"version": POLICY_VERSION, "contract_id": self.contract.contract_id, "request_id": request.request_id, "task_id": request.task_id, "job_cleanup": True, "profile_cleanup": True}

    def execute(self, output, arguments):
        """worker 성공 주장이 아니라 부모 계약 검사를 통과한 결과만 가져온다."""
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        root = self._stage(request)
        outcome = run_snapshot_eda(request, root, self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})

        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        resources = launch.get("resources", {})
        job_valid = (
            isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
            and resources.get("active_processes") == 0
            and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
            and resources.get("configured_process_limit") == self.limits.processes
        )

        # 입력 거부 응답은 정상 프로세스 종료 확인 후에만 분류한다.
        # 종료 코드 2를 성공으로 바꾸지 않으며 부분 산출물도 등록하지 않는다.
        if (
            outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
            and job_valid and launch.get("status") == "exited" and launch.get("exit_code") == 2
        ):
            reply_path = Path(request.output_directory) / "worker_result.json"

            if reply_path.exists():
                reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))

                if reply.matches(request) and reply.status == "blocked_input" and not reply.artifact_refs:
                    return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if outcome.get("request_id") != request.request_id or outcome.get("profile_cleanup") is not True or not launch_succeeded(launch) or not job_valid:
            code = {"timed_out": "WORKER_TIMED_OUT", "resource_exceeded": "WORKER_RESOURCE_EXCEEDED"}.get(launch.get("status"), "WORKER_FAILED")
            return ToolResult("failed_runtime", errors=(code,))

        staged = _root(request.output_directory)

        if {path.name for path in staged.iterdir()} != {"eda", "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("eda",) or not validate_eda_report(staged / "eda", self.contract.profile):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / "eda", output / "eda")
        (output / "worker_receipt.json").write_text(json.dumps(self._receipt(request)), encoding="utf-8")
        return ToolResult("passed", artifact_refs=("eda",))

    def validate(self, output, arguments):
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return False

        names = {path.name for path in _root(output).iterdir()}

        if not {"eda", "worker_receipt.json"}.issubset(names) or names - {"eda", "worker_receipt.json", "AGENT_COMPLETE.json"}:
            return False

        receipt = json.loads(_read_file(output / "worker_receipt.json", 65536))
        return receipt == self._receipt(self._request(output.name)) and validate_eda_report(output / "eda", self.contract.profile)

    def registered_tool(self):
        return RegisteredTool("run_eda", self.execute, self.validate, requires_approval=True)

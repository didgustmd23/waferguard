"""표형 입력 검사·생성을 별도 1회 승인으로 실행하는 기존 실행기 연결."""

from contextlib import contextmanager
from dataclasses import asdict
import functools
import json
from pathlib import Path
import re
import shutil

from .appcontainer_probe import launch_succeeded
from .approval_store import ApprovalStore
from .artifacts import _root, check_artifacts
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .delegated_approval import record_delegated
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor
from .input_snapshot import Snapshot, create_snapshot, verify_snapshot
from .isolated_eda import CancelSignal, _read_file
from .protected_store import StoreUnavailable, check_object, check_tree, create_store, store_lease, trusted_principals
from .snapshot_eda import _digest
from .split_evidence import read_committed_artifact
from .state import LocalStateStore
from .tabular_input_results import read_inspection, read_materialization
from .tabular_input_transport import code_manifest, run_input_worker
from .tabular_input_worker import build_packet, POLICY_VERSION, MAX_OUTPUT_BYTES
from .worker_contract import WorkerLimits, WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy
from .worker_runtime import MESSAGES as RUNTIME_MESSAGES
from .workflow_approval import approval_progress, preview_approval


# 입력 실행 저장소(실행 기록·승인 DB·작업 폴더) 보호 실패. 입력 데이터 오류와 구분한다.
STORE_MESSAGES = {
    "INPUT_STORE_PERMISSION_INVALID": "입력 실행 기록 저장소(작업 폴더의 input_runs)의 소유자·권한이 허용 범위와 다르거나 경로가 "
                                      "바뀌었습니다(다른 사용자·그룹의 쓰기·삭제·권한 변경, 상속 권한, 링크 등). 입력 데이터 문제가 "
                                      "아닙니다. 권한을 자동으로 고치지 않습니다. 이전 버전에서 만든 작업 폴더도 이 경우에 해당합니다. "
                                      "관리자가 해당 폴더를 확인한 뒤 새 작업으로 다시 시작하세요.",
    "INPUT_STORE_PREPARE_FAILED": "입력 실행 기록 저장소를 보호 권한으로 만들지 못했습니다. 디스크 공간과 작업 폴더 권한을 확인한 뒤 "
                                  "새 작업으로 다시 시작하세요.",
}


def code_id():
    """입력 동의와 위임 발급에 결합할 현재 코드 식별값이다."""
    return _digest(code_manifest())


class InputTool:
    """검사 또는 생성 하나만 등록하며 검증된 선행 검사만 생성에 복사한다."""

    def __init__(self, states, config, service):
        self.states, self.config, self.service = states, config, service
        self.cancel = CancelSignal(states)
        self.code_id = code_id()
        self.arguments = config["packet"]
        self.limits = WorkerLimits(**self.arguments["limits"])
        self.snapshot = Snapshot(**self.arguments["snapshot"])
        packet = build_packet(service.action, self.arguments["snapshot"], self.arguments["read_spec"],
                              self.arguments["limits"], self.arguments["code_id"], runtime_id=self.arguments["runtime_id"],
                              inspection_id=self.arguments["inspection_id"], inspection_sha256=self.arguments["inspection_sha256"],
                              max_output_bytes=self.arguments["max_output_bytes"])
        if _digest(packet) != _digest(self.arguments):
            raise ValueError("입력 설정 필드가 계약과 다릅니다.")
        self.config_id = _digest(packet)
        self.plan = ExecutionPlan(source_id=self.snapshot.source_id, config_id=self.config_id, policy_version=POLICY_VERSION,
                                  steps=(ExecutionStep("preflight", "run_preflight"),
                                         ExecutionStep("ingest", "run_ingest", json.dumps(packet))))

    def parent(self):
        """생성 인자에 고정한 선행 검사의 커밋·내용·입력·해석을 다시 확인한다."""
        if self.service.action == "inspect":
            if self.config["inspection_run"] is not None:
                raise ValueError("검사 단계의 선행 실행은 허용하지 않습니다.")
            return None
        parent = TabularInputService("inspect", root=self.service.root)
        folder, result = parent.verified(self.config["inspection_run"])
        report = result["inspection"]
        # 선행 검사도 같은 승인 런타임으로 실행된 결과여야 한다.
        inspected_with = parent._open(self.config["inspection_run"])[1].arguments["runtime_id"]
        if (report["status"] != "ready" or report["inspection_id"] != self.arguments["inspection_id"]
                or inspected_with != self.arguments["runtime_id"]
                or result["inspection_sha256"] != self.arguments["inspection_sha256"]
                or _digest(report["spec"]) != _digest(self.arguments["read_spec"])
                or _digest(report["snapshot"]) != _digest(self.arguments["snapshot"])
                or _digest(report["limits"]) != _digest(self.arguments["limits"])):
            raise ValueError("생성 대상과 선행 검사가 일치하지 않습니다.")
        return folder, result

    def _check_arguments(self, arguments):
        if _digest(arguments) != self.config_id or self.arguments["code_id"] != code_id():
            raise ValueError("입력·해석·코드가 승인 범위와 다릅니다.")
        if not verify_snapshot(Path(self.config["snapshot_directory"]), self.snapshot):
            raise ValueError("승인한 작업 복사본이 변경됐습니다.")
        self.parent()

    def _work_root(self, task_id):
        if type(task_id) is not str or re.fullmatch(r"[0-9a-f]{32}", task_id) is None:
            raise ValueError("내부 작업 ID가 필요합니다.")
        return _root(self.states.directory.parent) / "workers" / task_id

    def _request(self, task_id):
        root = self._work_root(task_id)
        files = tuple(str(root / "input" / name) for name in ("snapshot/input.bin", "snapshot/snapshot.json", "settings.json"))
        if self.service.action == "materialize":
            files += (str(root / "input/inspection.json"),)
        state = self.states.load()
        return WorkerRequest(task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
                             stage="ingest", action_id="run_ingest", source_id=self.snapshot.source_id,
                             config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=POLICY_VERSION,
                             limits=self.limits, input_files=files, dependency_directories=(),
                             output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"))

    def _stage(self, request):
        """승인 소비 뒤 작업 복사본·설정과 해시를 확인한 검사 sidecar만 복사한다."""
        self._check_arguments(self.arguments)
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir()
        # 런타임은 작업 폴더에 복사하지 않는다. transport가 공유 런타임을 골라 검증한다.
        for name in ("input", "tools", "output", "scratch"):
            (root / name).mkdir()
        source = Path(self.config["snapshot_directory"])
        staged = create_snapshot(source / "input.bin", root / "input/snapshot", max_input_bytes=self.snapshot.max_input_bytes)
        if staged != self.snapshot:
            raise ValueError("복사한 입력이 승인 대상과 다릅니다.")
        parent = self.parent()
        if parent is not None:
            raw = _read_file(parent[0] / "inspection.json", 16 * 1024 * 1024)
            import hashlib
            if hashlib.sha256(raw).hexdigest() != self.arguments["inspection_sha256"]:
                raise ValueError("복사할 검사 파일이 변경됐습니다.")
            (root / "input/inspection.json").write_bytes(raw)
        (root / "input/settings.json").write_text(json.dumps(self.arguments, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(tuple(root / name for name in ("input", "tools")),
                                            root / "output", root / "scratch", protected_roots=(self.states.directory, source)), self.limits)
        return root

    def _evidence(self, request, outcome):
        resource = outcome.get("launch", {}).get("resources", {})
        return (outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
                and resource.get("cleanup_confirmed") is True and type(resource.get("active_processes")) is int
                and resource["active_processes"] == 0
                and resource.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
                and resource.get("configured_process_limit") == self.limits.processes)

    def _read(self, directory, request, *, committed=False):
        reader = read_inspection if self.service.action == "inspect" else read_materialization
        return reader(directory, request, committed=committed)

    def execute(self, output, arguments):
        self._check_arguments(arguments)
        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))
        request = self._request(output.name)
        # 실행 런타임은 승인한 packet의 ID로만 고른다. 원본 환경이 바뀌었어도 다른 ID를 쓰지 않는다.
        outcome = run_input_worker(request, self._stage(request), self.cancel, runtime_id=self.arguments["runtime_id"])
        self._check_arguments(arguments)
        if outcome.get("runtime_error") is not None:
            # 원인 코드는 transport.json에 남고 runtime_failure로 읽는다. 실행기에는 고정 실패 분류만 넘긴다.
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))
        if not self._evidence(request, outcome):
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))
        launch = outcome.get("launch", {})
        if self.cancel.is_set() or launch.get("status") == "cancelled":
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))
        staged = _root(request.output_directory)
        if launch.get("status") == "exited" and launch.get("exit_code") == 2:
            reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))
            if reply.matches(request) and reply.status == "blocked_input" and not reply.artifact_refs:
                return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))
        if not launch_succeeded(launch):
            return ToolResult("failed_runtime", errors=("WORKER_FAILED",))
        reply = WorkerResult.from_json(_read_file(staged / "worker_result.json", 65536).decode("utf-8"))
        if ({path.name for path in staged.iterdir()} != {"operation", "worker_result.json"}
                or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != ("operation",)):
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))
        self._read(staged / "operation", request)
        for path in (staged / "operation").iterdir():
            if path.is_dir():
                shutil.copytree(path, output / path.name)
            else:
                shutil.copyfile(path, output / path.name)
        return ToolResult("passed", artifact_refs=("inspection.json", "binding.json") + (("table",) if self.service.action == "materialize" else ()))

    def validate(self, output, arguments):
        self._check_arguments(arguments)
        request = self._request(output.name)
        root = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        # 실행 기록은 신뢰 주체만 쓸 수 있어야 한다(worker·다른 사용자 ACE가 있으면 결과를 등록하지 않음).
        if not check_object(root / "transport.json", trusted_principals()):
            return False
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))
        # 실행 기록의 런타임은 승인한 packet의 ID와 같아야 한다(형식만 맞는 다른 ID 거부).
        runtime = outcome.get("runtime")
        if (self.cancel.is_set() or saved != request or not self._evidence(request, outcome)
                or type(runtime) is not dict or runtime.get("id") != self.arguments["runtime_id"]
                or not launch_succeeded(outcome.get("launch", {})) or not reply.matches(request)
                or reply.status != "succeeded" or reply.artifact_refs != ("operation",)):
            return False
        return self._read(output, request, committed=(output / "AGENT_COMPLETE.json").exists()) == self._read(Path(request.output_directory) / "operation", request)


def _preflight(output, arguments):
    """원본을 읽거나 worker를 실행하지 않는 실행기 준비 단계."""
    if arguments:
        return ToolResult("blocked_input")
    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def _protected(method):
    """공개 연산을 입력 실행 저장소 보유·검사 안에서 실행한다. 보유는 연산이 끝나면(예외 포함) 놓는다."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._store(create=method.__name__ == "create"):
            return method(self, *args, **kwargs)
    return wrapper


class TabularInputService:
    """검사·생성을 같은 공개 서비스 연산으로 연결하되 승인은 각각 소비한다."""

    def __init__(self, action, *, root=None):
        if action not in ("inspect", "materialize"):
            raise ValueError("등록되지 않은 입력 도구입니다.")
        self.action = action
        self.root = Path(root).absolute() if root is not None else Path(__file__).resolve().parents[2] / "logs/agents/tabular_input"

    @contextmanager
    def _store(self, *, create=False):
        """저장소 루트를 처음이면 보호 SD로 만들고, 보유한 핸들로 소유자·보호·ACE를 검사한다. 기존 권한을 고치지 않는다.

        보유 중에는 루트와 상위 경로(작업 폴더 등)의 이름 변경이 막힌다. 실패는 입력 데이터 오류와 다른 고정 코드다.
        """
        try:
            trusted = trusted_principals()
            if create:
                create_store(self.root, trusted)
        except (OSError, ValueError):
            raise StoreUnavailable("INPUT_STORE_PREPARE_FAILED") from None
        try:
            lease = store_lease(self.root, trusted)
            lease.__enter__()
        except StoreUnavailable:
            raise StoreUnavailable("INPUT_STORE_PERMISSION_INVALID") from None
        try:
            yield
        finally:
            lease.__exit__(None, None, None)

    def _directory(self, name, *, create=False):
        # workers는 같은 폴더의 작업 폴더 이름이라 실행 이름으로 쓰지 않는다.
        if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None or name == "workers":
            raise ValueError("안전한 새 실행 이름이 필요합니다.")
        base = self.root / self.action
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return _root(base) / name

    @_protected
    def create(self, name, *, snapshot_directory, snapshot, spec, limits, runtime_id, inspection_run=None,
               max_output_bytes=MAX_OUTPUT_BYTES):
        """검증된 입력으로 새 실행 기록만 만든다. 승인·worker는 시작하지 않는다.

        runtime_id는 호출자가 승인 전에 고정한 공유 런타임 구성 ID다(worker_runtime.current_runtime_id).
        """
        packet = build_packet(self.action, snapshot, spec, limits, code_id(), runtime_id=runtime_id,
                              inspection_id=None if self.action == "inspect" else "0" * 64,
                              inspection_sha256=None if self.action == "inspect" else "0" * 64, max_output_bytes=max_output_bytes)
        if self.action == "materialize":
            _, parent = TabularInputService("inspect", root=self.root).verified(inspection_run)
            packet.update(inspection_id=parent["inspection"]["inspection_id"], inspection_sha256=parent["inspection_sha256"])
        config = {"snapshot_directory": str(snapshot_directory), "packet": packet, "inspection_run": inspection_run}
        directory = self._directory(name, create=True)
        if directory.exists() or directory.is_symlink():
            raise ValueError("기존 실행은 덮어쓰지 않습니다.")
        states = LocalStateStore(directory)
        tool = InputTool(states, config, self)
        tool._check_arguments(packet)
        state = states.create(_current_actor(), policy_version=POLICY_VERSION)
        (directory / "selection.json").write_text(json.dumps(config, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        ApprovalStore(directory / "approvals.sqlite3")
        append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)

    def _open(self, name):
        directory = _root(self._directory(name))
        # 실행 기록·승인 DB·완료 결과는 신뢰 주체만 쓸 수 있어야 한다(worker·다른 사용자 ACE 거부).
        trusted = trusted_principals()
        if not (check_object(directory.parent, trusted) and check_tree(directory, trusted)):
            raise StoreUnavailable("INPUT_STORE_PERMISSION_INVALID")
        config = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))
        if type(config) is not dict or set(config) != {"snapshot_directory", "packet", "inspection_run"}:
            raise ValueError("입력 실행 기록의 필드가 올바르지 않습니다.")
        states = LocalStateStore(directory)
        if states.load().actor_id != _current_actor():
            raise ValueError("실행을 만든 계정에서만 접근하세요.")
        tool = InputTool(states, config, self)
        if config["packet"]["code_id"] != tool.code_id:
            raise ValueError("입력 실행 코드가 바뀌었습니다. 새 승인이 필요합니다.")
        manager = RulesExecutor(states, ApprovalStore(directory / "approvals.sqlite3"), tool.plan,
                                (RegisteredTool("run_preflight", _preflight, _validate_preflight),
                                 RegisteredTool("run_ingest", tool.execute, tool.validate, requires_approval=True)))
        return config, tool, manager

    def _summary(self, config):
        packet = config["packet"]
        return lambda tool: {"stage": tool.states.load().current_stage, "limits": packet["limits"],
                             "run_name": tool.states.directory.name, "action": self.action,
                             "snapshot_sha256": packet["snapshot"]["sha256"], "input_code_id": packet["code_id"],
                             "read_spec": packet["read_spec"], "runtime_id": packet["runtime_id"],
                             "inspection_id": packet["inspection_id"],
                             "inspection_sha256": packet["inspection_sha256"], "max_output_bytes": packet["max_output_bytes"]}

    @_protected
    def status(self, name):
        config, tool, manager = self._open(name)
        state = tool.states.load()
        actions = []
        if not state.blockers and state.status in {"created", "running", "awaiting_approval"}:
            progress = approval_progress(manager, state.current_stage)
            if not progress["task_exists"]:
                actions = [{"id": ("execute" if progress["valid_approval_ids"] else "approve") if state.status == "awaiting_approval" else "next"}]
        return {"status": state.status, "stage": state.current_stage, "revision": state.revision, "actions": actions}

    @_protected
    def preview_approval(self, name):
        config, tool, manager = self._open(name)
        return preview_approval(tool, manager, self._summary(config))

    @_protected
    def record_delegated_approval(self, name, preview, *, authorize):
        config, tool, manager = self._open(name)
        return record_delegated(tool, manager, self._summary(config), preview, authorize=authorize)

    @_protected
    def next(self, name, *, expected_stage, expected_revision, approval_id=None):
        _, tool, manager = self._open(name)
        state = tool.states.load()
        if type(expected_revision) is not int or state.current_stage != expected_stage or state.revision != expected_revision:
            raise ValueError("선택한 단계·revision이 바뀌었습니다.")
        return manager.run_next(approval_id=approval_id)

    @_protected
    def verified(self, name):
        """실제 승인 소비·커밋과 현재 결과 형식을 재검증한다."""
        _, tool, manager = self._open(name)
        if tool.states.load().status != "completed":
            raise ValueError("커밋된 입력 결과가 없습니다.")
        evidence = read_committed_artifact(tool.states.directory, manager.scope_for("ingest"))
        output = evidence.directory
        request = tool._request(output.name)
        return output, tool._read(output, request, committed=True)

    @_protected
    def source_contract(self, name):
        """생성 결과를 후속 v2 단계가 소비할 해시·승인 증거 계약으로 묶는다."""
        from src.preprocessing.tabular_bundle import BundleRef
        from .tabular_source_contract import TabularSourceContract
        if self.action != "materialize":
            raise ValueError("생성 완료 결과만 표형 출처로 쓸 수 있습니다.")
        folder, result = self.verified(name)
        config, tool, manager = self._open(name)
        packet = config["packet"]
        source = TabularSourceContract(snapshot_directory=config["snapshot_directory"], snapshot=tool.snapshot,
                                       bundle_directory=str(folder / "table"), bundle=BundleRef.from_dict(result["bundle"]),
                                       read_spec_json=json.dumps(packet["read_spec"]), inspection_id=packet["inspection_id"],
                                       inspection_sha256=packet["inspection_sha256"], max_output_bytes=packet["max_output_bytes"],
                                       materialization=read_committed_artifact(tool.states.directory, manager.scope_for("ingest")),
                                       limits=tool.limits, execution_code_id=packet["code_id"],
                                       execution_runtime_id=packet["runtime_id"])
        source.check_input()
        return source

    @_protected
    def runtime_failure(self, name):
        """현재 단계 실행이 공유 런타임·작업 폴더 보호 검사에서 멈췄으면 고정 코드를, 아니면 None을 돌려준다."""
        _, tool, manager = self._open(name)
        task = manager.approvals.get_task(manager.scope_for(tool.states.load().current_stage))
        if task is None:
            return None
        record = tool._work_root(manager._output(task).name) / "transport.json"
        if record.exists() and not check_object(record, trusted_principals()):
            return "INPUT_STORE_PERMISSION_INVALID"
        try:
            outcome = _decode(_read_file(record, 65536).decode("utf-8"))
        except (OSError, ValueError, UnicodeError):
            return None
        code = outcome.get("runtime_error") if type(outcome) is dict else None
        return code if code in RUNTIME_MESSAGES or code in STORE_MESSAGES else None

    @_protected
    def results(self, name):
        folder, _ = self.verified(name)
        return {"status": "completed", "items": [{"kind": "artifact_list", "label": self.action, "path": str(folder)}]}

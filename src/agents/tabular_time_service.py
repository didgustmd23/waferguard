"""표형 v2 시간 결측 복원의 미리보기·적용을 각각 1회 승인으로 실행하는 기존 실행기 연결. 입력 도구와 같은 보호 저장소·운송·결과 재검증을 쓴다.

입력 도구(tabular_input_*)의 코드 지문·실행 기록을 바꾸지 않도록 별도 정책(tabular_time_restore_v1)·실행 루트(<작업 폴더>/time_runs)를 쓴다.
적용 결과는 원래 생성 표를 바꾸지 않은 새 묶음(파생 표)이며, 내용 기반 출처(TimeRestoredSource)로 다음 단계가 다시 확인한다.
"""

from contextlib import contextmanager
from dataclasses import asdict
import functools
import hashlib
import json
from pathlib import Path
import re
import shutil

from src.preprocessing.tabular_bundle import BundleRef, inspect_bundle
from .appcontainer_probe import launch_succeeded
from .approval_store import ApprovalStore
from .artifacts import _root
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .delegated_approval import record_delegated
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor
from .isolated_eda import CancelSignal, _read_file
from .protected_store import StoreUnavailable, check_object, check_tree, create_store, store_lease, trusted_principals
from .snapshot_eda import _digest
from .split_evidence import read_committed_artifact
from .state import LocalStateStore
from .tabular_input_transport import CONTROL_FILES as INPUT_CONTROLS, WORKER_FILES as INPUT_WORKER_FILES, run_input_worker
# 파생 표 출처 계약은 하위 단계(분할·내보내기)가 같은 인터페이스로 쓰도록 출처 계약 모듈에 둔다. 기존 import를 위해 다시 내보낸다.
from .tabular_source_contract import TabularSourceContract, TimeRestoredSource
from .tabular_time_results import read_application, read_preview
from .tabular_time_worker import (BUNDLE_FILES, MAX_OUTPUT_BYTES, MAX_PREVIEW_BYTES, POLICY_VERSION, build_packet,
                                  rebuild_packet)
from .worker_contract import WorkerLimits, WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy
from .worker_runtime import MESSAGES as RUNTIME_MESSAGES


# ==========================================
# 고정 코드 목록
# - worker 복사 목록은 입력 worker 목록(공유 런타임 확인 포함)에 복원 판정 모듈과 시간 worker만 더한다
# - 승인 코드 지문은 입력 도구 지문 범위(조정기·동의·v2 어댑터 포함)에 시간 복원 부모 코드를 더한다
# ==========================================
ENTRY = "src/agents/tabular_time_worker.py"
WORKER_FILES = tuple(dict.fromkeys(INPUT_WORKER_FILES + ("src/preprocessing/time_restore.py", ENTRY)))
CONTROL_FILES = tuple(dict.fromkeys(INPUT_CONTROLS + WORKER_FILES + (
    "src/agents/tabular_time_service.py", "src/agents/tabular_time_results.py", "src/agents/tabular_time_flow.py",
)))
STORE_MESSAGES = {
    "TIME_STORE_PERMISSION_INVALID": "시간 복원 실행 기록 저장소(작업 폴더의 time_runs)의 소유자·권한이 허용 범위와 다르거나 경로가 "
                                     "바뀌었습니다. 데이터 문제가 아닙니다. 권한을 자동으로 고치지 않습니다. 관리자가 해당 폴더를 확인한 뒤 "
                                     "새 작업으로 다시 시작하세요.",
    "TIME_STORE_PREPARE_FAILED": "시간 복원 실행 기록 저장소를 보호 권한으로 만들지 못했습니다. 디스크 공간과 작업 폴더 권한을 확인하세요.",
}
WORKER_CODES = frozenset({"INPUT_INVALID", "TIME_PREVIEW_CHANGED"})


def code_manifest():
    """시간 복원 승인·계산·결과 검증에 관여하는 고정 파일의 지문을 반환한다."""
    from .artifacts import _fingerprint
    project = Path(__file__).resolve().parents[2]
    return {name: _fingerprint(project / name)["sha256"] for name in CONTROL_FILES}


def code_id():
    """결정 T·R 동의와 위임 발급에 결합할 현재 코드 식별값이다."""
    return _digest(code_manifest())


def run_time_worker(request, root, cancel_event, *, runtime_id):
    """입력 worker와 같은 공유 런타임·ACL·Job 운송으로 시간 worker만 실행한다."""
    return run_input_worker(request, root, cancel_event, runtime_id=runtime_id, entry=ENTRY, worker_files=WORKER_FILES,
                            manifest=code_manifest, container_prefix="waferguard.tabulartime.")


class TimeTool:
    """미리보기 또는 적용 하나만 등록한다. 적용은 검증된 미리보기 파일만 입력으로 복사한다."""

    def __init__(self, states, config, service):
        self.states, self.config, self.service = states, config, service
        self.cancel = CancelSignal(states)
        self.code_id = code_id()
        self.arguments = config["packet"]
        packet = rebuild_packet(self.arguments)
        self.source = TabularSourceContract.from_dict(config["source"])
        if (_digest(packet) != _digest(self.arguments) or packet["action"] != service.action
                or packet["source_id"] != self.source.source_id or packet["bundle"] != asdict(self.source.bundle)):
            raise ValueError("시간 복원 설정 필드가 계약과 다릅니다.")
        self.limits = WorkerLimits(**packet["limits"])
        self.config_id = _digest(packet)
        self.plan = ExecutionPlan(source_id=packet["source_id"], config_id=self.config_id, policy_version=POLICY_VERSION,
                                  steps=(ExecutionStep("preflight", "run_preflight"),
                                         ExecutionStep("ingest", "run_ingest", json.dumps(packet))))

    def parent(self):
        """적용 인자에 고정한 미리보기의 커밋·내용·규칙·입력·런타임을 다시 확인한다."""
        if self.service.action == "preview":
            if self.config["preview_run"] is not None:
                raise ValueError("미리보기 단계의 선행 실행은 허용하지 않습니다.")
            return None
        parent = TabularTimeService("preview", root=self.service.root)
        folder, result = parent.verified(self.config["preview_run"])
        previewed = parent._open(self.config["preview_run"])[1].arguments
        if (result["preview_sha256"] != self.arguments["preview_sha256"] or result["preview"]["status"] != "ready"
                or not result["preview"]["rows"]
                or any(_digest(previewed[key]) != _digest(self.arguments[key])
                       for key in ("source_id", "bundle", "rules", "rules_signature", "runtime_id", "limits", "code_id"))):
            raise ValueError("적용 대상과 확인한 미리보기가 일치하지 않습니다.")
        return folder, result

    def _check_arguments(self, arguments):
        if _digest(arguments) != self.config_id or self.arguments["code_id"] != code_id():
            raise ValueError("입력·규칙·코드가 승인 범위와 다릅니다.")
        self.source.check_input()
        self.parent()

    def _work_root(self, task_id):
        if type(task_id) is not str or re.fullmatch(r"[0-9a-f]{32}", task_id) is None:
            raise ValueError("내부 작업 ID가 필요합니다.")
        return _root(self.states.directory.parent) / "workers" / task_id

    def _request(self, task_id):
        root = self._work_root(task_id)
        files = tuple(str(root / "input/bundle" / name) for name in BUNDLE_FILES) + (str(root / "input/settings.json"),)
        if self.service.action == "apply":
            files += (str(root / "input/preview.json"),)
        state = self.states.load()
        return WorkerRequest(task_id=task_id, run_id=state.run_id, actor_id=state.actor_id,
                             stage="ingest", action_id="run_ingest", source_id=self.arguments["source_id"],
                             config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=POLICY_VERSION,
                             limits=self.limits, input_files=files, dependency_directories=(),
                             output_directory=str(root / "output/result"), scratch_directory=str(root / "scratch/task"))

    def _stage(self, request):
        """승인 소비 뒤 원래 생성 묶음·설정과 해시를 확인한 미리보기만 복사한다. 원래 묶음은 읽기만 한다."""
        self._check_arguments(self.arguments)
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir()
        for name in ("input", "tools", "output", "scratch"):
            (root / name).mkdir()
        source = Path(self.source.bundle_directory)
        staged = root / "input/bundle"
        staged.mkdir()
        for name in BUNDLE_FILES:
            shutil.copyfile(source / name, staged / name)
        inspect_bundle(staged, expected=self.source.bundle, max_bytes=self.source.bundle.size_bytes)
        parent = self.parent()
        if parent is not None:
            raw = _read_file(parent[0] / "preview.json", MAX_PREVIEW_BYTES)
            if hashlib.sha256(raw).hexdigest() != self.arguments["preview_sha256"]:
                raise ValueError("복사할 미리보기 파일이 변경됐습니다.")
            (root / "input/preview.json").write_bytes(raw)
        (root / "input/settings.json").write_text(json.dumps(self.arguments, ensure_ascii=False, sort_keys=True, allow_nan=False),
                                                  encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(tuple(root / name for name in ("input", "tools")),
                                              root / "output", root / "scratch", protected_roots=(self.states.directory, source)),
                             self.limits)
        return root

    def _evidence(self, request, outcome):
        resource = outcome.get("launch", {}).get("resources", {})
        return (outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
                and resource.get("cleanup_confirmed") is True and type(resource.get("active_processes")) is int
                and resource["active_processes"] == 0
                and resource.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
                and resource.get("configured_process_limit") == self.limits.processes)

    def _read(self, directory, request, *, committed=False):
        if self.service.action == "preview":
            return read_preview(directory, request, parent_directory=Path(self.source.bundle_directory), committed=committed)
        return read_application(directory, request, parent_directory=Path(self.source.bundle_directory), committed=committed)

    def execute(self, output, arguments):
        self._check_arguments(arguments)
        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))
        request = self._request(output.name)
        # 실행 런타임은 승인한 packet의 ID로만 고른다. 원본 환경이 바뀌었어도 다른 ID를 쓰지 않는다.
        outcome = run_time_worker(request, self._stage(request), self.cancel, runtime_id=self.arguments["runtime_id"])
        self._check_arguments(arguments)
        if outcome.get("runtime_error") is not None or not self._evidence(request, outcome):
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
        return ToolResult("passed", artifact_refs=("preview.json", "binding.json") + (("table",) if self.service.action == "apply" else ()))

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
        runtime = outcome.get("runtime")
        if (self.cancel.is_set() or saved != request or not self._evidence(request, outcome)
                or type(runtime) is not dict or runtime.get("id") != self.arguments["runtime_id"]
                or not launch_succeeded(outcome.get("launch", {})) or not reply.matches(request)
                or reply.status != "succeeded" or reply.artifact_refs != ("operation",)):
            return False
        return (self._read(output, request, committed=(output / "AGENT_COMPLETE.json").exists())
                == self._read(Path(request.output_directory) / "operation", request))


def _preflight(output, arguments):
    """원본을 읽거나 worker를 실행하지 않는 실행기 준비 단계."""
    if arguments:
        return ToolResult("blocked_input")
    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


def _protected(method):
    """공개 연산을 시간 복원 실행 저장소 보유·검사 안에서 실행한다. 보유는 연산이 끝나면(예외 포함) 놓는다."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._store(create=method.__name__ == "create"):
            return method(self, *args, **kwargs)
    return wrapper


class TabularTimeService:
    """미리보기·적용을 같은 공개 서비스 연산으로 연결하되 승인은 각각 소비한다."""

    def __init__(self, action, *, root):
        if action not in ("preview", "apply"):
            raise ValueError("등록되지 않은 시간 복원 도구입니다.")
        self.action = action
        self.root = Path(root).absolute()

    @contextmanager
    def _store(self, *, create=False):
        """저장소 루트를 처음이면 보호 SD로 만들고, 보유한 핸들로 소유자·보호·ACE를 검사한다. 기존 권한을 고치지 않는다."""
        try:
            trusted = trusted_principals()
            if create:
                create_store(self.root, trusted)
        except (OSError, ValueError):
            raise StoreUnavailable("TIME_STORE_PREPARE_FAILED") from None
        try:
            lease = store_lease(self.root, trusted)
            lease.__enter__()
        except StoreUnavailable:
            raise StoreUnavailable("TIME_STORE_PERMISSION_INVALID") from None
        try:
            yield
        finally:
            lease.__exit__(None, None, None)

    def _directory(self, name, *, create=False):
        if type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None or name == "workers":
            raise ValueError("안전한 새 실행 이름이 필요합니다.")
        base = self.root / self.action
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return _root(base) / name

    @_protected
    def create(self, name, *, source, rules, limits, runtime_id, preview_run=None, max_output_bytes=MAX_OUTPUT_BYTES):
        """검증된 출처·규칙으로 새 실행 기록만 만든다. 승인·worker는 시작하지 않는다.

        runtime_id는 호출자가 승인 전에 고정한 공유 런타임 구성 ID다. 적용은 미리보기와 같은 ID여야 한다.
        """
        if type(source) is not TabularSourceContract:
            raise ValueError("검증된 표형 출처 계약이 필요합니다.")
        packet = build_packet(self.action, source.source_id, asdict(source.bundle), rules, limits, code_id(),
                              runtime_id=runtime_id, max_output_bytes=max_output_bytes,
                              preview_sha256=None if self.action == "preview" else "0" * 64)
        if self.action == "apply":
            _, parent = TabularTimeService("preview", root=self.root).verified(preview_run)
            packet = rebuild_packet(dict(packet, preview_sha256=parent["preview_sha256"]))
        config = {"source": source.to_dict(), "packet": packet, "preview_run": preview_run}
        directory = self._directory(name, create=True)
        if directory.exists() or directory.is_symlink():
            raise ValueError("기존 실행은 덮어쓰지 않습니다.")
        states = LocalStateStore(directory)
        tool = TimeTool(states, config, self)
        tool._check_arguments(packet)
        state = states.create(_current_actor(), policy_version=POLICY_VERSION)
        (directory / "selection.json").write_text(json.dumps(config, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        ApprovalStore(directory / "approvals.sqlite3")
        append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)

    def _open(self, name):
        directory = _root(self._directory(name))
        trusted = trusted_principals()
        if not (check_object(directory.parent, trusted) and check_tree(directory, trusted)):
            raise StoreUnavailable("TIME_STORE_PERMISSION_INVALID")
        config = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))
        if type(config) is not dict or set(config) != {"source", "packet", "preview_run"}:
            raise ValueError("시간 복원 실행 기록의 필드가 올바르지 않습니다.")
        states = LocalStateStore(directory)
        if states.load().actor_id != _current_actor():
            raise ValueError("실행을 만든 계정에서만 접근하세요.")
        tool = TimeTool(states, config, self)
        if config["packet"]["code_id"] != tool.code_id:
            raise ValueError("시간 복원 실행 코드가 바뀌었습니다. 새 미리보기가 필요합니다.")
        manager = RulesExecutor(states, ApprovalStore(directory / "approvals.sqlite3"), tool.plan,
                                (RegisteredTool("run_preflight", _preflight, _validate_preflight),
                                 RegisteredTool("run_ingest", tool.execute, tool.validate, requires_approval=True)))
        return config, tool, manager

    def _summary(self, config):
        packet = config["packet"]
        return lambda tool: {"stage": tool.states.load().current_stage, "limits": packet["limits"],
                             "run_name": tool.states.directory.name, "action": self.action, "source_id": packet["source_id"],
                             "rules_signature": packet["rules_signature"], "time_code_id": packet["code_id"],
                             "runtime_id": packet["runtime_id"], "preview_sha256": packet["preview_sha256"],
                             "max_output_bytes": packet["max_output_bytes"]}

    @_protected
    def status(self, name):
        from .workflow_approval import approval_progress
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
        from .workflow_approval import preview_approval
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
            raise ValueError("커밋된 시간 복원 결과가 없습니다.")
        output = read_committed_artifact(tool.states.directory, manager.scope_for("ingest")).directory
        return output, tool._read(output, tool._request(output.name), committed=True)

    @_protected
    def derived_source(self, name):
        """적용 결과를 다음 단계가 다시 확인할 파생 표 출처 계약으로 묶는다."""
        if self.action != "apply":
            raise ValueError("적용 완료 결과만 파생 표 출처로 쓸 수 있습니다.")
        folder, result = self.verified(name)
        config, tool, manager = self._open(name)
        packet = config["packet"]
        source = TimeRestoredSource(parent=tool.source, bundle_directory=str(folder / "table"),
                                    bundle=BundleRef.from_dict(result["bundle"]),
                                    application=read_committed_artifact(tool.states.directory, manager.scope_for("ingest")),
                                    rules_json=json.dumps(packet["rules"], ensure_ascii=False, sort_keys=True),
                                    preview_sha256=packet["preview_sha256"],
                                    restoration_digest=result["preview"]["restoration_digest"],
                                    restored_count=result["preview"]["restored_count"], limits=tool.limits,
                                    execution_code_id=packet["code_id"], execution_runtime_id=packet["runtime_id"],
                                    max_output_bytes=packet["max_output_bytes"])
        source.check_input()
        return source

    @_protected
    def worker_diagnostic(self, name):
        """중단된 단계의 worker 분류(고정 stage·code)만 허용 목록으로 읽는다. 원본 값은 없다. 없으면 None."""
        _, tool, manager = self._open(name)
        task = manager.approvals.get_task(manager.scope_for(tool.states.load().current_stage))
        if task is None:
            return None
        try:
            item = _decode(_read_file(tool._work_root(manager._output(task).name) / "scratch/task/diagnostic.json", 4096).decode("utf-8"))
        except (OSError, ValueError, UnicodeError):
            return None
        if (type(item) is not dict or set(item) != {"stage", "code"} or item["code"] not in WORKER_CODES
                or item["stage"] not in ("REQUEST", "COMPUTE")):
            return None
        return dict(item)

    @_protected
    def runtime_failure(self, name):
        """현재 단계 실행이 공유 런타임·작업 폴더 보호 검사에서 멈췄으면 고정 코드를, 아니면 None을 돌려준다."""
        _, tool, manager = self._open(name)
        task = manager.approvals.get_task(manager.scope_for(tool.states.load().current_stage))
        if task is None:
            return None
        record = tool._work_root(manager._output(task).name) / "transport.json"
        if record.exists() and not check_object(record, trusted_principals()):
            return "TIME_STORE_PERMISSION_INVALID"
        try:
            outcome = _decode(_read_file(record, 65536).decode("utf-8"))
        except (OSError, ValueError, UnicodeError):
            return None
        code = outcome.get("runtime_error") if type(outcome) is dict else None
        # 공유 운송은 작업 폴더 권한 실패를 입력 도구 코드로 남긴다. 시간 복원 저장소 코드로 바꿔 안내한다.
        if code == "INPUT_STORE_PERMISSION_INVALID":
            return "TIME_STORE_PERMISSION_INVALID"
        return code if code in RUNTIME_MESSAGES or code in STORE_MESSAGES else None

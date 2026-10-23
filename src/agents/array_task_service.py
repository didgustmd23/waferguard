"""배열 통합 작업의 실행 도구와 서비스 API. 기존 실행기·승인 저장소·위임 발급·격리 운송을 그대로 쓴다.

공통 조정기는 이 서비스의 create·status·preview_approval·record_delegated_approval·next·results만 부른다.
단계마다 1회 승인을 소비한 뒤 작업 복사본만 격리 worker에 넘기고, 결과는 부모 검증기를 통과한 것만 커밋한다.
"""

import json
import os
from pathlib import Path
import re
import shutil
import sys

from .appcontainer_probe import launch_succeeded
from .approval_store import ApprovalStore
from .array_task_results import read_application, read_candidate, read_profile
from .array_task_transport import code_manifest, run_array_worker
from .array_task_worker import OUTPUTS, POLICY_VERSION, STAGES
from .artifacts import MARKER_NAME, _root, check_artifacts
from .audit import append_event
from .cli import _current_actor
from .contracts import ToolResult
from .delegated_approval import record_delegated
from .executor import ExecutionPlan, ExecutionStep, RegisteredTool, RulesExecutor
from .input_snapshot import Snapshot, create_snapshot
from .isolated_eda import CancelSignal, _read_file
from .snapshot_eda import _digest
from .state import LocalStateStore
from .worker_contract import WorkerLimits, WorkerRequest, WorkerResult, _decode
from .worker_paths import WorkerPathPolicy
from .workflow_approval import approval_progress, preview_approval


MODULES = {"array_profile": "audit", "array_prepare": "derive_preparation", "array_apply": "apply_preparation"}
PARENTS = {"array_apply": "array_prepare"}
DIAGNOSTIC_CODES = frozenset({"PASSED", "ACCESS_DENIED", "INPUT_INVALID", "IO_FAILED", "RUNTIME_FAILED"})
DETAILS = frozenset({
    "NPZ_MEMBERS_INVALID", "NPZ_TOO_LARGE", "NPZ_MEMBER_INVALID", "NPY_HEADER_INVALID", "NPY_DTYPE_UNSUPPORTED",
    "MAPPING_INVALID", "MAPPING_SAMPLES_INVALID", "MAPPING_CHANNEL_INVALID", "MAPPING_META_INVALID", "SAMPLE_COUNT_INVALID",
    "GROUP_MISSING", "SAMPLE_ID_INVALID", "LABEL_MISSING_UNDECIDED", "SPLIT_EMPTY", "NONFINITE_PRESENT",
    "CHANNEL_ALL_NONFINITE", "STATE_NONFINITE", "OUTPUT_NONFINITE",
})


def code_id():
    """배열 작업 연결 코드 지문. 동의 C·D 내용과 위임 발급 직전 값이 같아야 한다."""
    return _digest(code_manifest())


def _base():
    return Path(__file__).resolve().parents[2] / "logs" / "agents" / "array_task"


def _directory(module_id, name, *, create_parent=False):
    if module_id not in MODULES or type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) is None:
        raise ValueError("등록된 배열 모듈과 소문자로 시작하는 실행 이름이 필요합니다.")

    root = _base() / module_id

    if create_parent:
        root.mkdir(parents=True, exist_ok=True)

    return _root(root) / name


def read_worker_diagnostic(root):
    """worker가 scratch에 남긴 고정 분류만 허용 목록으로 읽는다. 없거나 형식이 다르면 None."""
    try:
        item = _decode(_read_file(root / "scratch" / "task" / "diagnostic.json", 4096).decode("utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None

    if (type(item) is not dict or set(item) - {"detail"} != {"stage", "code"} or item["code"] not in DIAGNOSTIC_CODES
            or type(item["stage"]) is not str or re.fullmatch(r"[A-Z_]{1,40}", item["stage"]) is None
            or ("detail" in item and item["detail"] not in DETAILS)):
        return None

    return dict(item)


# ==========================================
# 실행 도구: 승인 소비 후 격리 worker 실행·부모 검증·커밋
# ==========================================

class ArrayTaskTool:
    """한 단계만 등록한다. 선행 결과(적용 단계의 후보)는 커밋·검증된 출력만 입력으로 복사한다."""

    def __init__(self, states, config):
        self.states = states
        self.config = config
        self.module_id = config["module_id"]
        self.stage = MODULES[self.module_id]
        self.limits = WorkerLimits(**config["limits"])
        self.snapshot = Snapshot(**config["snapshot"])
        self.cancel = CancelSignal(states)
        self.code_id = code_id()
        self.arguments = dict(config["packet"], code_id=config["code_id"])
        _decode(json.dumps(self.arguments))
        self.config_id = _digest(self.arguments)
        self.plan = ExecutionPlan(
            source_id=self.snapshot.source_id, config_id=self.config_id, policy_version=POLICY_VERSION,
            steps=(ExecutionStep("preflight", "run_preflight"), ExecutionStep(self.stage, STAGES[self.stage], json.dumps(self.arguments))))

    def parent(self):
        """적용 단계의 선행 후보. 승인 인자에 결합한 실행 이름·후보 서명과 매번 다시 대조한다."""
        if self.config["parent"] is None:
            return None

        bound = self.arguments.get("candidate")
        service = Service(PARENTS[self.module_id])

        # 후보 서명은 완료 표식 해시와 검증된 내용을 묶는다. 다른 후보나 바뀐 후보는 적용하지 않는다.
        if (type(bound) is not dict or bound.get("run") != self.config["parent"]
                or service.signature(self.config["parent"]) != bound.get("signature")):
            raise ValueError("적용할 후보가 동의한 후보와 다릅니다.")

        folder, verified = service.verified(self.config["parent"])

        if verified["candidate"]["binding"]["snapshot"] != self.config["snapshot"]:
            raise ValueError("선행 후보를 검증하지 못했습니다.")

        return folder, verified

    def _check_arguments(self, arguments):
        if arguments != self.arguments or self.code_id != code_id() or self.config["code_id"] != self.code_id:
            raise ValueError("입력·설정·코드가 승인 범위와 다릅니다.")

        self.parent()

    def _work_root(self, task_id):
        if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
            raise ValueError("내부 작업 ID가 필요합니다.")

        return _root(self.states.directory.parent) / "workers" / task_id

    def _request(self, task_id):
        root = self._work_root(task_id)
        state = self.states.load()
        files = tuple(str(root / "input" / name) for name in ("snapshot/input.bin", "snapshot/snapshot.json", "array_task.json"))

        if self.stage == "apply_preparation":
            files += tuple(str(root / "input/candidate" / name) for name in ("assignment.json", "state.json", "candidate.json"))

        return WorkerRequest(
            task_id=task_id, run_id=state.run_id, actor_id=state.actor_id, stage=self.stage, action_id=STAGES[self.stage],
            source_id=self.plan.source_id, config_id=self.config_id, plan_id=self.plan.plan_id, policy_version=POLICY_VERSION,
            limits=self.limits, input_files=files, dependency_directories=(), output_directory=str(root / "output/result"),
            scratch_directory=str(root / "scratch/task"))

    def _stage(self, request):
        """승인 소비·예약 후 작업 복사본과(적용 단계면) 검증된 후보만 새 작업 폴더에 복사한다."""
        root = self._work_root(request.task_id)
        root.parent.mkdir(exist_ok=True)
        _root(root.parent)
        root.mkdir()

        for name in ("runtime", "input", "tools", "output", "scratch"):
            (root / name).mkdir()

        source = Path(self.config["snapshot_directory"])
        staged = create_snapshot(source / "input.bin", root / "input/snapshot", max_input_bytes=self.snapshot.max_input_bytes)

        if staged != self.snapshot:
            raise ValueError("작업용 복사본이 승인 대상과 다릅니다.")

        parent = self.parent()

        if parent is not None:
            shutil.copytree(parent[0], root / "input/candidate")

            if read_candidate(root / "input/candidate", Service(PARENTS[self.module_id]).packet(self.config["parent"])) != parent[1]:
                raise ValueError("복사한 후보가 검증한 후보와 다릅니다.")

        (root / "input/array_task.json").write_text(json.dumps(self.arguments, ensure_ascii=False), encoding="utf-8")
        (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
        Path(request.output_directory).mkdir()
        Path(request.scratch_directory).mkdir()
        request.check_policy(WorkerPathPolicy(
            tuple(root / name for name in ("runtime", "input", "tools")), root / "output", root / "scratch",
            protected_roots=(self.states.directory, source)), self.limits)
        return root

    def _evidence(self, request, outcome):
        resources = outcome.get("launch", {}).get("resources", {})
        return (outcome.get("request_id") == request.request_id and outcome.get("profile_cleanup") is True
                and isinstance(resources, dict) and resources.get("cleanup_confirmed") is True
                and resources.get("active_processes") == 0
                and resources.get("configured_job_memory_bytes") == self.limits.memory_mib * 1024 * 1024
                and resources.get("configured_process_limit") == self.limits.processes)

    def _read(self, folder):
        """단계별 부모 검증. 적용 단계는 선행 후보와의 교차 대조까지 한다."""
        if self.stage == "audit":
            return read_profile(folder)

        if self.stage == "derive_preparation":
            return read_candidate(folder, self.arguments)

        return read_application(folder, self.arguments, self.parent()[1])

    def execute(self, output, arguments):
        self._check_arguments(arguments)

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        request = self._request(output.name)
        outcome = run_array_worker(request, self._stage(request), self.cancel)
        self._check_arguments(arguments)
        launch = outcome.get("launch", {})
        name = OUTPUTS[self.stage]

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

        if {path.name for path in staged.iterdir()} != {name, "worker_result.json"}:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        reply = WorkerResult.from_json(_read_file(reply_path, 65536).decode("utf-8"))

        if not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != (name,) or self._read(staged / name) is None:
            return ToolResult("blocked_input", errors=("WORKER_INPUT_BLOCKED",))

        if self.cancel.is_set():
            return ToolResult("blocked_input", errors=("WORKER_CANCELLED",))

        shutil.copytree(staged / name, output / name)
        return ToolResult("passed", artifact_refs=(name,))

    def validate(self, output, arguments):
        self._check_arguments(arguments)
        name = OUTPUTS[self.stage]

        if self.cancel.is_set() or {path.name for path in _root(output).iterdir()} - {name, "AGENT_COMPLETE.json"}:
            return False

        request = self._request(output.name)
        root = self._work_root(output.name)
        saved = WorkerRequest.from_json(_read_file(root / "input/request.json", 65536).decode("utf-8"))
        outcome = _decode(_read_file(root / "transport.json", 65536).decode("utf-8"))
        reply = WorkerResult.from_json(_read_file(Path(request.output_directory) / "worker_result.json", 65536).decode("utf-8"))

        if (saved != request or not self._evidence(request, outcome) or not launch_succeeded(outcome.get("launch", {}))
                or not reply.matches(request) or reply.status != "succeeded" or reply.artifact_refs != (name,)):
            return False

        result = self._read(output / name)
        return result is not None and result == self._read(Path(request.output_directory) / name)

    def registered_tool(self):
        return RegisteredTool(STAGES[self.stage], self.execute, self.validate, requires_approval=True)


def _preflight(output, arguments):
    if arguments or os.name != "nt" or sys.version_info[:2] not in ((3, 10), (3, 12)):
        return ToolResult("blocked_input")

    (output / "preflight.json").write_text(json.dumps({"version": POLICY_VERSION}), encoding="utf-8")
    return ToolResult("passed")


def _validate_preflight(output, arguments):
    return not arguments and _decode(_read_file(output / "preflight.json", 4096).decode()) == {"version": POLICY_VERSION}


# ==========================================
# 서비스 API (공통 조정기가 부르는 연산)
# ==========================================

class Service:
    def __init__(self, module_id):
        if module_id not in MODULES:
            raise ValueError("등록되지 않은 배열 모듈입니다.")

        self.module_id, self.stage = module_id, MODULES[module_id]

    def create(self, name, *, snapshot_directory, snapshot, packet, limits, parent=None):
        """새 실행만 만든다. 기존 실행은 덮어쓰지 않으며 worker·승인은 시작하지 않는다."""
        if (parent is None) != (self.module_id not in PARENTS):
            raise ValueError("선행 단계 연결이 모듈 계약과 다릅니다.")

        directory = _directory(self.module_id, name, create_parent=True)

        if directory.exists() or directory.is_symlink():
            raise ValueError("기존 실행은 덮어쓰지 않습니다. 새 이름을 사용하세요.")

        config = {"module_id": self.module_id, "snapshot_directory": str(snapshot_directory), "snapshot": snapshot,
                  "packet": dict(packet, version=POLICY_VERSION, stage=self.stage, snapshot=snapshot, limits=limits),
                  "limits": limits, "parent": parent, "code_id": code_id()}
        payload = json.dumps(config, ensure_ascii=False, allow_nan=False)
        _decode(payload)
        states = LocalStateStore(directory)
        ArrayTaskTool(states, config)
        state = states.create(_current_actor(), policy_version=POLICY_VERSION)
        (directory / "selection.json").write_text(payload, encoding="utf-8")
        ApprovalStore(directory / "approvals.sqlite3")
        append_event(directory / "events.jsonl", "run_created", "RUN_CREATED", stage=state.current_stage, revision=state.revision)

    def _open(self, name):
        directory = _root(_directory(self.module_id, name))
        config = _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))

        if (type(config) is not dict or set(config) != {"module_id", "snapshot_directory", "snapshot", "packet", "limits", "parent", "code_id"}
                or config["module_id"] != self.module_id):
            raise ValueError("배열 실행 기록의 필드가 올바르지 않습니다.")

        states = LocalStateStore(directory)

        if states.load().actor_id != _current_actor():
            raise ValueError("실행을 만든 OS 계정에서만 접근하세요.")

        tool = ArrayTaskTool(states, config)

        if config["code_id"] != tool.code_id:
            raise ValueError("배열 작업 코드가 바뀌었습니다. 새 실행과 승인이 필요합니다.")

        manager = RulesExecutor(states, ApprovalStore(directory / "approvals.sqlite3"), tool.plan, (
            RegisteredTool("run_preflight", _preflight, _validate_preflight), tool.registered_tool()))
        return config, tool, manager

    def config(self, name):
        """실행 기록(입력 snapshot·설정). 계보 대조용이며 실행·승인은 하지 않는다."""
        directory = _root(_directory(self.module_id, name))
        return _decode(_read_file(directory / "selection.json", 65536).decode("utf-8"))

    def packet(self, name):
        config = self.config(name)
        return dict(config["packet"], code_id=config["code_id"])

    def _summary(self, config):
        def factory(tool):
            # 승인 요약은 단계·자원·입력 해시·코드 지문을 담는다. 위임 판정이 동의 내용과 대조한다.
            return {"stage": tool.states.load().current_stage, "limits": config["limits"],
                    "snapshot_sha256": config["snapshot"]["sha256"], "array_code_id": tool.code_id}

        return factory

    def status(self, name):
        config, tool, manager = self._open(name)
        state = tool.states.load()
        actions = []

        if not state.blockers and state.status in {"created", "running", "awaiting_approval"}:
            progress = approval_progress(manager, state.current_stage)

            if not progress["task_exists"]:
                actions = [{"id": ("execute" if progress["valid_approval_ids"] else "approve")
                            if state.status == "awaiting_approval" else "next"}]

        return {"status": state.status, "stage": state.current_stage, "revision": state.revision, "actions": actions}

    def preview_approval(self, name):
        config, tool, manager = self._open(name)
        return preview_approval(tool, manager, self._summary(config))

    def record_delegated_approval(self, name, preview, *, authorize):
        config, tool, manager = self._open(name)
        return record_delegated(tool, manager, self._summary(config), preview, authorize=authorize)

    def next(self, name, *, expected_stage, expected_revision, approval_id=None):
        config, tool, manager = self._open(name)
        state = tool.states.load()

        if type(expected_stage) is not str or type(expected_revision) is not int or state.current_stage != expected_stage or state.revision != expected_revision:
            raise ValueError("선택한 단계·revision이 변경됐습니다. 실행하지 않습니다.")

        return manager.run_next(approval_id=approval_id)

    def output(self, name):
        config, tool, manager = self._open(name)
        task = manager.approvals.get_task(manager.scope_for(self.stage))

        if task is None or task["status"] != "committed" or tool.states.load().status != "completed":
            raise ValueError("커밋된 결과가 없습니다.")

        output = manager._output(task)
        step = next(item for item in manager.plan.steps if item.stage == self.stage)

        # 다시 읽을 때도 완료 표식의 파일 해시를 대조한다. 커밋 뒤 바뀐 결과는 쓰지 않는다.
        if not check_artifacts(output, manager._context(step)).reusable:
            raise ValueError("커밋된 결과 파일이 완료 기록과 다릅니다.")

        return output

    def verified(self, name):
        """커밋된 결과를 부모 검증기로 다시 읽은 내용. 읽을 때마다 검증한다."""
        config, tool, manager = self._open(name)
        output = self.output(name)
        result = tool._read(output / OUTPUTS[self.stage])

        if result is None:
            raise ValueError("결과 검증을 통과하지 못했습니다.")

        return output / OUTPUTS[self.stage], result

    def signature(self, name):
        """커밋된 결과의 서명. 완료 표식 바이트와 검증된 내용을 함께 해시한다. 인증 서명은 아니다."""
        import hashlib

        folder, result = self.verified(name)
        marker = _read_file(folder.parent / MARKER_NAME, 1024 * 1024)
        content = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        return hashlib.sha256(hashlib.sha256(marker).digest() + content).hexdigest()

    def results(self, name):
        folder, _ = self.verified(name)
        return {"status": "completed", "items": [{"kind": "artifact_list", "label": self.stage, "path": str(folder)}]}

    def worker_diagnostic(self, name):
        """중단된 단계의 worker 진단 분류(고정 stage·code·세부 코드)만 돌려준다."""
        config, tool, manager = self._open(name)
        task = manager.approvals.get_task(manager.scope_for(tool.states.load().current_stage))

        if task is None:
            return None

        return read_worker_diagnostic(tool._work_root(manager._output(task).name))


def service(module_id):
    return Service(module_id)

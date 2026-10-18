"""WM 공통 진입점에서 기존 승인 화면·실행기를 호출한다. 자동 진행은 하지 않는다."""

from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

from .approvals import validate_approval
from .artifacts import _fingerprint
from .executor import ExecutionResult, RulesExecutor
from .state import LocalStateStore
from .wm_module_catalogue import _cli
from .wm_module_plan import load_wm_modules
from .wm_run_bindings import load_bindings
from .wm_run_evidence import SPECS, WmRunReader, _scope, load_run_tool
from .wm_workflow_guidance import recommend_next
from .worker_paths import _path


def _approved_id(tool, identifier):
    """다른 유효 승인이 있어도 사용자가 지정한 정확한 ID·범위만 허용한다."""
    if type(identifier) is not str or re.fullmatch(r"[a-f0-9]{32}", identifier) is None:
        raise ValueError("실행에는 현재 단계의 정확한 승인 ID가 필요합니다.")

    path = _path(tool.states.directory / "approvals.sqlite3", directory=False)

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute("SELECT scope_json,created_at,expires_at,status FROM approvals WHERE approval_id=?",
                                 (identifier,)).fetchone()

    if row is None:
        raise ValueError("지정한 승인 기록이 없습니다.")

    from .approval_store import ApprovalStore

    # 기존 역직렬화·검증 규칙을 쓰되 저장소 인스턴스나 쓰기 트랜잭션은 만들지 않는다.
    record = ApprovalStore._record(dict(zip(("approval_id", "scope_json", "created_at", "expires_at", "status"),
                                           (identifier, *row))))
    scope = _scope(tool, tool.states.load().current_stage)

    if validate_approval(record, scope, now=datetime.now(timezone.utc)) != "APPROVAL_VALID":
        raise ValueError("승인 범위·유효기간·미사용 상태가 현재 단계와 다릅니다.")


class _ApprovalStates(LocalStateStore):
    """기존 승인 화면이 확인 도중 다른 단계로 바뀌면 저장 전에 거부한다."""

    def __init__(self, states, check):
        super().__init__(states.directory)
        self._check = check

    def load(self):
        return self._check()

    def save(self, *args, **kwargs):
        raise ValueError("승인 확인 단계에서는 실행 상태를 변경하지 않습니다.")


class _SelectedExecutor(RulesExecutor):
    """기존 OS 실행 잠금을 획득한 뒤에도 요청한 단계가 그대로인지 확인한다."""

    def __init__(self, manager, check, tool):
        super().__init__(manager.states, manager.approvals, manager.plan, tuple(manager.tools.values()))
        self._selection_check = check
        self._selected_tool = tool

    def _run_step(self, state, approval_id):
        # 안내 후 다른 실행자가 먼저 단계를 진행한 경우 다음 단계를 대신 실행하지 않는다.
        try:
            self._selection_check()

            if approval_id is not None:
                _approved_id(self._selected_tool, approval_id)
        except (ValueError, TypeError, OSError, KeyError, RuntimeError, sqlite3.Error):
            return ExecutionResult("blocked_input", "WM_SELECTION_CHANGED", state.current_stage)

        return super()._run_step(state, approval_id)


def dispatch_stage(pipeline, bindings, step_id, name, stage, command, *, approval_id=None):
    """명시한 작업의 한 단계만 호출한다. 승인 화면의 사용자 입력을 대행하지 않는다."""
    if command not in {"next", "approve"}:
        raise ValueError("공통 실행 명령은 next 또는 approve만 허용합니다.")

    # 사용자 CLI의 상대경로만 현재 작업 폴더 기준으로 절대경로화한다.
    # worker 경로 검증 자체는 완화하지 않으며 링크·탈출 검사도 그대로 적용한다.
    paths = [_path(Path(value).absolute(), directory=False) for value in (pipeline, bindings)]

    if any(path.stat().st_size > 65536 for path in paths):
        raise ValueError("계획·배정 파일은 각각 64 KiB 이하여야 합니다.")

    stamps = [_fingerprint(path) for path in paths]
    plan = load_wm_modules(paths[0])
    runs = load_bindings(paths[1], plan)
    advice = recommend_next(plan, runs)
    guidance = advice["guidance"]

    if (advice["status"] != "needs_user_action" or advice.get("step_id") != step_id
            or runs.get(step_id) != name or guidance.get("run_name") != name):
        raise ValueError("요청한 작업이 현재 진행 가능한 첫 미완료 단계가 아닙니다.")

    action = guidance["action"]

    if command == "approve":
        if action != "approve_stage" or approval_id is not None:
            raise ValueError("승인 대기 단계만 승인할 수 있습니다. 유효 승인은 중복 발급하지 않습니다.")
    elif action not in {"continue_local_stage", "request_approval", "execute_approved_stage"}:
        raise ValueError("현재 상태에서는 다음 단계를 실행할 수 없습니다.")
    elif (action == "execute_approved_stage") != (approval_id is not None):
        raise ValueError("승인된 단계 실행에만 승인 ID를 명시하세요. 승인 요청과 실행은 구분합니다.")

    step = next(item for item in plan.steps if item.step_id == step_id)
    tool, module, _, _ = load_run_tool(step.module_id, name, require_complete=False)
    states = tool.states
    expected = states.load()

    if expected.current_stage != stage:
        raise ValueError("지정한 내부 단계가 현재 단계와 다릅니다. 상태를 다시 확인하세요.")

    reader = WmRunReader()
    reader.check_parent(step.module_id, tool)
    reference = dict(step.inputs)["evidence"]

    if reference != "source.evidence":
        parent_name = runs[reference.split(".", 1)[0]]

        if getattr(tool, SPECS[step.module_id][1], None) != parent_name:
            raise ValueError("계획에 배정된 부모와 실제 작업의 부모가 다릅니다.")

    def check_selection():
        """입력 파일·상태 변경 시 기존 범위로만 멈춘다. 새 작업을 선택하지 않는다."""
        if any(_fingerprint(_path(path, directory=False)) != stamp for path, stamp in zip(paths, stamps)):
            raise ValueError("계획 또는 배정이 변경되었습니다.")

        latest = states.load()

        if latest != expected or tool.cancel.is_set():
            raise ValueError("선택한 작업의 상태가 변경되었거나 취소되었습니다.")

        return latest

    check_selection()
    reader.check_unchanged()

    if approval_id is not None:
        _approved_id(tool, approval_id)

    cli = _cli(module.cli_id)

    if command == "approve":
        # 기존 대화형 화면·확인 문구·잠금·범위 재검사를 그대로 유지한다.
        tool.states = _ApprovalStates(states, check_selection)
        result = cli.approve(tool)
    else:
        manager = cli.executor(tool)
        result = asdict(_SelectedExecutor(manager, check_selection, tool).run_next(approval_id=approval_id))

    return {**result, "step_id": step_id, "run_name": name,
            "worker_started": tool.worker_started, "local_only": True}

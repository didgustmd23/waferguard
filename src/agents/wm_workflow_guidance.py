"""WM 계획의 다음 행동을 로컬에서 안내한다. 생성·승인·실행 권한은 제공하지 않는다."""

from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
import sqlite3

from .approvals import ApprovalRecord, validate_approval
from .artifacts import ArtifactContext, _fingerprint, _root, check_artifacts
from .cli import _current_actor
from .state import LocalStateStore
from .wm_module_catalogue import _cli, wm_modules
from .wm_run_bindings import inspect_bindings
from .wm_run_evidence import SPECS, WmRunReader, _committed, _scope, check_run_name, load_run_tool
from .worker_paths import _path


def _state(module, name):
    """중단 조건을 먼저 읽는다. 보안 잠금·취소를 다음 실행 명령으로 바꾸지 않는다."""
    check_run_name(name)
    directory = _root(_cli(module.cli_id)._directory(name))
    path = _path(directory / "state.json", directory=False)
    stamp = _fingerprint(path)
    state = LocalStateStore(directory).load()

    if state.actor_id != _current_actor() or _fingerprint(path) != stamp:
        raise ValueError("작업 사용자 또는 조회 중 상태가 변경되었습니다.")

    return state, (directory / "CANCEL_REQUESTED").exists()


def _query_progress(tool, module, *, now):
    """정확한 현재 범위의 승인만 조회한다. 만료 상태를 DB에 기록하지도 않는다."""
    state = tool.states.load()
    scope = _scope(tool, state.current_stage)
    scope_json = json.dumps(asdict(scope), ensure_ascii=False, sort_keys=True)
    database = _path(tool.states.directory / "approvals.sqlite3", directory=False)

    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        task = connection.execute("SELECT task_id,status FROM task_runs WHERE scope_json=?", (scope_json,)).fetchone()
        reservation = connection.execute("SELECT reservation_id FROM reservations WHERE scope_json=?", (scope_json,)).fetchone()
        records = connection.execute(
            "SELECT approval_id,created_at,expires_at,status FROM approvals "
            "WHERE scope_json=? AND status='pending' LIMIT 33", (scope_json,),
        ).fetchall()

        # 현재 단계가 시작되지 않았을 때만 앞 단계 완료 여부를 대조한다.
        # 예약만 남거나 종료가 불명확하면 자동 재시도 명령을 제시하지 않는다.
        stages = [step.stage for step in tool.plan.steps]

        if state.current_stage not in stages or len(records) > 32:
            raise ValueError("현재 단계 또는 승인 조회 범위가 올바르지 않습니다.")

        previous = []

        for stage in stages[:stages.index(state.current_stage)]:
            prior_scope = _scope(tool, stage)
            prior_json = json.dumps(asdict(prior_scope), ensure_ascii=False, sort_keys=True)
            row = connection.execute("SELECT task_id,status FROM task_runs WHERE scope_json=?", (prior_json,)).fetchone()

            if row is None or row[1] != "committed" or re.fullmatch(r"[a-f0-9]{32}", row[0]) is None:
                raise ValueError("현재 단계 앞의 완료 기록이 필요합니다.")

            context = ArtifactContext(prior_scope.source_id, prior_scope.config_id, prior_scope.plan_id,
                                      prior_scope.policy_version, prior_scope.tool_id)

            if not check_artifacts(tool.states.directory / "attempts" / row[0], context).reusable:
                raise ValueError("선행 단계 산출물의 무결성이 다릅니다.")

            previous.append(stage)

    for stage in previous:
        if stage in module.review_stages:
            _committed(tool, stage)

    valid = False

    for identifier, created, expires, status in records:
        record = ApprovalRecord(identifier, scope, datetime.fromisoformat(created), datetime.fromisoformat(expires), status)
        valid = valid or validate_approval(record, scope, now=now) == "APPROVAL_VALID"

    return {"task_exists": task is not None or reservation is not None,
            "valid_approval_available": valid,
            "requires_approval": state.current_stage in (module.worker_stage, *module.review_stages)}


def _guidance(action, message, module=None, name=None, command=None, *, approval=False):
    """임의 셸 문자열을 생성하지 않고 고정 CLI·하위 명령·검증된 작업 이름만 반환한다."""
    result = {"action": action, "message": message}

    if module is not None:
        result["cli"] = "src.agents." + module.cli_id

    if name is not None:
        result["run_name"] = name

    if command is not None:
        result["command"] = command
        result["approval_id_required"] = approval

    return result


def _pending_action(plan, runs, step, module, *, now):
    """미완료 입력도 코드·부모 증거를 확인한 뒤 조건부 안내만 한다."""
    name = runs[step.step_id]
    state, cancel_file = _state(module, name)

    if "SECURITY_LOCKED" in state.blockers:
        return _guidance("security_review", "보안 잠금 상태입니다. 검토 없이 진행하지 마세요.")

    if state.cancel_requested or cancel_file or state.status == "cancelled":
        return _guidance("stop_cancelled", "취소된 작업입니다. 기존 승인으로 재개하지 않습니다.")

    if state.blockers or state.status in {"blocked_input", "failed_runtime"}:
        return _guidance("review_local_report", "검토 필요 또는 실패 상태입니다. 로컬 보고서를 먼저 확인하세요.", module, name, "status")

    if state.status not in {"created", "running", "awaiting_approval"}:
        return _guidance("repeat_verification", "상태가 바뀌었습니다. 읽기 전용 연결 검사를 다시 수행하세요.")

    tool, _, _, stamps = load_run_tool(step.module_id, name, require_complete=False)
    reader = WmRunReader()
    reader.check_parent(step.module_id, tool)
    reference = dict(step.inputs)["evidence"]

    if reference != "source.evidence":
        expected = runs[reference.split(".", 1)[0]]

        if getattr(tool, SPECS[step.module_id][1], None) != expected:
            raise ValueError("계획과 다른 선행 작업입니다.")

    progress = _query_progress(tool, module, now=now)
    reader.check_unchanged()

    for filename, expected in stamps.items():
        if _fingerprint(_path(tool.states.directory / filename, directory=False)) != expected:
            raise ValueError("안내 준비 중 실행 기록이 변경되었습니다.")

    latest, cancel_file = _state(module, name)

    if latest != state or cancel_file:
        raise ValueError("안내 준비 중 작업 상태가 변경되었습니다.")

    if progress["task_exists"]:
        return _guidance("wait_or_review_task", "이미 예약·실행·완료 이력이 있습니다. 자동 재시도하지 말고 상태를 확인하세요.", module, name, "status")

    if state.status == "awaiting_approval":
        if not progress["requires_approval"]:
            raise ValueError("승인이 필요하지 않은 단계가 승인 대기 상태입니다.")

        if progress["valid_approval_available"]:
            return _guidance("execute_approved_stage", "현재 범위의 유효한 미사용 승인이 있습니다. 실행 시 승인 ID를 직접 지정하세요.",
                             module, name, "next", approval=True)

        return _guidance("approve_stage", "현재 단계의 설정·영향을 확인하고 사용자 승인을 진행하세요. 만료·소비된 승인은 재사용하지 않습니다.",
                         module, name, "approve")

    if progress["requires_approval"]:
        return _guidance("request_approval", "다음 단계 호출로 승인 대기 상태를 요청하세요. 아직 승인하거나 worker를 실행하는 명령은 아닙니다.",
                         module, name, "next")

    return _guidance("continue_local_stage", "승인이 필요 없는 현재 로컬 단계 하나를 진행할 수 있습니다. 실행 직전에 기존 실행기가 다시 검증합니다.",
                     module, name, "next")


def recommend_next(plan, runs, *, now=None):
    """전체 완료는 재실행 없이 종료하며, 안내 결과를 실행 허가나 승인으로 저장하지 않는다."""
    now = datetime.now(timezone.utc) if now is None else now

    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("안내 시각에는 시간대가 필요합니다.")

    inspected = inspect_bindings(plan, runs)
    base = {"pipeline_id": plan.pipeline_id, "steps": inspected["steps"], "worker_started": False,
            "approval_consumed": False, "advisory_only": True, "local_only": True}

    if inspected["status"] == "bindings_verified":
        return {**base, "status": "completed", "guidance": _guidance("finish", "모든 단계의 완료·부모 연결이 확인됐습니다. 재실행할 작업이 없습니다.")}

    stop = inspected.get("next_action")

    if inspected["status"] == "verification_failed" or type(stop) is not dict:
        return {**base, "status": "blocked", "guidance": _guidance("repeat_verification", "조회 중 변경을 확인했습니다. 재검증 전에는 진행하지 마세요.")}

    step = next(item for item in plan.steps if item.step_id == stop["step_id"])
    module = next(item for item in wm_modules() if item.contract.module_id == step.module_id)

    if stop["code"] == "RUN_NOT_ASSIGNED":
        guidance = _guidance("select_or_create_run", "기존 작업을 선택하거나 생성 설정을 확인한 뒤 배정 파일에 연결하세요.", module, command="create --help")
    elif stop["code"] == "RUN_NOT_COMPLETE":
        try:
            guidance = _pending_action(plan, runs, step, module, now=now)
        except (ValueError, TypeError, OSError, KeyError, RuntimeError, StopIteration):
            guidance = _guidance("review_local_evidence", "현재 코드·입력·부모 증거·승인 기록을 확인할 수 없습니다. 자동 진행하지 않습니다.")
    else:
        guidance = _guidance("review_local_evidence", "작업 연결 또는 완료 증거가 다릅니다. 로컬 기록을 검토하세요.")

    blocked = guidance["action"] in {"security_review", "stop_cancelled", "review_local_report",
                                      "review_local_evidence", "repeat_verification", "wait_or_review_task"}
    return {**base, "status": "blocked" if blocked else "needs_user_action",
            "step_id": step.step_id, "guidance": guidance}

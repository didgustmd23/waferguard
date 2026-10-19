"""화면·터미널과 독립적인 승인 미리보기·저장. 작업 실행은 기존 실행기가 담당한다."""

from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import sqlite3

from .approval_store import ApprovalStore
from .approvals import validate_approval
from .audit import append_event
from .executor import _run_lock
from .worker_paths import _path


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class WorkflowApprovalPreview:
    """사용자가 확인할 범위의 불변 복사본. 승인 ID·실행 권한을 포함하지 않는다."""

    payload_json: str

    def to_dict(self):
        return json.loads(self.payload_json)


def approval_progress(manager, stage):
    """현재 범위의 미사용 승인·작업 유무를 읽기 전용으로 확인한다."""
    scope = manager.scope_for(stage)
    database = _path(manager.states.directory / "approvals.sqlite3", directory=False)
    scope_json = _json(asdict(scope))

    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        rows = connection.execute(
            "SELECT * FROM approvals WHERE scope_json=? AND status='pending' LIMIT 33", (scope_json,),
        ).fetchall()
        task = connection.execute("SELECT status FROM task_runs WHERE scope_json=?", (scope_json,)).fetchone()
        reservation = connection.execute("SELECT 1 FROM reservations WHERE scope_json=?", (scope_json,)).fetchone()

    if len(rows) > 32:
        raise ValueError("현재 범위의 승인 조회 상한을 초과했습니다.")

    now = datetime.now(timezone.utc)
    pending = [ApprovalStore._record(row) for row in rows]
    valid = [item.approval_id for item in pending if validate_approval(item, scope, now=now) == "APPROVAL_VALID"]
    return {"valid_approval_ids": valid, "task_exists": task is not None or reservation is not None}


def preview_approval(tool, manager, summary_factory):
    """입력·코드·상태·필수 승인 단계를 확인한다. 승인 발급·worker 실행은 없다."""
    state = tool.states.load()
    step = next((item for item in manager.plan.steps if item.stage == state.current_stage), None)

    if (state.status != "awaiting_approval" or state.cancel_requested or state.blockers
            or tool.cancel.is_set() or step is None
            or not manager.tools[step.action_id].requires_approval):
        raise ValueError("취소·차단되지 않은 필수 승인 대기 단계가 필요합니다.")

    tool._check_arguments(tool.arguments)
    progress = approval_progress(manager, state.current_stage)

    if progress["valid_approval_ids"] or progress["task_exists"]:
        raise ValueError("현재 범위에 유효 승인 또는 작업 이력이 있습니다. 중복 발급하지 않습니다.")

    summary = summary_factory(tool)
    scope = manager.scope_for(state.current_stage)

    if tool.states.load() != state or tool.cancel.is_set():
        raise ValueError("승인 미리보기 중 실행 상태가 변경되었습니다.")

    return WorkflowApprovalPreview(_json({"schema_version": "workflow_approval_preview_v1",
                                         "state": asdict(state), "scope": asdict(scope),
                                         "summary": summary}))


def record_approval(tool, manager, summary_factory, preview, *, confirmed=False):
    """확인한 같은 범위를 잠금 안에서 재검사한 뒤 한 번만 저장한다."""
    if confirmed is not True or not isinstance(preview, WorkflowApprovalPreview):
        raise ValueError("승인 범위에 대한 명시적 사용자 확인이 필요합니다.")

    with _run_lock(tool.states.directory) as acquired:
        if not acquired:
            raise ValueError("다른 작업이 진행 중입니다.")

        if preview_approval(tool, manager, summary_factory) != preview:
            raise ValueError("사용자 확인 이후 입력·설정·코드·단계가 변경되었습니다.")

        state = tool.states.load()
        scope = manager.scope_for(state.current_stage)
        append_event(tool.states.directory / "events.jsonl", "approval_requested", "LOCAL_REVIEW_CONFIRMED",
                     stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    return {"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat()}

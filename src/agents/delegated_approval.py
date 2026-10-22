"""작업 동의 범위 안의 위임 승인 발급. 기존 실행기·승인 저장소를 쓰는 모든 서비스가 같은 경로를 쓴다.

공통 workflow_approval.record_approval과 같은 잠금·범위 재검사·중복 발급 거부를 거치되, 동의 판정을 잠금 안에서 하고
직접 확인 이벤트 대신 위임 이벤트를 남긴다. 공통 workflow_approval.py는 웨이퍼 모듈 코드 지문에 포함돼 수정하지 않는다.
"""

from dataclasses import asdict
import hashlib
import json

from .audit import append_event
from .executor import _run_lock
from .workflow_approval import WorkflowApprovalPreview, preview_approval


def record_delegated(tool, manager, summary_factory, preview, *, authorize):
    """잠금 안에서 미리보기를 다시 계산해 같을 때만 동의 판정 후 1회 승인을 기록한다."""
    if not isinstance(preview, WorkflowApprovalPreview) or not callable(authorize):
        raise ValueError("현재 승인 미리보기와 동의 범위 판정이 필요합니다.")

    with _run_lock(tool.states.directory) as acquired:
        if not acquired:
            raise ValueError("다른 작업이 진행 중입니다.")

        if preview_approval(tool, manager, summary_factory) != preview:
            raise ValueError("동의 판정 이후 입력·설정·코드·단계가 변경되었습니다.")

        authorize(preview.to_dict())
        state = tool.states.load()
        scope = manager.scope_for(state.current_stage)
        append_event(tool.states.directory / "events.jsonl", "approval_requested", "DELEGATED_APPROVAL_RECORDED",
                     stage=state.current_stage, revision=state.revision)
        record = manager.approvals.record_confirmed_approval(scope)

    digest = hashlib.sha256(json.dumps(asdict(scope), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"approval_id": record.approval_id, "expires_at": record.expires_at.isoformat(), "scope_sha256": digest}

"""CSV·WM에서 같은 승인 중심 조작을 제공한다. 렌더링만으로 실행하지 않는다."""

from .approval_flow import ApprovalFlow, mark_flow_pending
from .commentary_view import after_completion, render_classification
from .progress_view import show_operation_progress
from .runtime_notice import render_permissions, require_execution_permissions
from .workflow_feedback import STATE_LABELS, STAGE_LABELS, render_approval_summary, show_error


def render_approval_flow(st, controller, *, workflow):
    flow = getattr(controller, "approval_flow", None)
    if flow is None:
        flow = ApprovalFlow(controller, workflow=workflow)
        controller.approval_flow = flow

    st.text("작업 이름: " + controller.name)
    permission_ok = render_permissions(st)
    st.caption("시작 후 다음 승인 지점까지 자동 진행합니다. 승인하면 해당 범위를 실행합니다. 실패 시 자동 재시도하지 않습니다.")
    state = flow.state or (controller.state if workflow else None)
    runtime = (state or {}).get("runtime") if workflow else None
    if workflow and runtime is None and controller.creation_receipt:
        runtime = controller.creation_receipt.get("runtime")
    runtime_ok = runtime is None or runtime["supported"]
    if not runtime_ok:
        st.warning(runtime.get("notice", "현재 환경은 이 모듈의 실행 지원 조건에 맞지 않습니다."))
    if state:
        stage = state["stage"]
        stage_label = STAGE_LABELS.get(stage, stage)
        if getattr(controller, "module_id", None) == "wm_cache_prepare" and stage == "commit_split":
            stage_label = "캐시 설정 확정"
        st.text("상태: " + STATE_LABELS.get(state["status"], state["status"]) + " / 단계: " + stage_label)

    if flow.busy:
        st.info("현재 요청을 처리 중입니다. 버튼을 잠갔습니다. 다음 승인 지점 또는 완료까지 기다려 주세요.")

    if flow.error or controller.error:
        st.error("요청이 중단됐거나 결과가 불명확합니다. 5 결과 확인에서 저장된 상태를 확인하세요. 자동 재시도하지 않습니다.")
        return

    if state and state["status"] in {"completed", "blocked_input", "failed_runtime", "security_locked", "cancelled"}:
        st.info("작업이 완료되었거나 중단되었습니다. 5 결과 확인에서 보고서를 확인하세요.")
        return

    if not flow.started:
        render_classification(st, controller, disabled=flow.busy)
        st.button("작업 시작·승인 지점까지 진행", key="approval_flow_start", disabled=flow.busy or not runtime_ok,
                  on_click=mark_flow_pending, args=(st, flow, "start"))

    if flow.ready and flow.summary is not None:
        st.subheader("이번 단계의 승인 범위")
        # backend의 미리보기는 승인 저장만 설명한다. UI의 결합 동작을 명확히 안내한다.
        summary = dict(flow.summary)
        summary.pop("effect", None)
        render_approval_summary(st, summary, executes=True)
        st.info("아래 버튼은 현재 범위의 일회 승인 저장과 실행을 함께 요청합니다. 다음 단계의 승인은 대신하지 않습니다.")
        scope_key = "approval_flow_confirm_" + "_".join(str(value) for value in flow.review_key)
        with st.form("approval_flow_approve"):
            st.checkbox("표시한 범위를 승인하고 현재 단계를 실행합니다", key=scope_key, disabled=flow.busy)
            supported = not workflow or controller.state["runtime"]["supported"]
            st.form_submit_button("승인하고 실행", type="primary", disabled=flow.busy or not permission_ok or not supported,
                                  on_click=mark_flow_pending, args=(st, flow, "approve", scope_key))

    # callback에서 예약한 명시적 클릭만 처리한다. run은 pending을 먼저 소비한다.
    action = flow.pending
    if action is None:
        return

    if action == "approve" and not require_execution_permissions(st):
        flow.pending = None
        flow.busy = False
        return

    try:
        if action == "start":
            from .commentary_view import freeze_classification

            freeze_classification(st, controller)
        with show_operation_progress(st, "다음 승인 지점 또는 완료까지 처리하고 있습니다."):
            result = flow.run(action)
    except Exception as error:
        show_error(st, error, "요청을 완료하지 못했습니다. 5 결과 확인에서 저장된 기록을 확인하세요.")
    else:
        if result.get("status") == "completed":
            after_completion(st, controller, workflow=workflow)
        if result.get("status") != "busy":
            st.rerun()

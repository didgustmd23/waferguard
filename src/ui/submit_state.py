"""명시적인 실행 클릭을 UI에 먼저 반영한다. callback에서 backend를 호출하지 않는다."""


def mark_execution_pending(st, controller, confirmation_key):
    # Streamlit은 클릭 callback을 다음 화면 재구성 전에 실행한다. 이 값으로
    # 실행 버튼을 비활성화한 화면을 먼저 전송한 뒤 기존 동기 명령을 한 번 호출한다.
    if st.session_state.get(confirmation_key) is True and not controller.busy:
        controller.execution_requested = True
        controller.busy = True


def clear_unsubmitted_execution(controller):
    """권한 검사에서 거부한 UI 예약만 해제한다. 실제 실행 중 busy는 해제하지 않는다."""
    if getattr(controller, "execution_requested", False):
        controller.execution_requested = False
        controller.busy = False

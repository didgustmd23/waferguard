"""입력 검사·새 작업 준비의 명시적 클릭을 잠그고 한 번만 전달한다."""

from contextlib import contextmanager


def mark_creation_pending(st, workflow, action, confirmation_key=None):
    """화면 재구성 전에 버튼만 잠근다. 파일 검사나 작업 생성은 하지 않는다."""
    if action not in {"preview", "create"} or workflow.creation_busy:
        return

    if confirmation_key is not None and st.session_state.get(confirmation_key) is not True:
        return

    workflow.creation_requested = action
    workflow.creation_busy = True
    workflow.creation_failure = None
    if action == "preview":
        workflow.creation_preview = None
        workflow.creation_selection = None


@contextmanager
def creation_submission(workflow, action):
    """예약을 먼저 소비하고 실행 요청과 같은 제어기 잠금을 사용한다."""
    if not workflow._command_lock.acquire(blocking=False):
        workflow.creation_requested = None
        raise ValueError("다른 요청을 처리 중입니다. 중복 제출하지 않습니다.")

    try:
        if workflow.creation_requested != action:
            raise ValueError("명시적으로 제출한 현재 요청이 필요합니다.")

        workflow.creation_requested = None
        yield
    finally:
        workflow.creation_busy = False
        workflow._command_lock.release()

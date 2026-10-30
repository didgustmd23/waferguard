"""통합 작업 실행 제출의 화면 잠금. 클릭 callback은 예약만 하고, 실행은 그 버튼 바로 아래에서 한 번만 한다.

Streamlit은 st.* 화면 갱신과 st.session_state 접근마다 대기 중인 재실행·중지 요청을 확인하고, 있으면 그 자리에서
RerunException/StopException(BaseException)을 낸다. 실행 중에 이런 지점이 있으면 조정기·worker 처리가 중간에 끊긴다.
그래서 예약·실행·결과는 세션에 한 번 넣어 둔 객체의 속성으로만 바꾸고, 실행 중에는 spinner 외의 화면 갱신·세션 상태
접근을 하지 않는다. 결과 반영(작업 연결·안내·세션 키 정리)은 다음 재실행 시작(reconcile)에서 한다.

같은 세션에서도 재실행이 겹칠 수 있다. 기본 설정(runner.fastReruns)에서 처리 중 위젯을 조작하면 Streamlit은 이전 재실행에
중지만 요청하고 새 재실행을 바로 시작한다. 이전 재실행은 yield point가 없는 실행을 끝까지 마친다. 새 재실행은 실행 중인
스레드가 살아 있으면 잠금을 유지하고 끝날 때까지 기다린다. 실행 시작은 모듈 잠금으로 한 번만 일어나게 한다.

예약은 클릭한 그 재실행에서만 실행한다. 다른 조작으로 실행 전에 끊긴 예약은 실행하지 않고 풀며 다시 누르게 한다
(그 사이 다른 탭·위젯이 바꾼 값으로 동의가 발급되지 않게). 같은 화면의 같은 버튼을 다시 누른 경우(빠른 두 번 클릭)만
새 재실행이 예약을 이어받는다. 서버 대조값처럼 사용자가 본 값은 클릭한 화면을 그릴 때 버튼 인자(bound)로 묶는다.
"""

from dataclasses import dataclass, field
import threading


KEY = "ui_task_submit"
PROCESSING = "처리 중입니다. 끝날 때까지 실행 버튼과 화면 이동을 잠갔습니다. 창을 닫거나 새로 고치지 마세요."
WAITING = "앞서 시작한 처리가 아직 진행 중입니다. 끝나면 결과를 반영합니다. 실행 버튼과 화면 이동은 계속 잠겨 있습니다."
UNCLEAR = ("error", "처리 요청이 끝났는지 화면에서 확인하지 못했습니다. 저장된 작업 기록으로도 완료를 확인하지 못했습니다. "
                    "자동으로 다시 실행하지 않습니다. 아래 저장된 작업 상태를 확인하세요.")
STALE = ("warning", "실행 전에 화면이 다시 그려져 제출을 진행하지 않았습니다. 동의·승인은 만들지 않았고 작업은 그대로입니다. "
                    "다음 행동: 화면의 현재 내용을 확인한 뒤 다시 누르세요.")
# 같은 세션의 겹친 재실행 사이에서 예약 이어받기·실행 시작·해제를 한 번만 일어나게 한다. 잠금 안에서는 화면·세션에 접근하지 않는다.
_LOCK = threading.Lock()


@dataclass
class TaskSubmit:
    """세션당 하나. request가 있으면 통합 작업의 실행 버튼·이동 위젯을 잠근다."""

    generation: int = 0  # 화면을 그린 시점. 실행·취소가 끝날 때마다 올려 이전 화면의 늦은 클릭을 거부한다.
    # reconcile을 지난 재실행 수. 예약이 클릭한 재실행에서 나온 것인지 가린다. 작업 상태가 아니라 비교에서 뺀다.
    runs: int = field(default=0, compare=False)
    # {"action", "task_id", "generation", "run", "bound", "progress", "state": reserved|ready|running|done|aborted, ...}
    request: dict | None = None
    notice: list | None = None  # 다음 화면에 한 번 보일 안내 [(종류, 문구)]


def submit_state(st):
    state = st.session_state.get(KEY)

    if not isinstance(state, TaskSubmit):
        state = TaskSubmit()
        st.session_state[KEY] = state

    return state


def _running(request):
    """어느 재실행이 call을 실행하는 중인지. call이 끝나면 예외 종류와 관계없이 done/aborted로 바뀐다.

    Streamlit은 이어지는 재실행을 같은 스레드에서 돌리기도 하므로 스레드 생존만으로 판정하지 않는다. 스레드가 없어졌는데
    running이면(프로세스 수준 중단) 실행 중으로 보지 않는다.
    """
    return request["state"] == "running" and request["thread"].is_alive()


def _unclear(request):
    """실행이 결과를 남기지 못하고 끝났다(BaseException). 저장된 작업 기록으로 이 동작의 완료를 확인할 수 있을 때만 확인 안내를 낸다."""
    progress = request["progress"]
    verify = request.get("verify")

    try:
        confirmed = verify(progress) if verify is not None else None
    except Exception:
        confirmed = None

    return {"link": progress.get("link"), "notice": [("info", confirmed) if confirmed else UNCLEAR]}


def reconcile(st):
    """재실행마다 화면을 그리기 전에 한 번(app.py) 호출한다. 끝난 실행의 결과를 반영한다.

    세션 쓰기는 요청을 지우기 전에 한다. 쓰기 중 재실행으로 끊기면 다음 재실행에서 같은 결과를 다시 반영한다.
    """
    state = submit_state(st)

    with _LOCK:
        state.runs += 1
        request = state.request

        if request is None or _running(request):
            return state

        if request["state"] in {"reserved", "ready"}:
            if request["run"] != state.runs:
                # 클릭한 재실행이 실행 전에 끊겼거나 예약한 화면을 그리지 않았다. 실행하지 않고 잠금을 푼다.
                state.request, state.notice = None, [STALE]
                state.generation += 1

            return state

    # done: call이 결과를 돌려줌. aborted(BaseException)·스레드 없는 running: 결과를 모르므로 저장 상태로 대조한다.
    outcome = (request.get("outcome") if request["state"] == "done" else None) or _unclear(request)

    if outcome.get("link"):
        st.session_state["ui_task_id"] = outcome["link"]

    for key in outcome.get("pop", ()):
        st.session_state.pop(key, None)

    with _LOCK:
        if state.request is request:
            state.notice = outcome.get("notice")
            state.request = None
            state.generation += 1

    return state


def locked(st):
    return submit_state(st).request is not None


def generation(st):
    return submit_state(st).generation


def idle(st, seen):
    """화면 이동 callback용. 실행 중이거나 이전 화면에서 누른 늦은 클릭이면 False."""
    state = submit_state(st)
    return state.request is None and state.generation == seen


def reserve(st, action, task_id, seen, bound=None):
    """클릭 callback. 실행 중·대기 중 예약이 있거나 이전 화면의 클릭이면 무시한다. 조정기를 호출하지 않는다.

    bound는 클릭한 화면을 그릴 때 정한 값(사용자가 본 대조값 등)이다. callback은 같은 재실행의 reconcile 전에 실행된다.
    같은 화면(generation·bound)의 같은 버튼을 실행 전에 다시 누르면 이전 재실행은 중지되므로 이 재실행이 예약을 이어받는다.
    """
    state = submit_state(st)

    with _LOCK:
        request = state.request

        if request is None:
            if state.generation == seen:
                state.request = {"action": action, "task_id": task_id, "generation": seen, "run": state.runs + 1,
                                 "bound": bound, "progress": {}, "state": "reserved"}
        elif (request["state"] in {"reserved", "ready"}
              and (request["action"], request["task_id"], request["generation"], request["bound"]) == (action, task_id, seen, bound)):
            request.update(run=state.runs + 1, state="reserved")


def reserved(st, action, task_id):
    """이번 화면에서 양식 값을 받아야 하는 예약인지."""
    request = submit_state(st).request
    return request is not None and request["state"] == "reserved" and (request["action"], request["task_id"]) == (action, task_id)


def bound(st):
    """이번 예약의 클릭한 화면 값."""
    return submit_state(st).request["bound"]


def decline(st, notice):
    """실행 전 확인(동의·권한 등)에서 멈춘다. 잠기지 않은 화면을 다시 그려 안내한다."""
    state = submit_state(st)

    with _LOCK:
        state.request = None
        state.generation += 1
        state.notice = notice

    st.rerun()


def show_notice(st):
    state = submit_state(st)
    notice, state.notice = state.notice, None

    for kind, text in notice or ():
        getattr(st, kind)(text)


def execute(st, call, label, verify=None):
    """예약한 버튼 바로 아래에서 call(progress)을 한 번 실행한다. 실행 중에는 spinner만 두고 세션 상태에 접근하지 않는다.

    call은 결과 dict를 돌려주고, 부분 진행(예: 만든 작업 ID)은 progress에 남긴다. verify(progress)는 결과를 받지 못했을 때
    저장된 작업 기록으로 이 동작의 완료를 확인하는 문구를 돌려준다(확인할 수 없으면 None).
    call 안의 일반 예외는 call이 안내로 바꾼다. 그 밖의 BaseException은 삼키지 않고 다시 던지며,
    요청은 aborted가 되어 다음 재실행에서 저장 상태 대조로 정리된다.
    """
    state = submit_state(st)
    request = state.request

    with _LOCK:
        # 이 재실행의 예약일 때만 준비한다. 같은 버튼 재클릭으로 새 재실행이 이어받았으면 그 상태를 덮어쓰지 않는다.
        mine = request is not None and request["state"] == "reserved" and request["run"] == state.runs

        if mine:
            request.update(state="ready", call=call, label=label, verify=verify)

    if not mine:
        st.rerun()

    with st.container():
        st.info(PROCESSING)

        with st.spinner(label, show_time=True):
            # 마지막 yield point 뒤에서 실행 권한을 잡는다. 그 사이 다른 재실행이 예약을 풀거나 이어받았으면 실행하지 않는다.
            with _LOCK:
                start = state.request is request and request["state"] == "ready" and request["run"] == state.runs

                if start:
                    request.update(state="running", thread=threading.current_thread())

            if start:
                try:
                    request["outcome"] = call(request["progress"])
                finally:
                    request["state"] = "done" if "outcome" in request else "aborted"

    st.rerun()


def wait_if_running(st, task_id, action=None):
    """다른 재실행이 이 작업(action이 있으면 그 동작)을 실행 중이면 이 자리에 안내·spinner를 두고 끝날 때까지 기다린다.

    기다리는 동안 세션 상태에 접근하지 않는다. 끝나면 결과 반영을 위해 다시 그린다. 실행 중이 아니면 그대로 돌아온다.
    """
    request = submit_state(st).request

    if (request is None or request["state"] != "running" or request["task_id"] != task_id
            or (action is not None and request["action"] != action)):
        return

    thread = request["thread"]

    if thread.is_alive():
        with st.container():
            st.info(WAITING)

            with st.spinner(request["label"], show_time=True):
                thread.join()

    st.rerun()


def finish(st):
    """화면 끝. 이번 재실행 예약인데 실행할 버튼을 그리지 않았으면(단계가 바뀜) 실행 없이 풀고, 다른 재실행이 실행 중이면 기다린다."""
    state = submit_state(st)
    request = state.request

    if request is None:
        return

    if request["state"] == "running":
        wait_if_running(st, request["task_id"])
        return

    with _LOCK:
        if state.request is request and request["state"] in {"reserved", "ready"}:
            state.request = None
            state.generation += 1

    st.rerun()

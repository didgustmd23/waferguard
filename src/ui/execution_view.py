"""사전 검사 → 범위 확인 → 승인 → 격리 실행 → 로컬 결과 표시 화면."""

import csv
from io import StringIO
import itertools

from .execution_service import UiExecution
from .result_service import load_results


def execution_available(st):
    """입력·설정을 바꾸어도 이미 시작한 작업의 상태 조회 연결은 보존한다."""
    execution = st.session_state.get("ui_active_execution")

    if not isinstance(execution, UiExecution) or not execution.started:
        return False

    try:
        execution.read_state()
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _submit(st, execution, action, *, confirmed=False):
    try:
        with st.spinner("로컬 계약 확인 및 요청 처리 중입니다. 작업을 다시 제출하지 마세요."):
            execution.command(action, confirmed=confirmed)
    except Exception:
        st.error("요청을 완료하지 못했습니다. 현재 작업의 로컬 기록을 확인하세요. "
                 "자동 재시도하거나 승인을 다시 발급하지 않습니다.")
    else:
        st.rerun()


def render_execution(st, receipt, candidate):
    st.header("4 실행 승인")
    execution = st.session_state.get("ui_active_execution")

    if not isinstance(execution, UiExecution) or execution.name != receipt["name"]:
        # 새 후보가 준비되어도 이전 실행 이력을 덮어쓰지 않고 세션에 함께 보존한다.
        if isinstance(execution, UiExecution):
            st.session_state.setdefault("ui_execution_history", {})[execution.name] = execution

        execution = UiExecution(receipt, candidate.run_selection_id)
        st.session_state["ui_active_execution"] = execution

    try:
        state = execution.read_state()
    except (OSError, ValueError, TypeError, KeyError):
        st.error("작업 기록이 변경되었거나 확인할 수 없습니다. 실행하지 않습니다.")
        return

    st.text("작업 이름: " + execution.name)
    st.text("현재 상태: " + state.status + " / 단계: " + state.current_stage)
    st.caption("승인은 실행과 별개입니다. 실행 중에는 현재 화면의 요청 처리가 끝날 때까지 기다려 주세요.")

    if execution.last_result:
        with st.expander("최근 요청 기술 상세"):
            st.json(execution.last_result)

    if execution.error:
        st.error("처리 실패 또는 결과 불명 상태입니다. 결과 화면에서 기록을 확인하고 새 작업으로 준비하세요.")
        return

    if state.status in {"created", "running"}:
        if st.button("사전 검사 진행", key="ui_preflight", help="입력·코드를 재검사하고 한 단계만 진행합니다. worker는 시작하지 않습니다."):
            _submit(st, execution, "preflight")
        return

    if state.status != "awaiting_approval":
        st.info("이 작업은 종료되었거나 중단되었습니다. 5번에서 상태와 결과를 확인하세요.")
        return

    if execution.approval_id is None:
        if execution.preview is None:
            if st.button("승인 범위 검사", key="ui_approval_preview", help="현재 입력·모듈·코드·자원 범위를 다시 검사해 표시합니다. 승인은 발급하지 않습니다."):
                _submit(st, execution, "preview")
            return

        st.subheader("이번 실행의 승인 범위")
        st.json(execution.preview["summary"])
        with st.form("approve_" + execution.name):
            confirmed = st.checkbox("위 입력·모듈·품질 기준·자원 범위의 이번 실행을 승인합니다", value=False,
                                    help="현재 작업과 코드에 결합된 일회 승인을 저장합니다. worker는 아직 실행하지 않습니다.")
            submitted = st.form_submit_button("이번 실행 승인", type="primary")

        if submitted:
            if confirmed:
                _submit(st, execution, "approve", confirmed=True)
            else:
                st.warning("이번 승인 범위를 확인해 주세요.")
        return

    st.success("이번 작업의 승인을 저장했습니다. 아래 실행 확인을 제출해야 worker가 시작됩니다.")
    with st.form("execute_" + execution.name):
        confirmed = st.checkbox("승인한 범위로 격리 worker를 지금 실행합니다", value=False,
                                help="일회 승인을 소비합니다. 입력·코드 변경 또는 만료 시 거부하며 자동 재시도하지 않습니다.")
        submitted = st.form_submit_button("승인한 작업 실행", type="primary")

    if submitted:
        if confirmed:
            _submit(st, execution, "execute", confirmed=True)
        else:
            st.warning("worker 실행 여부를 확인해 주세요.")


def render_results(st):
    st.header("5 결과 확인")
    if not execution_available(st):
        st.info("연결된 작업 기록이 없습니다.")
        return

    active = st.session_state["ui_active_execution"]
    history = dict(st.session_state.get("ui_execution_history", {}))
    history[active.name] = active
    name = st.selectbox("조회할 작업", tuple(history), index=list(history).index(active.name),
                        help="이 브라우저 세션에서 연결한 작업입니다. 입력·설정 변경 후에도 이전 이력을 확인할 수 있습니다.")
    execution = history[name]

    try:
        state = execution.read_state()
    except (OSError, ValueError, TypeError, KeyError):
        st.error("선택한 작업 기록을 확인하지 못했습니다. 저장된 파일은 변경하지 않습니다.")
        return
    st.text("상태: " + state.status + " / 단계: " + state.current_stage)
    st.text("로컬 기록: " + str(execution.directory))

    if execution.last_result:
        with st.expander("최근 요청 기술 상세"):
            st.json(execution.last_result)

    if st.button("상태 새로 확인", key="ui_refresh_status", help="저장된 상태만 다시 읽습니다. 모델·승인·worker는 호출하지 않습니다."):
        st.rerun()

    if state.status == "blocked_input":
        st.warning("품질 검토 또는 입력 차단으로 중단되었습니다. 후속 분석은 자동 실행하지 않습니다.")
    elif state.status in {"failed_runtime", "security_locked", "cancelled"} or execution.error:
        st.error("작업이 완료되지 않았습니다. 부분 결과를 정상 완료로 표시하지 않습니다.")

    if st.button("로컬 결과 검사·열기", key="ui_load_results",
                 disabled=state.status not in {"completed", "blocked_input"},
                 help="결과 계약·완료 표식·파일 정합성을 검사합니다. LLM에 데이터를 전송하지 않습니다."):
        try:
            st.session_state["ui_result_view"] = (name, load_results(execution))
        except Exception:
            st.session_state.pop("ui_result_view", None)
            st.error("표시 가능한 결과 계약을 확인하지 못했습니다. 로컬 기록을 확인하세요.")

    loaded = st.session_state.get("ui_result_view")
    if loaded is None or loaded[0] != name:
        return

    result = loaded[1]
    st.caption(result["notice"])
    st.text("결과 상태: " + result["status"])
    st.text(result["directory"])

    render_result_files(st, result["files"])
    st.caption("결과 해설을 위한 LLM 전송과 외부 추적은 활성화하지 않았습니다.")


def render_result_files(st, files):
    """CSV·WM 공통 표시기. HTML·Markdown 외부 참조는 해석하지 않는다."""
    for item in files:
        with st.expander(item["name"], expanded=item["suffix"] == ".png"):
            if item["suffix"] == ".png":
                st.image(item["data"])
            elif item["suffix"] == ".csv":
                # 전체 표를 브라우저로 보내지 않는다. 처음 200행·100열만 표시한다.
                rows = list(itertools.islice(csv.reader(StringIO(item["data"].decode("utf-8-sig"))), 201))
                st.table([row[:100] for row in rows])
                st.caption("최대 201행(헤더 포함)·100열 미리보기입니다. 원본 보고서 파일은 변경하지 않습니다.")

                if rows and "missing_ratio" in rows[0] and len(rows) > 1:
                    index = rows[0].index("missing_ratio")
                    st.bar_chart({"특징": [row[0] for row in rows[1:]],
                                  "결측 비율": [float(row[index]) for row in rows[1:]]},
                                 x="특징", y="결측 비율")
            else:
                # 보고서의 외부 링크·HTML·이미지 참조를 자동으로 해석하지 않는다.
                st.text(item["data"].decode("utf-8")[:100000])


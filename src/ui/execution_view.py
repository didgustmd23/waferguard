"""사전 검사 → 범위 확인 → 승인 → 격리 실행 → 로컬 결과 표시 화면."""

import csv
from io import StringIO
import itertools

from .execution_service import UiExecution
from .result_service import load_results
from .progress_view import show_operation_progress
from .report_view import render_report
from .runtime_notice import render_permissions, require_execution_permissions
from .submit_state import clear_unsubmitted_execution
from .approval_flow_view import render_approval_flow


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
    if action in {"approve", "execute"} and not require_execution_permissions(st):
        clear_unsubmitted_execution(execution)
        return

    try:
        with show_operation_progress(st, "로컬 계약 확인 및 요청 처리 중입니다. 작업을 다시 제출하지 마세요."):
            result = execution.command(action, confirmed=confirmed)
    except Exception:
        st.error("요청을 완료하지 못했습니다. 현재 작업의 로컬 기록을 확인하세요. "
                 "자동 재시도하거나 승인을 다시 발급하지 않습니다.")
    else:
        if result.get("status") == "busy":
            st.info("현재 요청을 처리 중입니다. 다시 제출하지 마세요.")
            return
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

    render_approval_flow(st, execution, workflow=False)


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
    from .commentary_view import render_commentary

    render_commentary(st, execution, result)


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
            elif item["suffix"] == ".md":
                render_report(st, item, files)
            else:
                # 보고서의 외부 링크·HTML·이미지 참조를 자동으로 해석하지 않는다.
                st.text(item["data"].decode("utf-8")[:100000])


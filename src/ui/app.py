"""단계 안내형 로컬 UI. 모델·승인·격리 실행은 각각 명시적 제출로만 호출한다."""

import streamlit as st

from src.ui.settings import render_settings
from src.ui.navigation import render_navigation
from src.ui.state import SCREENS, default_settings
from src.ui import dialogue_view
from src.ui import input_view
from src.ui.candidate_view import render_candidate_approval
from src.ui.execution_view import render_results as render_execution_results


def render_request():
    """모델 호출을 포함한 명시적 제출 경계는 대화 화면 모듈에서 처리한다."""
    dialogue_view.render_request(st)


def render_review():
    """기존 계약을 재검사한 제안만 읽기 전용으로 표시한다."""
    dialogue_view.render_review(st)


def render_input():
    """기존 snapshot·프로파일을 사용자가 확인 제출한 경우만 로컬 결합 검사한다."""
    input_view.render_input(st)


def render_approval():
    """기존 일회 승인·격리 실행기의 명시적 제출 화면을 연다."""
    render_candidate_approval(st)


def render_results():
    """현재 로컬 상태와 검사된 표·그래프·보고서를 표시한다."""
    render_execution_results(st)


def main():
    st.set_page_config(page_title="WaferGuard 전처리 작업실", page_icon="🧩", layout="wide")

    if "ui_settings" not in st.session_state:
        try:
            st.session_state["ui_settings"] = default_settings()
        except (OSError, ValueError, TypeError, KeyError):
            st.error("기본 모델 설정을 읽을 수 없습니다. 프로젝트의 로컬 모델 JSON을 확인하세요.")
            return

    if "ui_settings_revision" not in st.session_state:
        st.session_state["ui_settings_revision"] = 0

    if "ui_draft" not in st.session_state:
        st.session_state["ui_draft"] = ""

    # 통합 작업 실행을 예약했거나 처리 중이면 화면 이동도 잠근다. 끝난 실행의 결과는 여기서 먼저 반영한다.
    from src.ui import task_submit

    task_submit.reconcile(st)
    locked = task_submit.locked(st)

    with st.sidebar:
        advanced = st.checkbox("고급 도구 표시", key="ui_show_advanced_tools", disabled=locked,
                               help="기존 LLM 제안과 등록 모듈의 단계별 실행 화면을 표시합니다. 표시 변경만으로 작업·승인·모델 호출은 발생하지 않습니다.")
        if advanced:
            if "ui_workflow_mode" not in st.session_state:
                st.session_state["ui_workflow_mode"] = "통합 작업"
            st.selectbox("작업 도구", ("통합 작업", "LLM 제안", "등록 모듈"), key="ui_workflow_mode", disabled=locked,
                         help="통합 작업을 기본으로 사용하세요. LLM 제안과 등록 모듈은 기존 단계별 실행을 위한 고급 도구입니다.")
        else:
            # 이전 세션의 고급 화면 선택이 남아 있어도 기본 화면에서는 실행 경로를 노출하지 않는다.
            # 각 화면의 초안·작업 연결·승인은 삭제하지 않는다.
            st.session_state["ui_workflow_mode"] = "통합 작업"
        if st.session_state.get("ui_workflow_mode") == "통합 작업":
            st.radio("화면", ("작업", "설정"), key="ui_task_screen", horizontal=True, disabled=locked)
        if st.session_state.get("ui_workflow_mode", "LLM 제안") == "LLM 제안":
            st.selectbox("제안 범위", ("품질·EDA", "CSV 분할·전처리"), key="ui_proposal_scope",
                         help="CSV 실행안도 같은 5단계를 사용합니다. 작업 준비와 각 실행 승인은 별도입니다.")

    if st.session_state.get("ui_workflow_mode") == "통합 작업":
        # 통합 작업은 단계 메뉴 대신 요청·진행·결정·결과 네 영역만 보여 준다.
        st.title("전처리 작업실")
        with st.container(border=True):
            if st.session_state.get("ui_task_screen", "작업") == "설정":
                render_settings(st)
            else:
                from src.ui.task_view import render_task

                render_task(st)
        return

    screen = render_navigation(st)
    st.title("전처리 작업실")
    st.caption("요청 작성부터 결과 확인까지, 단계별로 진행합니다.")
    st.info("제안 → 입력 확인 → 사전 검사 → 승인 → 격리 실행 → 로컬 결과 확인 순서로 진행합니다.")

    renderers = (render_request, render_review, render_input, render_approval, render_results, lambda: render_settings(st))
    if st.session_state.get("ui_workflow_mode") == "등록 모듈":
        from src.ui import workflow_view

        renderers = (workflow_view.request, workflow_view.review, workflow_view.inputs,
                     workflow_view.approval, workflow_view.results, render_settings)
        renderers = tuple((lambda renderer=renderer: renderer(st)) for renderer in renderers)

    from src.ui.csv_plan_service import selected

    if selected(st.session_state):
        from src.ui import csv_plan_view

        renderers = tuple((lambda renderer=renderer: renderer(st)) for renderer in (
            csv_plan_view.request, csv_plan_view.review, csv_plan_view.inputs,
            csv_plan_view.approval, csv_plan_view.results, render_settings))

    with st.container(border=True):
        renderers[SCREENS.index(screen)]()

    with st.expander("기술 상세 보기"):
        csv_plan = st.session_state.get("ui_csv_plan") if selected(st.session_state) else None
        dialogue = st.session_state.get("ui_csv_dialogue" if selected(st.session_state) else "ui_dialogue")
        execution = st.session_state.get("ui_active_execution")
        workflow = (csv_plan.active if csv_plan is not None else
                    st.session_state.get("ui_workflow") if st.session_state.get("ui_workflow_mode") == "등록 모듈" else None)
        st.json({"ui_phase": "csv_workflow" if selected(st.session_state) else "registered_workflow" if workflow is not None else "local_dialogue", "settings_revision": st.session_state["ui_settings_revision"],
                 "dialogue_phase": None if dialogue is None else dialogue.phase,
                 "approval_recorded": (workflow.approval_id is not None if workflow is not None
                                       else execution is not None and execution.approval_id is not None),
                 "worker_started": (bool((workflow.last_result or {}).get("worker_started")) if workflow is not None
                                    else execution is not None and execution.worker_started),
                 "langsmith_tracing": False,
                 "settings_storage": "session_memory_only",
                 "workflow_storage": "local_plan_and_creation_records" if csv_plan is not None else None})


if __name__ == "__main__":
    main()

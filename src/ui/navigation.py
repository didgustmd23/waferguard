"""단계 메뉴의 표현과 이동만 담당한다. 모델 호출·승인·실행 부작용은 없다."""

from src.ui.state import SCREENS, SCREEN_LOCKS, resolve_screen
from src.ui.dialogue_view import review_available
from src.ui.input_view import input_available
from src.ui.candidate_view import current_candidate_receipt
from src.ui.execution_view import execution_available


STEP_HINTS = (
    "요청할 전처리 작업을 작성합니다",
    "모듈과 처리 순서를 검토합니다",
    "데이터와 프로파일을 선택합니다",
    "범위를 확인하고 실행을 승인합니다",
    "진행 상태와 표·그래프를 확인합니다",
)


def availability(st):
    """선택한 진입 경로의 메뉴 조건만 조회한다. worker·승인을 호출하지 않는다."""
    from .csv_plan_service import selected

    if selected(st.session_state):
        from .csv_plan_view import flags

        return flags(st)

    if st.session_state.get("ui_workflow_mode") == "등록 모듈":
        from .workflow_view import flags

        return flags(st)

    return {"proposal_available": review_available(st), "saved_available": input_available(st),
            "candidate_available": current_candidate_receipt(st) is not None,
            "execution_available": execution_available(st)}


def select_screen(st, screen):
    """클릭 콜백에서는 이동할 화면만 바꾼다. 잠긴 화면 요청은 첫 화면으로 돌린다."""
    st.session_state["ui_screen"] = resolve_screen(
        screen, **availability(st),
    )


def render_navigation(st):
    """현재 단계는 강조하고 선행 조건이 없는 단계는 잠금 사유를 표시한다."""
    flags = availability(st)
    available = flags["proposal_available"]
    saved = flags["saved_available"]
    prepared = flags["candidate_available"]
    executed = flags["execution_available"]
    screen = resolve_screen(st.session_state.get("ui_screen", SCREENS[0]),
                            proposal_available=available, saved_available=saved, candidate_available=prepared,
                            execution_available=executed)
    st.session_state["ui_screen"] = screen

    with st.sidebar:
        st.title("WaferGuard")
        st.caption("전처리 작업실 · 로컬 전용")
        st.divider()
        st.subheader("작업 순서")

        for index, step in enumerate(SCREENS[:-1]):
            locked = step in SCREEN_LOCKS and not (index == 1 and available or index == 2 and saved
                                                   or index == 3 and prepared or index == 4 and executed)
            st.button(
                step + (" · 잠김" if locked else ""),
                key=f"ui_step_{index + 1}",
                type="primary" if screen == step else "secondary",
                width="stretch",
                disabled=locked,
                help=SCREEN_LOCKS[step] if locked else STEP_HINTS[index],
                on_click=select_screen,
                args=(st, step),
            )

        st.caption("잠긴 단계의 버튼에 마우스를 올리면 필요한 조건을 볼 수 있습니다.")
        st.divider()
        st.button(
            "설정", key="ui_settings_navigation", width="stretch",
            type="primary" if screen == "설정" else "secondary",
            help="모델·자원 기본값과 전달 정책 안내를 확인합니다. 작업은 실행하지 않습니다.",
            on_click=select_screen, args=(st, "설정"),
        )
        applied = st.session_state["ui_settings"]
        st.caption("모드: " + applied["mode"])
        st.caption("모델: " + applied["model"]["model"])
        st.caption("연결 상태: 아직 확인하지 않음")

    return screen

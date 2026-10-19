"""실행 후보 준비와 생성 영수증 검사 후 승인 화면 연결."""

from .candidate_service import UiCandidate
from .dialogue_service import fingerprint
from .dialogue_view import current_dialogue


def current_candidate_receipt(st):
    dialogue = current_dialogue(st)
    candidate = st.session_state.get("ui_execution_candidate")

    if dialogue is None or not isinstance(candidate, UiCandidate):
        return None

    return candidate.verified_receipt(dialogue, st.session_state["ui_settings"])


def render_candidate_creation(st, preview, dialogue, settings):
    """새 미리보기마다 새 확인을 받는다. 화면 재표시는 작업 생성 명령이 아니다."""
    candidate = st.session_state.get("ui_execution_candidate")

    if candidate is None or candidate.preview_id != fingerprint(preview):
        candidate = UiCandidate(preview, dialogue, settings)
        st.session_state["ui_execution_candidate"] = candidate

    st.subheader("실행 후보 준비")
    selection = preview["result"]["selection"]
    st.text("snapshot: " + selection["snapshot_directory"])
    st.text("프로파일: " + selection["profile_path"])
    st.text("처리 순서: " + " → ".join(step["module_id"] for step in selection["pipeline"]["steps"]))
    st.caption("입력·모듈·품질 기준·자원 상세는 위 입력 결합 미리보기에서 확인하세요. 실행 승인과 작업 시작은 별도입니다.")

    if candidate.phase == "created":
        receipt = candidate.verified_receipt(dialogue, settings)

        if receipt is None:
            st.error("생성된 작업 기록을 확인할 수 없습니다. 4번을 열지 않으며 다시 생성하지 않습니다.")
        else:
            st.success("실행 후보를 준비했습니다. 왼쪽 ‘4 실행 승인’에서 준비 기록을 확인할 수 있습니다.")
            st.text("작업 이름: " + receipt["name"])
            st.text("로컬 작업 폴더: " + receipt["run_dir"])
        return

    if candidate.phase != "ready":
        st.warning("이 요청은 처리 중이거나 실패·중단되었습니다. 자동 재시도하지 않습니다. "
                   "부분 기록을 확인한 뒤 입력 미리보기를 다시 검사하고 새 작업 이름을 사용하세요.")
        return

    with st.form("create_candidate_" + candidate.preview_id):
        name = st.text_input("실행 후보 이름", value="ui_run01", max_chars=48,
                             help="소문자로 시작하는 영문 소문자·숫자·밑줄로 48자 이하입니다. 기존 작업은 덮어쓰지 않습니다.")
        confirmed = st.checkbox("입력·모듈·설정·자원을 확인했으며 작업 준비에 동의합니다", value=False,
                                help="미승인 작업과 준비 파일만 생성합니다. 승인 토큰 발급·worker 실행·LLM 호출은 하지 않습니다.")
        submitted = st.form_submit_button("실행 후보 생성", type="primary")

    if submitted:
        if not confirmed:
            st.warning("작업 준비 범위를 확인하고 동의해 주세요.")
            return

        try:
            receipt = candidate.create(name, dialogue, settings, confirmed=True)
        except Exception:
            st.error("실행 후보를 준비하지 못했습니다. 입력·설정 변경 또는 기존 이름을 확인하세요. "
                     "부분 기록이 남을 수 있으며 자동 재시도하지 않습니다.")
        else:
            if receipt["status"] == "busy":
                st.warning("같은 작업을 준비 중입니다. 중복 생성하지 않았습니다.")
            else:
                st.rerun()


def render_candidate_approval(st):
    receipt = current_candidate_receipt(st)

    if receipt is None:
        st.warning("현재 범위에서 확인된 실행 후보가 없습니다. 3번에서 작업을 먼저 준비하세요.")
        return

    from .execution_view import render_execution

    render_execution(st, receipt, st.session_state["ui_execution_candidate"])

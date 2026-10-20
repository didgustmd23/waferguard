"""요청·누적 전송 미리보기·질문·제안 검토 화면. 부작용은 명시적 제출에서만 발생한다."""

from src.agents.llm_proposal import DESCRIPTIONS
from src.agents.llm_proposal_store import list_saved_proposals
from .dialogue_service import UiDialogue
from .state import MODES, validate_draft
from .source_selection import render_source_selection, selected_source


def current_dialogue(st):
    """설정 변경으로 이전 미리보기·제안의 유효성이 사라지면 세션 대화를 닫는다."""
    dialogue = st.session_state.get("ui_dialogue")
    source = selected_source(st.session_state)

    if dialogue is not None and (not dialogue.matches(st.session_state["ui_settings"])
                                 or source is None
                                 or getattr(dialogue, "ui_source_id", None) != source["selection_id"]):
        dialogue.close()
        del st.session_state["ui_dialogue"]
        st.session_state.pop("ui_input_preview", None)
        st.session_state.pop("ui_execution_candidate", None)
        return None

    return dialogue


def review_available(st):
    dialogue = current_dialogue(st)
    return dialogue is not None and dialogue.validated_proposal(st.session_state["ui_settings"]) is not None


def render_request(st):
    st.header("1 요청·대화")
    st.caption("요청 문장·이전 질문과 답변·공개 모듈 명세만 로컬 모델에 전달합니다. "
               "파일·통계·비밀정보를 입력하지 마세요. LangSmith 추적은 비활성화합니다.")
    settings = st.session_state["ui_settings"]
    render_source_selection(st)
    dialogue = current_dialogue(st)
    source = selected_source(st.session_state)

    if dialogue is not None:
        for exchange in dialogue.history:
            st.caption("보낸 요청 / 답변")
            st.text(exchange["user"])

            for question in exchange["questions"]:
                st.text("확인 질문: " + question)

        if dialogue.phase == "proposal_validated":
            st.success("실행안 검증을 통과했습니다. 왼쪽 ‘2 제안 검토’에서 내용을 확인하세요.")
        elif dialogue.phase == "clarification_limit_reached":
            st.warning("최대 대화 횟수에 도달했습니다. 요청을 구체화해 새 대화를 시작하세요.")
        elif dialogue.phase == "failed":
            st.error("모델 통신 또는 응답 검증에 실패했습니다. 자동 재전송하지 않습니다. "
                     "이전 요청이 모델에 도달했을 수 있으므로 확인 후 새 대화를 시작하세요.")

        if st.button("새 대화 시작", key="ui_new_dialogue",
                     help="현재 세션 대화·제안·전송 미리보기를 비웁니다. 소비된 명령 기록은 지우지 않습니다."):
            dialogue.close()
            del st.session_state["ui_dialogue"]
            st.session_state["ui_draft"] = ""
            st.session_state.pop("ui_input_preview", None)
            st.session_state.pop("ui_execution_candidate", None)
            st.rerun()
            return

    if dialogue is None and source is not None and settings["mode"] == MODES[0]:
        render_saved_restore(st, settings, source)

    accepting = dialogue is None or dialogue.phase in ("new", "awaiting_answer")
    with st.form("request_form"):
        text = st.text_area(
            "확인 질문에 대한 답변" if dialogue is not None and dialogue.phase == "awaiting_answer" else "전처리 요청",
            value=st.session_state["ui_draft"], height=180,
            placeholder="예: 품질 검사를 통과하면 결측률과 상관관계를 확인해 주세요.",
            help="작업 목적만 입력하세요. 보관한 문장은 모델 전송 확인 전까지 세션에만 유지합니다.",
            disabled=not accepting,
        )
        submitted = st.form_submit_button("요청 초안 보관", type="primary", disabled=not accepting)

    if submitted and accepting:
        try:
            st.session_state["ui_draft"] = validate_draft(text)

            if dialogue is not None:
                dialogue.pending = None

            st.success("초안을 보관했습니다. 아직 모델에 보내지 않았습니다.")
        except ValueError as error:
            st.error(str(error))

    local_mode = settings["mode"] == MODES[0]
    if not local_mode:
        st.info("규칙 기반 화면 연결은 준비 중입니다. 이 모드에서는 모델을 호출하지 않습니다.")

    if st.button("전송 내용 미리보기", key="ui_preview_request",
                 disabled=not accepting or not local_mode or source is None or not st.session_state["ui_draft"],
                 help="보관한 초안과 누적 대화를 미리 봅니다. 이 버튼만으로 모델을 호출하지 않습니다."):
        try:
            if dialogue is None:
                dialogue = UiDialogue(settings)
                dialogue.ui_source_id = source["selection_id"]
                st.session_state["ui_dialogue"] = dialogue

            dialogue.prepare(st.session_state["ui_draft"], settings)
        except ImportError:
            st.error("대화 의존성이 없습니다. 현재 Python 환경의 requirements-agents.txt 설치를 확인하세요.")
        except Exception:
            st.error("전송 미리보기를 만들 수 없습니다. 입력 길이·설정·대화 상태를 확인하세요.")

    if dialogue is None or dialogue.pending is None:
        return

    pending = dialogue.pending
    preview = pending["preview"]
    st.subheader("전송 전 확인")
    st.text("목적지: " + preview["endpoint"] + "\n모델: " + preview["payload"]["model"])
    st.text(preview["payload"]["messages"][1]["content"])

    with st.expander("공개 모듈 명세·설정을 포함한 전체 전송 내용"):
        st.json(preview["payload"])

    # 명령마다 다른 폼 키를 사용해 이전 전송의 체크 상태를 재사용하지 않는다.
    with st.form("send_" + pending["command_id"]):
        confirmed = st.checkbox("위 내용을 이 PC의 로컬 모델에 전달하는 데 동의합니다", value=False)
        send = st.form_submit_button("확인한 내용 전송", type="primary")

    if send:
        if not confirmed:
            st.warning("전송 내용을 확인하고 동의해 주세요.")
            return

        try:
            with st.spinner("로컬 모델이 실행안을 제안하고 있습니다. 자동 재시도하지 않습니다."):
                outcome = dialogue.send(pending["command_id"], settings, confirmed=True)
        except Exception:
            # 모델·라이브러리 예외 원문에 문장이나 응답이 섞일 수 있어 UI로 노출하지 않는다.
            st.error("모델 전송을 완료하지 못했습니다. 자동 재시도하지 않습니다. "
                     "입력·설정·Ollama 상태를 확인하세요. 응답 수신 전 중단되었다면 전달 여부는 불명확합니다.")
        else:
            if outcome["status"] in ("busy", "already_consumed"):
                st.warning("이미 처리 중이거나 소비된 요청입니다. 다시 호출하지 않았습니다.")
            else:
                st.session_state["ui_draft"] = ""
                st.rerun()


def _refresh_saved(st):
    try:
        st.session_state["ui_saved_proposals"] = list_saved_proposals()
    except (OSError, ValueError):
        st.session_state["ui_saved_proposals"] = []
        st.error("저장 제안 폴더를 읽지 못했습니다. 폴더 위치와 링크 사용 여부를 확인하세요.")


def render_saved_restore(st, settings, source):
    """저장 제안 목록에서 고른 뒤 명시적으로 불러온다. 목록 표시·선택은 파일 이름과 시각만 읽는다."""
    st.subheader("저장된 제안 불러오기")
    st.caption("‘불러오기’를 누르면 저장 파일을 현재 계약으로 다시 검사합니다. "
               "모델 호출·승인 발급·작업 생성은 하지 않으며, 불러온 제안도 실행 승인이 아닙니다.")

    if "ui_saved_proposals" not in st.session_state:
        _refresh_saved(st)

    items = st.session_state["ui_saved_proposals"]
    times = {item["name"]: item["saved_at"].replace("T", " ")[:19] for item in items}

    with st.form("saved_proposal_restore"):
        if items:
            name = st.selectbox("저장된 제안", list(times), format_func=lambda value: f"{value} · 저장 {times[value]}",
                                help="최근 저장 순입니다. 저장 시각은 파일 수정 시각이며, 목록에 있다고 검증을 통과한 것은 아닙니다.")
        else:
            name = None
            st.info("저장된 제안이 없습니다. 새 요청으로 제안을 만들고 ‘2 제안 검토’에서 저장하세요.")

        refresh = st.form_submit_button("목록 새로고침", help="저장 폴더의 파일 이름과 수정 시각만 다시 읽습니다.")
        restore = st.form_submit_button("불러오기", type="primary", disabled=not items,
                                        help="선택한 제안을 기존 복원 검사기로 확인합니다. 모델·worker를 실행하지 않습니다.")

    if refresh:
        _refresh_saved(st)
        st.rerun()
        return

    if not restore or name is None:
        return

    dialogue = None

    try:
        dialogue = UiDialogue(settings)
        dialogue.ui_source_id = source["selection_id"]
        dialogue.restore_saved(name, settings)
    except Exception as error:
        if dialogue is not None:
            dialogue.close()

        _restore_error(st, error)
    else:
        st.session_state["ui_dialogue"] = dialogue
        st.rerun()


def _restore_error(st, error):
    # 저장 파일 내용이나 예외 원문을 화면에 그대로 노출하지 않는다.
    if isinstance(error, ImportError):
        st.error("대화 의존성이 없습니다. 현재 Python 환경의 requirements-agents.txt 설치를 확인하세요.")
    else:
        st.error("선택한 제안을 불러오지 못했습니다. 파일이 삭제됐거나 현재 등록 모듈·설정·계획 지문 "
                 "검사를 통과하지 못했습니다. 파일을 수정하지 않았습니다. ‘목록 새로고침’ 후 확인하세요.")


def render_review(st):
    st.header("2 제안 검토")
    dialogue = current_dialogue(st)
    checked = None if dialogue is None else dialogue.validated_proposal(st.session_state["ui_settings"])

    if checked is None:
        st.warning("현재 설정에서 검증된 제안이 없습니다. 요청 화면에서 먼저 제안을 생성하세요.")
        return

    pipeline = checked["proposal"]["pipeline"]
    st.success("모듈·버전·설정·품질 검사 의존성의 정적 검증을 통과한 제안입니다.")
    st.caption("입력 데이터 적합성 검사나 전처리 실행 승인은 아닙니다.")

    for index, step in enumerate(pipeline["steps"], start=1):
        st.subheader(f"{index}. {step['module_id']}")
        st.text(DESCRIPTIONS[step["module_id"]])
        st.caption("버전: " + step["version"] + " / 선행 단계: " + (", ".join(step["depends_on"]) or "없음"))
        st.json(step["settings"])

    with st.expander("제안 원문·계획 서명"):
        st.json(checked)

    saved = dialogue.saved_proposal(st.session_state["ui_settings"])

    if saved is not None:
        st.success("제안을 저장하고 복원 검사를 통과했습니다. 왼쪽 ‘3 입력 선택’으로 진행하세요.")
        st.text("저장 이름: " + saved["name"])
        st.text("로컬 파일: " + dialogue.saved["local_file"])
        return

    if dialogue.saved is not None:
        st.error("이전에 저장한 제안을 복원할 수 없거나 내용이 달라졌습니다. "
                 "기존 파일은 덮어쓰지 않습니다. 저장 파일을 확인한 뒤 새 대화를 시작하세요.")
        return

    st.caption("제안 JSON만 새 파일로 저장합니다. 요청 원문·대화·승인 토큰은 저장하지 않습니다.")
    with st.form("proposal_save_" + checked["plan_signature"]):
        name = st.text_input("제안 저장 이름", value="ui_proposal01",
                             help="소문자로 시작하는 영문 소문자·숫자·밑줄 이름을 지정하세요. 기존 파일은 덮어쓰지 않습니다.")
        confirmed = st.checkbox("모듈·순서·설정을 검토했으며 제안 저장에 동의합니다", value=False,
                                help="로컬 제안 저장에 대한 확인입니다. 입력 접근·실행 승인은 별도입니다.")
        submitted = st.form_submit_button("검토한 제안 저장", type="primary")

    if submitted:
        if not confirmed:
            st.warning("제안 내용을 검토하고 저장에 동의해 주세요.")
            return

        try:
            dialogue.save(name, st.session_state["ui_settings"], confirmed=True)
        except FileExistsError:
            st.error("같은 이름의 제안이 이미 있습니다. 기존 파일을 덮어쓰지 않으므로 다른 이름을 지정하세요.")
        except Exception:
            st.error("제안 저장·복원 검증을 완료하지 못했습니다. 이름·현재 설정·저장 파일을 확인하세요. 자동 재시도하지 않습니다.")
        else:
            st.rerun()

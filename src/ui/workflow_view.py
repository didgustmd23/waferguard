"""등록 모듈용 공통 5단계 화면. 특정 데이터 처리 함수나 CLI를 직접 호출하지 않는다."""

from .path_picker import path_input
from .workflow_service import UiWorkflow, workflow_modules
from .workflow_results import load_workflow_results
from .execution_view import render_result_files


def current(st):
    workflow = st.session_state.get("ui_workflow")
    return workflow if isinstance(workflow, UiWorkflow) else None


def flags(st):
    """메뉴 잠금은 세션 정보로만 판정한다. 파일·DB 정밀 조회를 하지 않는다."""
    workflow = current(st)
    selected = workflow is not None
    return {"proposal_available": selected,
            "saved_available": selected and workflow.reviewed,
            "candidate_available": selected and workflow.name is not None,
            "execution_available": selected and workflow.name is not None}


def request(st):
    st.header("1 요청·대화")
    st.info("등록 모듈을 직접 선택합니다. 이 경로에서는 LLM을 호출하지 않습니다.")
    specs = workflow_modules()
    mapping = {item["contract"]["module_id"]: item for item in specs}
    with st.form("workflow_select"):
        selected = st.selectbox("등록 모듈", tuple(mapping), key="workflow_module_select",
                                format_func=lambda key: mapping[key]["title"],
                                help="등록부의 공통 서비스가 연결된 모듈만 표시합니다.")
        submitted = st.form_submit_button("이 모듈 선택", type="primary")

    if submitted:
        previous = current(st)
        if previous is not None and previous.name:
            st.session_state.setdefault("ui_workflow_history", []).append(previous)

        st.session_state["ui_workflow"] = UiWorkflow(selected, mapping[selected]["contract"]["version"])
        st.rerun()

    workflow = current(st)
    if workflow is None:
        return

    st.success("선택한 모듈: " + mapping[workflow.module_id]["title"])
    with st.form("workflow_reconnect"):
        name = st.text_input("기존 작업 이름", help="기존 작업을 이어서 확인할 때만 입력하세요. 기록을 덮어쓰지 않습니다.", max_chars=128)
        connect = st.form_submit_button("기존 작업 확인·연결")

    if connect:
        try:
            with st.spinner("기존 작업 계약을 확인하고 있습니다."):
                workflow.connect(name)
        except Exception:
            st.error("기존 작업을 연결하지 못했습니다. 모듈·이름·코드 변경 여부를 확인하세요.")
        else:
            st.rerun()


def review(st):
    st.header("2 제안 검토")
    workflow = current(st)
    if workflow is None:
        return

    service = workflow.service()
    capabilities = service.capabilities()
    st.json({"module_id": workflow.module_id, "version": workflow.version,
             "creation_supported": capabilities["creation_supported"],
             "runtime": capabilities["runtime"]})
    st.caption("기존 계획의 선행 결과를 검증하며 한 단계씩 진행합니다. 생성·승인·실행은 각각 별도 확인입니다.")
    if not capabilities["runtime"]["supported"]:
        st.warning("현재 Python 환경의 WM 격리 실행은 검증되지 않았습니다. 호환 환경 확인 전에는 worker를 실행하지 않습니다.")

    if not capabilities["creation_supported"]:
        st.info("이 모듈은 기존 작업 연결만 지원합니다. 새 작업 생성은 기존 CLI를 사용하세요.")

    if st.button("모듈 검토 완료", key="workflow_review"):
        workflow.reviewed = True
        st.rerun()


def _field(st, field, prefix):
    """설정 명세의 고정 자료형만 지원한다. 사용자 코드·위젯 import는 받지 않는다."""
    key = prefix + field["name"]
    common = {"key": key, "help": field["help"]}
    if field["kind"] in {"directory", "json_file"}:
        return path_input(st, field["title"], kind=field["kind"], value=field["default"], **common)
    if field["kind"] == "text":
        return st.text_input(field["title"], value=field["default"], max_chars=4096, **common)
    if field["kind"] == "boolean":
        return st.checkbox(field["title"], value=field["default"], **common)
    if field["kind"] in {"integer", "number"}:
        return st.number_input(field["title"], value=field["default"],
                               min_value=field["minimum"], max_value=field["maximum"], **common)

    raise ValueError("등록되지 않은 설정 표시 자료형입니다.")


def inputs(st):
    st.header("3 입력 선택")
    workflow = current(st)
    if workflow is None or not workflow.reviewed:
        return

    if workflow.name is not None:
        st.success("연결한 작업: " + workflow.name)
        if workflow.creation_receipt:
            st.json(workflow.creation_receipt)
            st.caption("후속 단계는 위의 새 bindings 경로를 사용하세요. 기존 배정 파일은 변경하지 않았습니다.")
        return

    schema = workflow.service().creation_schema()
    if not schema["supported"]:
        st.info("현재 모듈은 새 작업 생성 미지원입니다. 1번에서 기존 작업을 연결하세요.")
        return

    st.caption(schema["notice"])
    with st.form("workflow_creation_" + workflow.module_id):
        values = {field["name"]: _field(st, field, "workflow_" + workflow.module_id + "_") for field in schema["fields"]}
        st.subheader("작업 자원 상한")
        defaults = st.session_state["ui_settings"]["limits"]
        limits = {}
        for key, title, maximum in (("memory_mib", "메모리 (MiB)", 32768),
                                    ("processes", "프로세스 수", 8), ("wall_seconds", "실행 시간 (초)", 86400)):
            limits[key] = st.number_input(title, min_value=1, max_value=maximum, value=defaults[key],
                                          help="이번 작업에만 적용하는 요청 상한입니다. 실제 강제는 기존 Job 실행기가 담당합니다.")
        preview = st.form_submit_button("생성 범위 검사·미리보기")

    if preview:
        try:
            with st.spinner("입력·선행 결과·생성 조건을 검사하고 있습니다. 큰 파일은 시간이 걸릴 수 있습니다."):
                workflow.preview_creation(values, limits)
        except Exception:
            st.error("생성 검사를 통과하지 못했습니다. 계획·배정·입력·설정·자원 상한을 확인하세요. 자동 재시도하지 않습니다.")

    if workflow.creation_preview is None:
        return

    st.json(workflow.creation_preview.packet)
    with st.form("workflow_create_confirm"):
        confirmed = st.checkbox("위 설정으로 새 작업과 배정 기록만 생성합니다", value=False,
                                help="실행 승인이나 worker 시작이 아닙니다. 입력이 바뀌면 미리보기를 다시 검사해야 합니다.")
        submitted = st.form_submit_button("새 작업 준비", type="primary")

    if submitted:
        try:
            workflow.create(values, limits, confirmed=confirmed)
        except Exception:
            st.error("작업을 생성하지 못했거나 결과를 확인하지 못했습니다. 기존 기록을 확인하세요. 자동 재생성하지 않습니다.")
        else:
            st.rerun()


def _command(st, workflow, action, confirmed=False):
    try:
        with st.spinner("현재 계약과 상태를 확인하고 있습니다. 같은 요청을 다시 제출하지 마세요."):
            workflow.command(action, confirmed=confirmed)
    except Exception:
        st.error("요청을 완료하지 못했습니다. 상태를 다시 확인하세요. 자동 재시도·재승인하지 않습니다.")
    else:
        st.rerun()


def approval(st):
    st.header("4 실행 승인")
    workflow = current(st)
    if workflow is None or workflow.name is None:
        return

    st.text("작업 이름: " + workflow.name)
    if st.button("현재 상태 확인", key="workflow_refresh"):
        _command(st, workflow, "refresh")

    state = workflow.state
    if state is None:
        st.info("현재 상태 확인을 눌러 계약을 검사하세요. 화면 갱신만으로 검사·승인·실행하지 않습니다.")
        return

    st.text("상태: " + state["status"] + " / 단계: " + state["stage"])
    if workflow.error:
        st.warning("이전 요청의 결과가 불명확하거나 실패했습니다. 현재 상태부터 다시 확인하세요.")
        return

    actions = {item["id"] for item in state["actions"]}
    if "next" in actions and st.button("현재 단계 한 번 진행", key="workflow_next",
                                       help="승인 필요 단계는 승인 대기로 전환합니다. 후속 단계를 자동 실행하지 않습니다."):
        _command(st, workflow, "next")

    if "approve" in actions:
        if st.button("승인 범위 검사", key="workflow_preview"):
            _command(st, workflow, "preview")

        if workflow.approval_preview is not None:
            st.json(workflow.approval_preview.to_dict()["summary"])
            with st.form("workflow_approve"):
                confirmed = st.checkbox("표시한 현재 단계의 범위를 승인합니다", value=False)
                submitted = st.form_submit_button("이번 단계 승인")
            if submitted:
                _command(st, workflow, "approve", confirmed)

    if "execute" in actions:
        if workflow.approval_id is None:
            workflow.approval_id = st.text_input("기존 미사용 승인 ID", help="재연결한 작업의 승인 ID입니다. 유효성은 실행기가 다시 검사합니다.", max_chars=128) or None

        with st.form("workflow_execute"):
            confirmed = st.checkbox("승인한 현재 단계만 실행합니다", value=False)
            submitted = st.form_submit_button("승인 단계 실행", disabled=not state["runtime"]["supported"])
        if not state["runtime"]["supported"]:
            st.warning("현재 환경은 WM 격리 실행 지원 대상이 아닙니다. 버전 제한을 우회하지 않습니다.")
        if submitted:
            _command(st, workflow, "execute", confirmed)

    if not actions:
        st.info("진행할 행동이 없습니다. 완료·품질 중단·실패 기록을 5번에서 확인하세요.")


def results(st):
    st.header("5 결과 확인")
    workflow = current(st)
    if workflow is None or workflow.name is None:
        return

    st.text("작업 이름: " + workflow.name)
    if st.button("결과 정합성 검사·열기", key="workflow_results"):
        st.session_state.pop("ui_workflow_results", None)
        try:
            with st.spinner("로컬 결과 계약과 표시 파일을 검사하고 있습니다."):
                result = load_workflow_results(workflow)
                st.session_state["ui_workflow_results"] = (workflow.module_id, workflow.name, result)
        except Exception:
            st.error("표시 가능한 결과를 검증하지 못했습니다. 실패 부분 결과를 완료로 표시하지 않습니다.")

    loaded = st.session_state.get("ui_workflow_results")
    if loaded is None or loaded[:2] != (workflow.module_id, workflow.name):
        return

    result = loaded[2]
    st.text("결과 상태: " + result["status"])
    st.caption(result["notice"])
    for path in result["directories"]:
        st.text("로컬 결과 폴더: " + path)
    render_result_files(st, result["files"])

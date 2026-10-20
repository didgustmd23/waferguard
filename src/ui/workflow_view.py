"""등록 모듈용 공통 5단계 화면. 특정 데이터 처리 함수나 CLI를 직접 호출하지 않는다."""

from .path_picker import path_input
from .workflow_service import UiWorkflow, workflow_modules
from .workflow_results import load_workflow_results
from .execution_view import render_result_files
from .progress_view import show_operation_progress
from .runtime_notice import render_permissions, require_execution_permissions
from .submit_state import clear_unsubmitted_execution
from .approval_flow_view import render_approval_flow
from .creation_submit import mark_creation_pending, creation_submission
from .workflow_feedback import (STATE_LABELS, STAGE_LABELS, diagnose_paths, render_paths,
                                show_error)


def current(st):
    from .csv_plan_service import selected

    if selected(st.session_state):
        plan = st.session_state.get("ui_csv_plan")
        return None if plan is None else plan.active

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
            with show_operation_progress(st, "기존 작업 계약을 확인하고 있습니다."):
                workflow.connect(name)
        except Exception as error:
            show_error(st, error, "기존 작업을 연결하지 못했습니다. 모듈·이름·코드 변경 여부를 확인하세요.")
        else:
            st.rerun()


def review(st):
    st.header("2 제안 검토")
    workflow = current(st)
    if workflow is None:
        return

    service = workflow.service()
    capabilities = service.capabilities()
    title = next((item["title"] for item in workflow_modules()
                  if item["contract"]["module_id"] == workflow.module_id), workflow.module_id)
    st.subheader(title)
    if capabilities["runtime"]["supported"]:
        st.success("현재 환경은 이 모듈의 실행 허용 조건에 맞습니다. 아직 데이터를 분석한 것은 아닙니다.")
    st.text("전처리는 CPU로 수행합니다. GPU 학습과는 별도입니다.")
    if capabilities["runtime"].get("notice"):
        st.info(capabilities["runtime"]["notice"])
    render_permissions(st)
    with st.expander("모듈 기술 상세"):
        st.json({"module_id": workflow.module_id, "version": workflow.version,
                 "creation_supported": capabilities["creation_supported"], "runtime": capabilities["runtime"]})
    st.caption("기존 계획의 선행 결과를 검증하며 한 단계씩 진행합니다. 생성·승인·실행은 각각 별도 확인입니다.")
    if not capabilities["runtime"]["supported"]:
        st.warning("현재 Python 환경의 이 모듈 실행은 지원 조건에 맞지 않습니다. 호환성 확인 전에는 실행하지 않습니다.")

    if not capabilities["creation_supported"]:
        st.info("이 모듈은 기존 작업 연결만 지원합니다. 새 작업 생성은 기존 CLI를 사용하세요.")

    st.caption("다음: 모듈 검토 완료 → 3 입력 선택. 생성·승인·실행은 각각 별도 확인입니다.")

    if st.button("모듈 검토 완료", key="workflow_review", disabled=workflow.reviewed):
        workflow.reviewed = True
        st.rerun()

    if workflow.reviewed:
        st.success("모듈 검토를 완료했습니다. 왼쪽 ‘3 입력 선택’에서 입력과 설정을 준비하세요.")


def _field(st, field, prefix):
    """설정 명세의 고정 자료형만 지원한다. 사용자 코드·위젯 import는 받지 않는다."""
    key = prefix + field["name"]
    common = {"key": key, "help": field["help"]}
    if field["kind"] in {"directory", "json_file", "csv_file"}:
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
            if workflow.creation_receipt.get("bindings"):
                st.text("후속 작업 배정 파일: " + workflow.creation_receipt["bindings"])
            with st.expander("생성 기록 기술 상세"):
                st.json(workflow.creation_receipt)
            st.caption("후속 단계는 위의 새 bindings 경로를 사용하세요. 기존 배정 파일은 변경하지 않았습니다.")
        return

    schema = workflow.service().creation_schema()
    if not schema["supported"]:
        st.info("현재 모듈은 새 작업 생성 미지원입니다. 1번에서 기존 작업을 연결하세요.")
        return

    st.caption(schema["notice"])
    if workflow.creation_failure is not None:
        show_error(st, workflow.creation_failure, "입력·생성 조건 또는 작업 준비 결과를 확인하지 못했습니다. 경로 진단과 기존 기록을 확인하세요.")

    busy = workflow.creation_busy
    if busy:
        st.info("요청을 처리 중입니다. 생성 범위 검사와 새 작업 준비 버튼을 잠갔습니다.")

    with st.form("workflow_creation_" + workflow.module_id):
        defaults = getattr(workflow, "creation_defaults", {})
        prefix = getattr(workflow, "creation_widget_prefix", "workflow_" + workflow.module_id + "_")
        values = {field["name"]: _field(st, dict(field, default=defaults.get(field["name"], field["default"])),
                                      prefix) for field in schema["fields"]}
        uses_job = schema.get("uses_job", True)
        st.subheader("작업 자원 상한" if uses_job else "로컬 전달 작업")
        defaults = st.session_state["ui_settings"]["limits"]
        limits = {} if uses_job else dict(defaults)
        if not uses_job:
            st.caption("고정 파일의 로컬 복사입니다. 전달 바이트 상한은 적용하지만 AppContainer·Job의 시간·RAM 제한을 적용하는 작업은 아닙니다.")
        resource_fields = (("memory_mib", "메모리 (MiB)", 32768),
                           ("processes", "프로세스 수", 8), ("wall_seconds", "실행 시간 (초)", 86400)) if uses_job else ()
        for key, title, maximum in resource_fields:
            limits[key] = st.number_input(title, min_value=1, max_value=maximum, value=defaults[key],
                                          help="이번 작업에만 적용하는 요청 상한입니다. 실제 강제는 기존 Job 실행기가 담당합니다.")
        st.form_submit_button("생성 범위 검사·미리보기", disabled=busy,
                              on_click=mark_creation_pending, args=(st, workflow, "preview"))

    render_paths(st, getattr(workflow, "path_diagnostics", []))

    if workflow.creation_preview is not None:
        st.success("생성 조건 확인 완료 — 아래 확인을 제출하면 작업만 준비합니다. 분석은 아직 실행하지 않습니다.")
        st.text("새 작업 이름: " + workflow.creation_preview.packet["run_name"])
        with st.expander("생성 범위 기술 상세"):
            st.json(workflow.creation_preview.packet)
        with st.form("workflow_create_confirm"):
            confirmation_key = "workflow_create_confirm_" + workflow.module_id + "_" + str(workflow.creation_selection)
            st.checkbox(getattr(workflow, "creation_confirmation_label", "위 설정으로 새 작업과 배정 기록만 생성합니다"), value=False,
                        key=confirmation_key, disabled=busy,
                        help="실행 승인이나 worker 시작이 아닙니다. 입력이 바뀌면 미리보기를 다시 검사해야 합니다.")
            st.form_submit_button("새 작업 준비", type="primary", disabled=busy,
                                  on_click=mark_creation_pending, args=(st, workflow, "create", confirmation_key))

    # 두 폼을 비활성화해 표시한 뒤 명시적인 클릭 한 건만 처리한다.
    action = workflow.creation_requested
    if action is not None:
        try:
            with creation_submission(workflow, action):
                if action == "preview":
                    workflow.creation_preview = None
                    workflow.creation_selection = None
                    workflow.path_diagnostics = diagnose_paths(schema, values)
                    if any(not row["valid"] for row in workflow.path_diagnostics):
                        raise ValueError("selected path rejected")

                    with show_operation_progress(st, "입력·선행 결과·생성 조건을 검사하고 있습니다."):
                        workflow.preview_creation(values, limits)
                else:
                    with show_operation_progress(st, "새 작업과 입력 복사본을 준비하고 있습니다."):
                        workflow.create(values, limits, confirmed=True)
        except Exception as error:
            workflow.creation_failure = error
            st.rerun()
        else:
            st.rerun()


def _command(st, workflow, action, confirmed=False):
    if action in {"approve", "execute"} and not require_execution_permissions(st):
        clear_unsubmitted_execution(workflow)
        return

    try:
        with show_operation_progress(st, "현재 요청을 처리하고 있습니다. 같은 요청을 다시 제출하지 마세요."):
            result = workflow.command(action, confirmed=confirmed)
    except Exception as error:
        show_error(st, error, "요청을 완료하지 못했습니다. 상태를 다시 확인하세요.")
    else:
        if result.get("status") == "busy":
            st.info("현재 요청을 처리 중입니다. 다시 제출하지 마세요.")
            return
        st.rerun()


def approval(st):
    st.header("4 실행 승인")
    workflow = current(st)
    if workflow is None or workflow.name is None:
        return

    render_approval_flow(st, workflow, workflow=True)


def results(st):
    st.header("5 결과 확인")
    workflow = current(st)
    if workflow is None or workflow.name is None:
        return

    st.text("작업 이름: " + workflow.name)
    if st.button("실행 상태 확인", key="workflow_result_state"):
        _command(st, workflow, "refresh")

    state = workflow.state
    if state is None:
        st.info("실행 상태 확인을 눌러 저장된 작업 기록부터 확인하세요.")
        return

    st.text("작업 상태: " + STATE_LABELS.get(state["status"], state["status"]))
    if state["status"] not in {"completed", "blocked_input"}:
        if state["status"] in {"failed_runtime", "security_locked", "cancelled"}:
            st.error("이 작업은 정상 완료되지 않았습니다. 결과 파일 손상으로 단정할 수는 없습니다.")
            st.text("중단 단계: " + STAGE_LABELS.get(state["stage"], state["stage"]))
            st.info("실행 실패 작업은 UI 서버 권한과 로컬 실행 기록을 확인하세요. "
                    "상세 원인이 저장되지 않은 기록일 수 있습니다. 기존 승인으로 자동 재실행하지 않습니다.")
            render_permissions(st)
        else:
            st.info("아직 결과를 열 수 있는 상태가 아닙니다. 4 실행 승인에서 현재 단계를 확인하세요.")
        return

    if state["status"] == "blocked_input":
        st.warning("입력 차단 또는 품질 검토로 멈춘 작업입니다. 저장된 검토 보고서가 있으면 아래에서 확인할 수 있습니다.")
    if st.button("결과 정합성 검사·열기", key="workflow_results"):
        st.session_state.pop("ui_workflow_results", None)
        try:
            with show_operation_progress(st, "로컬 결과 계약과 표시 파일을 검사하고 있습니다."):
                result = load_workflow_results(workflow)
                st.session_state["ui_workflow_results"] = (workflow.module_id, workflow.name, result)
        except Exception as error:
            show_error(st, error, "실행 기록은 있으나 결과 파일·해시·표시 계약을 확인하지 못했습니다. "
                       "결과를 수정하거나 worker를 재실행하지 말고 로컬 기록을 확인하세요.")

    loaded = st.session_state.get("ui_workflow_results")
    if loaded is None or loaded[:2] != (workflow.module_id, workflow.name):
        return

    result = loaded[2]
    st.text("결과 상태: " + STATE_LABELS.get(result["status"], result["status"]))
    st.caption(result["notice"])
    for path in result["directories"]:
        st.text("로컬 결과 폴더: " + path)
    render_result_files(st, result["files"])
    from .commentary_view import render_commentary

    render_commentary(st, workflow, result)

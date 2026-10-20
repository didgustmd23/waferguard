"""CSV 실행안도 같은 5단계 화면에서 제안·검토·준비하고 기존 승인·결과 UI를 재사용한다."""

import json
from pathlib import Path

from src.agents import llm_workflow_store as store
from src.agents.llm_workflow_contract import validate_workflow_proposal
from src.agents.llm_workflow_proposal import checked_context, restore_context
from src.agents.worker_contract import _decode
from src.agents.workflow_service_registry import CSV_IDS
from .csv_plan_service import CsvPlanDialogue, CsvPlanSession
from .state import MODES
from . import workflow_view


def flags(st):
    dialogue = st.session_state.get("ui_csv_dialogue")
    checked = None if dialogue is None else dialogue.validated_proposal(st.session_state["ui_settings"])
    plan = st.session_state.get("ui_csv_plan")
    return {"proposal_available": checked is not None or plan is not None,
            "saved_available": plan is not None,
            "candidate_available": plan is not None and plan.active is not None and plan.active.name is not None,
            "execution_available": plan is not None and plan.active is not None and plan.active.name is not None}


def _error(st):
    st.error("현재 요청을 확인하지 못했습니다. 규칙·파일·연결 기록을 확인하세요. 자동 재시도하지 않습니다.")


def request(st):
    st.header("1 요청·대화")
    st.caption("CSV 실행안 제안용입니다. 원본·경로·통계를 문장에 넣지 마세요. 실행·승인은 별도입니다.")
    settings = st.session_state["ui_settings"]

    with st.form("csv_plan_restore"):
        name = st.text_input("저장된 CSV 실행안 이름", help="불러오기는 현재 기록 확인만 하며 모델·승인을 재실행하지 않습니다.")
        restore = st.form_submit_button("CSV 실행안 불러오기")

    if restore:
        st.session_state.pop("ui_csv_plan", None)
        try:
            plan = CsvPlanSession(store.load_workflow(name))
            plan.restore()
            st.session_state["ui_csv_plan"] = plan
            previous = st.session_state.pop("ui_csv_dialogue", None)

            if previous is not None:
                previous.close()
        except Exception:
            _error(st)
        else:
            st.success("저장 실행안을 연결했습니다. 3 입력 선택에서 현재 작업을 확인하세요.")

    dialogue = st.session_state.get("ui_csv_dialogue")
    if dialogue is not None and not dialogue.matches(settings):
        dialogue.close()
        st.session_state.pop("ui_csv_dialogue", None)
        dialogue = None

    if dialogue is None:
        example = Path(__file__).resolve().parents[2] / "configs/agents/llm/workflow_proposal_example.json"
        steps = json.loads(example.read_text(encoding="utf-8"))["workflow"]["steps"]
        default = {step["module_id"]: step["settings"] for step in steps}
        default["csv_local_handoff"] = {}

        with st.form("csv_plan_rules"):
            st.caption("예제 규칙입니다. 전체 유효값을 검토·편집한 뒤 명시적으로 선택하세요. 실제 입력 적합성을 보증하지 않습니다.")
            rules = st.text_area("CSV 실행안 규칙 JSON", value=json.dumps(default, ensure_ascii=False, indent=2), height=220,
                                 help="현재는 새 분할부터 시작합니다. 규칙 변경은 새 대화가 필요합니다.")
            apply = st.form_submit_button("이 규칙으로 새 대화 준비", disabled=settings["mode"] != MODES[0])

        if apply:
            try:
                context = {"allowed_modules": tuple((item, "1.0.0") for item in CSV_IDS),
                           "selected_settings": _decode(rules)}
                checked_context(**context)
                st.session_state["ui_csv_dialogue"] = CsvPlanDialogue(settings, context)
                st.session_state.pop("ui_csv_plan", None)
            except Exception:
                _error(st)
            else:
                st.rerun()

        return

    for exchange in dialogue.history:
        st.text("요청 / 답변: " + exchange["user"])
        for question in exchange["questions"]:
            st.text("확인 질문: " + question)

    if st.button("CSV 새 대화 시작", disabled=dialogue.phase == "sending"):
        dialogue.close()
        st.session_state.pop("ui_csv_dialogue", None)
        st.session_state.pop("ui_csv_plan", None)
        st.rerun()
        return

    accepting = dialogue.phase in {"new", "awaiting_answer"}

    if dialogue.phase == "proposal_validated":
        st.success("제안을 검증했습니다. 2 제안 검토로 진행하세요.")
    elif not accepting:
        st.warning("대화가 종료되었거나 실패했습니다. 자동 재전송하지 않습니다.")

    with st.form("csv_plan_request"):
        text = st.text_area("CSV 요청 / 확인 질문 답변", disabled=not accepting,
                            placeholder="예: 선택한 규칙으로 분할부터 후보 산출·적용까지 제안해 주세요.")
        preview = st.form_submit_button("CSV 전송 내용 미리보기", disabled=not accepting)

    if preview:
        try:
            dialogue.prepare(text, settings)
        except Exception:
            _error(st)

    if dialogue.pending is None:
        return

    pending = dialogue.pending
    with st.expander("실제 로컬 모델 전달 내용", expanded=True):
        st.json(pending["preview"])

    with st.form("csv_send_" + pending["command_id"]):
        confirmed = st.checkbox("위 공개 규칙과 문장을 로컬 모델에 전달합니다")
        send = st.form_submit_button("CSV 확인한 내용 전송")

    if send:
        if not confirmed:
            st.warning("전달 자료를 확인해 주세요.")
            return

        try:
            with st.spinner("로컬 모델이 제안 중입니다. 자동 재시도하지 않습니다."):
                dialogue.send(pending["command_id"], settings, confirmed=True)
        except Exception:
            _error(st)
        else:
            st.rerun()


def review(st):
    st.header("2 제안 검토")
    plan = st.session_state.get("ui_csv_plan")

    if plan is not None:
        st.success("저장 실행안: " + plan.saved["name"] + " — 3 입력 선택으로 진행하세요.")
        st.json(plan.saved["proposal"].to_dict()["proposal"])
        return

    dialogue = st.session_state.get("ui_csv_dialogue")
    checked = None if dialogue is None else dialogue.validated_proposal(st.session_state["ui_settings"])

    if checked is None:
        st.warning("먼저 현재 설정에서 검증된 CSV 제안을 생성하세요.")
        return

    st.json(checked["proposal"])
    st.caption("제안·규칙만 저장합니다. 대화 원문·승인 토큰은 저장하지 않습니다.")

    with st.form("csv_plan_save"):
        name = st.text_input("CSV 실행안 저장 이름", value="csv_plan01")
        confirmed = st.checkbox("제안 순서·규칙을 검토했으며 로컬 저장에 동의합니다")
        save = st.form_submit_button("검토한 CSV 실행안 저장")

    if save and confirmed:
        try:
            context = restore_context(dialogue.context_json)
            proposal = validate_workflow_proposal(json.dumps(checked["proposal"]), **context)
            # 현재 UI 지원 범위를 저장하기 전에 확인한다.
            if context["existing_inputs"] or checked["proposal"]["workflow"]["steps"][0]["module_id"] != CSV_IDS[0]:
                raise ValueError("새 분할부터 시작하는 실행안이 필요합니다.")

            plan = CsvPlanSession(store.save_workflow(name, proposal, context))
            plan.restore()
            st.session_state["ui_csv_plan"] = plan
        except Exception:
            _error(st)
        else:
            st.rerun()
    elif save:
        st.warning("제안 순서·규칙을 검토하고 저장에 동의해 주세요.")


def _progress(st):
    plan = st.session_state.get("ui_csv_plan")

    if plan is None:
        return None

    st.caption(f"실행안 {plan.saved['name']} · 작업 {plan.index + 1}/{len(plan.steps)} · {plan.steps[plan.index]['module_id']}")
    if plan.active is not None and plan.active.state is not None and plan.active.state["status"] == "completed":
        if plan.index < len(plan.steps) - 1 and st.button("완료 증거 확인·다음 작업 준비"):
            try:
                plan.advance()
            except Exception:
                _error(st)
            else:
                st.session_state.pop("ui_workflow_results", None)
                st.session_state["ui_screen"] = "3 입력 선택"
                st.rerun()

    return plan


def inputs(st):
    plan = _progress(st)

    if plan is None:
        return

    report = getattr(plan.active, "candidate_report", None)
    if report is not None:
        st.caption("검증한 후보 보고서의 로컬 표시 복사본입니다. 적용 준비 전에 검토하세요.")
        with st.expander("고정 후보 보고서", expanded=True):
            st.text(report)

    workflow_view.inputs(st)


def approval(st):
    if _progress(st) is not None:
        workflow_view.approval(st)


def results(st):
    if _progress(st) is not None:
        workflow_view.results(st)

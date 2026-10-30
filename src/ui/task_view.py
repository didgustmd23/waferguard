"""통합 작업 화면. 요청·진행 상황·필요한 결정·최종 결과 네 영역만 두고 기술 정보는 접어 둔다.

데이터 형태·도메인을 모른다. 유형별 문구·양식·결과 요약은 어댑터와 양식 등록부에서 받는다.
화면 표시만으로는 복사·worker·모델 호출이 없다. 버튼 제출 때만 작업 조정기를 호출하며, 실패는 자동으로 다시 시도하지 않는다.
제출은 task_submit으로 예약만 하고, 잠긴 화면을 다 그린 뒤 render_task 끝에서 한 번 실행한다(처리 중 재실행으로 끊기지 않게).
입력 자료에서 온 문자열(컬럼명·모델 문장·질문 대상)은 일반 텍스트나 표로만 표시하고 마크다운으로 해석하지 않는다.
"""

import pandas as pd

from src.agents import task_adapters, task_rag, task_store
from src.agents import task_orchestrator as orchestrator
from src.agents.task_plan import PlanError, flatten
from src.agents.task_report import chart_rows, code_label, recommendation_notice, table_columns
from . import runtime_notice, task_submit
from .path_picker import path_input
from .runtime_notice import render_permissions
from .task_forms import form_for


PHASE_LABELS = {"created": "작업 준비", "diagnosing": "입력 확인·진단 중", "awaiting_input": "입력 해석 선택 필요",
                "awaiting_review": "계획 확인 필요",
                "processing": "처리 단계 진행 중", "awaiting_apply": "적용 확인 필요",
                "applying": "적용·전달 중", "completed": "완료", "needs_attention": "확인이 필요한 문제"}
# 조정기가 처리 도중에만 두는 단계와, 사용자 결정·완료를 기다리며 멈춘 단계.
IN_PROGRESS = {"diagnosing", "processing", "applying"}
RESTING = {"awaiting_input", "awaiting_review", "awaiting_apply", "completed", "needs_attention"}
DETAIL_LABELS = {"duplicated_names": "중복된 이름 수", "affected_columns": "영향받는 컬럼 수"}
DIAGNOSTIC_LABELS = {"INPUT_INVALID": "입력 검증 실패", "ACCESS_DENIED": "파일 접근 거부", "IO_FAILED": "파일 입출력 실패",
                     "RUNTIME_FAILED": "실행 중 오류", "PASSED": "통과"}
NO_ANSWER = "(답하지 않음)"
INPUT_STATUS = {"inspecting": "검사 중", "waiting": "선택 대기", "materializing": "작업용 표 생성 중", "materialized": "완료"}
MODEL_LIMITATION_NOTICE = (
    "현재 로컬 모델의 한계: 확인한 사례에서 진단 수치 해석, 라벨·그룹 구분, 전처리 권장 이유에 오류가 관찰됐습니다. "
    "응답 형식 검사를 통과해도 내용의 타당성을 보장하지 않습니다. 계산 결과와 모델 의견을 구분해 확인하세요. "
    "이 한계를 모든 모델에 일반화하지 않으며, 다른 로컬·API 모델의 품질은 별도 평가가 필요합니다."
)
RECORD_ERRORS = {
    "RECORD_NOT_FOUND": "작업 기록을 찾지 못했습니다. 기록 폴더가 옮겨졌거나 지워졌을 수 있습니다. 새 작업으로 시작하세요.",
    "RECORD_TOO_LARGE": "작업 기록이 크기 상한(512 KiB)을 넘어 열지 않았습니다. 기록은 그대로 두었습니다. 컬럼이 매우 많은 입력이면 관리자에게 알려 주세요.",
    "RECORD_PROTECT_FAILED": "작업 폴더를 보호 권한으로 만들지 못했습니다. 디스크 공간과 logs/agents/tasks 폴더 권한·상위 경로 링크를 "
                             "확인한 뒤 다시 시작하세요. 작업은 만들지 않았습니다.",
    "RECORD_UNPROTECTED": "작업 기록이 보호 경계 밖에 있어(이전 버전에서 만든 작업이거나 다른 사용자·그룹의 쓰기 권한이 있음) 승인·진행·복구에 "
                          "쓰지 않았습니다. 권한을 자동으로 고치지 않습니다. 저장된 결과는 그대로 볼 수 있습니다. 새 작업으로 시작하세요.",
    "RECORD_OWNER_MISMATCH": "다른 Windows 사용자가 만든 작업이라 열지 않았습니다. 해당 사용자로 실행한 UI에서 여세요.",
    "RECORD_INVALID": "작업 기록 형식이 현재 버전과 맞지 않아 열지 않았습니다. 기록은 그대로 두었습니다.",
    "RECORD_UNREADABLE": "작업 기록 파일을 읽지 못했습니다(형식 오류 또는 링크). 기록은 그대로 두었습니다.",
    "RECORD_PART_INVALID": "작업 기록의 분리 저장 파일(parts)이 없거나 기록의 해시와 달라 열지 않았습니다. 기록은 그대로 두었습니다.",
}


RECOVERY_ERRORS = {
    "RECOVERY_NOT_APPLICABLE": "완료 기록 복구 대상 상태가 아닙니다.",
    "RECOVERY_RECORD_MISMATCH": "작업 기록의 단계·이름·입력이 실행 기록과 맞지 않습니다.",
    "RECOVERY_CONSENT_INVALID": "동의 기록의 형식·내용 해시가 확정 계획과 맞지 않습니다.",
    "RECOVERY_STEP_UNVERIFIED": "단계 실행을 현재 코드로 다시 열어 검증하지 못했습니다. 실행 뒤 코드가 바뀌었거나 결과가 바뀌었을 수 있습니다.",
    "RECOVERY_CANDIDATE_CHANGED": "확인한 적용 대상 서명이 바뀌었습니다.",
    "RECOVERY_RESULT_UNVERIFIED": "결과 파일 해시·커밋 기록 검증을 통과하지 못했습니다.",
}


def current_task(st):
    task_id = st.session_state.get("ui_task_id")

    if task_id is None:
        return None

    try:
        return task_store.load_task(task_id)
    except task_store.TaskRecordError as error:
        st.error(RECORD_ERRORS.get(error.code, RECORD_ERRORS["RECORD_UNREADABLE"]))
    except (OSError, ValueError):
        st.error(RECORD_ERRORS["RECORD_UNREADABLE"])

    # 기록은 지우지 않는다. 화면 연결만 풀어 새 작업이나 다른 작업을 열 수 있게 한다.
    st.session_state.pop("ui_task_id", None)
    _new_task_button(st, "task_new_after_record_error")
    return None


def _new_task_button(st, key):
    st.button("새 작업 시작", key=key, disabled=task_submit.locked(st), on_click=_new_task,
              args=(st, task_submit.generation(st)))


def _new_task(st, seen):
    # 처리 중이거나 이전 화면에서 늦게 도착한 클릭은 작업 연결을 풀지 않는다.
    if task_submit.idle(st, seen):
        st.session_state.pop("ui_task_id", None)


def _stop_revise(st, task_id, seen):
    if task_submit.idle(st, seen):
        st.session_state.pop("task_revise_" + task_id, None)


def _start_revise(st, task_id, attempt, seen):
    if task_submit.idle(st, seen):
        st.session_state["task_revise_" + task_id] = attempt


def _reserve_button(st, action, task_id, bound=None):
    """실행 제출 버튼의 공통 인자. 잠금 중에는 비활성이고, 클릭은 callback에서 예약만 한다.

    bound는 이 화면에 그린 값(서버 대조값 등)이다. 클릭한 화면에서 사용자가 본 값으로 실행하도록 버튼 인자에 묶는다.

    실행에 쓰는 값 위젯(동의·경로·목표·선택, 양식 밖 동의 체크박스·검색 근거 선택 포함)은 잠그지 않는다. Streamlit은
    비활성으로 그린 위젯의 이번 실행 값을 버리므로, 체크와 클릭이 한 번의 재실행에 합쳐지면 제출 값이 사라진다.
    제출 버튼이 잠기면 값 변경은 실행을 일으키지 않고, 처리 중의 재실행 요청은 실행이 끝난 뒤에 반영된다.
    잠긴 위젯에 늦게 도착한 클릭은 Streamlit이 callback을 실행하지 않는다.
    """
    return {"disabled": task_submit.locked(st), "on_click": task_submit.reserve,
            "args": (st, action, task_id, task_submit.generation(st), bound)}


def _checked(st, agreed, warning):
    """예약한 실행의 실행 전 확인. 동의·권한이 없으면 잠금을 풀고 안내와 함께 화면을 다시 그린다."""
    if not agreed:
        task_submit.decline(st, [("warning", warning)])

    if not runtime_notice.execution_permitted():
        task_submit.decline(st, [("error", runtime_notice.EXECUTION_PERMISSION_ERROR)])


def _record_error(error):
    return ("error", RECORD_ERRORS.get(error.code, RECORD_ERRORS["RECORD_UNREADABLE"]))


def _verify_stored(task_id, revision):
    """결과를 받지 못한 실행의 완료를 저장된 작업 기록으로만 판정한다(task_submit이 다음 재실행에서 호출).

    기록을 다시 열어(보호·형식 검증) 대기·완료·문제 단계에 있고, 제출할 때 본 revision보다 기록이 진행했을 때만 끝난 것으로
    본다. 진행 단계·변화 없음·열 수 없음은 확인 불가다. 문제 단계도 끝난 것으로 보지만 성공으로 표시하지 않고 단계만 알린다.
    작업 시작은 제출 때 작업 ID가 없으므로 실행이 남긴 작업 ID(progress)로 대조한다.
    """
    def verify(progress):
        target = task_id or progress.get("link")

        if target is None:
            return None

        record = task_store.load_task(target)

        if record["phase"] not in RESTING or (revision is not None and record["revision"] <= revision):
            return None

        # 다른 탭이 같은 작업을 진행시켰을 수도 있으므로 이 요청의 성공으로 단정하지 않고 저장된 단계만 알린다.
        return ("이 요청의 결과를 화면에서 직접 받지 못했지만, 저장된 작업 기록은 처리 도중이 아닌 단계로 진행했습니다. 현재 단계: "
                + PHASE_LABELS[record["phase"]] + ". 아래 화면의 안내에 따라 진행하세요. 자동으로 다시 실행하지 않았습니다.")

    return verify


def _rows(st, rows, columns=None):
    # 컬럼명·값은 입력 데이터에서 온 문자열이다. 마크다운이 아니라 표로만 표시한다.
    # 저장된 사실은 키가 정렬돼 있으므로 열 순서는 Markdown 보고서와 같은 table_columns로 정한다.
    # columns는 표를 만드는 쪽이 정한 순서다(없으면 호환용 기본 순서).
    if rows:
        columns = table_columns(rows, columns)
        st.dataframe(pd.DataFrame([{key: str(row.get(key, "")) for key in columns} for row in rows], columns=columns),
                     hide_index=True)
    else:
        st.text("(없음)")


def _chart(st, chart):
    """보고서 그래프 한 개. 값이 없으면 사유만 보인다. 이름·문구는 Markdown으로 해석하지 않고 축·범례 글자로만 쓴다."""
    import altair as alt

    st.text("■ " + chart["title"])

    for text in chart["notes"]:
        st.text("- " + text)

    if not chart["rows"]:
        return

    category, values = chart["category"], chart["values"]
    # 막대 순서는 사실의 행 순서 그대로다(구간 순서·건수 순위). 축 이름 길이는 사실에서 이미 제한했다.
    order = [row[category] for row in chart["rows"]]
    data = pd.DataFrame([{category: row[category], "계열": key, "값": row[key]} for row in chart["rows"] for key in values])
    # 가로 막대(클래스처럼 이름이 긴 축)는 이름 축을 y에, 값을 x에 둔다. 계열이 여럿이면 같은 이름 안에 나란히 둔다.
    across, along, offset = (alt.Y, alt.X, alt.YOffset) if chart["horizontal"] else (alt.X, alt.Y, alt.XOffset)
    # 한글 라벨은 그릴 때 계산보다 넓어질 수 있어, 가로 막대의 이름 축 제목은 라벨 옆이 아니라 축 위에 가로로 둔다.
    title = {"titleAngle": 0, "titleAlign": "right", "titleBaseline": "bottom", "titleX": 0, "titleY": -6} if chart["horizontal"] else {}
    encoding = [across(category + ":N", sort=order, title=category, axis=alt.Axis(labelLimit=240, **title)),
                along("값:Q", title=chart["unit"]),
                alt.Color("계열:N", sort=list(values), title=None, legend=alt.Legend(orient="bottom"))]

    if len(values) > 1:
        encoding.append(offset("계열:N", sort=list(values)))

    # 높이는 막대 영역만 정한다(fit-x). 기본 fit은 축·범례까지 이 높이에 넣어 가로 막대가 0px로 눌린다.
    height = max(80, 28 * len(order) * len(values)) if chart["horizontal"] else 240
    figure = alt.Chart(data).mark_bar().encode(*encoding, tooltip=[category + ":N", "계열:N", "값:Q"])
    st.altair_chart(figure.properties(height=height, autosize=alt.AutoSizeParams(type="fit-x", contains="padding")),
                    width="stretch")
    _rows(st, chart_rows(chart))


def render_task(st):
    st.header("통합 작업")
    st.caption("데이터와 목표를 주면 진단 → 권장안·질문 → 계획 확인 → 처리 → 적용 → 결과까지 이어서 진행합니다. "
               "필요한 결정만 묻고 그 사이는 기존 격리 실행기로 자동 진행합니다.")
    # 끝난 실행의 결과(작업 연결·안내)는 app.py가 재실행 시작에서 task_submit.reconcile로 반영했다.
    # 실행 중 안내·spinner는 실행한 버튼 바로 아래에 둔다(task_submit.execute·wait_if_running).
    task = current_task(st)

    if task is None:
        task_submit.show_notice(st)
        _render_request(st)
        task_submit.finish(st)
        return

    _render_progress(st, task)
    task_submit.show_notice(st)
    _render_decision(st, task)

    if task["phase"] == "completed":
        _render_result(st, task)

    with st.expander("기술 상세"):
        st.json({"task_id": task["task_id"], "kind": task["kind"], "phase": task["phase"], "attempt": task["attempt"],
                 "diagnostic_run": (task["diagnostic"] or {}).get("run_name"),
                 "runs": {step: item["run_name"] for step, item in task["steps"].items()},
                 "consents": {key: {"consent_id": value["consent_id"], "expires_at": value["expires_at"]}
                              for key, value in task["consents"].items()}})

    # 예약한 버튼을 그리지 못했거나(단계가 바뀜) 다른 재실행이 처리 중이면 여기서 정리하거나 기다린다.
    task_submit.finish(st)


def _render_request(st):
    st.subheader("요청")
    settings = st.session_state["ui_settings"]
    limits, model = settings["limits"], settings["model"]
    permission_ok = render_permissions(st)
    # 새 작업은 현재 세대 유형(CSV→v2 표형, NPZ→배열)으로만 만든다. 저장된 작업은 기록의 유형으로 연다.
    specs = task_adapters.input_specs(generation="current")
    # 검색 근거 선택은 동의 문구를 바로 바꾸도록 양식 밖에 둔다. 기본 OFF. 지원 여부는 제출 뒤 서버가 다시 판정한다.
    locked = task_submit.locked(st)
    rag_mode = st.radio("검색 근거 (실험적 Graph RAG)", list(task_rag.MODES), format_func=RAG_CHOICES.get, horizontal=True,
                        key="task_rag_mode")

    if rag_mode == "on":
        _render_rag_availability(st, specs)

    with st.form("task_request_form"):
        # 등록된 입력 형식을 모두 안내한다. 실제 유형은 제출 뒤 선택 파일로 서버가 판별하고, 동의 내용도 그 유형으로 고정한다.
        label = "입력 파일 (" + " · ".join(spec["label"] for spec in specs) + ")"
        source = path_input(st, label, key="task_source", kind=specs[0]["picker"] if len(specs) == 1 else "data_file",
                            help=" / ".join(spec["help"] for spec in specs), browse_disabled=locked)
        goal = st.text_area("목표", key="task_goal", max_chars=4000,
                            placeholder="예: [예측 대상] 예측용으로 분할·전처리해 주세요 (아래 '목표 작성 안내' 참고)")
        _render_goal_guide(st)

        for spec in specs:
            adapter = task_adapters.adapter(spec["kind_id"])
            lines = adapter.consent_a_text(model["model"]) + task_rag.consent_text(
                rag_mode, model["model"], supported=task_rag.profile_of(adapter) is not None)
            st.info("작업 동의 범위(결정 A) — " + spec["label"] + "을(를) 고른 경우\n" + "\n".join("- " + line for line in lines) +
                    f"\n- 자원 상한: 메모리 {limits['memory_mib']} MiB · 프로세스 {limits['processes']}개 · 시간 {limits['wall_seconds']}초")
        agreed = st.checkbox("위 범위에 동의하고 작업을 시작합니다", key="task_consent_a")
        # 동의 문구는 이 화면의 검색 근거 선택으로 그렸다. 클릭 뒤 선택을 바꿔도 그린 값으로만 시작한다.
        options = _reserve_button(st, "start", None, bound=rag_mode)
        st.form_submit_button("작업 시작", type="primary", **dict(options, disabled=options["disabled"] or not permission_ok))

    # 작업이 아직 화면에 연결되지 않았으므로 작업 ID 없이 시작 동작으로 찾는다.
    task_submit.wait_if_running(st, None, "start")

    if not task_submit.reserved(st, "start", None):
        return

    _checked(st, agreed, "작업 동의 범위를 확인하고 체크해 주세요.")
    shown_rag = task_submit.bound(st)
    task_submit.execute(st, lambda progress: _start(progress, goal, source, limits, model, shown_rag),
                        "입력을 복사하고 격리 진단과 로컬 모델 권장안을 준비하고 있습니다.", _verify_stored(None, None))


def _start(progress, goal, source, limits, model, rag_mode):
    """잠긴 실행. 화면·세션 상태에 접근하지 않고 결과만 돌려준다. 만든 작업 ID는 진단 전에 progress에 남겨 결과를 받지
    못해도 화면에 연결하고 저장 상태로 대조한다."""
    try:
        task = orchestrator.start_task(goal, source, limits, model, input_generation="current", rag_mode=rag_mode)
        progress["link"] = task["task_id"]
    except task_store.TaskRecordError as error:
        return {"notice": [_record_error(error)]}
    except task_rag.RagUnsupported:
        return {"notice": [("error", "선택한 파일 유형은 검색 근거(실험적 Graph RAG)를 지원하지 않습니다(표형 CSV만 지원). 위에서 OFF를 "
                                     "고른 뒤 다시 시작하세요. 작업은 만들지 않았습니다.")]}
    except Exception:
        return {"notice": [("error", "작업을 시작하지 못했습니다. 지원하는 입력 파일 경로와 목표 문장(비어 있지 않게)을 확인하세요.")]}

    try:
        orchestrator.diagnose(task["task_id"])
    except task_store.TaskRecordError as error:
        return {"link": task["task_id"], "notice": [_record_error(error)]}
    except Exception:
        return {"link": task["task_id"], "notice": [("error", "진단을 시작하지 못했습니다. 작업 상태를 확인하세요. 자동 재시도하지 않습니다.")]}

    return {"link": task["task_id"]}


RAG_CHOICES = {"off": "OFF (기본)", "on": "ON (실험적, 표형 CSV만)"}
GOAL_NOTICE = ("알고 있는 정보만 적으면 됩니다. 아래 작성 안내의 항목은 모두 선택 사항이며, 모르는 항목은 비워 두세요. "
               "목표 문장은 적은 그대로 로컬 모델 권장안 요청에 전달됩니다. 모델이 컬럼 의미를 확인하는 것은 아니며, "
               "자세히 적으면 권장안이 달라질 수 있지만 더 정확해진다고 확인된 것은 아닙니다.")
GOAL_ITEMS = ("예측하거나 분석하려는 정확한 대상 컬럼명",
              "입력 특징에서 제외하거나 식별자로만 둘 컬럼명",
              "반복 측정 그룹(같은 대상이 여러 행에 나옴)이나 시간 순서를 고려해야 하는지",
              "평가 분할에 필요한 조건 (예: 같은 그룹은 한 구간에만 두기, 시간 순서대로 나누기)",
              "컬럼의 의미와 업무상 제약")
GOAL_EXAMPLE = ("승객 생존 여부를 예측하기 위한 전처리를 해 주세요. 예측 대상 컬럼은 [정확한 컬럼명]이고, [식별자 컬럼명]은 "
                "입력 특징에서 제외해 주세요. 데이터에 그룹이나 시간 순서가 있으면 분할에 반영해 주세요. "
                "모르는 항목은 추측하지 말고 확인 질문으로 남겨 주세요.")


def _render_goal_guide(st):
    """목표 입력 안내. 표시만 하며 입력란 값·전달 내용은 바꾸지 않는다. 예시는 자동으로 입력하거나 제출하지 않는다."""
    st.caption(GOAL_NOTICE)

    with st.expander("목표 작성 안내 (선택 사항)"):
        st.text("알고 있으면 목표에 적어 주세요.")

        for item in GOAL_ITEMS:
            st.text("- " + item)

        st.text("예시 (대괄호 부분을 내 데이터의 실제 이름으로 바꿔 쓰세요. 입력란에 자동으로 들어가지 않습니다):")
        st.text(GOAL_EXAMPLE)


def _render_rag_availability(st, specs):
    """요청 화면 안내. 지식집·색인을 임베딩 없이 검사한 결과만 보인다(검색·모델 호출 없음)."""
    profiles = {task_rag.profile_of(task_adapters.adapter(spec["kind_id"])) for spec in specs} - {None}

    for profile in profiles:
        ready, code = task_rag.availability(profile)

        if ready:
            st.text("검색 근거 지식집·색인: 검사 통과 (실험적 기능이며 권장 정확성 개선은 확인되지 않았습니다)")
        else:
            st.warning("검색 근거 지식집·색인을 지금 사용할 수 없습니다: " + task_rag.CODE_LABELS.get(code, "분류되지 않음")
                       + ". ON으로 시작해도 근거 없이 기존 방식으로 1회 요청하고 그 사유를 기록합니다.")


def _render_rag(st, draft):
    """저장된 검색 근거 사용 기록. 검색을 다시 하지 않으며 지식 문장은 표·일반 텍스트로만 보인다."""
    rag = task_rag.view(draft)

    if rag is None:
        return

    for line in rag["lines"]:
        st.text(line)

    if rag["items"]:
        with st.expander(f"생성 요청에 포함한 지식 {len(rag['items'])}개 (적용 조건·출처·검토 상태)"):
            _rows(st, rag["items"], task_rag.ITEM_COLUMNS)

    if rag["relations"]:
        with st.expander(f"생성 요청에 포함한 관계 {len(rag['relations'])}개"):
            _rows(st, rag["relations"], task_rag.RELATION_COLUMNS)


def _render_progress(st, task):
    st.subheader("진행 상황")
    st.text("현재: " + PHASE_LABELS[task["phase"]])
    diagnostic = (task["diagnostic"] or {}).get("status")
    lines = ["입력 진단: " + ("완료" if diagnostic == "completed" else "대기" if diagnostic is None else "진행 중")]
    selection = task.get("input_selection")

    if selection:
        lines.insert(0, "입력 해석: " + INPUT_STATUS.get(selection.get("status"), "확인 필요"))

    for step, label in task_adapters.adapter(task["kind"]).STEP_LABELS.items():
        status = task["steps"].get(step, {}).get("status")
        lines.append(label + ": " + {"completed": "완료", None: "대기"}.get(status, "진행 중"))

    st.text(" · ".join(lines))


def _render_decision(st, task):
    phase, problem = task["phase"], task.get("problem") or {}

    if phase in IN_PROGRESS:
        # 다른 재실행이 이 작업을 처리하는 중이면 중단·복구 안내 대신 이 자리에서 끝날 때까지 기다린다.
        task_submit.wait_if_running(st, task["task_id"])

    if phase == "awaiting_input":
        _render_input(st, task)
    elif phase == "awaiting_review" or orchestrator.revisable(task):
        _render_review(st, task)
    elif phase == "awaiting_apply":
        if st.session_state.get("task_revise_" + task["task_id"]) == task["attempt"] and orchestrator.apply_revisable(task):
            st.text("적용하지 않고 계획을 고치는 중입니다. 확인하면 새 동의(결정 C)로 새 시도를 진행합니다. 지금 후보와 사용 기록은 보존됩니다.")
            st.button("적용 확인으로 돌아가기", key="task_revise_cancel", on_click=_stop_revise, disabled=task_submit.locked(st),
                      args=(st, task["task_id"], task_submit.generation(st)))
            _render_review(st, task)
        else:
            _render_apply(st, task)
    elif phase == "needs_attention":
        st.subheader("필요한 결정")
        _render_problem(st, task, problem)
        _new_task_button(st, "task_new_after_problem")
    elif orchestrator.recoverable(task):
        st.subheader("필요한 결정 — 완료 기록 복구")
        st.warning("모든 처리 단계가 끝났지만 완료 기록을 저장하기 전에 멈췄습니다.")
        st.text("복구는 기존 결과·승인 사용 기록·결과 파일 해시·적용 대상 서명을 검증한 뒤 보고서를 다시 조립해 완료로 저장합니다. "
                "worker 재실행·새 승인·데이터 변경은 없습니다. 검증에 실패하면 기록을 바꾸지 않습니다.")
        st.button("검증 후 완료 기록 복구", key="task_recover_completion", type="primary",
                  **_reserve_button(st, "recover", task["task_id"]))

        if task_submit.reserved(st, "recover", task["task_id"]):
            labels = task_adapters.adapter(task["kind"]).STEP_LABELS
            task_submit.execute(st, lambda progress: _recover(task["task_id"], labels), "기존 결과와 승인 사용 기록을 검증하고 있습니다.",
                                _verify_stored(task["task_id"], task["revision"]))
    elif phase in IN_PROGRESS and task_submit.locked(st):
        # 예약이 남아 있으면 이번 재실행이 정리한다(finish). 중단으로 표시하지 않는다.
        st.info("처리 결과를 반영하고 있습니다. 잠시 후 화면이 갱신됩니다.")
    elif phase in IN_PROGRESS:
        st.warning("이전 진행이 끝나지 않은 채 중단되었습니다. 자동으로 다시 실행하지 않습니다. 결과 확인 후 새 작업으로 시작하세요.")
        _new_task_button(st, "task_new_after_interrupt")


def _recover(task_id, labels):
    try:
        orchestrator.recover_completion(task_id)
    except task_store.TaskRecordError as error:
        return {"notice": [_record_error(error)]}
    except orchestrator.RecoveryError as error:
        label = labels.get(error.step, "") if error.step else ""
        return {"notice": [("error", "검증을 통과하지 못해 복구하지 않았습니다. 기록은 그대로입니다. "
                                     + RECOVERY_ERRORS.get(error.code, "") + (" (단계: " + label + ")" if label else ""))]}
    except Exception:
        return {"notice": [("error", "복구 중 예상하지 못한 오류로 멈췄습니다. 기록은 그대로입니다. 자동 재시도하지 않습니다.")]}

    return {}


def _render_input(st, task):
    """입력 해석 대기. 검사 사유(고정 코드·개수)와 선택지를 보이고, 제출할 때만 새 결정 A로 검사·생성을 이어 간다."""
    st.subheader("필요한 결정 — 입력 해석")
    adapter = task_adapters.adapter(task["kind"])
    view = adapter.input_review(task)
    st.warning(view["message"])

    for key, value in view["detail"].items():
        st.text("확인된 사유: " + DETAIL_LABELS.get(key, "개수") + " " + str(value))

    # 사용자가 실제로 본 것은 클릭한 화면에 그린 대기 상태다. 그 revision·검사 ID를 버튼에 묶어 서버와 대조해 오래된 제출을 거부한다.
    shown = (task["revision"], view["inspection_id"])
    st.text("다음 행동: 아래에서 처리 방법을 고르거나, 파일의 컬럼명을 고친 뒤 새 작업으로 시작하세요.")

    with st.form("task_input_form"):
        selection = {}

        for number, (name, choice) in enumerate(view["choices"].items(), start=1):
            values = list(choice["options"])
            labels = [choice["options"][value] for value in values]
            current = values.index(choice["current"]) if choice["current"] in values else None
            picked = st.selectbox(choice["label"], labels, index=current, key=f"task_input_{number}", placeholder="처리 방법 선택")

            if picked is not None:
                selection[name] = values[labels.index(picked)]

        st.info(view["consent"])
        agreed = st.checkbox("위 범위에 동의합니다", key="task_input_consent")
        st.form_submit_button("고른 해석으로 진행", type="primary", **_reserve_button(st, "input", task["task_id"], bound=shown))

    _new_task_button(st, "task_new_after_input")
    task_submit.wait_if_running(st, task["task_id"], "input")

    if not task_submit.reserved(st, "input", task["task_id"]):
        return

    _checked(st, agreed, "동의 범위를 확인하고 체크해 주세요.")
    from datetime import datetime, timezone

    expires = (task["consents"].get("A") or {}).get("expires_at")

    if expires is None or datetime.fromisoformat(expires) <= datetime.now(timezone.utc):
        task_submit.decline(st, [("error", "결정 A 동의가 만료되어 진행하지 않았습니다. 작업은 그대로입니다. 다음 행동: 새 작업으로 다시 시작하세요.")])

    seen = task_submit.bound(st)
    task_submit.execute(st, lambda progress: _resume_input(task["task_id"], selection, seen),
                        "고른 해석으로 입력을 다시 검사하고 작업용 표·진단·로컬 모델 권장안을 준비하고 있습니다.",
                        _verify_stored(task["task_id"], task["revision"]))


def _resume_input(task_id, selection, seen):
    try:
        if seen is None:
            raise ValueError("확인한 대기 상태가 없습니다.")

        orchestrator.resume_input(task_id, selection, expected_revision=seen[0], expected_inspection_id=seen[1])
    except task_store.TaskRecordError as error:
        return {"notice": [_record_error(error)]}
    except ValueError:
        return {"notice": [("error", "제출을 받지 않았습니다. 새 동의·실행은 만들지 않았고 작업은 그대로입니다."),
                           ("text", "원인: 화면이 오래됐거나(다른 탭에서 이미 진행), 고른 해석이 지금과 같거나, 이 검사 사유로 고를 수 없는 "
                                    "해석입니다. 다음 행동: 화면을 새로 고쳐 현재 상태를 확인한 뒤 처리 방법을 다시 고르세요.")]}
    except Exception:
        return {"notice": [("error", "입력 준비를 시작하지 못했습니다. 작업 상태를 확인하세요. 자동 재시도하지 않습니다.")]}

    return {}


def _render_findings(st, adapter, problem):
    """확인된 지적 목록. 대상 이름(컬럼명 등)은 표로만 표시한다."""
    findings = problem.get("findings") or []

    if findings:
        labels = getattr(adapter, "FINDING_LABELS", {})
        st.text(f"확인된 지적 {len(findings)}건")
        _rows(st, [{"지적": labels.get(item["code"], item["code"]), "대상": item["subject"] or "전체"} for item in findings],
              ["지적", "대상"])


def _render_problem(st, task, problem):
    """확인된 분류와 추정 원인을 나눠 보여 준다. 메시지는 고정 문구, 관련 항목 이름은 일반 텍스트다."""
    st.error(problem.get("message", "작업이 중단되었습니다."))
    diagnostic = problem.get("diagnostic")

    if diagnostic:
        st.text("확인된 분류(worker 진단): " + DIAGNOSTIC_LABELS.get(diagnostic["code"], "분류 확인 불가")
                + " · 단계 " + str(diagnostic["stage"]))

    if problem.get("error"):
        # 원인 추적용 고정 정보(예외 종류·단계)만 접어서 보여 준다.
        with st.expander("실패 진단 정보"):
            st.json(problem["error"])

    for key, value in (problem.get("detail") or {}).items():
        label = DETAIL_LABELS.get(key) or getattr(task_adapters.adapter(task["kind"]), "FINDING_LABELS", {}).get(key, "개수")
        st.text("확인된 사유: " + label + " " + str(value))

    adapter = task_adapters.adapter(task["kind"])

    if _render_hint(st, adapter, problem):
        pass
    elif not problem.get("actions"):
        st.text("다음 행동: 원인을 확인한 뒤 새 작업으로 시작하세요.")

    for action in problem.get("actions") or []:
        _render_action(st, task, adapter, action)


def _render_hint(st, adapter, problem):
    """원인 안내. worker 세부 코드로 확인된 원인과 추정을 구분해 표시한다."""
    hint = problem.get("hint")
    text = adapter.HINTS.get((hint or {}).get("code"))

    if not text:
        return False

    st.warning("확인된 원인 (worker 세부 코드)" if text.get("confirmed") else "추정 원인 (확인된 사실이 아닙니다)")
    st.text(text["cause"])

    if hint["subjects"]:
        st.text("관련 항목: " + ", ".join(hint["subjects"]) + (" (로컬 진단 기준 후보)" if text.get("confirmed") else ""))

    st.text("다음 행동: " + text["next"])
    return True


def _render_action(st, task, adapter, action):
    """어댑터가 제시한 해결 동작을 동의 문구와 함께 보여 준다. 제출 전에는 아무것도 실행하지 않는다."""
    spec = adapter.ACTIONS.get(action)

    if spec is None:
        return

    st.text("다음 행동: " + spec["label"] + " 또는 파일을 고친 뒤 새 작업으로 시작할 수 있습니다.")

    with st.form("task_action_" + action):
        st.info("동의 범위(결정 A 갱신): " + spec["consent"])
        agreed = st.checkbox("위 범위에 동의합니다", key="task_action_consent_" + action)
        st.form_submit_button(spec["label"], type="primary", **_reserve_button(st, "action:" + action, task["task_id"]))

    task_submit.wait_if_running(st, task["task_id"], "action:" + action)

    if not task_submit.reserved(st, "action:" + action, task["task_id"]):
        return

    _checked(st, agreed, "동의 범위를 확인하고 체크해 주세요.")
    task_submit.execute(st, lambda progress: _resolve(task["task_id"], action),
                        "작업용 복사본을 격리 실행으로 정비하고 진단과 로컬 모델 권장안을 준비하고 있습니다.",
                        _verify_stored(task["task_id"], task["revision"]))


def _resolve(task_id, action):
    try:
        orchestrator.resolve_problem(task_id, action)
    except task_store.TaskRecordError as error:
        return {"notice": [_record_error(error)]}
    except Exception:
        return {"notice": [("error", "해결 동작을 시작하지 못했습니다. 작업 상태를 확인하세요. 자동 재시도하지 않습니다.")]}

    return {}


def _numbered(questions):
    """화면의 질문 번호(필수 → 안내 순). 질문 위젯과 거부 안내가 같은 번호를 쓴다."""
    return list(enumerate(sorted(questions, key=lambda value: value["level"] != "required"), start=1))


def _question_widgets(st, questions, key):
    """질문을 필수 → 안내 순으로 그린다. 답한 질문만 돌려준다. 권장 선택지도 사용자가 고르기 전에는 반영하지 않는다."""
    answers = {}

    for number, item in _numbered(questions):
        # 출처를 함께 보인다. 모델 질문의 문장·수치는 확인된 사실이 아니다.
        source = "(모델 질문 · 확인된 사실 아님) " if item["source"] == "model" else "(진단 계산 기준 질문) "
        st.text(f"질문 {number}. " + ("[필수] " if item["level"] == "required" else "[안내] ") + source + item["text"])

        if item["subjects"]:
            st.text("관련 항목: " + ", ".join(item["subjects"]))

        if not item["options"]:
            continue

        labels = [NO_ANSWER] + [option["label"] + (" (권장)" if option["value"] == item["recommended"] else "")
                                for option in item["options"]]
        # 위젯 이름은 마크다운으로 해석되므로 질문 ID(컬럼 이름 포함 가능) 대신 고정 번호만 쓴다.
        chosen = st.selectbox(f"질문 {number} 답변", labels, key=key + "_q_" + item["id"])

        if chosen != NO_ANSWER:
            answers[item["id"]] = item["options"][labels.index(chosen) - 1]["value"]

    return answers


def _render_unresolved(st, task, state, draft):
    """제출한 계획에서 남은 필수 질문의 문장·관련 항목·해결 방법. 질문 ID를 설정 경로로 해석하지 않고 값을 바꾸지 않는다."""
    try:
        unresolved = orchestrator.unresolved_questions(task, draft["plan"], draft["answers"])
    except (PlanError, ValueError):
        unresolved = []

    if not unresolved:
        st.text("남은 질문을 다시 계산하지 못했습니다. 위 질문 답변과 계획 양식을 확인한 뒤 다시 미리보기하세요.")
        return

    numbers = {item["id"]: number for number, item in _numbered(state["questions"])}

    for item in unresolved:
        # 질문 문장·관련 항목은 입력 유래 이름을 포함할 수 있어 일반 텍스트로만 보인다.
        where = f"위 질문 {numbers[item['id']]}" if item["shown"] and item["id"] in numbers else "제출한 계획에서 새로 생긴 질문"
        st.text(f"남은 필수 질문({where}): {item['text']}")
        related = ([item["path"]] if item.get("path") else []) + list(item.get("subjects") or [])
        st.text("관련 항목: " + (", ".join(str(value) for value in related) or "전체 계획"))

        if item["shown"]:
            st.text("해결: 위 질문에 답하거나, 계획 양식에서 관련 항목을 고친 뒤 다시 미리보기하세요.")
        else:
            st.text("해결: 이 질문은 위 질문 목록에 없습니다. 계획 양식에서 관련 항목(역할 표·분할 방식 등)을 서로 맞게 고친 뒤 "
                    "다시 미리보기하세요. 입력한 값은 그대로 두었고 자동으로 바꾸지 않았습니다.")


def _render_rejected_items(st, error, questions, plan, answers):
    """거부된 계획 항목을 그 값을 정한 질문 번호·양식 값과 함께 보여 준다. 답변·계획은 바꾸지 않는다."""
    if not error.subjects:
        return

    numbered = {item["path"]: (number, item) for number, item in _numbered(questions)
                if item["path"] is not None and item["id"] in answers}
    lines = []

    for subject in error.subjects:
        if subject in numbered:
            number, item = numbered[subject]
            lines.append(f"질문 {number} 답변: {subject} = {answers[item['id']]}")
        elif "." in subject and subject.split(".", 1)[0] in plan and isinstance(plan[subject.split(".", 1)[0]], dict):
            section, name = subject.split(".", 1)
            lines.append(f"계획 양식: {subject} = {plan[section].get(name)}")
        else:
            lines.append("관련 항목: " + subject)

    for line in lines:
        st.text(line)

    if any(line.startswith(("질문 ", "계획 양식")) for line in lines):
        st.text("함께 쓸 수 없는 값입니다. 위 질문 답변이나 계획 양식에서 한쪽을 골라 다시 확인하세요. 다른 답변과 계획은 바꾸지 않았습니다.")


def _render_review(st, task):
    st.subheader("필요한 결정 — 계획 확인")
    adapter = task_adapters.adapter(task["kind"])
    problem = task.get("problem")

    draft = task["draft"]
    state = orchestrator.review_state(task)

    if problem:
        # 이전 형식 기록이면 검증된 결과로 다시 읽은 원인·지적·안내를 쓴다. 양식의 일괄 제외 선택도 같은 지적을 쓴다.
        problem = state["problem"]
        task = dict(task, problem=problem)
        st.warning(problem["message"])
        _render_findings(st, adapter, problem)

        if not problem.get("retryable"):
            st.text("이전 시도는 처리 단계에서 멈췄습니다. 완료된 단계와 사용한 승인 기록은 보존되며, 계획을 고쳐 확인하면 새 동의로 "
                    "새 시도를 진행합니다(같은 계획 자동 재실행 없음).")
            _render_hint(st, adapter, problem)
    # 작업 기록 값은 마크다운으로 해석하지 않는다. 사유는 고정 코드의 설명만 보여 준다.
    notice = recommendation_notice(task)

    if draft["source"] == "model":
        st.text("권장안: 로컬 LLM 권장안 (형식·허용값 검사 통과. 내용의 타당성은 확인되지 않았습니다)")
    elif "rejection" in draft or len(notice) > 1:
        # 계획 검증으로 권장안 전체를 거부한 기록: 전처리 실패나 권장안 성공이 아니며 확인된 하위 원인만 고정 문구로 보인다.
        st.text("권장안: 기본 계획 — " + notice[0])

        for line in notice[1:]:
            st.text(line)
    else:
        st.text("권장안: 기본 계획 (로컬 LLM 권장안을 사용할 수 없었습니다: "
                + (code_label(task, draft.get("code")) or "사유 코드 확인 불가") + ")")
    st.text(MODEL_LIMITATION_NOTICE)
    _render_rag(st, draft)

    coverage = draft.get("coverage")

    if coverage and coverage["total"]:
        st.text(f"모델 검토 범위: 판단 대상 {coverage['total']}개 중 상세 집계 {coverage['detailed']}개 · 이름만 {coverage['named']}개 · "
                f"요약만 {coverage['summarized']}개")

        if coverage["summarized"]:
            st.text("요약만 제공한 대상은 모델이 역할을 권하지 않았습니다(기본 계획 유지). 아래 계획 양식에서 직접 바꿀 수 있습니다.")

            with st.expander("요약만 제공한 대상 목록"):
                _rows(st, [{"대상": name} for name in coverage["summarized_ids"]])

    for text in draft.get("summary", []):
        st.text("- 모델 요약(확인된 사실 아님): " + text)

    for text in draft.get("notes", []):
        st.text("- 모델 메모(확인된 사실 아님): " + text)

    _render_diagnosis(st, task, adapter)

    if hasattr(adapter, "preview_images"):
        _render_previews(st, task, adapter)

    _render_recommended(st, state)
    _render_guide(st, state.get("guide") or [], "옵션 설명 (현재 표시한 계획 기준)")
    key = "task_review_" + task["task_id"][:8] + str(task["attempt"]) + ("r" if task["phase"] == "awaiting_apply" else "")

    with st.form(key):
        if state["questions"]:
            st.markdown("#### 질문")
            answers = _question_widgets(st, state["questions"], key)
            st.caption("질문 답변은 해당 항목 하나만 바꿉니다. 아래 양식에서 같은 항목을 다른 값으로 바꾸면 충돌로 표시하고 어느 쪽을 쓸지 묻습니다.")
        else:
            answers = {}

        st.markdown("#### 계획")
        plan = form_for(task["kind"])(st, task, state["plan"], state["fields"], key)
        submitted = st.form_submit_button("최종 계획 미리보기", type="primary", disabled=task_submit.locked(st))

    if submitted:
        # 미리보기 대상은 제출한 양식·답변 그대로다. 확정은 아래 동의 단계에서 이 값으로만 한다.
        st.session_state[key + "_draft"] = {"plan": plan, "answers": answers}

    draft = st.session_state.get(key + "_draft")

    if draft is None:
        st.text("양식과 질문 답변을 정한 뒤 '최종 계획 미리보기'를 누르세요. 확정 전에 답변을 반영한 최종 계획과 경고를 보여 줍니다.")
        return

    _render_final_plan(st, task, adapter, state, key, draft)


def _render_previews(st, task, adapter):
    """입력 확인용 미리보기. 표시할 때마다 결과 검증을 거친 이미지만 연다. 모델에는 보내지 않는다."""
    with st.expander("입력 미리보기 (로컬 전용, 처리 결과 아님)"):
        try:
            images = adapter.preview_images(task)
        except Exception:
            images = []
            st.text("미리보기를 결과 검증으로 다시 확인하지 못해 표시하지 않습니다.")

        if not images:
            st.text("(표시할 미리보기 없음: 표본 형상이 이미지 형태가 아니거나 표본 배열이 정해지지 않았습니다)")

        for image in images:
            # 캡션은 고정 문구와 숫자만 담는다. 입력 유래 문자열은 넣지 않는다.
            st.image(image["data"], caption=image["label"])


def _render_diagnosis(st, task, adapter):
    """진단 안내 줄과 진단 요약 표. 열 순서는 유형 어댑터가 정한 columns를 쓰고, 없으면 기본 순서다."""
    view = adapter.diagnosis_view(task)

    for line in adapter.input_notices(task) + view["lines"]:
        st.text(line)

    with st.expander("진단 요약 표"):
        _rows(st, view["rows"], view.get("columns"))


def _render_recommended(st, state):
    """모델 권장 변경(기본 계획 대비). 근거는 모델 의견이며 확인 열은 계산으로 대조한 결과다."""
    if not state["recommended"]:
        return

    st.markdown("#### 권장 변경 (기본 계획 대비)")
    before = {item["path"]: item["before"] for item in state["changes"]}
    _rows(st, [{"항목": item["path"], "기본값": before.get(item["path"]), "권장값": item["value"], "근거": "모델 의견: " + item["reason"],
                "확인": item.get("check") or "-"} for item in state["recommended"]], ["항목", "기본값", "권장값", "근거", "확인"])


def _render_guide(st, guide, title):
    """옵션 설명. 계산으로 확인한 사실·모델 의견·사전 경고를 나눠 보여 준다. 모델 문장은 일반 텍스트로만 표시한다."""
    if not guide:
        return

    warnings = [(item["label"], text) for item in guide for text in item["warnings"]]

    # 경고 문장에는 컬럼명이 들어가므로 강조 상자에는 고정 문구만 두고 내용은 일반 텍스트로 보인다.
    if warnings:
        st.warning(f"사전 경고 {len(warnings)}건 (진단 집계 기준, 설정은 바꾸지 않았습니다)")

        for label, text in warnings:
            st.text("- " + label + ": " + text)

    with st.expander(title):
        _rows(st, [{"옵션": item["label"], "설정 이름": item["name"], "현재 값": item["value_text"], "출처": item["source"],
                    "적용 대상": item["applies"]} for item in guide if item["basic"] or item["source"] != "기본 계획"],
              ["옵션", "설정 이름", "현재 값", "출처", "적용 대상"])

        for item in guide:
            st.text("■ " + item["label"])
            st.text("설정: " + item["name"] + " = " + str(item["value"]) + " (" + item["value_text"] + ") · 출처: " + item["source"])
            st.text("처리 내용: " + item["what"])
            st.text("적용 대상: " + item["applies"])

            for fact in item["facts"]:
                st.text("계산으로 확인한 사실(진단 집계): " + fact)

            if item["opinion"]:
                st.text("모델 의견(확인된 사실 아님): " + item["opinion"])

            st.text("주의: " + item["caution"])
            st.text("사용하지 않으면: " + item["off"])


def _render_final_plan(st, task, adapter, state, key, draft):
    """답변을 반영한 최종 계획·충돌·검증 오류를 확정 전에 보여 준다. 충돌은 사용자가 고를 때까지 확정할 수 없다."""
    st.markdown("#### 최종 적용 계획 (확정 전 확인)")

    try:
        preview = orchestrator.preview_plan(task, draft["plan"], draft["answers"])
    except PlanError as error:
        if error.code == "ANSWER_FORM_CONFLICT":
            _render_conflicts(st, task, state, key, draft)
            return

        st.error(error.message + " 작업은 만들지 않았습니다.")

        if error.code == "REQUIRED_QUESTION_UNANSWERED":
            _render_unresolved(st, task, state, draft)
        else:
            _render_rejected_items(st, error, state["questions"], draft["plan"], draft["answers"])
        return

    labels = {"base": "기본 계획", "model": "모델 권장", "user": "사용자 변경", "answer": "질문 답변", "previous": "직전 확정 계획"}
    origins = preview["origins"]
    changed = [{"항목": path, "값": value, "출처": labels.get(origins[path]["source"], origins[path]["source"]),
                "모델 의견": origins[path].get("reason") or ""}
               for path, value in flatten(preview["plan"]) if origins[path]["source"] != "base"]
    st.text("기본 계획과 다른 항목 (출처별). 표에 없는 항목은 기본 계획 값입니다.")
    _rows(st, changed, ["항목", "값", "출처", "모델 의견"])

    if preview["changes"]:
        st.text("이번에 고친 항목 (표시한 계획 → 최종 계획)")
        _rows(st, [{"항목": item["path"], "표시한 값": item["before"], "최종 값": item["after"]} for item in preview["changes"]],
              ["항목", "표시한 값", "최종 값"])

    _render_guide(st, preview["guide"], "옵션 설명 (최종 계획 기준)")

    with st.expander("실행에 쓰일 전체 설정 (최종 계획, 이 값 그대로 확정·실행합니다)"):
        _rows(st, [{"항목": path, "값": value, "출처": labels.get(origins[path]["source"], origins[path]["source"])}
                   for path, value in flatten(preview["plan"])])

    if preview.get("blocked"):
        # 검증은 통과했지만 이번 단계에서 실행하지 않는 계획이다. 동의·확정 버튼을 내지 않고 이유와 다음 행동만 보인다(서버도 거부).
        st.warning(preview["blocked"]["message"])
        return

    st.info(adapter.consent_c_text())
    consent_key = "task_consent_c_" + key
    agreed = st.checkbox("위 최종 계획과 진행 범위에 동의합니다", key=consent_key)
    # 확정은 클릭한 화면에서 사용자가 본 최종 계획의 결합값으로만 한다(다른 탭에서 작업이 바뀌면 서버가 거부).
    st.button("확인하고 진행", key=key + "_confirm", type="primary",
              **_reserve_button(st, "confirm", task["task_id"], bound=preview["digest"]))

    task_submit.wait_if_running(st, task["task_id"], "confirm")

    if not task_submit.reserved(st, "confirm", task["task_id"]):
        return

    _checked(st, agreed, "최종 계획과 진행 범위를 확인하고 체크해 주세요.")
    seen = task_submit.bound(st)
    task_submit.execute(st, lambda progress: _confirm(task["task_id"], draft, seen or "-", [key + "_draft", "task_revise_" + task["task_id"]]),
                        "확인한 계획으로 처리 단계를 격리 실행하고 있습니다.", _verify_stored(task["task_id"], task["revision"]))


def _confirm(task_id, draft, digest, done_keys):
    try:
        orchestrator.confirm_plan(task_id, draft["plan"], draft["answers"], expected_digest=digest)
    except task_store.TaskRecordError as error:
        return {"notice": [_record_error(error)]}
    except PlanError as error:
        return {"notice": [("error", error.message + " 작업은 만들지 않았습니다.")]}
    except ValueError:
        return {"notice": [("error", "계획 검증을 통과하지 못했습니다. 작업은 만들지 않았습니다.")]}
    except Exception:
        return {"notice": [("error", "처리를 시작하지 못했습니다. 작업 상태를 확인하세요. 자동 재시도하지 않습니다.")]}

    # 확정한 미리보기·계획 수정 표시는 성공했을 때만 지운다. 실패하면 같은 화면에서 원인을 보고 다시 고칠 수 있다.
    return {"pop": done_keys}


def _render_conflicts(st, task, state, key, draft):
    """양식 값과 답변이 다른 항목마다 사용자가 한쪽을 고르게 한다. 고르기 전에는 어느 값도 쓰지 않는다."""
    from src.agents.task_plan import answer_conflicts

    conflicts = answer_conflicts(orchestrator.review_state(task)["plan"], draft["plan"], state["questions"], draft["answers"])
    st.error("계획 양식에서 바꾼 값과 질문 답변이 같은 항목에 서로 다른 값을 정했습니다. 자동으로 고르지 않습니다. 항목마다 쓸 값을 고르세요.")
    numbers = {item["id"]: number for number, item in _numbered(state["questions"])}
    choices = {}

    for index, item in enumerate(conflicts, start=1):
        st.text(f"충돌 {index}: {item['path']} — 양식 값 {item['form']} / 질문 {numbers.get(item['id'], '?')} 답변 {item['answer']}")
        choices[item["id"]] = st.radio(f"충돌 {index}에 쓸 값", ("양식 값", "질문 답변"), index=None, key=key + f"_conflict_{index}")

    if not st.button("고른 값으로 다시 미리보기", key=key + "_resolve", disabled=task_submit.locked(st)):
        return

    if any(choice is None for choice in choices.values()):
        st.warning("모든 충돌 항목에서 쓸 값을 고르세요.")
        return

    plan, answers = draft["plan"], dict(draft["answers"])

    for item in conflicts:
        if choices[item["id"]] == "양식 값":
            # 양식 값을 쓰면 그 답변을 빼고 다시 검증한다. 필수 질문이면 최종 계획에서 조건이 사라져야 확정할 수 있다.
            answers.pop(item["id"], None)
        else:
            section, name = item["path"].split(".", 1)
            plan = {**plan, section: {**plan[section], name: item["answer"]}}

    st.session_state[key + "_draft"] = {"plan": plan, "answers": answers}
    st.text("고른 값으로 미리보기를 다시 계산합니다. 위 양식을 다시 제출하면 양식 값 기준으로 새로 계산합니다.")
    st.rerun()


def _render_apply(st, task):
    st.subheader("필요한 결정 — 적용")
    adapter = task_adapters.adapter(task["kind"])
    st.text(adapter.consent_d_text())
    candidate = task["candidate"]
    # 사용자가 실제로 본 것은 클릭한 화면에 그린 후보다. 그 서명·시도 번호를 버튼에 묶어 서버와 대조한다.
    shown = (candidate["signature"], task["attempt"])
    facts = candidate.get("facts")

    if facts is None:
        # 2026-10-21 이전 기록은 원문 보고서만 있다.
        st.text(candidate.get("report") or "적용 대상 보고서를 표시할 수 없습니다.")
    else:
        _rows(st, [{"항목": item["label"], "값": item["value"]} for item in facts["metrics"]])

        # 처리 영향 경고는 사실만 보여 주고 설정을 바꾸지 않는다. 고치려면 사용자가 '계획 수정'을 고른다.
        if facts.get("warnings"):
            st.warning(f"처리 영향 경고 {len(facts['warnings'])}건 (Train 후보 계산 기준)")

        for text in facts.get("warnings", []):
            st.text("- " + text)

        for text in facts.get("notes", []):
            st.caption(text)

        for table in facts["tables"]:
            st.markdown("#### " + table["title"])
            _rows(st, table["rows"], table.get("columns"))

            # 표 단위 설명(있는 유형·후보만)은 해당 표 바로 아래에 둔다. 이전 facts에는 없다.
            for text in table.get("notes", []):
                st.caption(text)

        with st.expander("원문 보고서 (일반 텍스트)"):
            st.text(facts["text"] or "원문 보고서를 표시할 수 없습니다.")

            if facts["text_truncated"]:
                st.warning("보고서가 길어 앞부분만 표시했습니다. 전체는 기술 상세의 해당 작업 폴더에서 확인하세요.")

    if orchestrator.apply_revisable(task):
        st.button("적용하지 않고 계획 수정", key="task_revise_plan", on_click=_start_revise, disabled=task_submit.locked(st),
                  args=(st, task["task_id"], task["attempt"], task_submit.generation(st)),
                  help="역할·전처리 방법을 고쳐 새 시도로 다시 계산합니다. 새 동의(결정 C)가 필요합니다.")

    with st.form("task_apply_form"):
        agreed = st.checkbox("이 적용 대상으로 적용과 결과 전달에 동의합니다 (결정 D)", key="task_consent_d")
        st.form_submit_button("적용하고 마무리", type="primary", **_reserve_button(st, "apply", task["task_id"], bound=shown))

    task_submit.wait_if_running(st, task["task_id"], "apply")

    if not task_submit.reserved(st, "apply", task["task_id"]):
        return

    _checked(st, agreed, "적용 대상을 확인하고 체크해 주세요.")
    seen = task_submit.bound(st)
    task_submit.execute(st, lambda progress: _apply(task["task_id"], seen), "적용하고 결과 전달 묶음과 결과 보고서를 만들고 있습니다.",
                        _verify_stored(task["task_id"], task["revision"]))


def _apply(task_id, seen):
    try:
        if seen is None:
            raise ValueError("확인한 적용 대상이 없습니다.")

        orchestrator.approve_apply(task_id, expected_signature=seen[0], expected_attempt=seen[1])
    except task_store.TaskRecordError as error:
        return {"notice": [_record_error(error)]}
    except Exception:
        # 오래된 화면이거나, 후보 계산 뒤 처리 코드가 바뀌어 후보를 현재 코드로 다시 열 수 없는 경우가 대표적이다.
        return {"notice": [("error", "적용을 시작하지 못했습니다. 동의(결정 D)는 발급하지 않았고 작업은 그대로입니다. 자동 재시도하지 않습니다."),
                           ("text", "다음 행동: 화면을 새로 고쳐 적용 대상을 다시 확인하세요. 그래도 같으면(후보 계산 뒤 처리 코드가 바뀐 경우 "
                                    "등) '적용하지 않고 계획 수정'으로 새 시도를 진행하세요. 새 동의(결정 C)로 후보를 다시 계산합니다.")]}

    return {}


def _rebuild(st, task_id, seen):
    from src.agents.task_report import rebuild_report

    if not task_submit.idle(st, seen):
        return

    try:
        rebuild_report(task_id)
    except Exception:
        st.session_state["task_rebuild_failed"] = task_id


def _render_result(st, task):
    from src.agents.task_report import verified_images

    st.subheader("최종 결과")
    report = task["report"]
    facts = report.get("facts")

    if task.get("recovered"):
        st.info("완료 기록 복구됨: 처리 단계는 모두 끝났지만 완료 저장이 멈췄던 작업입니다. 기존 결과·승인 사용 기록을 검증해 "
                "완료로 저장했습니다(worker 재실행·새 승인 없음).")
        labels = task_adapters.adapter(task["kind"]).STEP_LABELS
        read_only = task["recovered"].get("read_only_steps") or []

        if read_only:
            st.text("실행 뒤 코드가 바뀐 단계(" + ", ".join(labels.get(step, step) for step in read_only) + ")는 실행 당시 코드 지문이 "
                    "동의·승인 증거와 같은지 확인한 뒤 현재 검증기로 읽기 전용 검증했습니다.")

    if st.session_state.get("task_rebuild_failed") == task["task_id"]:
        st.error("보고서를 다시 조립하지 못했습니다. 결과 검증을 통과하지 못한 단계가 있을 수 있습니다. 기존 보고서는 그대로 둡니다.")

    if facts is None or "metrics" not in facts:
        st.info("이전 형식의 보고서입니다. 검증된 결과 파일로 다시 조립할 수 있습니다(재계산·재실행·모델 호출 없음).")
        st.button("보고서 다시 조립", key="task_rebuild_report", on_click=_rebuild, disabled=task_submit.locked(st),
                  args=(st, task["task_id"], task_submit.generation(st)))
        st.text(report["text"])
        _new_task_button(st, "task_new_after_done")
        return

    st.text("목표: " + facts["goal"])
    recommendation = facts["recommendation"]
    st.text("처리: 완료 · 로컬 LLM 권장안: " + ("사용함" if recommendation["source"] == "model" else
                                         "사용 못 함(" + (code_label(task, recommendation["code"]) or "사유 코드 확인 불가") + ") — 기본 계획에서 시작"))
    st.text(MODEL_LIMITATION_NOTICE)
    _render_rag(st, task["draft"])
    # 2026-10-22 이전에 저장한 보고서 사실에는 경고 절이 없다. 새 보고서는 공통 조립에서 항상 경고 목록을 둔다.
    legacy = "warnings" not in facts

    if legacy:
        st.info("이전 형식의 보고서입니다(예상·실측 구분과 경고 절 없음). 검증된 결과 파일로 다시 조립할 수 있습니다(재계산·재실행·모델 호출 없음).")
        st.button("보고서 다시 조립", key="task_rebuild_report", on_click=_rebuild, disabled=task_submit.locked(st),
                  args=(st, task["task_id"], task_submit.generation(st)))

    st.markdown("#### 핵심 결과와 주요 문제")

    for line in facts.get("highlights", []) + facts.get("lines", []):
        st.text(line)

    _rows(st, [{"항목": item["label"], "값": item["value"]} for item in facts["metrics"]])

    with st.expander("주요 문제 (진단·품질 검사)", expanded=legacy):
        _rows(st, facts["problems"])

    # 처리 전 EDA·계획과 결정 절은 이를 주는 유형만 채운다. 저장된 문구는 Markdown으로 해석하지 않는다.
    for key, heading, note in (("eda_tables", "#### 처리 전 데이터와 기본 EDA", facts.get("eda_note")),
                               ("decision_tables", "#### 계획과 결정", None)):
        if facts.get(key):
            st.markdown(heading)

            if note:
                st.text(note)

            for table in facts[key]:
                st.text("■ " + table["title"])

                for text in table["notes"]:
                    st.text("- " + text)

                _rows(st, table["rows"], table.get("columns"))

    # 분할 표는 이를 주는 유형만 채운다. 이전 형식·다른 유형 보고서에는 없다.
    if facts.get("split_table"):
        st.markdown("#### 분할 결과")
        # 저장된 사실 문자열은 Markdown으로 해석하지 않는다.
        st.text(facts["split_table"]["title"])
        _rows(st, facts["split_table"]["rows"])

    st.markdown("#### 컬럼별 처리와 제외 사유")
    _rows(st, facts["processing"])
    st.markdown("#### 실제 전후 비교")

    if facts.get("comparison_note"):
        st.caption(facts["comparison_note"])

    _rows(st, facts["comparison"])
    st.caption("검증된 결과 파일에 없는 수치는 계산하지 않고 '미집계' 또는 '미산출(사유)'로 표시합니다.")
    st.markdown("#### 그래프")

    # 보고서 그래프는 이를 주는 유형만 채운다. 저장 보고서의 값 표와 같은 사실로 그린다.
    for chart in facts.get("charts") or []:
        _chart(st, chart)

    if facts.get("charts"):
        st.text("■ 분할 EDA 그래프 (검증된 PNG)")

    # 저장된 경로를 다시 열지 않고, 표시할 때마다 기록 검사·결과 검증을 거친 그래프만 보여 준다.
    try:
        images = verified_images(task)
    except Exception:
        images = []
        st.text("그래프를 결과 검증으로 다시 확인하지 못해 표시하지 않습니다.")

    notes = facts.get("image_notes") or {}

    for image in images:
        st.image(image["data"], caption=image["label"])

        # 설명은 보고서 사실에 저장된 문구이므로 Markdown으로 해석하지 않는 글자로 보인다. 설명이 없으면 이름만 보인다.
        if image["label"] in notes:
            st.text(notes[image["label"]])

    if not images:
        st.text("(검증된 그래프 없음)")

    if not legacy:
        st.markdown("#### 경고: 정보 손실·미처리·미검증")

        for kind in ("정보 손실", "미처리", "미검증"):
            items = [item["내용"] for item in facts["warnings"] if item["구분"] == kind]
            st.text(kind + " " + str(len(items)) + "건")

            for text in items:
                st.text("- " + text)

    st.markdown("#### 참고 사항" if not legacy else "#### 주의사항")

    for text in facts["cautions"]:
        st.text("- " + text)

    st.markdown("#### 적용 구간과 결과 파일 활용")

    for text in facts.get("usage", []):
        st.text("- " + text)

    _rows(st, facts["files"])

    with st.expander("결과 기술 상세 (경로·해시·내부 작업)"):
        st.json(facts["technical"])
        st.text("보고서 파일: " + report["path"])

    _new_task_button(st, "task_new_after_done")

"""1번 데이터 선택 아래의 역할 지정 전 입력 진단과 프로파일 작성. 렌더링만으로 파일 쓰기·worker 실행이 없다.

진단 실행은 기존 승인 흐름(render_approval_flow)을 그대로 쓰고, 결과는 화면 표시와 사용자 역할 지정에만 쓴다.
역할을 추천·자동 확정하지 않으며 진단 결과를 LLM 대화·해설로 보내지 않는다.
"""

import pandas as pd

from src.agents.profile_store import EXCLUDED, ROLE_LABELS, build_profile, save_profile
from .approval_flow_view import render_approval_flow
from .profiling_service import (DIAGNOSTIC_NOTICE, UiInputProfiling, blocked_reason, create_profiling,
                                effective_snapshot, link_record, load_profile_result, missing_over,
                                register_derived_input)
from .source_selection import CSV_SNAPSHOT, CSV_SOURCE, selected_source


SESSION_KEYS = ("ui_input_profiling", "ui_input_profile_result", "ui_profile_roles", "ui_profile_preview", "ui_profile_saved",
                "ui_input_profile_blocked", "ui_input_profile_register_failed", "ui_profile_bulk_notice")
# 일괄 제외는 특징 역할만 바꾼다. 사용자가 지정한 target·식별자·group·timestamp는 유지한다.
FEATURE_LABELS = ("수치형", "범주형")
EXCLUDE_NOTICE = "분석 대상에서 제외하며 데이터 파일은 변경하지 않습니다."
DEDUPE_OPTION = "ui_profile_dedupe_option"
NEXT_INPUT = ("‘다른 CSV 선택하기’를 누르면 위 입력 데이터 칸이 비워집니다. 새 CSV를 선택하면 진단을 다시 준비할 수 있습니다. "
              "차단된 작업 기록과 소비된 승인은 보존하며 재사용하지 않습니다.")
RETRY_NOTICE = ("‘같은 입력으로 새 진단 작업 준비’를 누르면 새 작업과 새 승인으로 다시 진행할 수 있습니다. "
                "자동 재시도하지 않으며 이전 작업 기록과 소비된 승인은 보존합니다.")
# 입력 선택 해제 시 비우는 1번 화면 경로 입력 키. 원본·snapshot 파일은 지우지 않는다.
SOURCE_INPUT_KEYS = tuple(prefix + suffix for prefix in ("source_csv", "source_snapshot")
                          for suffix in ("_path", "_profile"))
SOURCE_INPUT_KEYS = SOURCE_INPUT_KEYS + tuple("_widget_" + key for key in SOURCE_INPUT_KEYS)
BLOCKED_MESSAGES = {
    "DUPLICATE_COLUMN_NAMES": "중복된 컬럼명이 있어 입력 진단을 차단했습니다. 역할을 컬럼 하나에 하나씩 대응할 수 없어 "
                              "기존 입력 검사 기준으로 거부한 것입니다. 사용자가 허용하지 않으면 컬럼명을 바꾸지 않습니다. "
                              "‘중복 컬럼명 자동 정비로 다시 준비’를 누르면 복사본에서만 중복 이름에 번호를 붙인 새 파생 입력을 만들 수 있습니다.",
    "EMPTY_COLUMN_NAME": "비어 있는 컬럼명이 있어 입력 진단을 차단했습니다. 컬럼명을 자동으로 채우지 않습니다. "
                         "모든 컬럼에 이름이 있는 CSV를 준비해 주세요.",
    "HEADER_UNREADABLE": "CSV 헤더를 읽지 못해 입력 진단을 차단했습니다. 인코딩·구분자를 확인하거나 헤더가 너무 길지 않은지(1 MiB 문자 이하) 확인해 주세요.",
    "INPUT_BLOCKED_OTHER": "CSV 형식 또는 입력 검사에서 차단했습니다. 헤더 중복·빈 이름은 아닙니다. 데이터 행·형식을 확인해 주세요.",
    "REASON_UNVERIFIED": "입력 복사본 재검증에 실패해 차단 사유를 판별하지 않았습니다. 선택한 snapshot이 승인 이후 바뀌었을 수 있습니다. "
                         "중복 여부를 단정하지 않으며 이 작업의 결과를 사용하지 않습니다.",
}
RECOVERY = {"blocked_input": "choose_other", "failed_runtime": "restart", "security_locked": "restart", "cancelled": "restart"}


def recovery_action(status):
    """종료된 진단 상태에만 다음 행동을 연결한다. 실행 중·불명 상태에는 버튼을 두지 않는다."""
    return RECOVERY.get(status)
PURPOSE_LABELS = {"EDA 검토용 (eda_review_v1)": "eda", "분할 검토용 (split_review_v1)": "split"}
SHOWN = ("column", "dtype_group", "missing_ratio", "blank_string_count", "observed_unique", "finite_count",
         "pos_inf_count", "neg_inf_count", "outlier_candidate_count")


def _base_directory(st, source):
    if source["mode"] == CSV_SNAPSHOT:
        return source["path"]

    copied = st.session_state.get("ui_source_snapshot")
    return copied["directory"] if copied is not None and copied["selection_id"] == source["selection_id"] else None


def _snapshot_directory(st, source):
    """실제 진단·분석에 쓸 snapshot. 같은 선택에서 등록한 파생 입력이 있으면 그것을 쓴다."""
    base = _base_directory(st, source)
    return None if base is None else effective_snapshot(st.session_state, source["selection_id"], base)


def _derived(st, source):
    derived = st.session_state.get("ui_derived_input")
    base = _base_directory(st, source)

    if isinstance(derived, dict) and derived.get("source_selection_id") == source["selection_id"] \
            and derived.get("original_directory") == base:
        return derived

    return None


def _discard(st):
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def _choose_other_input(st):
    """차단된 입력의 선택만 세션에서 해제한다. 파일·작업 기록·승인은 지우지 않는다."""
    _discard(st)

    for key in ("ui_source_selection", "ui_source_snapshot", "ui_input_preview", "ui_derived_input", DEDUPE_OPTION,
                *SOURCE_INPUT_KEYS):
        st.session_state.pop(key, None)


def _retry_with_dedupe(st):
    """사용자가 누른 경우에만 같은 입력의 진단 연결을 풀고 자동 정비 옵션을 켠다. 정비 작업은 새 승인이 필요하다."""
    _discard(st)
    st.session_state[DEDUPE_OPTION] = True


def current_profiling(st, source, directory):
    """입력 선택이나 snapshot이 바뀌면 이전 진단·역할 초안·미리보기를 버린다. 저장 파일은 지우지 않는다."""
    profiling = st.session_state.get("ui_input_profiling")

    if profiling is not None and (not isinstance(profiling, UiInputProfiling)
                                  or profiling.source_selection_id != source["selection_id"]
                                  or profiling.snapshot_directory != directory):
        _discard(st)
        return None

    return profiling


def render_input_profiling(st):
    source = selected_source(st.session_state)

    if source is None or source["mode"] not in (CSV_SOURCE, CSV_SNAPSHOT):
        return

    st.subheader("기초 분석으로 컬럼 역할·기준 정하기 (선택)")
    if not st.toggle("입력 진단 열기", key="ui_profiling_open",
                     help="프로파일이 없을 때 승인된 격리 작업으로 컬럼·자료형·결측·기초 통계·이상치 후보를 먼저 확인합니다."):
        return

    st.caption("진단 작업은 별도 승인 후 격리 실행합니다. 결과는 이 PC 화면에만 표시하며 LLM에는 보내지 않습니다. "
               "역할은 추천하지 않으며 데이터를 삭제·변환하지 않습니다.")

    if source["mode"] == CSV_SOURCE:
        from .input_view import prepare_selected_input

        prepare_selected_input(st)

    try:
        directory = _snapshot_directory(st, source)
    except ValueError:
        st.error("등록한 파생 입력이 등록 기록·원본 연결과 맞지 않아 사용하지 않습니다. 다른 CSV를 선택하거나 입력을 다시 선택하세요.")
        st.button("다른 CSV 선택하기", key="ui_profiling_choose_other", on_click=_choose_other_input, args=(st,))
        return

    if directory is None:
        st.info("먼저 선택한 CSV의 snapshot을 준비하세요.")
        return

    derived = _derived(st, source)

    if derived is not None:
        if derived["changed"]:
            st.success(derived["message"])
            st.caption("파생 입력: " + derived["directory"] + " · 대응표·원본 연결 기록: " + derived["record_path"])
        else:
            st.info(derived["message"])

    profiling = current_profiling(st, source, directory)

    if profiling is None:
        _render_creation(st, source, directory, offer_dedupe=derived is None)
        return

    render_approval_flow(st, profiling, workflow=False)

    try:
        status = profiling.read_state().status
    except Exception:
        st.error("진단 작업 상태를 확인하지 못했습니다. 자동 재시도하지 않습니다.")
        return

    action = recovery_action(status)

    if action == "choose_other":
        _render_blocked(st, profiling)
        return

    if action == "restart":
        st.info(RETRY_NOTICE)
        st.button("같은 입력으로 새 진단 작업 준비", key="ui_profiling_restart", on_click=_discard, args=(st,),
                  help="세션의 진단 작업 연결만 해제합니다. 작업 기록·승인·파일은 그대로 둡니다.")
        return

    if status != "completed":
        return

    if profiling.purpose == "header_dedupe":
        if st.session_state.get("ui_input_profile_register_failed") == profiling.name:
            _render_register_failed(st)
        else:
            _register(st, profiling)
        return

    result = st.session_state.get("ui_input_profile_result")

    if result is None or result["run_name"] != profiling.name:
        try:
            result = load_profile_result(profiling)
        except Exception:
            st.error("검증된 진단 결과를 열지 못했습니다. 결과를 표시하거나 역할 지정에 사용하지 않습니다.")
            return

        st.session_state["ui_input_profile_result"] = result

    render_profile_result(st, result)
    render_profile_builder(st, result)


def _render_blocked(st, profiling):
    """차단 사유를 고정 문구로만 보여 준다. 컬럼명·값·경로는 표시하지 않고 자동 재시도하지 않는다."""
    reason = st.session_state.get("ui_input_profile_blocked")

    if reason is None or reason["run_name"] != profiling.name:
        try:
            reason = dict(blocked_reason(profiling) or {"code": "INPUT_BLOCKED_OTHER"}, run_name=profiling.name)
        except Exception:
            # 재검증 실패는 저장하지 않는다. 다음 화면 표시에서 다시 판별한다.
            reason = {"code": "REASON_UNVERIFIED", "run_name": profiling.name}
        else:
            st.session_state["ui_input_profile_blocked"] = reason

    st.error(BLOCKED_MESSAGES[reason["code"]])

    if reason["code"] == "DUPLICATE_COLUMN_NAMES":
        st.caption(f"중복된 이름 {reason['duplicated_names']}개 · 해당 컬럼 {reason['affected_columns']}개 (이름은 표시하지 않습니다)")

    if reason["code"] == "DUPLICATE_COLUMN_NAMES" and profiling.purpose == "input_profile":
        st.button("중복 컬럼명 자동 정비로 다시 준비", key="ui_profiling_dedupe_retry", on_click=_retry_with_dedupe,
                  args=(st,), help="자동 정비 옵션을 켜고 새 정비 작업을 준비할 수 있게 합니다. 정비 작업은 별도 승인 후 실행합니다.")

    st.info(NEXT_INPUT)
    st.button("다른 CSV 선택하기", key="ui_profiling_choose_other", on_click=_choose_other_input, args=(st,),
              help="1번 화면의 입력 선택만 해제합니다. 원본·snapshot·작업 기록은 지우지 않습니다.")


def _register(st, profiling):
    """완료된 정비 run을 검증해 파생 입력으로 등록하고 진단 단계로 넘어간다. 실패하면 등록하지 않는다."""
    try:
        derived = register_derived_input(profiling)
    except Exception:
        # 실패를 기록해 화면을 다시 그려도 등록을 자동으로 재시도하지 않는다.
        st.session_state["ui_input_profile_register_failed"] = profiling.name
        _render_register_failed(st)
        return

    st.session_state["ui_derived_input"] = derived
    _discard(st)
    st.rerun()


def _render_register_failed(st):
    st.error("정비 결과를 파생 입력으로 등록하지 못했습니다. 부분 결과는 입력으로 사용하지 않습니다. 자동 재시도하지 않습니다.")
    st.button("같은 입력으로 새 정비 작업 준비", key="ui_profiling_restart", on_click=_discard, args=(st,),
              help="새 정비 작업과 새 승인으로 다시 진행합니다. 이전 작업 기록과 남은 폴더는 지우지 않습니다.")


def _render_creation(st, source, directory, *, offer_dedupe=True):
    limits = st.session_state["ui_settings"]["limits"]

    with st.form("input_profiling_create_" + source["selection_id"][:16]):
        st.caption(f"자원 상한: 메모리 {limits['memory_mib']} MiB · 프로세스 {limits['processes']}개 · 시간 {limits['wall_seconds']}초 (설정 화면 값)")
        dedupe = offer_dedupe and st.checkbox(
            "중복 컬럼명이 있으면 복사본에서 자동 정비 (예: zero → zero_1~zero_19)", key=DEDUPE_OPTION,
            help="켜면 먼저 정비 작업을 승인·실행해 새 파생 입력을 만들고, 진단은 그 파생 입력으로 새 승인을 받아 진행합니다. "
                 "원본·기존 snapshot·데이터 값·컬럼 순서는 바꾸지 않습니다. 끄면 중복 컬럼명은 기존처럼 차단합니다.")
        confirmed = st.checkbox("선택한 snapshot의 입력 진단 작업 준비에 동의합니다", value=False,
                                help="미승인 작업 기록만 만듭니다. 승인·worker 실행·LLM 호출은 하지 않습니다.")
        submitted = st.form_submit_button("입력 진단 작업 준비")

    if not submitted:
        return

    if not confirmed:
        st.warning("작업 준비 범위를 확인하고 동의해 주세요.")
        return

    try:
        st.session_state["ui_input_profiling"] = create_profiling(source["selection_id"], directory, limits, confirmed=True,
                                                                  header_dedupe=dedupe is True)
    except Exception:
        st.error("입력 진단 작업을 준비하지 못했습니다. snapshot과 자원 설정을 확인하세요. 자동 재시도하지 않습니다.")
    else:
        st.rerun()


def render_profile_result(st, result):
    st.subheader("입력 진단 결과")
    st.caption(DIAGNOSTIC_NOTICE)
    summary = result["summary"]
    columns = st.columns(3)
    columns[0].metric("행 수", summary["row_count"])
    columns[1].metric("컬럼 수", summary["column_count"])
    columns[2].metric("중복 행 수", summary["repeated_rows"])
    st.dataframe(pd.DataFrame(result["columns"]), hide_index=True)
    st.caption("결측은 빈 칸 기준이며 공백 문자열은 별도 건수입니다. 품질 검사는 공백 문자열도 결측으로 봅니다. "
               "수치 컬럼의 최솟값·최댓값·분위수는 원본 값 그대로이므로 ID형 컬럼은 화면 공유에 주의하세요.")


def _roles(st, result):
    draft = st.session_state.get("ui_profile_roles")

    if draft is None or draft["result_id"] != result["result_id"]:
        draft = {"result_id": result["result_id"], "revision": 0,
                 "roles": {row["column"]: EXCLUDED for row in result["columns"]}}
        st.session_state["ui_profile_roles"] = draft

    return draft


def _bulk_numeric(st, result):
    """사용자가 누를 때만 제외로 남은 수치 자료형 컬럼을 수치형으로 바꾼다. 다른 역할은 바꾸지 않는다."""
    draft = _roles(st, result)

    for row in result["columns"]:
        if row["dtype_group"] == "numeric" and draft["roles"][row["column"]] == EXCLUDED:
            draft["roles"][row["column"]] = "수치형"

    draft["revision"] += 1
    st.session_state.pop("ui_profile_preview", None)


def uninformative_columns(result):
    """진단 결과 기준 상수(관측 고유값 1개)·전체 결측(0개) 컬럼. 원본을 다시 읽지 않는다."""
    return [row["column"] for row in result["columns"]
            if row["observed_unique"] is not None and row["observed_unique"] <= 1]


def _bulk_exclude_uninformative(st, result):
    """사용자가 누를 때만 상수·전체 결측 특징을 제외로 바꾸고 적용 전후 개수를 남긴다. 데이터 파일은 바꾸지 않는다."""
    draft = _roles(st, result)
    before = sum(1 for role in draft["roles"].values() if role == EXCLUDED)
    targets = uninformative_columns(result)
    changed = kept = 0

    for name in targets:
        role = draft["roles"][name]

        if role in FEATURE_LABELS:
            draft["roles"][name] = EXCLUDED
            changed += 1
        elif role != EXCLUDED:
            kept += 1

    draft["revision"] += 1
    st.session_state.pop("ui_profile_preview", None)
    st.session_state["ui_profile_bulk_notice"] = {
        "result_id": result["result_id"], "targets": len(targets), "changed": changed, "kept": kept,
        "before": before, "after": sum(1 for role in draft["roles"].values() if role == EXCLUDED),
    }


def render_profile_builder(st, result):
    st.subheader("컬럼 역할·기준 지정")
    st.caption("초기 역할은 모두 ‘제외’입니다. 제외 컬럼은 프로파일에 쓰지 않습니다. 이상치 처리 방식은 이후 전처리 계획에서 선택합니다.")
    draft = _roles(st, result)
    st.button("제외로 남은 수치 자료형 컬럼을 모두 수치형으로", key="ui_profile_bulk_numeric",
              on_click=_bulk_numeric, args=(st, result),
              help="누를 때만 적용합니다. 이후 표에서 개별 수정할 수 있습니다.")
    st.button("상수·전체 결측 컬럼 일괄 제외", key="ui_profile_bulk_exclude",
              on_click=_bulk_exclude_uninformative, args=(st, result),
              help="진단 결과에서 관측 고유값이 1개(상수) 또는 0개(전체 결측)인 수치형·범주형 컬럼만 제외로 바꿉니다. "
                   "target·식별자·group·timestamp는 바꾸지 않습니다. " + EXCLUDE_NOTICE)
    notice = st.session_state.get("ui_profile_bulk_notice")

    if notice is not None and notice["result_id"] == result["result_id"]:
        st.info(f"상수·전체 결측 컬럼 {notice['targets']}개 중 {notice['changed']}개를 제외로 바꿨습니다. "
                f"제외 컬럼 수: 적용 전 {notice['before']}개 → 적용 후 {notice['after']}개. "
                f"target·식별자·group·timestamp로 지정한 {notice['kept']}개는 바꾸지 않았습니다. "
                + EXCLUDE_NOTICE + " 아래 표에서 개별 수정할 수 있습니다.")

    purpose = PURPOSE_LABELS[st.selectbox("프로파일 목적", tuple(PURPOSE_LABELS), key="ui_profile_purpose")]
    table = pd.DataFrame([{**{name: row[name] for name in SHOWN}, "role": draft["roles"][row["column"]]}
                          for row in result["columns"]])

    with st.form("profile_builder_form"):
        edited = st.data_editor(
            table, key=f"ui_profile_editor_{result['result_id'][:12]}_{draft['revision']}", hide_index=True,
            disabled=list(SHOWN),
            column_config={"role": st.column_config.SelectboxColumn("역할", options=list(ROLE_LABELS), required=True)},
        )
        profile_id = st.text_input("프로파일 이름 (파일 이름)", value="ui_profile01", max_chars=64,
                                   help="소문자로 시작하는 영문 소문자·숫자·밑줄입니다. 같은 이름은 덮어쓰지 않습니다.")
        plot = st.number_input("그래프 특징 상한", min_value=1, max_value=30, value=12)
        threshold = st.number_input("결측률 검토 기준", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        level = st.selectbox("결측률 초과 통지 수준", ("review", "block", "ignore"))
        split = None

        if purpose == "split":
            ratios = [st.number_input(title, min_value=0.01, max_value=0.98, value=value, step=0.05)
                      for title, value in (("Train 비율", 0.70), ("Validation 비율", 0.15), ("Test 비율", 0.15))]
            split = {"strategy": st.selectbox("분할 방식", ("random", "group", "time")),
                     "train_ratio": ratios[0], "validation_ratio": ratios[1], "test_ratio": ratios[2],
                     "random_state": int(st.number_input("난수 시드", min_value=0, max_value=2**32 - 1, value=42)),
                     "stratify": st.checkbox("target 기준 계층 분할 (random만)", value=False),
                     "require_all_classes": st.checkbox("모든 구간에 모든 클래스 요구", value=True)}

        submitted = st.form_submit_button("프로파일 검증 미리보기")

    if submitted:
        st.session_state.pop("ui_profile_preview", None)
        draft["roles"] = dict(zip(edited["column"], edited["role"]))

        try:
            profile = build_profile(
                tuple(row["column"] for row in result["columns"]), draft["roles"], purpose=purpose,
                profile_id=profile_id, max_plot_features=int(plot),
                quality={"missing_ratio_threshold": float(threshold), "high_missing": level}, split=split,
            )
        except Exception:
            st.error("프로파일 계약 검사를 통과하지 못했습니다. 특징 컬럼 1개 이상, target·group·timestamp 각 1개 이하, "
                     "이름 형식, 분할 비율 합 1을 확인하세요.")
        else:
            st.session_state["ui_profile_preview"] = {"result_id": result["result_id"], "profile": profile,
                                                      "threshold": float(threshold)}

    _render_preview(st, result)


def _render_preview(st, result):
    preview = st.session_state.get("ui_profile_preview")

    if preview is None or preview["result_id"] != result["result_id"]:
        return

    profile = preview["profile"]
    roles = profile.roles
    st.success("프로파일 계약 검사를 통과했습니다. 아직 저장하지 않았습니다.")
    st.text(f"수치형 {len(roles.numeric)} · 범주형 {len(roles.categorical)} · 식별자 {len(roles.identifiers)} · "
            f"target {roles.target is not None:d} · group {roles.group is not None:d} · timestamp {roles.timestamp is not None:d}")
    st.caption(f"결측률 기준 {preview['threshold']:g} 초과 컬럼 {missing_over(result, preview['threshold'])}개 — 표시만 하며 자동 제외하지 않습니다.")
    rules = profile.split_rules

    if rules is not None and ((rules.strategy == "group" and roles.group is None)
                              or (rules.strategy == "time" and roles.timestamp is None)
                              or (rules.stratify and roles.target is None)):
        st.warning("분할 방식에 필요한 group·timestamp·target 역할이 없습니다. 이후 분할 작업에서 거부될 수 있습니다.")

    with st.expander("저장할 프로파일 JSON"):
        st.json(profile.to_dict())

    saved = st.session_state.get("ui_profile_saved")

    if saved is not None and saved["profile_signature"] == profile.signature:
        st.success("저장했습니다: " + saved["local_file"])
        st.caption("같은 입력 선택의 3 입력 선택(EDA) 또는 등록 모듈 분할 검토 프로파일 입력란에 기본값으로 연결했습니다. "
                   "컬럼 실재·자료형은 이후 승인된 격리 검사에서 확인합니다. 실행 승인은 아닙니다.")
        return

    with st.form("profile_save_form"):
        confirmed = st.checkbox("새 로컬 프로파일 파일만 생성합니다. 실행 승인이나 데이터 검사가 아닙니다", value=False)
        submitted = st.form_submit_button("프로파일 저장", type="primary")

    if not submitted:
        return

    if not confirmed:
        st.warning("저장 범위를 확인하고 동의해 주세요.")
        return

    try:
        # 화면에 표시한 미리보기 객체 그대로 저장한다. 다시 만들지 않는다.
        saved = save_profile(profile)
    except FileExistsError:
        st.error("같은 이름의 프로파일이 이미 있습니다. 새 이름으로 미리보기를 다시 만드세요. 기존 파일은 바꾸지 않았습니다.")
    except Exception:
        st.error("프로파일을 저장하지 못했습니다. 저장 위치를 확인하세요. 입력란에 연결하지 않았습니다.")
    else:
        st.session_state["ui_profile_saved"] = saved
        derived = st.session_state.get("ui_derived_input")
        uses_derived = (isinstance(derived, dict) and derived.get("changed") is True
                        and derived.get("directory") == result["snapshot_directory"])
        st.session_state["ui_linked_profile"] = link_record(result, saved, derived=uses_derived)
        st.rerun()

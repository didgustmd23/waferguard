"""저장 제안·snapshot·프로파일을 결합하고 별도 확인 후 미승인 실행 후보를 준비한다."""

from src.agents.worker_contract import WorkerLimits
from .dialogue_view import current_dialogue
from .source_selection import CSV_SOURCE, CSV_SNAPSHOT, prepare_source_snapshot, selected_source
from src.agents.input_snapshot import InputTooLarge, MIB
from .path_picker import path_input
from .dialogue_service import fingerprint
from .candidate_view import render_candidate_creation


def prepare_selected_input(st):
    """첫 화면의 선택을 이어받는다. 원본은 별도 동의 후에만 바이트 복사한다."""
    source = selected_source(st.session_state)

    if source is None or source["mode"] not in (CSV_SOURCE, CSV_SNAPSHOT):
        return "", "", False

    if source["mode"] == CSV_SNAPSHOT:
        return source["path"], source["profile"], True

    st.text("선택한 CSV 원본: " + source["path"])
    copied = st.session_state.get("ui_source_snapshot")

    if copied is not None and copied["selection_id"] == source["selection_id"]:
        st.success("선택한 CSV의 로컬 snapshot을 준비했습니다. 원본은 변경하지 않았습니다.")
        return copied["directory"], source["profile"], True

    with st.form("copy_source_" + source["selection_id"]):
        maximum = st.number_input("입력 복사 크기 상한 (MiB)", min_value=1, max_value=4096,
                                   value=min(4096, max(8, (source["size_bytes"] * 5 // 4 + MIB - 1) // MIB)),
                                   help="디스크에 만들 복사본의 크기 상한입니다. worker 메모리 상한이 아니며 초과 파일을 자르지 않습니다.")
        confirmed = st.checkbox("선택한 원본을 새 로컬 snapshot으로 복사하는 데 동의합니다", value=False,
                                help="원본 바이트만 복사합니다. CSV 해석·EDA·LLM 전송·worker 실행은 하지 않습니다.")
        submitted = st.form_submit_button("선택한 CSV의 snapshot 준비")

    if submitted:
        if not confirmed:
            st.warning("복사 내용을 확인하고 동의해 주세요.")
        else:
            try:
                copied = prepare_source_snapshot(source, maximum)
            except InputTooLarge as error:
                st.error(f"입력 크기 {error.size / MIB:.2f} MiB가 상한 {error.maximum / MIB:.2f} MiB를 넘습니다. 자르거나 분석하지 않았습니다.")
            except Exception:
                st.error("복사본을 준비하지 못했습니다. 원본 변경·경로·크기 상한을 확인하세요. 자동 재시도하지 않습니다.")
            else:
                st.session_state["ui_source_snapshot"] = copied
                st.rerun()

    return None, source["profile"], True


def input_available(st):
    """세션의 현재 제안과 저장 파일이 같을 때만 입력 선택 화면을 연다."""
    dialogue = current_dialogue(st)
    return dialogue is not None and dialogue.saved_proposal(st.session_state["ui_settings"]) is not None


def preview_input(dialogue, settings, snapshot_directory, profile_path):
    """기존 바인더를 재사용한다. 경로는 로컬에서만 쓰고 모델에는 보내지 않는다."""
    saved = dialogue.saved_proposal(settings)

    if saved is None:
        raise ValueError("검토한 제안을 먼저 저장하세요.")

    # 실제 파일 접근은 사용자가 선택 확인을 제출했을 때만 진행한다.
    from src.agents.llm_execution_candidate import preview_candidate

    result = preview_candidate(saved["name"], snapshot_directory, profile_path,
                               WorkerLimits(**settings["limits"]))
    selection = result["selection"]

    if (selection["proposal_plan_signature"] != saved["plan_signature"]
            or selection["pipeline"] != saved["proposal"]["pipeline"]
            or dialogue.saved_proposal(settings) != saved):
        raise ValueError("입력 확인 중 저장 제안이 변경되었습니다.")

    return result


def render_input(st):
    st.header("3 입력 선택")
    dialogue = current_dialogue(st)
    settings = st.session_state["ui_settings"]
    saved = None if dialogue is None else dialogue.saved_proposal(settings)

    if saved is None:
        st.warning("현재 제안을 검토·저장해야 입력을 선택할 수 있습니다.")
        return

    st.text("저장 제안: " + saved["name"])
    default_snapshot, default_profile, fixed_snapshot = prepare_selected_input(st)

    if default_snapshot is None:
        st.info("먼저 위에서 선택한 CSV의 snapshot을 준비하세요. 내용 분석은 별도 승인 이후입니다.")
        return

    st.info("선택한 입력의 snapshot과 프로파일 JSON을 확인하세요. 경로·데이터는 LLM에 전송하지 않습니다.")
    st.caption("snapshot 폴더에는 input.bin과 snapshot.json이 필요합니다. "
               "input.bin은 원본 바이트를 보존한 복사본이며 확장자로 자료형을 추측하지 않습니다.")

    with st.form("input_selection_form"):
        namespace = fingerprint({"source": getattr(dialogue, "ui_source_id", None),
                                 "snapshot": default_snapshot})[:16]
        snapshot = path_input(
            st, "snapshot 폴더 경로", key="input_snapshot_" + namespace, kind="directory",
            value=default_snapshot, disabled=fixed_snapshot,
            help="경로 입력 또는 폴더 찾기로 지정합니다. 첫 화면에서 선택한 고정 입력은 그 화면에서 변경하세요.",
        )
        profile = path_input(
            st, "프로파일 JSON 경로", key="input_profile_" + namespace, kind="json_file", value=default_profile,
            help="기존 로컬 JSON을 직접 입력하거나 파일 찾기로 지정하세요. 경로와 내용은 LLM에 보내지 않습니다.",
        )
        confirmed = st.checkbox(
            "선택한 로컬 입력의 정합성 검사를 진행합니다", value=False,
            help="제안·snapshot·프로파일·자원을 결합해 검사합니다. 실행 승인이나 worker 시작은 아닙니다.",
        )
        submitted = st.form_submit_button("입력 결합 미리보기", type="primary")

    limits = settings["limits"]
    st.caption(f"메모리 {limits['memory_mib']} MiB · 프로세스 {limits['processes']}개 · 시간 {limits['wall_seconds']}초")
    input_draft_id = fingerprint({"snapshot": default_snapshot if fixed_snapshot else snapshot, "profile": profile})
    previous = st.session_state.get("ui_input_preview")

    if previous is not None and previous.get("input_draft_id") != input_draft_id:
        st.session_state.pop("ui_input_preview", None)
        st.session_state.pop("ui_execution_candidate", None)
        st.info("입력 경로가 바뀌었습니다. 이전 확인을 사용하지 않으며 입력 결합 미리보기를 다시 검사해야 합니다.")

    if submitted:
        # 이전 성공 미리보기를 새 입력의 검사 결과처럼 남기지 않는다.
        st.session_state.pop("ui_input_preview", None)
        st.session_state.pop("ui_execution_candidate", None)

        if not confirmed:
            st.warning("선택한 입력의 검사에 동의해 주세요.")
        else:
            try:
                # 고정 입력은 브라우저 위젯 값 대신 서버가 보관한 선택 경로를 사용한다.
                result = preview_input(dialogue, settings, default_snapshot if fixed_snapshot else snapshot, profile)
            except Exception:
                st.error("입력 결합 검사를 통과하지 못했습니다. snapshot·프로파일 경로와 "
                         "현재 저장 제안·입력 계약을 확인하세요. 파일을 변경하거나 worker를 실행하지 않았습니다.")
            else:
                st.session_state["ui_input_preview"] = {
                    "settings_id": dialogue.settings_id,
                    "plan_signature": saved["plan_signature"],
                    "proposal_name": saved["name"],
                    "input_draft_id": input_draft_id,
                    "result": result,
                }

    preview = st.session_state.get("ui_input_preview")

    if (preview is not None and preview["settings_id"] == dialogue.settings_id
            and preview["plan_signature"] == saved["plan_signature"]
            and preview["proposal_name"] == saved["name"]):
        st.success("선택 시점의 입력 결합 검사를 통과했습니다. 실행 후보 생성·실행 승인은 하지 않았습니다.")
        st.caption("미리보기는 선택 시점의 결과입니다. 이후 파일 변경은 자동 감시하지 않으며 실제 작업 준비 시 재검사가 필요합니다.")

        with st.expander("로컬 입력 결합 상세"):
            st.json(preview["result"])
        render_candidate_creation(st, preview, dialogue, settings)

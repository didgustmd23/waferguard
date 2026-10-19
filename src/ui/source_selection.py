"""요청 전 로컬 데이터 선택. 파일 내용은 분석하거나 모델에 전달하지 않는다."""

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from src.agents.artifacts import _root
from src.agents.input_snapshot import MIB, _identity, create_snapshot
from src.agents.worker_paths import _path
from .dialogue_service import fingerprint
from .path_picker import path_input


UNSELECTED = "데이터를 선택하세요"
CSV_SOURCE = "CSV 원본 파일"
CSV_SNAPSHOT = "기존 CSV snapshot"
PLAN_ONLY = "데이터 없이 제안만 검토"
UNSUPPORTED = "맵·이미지·pickle (UI 연결 예정)"
SOURCE_MODES = (UNSELECTED, CSV_SOURCE, CSV_SNAPSHOT, PLAN_ONLY, UNSUPPORTED)
SNAPSHOT_ROOT = Path(__file__).resolve().parents[2] / "logs/agents/input_snapshots/ui"


def selected_source(state):
    """미확인 폼 값은 사용하지 않고 명시적으로 선택 완료한 정보만 반환한다."""
    mode = state.get("ui_source_mode", UNSELECTED)

    if mode == PLAN_ONLY:
        return {"mode": PLAN_ONLY, "selection_id": "proposal_only"}

    selected = state.get("ui_source_selection")
    return selected if selected is not None and selected["mode"] == mode else None


def select_source(mode, path, profile=""):
    """명시적 제출 시 일반 로컬 경로·메타데이터만 확인한다. CSV 내용 검사는 후속 worker 역할이다."""
    if mode not in (CSV_SOURCE, CSV_SNAPSHOT) or not path.strip():
        raise ValueError("지원하는 데이터 유형과 로컬 경로를 지정하세요.")

    local = _path(Path(path).absolute(), directory=mode == CSV_SNAPSHOT)

    if mode == CSV_SOURCE and local.suffix.lower() != ".csv":
        raise ValueError("CSV 원본은 .csv 파일을 선택하세요. 확장자가 내용의 유효성을 보증하지는 않습니다.")

    payload = local if mode == CSV_SOURCE else _path(local / "input.bin", directory=False)

    if mode == CSV_SNAPSHOT:
        _path(local / "snapshot.json", directory=False)

    profile_path = str(_path(Path(profile).absolute(), directory=False)) if profile.strip() else ""
    info = payload.stat()
    value = {"mode": mode, "path": str(local), "profile": profile_path,
             "identity": list(_identity(info)), "size_bytes": info.st_size}
    return {**value, "selection_id": fingerprint(value)}


def prepare_source_snapshot(selection, max_input_mib, *, root=SNAPSHOT_ROOT):
    """선택한 원본이 그대로일 때만 새 snapshot으로 복사한다. 해석·역직렬화는 하지 않는다."""
    if (selection["mode"] != CSV_SOURCE or type(max_input_mib) is not int
            or not 1 <= max_input_mib <= 4096):
        raise ValueError("CSV 원본과 1~4096 MiB 복사 상한이 필요합니다.")

    current = select_source(CSV_SOURCE, selection["path"], selection["profile"])

    if current != selection:
        raise ValueError("선택 후 원본 정보가 바뀌었습니다. 첫 화면에서 다시 선택하세요.")

    # 기존 CLI와 같은 복사 함수를 재사용하며 링크 없는 출력 경로만 준비한다.
    root = Path(root).absolute()
    for component in reversed((root, *root.parents)):
        component.mkdir(exist_ok=True)
        _root(component)

    destination = root / ("source_" + uuid4().hex)
    snapshot = create_snapshot(Path(selection["path"]), destination,
                               max_input_bytes=max_input_mib * MIB)

    if select_source(CSV_SOURCE, selection["path"], selection["profile"]) != selection:
        raise ValueError("복사 과정에서 원본 정보가 바뀌었습니다. 결과를 후속 입력으로 사용하지 마세요.")

    return {"selection_id": selection["selection_id"], "directory": str(destination),
            "snapshot": asdict(snapshot)}


def remember_source_mode(st):
    """화면 이동으로 위젯 키가 정리되어도 선택 모드는 별도 세션 키에 보존한다."""
    st.session_state["ui_source_mode"] = st.session_state["ui_source_mode_widget"]


def render_source_selection(st):
    st.subheader("입력 데이터")
    stored_mode = st.session_state.get("ui_source_mode", UNSELECTED)
    mode = st.selectbox("데이터 유형", SOURCE_MODES, key="ui_source_mode_widget",
                        index=SOURCE_MODES.index(stored_mode), on_change=remember_source_mode, args=(st,),
                        help="현재 대화 제안은 표형 CSV용입니다. 맵·이미지·pickle은 기존 CLI가 있어도 이 화면에는 아직 연결하지 않았습니다.")
    st.session_state["ui_source_mode"] = mode

    if mode in (CSV_SOURCE, CSV_SNAPSHOT):
        previous = selected_source(st.session_state)
        with st.form("source_selection_form"):
            prefix = "source_csv" if mode == CSV_SOURCE else "source_snapshot"
            path = path_input(st, "데이터 경로", key=prefix + "_path",
                              kind="csv_file" if mode == CSV_SOURCE else "directory",
                              value="" if previous is None else previous["path"],
                              help="경로를 직접 입력하거나 옆의 찾기 버튼으로 선택하세요. 경로는 LLM에 보내지 않습니다.")
            profile = path_input(st, "프로파일 JSON 경로 (선택)", key=prefix + "_profile", kind="json_file",
                                 value="" if previous is None else previous["profile"],
                                 help="기존 역할·품질 기준 JSON이 있으면 직접 입력하거나 파일 찾기로 지정하세요. 3번에서도 지정할 수 있습니다.")
            submitted = st.form_submit_button("로컬 데이터 선택 확인",
                                               help="경로와 크기 등 파일 정보만 확인합니다. 내용 분석·복사·모델 호출은 하지 않습니다.")

        if submitted:
            try:
                selection = select_source(mode, path, profile)
            except Exception:
                st.session_state.pop("ui_source_selection", None)
                st.error("입력 경로를 확인할 수 없습니다. 로컬 일반 CSV 파일 또는 snapshot 폴더와 프로파일 경로를 확인하세요.")
            else:
                previous = st.session_state.get("ui_source_selection")
                st.session_state["ui_source_selection"] = selection

                if previous != selection:
                    st.session_state.pop("ui_source_snapshot", None)
                    st.session_state.pop("ui_input_preview", None)

    selected = selected_source(st.session_state)

    if selected is not None and mode != PLAN_ONLY:
        st.text("선택한 경로: " + selected["path"])
        st.caption(f"선택한 크기: {selected['size_bytes'] / MIB:.2f} MiB · 아직 내용 분석은 하지 않았습니다.")
    elif mode == PLAN_ONLY:
        st.info("데이터 없이 처리 순서만 제안받습니다. 실제 데이터의 분석 결과가 아닙니다.")
    elif mode == UNSUPPORTED:
        st.warning("이 유형은 UI 연결 전입니다. CSV용 모델 제안으로 대신 진행하지 않습니다.")
    else:
        st.info("데이터 유형·경로를 선택하거나, ‘데이터 없이 제안만 검토’를 명시적으로 선택하세요.")

    st.divider()

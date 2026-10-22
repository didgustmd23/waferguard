"""사용자용 상태·경로 진단. 임의 예외 원문이나 파일 내용을 화면·LLM에 전달하지 않는다."""

from pathlib import Path

from src.agents.worker_paths import _path


STATE_LABELS = {"created": "작업 준비됨", "running": "진행 중", "awaiting_approval": "사용자 승인 대기",
                "completed": "실행 완료", "blocked_input": "입력 차단·품질 검토 필요",
                "failed_runtime": "실행 실패", "security_locked": "보안 잠금", "cancelled": "취소됨"}
STAGE_LABELS = {"preflight": "사전 검사", "run_eda": "품질 검사·EDA", "review_eda": "보고서 검토",
                "derive_preparation": "전처리 후보 산출", "commit_split": "분할 확정·선행 조건 확인",
                "prepare_data": "데이터 준비", "validate_contract": "입력 계약 검사"}


def diagnose_paths(schema, values):
    """제출 시에만 선택 경로를 검사한다. 결과를 저장해 재표시 때 파일을 다시 읽지 않는다."""
    rows = []
    for field in schema["fields"]:
        if field["kind"] not in {"directory", "json_file", "csv_file"}:
            continue

        value = values[field["name"]]
        row = {"title": field["title"], "entered": value if type(value) is str else "문자열 경로가 아닙니다.",
               "resolved": None, "valid": False}
        try:
            if type(value) is not str or not value.strip() or len(value) > 4096 or any(ord(c) < 32 for c in value):
                raise ValueError("invalid path")

            # 상대/절대 모두 기존 worker 경로 검사로 확인한다. 승인 입력을 치환하지 않는다.
            row["resolved"] = str(_path(Path(value).absolute(), directory=field["kind"] == "directory"))
            row["valid"] = True
        except (ValueError, TypeError, OSError):
            pass

        rows.append(row)

    return rows


def render_paths(st, rows):
    if not rows:
        return

    with st.expander("입력 경로 진단", expanded=any(not row["valid"] for row in rows)):
        st.caption("최근 미리보기 제출 시점의 검사 결과입니다. 입력을 변경했다면 다시 검사하세요.")
        for row in rows:
            st.text(row["title"] + (": 확인됨" if row["valid"] else ": 확인 필요"))
            st.text("입력: " + row["entered"][:4096])
            if row["resolved"]:
                st.text("확인된 경로: " + row["resolved"])
            else:
                st.caption("파일·폴더 종류, 존재 여부, 접근 권한, 따옴표·공백, 링크 여부를 확인하세요.")


def show_error(st, error, fallback):
    """알려진 고정 조건만 설명한다. 로컬 경로가 섞인 예외 원문을 그대로 출력하지 않는다."""
    known = {
        "selected path rejected":
            ("INPUT_PATH_INVALID", "선택 경로를 확인하지 못했습니다. 입력 경로 진단에서 문제 항목을 확인하세요."),
        "기존 작업 또는 생성 기록을 덮어쓰지 않습니다. 새 이름을 사용하세요.":
            ("WORKFLOW_NAME_EXISTS", "이미 사용한 작업 이름입니다. 기존 작업으로 연결하거나 새 이름을 지정하세요."),
        "작업 배정이 현재 모듈 계획의 버전·서명과 다릅니다.":
            ("PIPELINE_BINDING_MISMATCH", "모듈 계획과 작업 배정 파일이 서로 맞지 않습니다. 같은 계획의 배정을 선택하세요."),
        "지원하는 첫 미배정 WM 단계만 생성할 수 있습니다.":
            ("STEP_NOT_AVAILABLE", "단계 ID가 없거나 이미 배정됐습니다. 선택한 계획·배정을 확인하세요."),
        "선행 작업을 먼저 검증하고 현재 첫 미배정 단계만 생성하세요.":
            ("PREVIOUS_STEP_REQUIRED", "앞 단계의 완료 기록을 먼저 연결·검증해야 합니다."),
    }
    code, message = known.get(str(error), ("WORKFLOW_REQUEST_FAILED", fallback))
    if isinstance(error, PermissionError):
        code, message = "LOCAL_ACCESS_DENIED", "선택한 로컬 파일·폴더에 접근하지 못했습니다. UI 서버 권한과 경로를 확인하세요."
    elif isinstance(error, FileNotFoundError):
        code, message = "LOCAL_PATH_MISSING", "선택한 파일·폴더가 없습니다. 입력 경로 진단을 확인하세요."

    st.error(message)
    st.caption("진단 코드: " + code + " · 자동 재시도·재승인하지 않습니다.")


def render_approval_summary(st, summary, *, executes=False):
    st.info("현재 단계 한 번에 대한 승인·실행입니다." if executes else
            "현재 단계 한 번에 대한 승인입니다. 승인만으로 worker를 실행하지 않습니다.")
    if "stage" in summary:
        st.text("승인 단계: " + STAGE_LABELS.get(summary["stage"], summary["stage"]))

    if summary.get("input_change"):
        st.warning(summary["input_change"])

    limits = summary.get("limits", {})
    for key, title, unit in (("memory_mib", "메모리 상한", " MiB"), ("processes", "프로세스 상한", "개"),
                             ("wall_seconds", "실행 시간 상한", "초")):
        if key in limits:
            st.text(title + ": " + str(limits[key]) + unit)

    with st.expander("승인 범위 기술 상세"):
        st.json(summary)

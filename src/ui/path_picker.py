"""직접 경로 입력과 명시적 클릭에 의한 로컬 파일·폴더 선택을 함께 제공한다."""

import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock


DIALOG_SCRIPT = Path(__file__).with_name("path_dialog.py")
DIALOG_KINDS = ("csv_file", "json_file", "npz_file", "data_file", "directory")
DIALOG_TIMEOUT_SECONDS = 180
_DIALOG_LOCK = Lock()


def pick_local_path(kind):
    """서버 PC의 창을 한 번만 연다. 원격 브라우저의 디스크 선택 기능이 아니다."""
    if os.name != "nt" or kind not in DIALOG_KINDS or not _DIALOG_LOCK.acquire(blocking=False):
        return {"status": "unavailable"}

    process = None
    try:
        # 고정 helper만 실행한다. 사용자의 경로나 문장은 명령줄에 넣지 않는다.
        process = subprocess.Popen(
            [sys.executable, "-I", str(DIALOG_SCRIPT), kind], shell=False,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        raw, _ = process.communicate(timeout=DIALOG_TIMEOUT_SECONDS)

        if process.returncode != 0 or len(raw) > 32768:
            return {"status": "unavailable"}

        result = json.loads(raw.decode("utf-8"))

        if type(result) is not dict:
            return {"status": "unavailable"}

        if result == {"status": "cancelled"}:
            return result

        path = result.get("path")
        if (set(result) == {"status", "path"} and result["status"] == "selected"
                and type(path) is str and 0 < len(path) <= 4096
                and not any(ord(char) < 32 for char in path)):
            return result

        return {"status": "unavailable"}
    except Exception:
        return {"status": "unavailable"}
    finally:
        try:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            _DIALOG_LOCK.release()


def fill_from_dialog(st, key, kind):
    """폼 재표시 전에 입력란만 갱신한다. 선택 확인·검사는 별도 제출에서 수행한다."""
    result = pick_local_path(kind)

    if result["status"] == "selected":
        st.session_state[key] = result["path"]
        st.session_state["_widget_" + key] = result["path"]
        notice = "선택한 경로를 입력했습니다. 아직 선택 확인·검사·복사는 하지 않았습니다."
    elif result["status"] == "cancelled":
        notice = "선택을 취소했습니다. 기존 경로는 유지합니다."
    else:
        notice = "선택 창을 열 수 없거나 시간이 초과되었습니다. 경로를 직접 입력해 주세요."

    st.session_state[key + "_notice"] = notice


def path_input(st, label, *, key, kind, value="", help, disabled=False, browse_disabled=False):
    """기존 폼 안에서 사용한다. 찾아보기 역시 폼 제출 버튼이지만 확인 동작은 아니다.

    browse_disabled는 찾아보기 버튼만 잠근다(처리 중 잠금). 경로 입력 값은 같은 실행의 제출 값을 유지한다.
    """
    widget_key = "_widget_" + key

    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(key, value)

    columns = st.columns([4, 1], vertical_alignment="bottom")
    with columns[0]:
        path = st.text_input(label, key=widget_key, disabled=disabled, max_chars=4096, help=help)

    with columns[1]:
        st.form_submit_button(
            "폴더 찾기" if kind == "directory" else "파일 찾기", key=key + "_browse",
            disabled=disabled or browse_disabled, on_click=fill_from_dialog, args=(st, key, kind),
            help="이 PC에 선택 창을 엽니다. 취소하면 기존 입력을 유지합니다. 로컬 단일 사용자 모드용입니다.",
        )

    st.session_state[key] = path
    notice = st.session_state.get(key + "_notice")

    if notice:
        st.caption(notice)

    return path

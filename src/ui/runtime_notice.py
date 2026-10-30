"""현재 UI 서버 프로세스의 권한 표시. 승격 요청·프로세스 실행은 하지 않는다."""

import ctypes
import os
import sys


def server_permissions():
    """터미널 제목이 아닌 이 Python 프로세스의 활성 관리자 권한을 확인한다."""
    elevated = None
    if os.name == "nt":
        try:
            check = ctypes.windll.shell32.IsUserAnAdmin
            check.argtypes = []
            check.restype = ctypes.c_int
            elevated = bool(check())
        except (AttributeError, OSError):
            pass

    return {"elevated": elevated, "pid": os.getpid(), "python": sys.executable}


def render_permissions(st):
    permissions = server_permissions()
    if permissions["elevated"] is True:
        st.success("UI 서버 권한: 관리자 — 격리 작업 준비를 진행할 수 있습니다.")
    else:
        label = "일반 권한" if permissions["elevated"] is False else "확인 불가"
        st.warning("UI 서버 권한: " + label + ". 현재 구현의 격리 폴더 권한 설정에는 관리자 권한이 필요합니다. "
                   "UI를 종료한 뒤 관리자 PowerShell에서 다시 실행하세요. 조회·입력 검토는 가능합니다.")

    with st.expander("UI 서버 기술 상세"):
        st.text("프로세스 ID: " + str(permissions["pid"]))
        st.text("Python: " + permissions["python"])

    return permissions["elevated"] is True


EXECUTION_PERMISSION_ERROR = ("현재 UI 서버에 관리자 권한이 없어 승인·실행 요청을 보내지 않았습니다. "
                              "관리자 PowerShell에서 UI를 다시 실행하세요.")


def execution_permitted():
    """화면 출력 없이 실행 권한만 판정한다. 안내는 호출한 화면이 EXECUTION_PERMISSION_ERROR로 보인다."""
    return server_permissions()["elevated"] is True


def require_execution_permissions(st):
    """승인 제출·실행 제출 시 다시 검사한다. 기존 관리자/승인 정책을 대체하지 않는다."""
    if execution_permitted():
        return True

    st.error(EXECUTION_PERMISSION_ERROR)
    return False

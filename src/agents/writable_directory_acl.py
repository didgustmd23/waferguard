"""새 작업의 빈 출력·임시 폴더만 준비한다. 호출자 WO는 현재 사용자 SID에만 부여한다."""

import ctypes as ct
from ctypes import wintypes as wt
import os
from pathlib import Path
import subprocess

from .artifacts import _root
from .appcontainer_probe import _acl


class DirectoryAclError(OSError):
    """외부 명령 출력 대신 고정된 실패 작업만 운송 계층에 전달한다."""

    def __init__(self, operation, code=None):
        super().__init__(code, "WRITABLE_DIRECTORY_SETUP_FAILED")
        self.acl_operation = operation


# 승격된 관리자 토큰의 기본 소유자는 사용자 대신 이 그룹일 수 있다. 다른 그룹·사용자는 허용하지 않는다.
ADMINISTRATORS_SID = "S-1-5-32-544"


def _win32():
    if os.name != "nt":
        raise DirectoryAclError("OWNER_CHECK")

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    advapi = ct.WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wt.HANDLE
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.LocalFree.argtypes = [ct.c_void_p]
    kernel.LocalFree.restype = ct.c_void_p
    advapi.OpenProcessToken.argtypes = [wt.HANDLE, wt.DWORD, ct.POINTER(wt.HANDLE)]
    advapi.OpenProcessToken.restype = wt.BOOL
    advapi.GetTokenInformation.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.POINTER(wt.DWORD)]
    advapi.GetTokenInformation.restype = wt.BOOL
    advapi.GetNamedSecurityInfoW.argtypes = [
        wt.LPWSTR, ct.c_int, wt.DWORD, ct.POINTER(ct.c_void_p), ct.c_void_p,
        ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p),
    ]
    advapi.GetNamedSecurityInfoW.restype = wt.DWORD
    advapi.ConvertSidToStringSidW.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    advapi.ConvertSidToStringSidW.restype = wt.BOOL
    return kernel, advapi


def _sid_text(kernel, advapi, sid):
    text = ct.c_void_p()

    if not advapi.ConvertSidToStringSidW(sid, ct.byref(text)):
        raise DirectoryAclError("OWNER_CHECK", ct.get_last_error())

    try:
        return ct.wstring_at(text)
    finally:
        kernel.LocalFree(text)


def _token_identity():
    """현재 프로세스 토큰의 사용자 SID·승격 여부·기본 소유자 SID를 조회한다. 토큰·특권은 바꾸지 않는다."""
    class SidAndAttributes(ct.Structure):
        _fields_ = [("sid", ct.c_void_p), ("attributes", wt.DWORD)]

    kernel, advapi = _win32()
    token = wt.HANDLE()

    try:
        # TOKEN_QUERY / TokenUser. 첫 호출은 필요한 버퍼 크기만 조회한다.
        if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ct.byref(token)):
            raise DirectoryAclError("OWNER_CHECK", ct.get_last_error())

        needed = wt.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ct.byref(needed))

        if ct.get_last_error() != 122 or not 0 < needed.value <= 65536:
            raise DirectoryAclError("OWNER_CHECK")

        buffer = ct.create_string_buffer(needed.value)

        if not advapi.GetTokenInformation(token, 1, buffer, needed.value, ct.byref(needed)):
            raise DirectoryAclError("OWNER_CHECK", ct.get_last_error())

        user = _sid_text(kernel, advapi, ct.cast(buffer, ct.POINTER(SidAndAttributes)).contents.sid)
        # TokenOwner: 이 프로세스가 새로 만드는 객체의 기본 소유자. TOKEN_OWNER는 SID 포인터 하나다.
        advapi.GetTokenInformation(token, 4, None, 0, ct.byref(needed))

        if ct.get_last_error() != 122 or not 0 < needed.value <= 65536:
            raise DirectoryAclError("OWNER_CHECK")

        owner_buffer = ct.create_string_buffer(needed.value)

        if not advapi.GetTokenInformation(token, 4, owner_buffer, needed.value, ct.byref(needed)):
            raise DirectoryAclError("OWNER_CHECK", ct.get_last_error())

        default_owner = _sid_text(kernel, advapi, ct.cast(owner_buffer, ct.POINTER(ct.c_void_p)).contents.value)
        # TokenElevation: 환경 변수·창 제목이 아니라 토큰의 실제 승격 여부만 사용한다.
        elevation = wt.DWORD()

        if not advapi.GetTokenInformation(token, 20, ct.byref(elevation), ct.sizeof(elevation), ct.byref(needed)):
            raise DirectoryAclError("OWNER_CHECK", ct.get_last_error())

        return user, elevation.value != 0, default_owner
    finally:
        if token:
            kernel.CloseHandle(token)


def _directory_owner(directory):
    """폴더 소유자 SID만 조회한다. DACL·SACL·소유자는 변경하지 않는다."""
    kernel, advapi = _win32()
    owner = ct.c_void_p()
    descriptor = ct.c_void_p()

    try:
        # SE_FILE_OBJECT / OWNER_SECURITY_INFORMATION
        result = advapi.GetNamedSecurityInfoW(
            str(directory), 1, 1, ct.byref(owner), None, None, None, ct.byref(descriptor),
        )

        if result:
            raise DirectoryAclError("OWNER_CHECK", result)

        if not owner:
            raise DirectoryAclError("OWNER_MISMATCH")

        return _sid_text(kernel, advapi, owner)
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)


def _owner_user_sid(directory):
    """새 작업 폴더의 소유자가 허용 대상일 때만 현재 사용자 SID를 반환한다.

    허용 대상은 두 가지뿐이다.
    - 소유자가 현재 프로세스 사용자 SID와 같다.
    - 다음을 모두 만족한다: 토큰이 실제로 승격됐고, 토큰의 기본 소유자가 Administrators 그룹(S-1-5-32-544)이며,
      폴더 소유자도 그 그룹이고, 사용자가 일반 계정 SID(S-1-5-21-)다. 이 프로세스가 방금 만든 새 폴더가
      승격 토큰의 기본 소유자로 만들어진 경우만 해당한다. SYSTEM·서비스 계정 실행은 이 예외에 포함하지 않는다.
    그 밖의 사용자·그룹 소유는 거부한다. 반환값은 항상 현재 사용자 SID이며 그룹 SID가 아니다.
    """
    user, elevated, default_owner = _token_identity()
    owner = _directory_owner(directory)

    if owner == user or (elevated is True and user.startswith("S-1-5-21-")
                         and default_owner == ADMINISTRATORS_SID and owner == ADMINISTRATORS_SID):
        return user

    raise DirectoryAclError("OWNER_MISMATCH")


def prepare_writable_directory(root, directory, container_sid):
    """검증한 빈 작업 폴더에만 호출자 WO와 worker M·Low를 설정한다.

    WRITE_OWNER는 Low 레이블 설정에 필요하다. 호출자 자신의 기존 권한에만
    추가하고 상속하지 않는다. worker에는 WO·FullControl을 부여하지 않는다.
    실패 시 다른 경로나 관리자 실행으로 자동 우회하지 않는다.
    """
    root = _root(root)
    directory = _root(directory)
    allowed = {root / "output" / "result", root / "scratch" / "task"}

    if directory not in allowed or any(directory.iterdir()):
        raise DirectoryAclError("TARGET_CHECK")

    caller_sid = _owner_user_sid(directory)

    if caller_sid == container_sid:
        raise DirectoryAclError("OWNER_MISMATCH")

    executable = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
    # /grant는 기존 명시 권한을 보존한다. /T·상속 플래그·소유권 변경은 사용하지 않는다.
    result = subprocess.run(
        [str(executable), str(directory), "/grant", f"*{caller_sid}:(WO)", "/Q"],
        capture_output=True, timeout=30, check=False,
    )

    if result.returncode:
        raise DirectoryAclError("CALLER_WRITE_OWNER", result.returncode)

    try:
        _acl(directory, container_sid, "M", low=True)
    except OSError as error:
        # 고정된 내부 오류 이름만 판별한다. 경로·현지화된 명령 출력은 기록하지 않는다.
        operation = {
            "PROBE_ACL_SETUP_FAILED": "WORKER_GRANT",
            "PROBE_INTEGRITY_SETUP_FAILED": "LOW_INTEGRITY",
        }.get(error.strerror, "WORKER_ACL")
        raise DirectoryAclError(operation, error.errno) from error

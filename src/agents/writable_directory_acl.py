"""새 작업의 빈 출력·임시 폴더만 일반 사용자 권한으로 준비한다."""

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


def _owner_user_sid(directory):
    """폴더 소유자와 현재 프로세스 사용자가 같을 때만 사용자 SID를 반환한다.

    관리자 그룹이나 환경 변수의 사용자 이름을 신뢰하지 않는다. 토큰의 실제
    사용자 SID를 조회하며, 조회만 수행하고 토큰 특권·폴더 소유자는 변경하지 않는다.
    """
    if os.name != "nt":
        raise DirectoryAclError("OWNER_CHECK")

    class SidAndAttributes(ct.Structure):
        _fields_ = [("sid", ct.c_void_p), ("attributes", wt.DWORD)]

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
    advapi.EqualSid.argtypes = [ct.c_void_p, ct.c_void_p]
    advapi.EqualSid.restype = wt.BOOL
    advapi.ConvertSidToStringSidW.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    advapi.ConvertSidToStringSidW.restype = wt.BOOL

    token = wt.HANDLE()
    descriptor = ct.c_void_p()
    text = ct.c_void_p()

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

        user = ct.cast(buffer, ct.POINTER(SidAndAttributes)).contents.sid
        owner = ct.c_void_p()
        # SE_FILE_OBJECT / OWNER_SECURITY_INFORMATION. DACL·SACL 자체는 변경하지 않는다.
        result = advapi.GetNamedSecurityInfoW(
            str(directory), 1, 1, ct.byref(owner), None, None, None, ct.byref(descriptor),
        )

        if result:
            raise DirectoryAclError("OWNER_CHECK", result)

        if not owner or not advapi.EqualSid(owner, user):
            raise DirectoryAclError("OWNER_MISMATCH")

        if not advapi.ConvertSidToStringSidW(user, ct.byref(text)):
            raise DirectoryAclError("OWNER_CHECK", ct.get_last_error())

        return ct.wstring_at(text)
    finally:
        if text:
            kernel.LocalFree(text)

        if descriptor:
            kernel.LocalFree(descriptor)

        if token:
            kernel.CloseHandle(token)


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

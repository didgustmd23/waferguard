"""부모가 관리하는 보호 저장소의 생성·보안 조회·의미상 권한 판정·경로 보유.

신뢰 주체는 현재 사용자·SYSTEM·Administrators다. 보호 폴더는 생성 순간부터 상속을 끊은 DACL을 가지며,
기존 폴더의 권한을 고치지 않는다. 신뢰 주체의 변경은 막지 않는다. 공유 런타임과 입력 실행 저장소가 함께 쓴다.
"""

from contextlib import contextmanager
import ctypes as ct
from ctypes import wintypes as wt
import os
from pathlib import Path
import re
import shutil
import uuid

from .artifacts import _root
from .writable_directory_acl import _token_identity


# 현재 사용자 외에 저장소 작성·소유를 허용하는 OS 신뢰 주체: SYSTEM, Administrators.
TRUSTED = ("S-1-5-18", "S-1-5-32-544")
# capability에 허용하는 마스크: FILE_GENERIC_READ|FILE_GENERIC_EXECUTE(0x1200A9)와 GENERIC_READ·GENERIC_EXECUTE.
READ_EXECUTE = 0x1200A9 | 0x80000000 | 0x20000000


class StoreUnavailable(ValueError):
    """고정 코드만 담는다. 경로·원문 예외는 넘기지 않는다. 기존 ValueError 처리 경로와 호환된다."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _advapi():
    advapi = ct.WinDLL("advapi32", use_last_error=True)
    advapi.GetNamedSecurityInfoW.argtypes = [wt.LPCWSTR, ct.c_int, wt.DWORD, ct.POINTER(ct.c_void_p), ct.c_void_p,
                                             ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    advapi.GetNamedSecurityInfoW.restype = wt.DWORD
    advapi.GetSecurityInfo.argtypes = [wt.HANDLE, ct.c_int, wt.DWORD, ct.POINTER(ct.c_void_p), ct.c_void_p,
                                       ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    advapi.GetSecurityInfo.restype = wt.DWORD
    advapi.GetSecurityDescriptorControl.argtypes = [ct.c_void_p, ct.POINTER(wt.WORD), ct.POINTER(wt.DWORD)]
    advapi.GetSecurityDescriptorControl.restype = wt.BOOL
    advapi.GetAce.argtypes = [ct.c_void_p, wt.DWORD, ct.POINTER(ct.c_void_p)]
    advapi.GetAce.restype = wt.BOOL
    advapi.ConvertSidToStringSidW.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    advapi.ConvertSidToStringSidW.restype = wt.BOOL
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ct.c_void_p]
    kernel.LocalFree.restype = ct.c_void_p
    return advapi, kernel


def _sid_text(advapi, kernel, sid):
    text = ct.c_void_p()

    if not advapi.ConvertSidToStringSidW(sid, ct.byref(text)):
        raise OSError(ct.get_last_error(), "RUNTIME_SECURITY_QUERY_FAILED")

    try:
        return ct.wstring_at(text)
    finally:
        kernel.LocalFree(text)


def _security(path=None, handle=None):
    """소유자·DACL 보호 여부·ACE 목록·무결성 레이블을 읽는다. 변경하지 않는다.

    ACE는 (형식, 플래그, 마스크, SID) 튜플이다. 허용(0)·거부(1)·레이블(0x11) 외 형식은 형식 번호만 남기고 검사에서 거부한다.
    """
    advapi, kernel = _advapi()
    owner, dacl, sacl, descriptor = ct.c_void_p(), ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    # OWNER | DACL | LABEL. 레이블 읽기는 READ_CONTROL만 필요하다.
    information = 0x1 | 0x4 | 0x10

    if handle is not None:
        result = advapi.GetSecurityInfo(handle, 1, information, ct.byref(owner), None, ct.byref(dacl), ct.byref(sacl),
                                        ct.byref(descriptor))
    else:
        result = advapi.GetNamedSecurityInfoW(str(path), 1, information, ct.byref(owner), None, ct.byref(dacl), ct.byref(sacl),
                                              ct.byref(descriptor))

    if result:
        raise OSError(result, "RUNTIME_SECURITY_QUERY_FAILED")

    try:
        control, revision = wt.WORD(), wt.DWORD()

        if not advapi.GetSecurityDescriptorControl(descriptor, ct.byref(control), ct.byref(revision)):
            raise OSError(ct.get_last_error(), "RUNTIME_SECURITY_QUERY_FAILED")

        def entries(acl):
            if not acl:
                return None

            items = []

            # ACL 헤더: 개정(1)·예약(1)·크기(2)·ACE 수(2)·예약(2)
            for index in range(ct.cast(acl, ct.POINTER(wt.WORD))[2]):
                ace = ct.c_void_p()

                if not advapi.GetAce(acl, index, ct.byref(ace)):
                    raise OSError(ct.get_last_error(), "RUNTIME_SECURITY_QUERY_FAILED")

                header = ct.cast(ace, ct.POINTER(ct.c_ubyte))

                if header[0] not in (0, 1, 0x11):
                    items.append((header[0], header[1], None, None))
                    continue

                # 허용·거부·레이블 ACE는 헤더(4)·마스크(4) 뒤에 SID가 온다.
                mask = ct.cast(ace.value + 4, ct.POINTER(wt.DWORD))[0]
                items.append((header[0], header[1], mask, _sid_text(advapi, kernel, ct.c_void_p(ace.value + 8))))

            return items

        label = None

        for kind, _, _, sid in entries(sacl) or []:
            if kind == 0x11 and sid and sid.startswith("S-1-16-"):
                label = int(sid.rsplit("-", 1)[1])

        return {"owner": _sid_text(advapi, kernel, owner) if owner else None,
                # SE_DACL_PRESENT(0x4)가 없거나 DACL 포인터가 없으면 NULL DACL(모두 허용)로 보고 거부한다.
                "dacl": entries(dacl) if control.value & 0x4 else None,
                "protected": bool(control.value & 0x1000), "label": label}
    finally:
        kernel.LocalFree(descriptor)


def _permission_ok(info, trusted, capability, *, root):
    """의미상 권한 검사. 신뢰 주체 외 SID의 허용 ACE는 모두 거부하고 capability는 읽기·실행 이하만 허용한다.

    거부 ACE는 권한을 줄이기만 하므로 통과시킨다. root(저장소·버전 폴더)는 상속을 끊은 DACL이어야 한다.
    Medium(0x2000) 미만 무결성 레이블은 Low 토큰 worker의 쓰기 경로가 될 수 있어 거부한다.
    """
    if info["owner"] not in trusted or info["dacl"] is None or (root and not info["protected"]):
        return False

    if info["label"] is not None and info["label"] < 0x2000:
        return False

    for kind, _, mask, sid in info["dacl"]:
        if kind == 1:
            continue

        # OWNER RIGHTS(S-1-3-4)는 현재 소유자에게만 적용되며 소유자는 위에서 신뢰 주체로 확인했다.
        if kind != 0 or not (sid in trusted or sid == "S-1-3-4" or (sid == capability and mask & ~READ_EXECUTE == 0)):
            return False

    return True


def _create_protected(path, user, capability=None):
    """새 폴더를 생성 순간부터 상속을 끊은 보호 DACL로 만든다. 생성 뒤 권한을 바꾸는 틈이 없다.

    신뢰 주체(현재 사용자·SYSTEM·Administrators)는 FILE_ALL_ACCESS, capability는 읽기·실행만 하위에 상속된다.
    이미 있으면 FileExistsError. 기존·상위 폴더 ACL은 바꾸지 않는다.
    """
    advapi = ct.WinDLL("advapi32", use_last_error=True)
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [wt.LPCWSTR, wt.DWORD, ct.POINTER(ct.c_void_p),
                                                                            ct.c_void_p]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wt.BOOL
    kernel.CreateDirectoryW.argtypes = [wt.LPCWSTR, ct.c_void_p]
    kernel.CreateDirectoryW.restype = wt.BOOL
    kernel.LocalFree.argtypes = [ct.c_void_p]
    kernel.LocalFree.restype = ct.c_void_p

    class SecurityAttributes(ct.Structure):
        _fields_ = [("length", wt.DWORD), ("descriptor", ct.c_void_p), ("inherit", wt.BOOL)]

    for sid in (user, capability):
        if sid is not None and re.fullmatch(r"S-1-[0-9]+(-[0-9]+)+", sid) is None:
            raise OSError("RUNTIME_ACL_SETUP_FAILED")

    # D:P = 보호(상속 차단) DACL. OICI = 하위 폴더·파일 상속. FA = FILE_ALL_ACCESS, 0x1200a9 = 읽기·실행.
    aces = "".join(f"(A;OICI;FA;;;{sid})" for sid in (user, *TRUSTED))
    sddl = "D:P" + aces + (f"(A;OICI;0x1200a9;;;{capability})" if capability else "")
    descriptor = ct.c_void_p()

    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ct.byref(descriptor), None):
        raise OSError(ct.get_last_error(), "RUNTIME_ACL_SETUP_FAILED")

    try:
        attributes = SecurityAttributes(ct.sizeof(SecurityAttributes), descriptor, False)

        if not kernel.CreateDirectoryW(str(path), ct.byref(attributes)):
            error = ct.get_last_error()
            raise FileExistsError(error, "RUNTIME_EXISTS") if error == 183 else OSError(error, "RUNTIME_ACL_SETUP_FAILED")
    finally:
        kernel.LocalFree(descriptor)


def _hold(directory):
    """목록 읽기·보안 읽기, 삭제 공유 없음으로 폴더를 연다. 보유 중에는 이 폴더와 상위 경로의 이름 변경·삭제가 막힌다."""
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
    kernel.CreateFileW.restype = wt.HANDLE
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    # FILE_LIST_DIRECTORY|FILE_READ_ATTRIBUTES|READ_CONTROL, FILE_SHARE_READ|WRITE, OPEN_EXISTING,
    # FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT(링크를 따라가지 않음)
    handle = kernel.CreateFileW(str(directory), 0x1 | 0x80 | 0x20000, 0x1 | 0x2, None, 3, 0x02000000 | 0x00200000, None)

    if handle in (None, wt.HANDLE(-1).value):
        raise OSError(ct.get_last_error(), "RUNTIME_HOLD_FAILED")

    return handle, lambda: kernel.CloseHandle(handle)


def trusted_principals():
    """현재 사용자·SYSTEM·Administrators SID. 첫 항목이 현재 사용자다."""
    return (_token_identity()[0], *TRUSTED)


def create_store(path, trusted):
    """저장소 루트를 보호 SD의 임시 폴더로 만든 뒤 rename으로 공개한다. 이미 있으면 그대로 둔다(검사는 store_lease)."""
    path = Path(path)

    if os.path.lexists(path):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    # 상위 경로에 링크가 있으면 다른 위치에 만들지 않는다(ValueError).
    parent = _root(path.parent)
    staging = parent / (".store-" + uuid.uuid4().hex)
    published = False

    try:
        _create_protected(staging, trusted[0])
        os.rename(staging, parent / path.name)
        published = True
    except OSError:
        # 다른 실행이 먼저 공개했으면 그 루트를 검사 단계에서 본다.
        if not os.path.lexists(path):
            raise
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


@contextmanager
def store_lease(path, trusted, *, capability=None):
    """저장소 루트를 보유한 뒤 그 핸들로 소유자·보호·ACE를 검사하고, with 블록이 끝나면 놓는다.

    보유 중에는 루트와 상위 경로의 이름 변경·삭제가 막힌다. 검사 실패는 STORE_PERMISSION_INVALID이며 권한을 고치지 않는다.
    """
    try:
        path = _root(path)
        handle, release = _hold(path)
    except (OSError, ValueError):
        raise StoreUnavailable("STORE_PERMISSION_INVALID") from None

    try:
        try:
            # 보유 뒤에는 상위 경로가 고정된다. 이 시점에 링크가 없어야 경로와 보유 객체가 결합된다.
            _root(path)
            allowed = _permission_ok(_security(handle=handle), trusted, capability, root=True)
        except Exception:
            allowed = False

        if not allowed:
            raise StoreUnavailable("STORE_PERMISSION_INVALID")

        yield path
    finally:
        release()


def check_object(path, trusted):
    """폴더·파일 하나의 소유자·ACE·레이블을 신뢰 주체 기준으로 검사한다(하위는 보지 않는다)."""
    from .artifacts import _is_link

    try:
        path = Path(path)
        if _is_link(path) or (path.is_file() and os.lstat(path).st_nlink != 1):
            return False
        return _permission_ok(_security(path), trusted, None, root=False)
    except (OSError, ValueError, TypeError):
        return False


def check_tree(path, trusted, *, extra=None, low_allowed=False):
    """폴더 트리의 모든 객체를 의미상 권한으로 검사한다. 링크·재분석 지점·하드 링크도 거부한다.

    extra는 {SID: 허용 최대 마스크}로, 신뢰 주체 외에 허용할 작업 SID 권한이다. low_allowed는 worker 쓰기 영역의
    Low 레이블 허용 여부다. 하나라도 다르면 False이며 아무것도 고치지 않는다.
    """
    from .artifacts import _is_link
    extra = extra or {}
    path = Path(path)

    def allowed(item):
        info = _security(item)

        if info["owner"] not in trusted or info["dacl"] is None:
            return False

        if info["label"] is not None and info["label"] < 0x2000 and not (low_allowed and info["label"] >= 0x1000):
            return False

        for kind, _, mask, sid in info["dacl"]:
            if kind == 1:
                continue

            if kind != 0 or not (sid in trusted or sid == "S-1-3-4" or (sid in extra and mask & ~extra[sid] == 0)):
                return False

        return True

    try:
        if _is_link(path) or not allowed(path):
            return False

        def failed(error):
            raise error

        # 목록 오류를 무시하면 그 하위를 검사하지 않고 통과할 수 있으므로 실패로 처리한다.
        for current, folders, names in os.walk(path, followlinks=False, onerror=failed):
            for name in folders + names:
                item = Path(current) / name

                if _is_link(item) or (item.is_file() and os.lstat(item).st_nlink != 1) or not allowed(item):
                    return False

        return True
    except (OSError, ValueError, TypeError):
        return False


"""격리 worker가 함께 쓰는 Python 런타임의 준비·식별·재사용.

원본 환경은 읽기만 한다. 구성별 버전 폴더를 상속을 끊은 보호 DACL의 임시 폴더에서 복사·검증한 뒤 rename으로
한 번만 공개하고, 공개한 버전은 덮어쓰거나 권한을 고치거나 자동 삭제하지 않는다. 실행은 승인에 고정한 ID의
버전만 쓰며, 검증부터 worker 종료까지 버전 폴더 핸들을 보유해 경로 교체를 막는다(lease). 해시·완료 기록·ACL은
서명된 무결성이 아니며, 신뢰 주체(현재 사용자·SYSTEM·Administrators)의 변경은 막지 않는다.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from .appcontainer_probe import runtime_files
from .artifacts import _is_link, _root
from .protected_store import READ_EXECUTE, TRUSTED, _create_protected, _hold, _permission_ok, _security
from .writable_directory_acl import _token_identity


SCHEMA = "worker_runtime_v1"
RECORD = "manifest.json"
# 공유 런타임 RX는 작업별 package SID가 아니라 이 이름에서 파생한 capability SID에만 준다.
CAPABILITY = "waferguardWorkerRuntime"
DEFAULT_BASE = Path(__file__).resolve().parents[2] / "logs/agents/worker_runtime"
MESSAGES = {
    "RUNTIME_SOURCE_INVALID": "현재 Python 환경에서 격리 실행용 파일 목록을 만들지 못했습니다(누락·링크·복사 상한 512MiB 초과). "
                              "앱을 실행한 Python 환경을 확인한 뒤 새 작업으로 다시 시작하세요.",
    "RUNTIME_PREPARE_FAILED": "격리 실행용 Python 런타임을 준비하지 못했습니다(복사·권한 설정 실패). 디스크 공간과 "
                              "logs/agents/worker_runtime 폴더 권한을 확인한 뒤 새 작업으로 다시 시작하세요.",
    "RUNTIME_VERIFY_FAILED": "준비된 격리 실행용 Python 런타임이 완료 기록과 다릅니다(변경·누락·추가 파일 또는 링크). "
                             "자동으로 덮어쓰거나 지우지 않습니다. 관리자가 logs/agents/worker_runtime의 해당 버전 폴더를 "
                             "다른 곳으로 옮긴 뒤 새 작업으로 다시 시작하세요.",
    "RUNTIME_PERMISSION_INVALID": "격리 실행용 Python 런타임 저장소의 소유자·권한이 허용 범위와 다릅니다(다른 사용자·그룹의 "
                                  "쓰기·삭제·권한 변경, 상속 권한, 낮은 무결성 레이블 등). 권한을 자동으로 고치지 않습니다. "
                                  "다른 Windows 계정이 만든 저장소도 이 경우에 해당합니다. 관리자가 logs/agents/worker_runtime의 "
                                  "해당 폴더를 확인해 다른 곳으로 옮긴 뒤 새 작업으로 다시 시작하세요.",
    "RUNTIME_CHANGED": "승인한 격리 실행용 Python 런타임이 준비돼 있지 않고, 현재 Python 환경 구성이 승인 때와 다릅니다. "
                       "같은 승인으로 다른 런타임을 고르지 않습니다. 새 작업으로 다시 시작해 현재 환경으로 실행·승인을 새로 "
                       "준비하세요.",
}
NOTICE = "원본 Python 환경으로 대신 실행하지 않았고 자동 재시도하지 않습니다."


class RuntimeUnavailable(Exception):
    """고정 코드만 담는다. 경로·원문 예외는 호출자에게 넘기지 않는다."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SharedRuntime:
    runtime_id: str
    directory: Path
    prepared: bool
    timings: dict
    capability: str

    @property
    def python(self):
        return self.directory / "runtime" / "python.exe"


def capability_sid():
    """고정 capability 이름의 SID 문자열. 시스템 상태는 바꾸지 않는다."""
    kernel = ct.WinDLL("kernelbase", use_last_error=True)
    advapi = ct.WinDLL("advapi32", use_last_error=True)
    local = ct.WinDLL("kernel32", use_last_error=True)
    derive = kernel.DeriveCapabilitySidsFromName
    derive.argtypes = [wt.LPCWSTR, ct.POINTER(ct.c_void_p), ct.POINTER(wt.DWORD), ct.POINTER(ct.c_void_p), ct.POINTER(wt.DWORD)]
    derive.restype = wt.BOOL
    advapi.ConvertSidToStringSidW.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    advapi.ConvertSidToStringSidW.restype = wt.BOOL
    local.LocalFree.argtypes = [ct.c_void_p]
    local.LocalFree.restype = ct.c_void_p
    groups, group_count, capabilities, capability_count = ct.c_void_p(), wt.DWORD(), ct.c_void_p(), wt.DWORD()

    if not derive(CAPABILITY, ct.byref(groups), ct.byref(group_count), ct.byref(capabilities), ct.byref(capability_count)):
        raise OSError(ct.get_last_error(), "RUNTIME_CAPABILITY_FAILED")

    try:
        if capability_count.value != 1:
            raise OSError("RUNTIME_CAPABILITY_FAILED")

        text = ct.c_void_p()

        if not advapi.ConvertSidToStringSidW(ct.cast(capabilities, ct.POINTER(ct.c_void_p))[0], ct.byref(text)):
            raise OSError(ct.get_last_error(), "RUNTIME_CAPABILITY_FAILED")

        try:
            return ct.wstring_at(text)
        finally:
            local.LocalFree(text)
    finally:
        # 반환된 SID 배열은 각 SID와 배열을 LocalFree로 해제한다.
        for array, count in ((groups, group_count.value), (capabilities, capability_count.value)):
            if array:
                for index in range(count):
                    local.LocalFree(ct.cast(array, ct.POINTER(ct.c_void_p))[index])
                local.LocalFree(array)


def _sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)

    return digest.hexdigest()


def _hashes(paths):
    # hashlib은 큰 블록에서 GIL을 놓으므로 읽기·해시를 스레드로 겹친다. 결과 순서는 입력 순서를 따른다.
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_sha256, paths))


def _inspect_file(path, trusted, capability):
    """같은 핸들로 보안 정보와 내용 해시를 읽는다. 권한이 맞지 않으면 해시 대신 None."""
    import msvcrt
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        if not _permission_ok(_security(handle=msvcrt.get_osfhandle(stream.fileno())), trusted, capability, root=False):
            return None

        while chunk := stream.read(1 << 20):
            digest.update(chunk)

    return digest.hexdigest()


def _relative(path, prefix):
    return path.relative_to(prefix).as_posix()


def source_manifest(prefix):
    """runtime_files() 허용 목록 그대로 원본 파일 구성을 내용 해시로 기록한다. 원본은 읽기만 한다."""
    prefix = Path(prefix)
    files = runtime_files(prefix)
    sizes = [path.stat().st_size for path in files]
    entries = [[_relative(path, prefix), size, digest] for path, size, digest in zip(files, sizes, _hashes(files))]
    return {"schema": SCHEMA, "python": ".".join(map(str, sys.version_info[:3])),
            "cache_tag": sys.implementation.cache_tag, "files": sorted(entries)}


def runtime_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def current_runtime_id(prefix=None):
    """승인에 고정할 현재 원본 구성의 런타임 ID. 원본 전체를 해시하며 아무것도 복사하지 않는다."""
    try:
        return runtime_id(source_manifest(Path(sys.prefix if prefix is None else prefix)))
    except (OSError, ValueError):
        raise RuntimeUnavailable("RUNTIME_SOURCE_INVALID") from None


def verify_runtime(directory, expected_id, *, capability, trusted, handle=None):
    """공개 버전을 승인 ID의 완료 기록·내용·구조·소유자·권한으로 검증한다. 고치지 않고 고정 코드나 None을 돌려준다.

    완료 기록은 그 내용의 해시가 승인 ID와 같을 때만 기준으로 쓴다. 파일 권한은 해시를 읽는 같은 핸들로 확인한다.
    handle을 주면 버전 폴더 자체의 보안은 보유 중인 그 핸들로 조회한다(검사한 객체와 보유한 객체가 같다).
    """
    import msvcrt
    directory = Path(directory)

    try:
        if (type(expected_id) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_id) is None or directory.name != expected_id
                or any(_is_link(item) for item in (directory, *directory.parents))):
            return "RUNTIME_VERIFY_FAILED"

        if not _permission_ok(_security(directory) if handle is None else _security(handle=handle), trusted, capability, root=True):
            return "RUNTIME_PERMISSION_INVALID"

        with open(directory / RECORD, "rb") as stream:
            if not _permission_ok(_security(handle=msvcrt.get_osfhandle(stream.fileno())), trusted, capability, root=False):
                return "RUNTIME_PERMISSION_INVALID"

            record = json.loads(stream.read(64 * 1024 * 1024 + 1))

        manifest = {key: value for key, value in record.items() if key != "runtime_id"} if type(record) is dict else None

        if (manifest is None or record.get("runtime_id") != expected_id or runtime_id(manifest) != expected_id
                or manifest.get("schema") != SCHEMA or type(manifest.get("files")) is not list):
            return "RUNTIME_VERIFY_FAILED"

        runtime = directory / "runtime"
        expected = {name: (size, digest) for name, size, digest in manifest["files"]}
        # 목록 파일의 상위 폴더만 허용한다. 빈 폴더·추가 폴더도 다른 구성으로 본다.
        folders = {parent.as_posix() for name in expected for parent in Path(name).parents}
        found = []

        if {item.name for item in directory.iterdir()} != {RECORD, "runtime"} or _is_link(runtime) or not runtime.is_dir():
            return "RUNTIME_VERIFY_FAILED"

        for current, subfolders, names in os.walk(runtime, followlinks=False):
            relative = Path(current).relative_to(runtime).as_posix()

            if relative not in folders or _is_link(Path(current)):
                return "RUNTIME_VERIFY_FAILED"

            # 하위 폴더도 소유자·권한을 따로 본다. 폴더에만 다른 권한을 넣은 경우를 거부한다.
            if not _permission_ok(_security(Path(current)), trusted, capability, root=False):
                return "RUNTIME_PERMISSION_INVALID"

            for name in subfolders:
                if _is_link(Path(current) / name):
                    return "RUNTIME_VERIFY_FAILED"

            found.extend(Path(current) / name for name in names)

        names = [_relative(path, runtime) for path in found]

        if sorted(names) != sorted(expected):
            return "RUNTIME_VERIFY_FAILED"

        for path, name in zip(found, names):
            info = os.lstat(path)

            if _is_link(path) or info.st_nlink != 1 or info.st_size != expected[name][0]:
                return "RUNTIME_VERIFY_FAILED"

        with ThreadPoolExecutor(max_workers=8) as pool:
            digests = list(pool.map(lambda path: _inspect_file(path, trusted, capability), found))

        if any(digest is None for digest in digests):
            return "RUNTIME_PERMISSION_INVALID"

        return None if all(digest == expected[name][1] for name, digest in zip(names, digests)) else "RUNTIME_VERIFY_FAILED"
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return "RUNTIME_VERIFY_FAILED"


def _prepare(base, prefix, manifest, identifier, cancel_event, sid, user):
    """보호 DACL의 새 staging에 복사·검증·기록한 뒤 rename으로 공개한다. 이미 공개돼 있으면 False(패배)를 돌려준다."""
    staging = base / (".staging-" + uuid.uuid4().hex)

    try:
        # 생성 순간부터 신뢰 주체 F·capability RX만 두는 보호 DACL. 하위 폴더·파일은 이 ACE만 상속한다.
        _create_protected(staging, user, sid)
        runtime = staging / "runtime"

        for name, _, _ in manifest["files"]:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeUnavailable("RUNTIME_CANCELLED")

            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prefix / name, target)

        (staging / RECORD).write_text(json.dumps(dict(manifest, runtime_id=identifier), sort_keys=True), encoding="utf-8")
        # staging 이름으로는 검증하지 않는다. 복사본 내용만 원본 manifest와 대조한다.
        copied = [runtime / name for name, _, _ in manifest["files"]]

        if [digest for _, _, digest in manifest["files"]] != _hashes(copied):
            raise RuntimeUnavailable("RUNTIME_PREPARE_FAILED")

        try:
            os.rename(staging, base / identifier)
        except FileExistsError:
            return False
        except OSError:
            if (base / identifier).exists():
                return False
            raise

        return True
    except RuntimeUnavailable:
        raise
    except OSError:
        # 원본 환경으로 대신 실행하지 않는다.
        raise RuntimeUnavailable("RUNTIME_PREPARE_FAILED") from None
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _base(base, trusted, capability):
    """저장소 루트를 보유한 뒤 그 핸들로 소유자·권한을 확인한다. 처음이면 보호 SD로 만들어 rename으로 공개한다.

    반환한 해제 함수를 부를 때까지 루트와 상위 경로의 이름 변경이 막힌다. 기존 권한을 고치지 않는다.
    """
    base = Path(base)

    try:
        if not os.path.lexists(base):
            base.parent.mkdir(parents=True, exist_ok=True)
            staging = base.parent / (".worker_runtime-" + uuid.uuid4().hex)

            try:
                _create_protected(staging, trusted[0])
                os.rename(staging, base)
            except OSError:
                # 다른 실행이 먼저 공개했으면 그 루트를 아래에서 보유·검사한다.
                if not os.path.lexists(base):
                    raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

        base = _root(base)
    except (OSError, ValueError):
        raise RuntimeUnavailable("RUNTIME_PREPARE_FAILED") from None

    try:
        handle, release = _hold(base)
    except OSError:
        raise RuntimeUnavailable("RUNTIME_PERMISSION_INVALID") from None

    try:
        # 보유 뒤 상위 경로가 고정된 상태에서 링크를 다시 확인해 경로와 보유 객체를 결합한다.
        _root(base)
        allowed = _permission_ok(_security(handle=handle), trusted, capability, root=True)
    except Exception:
        # 어떤 조회 실패도 보유를 남기지 않고 권한 이상으로 멈춘다.
        allowed = False

    if not allowed:
        release()
        raise RuntimeUnavailable("RUNTIME_PERMISSION_INVALID")

    return base, release


@contextmanager
def runtime_lease(expected_id, *, prefix=None, base=None, cancel_event=None):
    """승인 ID의 공개 버전을 보유·검증해 넘기고, with 블록이 끝날 때(worker 종료 후) 놓는다.

    공개본이 없으면 현재 원본 구성이 승인 ID와 같을 때만 준비한다. 다르면 RUNTIME_CHANGED로 멈추며 다른 ID를 고르지 않는다.
    """
    if type(expected_id) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_id) is None:
        raise RuntimeUnavailable("RUNTIME_VERIFY_FAILED")

    try:
        # 준비·실행에 같은 값을 쓴다. 함수가 없는 Windows(AttributeError)도 준비 실패로 분류한다.
        sid = capability_sid()
        trusted = (_token_identity()[0], *TRUSTED)
    except (OSError, AttributeError):
        raise RuntimeUnavailable("RUNTIME_PREPARE_FAILED") from None

    prefix = Path(sys.prefix if prefix is None else prefix)
    # 루트는 준비·검증·worker 종료까지 보유한다. 그동안 상위 경로를 다른 폴더로 바꿔치기할 수 없다.
    base, release_base = _base(DEFAULT_BASE if base is None else base, trusted, sid)

    try:
        with _version_lease(base, expected_id, prefix, cancel_event, sid, trusted) as selected:
            yield selected
    finally:
        release_base()


@contextmanager
def _version_lease(base, expected_id, prefix, cancel_event, sid, trusted):
    directory = base / expected_id
    timings, prepared = {}, False

    if not os.path.lexists(directory):
        mark = time.perf_counter()

        try:
            manifest = source_manifest(prefix)
        except (OSError, ValueError):
            raise RuntimeUnavailable("RUNTIME_SOURCE_INVALID") from None

        timings["source_manifest_seconds"] = round(time.perf_counter() - mark, 3)

        if runtime_id(manifest) != expected_id:
            raise RuntimeUnavailable("RUNTIME_CHANGED")

        mark = time.perf_counter()
        prepared = _prepare(base, prefix, manifest, expected_id, cancel_event, sid, trusted[0])
        timings["prepare_seconds"] = round(time.perf_counter() - mark, 3)

    try:
        handle, release = _hold(directory)
    except OSError:
        raise RuntimeUnavailable("RUNTIME_VERIFY_FAILED") from None

    try:
        mark = time.perf_counter()
        code = verify_runtime(directory, expected_id, capability=sid, trusted=trusted, handle=handle)

        if code is not None:
            raise RuntimeUnavailable(code)

        timings["verify_seconds"] = round(time.perf_counter() - mark, 3)
        yield SharedRuntime(expected_id, directory, prepared, timings, sid)
    finally:
        release()

"""Windows AppContainer 실험용 최소 네이티브 실행 래퍼. 범용 worker 실행기가 아니다."""

import ctypes as ct
from ctypes import wintypes as wt
import os
import time

from .job_object import WindowsJob


class StartupInfo(ct.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("reserved", wt.LPWSTR), ("desktop", wt.LPWSTR), ("title", wt.LPWSTR),
        ("x", wt.DWORD), ("y", wt.DWORD), ("xsize", wt.DWORD), ("ysize", wt.DWORD),
        ("xchars", wt.DWORD), ("ychars", wt.DWORD), ("fill", wt.DWORD), ("flags", wt.DWORD),
        ("show", wt.WORD), ("reserved_size", wt.WORD), ("reserved_bytes", ct.c_void_p),
        ("stdin", wt.HANDLE), ("stdout", wt.HANDLE), ("stderr", wt.HANDLE),
    ]


class StartupInfoEx(ct.Structure):
    _fields_ = [("startup", StartupInfo), ("attributes", ct.c_void_p)]


class ProcessInfo(ct.Structure):
    _fields_ = [("process", wt.HANDLE), ("thread", wt.HANDLE), ("pid", wt.DWORD), ("tid", wt.DWORD)]


class SidAndAttributes(ct.Structure):
    _fields_ = [("sid", ct.c_void_p), ("attributes", wt.DWORD)]


class SecurityCapabilities(ct.Structure):
    _fields_ = [("sid", ct.c_void_p), ("capabilities", ct.c_void_p), ("count", wt.DWORD), ("reserved", wt.DWORD)]


class AppContainer:
    """매 실험에서 새 프로필만 만들고 종료 후 자신이 만든 프로필만 제거한다."""

    def __init__(self, name):
        if os.name != "nt":
            raise OSError("WINDOWS_REQUIRED")

        self.name = name
        self.sid = ct.c_void_p()
        self.created = False
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        self.userenv = ct.WinDLL("userenv", use_last_error=True)
        self.advapi = ct.WinDLL("advapi32", use_last_error=True)
        self._signatures()

    def _signatures(self):
        """64비트 포인터가 잘리지 않도록 모든 네이티브 인자를 명시한다."""
        self.userenv.CreateAppContainerProfile.argtypes = [wt.LPCWSTR, wt.LPCWSTR, wt.LPCWSTR, ct.c_void_p, wt.DWORD, ct.POINTER(ct.c_void_p)]
        self.userenv.CreateAppContainerProfile.restype = ct.c_long
        self.userenv.DeleteAppContainerProfile.argtypes = [wt.LPCWSTR]
        self.userenv.DeleteAppContainerProfile.restype = ct.c_long
        self.advapi.ConvertSidToStringSidW.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
        self.advapi.ConvertSidToStringSidW.restype = wt.BOOL
        self.advapi.FreeSid.argtypes = [ct.c_void_p]
        self.advapi.ConvertStringSidToSidW.argtypes = [wt.LPCWSTR, ct.POINTER(ct.c_void_p)]
        self.advapi.ConvertStringSidToSidW.restype = wt.BOOL
        self.kernel.LocalFree.argtypes = [ct.c_void_p]
        self.kernel.LocalFree.restype = ct.c_void_p
        self.kernel.InitializeProcThreadAttributeList.argtypes = [ct.c_void_p, wt.DWORD, wt.DWORD, ct.POINTER(ct.c_size_t)]
        self.kernel.UpdateProcThreadAttribute.argtypes = [ct.c_void_p, wt.DWORD, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_void_p]
        self.kernel.DeleteProcThreadAttributeList.argtypes = [ct.c_void_p]
        self.kernel.CreateProcessW.argtypes = [wt.LPCWSTR, wt.LPWSTR, ct.c_void_p, ct.c_void_p, wt.BOOL, wt.DWORD, ct.c_void_p, wt.LPCWSTR, ct.POINTER(StartupInfoEx), ct.POINTER(ProcessInfo)]
        self.kernel.CreateProcessW.restype = wt.BOOL
        self.kernel.ResumeThread.argtypes = [wt.HANDLE]
        self.kernel.ResumeThread.restype = wt.DWORD
        self.kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        self.kernel.WaitForSingleObject.restype = wt.DWORD
        self.kernel.GetExitCodeProcess.argtypes = [wt.HANDLE, ct.POINTER(wt.DWORD)]
        self.kernel.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
        self.kernel.CloseHandle.argtypes = [wt.HANDLE]

    def create(self):
        result = self.userenv.CreateAppContainerProfile(self.name, self.name, "Synthetic compatibility probe", None, 0, ct.byref(self.sid))

        if result < 0:
            raise OSError(ct.c_uint32(result).value, "APPCONTAINER_PROFILE_CREATE_FAILED")

        self.created = True
        text = ct.c_void_p()

        if not self.advapi.ConvertSidToStringSidW(self.sid, ct.byref(text)):
            raise OSError(ct.get_last_error(), "APPCONTAINER_SID_FAILED")

        try:
            return ct.wstring_at(text)
        finally:
            self.kernel.LocalFree(text)

    def close(self):
        """프로필 정리 실패를 숨기지 않고 호출자에게 반환한다."""
        cleaned = True

        if self.created:
            cleaned = self.userenv.DeleteAppContainerProfile(self.name) >= 0
            self.created = not cleaned

        if self.sid:
            self.advapi.FreeSid(self.sid)
            self.sid = ct.c_void_p()

        return cleaned

    def launch(self, executable, arguments, environment, cwd, *, seconds=60, on_started=None, limits=None, cancel_event=None,
               capabilities=()):
        """고정 probe를 실행한다. 핸들 상속·셸·네트워크 권한은 주지 않는다.

        capabilities는 비어 있거나 공유 런타임 읽기용 고정 capability SID 하나(worker_runtime.capability_sid())만 받는다.
        다른 이름 파생·네트워크 capability는 모두 거부한다. 기본값은 기존처럼 capability 0개다.
        """
        import subprocess

        if capabilities != ():
            from .worker_runtime import capability_sid

            if type(capabilities) is not tuple or capabilities != (capability_sid(),):
                raise ValueError("APPCONTAINER_CAPABILITY_INVALID")

        length = ct.c_size_t()
        self.kernel.InitializeProcThreadAttributeList(None, 1, 0, ct.byref(length))

        if not length.value:
            raise OSError("APPCONTAINER_ATTRIBUTES_FAILED")

        attributes = ct.create_string_buffer(length.value)

        if not self.kernel.InitializeProcThreadAttributeList(attributes, 1, 0, ct.byref(length)):
            raise OSError("APPCONTAINER_ATTRIBUTES_FAILED")

        process = ProcessInfo()
        sids = []
        started = False
        finished = False
        job = None

        try:
            # 사용자 한도가 전달되면 OS 제한 설정부터 완료한다.
            if limits is not None:
                job = WindowsJob(limits)

            for text in capabilities:
                sid = ct.c_void_p()

                if not self.advapi.ConvertStringSidToSidW(text, ct.byref(sid)):
                    raise OSError(ct.get_last_error(), "APPCONTAINER_CAPABILITIES_FAILED")

                sids.append(sid)

            # SID_AND_ATTRIBUTES 배열. SE_GROUP_ENABLED(0x4)로 켠다.
            entries = (SidAndAttributes * max(len(sids), 1))(*[SidAndAttributes(sid, 0x4) for sid in sids])
            security = SecurityCapabilities(self.sid, ct.cast(entries, ct.c_void_p) if sids else None, len(sids), 0)

            if not self.kernel.UpdateProcThreadAttribute(attributes, 0, 0x20009, ct.byref(security), ct.sizeof(security), None, None):
                raise OSError("APPCONTAINER_CAPABILITIES_FAILED")

            startup = StartupInfoEx()
            startup.startup.cb = ct.sizeof(startup)
            startup.attributes = ct.cast(attributes, ct.c_void_p)
            command = ct.create_unicode_buffer(subprocess.list2cmdline([str(executable), *arguments]))
            # Windows의 환경 블록은 대소문자를 구분하지 않는 키 순서로 정렬한다.
            entries = sorted(environment.items(), key=lambda item: item[0].casefold())
            env = ct.create_unicode_buffer("\0".join(f"{key}={value}" for key, value in entries) + "\0\0")
            flags = 0x80000 | 0x400 | 0x8000000 | 0x4

            if not self.kernel.CreateProcessW(str(executable), command, None, None, False, flags, env, str(cwd), ct.byref(startup), ct.byref(process)):
                raise OSError(ct.get_last_error(), "APPCONTAINER_PROCESS_CREATE_FAILED")

            started = True

            if job is not None:
                job.assign(process.process)

            # 수집기를 사용하는 합성 검사에서만 PID를 로컬에 기록한다.
            # 콜백 실패 시 아직 정지된 프로세스를 finally에서 종료한다.
            if on_started is not None:
                on_started(process.pid)

            started_at = time.monotonic()

            if self.kernel.ResumeThread(process.thread) == 0xFFFFFFFF:
                raise OSError("APPCONTAINER_RESUME_FAILED")

            if job is not None:
                resources = job.monitor(process.process, started_at=started_at, cancel_event=cancel_event)
                finished = resources["cleanup_confirmed"]

                if resources["status"] in {"succeeded", "failed_runtime"} and resources["exit_code"] is not None:
                    return {"status": "exited", "exit_code": resources["exit_code"], "resources": resources}

                return {"status": resources["status"], "resources": resources}

            waited = self.kernel.WaitForSingleObject(process.process, seconds * 1000)

            if waited == 258:
                return {"status": "timed_out", "code": "PROBE_TIMED_OUT"}

            if waited != 0:
                raise OSError("APPCONTAINER_WAIT_FAILED")

            finished = True
            code = wt.DWORD()

            if not self.kernel.GetExitCodeProcess(process.process, ct.byref(code)):
                raise OSError("APPCONTAINER_EXIT_QUERY_FAILED")

            return {"status": "exited", "exit_code": code.value}
        finally:
            if started and not finished:
                self.kernel.TerminateProcess(process.process, 1)
                self.kernel.WaitForSingleObject(process.process, 5000)

            if job is not None:
                job.close()

            for handle in (process.thread, process.process):
                if handle:
                    self.kernel.CloseHandle(handle)

            self.kernel.DeleteProcThreadAttributeList(attributes)

            for sid in sids:
                self.kernel.LocalFree(sid)

"""Windows Job Object 기반 worker 자원 제한. 파일·네트워크 격리는 별도다."""

import ctypes as ct
from ctypes import wintypes as wt
import os
import time

from .worker_contract import WorkerLimits


ACTIVE_PROCESS_LIMIT = 0x00000008
JOB_MEMORY_LIMIT = 0x00000200
KILL_ON_JOB_CLOSE = 0x00002000
RESOURCE_MESSAGES = {3, 9, 10}  # 프로세스 수·프로세스 메모리·Job 메모리 초과


class BasicLimits(ct.Structure):
    _fields_ = [
        ("process_time", ct.c_int64), ("job_time", ct.c_int64), ("flags", wt.DWORD),
        ("min_working_set", ct.c_size_t), ("max_working_set", ct.c_size_t),
        ("active_processes", wt.DWORD), ("affinity", ct.c_size_t),
        ("priority", wt.DWORD), ("scheduling", wt.DWORD),
    ]


class IoCounters(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in (
        "read_operations", "write_operations", "other_operations",
        "read_bytes", "write_bytes", "other_bytes",
    )]


class ExtendedLimits(ct.Structure):
    _fields_ = [
        ("basic", BasicLimits), ("io", IoCounters),
        ("process_memory", ct.c_size_t), ("job_memory", ct.c_size_t),
        ("peak_process_memory", ct.c_size_t), ("peak_job_memory", ct.c_size_t),
    ]


class CompletionPort(ct.Structure):
    _fields_ = [("key", ct.c_void_p), ("port", wt.HANDLE)]


class Accounting(ct.Structure):
    _fields_ = [
        ("user_time", ct.c_int64), ("kernel_time", ct.c_int64),
        ("period_user_time", ct.c_int64), ("period_kernel_time", ct.c_int64),
        ("page_faults", wt.DWORD), ("total_processes", wt.DWORD),
        ("active_processes", wt.DWORD), ("terminated_processes", wt.DWORD),
    ]


def build_limits(limits):
    """사용자 요청을 OS 구조로 옮긴다. 기본값·상한·무제한 옵션을 새로 만들지 않는다."""
    if not isinstance(limits, WorkerLimits):
        raise TypeError("WorkerLimits 자원 제한 계약이 필요합니다.")

    information = ExtendedLimits()
    information.basic.flags = ACTIVE_PROCESS_LIMIT | JOB_MEMORY_LIMIT | KILL_ON_JOB_CLOSE
    information.basic.active_processes = limits.processes
    information.job_memory = limits.memory_mib * 1024 * 1024

    return information


class WindowsJob:
    """이름 없는 비상속 Job. 일반 CreateProcess 자식의 breakaway를 허용하지 않는다."""

    def __init__(self, limits):
        information = build_limits(limits)

        if os.name != "nt":
            raise OSError("Windows에서만 Job Object를 사용할 수 있습니다.")

        self.limits = limits
        self.handle = None
        self.port = None
        self.messages = set()
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        self._signatures()

        try:
            self.handle = self.kernel.CreateJobObjectW(None, None)

            if not self.handle:
                self._error("JOB_CREATE_FAILED")

            if not self.kernel.SetInformationJobObject(self.handle, 9, ct.byref(information), ct.sizeof(information)):
                self._error("JOB_LIMIT_SETUP_FAILED")

            self.port = self.kernel.CreateIoCompletionPort(ct.c_void_p(-1), None, 0, 1)

            if not self.port:
                self._error("JOB_NOTIFICATION_PORT_FAILED")

            association = CompletionPort(ct.c_void_p(1), self.port)

            if not self.kernel.SetInformationJobObject(self.handle, 7, ct.byref(association), ct.sizeof(association)):
                self._error("JOB_NOTIFICATION_SETUP_FAILED")
        except BaseException:
            self.close()
            raise

    def _signatures(self):
        """64비트 핸들·SIZE_T가 잘리지 않도록 Win32 인자·반환형을 고정한다."""
        kernel = self.kernel
        kernel.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
        kernel.CreateJobObjectW.restype = wt.HANDLE
        kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
        kernel.SetInformationJobObject.restype = wt.BOOL
        kernel.QueryInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p]
        kernel.QueryInformationJobObject.restype = wt.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        kernel.AssignProcessToJobObject.restype = wt.BOOL
        kernel.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
        kernel.TerminateJobObject.restype = wt.BOOL
        kernel.CreateIoCompletionPort.argtypes = [wt.HANDLE, wt.HANDLE, ct.c_size_t, wt.DWORD]
        kernel.CreateIoCompletionPort.restype = wt.HANDLE
        kernel.GetQueuedCompletionStatus.argtypes = [wt.HANDLE, ct.POINTER(wt.DWORD), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_void_p), wt.DWORD]
        kernel.GetQueuedCompletionStatus.restype = wt.BOOL
        kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        kernel.WaitForSingleObject.restype = wt.DWORD
        kernel.GetExitCodeProcess.argtypes = [wt.HANDLE, ct.POINTER(wt.DWORD)]
        kernel.GetExitCodeProcess.restype = wt.BOOL
        kernel.CloseHandle.argtypes = [wt.HANDLE]
        kernel.CloseHandle.restype = wt.BOOL

    @staticmethod
    def _error(code):
        raise OSError(ct.get_last_error(), code)

    def assign(self, process_handle):
        """정지 상태 worker를 연결한다. 실패 시 제한 없는 실행으로 우회하지 않는다."""
        if not self.handle or not self.kernel.AssignProcessToJobObject(self.handle, process_handle):
            self._error("JOB_ASSIGN_FAILED")

    def snapshot(self):
        """실행 중·종료된 프로세스를 포함한 Job 통계를 조회한다."""
        accounting = Accounting()
        limits = ExtendedLimits()

        for kind, information in ((1, accounting), (9, limits)):
            if not self.kernel.QueryInformationJobObject(self.handle, kind, ct.byref(information), ct.sizeof(information), None):
                self._error("JOB_ACCOUNTING_QUERY_FAILED")

        return {
            "active_processes": accounting.active_processes,
            "total_processes": accounting.total_processes,
            "peak_job_memory_bytes": limits.peak_job_memory,
            "configured_job_memory_bytes": limits.job_memory,
            "configured_process_limit": limits.basic.active_processes,
        }

    def _poll_messages(self):
        """통지는 진단 근거다. 미수신을 제한 미발생의 증명으로 사용하지 않는다."""
        for _ in range(256):
            message = wt.DWORD()
            key = ct.c_size_t()
            process = ct.c_void_p()
            ok = self.kernel.GetQueuedCompletionStatus(self.port, ct.byref(message), ct.byref(key), ct.byref(process), 0)

            if not ok:
                if ct.get_last_error() != 258:
                    self._error("JOB_NOTIFICATION_READ_FAILED")

                break

            if key.value == 1 and message.value in RESOURCE_MESSAGES:
                self.messages.add(message.value)

    def terminate_and_drain(self):
        """루트뿐 아니라 Job 전체를 종료하고 최대 5초 동안 잔존 프로세스 0을 확인한다."""
        if self.snapshot()["active_processes"]:
            if not self.kernel.TerminateJobObject(self.handle, 1):
                self._error("JOB_TERMINATION_FAILED")

        deadline = time.monotonic() + 5

        while self.snapshot()["active_processes"]:
            if time.monotonic() >= deadline:
                return False

            time.sleep(0.02)

        return True

    def monitor(self, process_handle, *, started_at, cancel_event=None):
        """실제 경과 시간·취소·OS 통지를 감시하고 모든 종료 경로에서 자식을 정리한다."""
        status, exit_code = "failed_runtime", None
        cleanup = False

        try:
            while True:
                self._poll_messages()

                if self.messages:
                    status = "resource_exceeded"
                    break

                if cancel_event is not None and cancel_event.is_set():
                    status = "cancelled"
                    break

                if time.monotonic() - started_at >= self.limits.wall_seconds:
                    status = "timed_out"
                    break

                waited = self.kernel.WaitForSingleObject(process_handle, 50)

                if waited == 0:
                    code = wt.DWORD()

                    if not self.kernel.GetExitCodeProcess(process_handle, ct.byref(code)):
                        self._error("JOB_PROCESS_EXIT_QUERY_FAILED")

                    exit_code = code.value
                    # 종료 직전 도착한 제한 통지도 확인한다.
                    self._poll_messages()
                    status = "resource_exceeded" if self.messages else "succeeded" if exit_code == 0 else "failed_runtime"
                    break

                if waited != 258:
                    self._error("JOB_PROCESS_WAIT_FAILED")
        finally:
            cleanup = self.terminate_and_drain()

        result = {
            "status": status, "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - started_at, 6),
            "resource_messages": sorted(self.messages), "cleanup_confirmed": cleanup,
            **self.snapshot(),
        }

        if not cleanup:
            result["status"] = "failed_runtime"
            result["cleanup_warning"] = "JOB_PROCESSES_REMAIN"

        return result

    def close(self):
        """비상속 마지막 Job 핸들을 닫으면 KILL_ON_JOB_CLOSE가 적용된다."""
        for name in ("handle", "port"):
            handle = getattr(self, name, None)

            if handle:
                self.kernel.CloseHandle(handle)
                setattr(self, name, None)

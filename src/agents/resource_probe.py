"""고정 합성 worker로 Job 제한을 검사한다. 이 도구 자체는 AppContainer가 아니다."""

import argparse
import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .appcontainer_native import ProcessInfo, StartupInfo
from .artifacts import _root
from .job_object import WindowsJob
from .worker_contract import WorkerLimits


CASES = {"normal", "memory", "spawn", "sleep", "descendant", "cancel"}


def launch_synthetic(case, limits, *, cancel_event=None, on_assigned=None):
    """고정 파일·고정 시나리오만 실행한다. 임의 명령·모듈을 받지 않는다."""
    if case not in CASES:
        raise ValueError("등록된 합성 자원 검사만 실행할 수 있습니다.")

    job = WindowsJob(limits)
    kernel = job.kernel
    process = ProcessInfo()
    created = False

    # 표준 CreateProcess는 STARTUPINFO를 사용하며 핸들을 상속하지 않는다.
    kernel.CreateProcessW.argtypes = [wt.LPCWSTR, wt.LPWSTR, ct.c_void_p, ct.c_void_p, wt.BOOL, wt.DWORD, ct.c_void_p, wt.LPCWSTR, ct.POINTER(StartupInfo), ct.POINTER(ProcessInfo)]
    kernel.CreateProcessW.restype = wt.BOOL
    kernel.ResumeThread.argtypes = [wt.HANDLE]
    kernel.ResumeThread.restype = wt.DWORD
    kernel.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
    kernel.TerminateProcess.restype = wt.BOOL
    startup = StartupInfo()
    startup.cb = ct.sizeof(startup)
    payload = Path(__file__).with_name("resource_child.py").resolve()
    mode = "sleep" if case == "cancel" else case
    command = ct.create_unicode_buffer(subprocess.list2cmdline([
        sys.executable, "-I", "-S", "-B", str(payload), mode,
    ]))

    # 합성 검사에도 키·PYTHONPATH·부모 PATH를 전달하지 않는다.
    environment = {name: os.environ[name] for name in ("SystemRoot", "USERPROFILE", "LOCALAPPDATA") if name in os.environ}
    environment["PATH"] = str(Path(sys.executable).parent)
    entries = sorted(environment.items(), key=lambda item: item[0].casefold())
    block = ct.create_unicode_buffer("\0".join(f"{key}={value}" for key, value in entries) + "\0\0")

    try:
        flags = 0x4 | 0x400 | 0x08000000  # 정지 상태·Unicode 환경·새 창 없음

        if not kernel.CreateProcessW(sys.executable, command, None, None, False, flags, block, str(payload.parent), ct.byref(startup), ct.byref(process)):
            job._error("SYNTHETIC_PROCESS_CREATE_FAILED")

        created = True
        job.assign(process.process)

        # 내부 검증 콜백이다. CLI·요청 JSON·LLM에서는 콜백을 지정할 수 없다.
        if on_assigned is not None:
            on_assigned(job, process.process)

        started_at = time.monotonic()

        if kernel.ResumeThread(process.thread) == 0xFFFFFFFF:
            job._error("SYNTHETIC_PROCESS_RESUME_FAILED")

        result = job.monitor(process.process, started_at=started_at, cancel_event=cancel_event)
        result["limits"] = asdict(limits)
        result["case"] = case
        result["is_security_isolation_test"] = False
        result["real_data_read"] = False

        return result
    finally:
        # Job 연결 실패 시에도 아직 정지된 루트 프로세스를 남기지 않는다.
        if created:
            kernel.TerminateProcess(process.process, 1)
            kernel.WaitForSingleObject(process.process, 5000)

        job.close()

        for handle in (process.thread, process.process):
            if handle:
                kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description="합성 worker의 OS 자원 제한 검사. 방화벽·감사 설정을 변경하지 않습니다.")
    parser.add_argument("--confirm-resource-probe", action="store_true")
    parser.add_argument("--case", choices=sorted(CASES), default="normal")
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--wall-seconds", type=int, default=300)
    arguments = parser.parse_args()

    if not arguments.confirm_resource_probe:
        parser.error("합성 프로세스 실행에는 --confirm-resource-probe 동의가 필요합니다.")

    limits = WorkerLimits(arguments.wall_seconds, arguments.memory_mib, arguments.processes)
    cancelled = threading.Event()
    timer = None

    if arguments.case == "cancel":
        timer = threading.Timer(0.5, cancelled.set)
        timer.start()

    try:
        result = launch_synthetic(arguments.case, limits, cancel_event=cancelled)
    finally:
        if timer is not None:
            timer.cancel()

    parent = Path(__file__).resolve().parents[2]

    for part in ("logs", "agents", "resource_probes"):
        parent = parent / part
        parent.mkdir(exist_ok=True)
        _root(parent)

    directory = parent / uuid.uuid4().hex
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("합성 자원 검사 결과:", directory / "result.json")
    print("자원 제한 검사이며 실데이터 worker·LLM을 활성화하지 않았습니다.")


if __name__ == "__main__":
    main()

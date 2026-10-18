"""합성 worker의 제한된 진행·프로세스 시도 기록. 보안 경계나 차단 판정기가 아니다."""

import json
import math
import os
from pathlib import Path
import sys
import threading
import time


STAGES = frozenset({
    "BOOT", "TOKEN", "FILE_ACCESS", "NETWORK", "IMPORT_NUMPY", "IMPORT_PANDAS",
    "IMPORT_SKLEARN", "IMPORT_MATPLOTLIB", "IMPORT_PYPLOT", "PIPELINE",
    "CSV", "PLOT_CREATE", "PLOT_SAVE", "RESULT",
})
STATES = frozenset({"START", "DONE", "FAILED"})
AUDIT_EVENTS = frozenset({"subprocess.Popen", "_winapi.CreateProcess", "os.system", "os.startfile"})
PACKAGES = frozenset({"numpy", "pandas", "sklearn", "matplotlib", "joblib", "PIL", "fontTools"})
PROGRAMS = {
    "fc-list": "FONTCONFIG", "fc-list.exe": "FONTCONFIG",
    "powershell.exe": "POWERSHELL", "pwsh.exe": "POWERSHELL",
    "wmic.exe": "WMIC", "python.exe": "PYTHON", "cmd.exe": "CMD",
    "where.exe": "WHERE", "dvipng.exe": "DVIPNG",
    "gswin64c.exe": "GHOSTSCRIPT", "gswin32c.exe": "GHOSTSCRIPT",
}
SITES = {
    ("matplotlib.font_manager", "_get_fontconfig_fonts"): "MATPLOTLIB_FONTCONFIG",
    ("matplotlib", "_get_executable_info"): "MATPLOTLIB_EXECUTABLE_CHECK",
    ("joblib.externals.loky.backend.context", "_count_physical_cores_win32"): "JOBLIB_CPU_QUERY",
}
RECORD_FIELDS = {"sequence", "seconds", "kind", "stage", "state", "audit_event", "program", "package", "site", "line"}


def caller_location():
    """호출 스택의 경로·지역 변수·코드 원문 대신 고정 패키지/위치 코드만 반환한다."""
    frame = sys._getframe(1)

    try:
        for _ in range(32):
            if frame is None:
                break

            module = frame.f_globals.get("__name__", "")
            package = module.split(".", 1)[0] if isinstance(module, str) else ""

            if package in PACKAGES:
                site = SITES.get((module, frame.f_code.co_name), "OTHER_APPROVED_PACKAGE")
                return package, site, min(max(frame.f_lineno, 0), 1000000)

            frame = frame.f_back
    finally:
        # frame 참조를 유지해 원본 데이터나 지역 변수를 붙잡지 않는다.
        del frame

    return "unknown", "UNKNOWN", 0


class ProbeJournal:
    """최대 64개 기록을 매번 flush/fsync한다. worker 강제 종료의 finally에 의존하지 않는다."""

    def __init__(self, path):
        self.path = Path(path)
        self.started_at = time.monotonic()
        self.stage = "BOOT"
        self.sequence = 0
        self.failed = False
        self._lock = threading.Lock()

    def _write(self, **fields):
        with self._lock:
            if self.failed or self.sequence >= 64:
                return

            record = {"sequence": self.sequence + 1, "seconds": round(time.monotonic() - self.started_at, 6), **fields}
            payload = (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")

            try:
                with self.path.open("ab", buffering=0) as stream:
                    if stream.write(payload) != len(payload):
                        raise OSError("PROGRESS_WRITE_INCOMPLETE")

                    os.fsync(stream.fileno())

                self.sequence += 1
            except OSError:
                # 진단 쓰기 실패가 새 프로세스 허용/거부 정책을 대신하지 않도록 한다.
                self.failed = True

    def checkpoint(self, stage, state):
        """자유 설명이나 파일명을 받지 않고 고정 단계·상태만 남긴다."""
        if stage not in STAGES or state not in STATES:
            raise ValueError("허용한 합성 진단 단계·상태가 아닙니다.")

        self.stage = stage
        self._write(kind="checkpoint", stage=stage, state=state)

    def audit(self, event, arguments):
        """OS 호출 전에 생성 시도를 기록한다. 인자 목록·환경·명령 문자열은 저장하지 않는다."""
        if event not in AUDIT_EVENTS:
            return

        # os.system/startfile의 문자열은 명령·문서 경로일 수 있으므로 해석하지 않는다.
        executable = arguments[0] if event in {"subprocess.Popen", "_winapi.CreateProcess"} and arguments else None
        program = "UNKNOWN"

        if isinstance(executable, str):
            basename = executable.replace("\\", "/").rsplit("/", 1)[-1].lower()
            program = PROGRAMS.get(basename, "UNKNOWN")
        elif executable is None and event == "subprocess.Popen" and len(arguments) > 1:
            # Windows Popen은 executable=None일 수 있다. 고정 실행 파일 접두사만 비교한다.
            # 전체 명령을 파싱·저장하지 않으며 경로가 복잡하면 UNKNOWN으로 남긴다.
            command = arguments[1]

            if isinstance(command, str):
                for name, code in PROGRAMS.items():
                    if command == name or command.startswith(name + " ") or command.startswith('"' + name + '" '):
                        program = code
                        break

        package, site, line = caller_location()
        self._write(kind="process_attempt", stage=self.stage, audit_event=event,
                    program=program, package=package, site=site, line=line)


def read_progress(path):
    """worker가 쓴 기록을 제한 크기·고정 필드로 검증한다. 신뢰된 보안 증거로 승격하지 않는다."""
    try:
        with Path(path).open("rb") as stream:
            payload = stream.read(65537)
    except OSError:
        return {"status": "unavailable"}

    if len(payload) > 65536:
        return {"status": "invalid"}

    # 강제 종료 중 마지막 한 줄이 끊어졌다면 완성된 앞부분만 사용한다.
    partial = bool(payload and not payload.endswith(b"\n"))
    lines = payload.split(b"\n")[:-1]
    records = []

    try:
        if len(lines) > 64:
            raise ValueError

        for line in lines:
            row = json.loads(line)

            if not isinstance(row, dict) or set(row) - RECORD_FIELDS:
                raise ValueError

            if type(row.get("sequence")) is not int or row["sequence"] != len(records) + 1:
                raise ValueError

            seconds = row.get("seconds")

            if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0 or row.get("stage") not in STAGES:
                raise ValueError

            if row.get("kind") == "checkpoint":
                if set(row) != {"sequence", "seconds", "kind", "stage", "state"} or row.get("state") not in STATES:
                    raise ValueError
            elif row.get("kind") == "process_attempt":
                if set(row) != {"sequence", "seconds", "kind", "stage", "audit_event", "program", "package", "site", "line"}:
                    raise ValueError

                if row["audit_event"] not in AUDIT_EVENTS or row["program"] not in {*PROGRAMS.values(), "UNKNOWN"} or row["package"] not in {*PACKAGES, "unknown"} or row["site"] not in {*SITES.values(), "OTHER_APPROVED_PACKAGE", "UNKNOWN"}:
                    raise ValueError

                if type(row["line"]) is not int or not 0 <= row["line"] <= 1000000:
                    raise ValueError
            else:
                raise ValueError

            records.append(row)
    except (ValueError, TypeError, UnicodeError):
        return {"status": "invalid"}

    checkpoints = [row for row in records if row["kind"] == "checkpoint"]

    return {
        "status": "partial" if partial else "available" if records else "empty",
        "record_count": len(records),
        "last_checkpoint": checkpoints[-1] if checkpoints else None,
        "process_attempts": [row for row in records if row["kind"] == "process_attempt"],
        "is_security_proof": False,
    }

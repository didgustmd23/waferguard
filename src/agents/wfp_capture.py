"""명시적 동의를 받은 합성 검사에서만 사용하는 로컬 WFP 이벤트 수집기.

방화벽·감사 정책·통신 예외를 수정하지 않는다. 고유 ETW 세션만 종료하며
수집 파일은 다른 프로세스의 메타데이터를 포함할 수 있어 외부로 보내지 않는다.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid

from .artifacts import _root


# 로컬 logman에서 확인한 HRESULT. 명령 출력의 언어와 무관하게 판정한다.
SESSION_NOT_FOUND = 0x80300002
INVALID_ARGUMENT = 0x80070057
DROP_KEYWORD = "0x0000010000000000"


def utc_now():
    """프로세스·연결 시도를 이벤트 시각과 대조하기 위한 UTC를 기록한다."""
    return datetime.now(timezone.utc).isoformat()


class WfpCapture:
    """실행 전용 이름·시간 제한·finally 종료를 갖춘 보조 진단 도구다."""

    def __init__(self, *, consent=False, seconds=20):
        if consent is not True:
            raise ValueError("시스템 네트워크 메타데이터 수집에 명시적으로 동의해야 합니다.")

        if type(seconds) is not int or not 10 <= seconds <= 60:
            raise ValueError("수집 시간은 10~60초 정수로 지정하세요.")

        self.seconds = seconds
        self.name = "waferguard.wfp." + uuid.uuid4().hex
        self._lock = threading.Lock()
        self._timer = None
        self._attempted = False
        self._started = False
        self._closed = False
        self.directory = None
        self.record = {
            "schema_version": "wfp_capture_v2", "status": "not_started",
            "session_name": self.name, "seconds": seconds,
            "provider": "Microsoft-Windows-WFP", "stop_confirmed": False,
            "drop_evidence_verified": False, "production_worker_enabled": False,
            "contains_system_metadata": True, "llm_export_allowed": False,
            "cleanup_confirmed": False, "keywords": DROP_KEYWORD,
        }

    def _command(self, arguments):
        """고정 실행 파일과 목록 인자만 사용한다. 셸 명령을 조립하지 않는다."""
        executable = Path(os.environ["SystemRoot"]) / "System32" / "logman.exe"
        completed = subprocess.run(
            [str(executable), *arguments], capture_output=True, timeout=10,
            check=False, creationflags=0x08000000,
        )

        # 다른 앱 정보가 섞일 수 있는 명령 출력을 저장하지 않고 숫자 코드만 반환한다.
        return completed.returncode & 0xFFFFFFFF

    def _save(self):
        """로컬 상관 정보만 저장한다. LLM이나 외부 감사 로그로 전달하지 않는다."""
        if self.directory is not None:
            (self.directory / "capture.json").write_text(
                json.dumps(self.record, ensure_ascii=False, indent=2), encoding="utf-8",
            )

    def _query_session(self, phase):
        """없음·존재·조회 실패를 구분한다. 알 수 없는 오류를 없음으로 취급하지 않는다."""
        try:
            code = self._command(["query", self.name, "-ets"])
            self.record[f"{phase}_return_code"] = code
        except (OSError, subprocess.TimeoutExpired):
            self.record[f"{phase}_error"] = "WFP_SESSION_QUERY_FAILED"
            return "unknown"

        if code == 0:
            return "present"

        if code == SESSION_NOT_FOUND:
            return "absent"

        return "unknown"

    def start(self, probe_root, sid, port):
        """런타임 복사 후 프로세스 시작 직전에만 수집한다. 준비 시간은 포함하지 않는다."""
        if os.name != "nt" or not ctypes.windll.shell32.IsUserAnAdmin():
            raise ValueError("WFP 수집은 사용자가 연 관리자 PowerShell에서만 실행하세요.")

        if self.directory is not None or self._closed:
            raise ValueError("이미 사용한 WFP 수집기를 다시 시작할 수 없습니다.")

        root = _root(probe_root)
        self.directory = root / "wfp"
        self.directory.mkdir(exist_ok=False)
        _root(self.directory)
        self.record.update({
            "probe_id": root.name, "appcontainer_sid": sid,
            "remote_address": "127.0.0.1", "remote_port": port,
            "expected_executable": str(root / "runtime" / "python.exe"),
            "trace_file": "events.etl", "start_requested_at": utc_now(),
        })
        self._save()

        # 고유 세션이 이미 있다면 소유권을 가정하지 않고 중단한다.
        session = self._query_session("preflight_query")

        if session == "present":
            self.record["status"] = "existing_session_not_touched"
            self.record["failure_code"] = "WFP_SESSION_ALREADY_EXISTS"
            self._save()
            raise ValueError("같은 이름의 진단 세션이 존재하여 시작하지 않습니다.")

        if session != "absent":
            self.record["status"] = "preflight_unconfirmed"
            self.record["failure_code"] = "WFP_SESSION_QUERY_FAILED"
            self._save()
            raise ValueError("기존 세션 여부를 확인하지 못해 수집을 시작하지 않습니다.")

        self._attempted = True
        try:
            # 직접 ETW 명령에는 예약 수집용 -rf를 섞지 않는다.
            # Dropped 키워드만 요청하고 32MiB 순환 파일과 Python 타이머를 사용한다.
            code = self._command([
                "create", "trace", self.name, "-ets", "-p", "Microsoft-Windows-WFP", DROP_KEYWORD, "5",
                "-o", str(self.directory / "events.etl"), "-f", "bincirc",
                "-max", "32",
            ])
            self.record["start_return_code"] = code

            if code != 0:
                failure = "WFP_CAPTURE_INVALID_ARGUMENT" if code == INVALID_ARGUMENT else "WFP_CAPTURE_START_FAILED"
                self.record["failure_code"] = failure
                raise OSError(code, failure)

            self._started = True
            self.record["status"] = "collecting"
            self.record["started_at"] = utc_now()
            self._save()
            self._timer = threading.Timer(self.seconds, self.stop)
            self._timer.daemon = True
            self._timer.start()
        except BaseException:
            self.record.setdefault("failure_code", "WFP_CAPTURE_START_NOT_CONFIRMED")
            # 명령 timeout·중단에서도 자신이 시도한 고유 세션의 종료를 시도한다.
            self.stop()
            raise

    def process_started(self, pid):
        """ResumeThread 전의 PID를 기록한다. PID만으로 차단 이벤트를 확정하지 않는다."""
        with self._lock:
            if self._closed or self.record["status"] != "collecting":
                raise ValueError("수집이 종료되어 검사 프로세스를 시작하지 않습니다.")

            self.record["process_id"] = pid
            self.record["process_resume_at"] = utc_now()
            self._save()

    def stop(self):
        """타이머·정상 종료·예외가 겹쳐도 동일 세션의 종료 명령은 한 번만 시도한다."""
        with self._lock:
            if self._closed:
                return

            self._closed = True

            if self._timer is not None:
                self._timer.cancel()

            if not self._attempted:
                return

            # 시작 실패 후 세션이 없으면 종료 명령을 불필요하게 호출하지 않는다.
            session = self._query_session("cleanup_query")

            if session == "absent":
                self.record["cleanup_status"] = "session_absent_confirmed"
                self.record["cleanup_confirmed"] = True
            else:
                self.record["stop_requested_at"] = utc_now()

                try:
                    code = self._command(["stop", self.name, "-ets"])
                    self.record["stop_return_code"] = code
                    self.record["stop_confirmed"] = code == 0
                except (OSError, subprocess.TimeoutExpired):
                    self.record["stop_error"] = "WFP_CAPTURE_STOP_NOT_CONFIRMED"

                # stop의 반환값과 별개로 종료 후 세션 부재를 확인한다.
                final_session = self._query_session("after_stop_query")
                self.record["cleanup_confirmed"] = final_session == "absent"
                self.record["cleanup_status"] = "session_absent_confirmed" if final_session == "absent" else "session_state_unconfirmed"

            if not self.record["cleanup_confirmed"]:
                self.record["status"] = "stop_unconfirmed"
            elif not self._started:
                self.record["status"] = "start_failed_cleanup_complete"
            else:
                self.record["status"] = "capture_complete_review_required"
            self.record["finished_at"] = utc_now()

            try:
                self._save()
            except OSError:
                # 결과 파일 쓰기 실패가 프로필·소켓 정리까지 막지 않도록 한다.
                self.record["record_error"] = "WFP_CAPTURE_RECORD_WRITE_FAILED"


def main():
    parser = argparse.ArgumentParser(description="합성 AppContainer 검사의 로컬 WFP 메타데이터 수집")
    parser.add_argument("--confirm-isolation-probe", action="store_true")
    parser.add_argument("--confirm-system-metadata-capture", action="store_true")
    parser.add_argument("--seconds", type=int, default=20)
    arguments = parser.parse_args()

    if not arguments.confirm_isolation_probe or not arguments.confirm_system_metadata_capture:
        parser.error("격리 검사와 시스템 메타데이터 수집의 두 동의 옵션이 모두 필요합니다.")

    capture = WfpCapture(consent=True, seconds=arguments.seconds)
    from .appcontainer_probe import run_probe

    root, result = run_probe(wfp_capture=capture)
    print("격리 검사:", result["status"])
    print("WFP 수집:", capture.record["status"])
    print("로컬 결과:", root / "result.json")

    if capture.directory is not None:
        print("로컬 수집 정보:", capture.directory / "capture.json")

    if "failure_code" in capture.record:
        print("수집 실패 코드:", capture.record["failure_code"])

    if capture.record.get("status") == "start_failed_cleanup_complete":
        print("수집 시작에 실패했으며 활성 세션 없음은 확인했습니다. 종료할 세션은 없습니다.")

    if capture._attempted and not capture.record["cleanup_confirmed"]:
        print("종료를 확인하지 못했습니다. 관리자 터미널에서 다음 고유 세션만 확인·종료하세요.")
        print(f"logman query {capture.name} -ets")
        print(f"logman stop {capture.name} -ets")

    print("ETL은 로컬에서만 확인하세요. 수집 완료는 차단 증명이나 worker 활성화를 뜻하지 않습니다.")
    raise SystemExit(2)


if __name__ == "__main__":
    main()

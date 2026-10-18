"""합성 로컬 서버의 양성 대조와 읽기 전용 통신 예외 진단."""

import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import uuid


class LoopbackObserver:
    """127.0.0.1에만 열고 고정 합성 토큰으로 부모·child 연결을 구분한다."""

    def __init__(self):
        self.tokens = {name: uuid.uuid4().hex for name in ("before", "child", "after")}
        self.counts = {name: 0 for name in (*self.tokens, "unknown")}
        self.controls = {"before": False, "after": False}
        self.healthy = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener = None
        self._thread = None

    def start(self):
        """외부 인터페이스를 열거나 방화벽 설정을 변경하지 않는다."""
        if self._listener is not None:
            raise ValueError("로컬 검사 서버를 중복 시작할 수 없습니다.")

        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(4)
        self._listener.settimeout(0.1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def port(self):
        if self._listener is None:
            raise ValueError("로컬 검사 서버가 시작되지 않았습니다.")

        return self._listener.getsockname()[1]

    def _serve(self):
        """작은 합성 식별 토큰만 수신하고 원문·주소·다른 트래픽은 보존하지 않는다."""
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    self.healthy = False

                return

            name = "unknown"
            recorded = False

            try:
                with connection:
                    connection.settimeout(1)
                    # 고정 32바이트를 끝까지 읽는다. TCP의 분할 수신도 처리한다.
                    payload = b""

                    while len(payload) < 32:
                        chunk = connection.recv(32 - len(payload))

                        if not chunk:
                            break

                        payload += chunk

                    name = next(
                        (key for key, token in self.tokens.items() if payload == token.encode("ascii")),
                        "unknown",
                    )

                    with self._lock:
                        self.counts[name] += 1

                    recorded = True

                    connection.sendall(b"ok")
            except OSError:
                # 연결은 왔으나 식별·응답이 실패한 경우도 숨기지 않는다.
                if not recorded:
                    with self._lock:
                        self.counts["unknown"] += 1

    def control(self, name):
        """동일 서버에 대한 검사 전후 연결·수신 확인. 실패를 예외로 숨기지 않는다."""
        if name not in self.controls:
            raise ValueError("허용하지 않는 로컬 대조 검사입니다.")

        succeeded = False

        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
                connection.settimeout(2)
                connection.sendall(self.tokens[name].encode("ascii"))
                succeeded = connection.recv(2) == b"ok"
        except OSError:
            pass

        self.controls[name] = succeeded

        return succeeded

    def snapshot(self):
        with self._lock:
            counts = dict(self.counts)

        return {"controls": dict(self.controls), "connections": counts, "server_healthy": self.healthy}

    def close(self):
        self._stop.set()

        if self._listener is not None:
            self._listener.close()

        if self._thread is not None:
            self._thread.join(timeout=2)

            if self._thread.is_alive():
                self.healthy = False


def query_loopback_exemption(sid):
    """해당 SID의 예외만 읽고 예외 추가·삭제 명령은 실행하지 않는다."""
    unknown = {"query_succeeded": False, "exemption_present": None}

    sid_pattern = r"S-1-15-2-(?:[0-9]+-){6}[0-9]+"

    if os.name != "nt" or not isinstance(sid, str) or not re.fullmatch(sid_pattern, sid):
        return unknown

    system_root = os.environ.get("SystemRoot")

    if not system_root:
        return unknown

    executable = Path(system_root) / "System32" / "CheckNetIsolation.exe"

    try:
        result = subprocess.run(
            [str(executable), "LoopbackExempt", "-s"], capture_output=True,
            timeout=10, check=False, creationflags=0x08000000,
        )
    except (OSError, subprocess.TimeoutExpired):
        return unknown

    if result.returncode or not result.stdout.strip():
        return unknown

    # SID는 ASCII다. 현지화된 제목이나 다른 앱 이름·SID 목록은 보존하지 않는다.
    # Windows 명령의 UTF-16 출력도 처리한다. 해석 실패를 예외 없음으로 간주하지 않는다.
    try:
        if b"\x00" in result.stdout:
            encoding = "utf-16" if result.stdout.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-16-le"
            output = result.stdout.decode(encoding)
        else:
            output = result.stdout.decode("ascii", errors="replace")
    except UnicodeError:
        return unknown

    identifiers = set(re.findall(r"S-1-15-2-(?:[0-9]+-){6}[0-9]+(?![0-9-])", output))

    return {"query_succeeded": True, "exemption_present": sid in identifiers}


def assess_loopback(child, observation, before, after):
    """대조·토큰·예외 설정을 함께 보되 timeout만으로 OS 차단을 확정하지 않는다."""
    checks = child.get("checks", {}) if isinstance(child, dict) else {}

    if not isinstance(checks, dict) or not isinstance(observation, dict):
        return {"status": "review_required", "code": "LOOPBACK_CONTROL_INCOMPLETE"}

    connections = observation.get("connections", {})
    controls = observation.get("controls", {})

    if not isinstance(connections, dict) or not isinstance(controls, dict):
        return {"status": "review_required", "code": "LOOPBACK_CONTROL_INCOMPLETE"}

    invalid_counts = any(
        type(connections.get(name)) is not int or connections[name] < 0
        for name in ("before", "child", "after", "unknown")
    )

    if invalid_counts:
        return {"status": "review_required", "code": "LOOPBACK_CONTROL_INCOMPLETE"}

    if checks.get("loopback_connected") is True or connections.get("child", 0) > 0:
        return {"status": "failed", "code": "LOOPBACK_CONNECTION_OBSERVED"}

    controls_ok = all(
        controls.get(name) is True and connections[name] == 1
        for name in ("before", "after")
    )

    if observation.get("server_healthy") is not True or not controls_ok or connections["unknown"] != 0:
        return {"status": "review_required", "code": "LOOPBACK_CONTROL_INCOMPLETE"}

    if checks.get("appcontainer_token") is not True or checks.get("network_capabilities_absent") is not True:
        return {"status": "review_required", "code": "NETWORK_TOKEN_NOT_VERIFIED"}

    exemptions_checked = all(
        isinstance(item, dict)
        and item.get("query_succeeded") is True
        and item.get("exemption_present") is False
        for item in (before, after)
    )

    if not exemptions_checked:
        return {"status": "review_required", "code": "LOOPBACK_EXCEPTION_NOT_VERIFIED"}

    outcome = checks.get("loopback_network", {})

    if not isinstance(outcome, dict):
        return {"status": "review_required", "code": "LOOPBACK_RESULT_INCONCLUSIVE"}

    if outcome.get("denied") is True and type(outcome.get("error_code")) is int and outcome["error_code"] in {1, 5, 13, 10013}:
        return {"status": "passed", "code": "LOOPBACK_ACCESS_DENIED"}

    if outcome.get("outcome") == "timeout":
        return {"status": "review_required", "code": "LOOPBACK_OS_DROP_EVIDENCE_MISSING"}

    return {"status": "review_required", "code": "LOOPBACK_RESULT_INCONCLUSIVE"}

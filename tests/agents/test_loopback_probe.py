"""합성 로컬 연결·예외 조회·보수적 판정을 검증한다. OS 설정은 변경하지 않는다."""

from copy import deepcopy
import socket
import subprocess
import unittest
from unittest.mock import patch

from src.agents.loopback_probe import LoopbackObserver, assess_loopback, query_loopback_exemption


class LoopbackProbeTest(unittest.TestCase):
    def evidence(self):
        """접근 거부가 명시된 정상 합성 증거를 매번 새로 만든다."""
        child = {"checks": {
            "appcontainer_token": True, "network_capabilities_absent": True,
            "loopback_connected": False,
            "loopback_network": {"denied": True, "error_code": 10013, "outcome": "access_denied"},
        }}
        observation = {
            "controls": {"before": True, "after": True},
            "connections": {"before": 1, "after": 1, "child": 0, "unknown": 0},
            "server_healthy": True,
        }
        exception = {"query_succeeded": True, "exemption_present": False}

        return child, observation, exception, deepcopy(exception)

    def test_positive_controls_and_child_observation(self):
        """전후 정상 연결과 child 토큰 수신을 구분한다. 실제 AppContainer는 만들지 않는다."""
        observer = LoopbackObserver()

        try:
            observer.start()
            self.assertTrue(observer.control("before"))

            with socket.create_connection(("127.0.0.1", observer.port), timeout=2) as connection:
                connection.sendall(observer.tokens["child"].encode("ascii"))
                self.assertEqual(connection.recv(2), b"ok")

            self.assertTrue(observer.control("after"))
            snapshot = observer.snapshot()
            self.assertEqual(snapshot["connections"], {"before": 1, "child": 1, "after": 1, "unknown": 0})
            self.assertNotIn(observer.tokens["child"], str(snapshot))
        finally:
            observer.close()

    def test_explicit_denial_required(self):
        """모든 대조가 통과해도 시간 초과만으로 OS 차단 성공을 선언하지 않는다."""
        evidence = self.evidence()
        self.assertEqual(assess_loopback(*evidence)["status"], "passed")
        evidence[0]["checks"]["loopback_network"] = {"denied": False, "error_code": None, "outcome": "timeout"}
        result = assess_loopback(*evidence)
        self.assertEqual(result["status"], "review_required")
        self.assertEqual(result["code"], "LOOPBACK_OS_DROP_EVIDENCE_MISSING")

    def test_observed_connection_fails(self):
        """child 접속 또는 서버 수신이 있으면 거부 주장보다 연결 증거를 우선한다."""
        for source in ("child", "server"):
            evidence = self.evidence()

            if source == "child":
                evidence[0]["checks"]["loopback_connected"] = True
            else:
                evidence[1]["connections"]["child"] = 1

            self.assertEqual(assess_loopback(*evidence)["status"], "failed")

    def test_incomplete_evidence_requires_review(self):
        """대조·토큰·예외 조회의 실패나 누락을 통과로 간주하지 않는다."""
        for position, key, value in (
            (1, "server_healthy", False), (1, "connections", {}),
            (2, "query_succeeded", False), (3, "exemption_present", True),
        ):
            evidence = self.evidence()
            evidence[position][key] = value
            self.assertEqual(assess_loopback(*evidence)["status"], "review_required")

        evidence = self.evidence()
        evidence[0]["checks"]["network_capabilities_absent"] = False
        self.assertEqual(assess_loopback(*evidence)["code"], "NETWORK_TOKEN_NOT_VERIFIED")
        self.assertEqual(assess_loopback({}, None, {}, {})["status"], "review_required")

    def test_readonly_exception_query(self):
        """조회 명령만 사용하고 다른 앱의 SID 목록은 결과에 포함하지 않는다."""
        sid = "S-1-15-2-1-2-3-4-5-6-7"

        for output, present in ((sid.encode(), True), (sid.encode("utf-16"), True), (b"No exemptions", False)):
            completed = subprocess.CompletedProcess([], 0, stdout=output)

            with patch("src.agents.loopback_probe.os.name", "nt"), patch.dict("os.environ", {"SystemRoot": "C:/Windows"}), patch("src.agents.loopback_probe.subprocess.run", return_value=completed) as run:
                result = query_loopback_exemption(sid)

            self.assertEqual(result, {"query_succeeded": True, "exemption_present": present})
            self.assertEqual(run.call_args.args[0][1:], ["LoopbackExempt", "-s"])
            self.assertNotIn(sid, str(result))

    def test_query_failure_is_unknown(self):
        """조회 실패를 통신 예외 없음으로 바꾸지 않는다."""
        sid = "S-1-15-2-1-2-3-4-5-6-7"

        with patch("src.agents.loopback_probe.os.name", "nt"), patch("src.agents.loopback_probe.subprocess.run", side_effect=OSError()):
            self.assertEqual(query_loopback_exemption(sid), {"query_succeeded": False, "exemption_present": None})


if __name__ == "__main__":
    unittest.main()

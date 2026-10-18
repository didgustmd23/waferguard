"""모의 WFP 명령만 검증한다. 실제 수집·ACL·방화벽은 변경하지 않는다."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.agents.wfp_capture import DROP_KEYWORD, INVALID_ARGUMENT, SESSION_NOT_FOUND, WfpCapture, main


class WfpCaptureTest(unittest.TestCase):
    def test_consent_and_limits(self):
        """동의 누락·허용 범위 밖 시간·bool 입력을 거부한다."""
        with self.assertRaises(ValueError):
            WfpCapture()

        for seconds in (True, 0, 9, 61, 20.0):
            with self.assertRaises(ValueError):
                WfpCapture(consent=True, seconds=seconds)

    def test_named_capture_and_stop_once(self):
        """고유 세션·파일 크기·타이머·중복 종료·로컬 PID 기록을 검사한다."""
        with tempfile.TemporaryDirectory() as temporary:
            capture = WfpCapture(consent=True)

            with patch("src.agents.wfp_capture.ctypes.windll.shell32.IsUserAnAdmin", return_value=1), patch.object(capture, "_command", side_effect=[SESSION_NOT_FOUND, 0, 0, 0, SESSION_NOT_FOUND]) as command, patch("src.agents.wfp_capture.threading.Timer") as timer:
                capture.start(Path(temporary), "synthetic_sid", 12345)
                capture.process_started(123)
                # 실제 시간을 기다리지 않고 타이머에 등록한 종료 콜백을 호출한다.
                timer.call_args.args[1]()
                capture.stop()

            calls = [call.args[0] for call in command.call_args_list]
            self.assertEqual(calls[0], ["query", capture.name, "-ets"])
            self.assertEqual(calls[1][:4], ["create", "trace", capture.name, "-ets"])
            self.assertEqual(calls[1][-2:], ["-max", "32"])
            self.assertNotIn("-rf", calls[1])
            self.assertIn(DROP_KEYWORD, calls[1])
            self.assertEqual(calls[3], ["stop", capture.name, "-ets"])
            self.assertEqual(len(calls), 5)
            timer.assert_called_once_with(20, capture.stop)
            timer.return_value.start.assert_called_once()
            saved = json.loads((Path(temporary) / "wfp/capture.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["stop_confirmed"])
            self.assertTrue(saved["cleanup_confirmed"])
            self.assertFalse(saved["drop_evidence_verified"])
            self.assertEqual(saved["process_id"], 123)

    def test_start_timeout_attempts_cleanup(self):
        """시작 명령 timeout에서도 고유 세션 종료를 시도한다."""
        with tempfile.TemporaryDirectory() as temporary:
            capture = WfpCapture(consent=True)

            with patch("src.agents.wfp_capture.ctypes.windll.shell32.IsUserAnAdmin", return_value=1), patch.object(capture, "_command", side_effect=[SESSION_NOT_FOUND, subprocess.TimeoutExpired("synthetic", 10), 0, 0, SESSION_NOT_FOUND]) as command:
                with self.assertRaises(subprocess.TimeoutExpired):
                    capture.start(Path(temporary), "synthetic_sid", 12345)

            self.assertEqual(command.call_args_list[3].args[0], ["stop", capture.name, "-ets"])
            self.assertTrue(capture.record["stop_confirmed"])
            self.assertTrue(capture.record["cleanup_confirmed"])

    def test_existing_session_is_not_stopped(self):
        """같은 이름의 기존 세션이 있으면 시작·종료 모두 하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            capture = WfpCapture(consent=True)

            with patch("src.agents.wfp_capture.ctypes.windll.shell32.IsUserAnAdmin", return_value=1), patch.object(capture, "_command", return_value=0) as command:
                with self.assertRaises(ValueError):
                    capture.start(Path(temporary), "synthetic_sid", 12345)

                capture.stop()

            self.assertEqual(command.call_count, 1)

    def test_stop_failure_is_not_success(self):
        """종료 실패는 기록하고 성공으로 바꾸지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            capture = WfpCapture(consent=True)

            with patch("src.agents.wfp_capture.ctypes.windll.shell32.IsUserAnAdmin", return_value=1), patch.object(capture, "_command", side_effect=[SESSION_NOT_FOUND, 0, 0, OSError(), 0]), patch("src.agents.wfp_capture.threading.Timer"):
                capture.start(Path(temporary), "synthetic_sid", 12345)
                capture.stop()

            self.assertEqual(capture.record["status"], "stop_unconfirmed")
            self.assertFalse(capture.record["stop_confirmed"])
            self.assertFalse(capture.record["cleanup_confirmed"])

    def test_start_invalid_argument_without_session(self):
        """실제 실패 이력을 재현하고 시작 오류를 종료 실패로 덮어쓰지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            capture = WfpCapture(consent=True)

            with patch("src.agents.wfp_capture.ctypes.windll.shell32.IsUserAnAdmin", return_value=1), patch.object(capture, "_command", side_effect=[SESSION_NOT_FOUND, INVALID_ARGUMENT, SESSION_NOT_FOUND]) as command:
                with self.assertRaises(OSError):
                    capture.start(Path(temporary), "synthetic_sid", 12345)

            self.assertEqual(capture.record["failure_code"], "WFP_CAPTURE_INVALID_ARGUMENT")
            self.assertEqual(capture.record["status"], "start_failed_cleanup_complete")
            self.assertTrue(capture.record["cleanup_confirmed"])
            self.assertNotIn("stop_return_code", capture.record)
            self.assertEqual(command.call_count, 3)

    def test_query_failure_does_not_start(self):
        """세션 조회의 권한 오류를 없음으로 오인하여 시작하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            capture = WfpCapture(consent=True)

            with patch("src.agents.wfp_capture.ctypes.windll.shell32.IsUserAnAdmin", return_value=1), patch.object(capture, "_command", return_value=0x80070005) as command:
                with self.assertRaises(ValueError):
                    capture.start(Path(temporary), "synthetic_sid", 12345)

                capture.stop()

            self.assertFalse(capture._attempted)
            self.assertEqual(capture.record["status"], "preflight_unconfirmed")
            self.assertEqual(command.call_count, 1)

    def test_successful_stop_requires_absence_check(self):
        """종료 명령 성공이어도 최종 세션 조회 실패라면 정리 완료를 주장하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            capture = WfpCapture(consent=True)

            with patch("src.agents.wfp_capture.ctypes.windll.shell32.IsUserAnAdmin", return_value=1), patch.object(capture, "_command", side_effect=[SESSION_NOT_FOUND, 0, 0, 0, 0x80070005]), patch("src.agents.wfp_capture.threading.Timer"):
                capture.start(Path(temporary), "synthetic_sid", 12345)
                capture.stop()

            self.assertTrue(capture.record["stop_confirmed"])
            self.assertFalse(capture.record["cleanup_confirmed"])
            self.assertEqual(capture.record["status"], "stop_unconfirmed")

    def test_closed_capture_rejects_process(self):
        """수집 종료 후 프로세스 재개를 거부한다."""
        capture = WfpCapture(consent=True)
        capture.stop()

        with self.assertRaises(ValueError):
            capture.process_started(123)

    def test_hresult_normalization(self):
        """부호 있는 Windows 반환값을 동일한 32비트 HRESULT로 정규화한다."""
        capture = WfpCapture(consent=True)
        completed = subprocess.CompletedProcess([], SESSION_NOT_FOUND - (1 << 32))

        with patch("src.agents.wfp_capture.subprocess.run", return_value=completed) as command:
            self.assertEqual(capture._command(["query", capture.name, "-ets"]), SESSION_NOT_FOUND)

        self.assertEqual(command.call_args.args[0][1:], ["query", capture.name, "-ets"])

    def test_cli_requires_both_flags(self):
        """기본 실행이나 한쪽 동의만으로는 probe를 호출하지 않는다."""
        for flags in ([], ["--confirm-isolation-probe"], ["--confirm-system-metadata-capture"]):
            with patch("sys.argv", ["wfp_capture", *flags]), patch("src.agents.appcontainer_probe.run_probe") as probe, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    main()

                probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()

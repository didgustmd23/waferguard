"""진행 기록의 정보 최소화·크기 제한·중단 복구·실제 Python 감사 이벤트를 검증한다."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.agents.probe_progress import ProbeJournal, caller_location, read_progress


class ProbeProgressTest(unittest.TestCase):
    def test_checkpoint_and_private_arguments(self):
        """고정 분류만 저장하고 실행 경로·명령·작업 폴더·환경 값을 버린다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            journal = ProbeJournal(path)
            journal.checkpoint("IMPORT_PYPLOT", "START")

            with patch("src.agents.probe_progress.caller_location", return_value=("matplotlib", "MATPLOTLIB_FONTCONFIG", 123)):
                journal.audit("subprocess.Popen", (
                    "C:/private-location/fc-list.exe", ["private-argument"],
                    "private-working-directory", {"SECRET": "private-value"},
                ))

            result = read_progress(path)
            attempt = result["process_attempts"][0]
            self.assertEqual(attempt["program"], "FONTCONFIG")
            self.assertEqual(attempt["site"], "MATPLOTLIB_FONTCONFIG")
            self.assertEqual(attempt["stage"], "IMPORT_PYPLOT")
            self.assertEqual(result["last_checkpoint"]["state"], "START")
            self.assertNotIn("private", path.read_text())
            self.assertNotIn("SECRET", path.read_text())
            self.assertFalse(result["is_security_proof"])

    def test_unknown_program_and_ignored_events(self):
        """허용하지 않은 실행 파일명과 일반 파일 접근 이벤트를 출력에 노출하지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            journal = ProbeJournal(path)
            journal.audit("open", ("private-file",))
            self.assertFalse(path.exists())
            journal.audit("_winapi.CreateProcess", ("private-program.exe", "private-command"))
            self.assertEqual(read_progress(path)["process_attempts"][0]["program"], "UNKNOWN")
            self.assertNotIn("private", path.read_text())

    def test_windows_implicit_executable(self):
        """Windows의 executable=None 호출도 고정 접두사만 분류하고 인자는 버린다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            journal = ProbeJournal(path)
            journal.audit("subprocess.Popen", (None, "fc-list.exe private-argument", None, {}))
            self.assertEqual(read_progress(path)["process_attempts"][0]["program"], "FONTCONFIG")
            self.assertNotIn("private", path.read_text())

    def test_partial_last_record_is_not_claimed_complete(self):
        """종료 중 끊긴 마지막 줄은 제외하되 앞선 저장 완료 기록을 보존한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            ProbeJournal(path).checkpoint("PIPELINE", "START")

            with path.open("ab") as stream:
                stream.write(b'{"sequence":2')

            result = read_progress(path)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["record_count"], 1)

    def test_write_failure_and_record_limit(self):
        """진단 쓰기 실패는 표시하고 최대 64개 이상의 기록은 생성하지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            journal = ProbeJournal(path)

            for _ in range(100):
                journal.checkpoint("BOOT", "START")

            self.assertEqual(read_progress(path)["record_count"], 64)
            self.assertLess(path.stat().st_size, 65536)

            failed = ProbeJournal(Path(directory) / "missing" / "progress.jsonl")
            failed.checkpoint("BOOT", "START")
            self.assertTrue(failed.failed)

    def test_untrusted_records_are_rejected(self):
        """자유 필드·잘못된 단계·비표준 시각·과대 파일을 거부한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            valid = {"sequence": 1, "seconds": 0.0, "kind": "checkpoint", "stage": "BOOT", "state": "START"}

            for row in ({**valid, "command": "private"}, {**valid, "stage": "PRIVATE_STAGE"}, {**valid, "seconds": float("nan")}):
                path.write_text(json.dumps(row) + "\n")
                self.assertEqual(read_progress(path)["status"], "invalid")

            path.write_bytes(b"x" * 65537)
            self.assertEqual(read_progress(path)["status"], "invalid")
            self.assertEqual(read_progress(path.with_name("missing"))["status"], "unavailable")

    def test_caller_location_is_fixed_code(self):
        """승인 패키지가 없는 호출 스택에서는 경로·함수명 대신 unknown을 반환한다."""
        package, site, line = caller_location()
        self.assertEqual((package, site, line), ("unknown", "UNKNOWN", 0))

    def test_real_subprocess_audit_in_separate_python(self):
        """별도 Python에서만 hook을 설치하여 테스트 실행기의 전역 hook을 오염시키지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            # 합성 프로세스 생성만 수행한다. 네트워크·원본 데이터는 사용하지 않는다.
            script = (
                "import sys,subprocess; from pathlib import Path; "
                "sys.path.insert(0,sys.argv[1]); "
                "from probe_progress import ProbeJournal; "
                "j=ProbeJournal(Path(sys.argv[2])); j.checkpoint('BOOT','START'); "
                "sys.addaudithook(j.audit); "
                "subprocess.run([sys.executable,'-I','-S','-c','pass'],check=True); "
                "j.checkpoint('BOOT','DONE')"
            )
            tools = Path(__file__).resolve().parents[2] / "src" / "agents"
            subprocess.run([sys.executable, "-I", "-S", "-c", script, str(tools), str(path)], check=True, timeout=10)
            result = read_progress(path)
            self.assertTrue(any(row["audit_event"] == "subprocess.Popen" for row in result["process_attempts"]))
            self.assertEqual(result["last_checkpoint"]["state"], "DONE")


if __name__ == "__main__":
    unittest.main()

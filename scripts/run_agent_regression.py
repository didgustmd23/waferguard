"""에이전트 합성 회귀의 진행·시간·실패를 로컬에 보존한다. 실제 격리 선택 검사는 끈다."""

from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import unittest
import uuid


PROJECT = Path(__file__).resolve().parents[1]


class RecordingResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.started = time.perf_counter()
        self.outcome_counts = (len(self.failures), len(self.errors), len(self.skipped))
        self.event({"event": "started", "test": test.id(), "number": self.testsRun})

    def stopTest(self, test):
        elapsed = time.perf_counter() - self.started
        counts = (len(self.failures), len(self.errors), len(self.skipped))
        self.event({"event": "finished", "test": test.id(), "number": self.testsRun,
                    "elapsed_seconds": round(elapsed, 3),
                    "new_failures": counts[0] - self.outcome_counts[0],
                    "new_errors": counts[1] - self.outcome_counts[1],
                    "new_skips": counts[2] - self.outcome_counts[2]})

        if self.testsRun % 25 == 0 or elapsed >= 10 or counts[:2] != self.outcome_counts[:2]:
            print(f"진행 {self.testsRun}/{self.total} · 실패 {counts[0]} · 오류 {counts[1]} · "
                  f"생략 {counts[2]} · 최근 {elapsed:.1f}초 · {test.id()}", file=self.console, flush=True)

        super().stopTest(test)

    def event(self, record):
        self.events.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.events.flush()


def main():
    # 테스트 탐색 전에 설정한다. 외부 환경이 1이어도 실제 선택 검사를 켜지 않는다.
    os.environ["WAFERGUARD_TEST_APPCONTAINER"] = "0"
    sys.path.insert(0, str(PROJECT))
    directory = PROJECT / "logs/agents/regression" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    console = sys.stdout
    started = time.perf_counter()
    summary = {"status": "interrupted", "native_opt_in_enabled": False}
    print("회귀 기록:", directory, flush=True)

    try:
        with (directory / "unittest.txt").open("x", encoding="utf-8") as log, (
            directory / "events.jsonl"
        ).open("x", encoding="utf-8") as events, redirect_stdout(log), redirect_stderr(log):
            suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests/agents"), top_level_dir=str(PROJECT))
            total = suite.countTestCases()
            print(f"전체 {total}개 발견 · 실제 AppContainer 선택 검사 비활성", file=console, flush=True)

            def result_factory(*args, **kwargs):
                result = RecordingResult(*args, **kwargs)
                result.console, result.events, result.total = console, events, total
                return result

            result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=result_factory).run(suite)
            summary = {"status": "passed" if result.wasSuccessful() else "failed",
                       "native_opt_in_enabled": False, "discovered": total, "tests_run": result.testsRun,
                       "failure_count": len(result.failures), "error_count": len(result.errors),
                       "skip_count": len(result.skipped), "expected_failure_count": len(result.expectedFailures),
                       "unexpected_success_count": len(result.unexpectedSuccesses),
                       "failures": [{"test": test.id(), "traceback": trace} for test, trace in result.failures],
                       "errors": [{"test": test.id(), "traceback": trace} for test, trace in result.errors],
                       "skips": [{"test": test.id(), "reason": reason} for test, reason in result.skipped]}
    finally:
        summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("검사 상태:", summary["status"], "· 경과:", summary["elapsed_seconds"], "초", file=console, flush=True)
        print("결과:", directory / "summary.json", file=console, flush=True)

    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""측정용 호출 시간 기록은 환경 변수가 있을 때만 고정 항목을 쓰고, 원래 호출 결과를 바꾸지 않는다."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agents.perf_timing import timed


class PerfTimingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "timing.jsonl"

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_disabled_without_environment_variable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WAFERGUARD_TIMING_LOG", None)
            with timed("status", "apply"):
                pass
        self.assertFalse(self.path.exists())

    def test_records_only_fixed_fields_and_keeps_exceptions(self):
        with patch.dict(os.environ, {"WAFERGUARD_TIMING_LOG": str(self.path)}):
            with timed("next", "handoff"):
                pass
            with self.assertRaises(ValueError), timed("status", "apply"):
                raise ValueError("원래 예외")
            # 정해지지 않은 종류·단계는 기록하지 않는다(자유 문자열 차단).
            with timed("secret token", "apply"), timed("status", "free text"):
                pass
        records = self.records()
        self.assertEqual([(r["kind"], r["step"], r["failed"]) for r in records], [("next", "handoff", False), ("status", "apply", True)])
        self.assertTrue(all(set(r) == {"kind", "step", "start_wall", "elapsed_s", "failed"} for r in records))
        self.assertTrue(all(r["elapsed_s"] >= 0 for r in records))

    def test_write_failure_does_not_change_the_call(self):
        with patch.dict(os.environ, {"WAFERGUARD_TIMING_LOG": str(self.path.parent / "missing" / "timing.jsonl")}):
            with timed("status", "apply"):
                value = 1
        self.assertEqual(value, 1)


if __name__ == "__main__":
    unittest.main()

"""공통 진행 관찰·UI 표시·모의 운송 연결. 실제 worker·승인·실데이터는 사용하지 않는다."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.agents.operation_progress import observe_progress, report_progress
from src.ui.progress_view import show_operation_progress


class ProgressObservationTest(unittest.TestCase):
    def test_nested_observers_restore_after_failure_and_do_not_cross_threads(self):
        outer, inner = [], []
        with observe_progress(outer.append):
            report_progress("contract_check")
            with self.assertRaises(ValueError), observe_progress(inner.append):
                report_progress("runtime_inventory")
                raise ValueError("synthetic")

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(report_progress, "worker_wait").result()

            report_progress("cleanup")

        report_progress("worker_wait")
        self.assertEqual([event.stage for event in outer], ["contract_check", "cleanup"])
        self.assertEqual([event.stage for event in inner], ["runtime_inventory"])

    def test_invalid_or_sensitive_messages_are_not_delivered(self):
        events = []
        with observe_progress(events.append):
            report_progress("D:/private/data.csv")
            report_progress("runtime_copy", completed=True, total=2)
            report_progress("runtime_copy", completed=3, total=2)
            report_progress("runtime_copy", completed=1)
            report_progress("runtime_copy", bytes_done=-1)

        self.assertEqual(events, [])

    def test_display_failure_does_not_change_execution_or_retry_it(self):
        callback = MagicMock(side_effect=RuntimeError("display unavailable"))
        with observe_progress(callback):
            report_progress("worker_wait")

        callback.assert_called_once()


class UiProgressTest(unittest.TestCase):
    def test_elapsed_timer_counts_and_throttle_without_rerun(self):
        st = MagicMock()
        with show_operation_progress(st, "실행 중"):
            report_progress("runtime_copy", completed=0, total=2, bytes_done=0)
            report_progress("runtime_copy", completed=1, total=2, bytes_done=1)
            report_progress("runtime_copy", completed=2, total=2, bytes_done=1024 ** 2)
            report_progress("worker_wait")

        st.spinner.assert_called_once_with("실행 중", show_time=True)
        messages = [call.args[0] for call in st.empty.return_value.info.call_args_list]
        self.assertTrue(any("파일 2/2" in text and "1.0 MiB" in text for text in messages))
        self.assertFalse(any("파일 1/2" in text for text in messages))
        self.assertIn("격리 worker 실행·결과 대기", messages)
        self.assertFalse(any("%" in text for text in messages))
        st.empty.return_value.empty.assert_called_once()
        st.rerun.assert_not_called()

    def test_exception_clears_display_and_observer_without_success_message(self):
        st = MagicMock()
        with self.assertRaises(ValueError):
            with show_operation_progress(st, "확인 중"):
                raise ValueError("synthetic")

        count = st.empty.return_value.info.call_count
        report_progress("worker_wait")
        self.assertEqual(st.empty.return_value.info.call_count, count)
        st.success.assert_not_called()
        st.rerun.assert_not_called()

    def test_wm_transport_reports_real_parent_phases_without_native_launch(self):
        from src.agents import wm_canonical_transport as transport

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("runtime", "input", "tools", "output", "scratch", "installed"):
                (root / name).mkdir()

            source = root / "installed/python.exe"
            source.write_bytes(b"synthetic")
            request = SimpleNamespace(task_id="synthetic", request_id="synthetic",
                output_directory=str(root / "output"), scratch_directory=str(root / "scratch"), limits=None)
            events = []
            with patch.object(transport, "AppContainer") as container, \
                    patch.object(transport, "_acl"), \
                    patch.object(transport, "runtime_files", return_value=[source]), \
                    patch.object(transport, "probe_environment", return_value={}), \
                    patch.object(transport.sys, "prefix", str(root / "installed")), \
                    observe_progress(events.append):
                container.return_value.close.return_value = True
                container.return_value.launch.return_value = {"status": "exited", "exit_code": 0}
                result = transport.run_wm_worker(request, root, threading.Event())

            self.assertTrue(result["profile_cleanup"])
            container.return_value.launch.assert_called_once()
            stages = [event.stage for event in events]
            self.assertEqual(stages[0], "contract_check")
            self.assertLess(stages.index("runtime_copy"), stages.index("worker_wait"))
            self.assertEqual(stages[-1], "cleanup")
            copied = next(event for event in events if event.completed == 1)
            self.assertEqual((copied.total, copied.bytes_done), (1, len(b"synthetic")))
            self.assertIn("src/agents/operation_progress.py", transport.CONTROL_FILES)
            self.assertNotIn("src/agents/operation_progress.py", transport.WORKER_FILES)


if __name__ == "__main__":
    unittest.main()

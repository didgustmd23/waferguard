"""Job 구조·실행 순서와 실제 Windows 합성 worker 자원 제한을 검사한다."""

import ctypes as ct
import os
import threading
import unittest
from unittest.mock import Mock, patch

from src.agents.appcontainer_native import AppContainer
from src.agents.appcontainer_probe import launch_succeeded
from src.agents.job_object import ACTIVE_PROCESS_LIMIT, JOB_MEMORY_LIMIT, KILL_ON_JOB_CLOSE, ExtendedLimits, build_limits
from src.agents.resource_probe import launch_synthetic
from src.agents.worker_contract import WorkerLimits


class JobContractTest(unittest.TestCase):
    def test_values_and_flags(self):
        """사용자 지정 값을 그대로 적용하며 breakaway·상속 제한 해제는 추가하지 않는다."""
        limits = build_limits(WorkerLimits(7, 123, 2))
        self.assertEqual(limits.job_memory, 123 * 1024 * 1024)
        self.assertEqual(limits.basic.active_processes, 2)
        self.assertEqual(limits.basic.flags, ACTIVE_PROCESS_LIMIT | JOB_MEMORY_LIMIT | KILL_ON_JOB_CLOSE)

        if os.name == "nt" and ct.sizeof(ct.c_void_p) == 8:
            self.assertEqual(ct.sizeof(ExtendedLimits), 144)

        with self.assertRaises(TypeError):
            build_limits({"memory_mib": 123})

    def test_assignment_failure_never_resumes(self):
        """Job 연결 실패 시 정지된 합성 프로세스를 종료하고 재개하지 않는다."""
        job = Mock()
        job.assign.side_effect = OSError("synthetic")

        def create(*arguments):
            process = arguments[-1]._obj
            process.process, process.thread = 123, 456
            return 1

        job.kernel.CreateProcessW.side_effect = create

        with patch("src.agents.resource_probe.WindowsJob", return_value=job):
            with self.assertRaises(OSError):
                launch_synthetic("normal", WorkerLimits())

        job.kernel.ResumeThread.assert_not_called()
        job.kernel.TerminateProcess.assert_called_once_with(123, 1)
        job.close.assert_called_once()

    def test_appcontainer_assignment_precedes_resume(self):
        """AppContainer에도 생성→Job 연결→콜백→재개→감시 순서를 적용한다."""
        container = object.__new__(AppContainer)
        container.kernel = Mock()
        container.sid = ct.c_void_p(1)
        order = []

        def attributes(*arguments):
            arguments[-1]._obj.value = 128
            return 1

        def create(*arguments):
            process = arguments[-1]._obj
            process.process, process.thread, process.pid = 123, 456, 789
            order.append("create")
            return 1

        container.kernel.InitializeProcThreadAttributeList.side_effect = attributes
        container.kernel.CreateProcessW.side_effect = create
        container.kernel.ResumeThread.side_effect = lambda handle: order.append("resume") or 0
        job = Mock()
        job.assign.side_effect = lambda handle: order.append("assign")
        resources = {"status": "succeeded", "exit_code": 0, "cleanup_confirmed": True}
        job.monitor.side_effect = lambda *args, **kwargs: order.append("monitor") or resources

        with patch("src.agents.appcontainer_native.WindowsJob", return_value=job):
            result = container.launch("python.exe", [], {}, ".", limits=WorkerLimits(), on_started=lambda pid: order.append("callback"))

        self.assertEqual(order, ["create", "assign", "callback", "resume", "monitor"])
        self.assertTrue(launch_succeeded(result))
        job.close.assert_called_once()
        result["resources"]["cleanup_confirmed"] = False
        self.assertFalse(launch_succeeded(result))


@unittest.skipUnless(os.name == "nt", "Windows Job Object 검사")
class NativeResourceTest(unittest.TestCase):
    def check_clean(self, result, status):
        self.assertEqual(result["status"], status, result)
        self.assertTrue(result["cleanup_confirmed"], result)
        self.assertEqual(result["active_processes"], 0)

    def test_normal(self):
        """정상 합성 작업은 완료되고 최대 커밋 메모리 통계를 남긴다."""
        result = launch_synthetic("normal", WorkerLimits(5, 128, 1))
        self.check_clean(result, "succeeded")
        self.assertGreater(result["peak_job_memory_bytes"], 0)

    def test_memory_limit(self):
        """64MiB Job에 최대 256MiB 요청을 시도하고 OS 제한 통지를 확인한다."""
        result = launch_synthetic("memory", WorkerLimits(5, 64, 1))
        self.check_clean(result, "resource_exceeded")
        self.assertIn(10, result["resource_messages"])
        # peak 통계와 강제 설정을 혼동하지 않고 OS에 실제 설정된 한도를 대조한다.
        self.assertEqual(result["configured_job_memory_bytes"], 64 * 1024 * 1024)

    def test_process_limit(self):
        """프로세스 상한 1이면 추가 자식 생성을 거부하고 Job 전체를 정리한다."""
        result = launch_synthetic("spawn", WorkerLimits(5, 128, 1))
        self.check_clean(result, "resource_exceeded")
        self.assertIn(3, result["resource_messages"])

    def test_timeout(self):
        """30초 대기 payload를 사용자 지정 1초 상한으로 종료한다."""
        result = launch_synthetic("sleep", WorkerLimits(1, 128, 1))
        self.check_clean(result, "timed_out")
        self.assertLess(result["elapsed_seconds"], 8)

    def test_cancel(self):
        """외부 취소 신호를 받으면 시간 상한을 기다리지 않고 종료한다."""
        cancelled = threading.Event()
        timer = threading.Timer(0.3, cancelled.set)
        timer.start()

        try:
            result = launch_synthetic("sleep", WorkerLimits(5, 128, 1), cancel_event=cancelled)
        finally:
            timer.cancel()

        self.check_clean(result, "cancelled")

    def test_descendants_are_drained(self):
        """루트가 먼저 종료돼도 허용된 자식 프로세스를 남기지 않는다."""
        result = launch_synthetic("descendant", WorkerLimits(5, 128, 2))
        self.check_clean(result, "succeeded")
        self.assertGreaterEqual(result["total_processes"], 2)
        self.assertEqual(result["configured_process_limit"], 2)

    def test_last_handle_close_kills_process(self):
        """마지막 Job 핸들을 닫으면 아직 정지된 worker도 OS가 종료한다."""
        def close_before_resume(job, process):
            job.close()
            self.assertEqual(job.kernel.WaitForSingleObject(process, 3000), 0)
            raise RuntimeError("합성 종료 검증 완료")

        with self.assertRaisesRegex(RuntimeError, "합성 종료 검증 완료"):
            launch_synthetic("sleep", WorkerLimits(5, 128, 1), on_assigned=close_before_resume)


if __name__ == "__main__":
    unittest.main()

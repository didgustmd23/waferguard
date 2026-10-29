"""입력 실행 저장소(실행 기록·승인 DB·작업 폴더)의 보호 생성·보유·의미상 권한 검사를 실제 ACL로 확인한다.

AppContainer 실행은 모의한다(OS 운송만). 승인·실행기·worker 계산·부모 검증은 실제 코드를 쓴다.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

from src.agents import protected_store as store
from src.agents import tabular_input_service as service
from src.agents import tabular_input_transport as transport
from src.agents.protected_store import StoreUnavailable
from src.agents.task_plan import TaskProblem
# 모듈로 가져와 TabularInputTest의 테스트가 이 모듈에서 다시 수집되지 않게 한다.
from tests.agents import test_tabular_input as inputs
from tests.agents.test_tabular_input import RUNTIME_ID, drive


AUTHENTICATED_USERS = "S-1-5-11"


def icacls(path, *arguments):
    executable = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
    return subprocess.run([str(executable), str(path), *arguments, "/Q"], capture_output=True, check=False).returncode


class StoreFixture(unittest.TestCase):
    """TabularInputTest의 합성 입력·모의 운송을 위임으로 쓴다(그 테스트들은 다시 실행하지 않는다). 저장소 루트는 runs다."""

    def setUp(self):
        self.fixture = inputs.TabularInputTest("test_inspect_and_materialize_consume_separate_approvals")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def __getattr__(self, name):
        if name == "fixture":
            raise AttributeError(name)
        return getattr(self.fixture, name)

    def ready(self, name="inspect1"):
        self.create(self.inspect, name)
        self.assertEqual(drive(self.inspect, name)["status"], "completed")
        return self.root / "runs"

    def assert_store_error(self, call, code="INPUT_STORE_PERMISSION_INVALID"):
        with self.assertRaises(StoreUnavailable) as caught:
            call()
        self.assertEqual(caught.exception.code, code)


class CreationTest(StoreFixture):
    def test_store_is_protected_from_creation_and_records_inherit_only_trusted(self):
        runs = self.ready()
        trusted = set(store.trusted_principals())
        info = store._security(runs)
        self.assertTrue(info["protected"])
        self.assertFalse(any(flags & 0x10 for _, flags, _, _ in info["dacl"]))
        self.assertEqual({sid for kind, _, _, sid in info["dacl"] if kind == 0}, trusted)
        # 실행 기록·승인 DB·상태·이벤트·완료 결과는 신뢰 주체 ACE만 갖는다(worker·다른 사용자 없음).
        run = runs / "inspect" / "inspect1"
        for name in ("selection.json", "approvals.sqlite3", "events.jsonl"):
            with self.subTest(file=name):
                self.assertTrue(store.check_object(run / name, store.trusted_principals()))
        self.assertTrue(store.check_tree(run, store.trusted_principals()))

    def test_preexisting_unprotected_store_is_rejected_and_not_fixed(self):
        runs = self.root / "runs"
        runs.mkdir()
        before = store._security(runs)
        self.assert_store_error(lambda: self.create(self.inspect, "inspect1"))
        self.assertEqual(list(runs.iterdir()), [])
        self.assertEqual(store._security(runs), before)

    def test_store_creation_failure_is_classified(self):
        with patch.object(store, "_create_protected", side_effect=OSError(5, "RUNTIME_ACL_SETUP_FAILED")):
            self.assert_store_error(lambda: self.create(self.inspect, "inspect1"), "INPUT_STORE_PREPARE_FAILED")
        self.assertFalse((self.root / "runs").exists())

    def test_junction_store_is_rejected(self):
        target = self.root / "elsewhere"
        target.mkdir()
        if subprocess.run(["cmd", "/c", "mklink", "/J", str(self.root / "runs"), str(target)],
                          capture_output=True, check=False).returncode:
            self.skipTest("이 환경에서는 junction을 만들 수 없습니다")
        self.addCleanup(lambda: (self.root / "runs").exists() and os.rmdir(self.root / "runs"))
        self.assert_store_error(lambda: self.create(self.inspect, "inspect1"))
        self.assertEqual(list(target.iterdir()), [])


class ReuseCheckTest(StoreFixture):
    """기존 보호 영역 재사용 시 소유자·DACL·상속·쓰기 권한·레이블을 검사하고 고치지 않는다."""

    def changed(self, target, *arguments):
        runs = self.ready()
        path = target(runs)
        if icacls(path, *arguments) != 0:
            self.skipTest("이 환경에서 시험용 ACL을 바꿀 수 없습니다")
        before = store._security(path)
        self.assert_store_error(lambda: self.inspect.status("inspect1"))
        self.assert_store_error(lambda: self.inspect.verified("inspect1"))
        self.assertEqual(store._security(path), before)

    def test_broad_write_on_approval_db_is_rejected(self):
        self.changed(lambda runs: runs / "inspect/inspect1/approvals.sqlite3", "/grant", f"*{AUTHENTICATED_USERS}:(M)")

    def test_delete_or_permission_change_on_run_record_is_rejected(self):
        for right in ("(D)", "(WDAC)", "(WO)"):
            with self.subTest(right=right):
                self.changed(lambda runs: runs / "inspect/inspect1/selection.json", "/grant", f"*{AUTHENTICATED_USERS}:{right}")
                shutil.rmtree(self.root / "runs")

    def test_child_folder_only_change_is_rejected(self):
        self.changed(lambda runs: next((runs / "inspect/inspect1/attempts").iterdir()), "/grant",
                     f"*{AUTHENTICATED_USERS}:(OI)(CI)(M)")

    def test_store_root_inheritance_or_broad_ace_is_rejected(self):
        self.changed(lambda runs: runs, "/inheritance:e")
        shutil.rmtree(self.root / "runs")
        self.changed(lambda runs: runs, "/grant", f"*{AUTHENTICATED_USERS}:(OI)(CI)(M)")

    def test_action_folder_and_low_label_are_checked(self):
        self.changed(lambda runs: runs / "inspect", "/grant", f"*{AUTHENTICATED_USERS}:(M)")
        shutil.rmtree(self.root / "runs")
        self.changed(lambda runs: runs / "inspect/inspect1/selection.json", "/setintegritylevel", "L")


class HoldTest(StoreFixture):
    def test_store_and_ancestors_cannot_be_renamed_during_execution(self):
        runs = self.root / "runs"
        attempts = []

        def swapping(request, root, cancel, **kwargs):
            for path in (runs, runs.parent):
                try:
                    os.rename(path, str(path) + "_swapped")
                    attempts.append(path.name)
                except OSError:
                    pass
            return self.fixture.transport(request, root, cancel, **kwargs)

        from tests.agents.test_tabular_input import transport as fake
        self.fixture.transport = fake
        self.mock.stop()
        with patch.object(service, "run_input_worker", side_effect=swapping):
            self.create(self.inspect, "inspect1")
            self.assertEqual(drive(self.inspect, "inspect1")["status"], "completed")
        self.assertEqual(attempts, [])
        # 연산이 끝나면 보유를 놓는다.
        os.rename(runs, self.root / "runs_moved")

    def test_hold_is_released_after_exception(self):
        self.create(self.inspect, "inspect1")
        # 실행 중 예외(실행기가 실패로 기록하거나 그대로 올림) 뒤에도 저장소 보유를 남기지 않는다.
        with patch.object(service.InputTool, "_request", side_effect=RuntimeError("boom")):
            try:
                drive(self.inspect, "inspect1")
            except Exception:
                pass
        with patch.object(service.TabularInputService, "_open", side_effect=RuntimeError("boom")), self.assertRaises(RuntimeError):
            self.inspect.status("inspect1")
        os.rename(self.root / "runs", self.root / "runs_moved")


class ReviewFixTest(StoreFixture):
    """r3 검토 보강: 보유 뒤 경로 재확인, 목록 오류, 상위 링크, 예약 이름, 실행 기록 객체 검사."""

    def test_path_rechecked_after_hold(self):
        self.ready()
        real, calls = store._root, []

        def second_fails(path):
            calls.append(path)
            if len(calls) == 2:
                raise ValueError("link appeared")
            return real(path)

        with patch.object(store, "_root", side_effect=second_fails):
            with self.assertRaises(StoreUnavailable) as caught:
                with store.store_lease(self.root / "runs", store.trusted_principals()):
                    pass
        self.assertEqual(caught.exception.code, "STORE_PERMISSION_INVALID")
        os.rename(self.root / "runs", self.root / "runs_moved")

    def test_listing_error_fails_tree_check(self):
        runs = self.ready()

        def walk(path, followlinks, onerror):
            onerror(OSError(5, "denied"))
            return iter(())

        with patch.object(store.os, "walk", side_effect=walk):
            self.assertFalse(store.check_tree(runs, store.trusted_principals()))

    def test_store_parent_link_is_rejected(self):
        target = self.root / "elsewhere"
        target.mkdir()
        if subprocess.run(["cmd", "/c", "mklink", "/J", str(self.root / "task"), str(target)],
                          capture_output=True, check=False).returncode:
            self.skipTest("이 환경에서는 junction을 만들 수 없습니다")
        self.addCleanup(lambda: (self.root / "task").exists() and os.rmdir(self.root / "task"))
        item = service.TabularInputService("inspect", root=self.root / "task" / "input_runs")
        with self.assertRaises(StoreUnavailable) as caught:
            item.create("inspect1", snapshot_directory=str(self.root / "snapshot"), snapshot=asdict(self.snapshot),
                        spec=self.spec.to_dict(), limits=asdict(self.limits), runtime_id=RUNTIME_ID)
        self.assertEqual(caught.exception.code, "INPUT_STORE_PREPARE_FAILED")
        self.assertEqual(list(target.iterdir()), [])

    def test_workers_is_reserved_run_name(self):
        with self.assertRaises(ValueError):
            self.create(self.inspect, "workers")

    def test_transport_record_with_other_write_is_not_used(self):
        def widened(request, root, cancel, **kwargs):
            outcome = inputs.transport(request, root, cancel, **kwargs)
            icacls(Path(root) / "transport.json", "/grant", f"*{AUTHENTICATED_USERS}:(M)")
            return outcome

        self.mock.stop()
        with patch.object(service, "run_input_worker", side_effect=widened):
            self.create(self.inspect, "inspect1")
            state = drive(self.inspect, "inspect1")
        # 넓은 쓰기 권한이 붙은 실행 기록은 결과 검증에 쓰지 않는다.
        self.assertNotEqual(state["status"], "completed")
        self.assertEqual(self.inspect.runtime_failure("inspect1"), "INPUT_STORE_PERMISSION_INVALID")


class JobFolderTest(unittest.TestCase):
    """실행 직전 작업 폴더 권한 검사: 코드·입력은 작업 SID 읽기·실행만, 쓰기는 자기 output·scratch만."""

    def setUp(self):
        import tempfile
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "workers" / ("a" * 32)
        for name in ("input", "tools", "output/result", "scratch/task"):
            (self.root / name).mkdir(parents=True)
        (self.root / "input/settings.json").write_text("{}", encoding="utf-8")
        (self.root / "tools/worker.py").write_text("x", encoding="utf-8")
        self.request = MagicMock(output_directory=str(self.root / "output/result"),
                                 scratch_directory=str(self.root / "scratch/task"))
        # 실제 작업 SID 대신 형식이 같은 합성 package SID를 쓴다. ACL 판정은 실제 icacls·보안 조회로 한다.
        self.sid = "S-1-15-2-1-2-3-4-5-6-7"

    def grant(self, path, permission):
        # 파일에 상속 플래그를 붙이면 icacls가 성공을 돌려주고도 ACE를 넣지 않으므로 폴더에만 붙인다.
        flags = "(OI)(CI)" if Path(path).is_dir() else ""
        if icacls(path, "/grant", f"*{self.sid}:{flags}{permission}") != 0:
            self.skipTest("이 환경에서 시험용 ACL을 바꿀 수 없습니다")

    def expected_grants(self):
        for name in ("input", "tools"):
            self.grant(self.root / name, "(RX)")
        for path in (self.request.output_directory, self.request.scratch_directory):
            self.grant(path, "(M)")
            icacls(path, "/setintegritylevel", "(OI)(CI)L")

    def test_expected_job_grants_pass(self):
        self.expected_grants()
        self.assertTrue(transport._job_permissions_ok(self.root, self.request, self.sid))

    def test_wrong_grants_are_rejected(self):
        cases = {
            "worker_write_tools": lambda: self.grant(self.root / "tools", "(M)"),
            "worker_write_input_file": lambda: self.grant(self.root / "input/settings.json", "(W)"),
            "worker_on_job_root": lambda: self.grant(self.root, "(RX)"),
            "broad_group_on_tools": lambda: icacls(self.root / "tools/worker.py", "/grant", f"*{AUTHENTICATED_USERS}:(M)"),
            "other_job_sid_on_input": lambda: icacls(self.root / "input", "/grant", "*S-1-15-2-9-9-9-9-9-9-9:(OI)(CI)(RX)"),
            "low_label_on_input": lambda: icacls(self.root / "input/settings.json", "/setintegritylevel", "L"),
            "worker_full_on_output": lambda: self.grant(self.request.output_directory, "(F)"),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                self.setUp()
                self.expected_grants()
                change()
                self.assertFalse(transport._job_permissions_ok(self.root, self.request, self.sid))

    def test_transport_does_not_launch_on_job_permission_failure(self):
        from tests.agents.test_worker_runtime import TransportTest
        case = TransportTest("test_launch_uses_approved_shared_runtime_and_holds_it")
        case.setUp()
        self.addCleanup(case.doCleanups)
        icacls(case.root / "tools", "/grant", f"*{AUTHENTICATED_USERS}:(OI)(CI)(M)")
        outcome = case.run_worker()
        self.assertEqual(outcome["runtime_error"], "INPUT_STORE_PERMISSION_INVALID")
        case.container.launch.assert_not_called()
        self.assertTrue(outcome["profile_cleanup"])
        case.container.close.assert_called_once()


class FlowTest(StoreFixture):
    def test_store_failure_is_shown_as_store_problem_not_input_error(self):
        from src.agents import tabular_input_flow as flow

        @flow._store_problem
        def failing():
            raise StoreUnavailable("INPUT_STORE_PERMISSION_INVALID")

        with self.assertRaises(TaskProblem) as caught:
            failing()
        self.assertEqual(caught.exception.code, "INPUT_STORE_PERMISSION_INVALID")
        self.assertIn("입력 데이터 문제가 아닙니다", caught.exception.message)
        self.assertIn("원본 Python 환경으로 대신 실행하지 않았고", caught.exception.message)

    def test_job_permission_failure_is_reported_through_runtime_failure(self):
        def denied(request, root, cancel, **kwargs):
            outcome = {"request_id": request.request_id, "profile_cleanup": True, "runtime_error": "INPUT_STORE_PERMISSION_INVALID"}
            (Path(root) / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
            return outcome

        self.mock.stop()
        with patch.object(service, "run_input_worker", side_effect=denied):
            self.create(self.inspect, "inspect1")
            self.assertEqual(drive(self.inspect, "inspect1")["status"], "failed_runtime")
        self.assertEqual(self.inspect.runtime_failure("inspect1"), "INPUT_STORE_PERMISSION_INVALID")

    def test_approval_binding_and_single_use_are_kept(self):
        runs = self.ready()
        _, tool, manager = self.inspect._open("inspect1")
        task = manager.approvals.get_task(manager.scope_for("ingest"))
        self.assertEqual(task["status"], "committed")
        self.assertEqual(tool.arguments["runtime_id"], RUNTIME_ID)
        self.assertTrue(store.check_tree(runs / "inspect/inspect1", store.trusted_principals()))


if __name__ == "__main__":
    unittest.main()

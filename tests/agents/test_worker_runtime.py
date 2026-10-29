"""공유 worker 런타임의 준비·재사용·권한 보호·승인 ID 결합과 입력 transport 연결을 검사한다.

저장소 권한은 임시 폴더에 실제 icacls·보안 정보 조회로 설정·확인한다(관리자 권한 불필요). AppContainer 실행은 모의한다.
실제 격리 실행 확인은 test_tabular_v2_appcontainer·별도 OS 시험이 담당한다.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from src.agents import tabular_input_transport as transport
from src.agents import worker_runtime as runtime
from src.agents.appcontainer_native import AppContainer
from src.agents.tabular_input_worker import shared_runtime
from src.agents.task_plan import TaskProblem
from src.agents.worker_contract import WorkerLimits, WorkerRequest


AUTHENTICATED_USERS = "S-1-5-11"


def tree_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def icacls(path, *arguments):
    executable = Path(os.environ["SystemRoot"]) / "System32" / "icacls.exe"
    return subprocess.run([str(executable), str(path), *arguments, "/Q"], capture_output=True, check=False).returncode


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        # runtime_files()가 실제로 고르는 배치를 작게 흉내 낸다(실행 파일·DLL·표준 라이브러리·허용 패키지).
        self.prefix = self.temp / "env"
        for name, data in (("python.exe", b"exe"), ("python312.dll", b"dll"), ("Lib/os.py", b"os"),
                           ("Lib/json/__init__.py", b"json"), ("DLLs/_ctypes.pyd", b"pyd"),
                           ("Lib/site-packages/numpy/__init__.py", b"np"), ("Lib/site-packages/other/x.py", b"other")):
            path = self.prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        # 저장소 루트의 상위(logs)까지 두어 상위 경로 이름 변경 차단을 시험한다.
        self.base = self.temp / "logs" / "shared"
        self.capability = runtime.capability_sid()

    def approved(self):
        return runtime.current_runtime_id(self.prefix)

    def select(self, expected=None, **kwargs):
        """승인 ID로 lease를 잡았다 놓는다. 반환값은 lease 안에서 받은 정보다."""
        with runtime.runtime_lease(expected or self.approved(), prefix=self.prefix, base=self.base, **kwargs) as selected:
            return selected

    def assert_code(self, code, expected=None, **kwargs):
        with self.assertRaises(runtime.RuntimeUnavailable) as caught:
            self.select(expected, **kwargs)
        self.assertEqual(caught.exception.code, code)


class PrepareReuseTest(RuntimeFixture):
    def test_first_prepare_then_reuse_without_copy(self):
        source = tree_hashes(self.prefix)
        first = self.select()
        self.assertTrue(first.prepared)
        self.assertEqual((first.directory, first.capability), (self.base / first.runtime_id, self.capability))
        # 허용 목록 밖 패키지(other)는 복사하지 않는다.
        self.assertEqual(sorted(tree_hashes(first.directory / "runtime")),
                         sorted(name for name in source if not name.startswith("Lib/site-packages/other")))
        with patch.object(runtime.shutil, "copyfile", wraps=shutil.copyfile) as copied, \
                patch.object(runtime, "_create_protected", wraps=runtime._create_protected) as protected:
            second = self.select()
        self.assertFalse(second.prepared)
        self.assertEqual((second.runtime_id, second.directory), (first.runtime_id, first.directory))
        # 재사용은 복사·권한 변경을 하지 않는다. 원본 환경은 그대로다.
        self.assertEqual((copied.call_count, protected.call_count), (0, 0))
        self.assertEqual(tree_hashes(self.prefix), source)
        self.assertEqual([path.name for path in self.base.iterdir()], [first.runtime_id])

    def test_changed_configuration_has_new_id_and_keeps_old(self):
        first = self.select()
        before = tree_hashes(first.directory)
        # 크기가 같은 내용 변경도 다른 구성이다(크기·버전 문자열만으로 판단하지 않음).
        (self.prefix / "Lib/os.py").write_bytes(b"OS")
        second = self.select()
        self.assertTrue(second.prepared)
        self.assertNotEqual(second.runtime_id, first.runtime_id)
        self.assertEqual(tree_hashes(first.directory), before)

    def test_identity_includes_python_version(self):
        manifest = runtime.source_manifest(self.prefix)
        self.assertNotEqual(runtime.runtime_id(manifest), runtime.runtime_id(dict(manifest, python="0.0.0")))


class ProtectionTest(RuntimeFixture):
    """저장소 루트·버전 폴더·하위 객체의 소유자·DACL·상속·레이블을 의미상 권한으로 확인한다."""

    def security(self, path):
        return runtime._security(path)

    def test_store_and_version_use_protected_trusted_dacl_with_capability_read(self):
        selected = self.select()
        user = runtime._token_identity()[0]
        trusted = {user, *runtime.TRUSTED}
        base, version = self.security(self.base), self.security(selected.directory)
        self.assertTrue(base["protected"] and version["protected"])
        self.assertEqual({sid for kind, _, _, sid in base["dacl"] if kind == 0} - {"S-1-3-4"}, trusted)
        self.assertEqual({sid for kind, _, _, sid in version["dacl"] if kind == 0} - {"S-1-3-4"}, trusted | {self.capability})
        file = self.security(selected.python)
        self.assertFalse(file["protected"])
        # 파일은 버전 폴더의 ACE만 상속한다. capability는 읽기·실행뿐이다.
        allowed = {sid: mask for kind, _, mask, sid in file["dacl"] if kind == 0}
        self.assertEqual(set(allowed) - {"S-1-3-4"}, trusted | {self.capability})
        self.assertEqual(allowed[self.capability] & ~runtime.READ_EXECUTE, 0)
        # 하위 객체의 허용 ACE는 버전 폴더와 같은 주체·마스크다(생성 시 보호 SD에서 내려온 값).
        self.assertEqual({(sid, mask) for kind, _, mask, sid in file["dacl"] if kind == 0},
                         {(sid, mask) for kind, _, mask, sid in version["dacl"] if kind == 0})

    def changed(self, change):
        selected = self.select()
        if change(selected.directory) != 0:
            self.skipTest("이 환경에서 시험용 ACL을 바꿀 수 없습니다")
        snapshot = tree_hashes(selected.directory)
        before = self.security(selected.directory / "runtime/Lib/os.py")
        self.assert_code("RUNTIME_PERMISSION_INVALID", selected.runtime_id)
        # 거부하고 권한·내용을 조용히 복구하지 않는다.
        self.assertEqual(tree_hashes(selected.directory), snapshot)
        self.assertEqual(self.security(selected.directory / "runtime/Lib/os.py"), before)

    def test_broad_group_write_on_file_is_rejected(self):
        self.changed(lambda d: icacls(d / "runtime/Lib/os.py", "/grant", f"*{AUTHENTICATED_USERS}:(M)"))

    def test_child_folder_only_permission_change_is_rejected(self):
        self.changed(lambda d: icacls(d / "runtime/Lib/json", "/grant", f"*{AUTHENTICATED_USERS}:(OI)(CI)(M)"))

    def test_delete_or_permission_change_right_is_rejected(self):
        for right in ("(D)", "(WDAC)", "(WO)"):
            with self.subTest(right=right):
                self.changed(lambda d: icacls(d / "runtime/Lib/os.py", "/grant", f"*{AUTHENTICATED_USERS}:{right}"))
                shutil.rmtree(self.base)

    def test_capability_write_is_rejected(self):
        self.changed(lambda d: icacls(d / "runtime/python.exe", "/grant", f"*{self.capability}:(M)"))

    def test_inherited_permissions_on_version_are_rejected(self):
        # 버전 폴더가 상위 상속을 다시 받으면(보호 해제) 거부한다.
        self.changed(lambda d: icacls(d, "/inheritance:e"))

    def test_low_integrity_label_is_rejected(self):
        self.changed(lambda d: icacls(d / "runtime/Lib/os.py", "/setintegritylevel", "L"))

    def test_unprotected_existing_store_is_not_used_or_fixed(self):
        # 다른 경로로 만든 보호되지 않은 저장소 루트(상속 ACL)는 쓰지 않고 고치지도 않는다.
        self.base.mkdir(parents=True)
        before = self.security(self.base)
        self.assert_code("RUNTIME_PERMISSION_INVALID")
        self.assertEqual(list(self.base.iterdir()), [])
        self.assertEqual(self.security(self.base), before)

    def test_store_with_broad_inheritable_write_is_rejected(self):
        self.select()
        if icacls(self.base, "/grant", f"*{AUTHENTICATED_USERS}:(OI)(CI)(M)") != 0:
            self.skipTest("이 환경에서 시험용 ACL을 바꿀 수 없습니다")
        self.assert_code("RUNTIME_PERMISSION_INVALID")

    def test_protected_folder_has_no_inherited_or_extra_entries_from_creation(self):
        # 생성 직후(권한 변경 단계 없이) 보호·비상속 ACE만 있어야 생성 중 개입 틈이 없다.
        user = runtime._token_identity()[0]
        target = self.temp / "created"
        runtime._create_protected(target, user, self.capability)
        info = self.security(target)
        self.assertTrue(info["protected"])
        self.assertFalse(any(flags & 0x10 for _, flags, _, _ in info["dacl"]))
        self.assertEqual({sid for kind, _, _, sid in info["dacl"] if kind == 0}, {user, *runtime.TRUSTED, self.capability})
        with self.assertRaises(FileExistsError):
            runtime._create_protected(target, user, self.capability)

    def test_store_is_held_during_preparation(self):
        # 루트 검사 뒤 준비(원본 해시·복사) 중에도 루트·상위 경로를 다른 폴더로 바꿀 수 없어야 한다.
        attempts = []
        real = runtime.source_manifest

        def swapping(prefix):
            for path in (self.base, self.base.parent):
                try:
                    os.rename(path, path.with_name(path.name + "_swapped"))
                    attempts.append(path.name)
                except OSError:
                    pass
            return real(prefix)

        with patch.object(runtime, "source_manifest", side_effect=swapping):
            self.assertTrue(self.select(self.approved()).prepared)
        self.assertEqual(attempts, [])

    def test_store_query_failure_releases_hold(self):
        self.select()
        # 루트 보안 조회의 예상 밖 예외도 권한 이상으로 멈추고, 루트 보유를 남기지 않는다.
        with patch.object(runtime, "_security", side_effect=ValueError("label")):
            self.assert_code("RUNTIME_PERMISSION_INVALID")
        os.rename(self.base, self.temp / "released")

    def test_lease_blocks_rename_of_version_store_and_ancestors(self):
        approved = self.approved()
        with runtime.runtime_lease(approved, prefix=self.prefix, base=self.base) as selected:
            for path in (selected.directory, self.base, self.base.parent):
                with self.subTest(path=path.name), self.assertRaises(OSError):
                    os.rename(path, path.with_name(path.name + "_moved"))
        # 놓은 뒤에는 보유가 끝난다(같은 사용자의 이동은 신뢰 주체 조작으로 범위 밖).
        os.rename(self.base, self.temp / "moved")
        self.assertTrue((self.temp / "moved" / approved).is_dir())

    def test_junction_store_is_rejected(self):
        target = self.temp / "elsewhere"
        target.mkdir()
        self.base.parent.mkdir(parents=True)
        if subprocess.run(["cmd", "/c", "mklink", "/J", str(self.base), str(target)], capture_output=True, check=False).returncode:
            self.skipTest("이 환경에서는 junction을 만들 수 없습니다")
        self.addCleanup(lambda: self.base.exists() and os.rmdir(self.base))
        self.assert_code("RUNTIME_PREPARE_FAILED")
        self.assertEqual(list(target.iterdir()), [])

    def test_semantic_rules(self):
        user = "S-1-5-21-1-2-3-1001"
        trusted = (user, *runtime.TRUSTED)
        good = {"owner": user, "protected": True, "label": None,
                "dacl": [(0, 3, 0x1F01FF, user), (0, 3, 0x1200A9, self.capability), (1, 0, 0x10000, AUTHENTICATED_USERS)]}
        self.assertTrue(runtime._permission_ok(good, trusted, self.capability, root=True))
        cases = {
            "untrusted_owner": dict(good, owner=AUTHENTICATED_USERS),
            "null_dacl": dict(good, dacl=None),
            "not_protected_root": dict(good, protected=False),
            "low_label": dict(good, label=0x1000),
            "other_sid_read_only": dict(good, dacl=good["dacl"] + [(0, 0, 0x1200A9, "S-1-5-32-545")]),
            "all_app_packages": dict(good, dacl=good["dacl"] + [(0, 0, 0x1200A9, "S-1-15-2-1")]),
            "capability_generic_write": dict(good, dacl=[(0, 3, 0x40000000, self.capability)]),
            "capability_delete": dict(good, dacl=[(0, 3, 0x10000 | 0x1200A9, self.capability)]),
            "unknown_ace_type": dict(good, dacl=good["dacl"] + [(5, 0, None, None)]),
        }
        for name, info in cases.items():
            with self.subTest(case=name):
                self.assertFalse(runtime._permission_ok(info, trusted, self.capability, root=True))
        self.assertTrue(runtime._permission_ok(dict(good, protected=False), trusted, self.capability, root=False))


class TamperTest(RuntimeFixture):
    def tampered(self, change):
        published = self.select()
        change(published.directory)
        snapshot = tree_hashes(published.directory)
        self.assert_code("RUNTIME_VERIFY_FAILED", published.runtime_id)
        # 실패해도 공개본을 고치거나 지우지 않는다.
        self.assertEqual(tree_hashes(published.directory), snapshot)
        return published

    def test_modified_missing_extra_and_record_changes_are_rejected(self):
        cases = {
            "modified": lambda d: (d / "runtime/Lib/os.py").write_bytes(b"XX"),
            "missing": lambda d: (d / "runtime/Lib/json/__init__.py").unlink(),
            "extra_file": lambda d: (d / "runtime/Lib/extra.py").write_bytes(b"x"),
            "extra_folder": lambda d: (d / "runtime/Lib/empty").mkdir(),
            "extra_top": lambda d: (d / "note.txt").write_bytes(b"x"),
            "record": lambda d: (d / "manifest.json").write_text(json.dumps({"runtime_id": d.name}), encoding="utf-8"),
            "record_other_files": lambda d: (d / "manifest.json").write_text(json.dumps(dict(
                json.loads((d / "manifest.json").read_text(encoding="utf-8")), python="0.0.0")), encoding="utf-8"),
            "no_record": lambda d: (d / "manifest.json").unlink(),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                published = self.tampered(change)
                shutil.rmtree(published.directory)

    def test_partial_version_folder_is_not_used_or_replaced(self):
        self.select()
        approved = self.approved()
        shutil.rmtree(self.base / approved)
        partial = self.base / approved
        (partial / "runtime").mkdir(parents=True)
        (partial / "runtime/python.exe").write_bytes(b"exe")
        # 보호 DACL이 없는 부분 폴더는 권한에서, 보호된 부분 폴더는 내용에서 거부한다. 어느 쪽도 고치거나 바꾸지 않는다.
        self.assert_code("RUNTIME_PERMISSION_INVALID")
        shutil.rmtree(partial)
        runtime._create_protected(partial, runtime._token_identity()[0], self.capability)
        (partial / "runtime").mkdir()
        (partial / "runtime/python.exe").write_bytes(b"exe")
        self.assert_code("RUNTIME_VERIFY_FAILED")
        self.assertEqual(tree_hashes(partial), {"runtime/python.exe": hashlib.sha256(b"exe").hexdigest()})

    def test_hard_link_and_junction_are_rejected(self):
        published = self.select()
        target = published.directory / "runtime/Lib/os.py"
        target.unlink()
        os.link(self.prefix / "Lib/os.py", target)
        self.assert_code("RUNTIME_VERIFY_FAILED", published.runtime_id)
        shutil.rmtree(published.directory)
        published = self.select()
        folder = published.directory / "runtime/Lib/json"
        shutil.rmtree(folder)
        if subprocess.run(["cmd", "/c", "mklink", "/J", str(folder), str(self.prefix / "Lib/json")],
                          capture_output=True, check=False).returncode:
            self.skipTest("이 환경에서는 junction을 만들 수 없습니다")
        self.addCleanup(lambda: folder.exists() and os.rmdir(folder))
        self.assert_code("RUNTIME_VERIFY_FAILED", published.runtime_id)


class ApprovedIdTest(RuntimeFixture):
    """승인에 고정한 ID만 실행하고, 환경이 바뀌어도 다른 ID를 조용히 고르지 않는다."""

    def test_source_change_after_approval_keeps_approved_published_runtime(self):
        approved = self.approved()
        self.select(approved)
        (self.prefix / "Lib/os.py").write_bytes(b"OS")
        selected = self.select(approved)
        self.assertEqual((selected.runtime_id, selected.prepared), (approved, False))
        self.assertEqual([path.name for path in self.base.iterdir()], [approved])

    def test_source_change_before_preparation_stops_without_new_version(self):
        approved = self.approved()
        (self.prefix / "Lib/os.py").write_bytes(b"OS")
        self.assert_code("RUNTIME_CHANGED", approved)
        self.assertEqual(list(self.base.iterdir()), [])

    def test_invalid_or_other_expected_id_is_rejected(self):
        approved = self.approved()
        self.select(approved)
        for expected in ("bad", None, "F" * 64):
            with self.subTest(expected=expected), self.assertRaises(runtime.RuntimeUnavailable) as caught:
                with runtime.runtime_lease(expected, prefix=self.prefix, base=self.base):
                    pass
            self.assertEqual(caught.exception.code, "RUNTIME_VERIFY_FAILED")
        # 형식만 맞는 다른 ID는 원본 구성과 달라 준비하지 않는다.
        self.assert_code("RUNTIME_CHANGED", "f" * 64)
        trusted = (runtime._token_identity()[0], *runtime.TRUSTED)
        self.assertEqual(runtime.verify_runtime(self.base / approved, "f" * 64, capability=self.capability, trusted=trusted),
                         "RUNTIME_VERIFY_FAILED")


class FailureConcurrencyTest(RuntimeFixture):
    def test_concurrent_prepare_publishes_one_version(self):
        barrier = threading.Barrier(4, timeout=30)
        real = runtime._create_protected

        def synchronized(path, user, capability=None):
            real(path, user, capability)
            if capability is not None:
                barrier.wait()

        results, errors = [], []

        def work():
            try:
                results.append(self.select())
            except Exception as error:
                errors.append(error)

        with patch.object(runtime, "_create_protected", side_effect=synchronized):
            threads = [threading.Thread(target=work) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(len({item.runtime_id for item in results}), 1)
        self.assertEqual(sum(item.prepared for item in results), 1)
        self.assertEqual([path.name for path in self.base.iterdir()], [results[0].runtime_id])
        self.assertEqual([path.name for path in self.base.parent.iterdir() if path.name.startswith(".worker_runtime-")], [])

    def test_copy_failure_leaves_nothing_and_next_prepare_succeeds(self):
        calls = {"count": 0}
        real = shutil.copyfile

        def failing(source, target):
            calls["count"] += 1
            if calls["count"] == 3:
                raise OSError("disk full")
            return real(source, target)

        with patch.object(runtime.shutil, "copyfile", side_effect=failing):
            self.assert_code("RUNTIME_PREPARE_FAILED")
        self.assertEqual(list(self.base.iterdir()), [])
        self.assertTrue(self.select().prepared)

    def test_copied_content_mismatch_is_not_published(self):
        real = shutil.copyfile

        def corrupt(source, target):
            real(source, target)
            if Path(target).name == "os.py":
                Path(target).write_bytes(b"XX")

        with patch.object(runtime.shutil, "copyfile", side_effect=corrupt):
            self.assert_code("RUNTIME_PREPARE_FAILED")
        self.assertEqual(list(self.base.iterdir()), [])

    def test_permission_setup_failures_stop_without_publish(self):
        real = runtime._create_protected

        def fail_version(path, user, capability=None):
            if capability is not None:
                raise OSError(5, "RUNTIME_ACL_SETUP_FAILED")
            return real(path, user, capability)

        with patch.object(runtime, "_create_protected", side_effect=fail_version):
            self.assert_code("RUNTIME_PREPARE_FAILED")
        self.assertEqual(list(self.base.iterdir()), [])
        shutil.rmtree(self.base)
        # 저장소 루트 보호 생성 실패·capability 조회 실패도 준비 실패로 분류하고 루트를 공개하지 않는다.
        with patch.object(runtime, "_create_protected", side_effect=OSError(5, "RUNTIME_ACL_SETUP_FAILED")):
            self.assert_code("RUNTIME_PREPARE_FAILED")
        self.assertFalse(self.base.exists())
        self.assertEqual([path.name for path in self.base.parent.iterdir()], [])
        with patch.object(runtime, "capability_sid", side_effect=AttributeError("DeriveCapabilitySidsFromName")):
            self.assert_code("RUNTIME_PREPARE_FAILED")

    def test_cancel_during_prepare_cleans_staging(self):
        cancel = threading.Event()
        cancel.set()
        self.assert_code("RUNTIME_CANCELLED", cancel_event=cancel)
        self.assertEqual(list(self.base.iterdir()), [])

    def test_invalid_source_is_classified(self):
        with patch.object(runtime, "runtime_files", side_effect=ValueError("link")):
            with self.assertRaises(runtime.RuntimeUnavailable) as caught:
                runtime.current_runtime_id(self.prefix)
        self.assertEqual(caught.exception.code, "RUNTIME_SOURCE_INVALID")


class WorkerBindingTest(RuntimeFixture):
    def test_worker_accepts_only_selected_runtime(self):
        published = self.select()
        python = published.python
        self.assertEqual(shared_runtime(python, published.runtime_id), python.parent)
        other = "e" * 64
        self.assertIsNone(shared_runtime(python, other))
        self.assertIsNone(shared_runtime(python, "bad"))
        self.assertIsNone(shared_runtime(self.prefix / "python.exe", published.runtime_id))
        # 폴더 이름이 같아도 완료 기록의 ID가 다르면 거부한다.
        fake = self.temp / "fake" / other / "runtime"
        fake.mkdir(parents=True)
        (fake.parent / "manifest.json").write_text(json.dumps({"runtime_id": published.runtime_id}), encoding="utf-8")
        self.assertIsNone(shared_runtime(fake / "python.exe", other))
        (fake.parent / "manifest.json").write_text("{", encoding="utf-8")
        self.assertIsNone(shared_runtime(fake / "python.exe", other))

    def test_native_launch_rejects_network_or_extra_capabilities(self):
        container = AppContainer("waferguard.test.capability")
        # 고정 런타임 capability 하나만 허용한다. 네트워크·다른 이름 파생 capability·중복·list는 거부한다.
        for value in (("S-1-15-3-1",), (self.capability,) * 2, [self.capability], (1,),
                      ("S-1-15-3-1024-9-9-9-9-9-9-9-9",), None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "APPCONTAINER_CAPABILITY_INVALID"):
                container.launch(Path("python.exe"), [], {}, self.temp, capabilities=value)
        self.assertFalse(container.created)


class TransportTest(RuntimeFixture):
    """입력 transport가 승인 ID의 공유 런타임만 실행하고 실행 중 보유하며 작업 폴더에 복사하지 않는지 확인한다."""

    def setUp(self):
        super().setUp()
        self.root = self.temp / "workers" / ("a" * 32)
        for name in ("input", "tools", "output/result", "scratch/task"):
            (self.root / name).mkdir(parents=True)
        (self.root / "input/settings.json").write_text("{}", encoding="utf-8")
        self.request = WorkerRequest(task_id="a" * 32, run_id="b" * 32, actor_id="actor", stage="ingest", action_id="run_ingest",
                                     source_id="c" * 64, config_id="d" * 64, plan_id="e" * 64, policy_version="tabular_input_v2",
                                     limits=WorkerLimits(60, 2048, 2), input_files=(str(self.root / "input/settings.json"),),
                                     dependency_directories=(),
                                     output_directory=str(self.root / "output/result"),
                                     scratch_directory=str(self.root / "scratch/task"))
        self.container = MagicMock()
        self.container.create.return_value = "S-1-15-2-1"
        self.container.close.return_value = True
        self.container.launch.return_value = {"status": "exited", "exit_code": 0}
        for target, value in (("AppContainer", MagicMock(return_value=self.container)),
                              ("prepare_writable_directory", MagicMock())):
            patcher = patch.object(transport, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.transport_acl = patch.object(transport, "_acl").start()
        self.addCleanup(patch.stopall)
        self.environment = patch.dict(os.environ, {"SystemRoot": os.environ.get("SystemRoot", "C:\\Windows"),
                                                    "LOCALAPPDATA": str(self.temp), "USERPROFILE": str(self.temp),
                                                    "PATH": "C:\\parent", "PYTHONPATH": "C:\\parent"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def run_worker(self, cancel=None, runtime_id=None, base=None):
        with patch.object(runtime.sys, "prefix", str(self.prefix)):
            return transport.run_input_worker(self.request, self.root, cancel or threading.Event(),
                                              runtime_id=runtime_id or self.approved(), runtime_base=base or self.base)

    def test_launch_uses_approved_shared_runtime_and_holds_it(self):
        renamed = []

        def launch(executable, arguments, *args, **kwargs):
            # 실행 중에는 저장소·버전 폴더 이름 변경이 막혀야 한다.
            for path in (self.base / arguments[-1], self.base):
                try:
                    os.rename(path, path.with_name(path.name + "_moved"))
                    renamed.append(path.name)
                except OSError:
                    pass
            return {"status": "exited", "exit_code": 0}

        self.container.launch.side_effect = launch
        approved = self.approved()
        outcome = self.run_worker(runtime_id=approved)
        self.assertEqual(renamed, [])
        selected = self.base / approved
        self.assertEqual(outcome["runtime"]["id"], approved)
        args, kwargs = self.container.launch.call_args
        self.assertEqual(args[0], selected / "runtime/python.exe")
        self.assertEqual(args[1][-1], approved)
        self.assertEqual(args[1][:3], ["-I", "-S", "-B"])
        environment = args[2]
        # 부모 PATH·PYTHONPATH는 넘기지 않는다. 임시·Matplotlib 쓰기는 작업 scratch다.
        self.assertEqual(environment["PATH"], str(selected / "runtime"))
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual((environment["TEMP"], environment["TMP"]), (str(self.root / "scratch/task"),) * 2)
        self.assertEqual(environment["MPLCONFIGDIR"], str(self.root / "scratch/task/mpl"))
        self.assertEqual(kwargs["capabilities"], (self.capability,))
        # 작업 SID에는 input·tools RX만 준다. 공유 런타임·작업 runtime 폴더 권한은 바꾸지 않는다.
        self.assertEqual([call.args[0] for call in self.transport_acl.call_args_list], [self.root / "input", self.root / "tools"])
        self.assertFalse((self.root / "runtime").exists())
        self.assertEqual(json.loads((self.root / "transport.json").read_text(encoding="utf-8")), outcome)
        self.assertTrue(outcome["profile_cleanup"])

    def test_changed_environment_is_not_run_with_new_id(self):
        approved = self.approved()
        (self.prefix / "Lib/os.py").write_bytes(b"OS")
        outcome = self.run_worker(runtime_id=approved)
        self.assertEqual(outcome["runtime_error"], "RUNTIME_CHANGED")
        self.container.create.assert_not_called()
        self.container.launch.assert_not_called()
        self.assertEqual(list(self.base.iterdir()), [])

    def test_runtime_failure_is_recorded_without_launch(self):
        self.base.mkdir(parents=True)
        outcome = self.run_worker()
        self.assertEqual(outcome["runtime_error"], "RUNTIME_PERMISSION_INVALID")
        self.assertNotIn("launch", outcome)
        self.container.create.assert_not_called()
        self.container.launch.assert_not_called()
        self.assertTrue(outcome["profile_cleanup"])
        self.assertEqual(json.loads((self.root / "transport.json").read_text(encoding="utf-8")), outcome)

    def test_cancel_before_and_after_preparation(self):
        cancel = threading.Event()
        cancel.set()
        outcome = self.run_worker(cancel)
        self.assertEqual(outcome["launch"], {"status": "cancelled"})
        self.container.create.assert_not_called()
        self.run_worker()
        self.container.reset_mock()
        outcome = self.run_worker(cancel)
        # 공개본이 있으면 선택은 끝나고 실행 직전 취소로 멈춘다. 프로필은 정리한다.
        self.assertEqual(outcome["launch"], {"status": "cancelled"})
        self.container.launch.assert_not_called()
        self.container.close.assert_called_once()

    def test_overlapping_runtime_and_job_folders_are_rejected(self):
        for base in (self.root / "input" / "rt", self.root / "output" / "rt", self.root / "rt"):
            with self.subTest(base=base):
                with self.assertRaises(ValueError):
                    self.run_worker(base=base)
                self.container.create.assert_not_called()
                self.container.launch.assert_not_called()
                shutil.rmtree(base)

    def test_launch_exception_still_cleans_profile_releases_and_records(self):
        self.container.launch.side_effect = OSError("APPCONTAINER_PROCESS_CREATE_FAILED")
        with self.assertRaises(OSError):
            self.run_worker()
        self.container.close.assert_called_once()
        recorded = json.loads((self.root / "transport.json").read_text(encoding="utf-8"))
        self.assertTrue(recorded["profile_cleanup"])
        self.assertNotIn("launch", recorded)
        # 예외 뒤에도 보유를 놓는다.
        os.rename(self.base, self.temp / "moved")
        os.rename(self.temp / "logs", self.temp / "logs_moved")


class ServiceTest(unittest.TestCase):
    """실제 승인 DB·실행기로 런타임 ID 결합·실패 분류·결과 검증을 본다(OS 운송만 모의)."""

    def setUp(self):
        from tests.agents.test_tabular_input import TabularInputTest
        self.fixture = TabularInputTest("test_inspect_and_materialize_consume_separate_approvals")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def run_with(self, outcome_change, *, runtime_id=None):
        from src.agents import tabular_input_service as service
        from tests.agents.test_tabular_input import RUNTIME_ID, drive, transport as fake
        seen = []

        def changed(request, root, cancel, **kwargs):
            seen.append(kwargs)
            outcome = outcome_change(fake(request, root, cancel, **kwargs), root)
            (Path(root) / "transport.json").write_text(json.dumps(outcome), encoding="utf-8")
            return outcome

        self.fixture.mock.stop()
        with patch.object(service, "run_input_worker", side_effect=changed):
            self.fixture.inspect.create("inspect1", snapshot_directory=str(self.fixture.root / "snapshot"),
                                        snapshot=__import__("dataclasses").asdict(self.fixture.snapshot),
                                        spec=self.fixture.spec.to_dict(), limits=__import__("dataclasses").asdict(self.fixture.limits),
                                        runtime_id=runtime_id or RUNTIME_ID, max_output_bytes=8 * 1024 * 1024)
            state = drive(self.fixture.inspect, "inspect1")
        return state, seen

    def test_approved_id_reaches_preview_transport_and_results(self):
        from tests.agents.test_tabular_input import RUNTIME_ID
        state, seen = self.run_with(lambda outcome, root: outcome)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(seen, [{"runtime_id": RUNTIME_ID}])
        _, tool, manager = self.fixture.inspect._open("inspect1")
        self.assertEqual(tool.arguments["runtime_id"], RUNTIME_ID)
        _, result = self.fixture.inspect.verified("inspect1")
        self.assertEqual(result["inspection"]["status"], "ready")
        binding = json.loads((self.fixture.inspect.verified("inspect1")[0] / "binding.json").read_text(encoding="utf-8"))
        self.assertEqual(binding["packet"]["runtime_id"], RUNTIME_ID)

    def test_preview_summary_and_config_include_runtime_id(self):
        from tests.agents.test_tabular_input import RUNTIME_ID
        self.fixture.mock.stop()
        self.fixture.create(self.fixture.inspect, "inspect1")
        for _ in range(3):
            state = self.fixture.inspect.status("inspect1")
            if state["status"] == "awaiting_approval":
                break
            self.fixture.inspect.next("inspect1", expected_stage=state["stage"], expected_revision=state["revision"])
        summary = self.fixture.inspect.preview_approval("inspect1").to_dict()["summary"]
        self.assertEqual(summary["runtime_id"], RUNTIME_ID)
        other = "1" * 64
        self.fixture.inspect.create("inspect2", snapshot_directory=str(self.fixture.root / "snapshot"),
                                    snapshot=__import__("dataclasses").asdict(self.fixture.snapshot), spec=self.fixture.spec.to_dict(),
                                    limits=__import__("dataclasses").asdict(self.fixture.limits), runtime_id=other)
        # 런타임 ID만 달라도 실행 설정(config_id)·승인 범위가 달라진다.
        first, second = (self.fixture.inspect._open(name)[1].config_id for name in ("inspect1", "inspect2"))
        self.assertNotEqual(first, second)

    def test_materialize_requires_inspection_with_same_runtime(self):
        from tests.agents.test_tabular_input import RUNTIME_ID, drive
        self.fixture.create(self.fixture.inspect, "inspect1")
        self.assertEqual(drive(self.fixture.inspect, "inspect1")["status"], "completed")
        dataclasses = __import__("dataclasses")
        with self.assertRaises(ValueError):
            self.fixture.materialize.create("materialize1", snapshot_directory=str(self.fixture.root / "snapshot"),
                                            snapshot=dataclasses.asdict(self.fixture.snapshot), spec=self.fixture.spec.to_dict(),
                                            limits=dataclasses.asdict(self.fixture.limits), runtime_id="1" * 64,
                                            inspection_run="inspect1")
        self.assertFalse((self.fixture.root / "runs/materialize/materialize1").exists())
        self.assertNotEqual(RUNTIME_ID, "1" * 64)

    def test_real_delegation_rejects_preview_with_other_runtime_id(self):
        from dataclasses import asdict
        from src.agents import task_consent
        from tests.agents.test_tabular_input import RUNTIME_ID
        self.fixture.mock.stop()
        self.fixture.create(self.fixture.inspect, "inspect1")
        for _ in range(3):
            state = self.fixture.inspect.status("inspect1")
            if state["status"] == "awaiting_approval":
                break
            self.fixture.inspect.next("inspect1", expected_stage=state["stage"], expected_revision=state["revision"])
        task_id = "a" * 32
        content = {"allowed": [["tabular_inspect", "ingest"]], "limits": asdict(self.fixture.limits)}
        task = {"task_id": task_id, "consents": {"A": task_consent.grant("A", task_id, content)}}
        events = self.fixture.inspect._open("inspect1")[1].states.directory / "events.jsonl"

        def issue(recorded_id):
            # 실제 동의 판정기와 서비스 위임 기록을 쓴다. 작업 기록의 ID와 미리보기 요약의 ID가 다르면 발급하지 않는다.
            authorize = task_consent.authorizer(task, "A", "tabular_inspect", "ingest", expected={},
                                                bound={"run_name": "inspect1", "runtime_id": recorded_id})
            return self.fixture.inspect.record_delegated_approval("inspect1", self.fixture.inspect.preview_approval("inspect1"),
                                                                  authorize=authorize)

        with self.assertRaisesRegex(ValueError, "동의 후 확보한 값과 다릅니다"):
            issue("1" * 64)
        self.assertNotIn("DELEGATED_APPROVAL_RECORDED", events.read_text(encoding="utf-8"))
        self.assertEqual(self.fixture.inspect.status("inspect1")["actions"], [{"id": "approve"}])
        issued = issue(RUNTIME_ID)
        self.assertRegex(issued["approval_id"], r"^[0-9a-f]+")
        self.assertIn("DELEGATED_APPROVAL_RECORDED", events.read_text(encoding="utf-8"))

    def test_runtime_error_stops_as_runtime_failure_with_code(self):
        state, _ = self.run_with(lambda outcome, root: {"request_id": outcome["request_id"], "profile_cleanup": True,
                                                        "runtime_error": "RUNTIME_CHANGED"})
        self.assertEqual(state["status"], "failed_runtime")
        self.assertEqual(self.fixture.inspect.runtime_failure("inspect1"), "RUNTIME_CHANGED")

    def test_runtime_record_must_match_approved_id(self):
        for change in (lambda outcome, root: dict(outcome, runtime={"id": "1" * 64, "prepared": False, "timings": {}}),
                       lambda outcome, root: dict(outcome, runtime={"id": "bad"}),
                       lambda outcome, root: {key: value for key, value in outcome.items() if key != "runtime"}):
            self.fixture.setUp()
            state, _ = self.run_with(change)
            # 형식이 맞아도 승인한 ID와 다른 실행 기록은 결과로 등록하지 않는다.
            self.assertEqual(state["status"], "blocked_input")

    def test_worker_rejects_packet_runtime_mismatch(self):
        from src.agents.tabular_input_worker import execute_request
        from tests.agents.test_tabular_input import RUNTIME_ID
        self.fixture.mock.stop()
        self.fixture.create(self.fixture.inspect, "inspect1")
        _, tool, _ = self.fixture.inspect._open("inspect1")
        request = tool._request("f" * 32)
        root = tool._stage(request)
        self.assertEqual(execute_request(request, root, "1" * 64).status, "blocked_input")
        diagnostic = json.loads((root / "scratch/task/diagnostic.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnostic, {"stage": "REQUEST", "code": "INPUT_INVALID"})
        self.assertNotEqual(RUNTIME_ID, "1" * 64)


class FlowMessageTest(unittest.TestCase):
    def test_runtime_failure_shows_cause_and_next_action(self):
        from src.agents import tabular_input_flow as flow

        class Service:
            action = "inspect"

            def __init__(self, code):
                self.code = code

            def status(self, name):
                return {"status": "failed_runtime", "stage": "ingest", "revision": 3}

            def runtime_failure(self, name):
                return self.code

        for code in runtime.MESSAGES:
            with self.subTest(code=code), self.assertRaises(TaskProblem) as caught:
                flow._drive(Service(code), "run", {}, None, {})
            self.assertEqual(caught.exception.code, code)
            self.assertIn(runtime.MESSAGES[code], caught.exception.message)
            self.assertIn("원본 Python 환경으로 대신 실행하지 않았고", caught.exception.message)
        with self.assertRaises(TaskProblem) as caught:
            flow._drive(Service(None), "run", {}, None, {})
        self.assertEqual(caught.exception.code, "INPUT_PREPARATION_FAILED")

    def test_delegation_binds_recorded_runtime_id(self):
        from types import SimpleNamespace
        from src.agents import tabular_input_flow as flow
        captured = {}

        class Service:
            action = "inspect"

            def status(self, name):
                return {"status": "awaiting_approval", "stage": "ingest", "revision": 1}

            def preview_approval(self, name):
                return SimpleNamespace(to_dict=lambda: {"summary": {}})

            def record_delegated_approval(self, name, preview, *, authorize):
                raise RuntimeError("stop")

        def authorizer(record, decision, module_id, stage, *, expected, bound):
            captured.update(bound)
            return lambda preview: None

        record = {"input_selection": {"runtime_id": "9" * 64}, "snapshot": {"snapshot": {"sha256": "8" * 64}}}
        scope = {"input_code_id": "7" * 64, "read_spec": {}, "max_output_bytes": 1}
        with self.assertRaises(RuntimeError):
            flow._drive(Service(), "run", record, SimpleNamespace(authorizer=authorizer), scope)
        # 위임 발급은 작업 기록에 고정한 런타임 ID와 승인 미리보기 요약의 값이 같을 때만 통과한다.
        self.assertEqual(captured["runtime_id"], "9" * 64)

    def test_prepare_input_stops_when_runtime_id_cannot_be_fixed(self):
        from src.agents import tabular_input_flow as flow
        record = {"task_id": "a" * 32, "consents": {"A": {}}, "limits": {}}
        with patch.object(flow, "_verify_snapshot"), patch.object(flow.task_consent, "check", return_value={}), \
                patch.object(flow, "_spec", return_value=MagicMock(to_dict=lambda: {})), \
                patch.object(flow, "code_id", return_value="c" * 64), \
                patch.object(flow, "current_runtime_id", side_effect=runtime.RuntimeUnavailable("RUNTIME_SOURCE_INVALID")):
            expected = {"input_code_id": "c" * 64, "input_policy": flow.POLICY_VERSION, "read_spec": {}, "input_attempt": 1,
                        "max_output_bytes": flow.MAX_OUTPUT_BYTES, "previous_inspection_id": None,
                        "previous_inspection_sha256": None, "limits": {}}
            with patch.object(flow.task_consent, "check", return_value=expected), self.assertRaises(TaskProblem) as caught:
                flow.prepare_input(record, Path("."), MagicMock())
        self.assertEqual(caught.exception.code, "RUNTIME_SOURCE_INVALID")
        self.assertNotIn("input_selection", record)


if __name__ == "__main__":
    unittest.main()

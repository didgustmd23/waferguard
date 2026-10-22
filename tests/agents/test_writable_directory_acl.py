"""ACL 명령은 모킹하며, 임시 빈 폴더의 대상 제한과 실패 시 중단을 검사한다."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.agents import writable_directory_acl as acl


class WritableDirectoryAclTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "output/result"
        self.target.mkdir(parents=True)

        for name, kwargs in (*self.owner_patches(), ("_acl", {})):
            patcher = patch.object(acl, name, **kwargs)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

        patcher = patch.object(acl.subprocess, "run", return_value=Mock(returncode=0))
        self.command = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, {"SystemRoot": "C:\\Windows"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def owner_patches(self):
        return (("_owner_user_sid", {"return_value": "S-1-5-21-123"}),)

    def prepare(self, target=None):
        acl.prepare_writable_directory(self.root, target or self.target, "S-1-15-2-123")

    def test_caller_right_is_additive_non_inheriting_and_worker_remains_modify(self):
        self.prepare()
        command = self.command.call_args.args[0]
        self.assertEqual(command[1:], [str(self.target), "/grant", "*S-1-5-21-123:(WO)", "/Q"])
        self._acl.assert_called_once_with(self.target, "S-1-15-2-123", "M", low=True)
        self.assertFalse(self.command.call_args.kwargs.get("shell", False))

    def test_existing_content_is_rejected_without_acl_change(self):
        (self.target / "existing.txt").touch()

        with self.assertRaises(acl.DirectoryAclError) as caught:
            self.prepare()

        self.assertEqual(caught.exception.acl_operation, "TARGET_CHECK")
        self._owner_user_sid.assert_not_called()
        self.command.assert_not_called()
        self._acl.assert_not_called()

    def test_root_parent_and_input_are_rejected(self):
        input_directory = self.root / "input"
        input_directory.mkdir()

        for target in (self.root, self.target.parent, input_directory):
            with self.subTest(target=target), self.assertRaises(acl.DirectoryAclError):
                self.prepare(target)

        self.command.assert_not_called()

    def test_owner_mismatch_never_grants_rights(self):
        self._owner_user_sid.side_effect = acl.DirectoryAclError("OWNER_MISMATCH")

        with self.assertRaises(acl.DirectoryAclError):
            self.prepare()

        self.command.assert_not_called()
        self._acl.assert_not_called()

    def test_worker_cannot_receive_caller_rights(self):
        self._owner_user_sid.return_value = "S-1-15-2-123"

        with self.assertRaises(acl.DirectoryAclError):
            self.prepare()

        self.command.assert_not_called()

    def test_caller_grant_failure_stops_before_worker_grant(self):
        self.command.return_value.returncode = 5

        with self.assertRaises(acl.DirectoryAclError) as caught:
            self.prepare()

        self.assertEqual(caught.exception.acl_operation, "CALLER_WRITE_OWNER")
        self._acl.assert_not_called()

    def test_worker_grant_and_integrity_failures_are_distinct(self):
        for message, operation in (
            ("PROBE_ACL_SETUP_FAILED", "WORKER_GRANT"),
            ("PROBE_INTEGRITY_SETUP_FAILED", "LOW_INTEGRITY"),
        ):
            self._acl.side_effect = OSError(5, message)

            with self.subTest(operation=operation), self.assertRaises(acl.DirectoryAclError) as caught:
                self.prepare()

            self.assertEqual(caught.exception.acl_operation, operation)
            self.assertEqual(caught.exception.errno, 5)

    def test_timeout_does_not_continue_to_worker_grant(self):
        self.command.side_effect = TimeoutError("synthetic_private_detail")

        with self.assertRaises(TimeoutError):
            self.prepare()

        self._acl.assert_not_called()


class OwnerDecisionTest(unittest.TestCase):
    """소유자 허용 규칙만 검사한다. 토큰·소유자 조회는 고정 값으로 대체한다."""

    USER = "S-1-5-21-1-2-3-1001"

    def decide(self, owner, elevated, default_owner=acl.ADMINISTRATORS_SID, user=USER):
        with patch.object(acl, "_token_identity", return_value=(user, elevated, default_owner)), \
                patch.object(acl, "_directory_owner", return_value=owner):
            return acl._owner_user_sid(Path("unused"))

    def test_user_owned_folder_is_allowed_with_or_without_elevation(self):
        self.assertEqual(self.decide(self.USER, False), self.USER)
        self.assertEqual(self.decide(self.USER, True), self.USER)

    def test_administrators_owner_only_with_elevated_token_returns_user_sid(self):
        self.assertEqual(self.decide(acl.ADMINISTRATORS_SID, True), self.USER)

        with self.assertRaises(acl.DirectoryAclError) as caught:
            self.decide(acl.ADMINISTRATORS_SID, False)

        self.assertEqual(caught.exception.acl_operation, "OWNER_MISMATCH")

    def test_other_owners_are_rejected_even_when_elevated(self):
        for owner in ("S-1-5-21-1-2-3-1002", "S-1-5-18", "S-1-5-32-545", "S-1-1-0", "S-1-5-32-544-1"):
            for elevated in (False, True):
                with self.subTest(owner=owner, elevated=elevated), self.assertRaises(acl.DirectoryAclError):
                    self.decide(owner, elevated)

    def test_token_default_owner_and_user_account_must_match_exception(self):
        """승격돼도 기본 소유자가 사용자이거나, 사용자가 SYSTEM·서비스 계정이면 관리자 소유 폴더를 거부한다."""
        for default_owner, user in ((self.USER, self.USER), ("S-1-5-32-545", self.USER),
                                    (acl.ADMINISTRATORS_SID, "S-1-5-18"), (acl.ADMINISTRATORS_SID, "S-1-5-80-1-2")):
            with self.subTest(default_owner=default_owner, user=user), self.assertRaises(acl.DirectoryAclError):
                self.decide(acl.ADMINISTRATORS_SID, True, default_owner, user)

    def test_elevation_flag_must_be_true_not_truthy(self):
        with self.assertRaises(acl.DirectoryAclError):
            self.decide(acl.ADMINISTRATORS_SID, 1)


class AdministratorsOwnedPreparationTest(WritableDirectoryAclTest):
    """승격 관리자 소유의 새 빈 폴더에도 기존 대상 제한·실패 중단 검사를 그대로 적용한다.

    소유자 판단은 실제 _owner_user_sid 규칙을 쓰고 토큰·소유자 조회만 고정 값으로 바꾼다.
    WO는 사용자 SID에만, worker는 기존 M·Low만 받는다.
    """

    def owner_patches(self):
        for name, value in (("_token_identity", ("S-1-5-21-123", True, acl.ADMINISTRATORS_SID)), ("_directory_owner", acl.ADMINISTRATORS_SID)):
            patcher = patch.object(acl, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

        return (("_owner_user_sid", {"wraps": acl._owner_user_sid}),)

    def test_owner_mismatch_never_grants_rights(self):
        with patch.object(acl, "_token_identity", return_value=("S-1-5-21-123", False, acl.ADMINISTRATORS_SID)):
            with self.assertRaises(acl.DirectoryAclError):
                self.prepare()

        self.command.assert_not_called()
        self._acl.assert_not_called()

    def test_worker_cannot_receive_caller_rights(self):
        with patch.object(acl, "_token_identity", return_value=("S-1-15-2-123", True, acl.ADMINISTRATORS_SID)):
            with self.assertRaises(acl.DirectoryAclError):
                self.prepare()

        self.command.assert_not_called()

    def test_group_sid_never_receives_write_owner(self):
        self.prepare()
        self.assertNotIn(acl.ADMINISTRATORS_SID, " ".join(self.command.call_args.args[0]))


@unittest.skipUnless(os.name == "nt", "Windows 토큰·ACL 조회 전용")
class Win32QueryTest(unittest.TestCase):
    """실제 토큰·폴더 소유자 조회만 수행한다. ACL은 변경하지 않는다."""

    def test_token_and_owner_queries_return_sid_strings(self):
        user, elevated, default_owner = acl._token_identity()
        self.assertTrue(user.startswith("S-1-5-"))
        self.assertIs(type(elevated), bool)
        self.assertTrue(default_owner.startswith("S-1-5-"))

        with tempfile.TemporaryDirectory() as directory:
            owner = acl._directory_owner(directory)
            self.assertTrue(owner.startswith("S-1-"))
            self.assertEqual(acl._owner_user_sid(Path(directory)), user)

    def test_junction_target_is_rejected_before_owner_query(self):
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            (root / "output").mkdir()
            link = root / "output" / "result"
            made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(real)], capture_output=True)

            if made.returncode:
                self.skipTest("junction을 만들 수 없는 환경")

            with patch.object(acl, "_owner_user_sid") as owner, patch.object(acl.subprocess, "run") as command:
                with self.assertRaises((acl.DirectoryAclError, ValueError)):
                    acl.prepare_writable_directory(root, link, "S-1-15-2-123")

            owner.assert_not_called()
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()

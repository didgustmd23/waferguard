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

        for name, kwargs in (
            ("_owner_user_sid", {"return_value": "S-1-5-21-123"}),
            ("_acl", {}),
        ):
            patcher = patch.object(acl, name, **kwargs)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

        patcher = patch.object(acl.subprocess, "run", return_value=Mock(returncode=0))
        self.command = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, {"SystemRoot": "C:\\Windows"})
        patcher.start()
        self.addCleanup(patcher.stop)

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


if __name__ == "__main__":
    unittest.main()

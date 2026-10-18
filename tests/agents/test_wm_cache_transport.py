"""실데이터·OS 권한 변경 없이 worker 준비 실패의 진단·정리 계약을 검증한다."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.agents import wm_cache_transport as transport


class CacheTransportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        for name in ('runtime', 'input', 'tools', 'output', 'scratch'):
            (self.root / name).mkdir()

        self.request = SimpleNamespace(task_id='a' * 32, request_id='b' * 64,
                                       output_directory=str(self.root / 'output'),
                                       scratch_directory=str(self.root / 'scratch'), limits=None)
        self.cancel = Mock()
        self.cancel.is_set.return_value = False
        self.container = Mock()
        self.container.create.return_value = 'synthetic-sid'
        self.container.close.return_value = True
        self.container.launch.return_value = {'status': 'exited', 'exit_code': 0}

        for name, replacement in (
            ('AppContainer', Mock(return_value=self.container)),
            ('code_manifest', Mock(return_value={'synthetic.py': 'synthetic'})),
            ('runtime_files', Mock(return_value=[])), ('_acl', Mock()),
            ('WORKER_FILES', ()), ('ENTRY', 'synthetic.py'),
            ('probe_environment', Mock(return_value={})),
            ('_fingerprint', Mock(return_value={'sha256': 'synthetic'})),
        ):
            mocking = patch.object(transport, name, replacement)
            mocking.start()
            self.addCleanup(mocking.stop)

        # 실제 소스 복사·권한 변경 대신 단계 순서와 실패 분류만 검사한다.
        mocking = patch.object(transport.shutil, 'copyfile')
        mocking.start()
        self.addCleanup(mocking.stop)

    def run_transport(self):
        outcome = transport.run_cache_worker(self.request, self.root, self.cancel)
        self.assertEqual(outcome, json.loads((self.root / 'transport.json').read_text()))
        self.container.close.assert_called_once()
        return outcome

    def test_profile_denied_is_recorded_without_private_exception(self):
        self.container.create.side_effect = PermissionError(5, 'private source value')
        outcome = self.run_transport()
        self.assertEqual(outcome['preparation_failure'],
                         {'stage': 'PROFILE_CREATE', 'code': 'SETUP_ACCESS_DENIED', 'os_error': 5})
        self.assertNotIn('private', json.dumps(outcome))
        self.assertNotIn('launch', outcome)
        self.container.launch.assert_not_called()

    def test_runtime_inventory_failure(self):
        transport.runtime_files.side_effect = ValueError('private package path')
        outcome = self.run_transport()
        self.assertEqual(outcome['preparation_failure']['stage'], 'RUNTIME_INVENTORY')
        self.assertEqual(outcome['preparation_failure']['code'], 'SETUP_CONTRACT_REJECTED')
        self.container.create.assert_not_called()

    def test_acl_denied_never_starts_worker(self):
        transport._acl.side_effect = PermissionError(13, 'private path')
        outcome = self.run_transport()
        self.assertEqual(outcome['preparation_failure']['stage'], 'INPUT_ACL')
        self.container.launch.assert_not_called()

    def test_success_retains_launch_and_cleanup(self):
        outcome = self.run_transport()
        self.assertEqual(outcome['launch']['exit_code'], 0)
        self.assertTrue(outcome['profile_cleanup'])
        self.assertNotIn('preparation_failure', outcome)

    def test_cleanup_failure_not_hidden(self):
        self.container.close.side_effect = OSError(5, 'private cleanup path')
        outcome = self.run_transport()
        self.assertFalse(outcome['profile_cleanup'])
        self.assertEqual(outcome['preparation_failure']['stage'], 'PROFILE_CLEANUP')

    def test_cancel_before_launch(self):
        self.cancel.is_set.return_value = True
        outcome = self.run_transport()
        self.assertEqual(outcome['launch'], {'status': 'cancelled'})
        self.container.launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()

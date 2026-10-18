"""선택 실행하는 합성 WM 캐시의 실제 Windows 격리 검사."""

import os
from pathlib import Path
import unittest

from tests.agents import test_wm_cache as fixtures
from src.agents import wm_cache_cli as cache_cli


@unittest.skipUnless(os.environ.get('WAFERGUARD_TEST_APPCONTAINER') == '1', '실제 Windows 격리 검사 선택 실행')
class WmCacheAppContainerTest(unittest.TestCase):
    setUp = fixtures.WmCacheTest.setUp
    transport = fixtures.WmCacheTest.transport
    proposal_transport = fixtures.WmCacheTest.proposal_transport
    commit_transport = fixtures.WmCacheTest.commit_transport
    check_transport = fixtures.WmCacheTest.check_transport
    approval = fixtures.WmCacheTest.approval

    def test_native_cache(self):
        """앞선 증거는 합성으로 준비하고 새 숫자 캐시는 실제 AppContainer+Job에서 생성한다."""
        approval = self.approval()
        result = self.manager.run_next(approval_id=approval.approval_id)
        diagnostics = [path.read_text() for path in (self.root / 'workers').glob('*/scratch/task/diagnostic.json')]
        outcomes = [path.read_text() for path in (self.root / 'workers').glob('*/transport.json')]
        self.assertEqual(result.status, 'completed', (result, diagnostics, outcomes))
        self.assertTrue(self.tool.worker_started)
        status = cache_cli.status(cache_cli.open_run('cache'))
        self.assertEqual(status['cache_status'], 'cache_complete')
        self.assertFalse(status['test_maps_read'])
        self.assertFalse(status['training_ready'])
        self.assertTrue((Path(status['local_cache_directory']) / 'train/index.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()

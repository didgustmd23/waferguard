"""선택 실행하는 실제 PyTorch·AppContainer·Job의 합성 Dataset 검사."""

import os
import unittest

from tests.agents import test_wm_dataset as fixtures
from src.agents import wm_dataset_cli as cli


@unittest.skipUnless(os.environ.get('WAFERGUARD_TEST_APPCONTAINER') == '1', '실제 Windows 격리 검사 선택 실행')
class DatasetAppContainerTest(unittest.TestCase):
    native_runtime = True
    setUp = fixtures.WmDatasetTest.setUp
    approval = fixtures.WmDatasetTest.approval

    def test_native_torch_batch(self):
        """완료 증거는 합성 대역이고 PyTorch 복사·DLL·두 구간 검사는 실제 격리 실행한다."""
        approval = self.approval()
        result = self.manager.run_next(approval_id=approval.approval_id)
        diagnostics = [path.read_text() for path in (self.base / 'workers').glob('*/scratch/task/diagnostic.json')]
        outcomes = [path.read_text() for path in (self.base / 'workers').glob('*/transport.json')]
        self.assertEqual(result.status, 'completed', (result, diagnostics, outcomes))
        self.assertTrue(self.tool.worker_started)
        status = cli.status(cli.open_run('dataset'))
        self.assertTrue(status['development_inputs_ready'])
        self.assertFalse(status['test_maps_read'])
        self.assertFalse(status['model_trained'])


if __name__ == '__main__':
    unittest.main()

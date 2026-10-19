"""합성 숫자 캐시로 Dataset 검사 승인·범위·변조·종료 계약을 확인한다."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests import test_wafer_cache_dataset as fixtures
from src.agents import wm_dataset_cli as cli
from src.agents.wm_canonical_contract import digest, file_hash
from src.agents.wm_cache_results import cache_files
from src.agents.wm_dataset_worker import execute_request
from src.agents.wm_dataset_transport import ENTRY, WORKER_FILES
from src.agents.worker_contract import WorkerLimits


class WmDatasetTest(unittest.TestCase):
    native_runtime = False

    def setUp(self):
        # 앞선 캐시 생성·승인 DB는 이 합성 fixture의 검사 대상이 아니다.
        # 기존 코어로 만든 작은 숫자 캐시를 완료 증거 대역에 연결한다.
        fixtures.CacheDatasetTest.setUp(self)
        fixtures.CacheDatasetTest.build(self, approved=True, canonical_dir=self.fixture.root)
        self.run_dir = self.base / 'runs/dataset'
        self.run_dir.parent.mkdir()
        evidence = {'cache_run': 'cache', 'run_id': 'synthetic-cache', 'plan_id': 'synthetic-plan',
                    'cache_directory': str(self.output), 'cache_files': cache_files(self.output, 32 * 1024 * 1024),
                    'cache_report': json.loads((self.output / 'cache.json').read_text())}
        evidence['evidence_id'] = digest(evidence)
        self.evidence = evidence

        for target in ('src.agents.wm_dataset_cli.read_cache_evidence', 'src.agents.wm_dataset.read_cache_evidence'):
            mocking = patch(target, return_value=evidence)
            mocking.start()
            self.addCleanup(mocking.stop)

        mocking = patch('src.agents.wm_dataset_cli._directory', return_value=self.run_dir)
        mocking.start()
        self.addCleanup(mocking.stop)

        if not self.native_runtime:
            # 모의 transport는 호스트의 PyTorch로 합성 데이터만 검사한다.
            # 실제 runtime 복사·토큰·DLL 로드는 별도 native 검사에서 확인한다.
            path = Path(sys.prefix) / 'Lib/site-packages/torch/__init__.py'
            records = {'Lib/site-packages/torch/__init__.py': file_hash(path, 1024 * 1024)}
            mocking = patch('src.agents.wm_dataset_cli.runtime_manifest', return_value=(records, str(fixtures.torch.__version__)))
            mocking.start()
            self.addCleanup(mocking.stop)
            # 코어·승인 회귀만 검사한다. 이 대역은 native 3.12 지원 승인이 아니다.
            mocking = patch('src.agents.wm_runtime.runtime_status', return_value={'supported': True})
            mocking.start()
            self.addCleanup(mocking.stop)

        cli.create('dataset', 'cache', WorkerLimits(300, 8192, 2), max_input_bytes=32 * 1024 * 1024,
                   max_runtime_bytes=8 * 1024 ** 3, confirmed=True)
        self.tool = cli.open_run('dataset')
        self.manager = cli.executor(self.tool)

    def approval(self):
        self.assertEqual(self.manager.run_next().code, 'STEP_COMPLETED')

        with patch('src.agents.wm_dataset.run_dataset_worker') as worker:
            self.assertEqual(self.manager.run_next().code, 'APPROVAL_REQUIRED')

        worker.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for('validate_contract'))

    def transport(self, request, root, signal):
        reply = execute_request(request, root)
        (Path(request.output_directory) / 'worker_result.json').write_text(reply.to_json(), encoding='utf-8')
        outcome = {'request_id': request.request_id, 'profile_cleanup': True, 'launch': {
            'status': 'exited', 'exit_code': 0 if reply.status == 'succeeded' else 2,
            'resources': {'status': 'succeeded', 'cleanup_confirmed': True, 'active_processes': 0,
                          'configured_job_memory_bytes': request.limits.memory_mib * 1024 * 1024,
                          'configured_process_limit': request.limits.processes},
        }}
        (root / 'transport.json').write_text(json.dumps(outcome), encoding='utf-8')
        return outcome

    def test_approval_completion_and_no_test_or_raw_input(self):
        """개발 두 구간만 검사하고 canonical·Test·모델을 worker에 전달하지 않는다."""
        approval = self.approval()
        before = cache_files(self.output, 32 * 1024 * 1024)
        from src.wafer_dl.dataset import check_first_batch

        def development_only(directory, split, batch_size, allow_test):
            self.assertIn(split, {'train', 'validation'})
            self.assertFalse(allow_test)
            return check_first_batch(directory, split, batch_size, allow_test)

        with patch('src.agents.wm_dataset.run_dataset_worker', side_effect=self.transport), \
                patch('src.wafer_dl.dataset.check_first_batch', side_effect=development_only):
            result = self.manager.run_next(approval_id=approval.approval_id)

        diagnostics = [path.read_text() for path in (self.base / 'workers').glob('*/scratch/task/diagnostic.json')]
        self.assertEqual(result.status, 'completed', (result, diagnostics))
        status = cli.status(cli.open_run('dataset'))
        self.assertTrue(status['development_inputs_ready'])
        self.assertFalse(status['test_maps_read'])
        self.assertFalse(status['model_trained'])
        task = self.manager.approvals.get_task(self.manager.scope_for('validate_contract'))
        source = self.tool._work_root(task['task_id']) / 'input'
        self.assertEqual({path.name for path in source.iterdir()},
                         {'analysis.json', 'files.json', 'runtime_files.json', 'request.json', 'cache'})
        self.assertFalse((source / 'cache/test').exists())
        self.assertEqual(cache_files(self.output, 32 * 1024 * 1024), before)

    def test_runtime_manifest_change_rejected(self):
        (self.run_dir / 'runtime_files.json').write_text('{}')

        with self.assertRaises(ValueError):
            cli.open_run('dataset')

    def test_changed_cache_rejected(self):
        (self.output / 'train/index.jsonl').write_bytes(b'changed')

        with self.assertRaises(ValueError):
            cli.open_run('dataset')

    def test_tampered_report_never_commits(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.transport(request, root, signal)
            path = Path(request.output_directory) / 'dataset/dataset_check.json'
            report = json.loads(path.read_text())
            report['test_maps_read'] = True
            path.write_text(json.dumps(report))
            return outcome

        with patch('src.agents.wm_dataset.run_dataset_worker', side_effect=changed):
            self.assertNotEqual(self.manager.run_next(approval_id=approval.approval_id).status, 'completed')

        self.assertNotIn('validate_contract', self.tool.states.load().artifact_refs)

    def test_cleanup_failure_never_commits(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.transport(request, root, signal)
            outcome['profile_cleanup'] = False
            return outcome

        with patch('src.agents.wm_dataset.run_dataset_worker', side_effect=changed):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, 'failed_runtime')

    def test_setup_failure_is_visible_and_never_retried(self):
        approval = self.approval()

        def failed(request, root, signal):
            outcome = {'request_id': request.request_id, 'profile_cleanup': True,
                       'preparation_failure': {'stage': 'RUNTIME_COPY', 'code': 'SETUP_ACCESS_DENIED', 'os_error': 5}}
            (root / 'transport.json').write_text(json.dumps(outcome))
            return outcome

        with patch('src.agents.wm_dataset.run_dataset_worker', side_effect=failed) as worker:
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, 'failed_runtime')
            status = cli.status(cli.open_run('dataset'))
            self.assertEqual(status['preparation_failure']['stage'], 'RUNTIME_COPY')
            self.manager.run_next(approval_id=approval.approval_id)
            worker.assert_called_once()

    def test_copied_code_has_no_project_dependency(self):
        project, copied = Path(__file__).resolve().parents[2], self.base / 'copied'

        for name in WORKER_FILES + (ENTRY,):
            target = copied / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, target)

        code = ('import sys; sys.path[:0]=[sys.argv[1],sys.argv[2]]; '
                'from src.agents.wm_dataset_worker import execute_request; '
                'from src.agents.wm_dataset_results import inspect_dataset; '
                'from src.agents.wm_dataset_runtime import decode_manifest; '
                'from src.wafer_dl.dataset import check_first_batch')
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(copied),
                                 str(Path(sys.prefix) / 'Lib/site-packages')], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

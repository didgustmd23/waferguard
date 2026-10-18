"""합성 데이터로 캐시 승인·Test 제외·색인 정합성·실패 시 미완료를 검증한다."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.agents import test_wm_transform as fixtures
from src.agents import wm_cache_cli as cache_cli
from src.agents.wm_cache_worker import execute_request
from src.agents.wm_cache_transport import ENTRY, WORKER_FILES
from src.agents.wm_canonical_contract import inventory
from src.agents.worker_contract import WorkerLimits


class WmCacheTest(unittest.TestCase):
    transport = fixtures.WmTransformTest.transport
    proposal_transport = fixtures.WmTransformTest.proposal_transport
    commit_transport = fixtures.WmTransformTest.commit_transport
    check_transport = fixtures.WmTransformTest.check_transport

    def setUp(self):
        fixtures.WmTransformTest.setUp(self)
        approval = fixtures.WmTransformTest.approval(self)

        with patch('src.agents.wm_transform.run_check_worker', side_effect=self.check_transport):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, 'completed')

        self.cache_dir = self.root / 'runs/cache'
        mocking = patch('src.agents.wm_cache_cli._directory', return_value=self.cache_dir)
        mocking.start()
        self.addCleanup(mocking.stop)
        cache_cli.create('cache', 'check', WorkerLimits(120, 2048, 2), 32 * 1024 * 1024, confirmed=True)
        self.tool = cache_cli.open_run('cache')
        self.manager = cache_cli.executor(self.tool)

    def approval(self):
        """앞 단계 재확인과 그림 확인 승인은 합성 fixture에서만 발급한다."""
        self.assertEqual(self.manager.run_next().code, 'STEP_COMPLETED')
        self.assertEqual(self.manager.run_next().code, 'STEP_COMPLETED')

        for stage in ('review_eda', 'commit_split'):
            self.assertEqual(self.manager.run_next().code, 'APPROVAL_REQUIRED')
            approval = self.manager.approvals.record_confirmed_approval(self.manager.scope_for(stage))
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).code, 'STEP_COMPLETED')

            if stage == 'review_eda':
                self.assertEqual(self.manager.run_next().code, 'STEP_COMPLETED')

        with patch('src.agents.wm_cache.run_cache_worker') as worker:
            self.assertEqual(self.manager.run_next().code, 'APPROVAL_REQUIRED')

        worker.assert_not_called()
        return self.manager.approvals.record_confirmed_approval(self.manager.scope_for('prepare_data'))

    def cache_transport(self, request, root, signal):
        reply = execute_request(request, root)
        self.validation_error = None

        if reply.status == 'succeeded':
            from src.agents.wm_cache_results import inspect_cache
            from src.agents.wm_canonical_contract import digest

            try:
                inspect_cache(Path(request.output_directory) / 'cache', request, self.tool.contract,
                              digest(self.tool.evidence), self.tool.evidence)
            except Exception as error:
                self.validation_error = repr(error)

        (Path(request.output_directory) / 'worker_result.json').write_text(reply.to_json(), encoding='utf-8')
        outcome = {'request_id': request.request_id, 'profile_cleanup': True, 'launch': {
            'status': 'exited', 'exit_code': 0 if reply.status == 'succeeded' else 2,
            'resources': {'status': 'succeeded', 'cleanup_confirmed': True, 'active_processes': 0,
                          'configured_job_memory_bytes': request.limits.memory_mib * 1024 * 1024,
                          'configured_process_limit': request.limits.processes},
        }}
        (root / 'transport.json').write_text(json.dumps(outcome), encoding='utf-8')
        return outcome

    def test_development_cache_and_no_test_decode(self):
        """Test가 같은 NPZ에 있어도 선택 배열만 복원하고 숫자 캐시·행 연결을 검증한다."""
        from src.wafer_dl.check_transform import read_checked_map

        approval = self.approval()
        split = self.tool.evidence['split_evidence']
        allowed = {row['sample_id'] for row in map(json.loads,
                   (Path(split['split_directory']) / 'split_manifest.jsonl').read_text().splitlines())
                   if row['split'] in {'train', 'validation'}}
        seen = set()
        before = inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes)

        def development_only(archive, row):
            self.assertIn(row['sample_id'], allowed)
            seen.add(row['sample_id'])
            return read_checked_map(archive, row)

        with patch('src.agents.wm_cache.run_cache_worker', side_effect=self.cache_transport), \
                patch('src.wafer_dl.check_transform.read_checked_map', side_effect=development_only):
            result = self.manager.run_next(approval_id=approval.approval_id)

        diagnostics = [path.read_text() for path in (self.root / 'workers').glob('*/scratch/task/diagnostic.json')]
        self.assertEqual(result.status, 'completed', (result, diagnostics, self.validation_error))
        self.assertEqual(seen, allowed)
        status = cache_cli.status(cache_cli.open_run('cache'))
        self.assertEqual(status['cache_status'], 'cache_complete')
        self.assertFalse(status['test_maps_read'])
        self.assertFalse(status['training_ready'])
        self.assertFalse((Path(status['local_cache_directory']) / 'test').exists())
        self.assertEqual(inventory(self.tool.contract.snapshot_directory, self.tool.contract.max_input_bytes), before)
        self.assertNotEqual(self.manager.run_next(approval_id=approval.approval_id).code, 'STEP_COMPLETED')

    def test_visual_review_is_required(self):
        self.manager.run_next()
        self.manager.run_next()

        with patch('src.agents.wm_cache.run_cache_worker') as worker:
            self.assertEqual(self.manager.run_next().code, 'APPROVAL_REQUIRED')
            self.assertEqual(self.tool.states.load().current_stage, 'review_eda')
            self.assertEqual(self.manager.run_next().code, 'APPROVAL_REQUIRED')

        worker.assert_not_called()

    def test_budget_rejected_before_new_run(self):
        other = self.root / 'runs/small'

        with patch('src.agents.wm_cache_cli._directory', return_value=other), self.assertRaises(ValueError):
            cache_cli.create('small', 'check', WorkerLimits(120, 2048, 2), 1, confirmed=True)

        self.assertFalse(other.exists())

    def test_changed_check_rejected(self):
        (Path(self.tool.evidence['check_directory']) / 'examples.png').write_bytes(b'changed')

        with self.assertRaises(ValueError):
            cache_cli.open_run('cache')

    def test_tampered_cache_never_commits(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.cache_transport(request, root, signal)
            (Path(request.output_directory) / 'cache/train/index.jsonl').write_bytes(b'{}\n')
            return outcome

        with patch('src.agents.wm_cache.run_cache_worker', side_effect=changed):
            result = self.manager.run_next(approval_id=approval.approval_id)

        self.assertNotEqual(result.status, 'completed')
        self.assertNotIn('prepare_data', self.tool.states.load().artifact_refs)

    def test_resource_evidence_required(self):
        approval = self.approval()

        def changed(request, root, signal):
            outcome = self.cache_transport(request, root, signal)
            outcome['launch']['resources']['cleanup_confirmed'] = False
            return outcome

        with patch('src.agents.wm_cache.run_cache_worker', side_effect=changed):
            self.assertEqual(self.manager.run_next(approval_id=approval.approval_id).status, 'failed_runtime')

    def test_setup_failure_is_visible_without_retry(self):
        """시작 전 실패의 고정 진단을 조회하되 소비한 승인으로 재실행하지 않는다."""
        approval = self.approval()

        def failed(request, root, signal):
            outcome = {'request_id': request.request_id, 'profile_cleanup': True,
                       'preparation_failure': {'stage': 'PROFILE_CREATE', 'code': 'SETUP_ACCESS_DENIED', 'os_error': 5}}
            (root / 'transport.json').write_text(json.dumps(outcome), encoding='utf-8')
            return outcome

        with patch('src.agents.wm_cache.run_cache_worker', side_effect=failed) as worker:
            result = self.manager.run_next(approval_id=approval.approval_id)
            self.assertEqual(result.status, 'failed_runtime')
            status = cache_cli.status(cache_cli.open_run('cache'))
            self.assertEqual(status['preparation_failure']['code'], 'SETUP_ACCESS_DENIED')
            self.assertFalse(status['worker_started'])
            self.manager.run_next(approval_id=approval.approval_id)
            worker.assert_called_once()

    def test_copied_worker_bootstraps(self):
        project, copied = Path(__file__).resolve().parents[2], self.root / 'copied_cache'

        for relative in WORKER_FILES + (ENTRY,):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)

        code = ('import sys; sys.path[:0]=[sys.argv[1],sys.argv[2]]; '
                'from src.agents.wm_cache_worker import execute_request; '
                'from src.agents.wm_cache_results import inspect_cache; '
                'from src.wafer_dl.isolated_cache import build_isolated_cache')
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(copied),
                                 str(Path(sys.prefix) / 'Lib/site-packages')], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

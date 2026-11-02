"""학습률 스케줄의 기본 호환·epoch별 동작·기록·대조 거부를 합성 데이터로 검증한다.

실제 데이터·GPU·Test를 쓰지 않는다. 학습은 CPU의 합성 Dataset으로만 수행한다.
"""

import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import torch

from src.wafer_dl import lr_compare, weight_compare
from src.wafer_dl.cache import file_digest
from src.wafer_dl.compare import (
    RUN_ARTIFACTS,
    _digest_files,
    build_run_contract,
    plan_runs,
    run_all,
)
from src.wafer_dl.train import load_train_config, train
from src.wafer_dl.validate_canonical import ValidationError
from tests.test_wafer_model_compare import (
    SyntheticDataset,
    write_cache_report,
    write_config,
)


CONFIG = Path('configs/wm811k/lr_v1/plateau_alpha0_75.json')
WEIGHT_CONFIG = Path('configs/wm811k/weight_v1/alpha0_75.json')
SEEDS = (42, 43, 44)
ALPHA = {'class_weight_alpha': 0.75, 'scratch_weight_factor': 1.0}
PLATEAU = {
    'type': 'ReduceLROnPlateau',
    'metric': 'macro_f1',
    'mode': 'max',
    'factor': 0.5,
    'patience': 2,
    'threshold': 0.0,
    'threshold_mode': 'abs',
    'cooldown': 0,
    'min_lr': 0.00001,
    'eps': 1e-8,
}


def scheduler_with(**changes):
    return {**PLATEAU, **changes}


class SchedulerConfigTest(unittest.TestCase):
    def load(self, **changes):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.json'
            write_config(path, **changes)
            return load_train_config(path)

    def test_repository_config_adds_only_fixed_schedule(self):
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        weight = json.loads(WEIGHT_CONFIG.read_text(encoding='utf-8'))
        self.assertEqual(config.pop('lr_scheduler'), PLATEAU)
        self.assertEqual(config, weight)
        self.assertEqual(load_train_config(CONFIG)['lr_scheduler'], PLATEAU)

    def test_config_without_key_stays_without_key(self):
        # 키를 채우면 기존 기준과의 조건 대조가 달라진다.
        self.assertNotIn('lr_scheduler', self.load())
        self.assertNotIn('lr_scheduler', load_train_config(WEIGHT_CONFIG))

    def test_invalid_schedules_are_rejected(self):
        missing = dict(PLATEAU)
        del missing['eps']
        for value in (
            'none',
            None,
            missing,
            scheduler_with(extra=1),
            scheduler_with(type='StepLR'),
            scheduler_with(metric='macro_recall'),
            scheduler_with(mode='min'),
            scheduler_with(threshold_mode='rel'),
            scheduler_with(factor=0),
            scheduler_with(factor=1),
            scheduler_with(factor=True),
            scheduler_with(factor='0.5'),
            scheduler_with(factor=math.nan),
            scheduler_with(threshold=-0.1),
            scheduler_with(threshold=math.inf),
            scheduler_with(min_lr=0.01),
            scheduler_with(min_lr=-1),
            scheduler_with(eps=-1),
            scheduler_with(patience=1.5),
            scheduler_with(patience=-1),
            scheduler_with(patience=True),
            scheduler_with(cooldown=-1),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, 'INVALID_|NONFINITE_JSON'):
                    self.load(lr_scheduler=value)


class ScheduledTrainingTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch('src.wafer_dl.train.WaferDataset', SyntheticDataset)
        patcher.start()
        self.addCleanup(patcher.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.count = 0

    def run_scores(self, scores, **changes):
        """Validation 점수를 고정한 CPU 학습을 수행하고 보고서·checkpoint를 돌려준다."""
        self.count += 1
        config = self.root / f'train{self.count}.json'
        output = self.root / f'run{self.count}'
        settings = {'epochs': len(scores), 'patience': 6, **changes}
        write_config(config, **settings)
        metrics = [{'macro_f1': score} for score in scores]
        with mock.patch('src.wafer_dl.train._evaluate', side_effect=metrics):
            report = train(self.root, output, config, 'cpu')
        checkpoint = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
        return report, checkpoint

    def test_fixed_scores_give_expected_rate_for_each_epoch(self):
        report, _ = self.run_scores(
            [0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5], lr_scheduler=PLATEAU
        )
        used = [epoch['learning_rate'] for epoch in report['history']]
        following = [epoch['next_learning_rate'] for epoch in report['history']]

        # 허용 2회를 넘긴 세 번째 불개선 epoch 뒤에 줄고 다음 epoch부터 적용된다.
        self.assertEqual(used, [0.001] * 5 + [0.0005] * 3)
        self.assertEqual(following, [0.001] * 4 + [0.0005] * 3 + [0.00025])
        self.assertEqual(
            [item['epoch'] for item in report['lr_reductions']], [5, 8]
        )

    def test_early_stop_keeps_six_and_is_not_reset_by_reduction(self):
        report, _ = self.run_scores([0.5, 0.6] + [0.5] * 10, lr_scheduler=PLATEAU)
        self.assertEqual(report['selected_epoch'], 2)
        self.assertEqual(len(report['history']), 8)

    def test_improvement_resets_both_counters(self):
        report, _ = self.run_scores(
            [0.5, 0.4, 0.4, 0.4, 0.7, 0.4, 0.4, 0.4], lr_scheduler=PLATEAU
        )
        self.assertEqual([item['epoch'] for item in report['lr_reductions']], [4, 8])
        self.assertEqual(report['selected_epoch'], 5)

    def test_rate_stops_at_minimum(self):
        report, _ = self.run_scores(
            [0.5] * 7,
            patience=20,
            lr_scheduler=scheduler_with(patience=0, min_lr=0.0004),
        )
        self.assertEqual(
            [epoch['next_learning_rate'] for epoch in report['history']],
            [0.001, 0.0005, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004],
        )
        self.assertEqual(len(report['lr_reductions']), 2)

    def test_best_checkpoint_keeps_schedule_state_and_strict_selection(self):
        _, checkpoint = self.run_scores(
            [0.5, 0.6, 0.6, 0.5, 0.5, 0.5], lr_scheduler=PLATEAU
        )
        record = checkpoint['lr_schedule']
        self.assertEqual(checkpoint['selected_epoch'], 2)
        self.assertEqual(record['settings'], PLATEAU)
        self.assertEqual(record['learning_rate'], 0.001)
        self.assertEqual(record['next_learning_rate'], 0.001)
        self.assertEqual(record['reductions'], [])
        self.assertEqual(record['scheduler_state']['best'], 0.6)
        self.assertEqual(record['scheduler_state']['num_bad_epochs'], 0)
        self.assertEqual(checkpoint['training_config']['lr_scheduler'], PLATEAU)

    def test_checkpoint_after_reduction_records_past_reductions(self):
        _, checkpoint = self.run_scores(
            [0.5, 0.4, 0.4, 0.4, 0.7], lr_scheduler=PLATEAU
        )
        record = checkpoint['lr_schedule']
        self.assertEqual(record['learning_rate'], 0.0005)
        self.assertEqual([item['epoch'] for item in record['reductions']], [4])

    def test_fixed_rate_creates_no_scheduler(self):
        name = 'torch.optim.lr_scheduler.ReduceLROnPlateau'
        with mock.patch(name, side_effect=AssertionError('호출 금지')):
            report, checkpoint = self.run_scores([0.5, 0.4, 0.4, 0.4])

        self.assertEqual({epoch['learning_rate'] for epoch in report['history']}, {0.001})
        self.assertEqual(report['lr_reductions'], [])
        self.assertIsNone(report['lr_scheduler'])
        self.assertIsNone(checkpoint['lr_schedule'])
        self.assertNotIn('lr_scheduler', checkpoint['training_config'])

    def test_schedule_without_reduction_changes_no_weights_or_random_state(self):
        scores = [0.1, 0.2, 0.3, 0.4]
        _, fixed = self.run_scores(scores)
        fixed_state = torch.get_rng_state()
        _, scheduled = self.run_scores(scores, lr_scheduler=PLATEAU)

        self.assertTrue(torch.equal(fixed_state, torch.get_rng_state()))
        for name, value in fixed['model_state_dict'].items():
            self.assertTrue(torch.equal(value, scheduled['model_state_dict'][name]), name)

    def test_reduction_changes_later_weights(self):
        scores = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9]
        _, fixed = self.run_scores(scores)
        _, scheduled = self.run_scores(scores, lr_scheduler=PLATEAU)

        self.assertFalse(all(
            torch.equal(value, scheduled['model_state_dict'][name])
            for name, value in fixed['model_state_dict'].items()
        ))


def schedule_run(scores, used, following, selected_epoch, reductions, epochs=30, patience=6):
    """schedule_summary가 읽는 항목만 가진 합성 실행을 만든다."""
    history = [
        {
            'epoch': index,
            'learning_rate': used[index - 1],
            'next_learning_rate': following[index - 1],
            'validation': {'macro_f1': score},
        }
        for index, score in enumerate(scores, start=1)
    ]
    selected_reductions = [item for item in reductions if item['epoch'] <= selected_epoch]
    return {
        'selected_epoch': selected_epoch,
        'contract': {'training_config': {
            'learning_rate': 0.001,
            'selection_metric': 'macro_f1',
            'epochs': epochs,
            'patience': patience,
            'lr_scheduler': PLATEAU,
        }},
        'report': {
            'history': history,
            'lr_scheduler': PLATEAU,
            'lr_reductions': reductions,
        },
        'lr_schedule': {
            'settings': PLATEAU,
            'learning_rate': used[selected_epoch - 1],
            'next_learning_rate': following[selected_epoch - 1],
            'reductions': selected_reductions,
            'scheduler_state': {},
        },
    }


class ScheduleSummaryTest(unittest.TestCase):
    def test_no_reduction(self):
        run = schedule_run([0.1, 0.2, 0.3], [0.001] * 3, [0.001] * 3, 3, [], epochs=3)
        result = lr_compare.schedule_summary(run)
        self.assertEqual(result['state'], 'no_reduction')
        self.assertEqual(result['epochs_after_first_reduction'], 0)
        self.assertFalse(result['selected_after_first_reduction'])

    def test_reduction_on_last_epoch_has_no_following_training(self):
        reduction = {'epoch': 3, 'learning_rate': 0.001, 'next_learning_rate': 0.0005}
        run = schedule_run(
            [0.5, 0.4, 0.4], [0.001] * 3, [0.001, 0.001, 0.0005], 1, [reduction],
            epochs=3, patience=2,
        )
        # 위 점수는 세 번째 불개선 전이므로 실제 스케줄과 달라 기록 불일치로 거부된다.
        with self.assertRaisesRegex(ValidationError, 'LR_COMPARE_RECORD_MISMATCH'):
            lr_compare.schedule_summary(run)

        scores = [0.5, 0.4, 0.4, 0.4]
        run = schedule_run(
            scores, [0.001] * 4, [0.001, 0.001, 0.001, 0.0005], 1,
            [{**reduction, 'epoch': 4}], epochs=4, patience=3,
        )
        result = lr_compare.schedule_summary(run)
        self.assertEqual(result['state'], 'reduced_without_training')
        self.assertEqual(result['epochs_after_first_reduction'], 0)

    def test_reduction_with_following_training(self):
        scores = [0.5, 0.4, 0.4, 0.4, 0.7]
        reduction = {'epoch': 4, 'learning_rate': 0.001, 'next_learning_rate': 0.0005}
        run = schedule_run(
            scores, [0.001] * 4 + [0.0005], [0.001] * 3 + [0.0005] * 2, 5, [reduction], epochs=5,
        )
        result = lr_compare.schedule_summary(run)
        self.assertEqual(result['state'], 'reduced_with_training')
        self.assertEqual(result['epochs_after_first_reduction'], 1)
        self.assertTrue(result['selected_after_first_reduction'])
        self.assertEqual(result['last_trained_learning_rate'], 0.0005)

    def test_changed_early_stop_budget_is_rejected(self):
        run = schedule_run(
            [0.5, 0.4, 0.4], [0.001] * 3, [0.001] * 3, 1, [], epochs=30, patience=6,
        )
        with self.assertRaisesRegex(ValidationError, 'LR_COMPARE_RECORD_MISMATCH'):
            lr_compare.schedule_summary(run)


class ScheduleCompareTest(unittest.TestCase):
    """합성 학습으로 만든 기준·후보를 대조한다. 실행은 한 번만 만들고 복사해 쓴다."""

    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.source = Path(temporary.name)
        patchers = [
            mock.patch('src.wafer_dl.train.WaferDataset', SyntheticDataset),
            mock.patch('src.wafer_dl.compare.WaferDataset', SyntheticDataset),
        ]
        for patcher in patchers:
            patcher.start()
            cls.addClassCleanup(patcher.stop)

        write_cache_report(cls.source / 'cache')
        common = {**ALPHA, 'epochs': 3, 'patience': 2}
        cls.build('baseline', common)
        cls.build('candidate', {**common, 'lr_scheduler': scheduler_with(patience=0)})
        cls.build('other_batch', {
            **common, 'batch_size': 2, 'lr_scheduler': scheduler_with(patience=0),
        })
        cls.build('other_weight', {
            **common, 'scratch_weight_factor': 0.9, 'lr_scheduler': scheduler_with(patience=0),
        })

    @classmethod
    def build(cls, name, settings):
        config = cls.source / f'{name}.json'
        write_config(config, **settings)
        failures = run_all(
            cls.source / 'cache',
            cls.source / name,
            config,
            'cpu',
            models=('SmallWaferCNN',),
            augmentations=('d4',),
            seeds=SEEDS,
        )
        assert failures == []

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(self.source / 'cache', self.root / 'cache')
        for name in ('baseline', 'candidate'):
            shutil.copytree(self.source / name, self.root / name)
        self.report = self.root / 'report'

    def compare(self, candidate=None, baseline=None, **options):
        return lr_compare.compare_schedule(
            'SmallWaferCNN',
            candidate or self.root / 'candidate',
            self.report,
            baseline or self.root / 'baseline',
            self.root / 'cache',
            **options,
        )

    def assert_rejected(self, code, **options):
        with self.assertRaisesRegex(ValidationError, code):
            self.compare(**options)
        self.assertFalse(self.report.exists())

    def reseal(self, name, seed, change_report=None, change_success=None):
        """합성 실행을 고친 뒤 파일 해시만 다시 맞춘다."""
        run_dir = self.root / name / f'SmallWaferCNN_d4_seed{seed}'
        report_path = run_dir / 'training_report.json'
        if change_report:
            report = json.loads(report_path.read_text(encoding='utf-8'))
            change_report(report)
            report_path.write_text(json.dumps(report), encoding='utf-8')
        success_path = run_dir / 'SUCCESS.json'
        success = json.loads(success_path.read_text(encoding='utf-8'))
        if change_success:
            change_success(success)
        success['artifacts'] = {
            item: file_digest(run_dir / item) for item in RUN_ARTIFACTS
        }
        success_path.write_text(json.dumps(success), encoding='utf-8')

    def test_comparison_reports_all_required_values(self):
        before = {
            path: file_digest(path)
            for path in (self.root / 'baseline').rglob('*') if path.is_file()
        }
        result = self.compare()

        self.assertEqual(result['selection_status'], '선정 보류')
        self.assertIsNone(result['selected_model'])
        self.assertFalse(result['test_split_used'])
        self.assertEqual(result['lr_scheduler'], scheduler_with(patience=0))
        self.assertEqual(set(result['schedules']), set(SEEDS))
        self.assertTrue(result['code']['same_code'])
        self.assertEqual(before, {
            path: file_digest(path)
            for path in (self.root / 'baseline').rglob('*') if path.is_file()
        })

        text = (self.report / 'comparison.md').read_text(encoding='utf-8')
        for expected in (
            '선정 보류', 'seed42 기준 → 후보', 'seed별 학습률 동작', 'lr_reductions',
            'training_minutes', 'Test는 사용하지 않았습니다',
        ):
            self.assertIn(expected, text)
        self.assertNotIn('합격했', text)

        summary = (self.report / 'summary.csv').read_text(encoding='utf-8-sig')
        for name in lr_compare.CLASS_NAMES:
            for metric in ('precision', 'recall', 'f1', 'fp', 'fn'):
                self.assertIn(f'{name},{metric}', summary)

        success = json.loads((self.report / 'SUCCESS.json').read_text(encoding='utf-8'))
        self.assertEqual(success['artifacts'].keys(), {
            'comparison.json', 'comparison.md', 'seed_differences.csv', 'summary.csv',
        })

    def test_code_difference_is_shown(self):
        files = {'train.py': 'b' * 64}
        for seed in SEEDS:
            self.reseal('baseline', seed, change_success=lambda success: success[
                'contract'].update(code_fingerprint={
                    'sha256': _digest_files(files), 'files': files,
                }))
        result = self.compare()
        self.assertFalse(result['code']['same_code'])
        self.assertIn('train.py', result['code']['differing_code_files'])
        text = (self.report / 'comparison.md').read_text(encoding='utf-8')
        self.assertIn('실제 신구 학습의 동등성은 측정하지 않았습니다', text)

    def test_mixed_code_is_rejected(self):
        files = {'train.py': 'b' * 64}
        self.reseal('candidate', 43, change_success=lambda success: success[
            'contract'].update(code_fingerprint={
                'sha256': _digest_files(files), 'files': files,
            }))
        self.assert_rejected('LR_COMPARE_CODE_MIXED')

    def test_baseline_with_schedule_is_rejected(self):
        self.assert_rejected(
            'LR_COMPARE_BASELINE_HAS_SCHEDULER', baseline=self.root / 'candidate'
        )

    def test_candidate_without_schedule_is_rejected(self):
        self.assert_rejected(
            'LR_COMPARE_CANDIDATE_HAS_NO_SCHEDULER', candidate=self.root / 'baseline'
        )

    def test_other_condition_is_rejected(self):
        self.assert_rejected(
            'LR_COMPARE_CONDITION_MISMATCH', candidate=self.source / 'other_batch'
        )

    def test_other_weight_is_rejected(self):
        self.assert_rejected(
            'LR_COMPARE_WEIGHT_MISMATCH', candidate=self.source / 'other_weight'
        )

    def test_missing_seed_is_rejected(self):
        shutil.rmtree(self.root / 'candidate' / 'SmallWaferCNN_d4_seed44')
        self.assert_rejected('COMPARE_RUN_INCOMPLETE')

    def test_incomplete_seed_list_is_rejected(self):
        self.assert_rejected('LR_COMPARE_SEEDS_INCOMPLETE', seeds=(42, 43))

    def test_other_model_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_INCOMPLETE'):
            lr_compare.compare_schedule(
                'ResNet18', self.root / 'candidate', self.report,
                self.root / 'baseline', self.root / 'cache',
            )
        with self.assertRaisesRegex(ValidationError, 'LR_COMPARE_MODEL_UNSUPPORTED'):
            lr_compare.compare_schedule(
                'MobileNetV3Small', self.root / 'candidate', self.report,
                self.root / 'baseline', self.root / 'cache',
            )

    def test_other_cache_is_rejected(self):
        (self.root / 'cache' / 'cache.json').write_text(json.dumps({
            'status': 'cache_complete', 'changed': True,
        }), encoding='utf-8')
        self.assert_rejected('LR_COMPARE_CACHE_MISMATCH')

    def test_changed_artifact_is_rejected(self):
        run_dir = self.root / 'candidate' / 'SmallWaferCNN_d4_seed42'
        with (run_dir / 'training_report.json').open('a', encoding='utf-8') as stream:
            stream.write(' ')
        self.assert_rejected('COMPARE_RUN_INTEGRITY_FAILED')

    def test_changed_rate_record_is_rejected(self):
        def change(report):
            report['history'][-1]['learning_rate'] = 0.5
        self.reseal('candidate', 42, change_report=change)
        self.assert_rejected('LR_COMPARE_RECORD_MISMATCH')

    def test_existing_report_and_partial_folder_are_preserved(self):
        self.report.mkdir()
        (self.report / 'keep.txt').write_text('보존', encoding='utf-8')
        with self.assertRaisesRegex(ValidationError, 'LR_COMPARE_REPORT_ALREADY_EXISTS'):
            self.compare()
        self.assertTrue((self.report / 'keep.txt').exists())

        shutil.rmtree(self.report)
        partial = self.report.with_name(self.report.name + '.tmp')
        partial.mkdir()
        self.assert_rejected('LR_COMPARE_PARTIAL_EXISTS')
        self.assertTrue(partial.exists())

    def test_weight_compare_rejects_scheduled_candidate(self):
        baseline = weight_compare.load_condition(
            self.root / 'baseline', SEEDS, 'SmallWaferCNN'
        )
        candidate = weight_compare.load_condition(
            self.root / 'candidate', SEEDS, 'SmallWaferCNN'
        )
        with self.assertRaisesRegex(ValidationError, 'WEIGHT_COMPARE_CONDITION_MISMATCH'):
            weight_compare.require_comparable(baseline, candidate)

    def test_changed_schedule_config_is_not_resumed(self):
        config = self.root / 'changed.json'
        write_config(
            config, **ALPHA, epochs=3, patience=2,
            lr_scheduler=scheduler_with(factor=0.25),
        )
        with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CONTRACT_MISMATCH'):
            plan_runs(
                self.root / 'cache', self.root / 'candidate', config, 'cpu',
                models=('SmallWaferCNN',), augmentations=('d4',), seeds=SEEDS,
            )

    def test_fixed_rate_config_is_not_mixed_into_scheduled_folder(self):
        config = self.root / 'fixed.json'
        write_config(config, **ALPHA, epochs=3, patience=2)
        with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CONTRACT_MISMATCH'):
            plan_runs(
                self.root / 'cache', self.root / 'candidate', config, 'cpu',
                models=('SmallWaferCNN',), augmentations=('d4',), seeds=SEEDS,
            )

    def test_completed_past_code_run_is_not_reused(self):
        files = {'train.py': 'b' * 64}
        self.reseal('candidate', 42, change_success=lambda success: success[
            'contract'].update(code_fingerprint={
                'sha256': _digest_files(files), 'files': files,
            }))
        config = self.source / 'candidate.json'
        with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CODE_MISMATCH'):
            plan_runs(
                self.root / 'cache', self.root / 'candidate', config, 'cpu',
                models=('SmallWaferCNN',), augmentations=('d4',), seeds=SEEDS,
            )

    def test_interrupted_run_is_preserved_and_retrained(self):
        run = 'SmallWaferCNN_d4_seed42'
        shutil.rmtree(self.root / 'candidate' / run)
        temporary = self.root / 'candidate' / (run + '.tmp')
        temporary.mkdir()
        (temporary / 'partial.txt').write_text('부분 결과', encoding='utf-8')

        failures = run_all(
            self.root / 'cache', self.root / 'candidate', self.source / 'candidate.json',
            'cpu', models=('SmallWaferCNN',), augmentations=('d4',), seeds=SEEDS,
        )
        self.assertEqual(failures, [])
        self.assertTrue((self.root / 'candidate' / run / 'SUCCESS.json').exists())
        self.assertTrue(
            (self.root / 'candidate' / (run + '.incomplete_1') / 'partial.txt').exists()
        )

    def test_contract_keeps_schedule_only_when_configured(self):
        fixed = build_run_contract(
            self.root / 'cache', self.source / 'baseline.json', 'cpu', None,
            'SmallWaferCNN', 'd4', 42,
        )
        scheduled = build_run_contract(
            self.root / 'cache', self.source / 'candidate.json', 'cpu', None,
            'SmallWaferCNN', 'd4', 42,
        )
        self.assertNotIn('lr_scheduler', fixed['training_config'])
        self.assertEqual(scheduled['training_config']['lr_scheduler'], scheduler_with(patience=0))


class CommandLineTest(unittest.TestCase):
    def test_failure_prints_code_and_next_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                'lr_compare', '--model', 'SmallWaferCNN',
                '--candidate-root', str(root / 'candidate'),
                '--report-dir', str(root / 'report'),
                '--baseline-root', str(root / 'baseline'),
            ]
            with mock.patch('sys.argv', argv), mock.patch('builtins.print') as output:
                self.assertEqual(lr_compare.main(), 1)
        lines = ' '.join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn('COMPARE_RUN_INCOMPLETE', lines)
        self.assertIn('compare run', lines)

    def test_default_baselines_are_separate_by_model(self):
        self.assertEqual(lr_compare.DEFAULT_BASELINES, {
            'SmallWaferCNN': 'models/wm811k/weight_cnn_v1/alpha0_75',
            'ResNet18': 'models/wm811k/weight_v1/alpha0_75',
        })


if __name__ == '__main__':
    unittest.main()

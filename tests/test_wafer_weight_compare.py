"""가중치 설정과 D4 모델별 기준 대조의 합성 데이터 테스트.

실제 학습·추론·Test 없이 기존 계산 호환, 후보 가중치, 대조 입력 검증, 판정표,
출력 충돌을 확인한다.
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import torch

from src.wafer_dl import compare, weight_compare as analysis
from src.wafer_dl.cache import file_digest
from src.wafer_dl.compare import (
    RUN_ARTIFACTS,
    RUN_CODE_FILES,
    RUN_COMPLETE_STATUS,
    RUN_CONTRACT_VERSION,
    _digest_files,
)
from src.wafer_dl.train import (
    CLASS_WEIGHT_METHOD,
    _save_checkpoint,
    class_weight_record,
    class_weights,
    load_train_config,
)
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


COUNTS = [101768, 3144, 439, 3653, 7685, 2811, 113, 688, 960]
SCRATCH = CLASS_NAMES.index('Scratch')
NONE = CLASS_NAMES.index('None')
LOC = CLASS_NAMES.index('Loc')
NEAR_FULL = CLASS_NAMES.index('Near-Full')
DONUT = CLASS_NAMES.index('Donut')
RANDOM = CLASS_NAMES.index('Random')
SEEDS = (42, 43, 44)
CONFIG_DIR = Path('configs/wm811k/weight_v1')


def previous_class_weights(class_counts, max_weight):
    """변경 전 class_weights 계산을 그대로 옮긴 독립 대조본."""
    counts = torch.tensor(class_counts, dtype=torch.float64)
    raw_weights = counts.sum() / (len(class_counts) * counts)
    low, high = 0.0, float(max_weight) / float(raw_weights.min())
    for _ in range(64):
        scale = (low + high) / 2
        if float(raw_weights.mul(scale).clamp(max=float(max_weight)).mean()) < 1:
            low = scale
        else:
            high = scale
    return raw_weights.mul(high).clamp(max=float(max_weight)).to(dtype=torch.float32)


def base_config(**changes):
    return {
        'training_version': 'wm_cnn_train_v1',
        'batch_size': 64,
        'epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'patience': 6,
        'seed': 42,
        'class_weighted': True,
        'max_class_weight': 10.0,
        'selection_metric': 'macro_f1',
        **changes,
    }


class ClassWeightTest(unittest.TestCase):
    def test_default_matches_previous_calculation_exactly(self):
        for counts in (COUNTS, [100, 10, 10, 10, 10, 10, 10, 10, 1], [7] * 9):
            for cap in (1, 1.5, 3, 5, 10, 100):
                with self.subTest(counts=counts, cap=cap):
                    self.assertTrue(torch.equal(
                        class_weights(counts, cap),
                        previous_class_weights(counts, cap),
                    ))
                    self.assertTrue(torch.equal(
                        class_weights(counts, cap, 1.0, 1.0),
                        previous_class_weights(counts, cap),
                    ))

    def test_independent_calculation_of_alpha_and_factor(self):
        """상한이 걸리지 않는 조건에서 손으로 쓴 식과 일치한다."""
        counts = torch.tensor(COUNTS, dtype=torch.float64)
        raw = (counts.sum() / (9 * counts)).pow(0.75)
        expected = raw / raw.mean()
        self.assertTrue(torch.allclose(
            class_weights(COUNTS, 10, 0.75).double(), expected, atol=1e-5
        ))

        raw = counts.sum() / (9 * counts)
        raw[SCRATCH] *= 0.7
        expected = raw / raw.mean()
        self.assertTrue(torch.allclose(
            class_weights(COUNTS, 10, 1.0, 0.7).double(), expected, atol=1e-5
        ))

    def test_scratch_factor_changes_only_scratch_ratio(self):
        before = class_weights(COUNTS, 10).double()
        after = class_weights(COUNTS, 10, 1.0, 0.7).double()
        ratio = after / before
        others = [index for index in range(9) if index != SCRATCH]

        self.assertAlmostEqual(float(ratio[SCRATCH] / ratio[NONE]), 0.7, places=5)
        self.assertLess(float(ratio[others].max() - ratio[others].min()), 1e-5)

    def test_alpha_lowers_every_rare_class_against_none(self):
        before = class_weights(COUNTS, 10).double()
        after = class_weights(COUNTS, 10, 0.75).double()
        for index in range(1, 9):
            self.assertLess(
                float(after[index] / after[NONE]), float(before[index] / before[NONE])
            )

    def test_cap_1_5_binds_scratch_and_changes_ratios(self):
        weights = class_weights(COUNTS, 1.5).double()
        self.assertAlmostEqual(float(weights[SCRATCH]), 1.5, places=5)
        self.assertAlmostEqual(float(weights[NEAR_FULL]), 1.5, places=5)
        self.assertLess(float(weights[SCRATCH] / weights[NONE]), 60)

    def test_cap_10_does_not_bind_on_baseline(self):
        self.assertLess(float(class_weights(COUNTS, 10).max()), 10)

    def test_invalid_values_are_rejected(self):
        for value in (0, -0.5, 1.5, float('nan'), float('inf'), True, '1'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, 'INVALID_CLASS_WEIGHT_ALPHA'):
                    class_weights(COUNTS, 10, value)
                with self.assertRaisesRegex(ValidationError, 'INVALID_SCRATCH_WEIGHT_FACTOR'):
                    class_weights(COUNTS, 10, 1.0, value)


class TrainConfigTest(unittest.TestCase):
    def load(self, config):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'train.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            return load_train_config(path)

    def test_previous_config_reads_as_default(self):
        config = load_train_config('configs/wm811k/train.json')
        self.assertEqual(config['class_weight_alpha'], 1.0)
        self.assertEqual(config['scratch_weight_factor'], 1.0)
        self.assertEqual(config['max_class_weight'], 10.0)

    def test_candidate_configs_change_only_weight_settings(self):
        base = load_train_config('configs/wm811k/train.json')
        expected = {
            'cap1_5': {'max_class_weight': 1.5},
            'alpha0_75': {'class_weight_alpha': 0.75},
            'scratch0_7': {'scratch_weight_factor': 0.7},
        }
        for name, changes in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    load_train_config(CONFIG_DIR / f'{name}.json'),
                    {**base, **changes},
                )

    def test_invalid_config_is_rejected(self):
        for changes in (
            {'class_weight_alpha': 0},
            {'class_weight_alpha': 1.1},
            {'class_weight_alpha': True},
            {'scratch_weight_factor': -1},
            {'scratch_weight_factor': False},
            {'unknown_weight': 1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValidationError, 'INVALID_TRAIN_CONFIG'):
                    self.load(base_config(**changes))

    def test_nan_and_infinity_are_rejected(self):
        for text in ('NaN', 'Infinity'):
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / 'train.json'
                    payload = json.dumps(base_config(class_weight_alpha=0.5))
                    path.write_text(payload.replace('0.5', text), encoding='utf-8')
                    with self.assertRaises(ValidationError):
                        load_train_config(path)


class WeightRecordTest(unittest.TestCase):
    def config(self, **changes):
        return {
            'class_weighted': True,
            'max_class_weight': 10.0,
            'class_weight_alpha': 1.0,
            'scratch_weight_factor': 0.7,
            **changes,
        }

    def test_record_names_all_classes(self):
        record = class_weight_record(class_weights(COUNTS, 10, 1.0, 0.7), self.config())
        self.assertEqual(record['method'], CLASS_WEIGHT_METHOD)
        self.assertEqual(list(record['weights']), list(CLASS_NAMES))

    def test_unweighted_run_records_no_weights(self):
        record = class_weight_record(
            class_weights(COUNTS, 10), self.config(class_weighted=False)
        )
        self.assertIsNone(record['weights'])

    def test_checkpoint_stores_record_loadable_with_weights_only(self):
        class Dataset:
            transform_config = {}
            protocol = 'p'
            source_split_manifest_sha256 = 'a' * 64

        record = class_weight_record(class_weights(COUNTS, 10), self.config())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'best_model.pt'
            _save_checkpoint(
                path,
                torch.nn.Linear(1, 1),
                {'selection_metric': 'macro_f1'},
                Dataset,
                {},
                1,
                'ResNet18',
                record,
            )
            checkpoint = torch.load(path, weights_only=True)
        self.assertEqual(checkpoint['class_weights'], record)


def matrix_with(changes=()):
    """대각 위주의 합성 혼동행렬에 (실제, 예측, 건수) 변경을 적용한다."""
    matrix = [[0] * 9 for _ in range(9)]
    for index in range(9):
        matrix[index][index] = 100
    for actual, predicted, count in changes:
        matrix[actual][actual] -= count
        matrix[actual][predicted] += count
    return matrix


class ComparisonTestCase(unittest.TestCase):
    model_name = 'ResNet18'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        (self.cache / 'cache.json').write_text(json.dumps({
            'status': 'cache_complete',
            'train_class_counts': COUNTS,
        }), encoding='utf-8')

        self.criteria = self.root / 'criteria.json'
        shutil.copyfile(CONFIG_DIR / 'decision_reference.json', self.criteria)

        self.old_code = self.fingerprint('train.py', 'b' * 64)
        self.new_code = self.fingerprint('train.py', 'c' * 64)
        self.baseline = self.root / 'baseline'
        self.candidate = self.root / 'scratch0_7'
        self.report = self.root / 'report'

        for seed in SEEDS:
            self.write_run(self.baseline, seed, matrix_with(), self.old_code, legacy=True)
            self.write_run(self.candidate, seed, matrix_with(), self.new_code)

    @staticmethod
    def fingerprint(name, digest):
        files = {name: digest}
        return {'sha256': _digest_files(files), 'files': files}

    def write_run(self, root, seed, matrix, code, legacy=False, config=None, recorded=None):
        """저장 혼동행렬을 가진 합성 실행을 쓴다. legacy는 이전 코드 형식이다."""
        run_dir = root / f'{self.model_name}_d4_seed{seed}'
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        settings = {
            'batch_size': 64,
            'epochs': 30,
            'class_weighted': True,
            'max_class_weight': 10.0,
            'seed': seed,
            'augmentation': 'd4',
        }
        if not legacy:
            settings.update({'class_weight_alpha': 1.0, 'scratch_weight_factor': 0.7})
        settings.update(config or {})

        checkpoint = {
            'model_name': self.model_name,
            'augmentation': 'd4',
            'training_config': settings,
            'split_protocol_id': 'split',
            'source_split_manifest_sha256': 'a' * 64,
            'transform_config': {'transform_version': 'wm_resize_v1'},
            'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
            'selected_epoch': 2,
            'validation_metrics': {'confusion_matrix': matrix},
        }
        torch.save(checkpoint, run_dir / 'best_model.pt')

        report = {
            'status': 'training_complete',
            'test_split_used': False,
            'device': 'cuda',
            'model_name': self.model_name,
            'seed': seed,
            'augmentation': 'd4',
            'split_protocol_id': 'split',
            'selected_epoch': 2,
            'history': [{'epoch': 2, 'validation': {'confusion_matrix': matrix}}],
        }
        if not legacy:
            weights = class_weights(
                COUNTS,
                settings['max_class_weight'],
                settings.get('class_weight_alpha', 1.0),
                settings.get('scratch_weight_factor', 1.0),
            )
            report['class_weights'] = recorded or {
                'weights': dict(zip(CLASS_NAMES, weights.tolist()))
            }
        (run_dir / 'training_report.json').write_text(json.dumps(report), encoding='utf-8')

        (run_dir / 'inference.json').write_text(json.dumps({
            'device': 'cuda',
            'ms_per_sample': 1.0,
        }), encoding='utf-8')

        (run_dir / 'SUCCESS.json').write_text(json.dumps({
            'status': RUN_COMPLETE_STATUS,
            'contract': {
                'contract_version': RUN_CONTRACT_VERSION,
                'model_name': self.model_name,
                'seed': seed,
                'device': 'cuda',
                'training_config': settings,
                'split_protocol_id': 'split',
                'source_split_manifest_sha256': 'a' * 64,
                'cache_sha256': file_digest(self.cache / 'cache.json'),
                'code_fingerprint': code,
            },
            'artifacts': {name: file_digest(run_dir / name) for name in RUN_ARTIFACTS},
        }), encoding='utf-8')

    def compare(self, **options):
        return analysis.compare_candidate(
            self.candidate,
            self.report,
            self.baseline,
            self.cache,
            self.criteria,
            **options,
        )

    def assert_rejected(self, code):
        with self.assertRaisesRegex(ValidationError, code):
            self.compare()
        self.assertFalse(self.report.exists())


class InputValidationTest(ComparisonTestCase):
    def test_old_code_baseline_is_read_without_building_model(self):
        with mock.patch.object(compare, 'build_model') as build:
            result = self.compare()

        self.assertFalse(build.called)
        self.assertFalse(result['code']['same_code'])
        self.assertEqual(result['code']['differing_code_files'], ['train.py'])

        report = (self.report / 'comparison.md').read_text(encoding='utf-8')
        self.assertIn('같은 코드로 학습한 비교가 아닙니다', report)

    def test_missing_baseline_seed_is_rejected(self):
        shutil.rmtree(self.baseline / 'ResNet18_d4_seed43')
        self.assert_rejected('COMPARE_RUN_INCOMPLETE')

    def test_seed_set_must_be_complete(self):
        with self.assertRaisesRegex(ValidationError, 'WEIGHT_COMPARE_SEEDS_INCOMPLETE'):
            self.compare(seeds=(42, 43))
        with self.assertRaisesRegex(ValidationError, 'WEIGHT_COMPARE_SEEDS_INCOMPLETE'):
            self.compare(seeds=(42, 42, 43))

    def test_other_training_condition_is_rejected(self):
        self.write_run(self.candidate, 43, matrix_with(), self.new_code, config={'epochs': 5})
        self.assert_rejected('WEIGHT_COMPARE_CONDITION_MISMATCH')

    def test_mixed_weight_settings_in_candidate_are_rejected(self):
        self.write_run(
            self.candidate, 43, matrix_with(), self.new_code,
            config={'scratch_weight_factor': 0.5},
        )
        self.assert_rejected('WEIGHT_COMPARE_CONDITION_MISMATCH')

    def test_mixed_code_in_candidate_is_rejected(self):
        self.write_run(
            self.candidate, 44, matrix_with(), self.fingerprint('train.py', 'd' * 64)
        )
        self.assert_rejected('WEIGHT_COMPARE_CODE_MIXED')

    def test_changed_artifact_is_rejected(self):
        (self.candidate / 'ResNet18_d4_seed42' / 'training_report.json').write_text(
            '{}', encoding='utf-8'
        )
        self.assert_rejected('COMPARE_RUN_INTEGRITY_FAILED')

    def test_other_cache_is_rejected(self):
        (self.cache / 'cache.json').write_text(json.dumps({
            'status': 'cache_complete',
            'train_class_counts': [1] * 9,
        }), encoding='utf-8')
        self.assert_rejected('WEIGHT_COMPARE_CACHE_MISMATCH')

    def test_same_weight_settings_are_rejected(self):
        for seed in SEEDS:
            self.write_run(
                self.candidate, seed, matrix_with(), self.new_code,
                config={'scratch_weight_factor': 1.0},
            )
        self.assert_rejected('WEIGHT_COMPARE_NO_WEIGHT_DIFFERENCE')

    def test_wrong_recorded_weights_are_rejected(self):
        self.write_run(
            self.candidate, 42, matrix_with(), self.new_code,
            recorded={'weights': {'None': 1.0}},
        )
        self.assert_rejected('WEIGHT_COMPARE_RECORD_MISMATCH')

    def test_changed_criteria_structure_is_rejected(self):
        criteria = json.loads(self.criteria.read_text(encoding='utf-8'))
        del criteria['donut_fn_increase_sum_max']
        self.criteria.write_text(json.dumps(criteria), encoding='utf-8')
        self.assert_rejected('INVALID_WEIGHT_CRITERIA')


class MetricTest(unittest.TestCase):
    def test_fp_fn_and_precision(self):
        matrix = matrix_with(((NONE, SCRATCH, 30), (LOC, SCRATCH, 20), (SCRATCH, RANDOM, 10)))
        metrics = analysis.matrix_metrics(matrix)

        self.assertEqual(metrics['Scratch']['fp'], 50)
        self.assertEqual(metrics['Scratch']['fn'], 10)
        self.assertAlmostEqual(metrics['Scratch']['precision'], 90 / 140)
        self.assertAlmostEqual(metrics['Scratch']['recall'], 0.9)
        self.assertEqual(metrics['Random']['fp'], 10)

    def test_empty_prediction_class_has_zero_precision(self):
        matrix = matrix_with()
        matrix[SCRATCH][SCRATCH] = 0
        matrix[SCRATCH][NONE] = 100
        self.assertEqual(analysis.matrix_metrics(matrix)['Scratch']['precision'], 0.0)


def rows_for(**values):
    """지표 이름별 seed 차이로 판정 입력 행을 만든다."""
    defaults = {
        ('overall', 'macro_f1'): (0, 0, 0),
        ('Scratch', 'precision'): (0.05, 0.05, 0.05),
        ('Scratch', 'recall'): (0, 0, 0),
        ('Random', 'recall'): (0, 0, 0),
        ('Near-Full', 'fn'): (0, 0, 0),
        ('Donut', 'fn'): (0, 0, 0),
    }
    defaults.update(values.get('changes', {}))
    return [
        {
            'seed': seed,
            'scope': scope,
            'metric': metric,
            'baseline': 1.0,
            'candidate': 1.0 + difference,
            'difference': difference,
        }
        for (scope, metric), differences in defaults.items()
        for seed, difference in zip(SEEDS, differences)
    ]


class CriteriaTest(unittest.TestCase):
    criteria = {
        'macro_f1_mean_difference_min_pp': -0.5,
        'scratch_precision_mean_difference_min_pp': 5.0,
        'scratch_precision_improved_seeds_min': 2,
        'scratch_recall_mean_difference_min_pp': -3.0,
        'random_recall_mean_difference_min_pp': -3.0,
        'near_full_fn_increase_sum_max': 1,
        'donut_fn_increase_sum_max': 3,
    }

    def results(self, **changes):
        rows = rows_for(changes=changes.get('changes', {}))
        return analysis.evaluate_criteria(
            analysis.summarize_rows(rows), rows, self.criteria
        )

    def failed(self, results):
        return [item['item'] for item in results if not item['passed']]

    def test_exact_boundaries_pass(self):
        results = self.results(changes={
            ('overall', 'macro_f1'): (-0.005, -0.005, -0.005),
            ('Scratch', 'recall'): (-0.03, -0.03, -0.03),
            ('Random', 'recall'): (-0.03, -0.03, -0.03),
            ('Near-Full', 'fn'): (1, 0, 0),
            ('Donut', 'fn'): (1, 1, 1),
        })
        self.assertEqual(self.failed(results), [])

    def test_values_just_past_boundary_fail(self):
        results = self.results(changes={
            ('overall', 'macro_f1'): (-0.00501, -0.00501, -0.00501),
            ('Scratch', 'precision'): (0.0499, 0.0499, 0.0499),
            ('Near-Full', 'fn'): (2, 0, 0),
            ('Donut', 'fn'): (2, 1, 1),
        })
        self.assertEqual(self.failed(results), [
            '평균 Macro-F1 차이(%p)',
            '평균 Scratch Precision 차이(%p)',
            'Near-Full FN 증가 합계',
            'Donut FN 증가 합계',
        ])

    def test_improved_seed_count_is_checked_separately(self):
        results = self.results(changes={
            ('Scratch', 'precision'): (0.2, -0.01, -0.01),
        })
        self.assertEqual(self.failed(results), ['Scratch Precision 개선 seed 수'])

    def test_percentage_points_are_differences_not_ratios(self):
        results = self.results(changes={
            ('Scratch', 'precision'): (0.06, 0.06, 0.06),
        })
        self.assertAlmostEqual(results[1]['value'], 6.0)

    def test_worse_seeds_are_shown(self):
        rows = rows_for(changes={('Donut', 'fn'): (2, -2, 0)})
        items = {
            (item['scope'], item['metric']): item
            for item in analysis.summarize_rows(rows)
        }
        self.assertEqual(items[('Donut', 'fn')]['worse_seeds'], '42')
        self.assertEqual(items[('Donut', 'fn')]['difference_sum'], 0)


class OutputTest(ComparisonTestCase):
    def test_paired_table_uses_same_seed_differences(self):
        self.write_run(
            self.candidate, 42, matrix_with(((NONE, RANDOM, 10),)), self.new_code
        )
        self.write_run(
            self.baseline, 42, matrix_with(((NONE, SCRATCH, 30),)), self.old_code, legacy=True
        )

        result = self.compare()

        rows = (self.report / 'seed_differences.csv').read_text(encoding='utf-8-sig')
        self.assertIn('42,Scratch,fp,30,0,-30', rows)
        self.assertIn('43,Scratch,fp,0,0,0', rows)
        self.assertIsNone(result['selected_model'])

    def test_report_states_scope_and_limits(self):
        result = self.compare()
        report = (self.report / 'comparison.md').read_text(encoding='utf-8')

        for text in (
            '선정 보류',
            '추가 튜닝',
            '통계적 우월성',
            '독립 표본 수가 아닙니다',
            '사용자가 수치를 승인한 기준이나',
            'Near-Full',
            'Donut',
        ):
            self.assertIn(text, report)

        self.assertFalse(result['test_split_used'])
        self.assertEqual(result['weight_settings']['candidate']['scratch_weight_factor'], 0.7)
        self.assertEqual(result['weight_settings']['baseline']['scratch_weight_factor'], 1.0)

    def test_all_nine_classes_are_reported(self):
        self.compare()
        rows = (self.report / 'summary.csv').read_text(encoding='utf-8-sig')
        for name in CLASS_NAMES:
            self.assertIn(f'{name},fn', rows)

    def test_inputs_stay_unchanged(self):
        paths = [
            *(self.baseline / 'ResNet18_d4_seed42').iterdir(),
            self.cache / 'cache.json',
            self.criteria,
        ]
        before = {path: file_digest(path) for path in paths}
        self.compare()
        self.assertEqual({path: file_digest(path) for path in paths}, before)

    def test_existing_report_is_not_overwritten(self):
        self.report.mkdir()
        marker = self.report / 'keep.txt'
        marker.write_text('기존', encoding='utf-8')

        with self.assertRaisesRegex(ValidationError, 'WEIGHT_COMPARE_REPORT_ALREADY_EXISTS'):
            self.compare()

        self.assertEqual(marker.read_text(encoding='utf-8'), '기존')

    def test_partial_report_is_not_accepted(self):
        partial = self.report.with_name('report.tmp')
        partial.mkdir()

        self.assert_rejected('WEIGHT_COMPARE_PARTIAL_EXISTS')
        self.assertTrue(partial.exists())

    def test_failure_keeps_partial_without_success(self):
        with mock.patch.object(analysis, 'write_report', side_effect=OSError('디스크 오류')):
            with self.assertRaises(OSError):
                self.compare()

        partial = self.report.with_name('report.incomplete_1')
        self.assertFalse(self.report.exists())
        self.assertTrue((partial / 'summary.csv').is_file())
        self.assertFalse((partial / 'SUCCESS.json').exists())

    def test_success_lists_written_files(self):
        self.compare()
        success = json.loads((self.report / 'SUCCESS.json').read_text(encoding='utf-8'))
        for name, digest in success['artifacts'].items():
            self.assertEqual(file_digest(self.report / name), digest)


class ModelSelectionTest(ComparisonTestCase):
    model_name = 'SmallWaferCNN'

    def compare_cnn(self):
        return self.compare(model_name='SmallWaferCNN')

    def assert_cnn_rejected(self, code):
        with self.assertRaisesRegex(ValidationError, code):
            self.compare_cnn()
        self.assertFalse(self.report.exists())

    def test_selected_model_names_files_and_reports(self):
        result = self.compare_cnn()

        self.assertEqual(result['model_name'], 'SmallWaferCNN')
        saved = json.loads((self.report / 'comparison.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['model_name'], 'SmallWaferCNN')
        report = (self.report / 'comparison.md').read_text(encoding='utf-8')
        self.assertTrue(report.startswith('# SmallWaferCNN+D4 가중치 대조: scratch0_7'))
        self.assertNotIn('ResNet18', report)
        self.assertIsNone(result['selected_model'])

    def test_candidate_is_compared_to_same_model_baseline_by_seed(self):
        matrix = matrix_with([(0, 8, 7)])
        self.write_run(self.candidate, 43, matrix, self.new_code)
        self.compare_cnn()

        rows = (self.report / 'seed_differences.csv').read_text(encoding='utf-8-sig')
        self.assertIn('43,Scratch,fp,0,7,7', rows)
        self.assertIn('42,Scratch,fp,0,0,0', rows)

    def test_default_model_does_not_read_cnn_runs(self):
        self.assert_rejected('COMPARE_RUN_INCOMPLETE')

    def test_other_model_baseline_folder_is_rejected(self):
        for seed in SEEDS:
            shutil.rmtree(self.baseline / f'SmallWaferCNN_d4_seed{seed}')
            self.model_name = 'ResNet18'
            self.write_run(self.baseline, seed, matrix_with(), self.old_code, legacy=True)
            self.model_name = 'SmallWaferCNN'

        self.assert_cnn_rejected('COMPARE_RUN_INCOMPLETE')

    def test_other_model_record_in_selected_folder_is_rejected(self):
        self.model_name = 'ResNet18'
        self.write_run(self.root / 'other', 43, matrix_with(), self.new_code)
        self.model_name = 'SmallWaferCNN'
        target = self.candidate / 'SmallWaferCNN_d4_seed43'
        shutil.rmtree(target)
        (self.root / 'other' / 'ResNet18_d4_seed43').rename(target)

        self.assert_cnn_rejected('COMPARE_RUN_CONTRACT_MISMATCH')

    def test_other_augmentation_is_rejected(self):
        self.write_run(
            self.candidate, 43, matrix_with(), self.new_code, config={'augmentation': 'none'},
        )
        self.assert_cnn_rejected('COMPARE_RUN_CONTRACT_MISMATCH')

    def test_missing_seed_is_rejected(self):
        shutil.rmtree(self.candidate / 'SmallWaferCNN_d4_seed44')
        self.assert_cnn_rejected('COMPARE_RUN_INCOMPLETE')

    def test_seed_record_mismatch_is_rejected(self):
        target = self.candidate / 'SmallWaferCNN_d4_seed43'
        shutil.rmtree(target)
        (self.candidate / 'SmallWaferCNN_d4_seed44').rename(target)
        self.write_run(self.candidate, 44, matrix_with(), self.new_code)

        self.assert_cnn_rejected('COMPARE_RUN_CONTRACT_MISMATCH')

    def test_other_condition_is_rejected(self):
        self.write_run(self.candidate, 43, matrix_with(), self.new_code, config={'epochs': 5})
        self.assert_cnn_rejected('WEIGHT_COMPARE_CONDITION_MISMATCH')

    def test_wrong_recorded_weights_are_rejected(self):
        self.write_run(
            self.candidate, 42, matrix_with(), self.new_code,
            recorded={'weights': {name: 1.0 for name in CLASS_NAMES}},
        )
        self.assert_cnn_rejected('WEIGHT_COMPARE_RECORD_MISMATCH')

    def test_code_difference_stays_visible(self):
        result = self.compare_cnn()

        self.assertFalse(result['code']['same_code'])
        self.assertEqual(result['code']['differing_code_files'], ['train.py'])

    def test_unsupported_model_is_rejected_before_reading(self):
        with self.assertRaisesRegex(ValidationError, 'WEIGHT_COMPARE_MODEL_UNSUPPORTED'):
            self.compare(model_name='MobileNetV3Small')
        self.assertFalse(self.report.exists())


class CommandLineTest(unittest.TestCase):
    base = ['weight_compare', '--candidate-root', 'c', '--report-dir', 'r']
    result = {
        'status': 'weight_comparison_complete',
        'candidate_name': 'c',
        'selection_status': '선정 보류',
        'code': {'same_code': True, 'differing_code_files': []},
        'all_reference_items_passed': False,
    }

    def run_main(self, *options):
        with mock.patch.object(analysis, 'compare_candidate', return_value=self.result) as call:
            with mock.patch('sys.argv', self.base + list(options)):
                with mock.patch('builtins.print'):
                    self.assertEqual(analysis.main(), 0)
        return call.call_args

    def test_existing_command_keeps_resnet_default(self):
        self.assertEqual(self.run_main().kwargs['model_name'], 'ResNet18')

    def test_model_option_selects_cnn(self):
        call = self.run_main('--model', 'SmallWaferCNN')
        self.assertEqual(call.kwargs['model_name'], 'SmallWaferCNN')

    def test_mobilenet_option_is_rejected(self):
        with mock.patch('sys.argv', self.base + ['--model', 'MobileNetV3Small']):
            with mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit):
                    analysis.main()


class BoundaryTest(unittest.TestCase):
    def test_candidate_configs_do_not_name_a_model(self):
        for name in ('cap1_5', 'alpha0_75', 'scratch0_7'):
            config = json.loads((CONFIG_DIR / f'{name}.json').read_text(encoding='utf-8'))
            self.assertFalse(any('model' in key for key in config))

    def test_comparison_stays_outside_training_fingerprint(self):
        self.assertNotIn('weight_compare.py', RUN_CODE_FILES)
        self.assertIn('train.py', RUN_CODE_FILES)


if __name__ == '__main__':
    unittest.main()

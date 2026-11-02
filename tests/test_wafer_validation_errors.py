"""ResNet18+D4 Validation 오분류 분석의 합성 데이터 테스트.

실제 캐시·checkpoint·GPU 없이 작은 가짜 모델로 계약 검증, 재현 대조, 집계,
이미지 선정, 산출물 공개 흐름을 확인한다.
"""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.wafer_dl import compare, validation_errors as analysis
from src.wafer_dl.cache import file_digest
from src.wafer_dl.compare import RUN_CODE_FILES, RUN_COMPLETE_STATUS, RUN_CONTRACT_VERSION
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


ROWS = 18
LOC = CLASS_NAMES.index('Loc')
NONE = CLASS_NAMES.index('None')
RANDOM = CLASS_NAMES.index('Random')
SCRATCH = CLASS_NAMES.index('Scratch')
TRAIN_CONFIG = {'seed': 42, 'augmentation': 'd4', 'class_weighted': True, 'max_class_weight': 10.0}


class FakeValidationDataset:
    """클래스마다 두 표본을 가진 Validation 대체 Dataset."""

    protocol = 'wm_split_test'
    source_split_manifest_sha256 = 'a' * 64
    transform_config = {'transform_version': 'wm_resize_v1'}
    opened = []
    closed = 0

    def __init__(self, cache_dir, split='train', allow_test=False):
        type(self).opened.append((split, allow_test))
        self.records = [
            ('validation/shard_000000.npy', index, index % 9, f'sample_{index:02d}')
            for index in range(ROWS)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image = torch.zeros((2, 8, 8), dtype=torch.float32)
        image[1, 1:7, 1:7] = 1
        image[0, 2, 2] = 1
        # 모델이 표본 위치를 알 수 있도록 유효 채널의 첫 칸에 번호를 넣는다.
        image[1, 0, 0] = index + 1
        return {
            'image': image,
            'target': torch.tensor(index % 9, dtype=torch.int64),
            'sample_id': f'sample_{index:02d}',
        }

    def close(self):
        type(self).closed += 1


class TableModel(nn.Module):
    """저장된 표대로 표본별 logit을 돌려주는 가짜 모델."""

    modes = []

    def __init__(self, num_classes=9):
        super().__init__()
        self.register_buffer('table', torch.zeros((ROWS, num_classes)))

    def forward(self, images):
        type(self).modes.append(self.training)
        return self.table[images[:, 1, 0, 0].long() - 1]


def correct_predictions():
    """모든 표본의 정답 라벨 목록을 만든다."""
    return [index % 9 for index in range(ROWS)]


def table_state(predictions):
    """표본별 예측이 나오도록 만든 모델 상태를 돌려준다."""
    table = torch.zeros((ROWS, 9))
    table[range(ROWS), predictions] = 5.0
    return {'table': table}


def confusion(predictions):
    """합성 예측의 혼동행렬을 계산한다."""
    matrix = [[0] * 9 for _ in range(9)]
    for index, predicted in enumerate(predictions):
        matrix[index % 9][predicted] += 1
    return matrix


class AnalysisTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'cache'
        self.runs = self.root / 'runs'
        self.output = self.root / 'out' / 'analysis'
        self.cache.mkdir()
        self.runs.mkdir()
        (self.cache / 'cache.json').write_text(json.dumps({
            'status': 'cache_complete',
            'split_protocol_id': FakeValidationDataset.protocol,
            'source_split_manifest_sha256':
                FakeValidationDataset.source_split_manifest_sha256,
            'train_class_counts': [5] * 9,
        }), encoding='utf-8')

        FakeValidationDataset.opened = []
        FakeValidationDataset.closed = 0
        TableModel.modes = []

        for target, name, value in (
            (analysis, 'WaferDataset', FakeValidationDataset),
            (analysis, 'build_model', lambda name, num_classes=9: TableModel(num_classes)),
            (compare, 'build_model', lambda name, num_classes=9: TableModel(num_classes)),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.write_run(42)
        self.write_run(43)
        self.write_run(44)

    def run_dir(self, seed):
        return self.runs / f'ResNet18_d4_seed{seed}'

    def write_run(self, seed, predictions=None, stored=None, **contract_changes):
        """검증을 통과하는 합성 실행 폴더를 쓴다."""
        predictions = predictions or correct_predictions()
        stored = stored or confusion(predictions)
        run_dir = self.run_dir(seed)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir()

        dataset = FakeValidationDataset
        config = {**TRAIN_CONFIG, 'seed': seed}
        torch.save({
            'checkpoint_version': 'wm_cnn_checkpoint_v2',
            'model_name': 'ResNet18',
            'model_config': {'num_classes': 9, 'input_channels': 2},
            'model_state_dict': table_state(predictions),
            'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
            'transform_config': dataset.transform_config,
            'split_protocol_id': dataset.protocol,
            'source_split_manifest_sha256': dataset.source_split_manifest_sha256,
            'training_config': config,
            'augmentation': 'd4',
            'selected_epoch': 2,
            'selection_metric': 'macro_f1',
            'validation_metrics': {'confusion_matrix': stored},
        }, run_dir / 'best_model.pt')

        (run_dir / 'training_report.json').write_text(json.dumps({
            'status': 'training_complete',
            'test_split_used': False,
            'device': 'cpu',
            'model_name': 'ResNet18',
            'seed': seed,
            'augmentation': 'd4',
            'split_protocol_id': dataset.protocol,
            'selected_epoch': 2,
            'validation_rows': ROWS,
            'history': [
                {'epoch': 1, 'validation': {'confusion_matrix': [[0] * 9] * 9}},
                {'epoch': 2, 'validation': {'confusion_matrix': stored}},
            ],
        }), encoding='utf-8')

        (run_dir / 'inference.json').write_text(json.dumps({
            'device': 'cpu',
            'ms_per_sample': 1.0,
        }), encoding='utf-8')

        contract = {
            'contract_version': RUN_CONTRACT_VERSION,
            'model_name': 'ResNet18',
            'seed': seed,
            'device': 'cpu',
            'training_config': config,
            'split_protocol_id': dataset.protocol,
            'source_split_manifest_sha256': dataset.source_split_manifest_sha256,
            'cache_sha256': file_digest(self.cache / 'cache.json'),
            'code_fingerprint': compare.code_fingerprint(),
            **contract_changes,
        }
        (run_dir / 'SUCCESS.json').write_text(json.dumps({
            'status': RUN_COMPLETE_STATUS,
            'contract': contract,
            'artifacts': {
                name: file_digest(run_dir / name) for name in compare.RUN_ARTIFACTS
            },
        }), encoding='utf-8')

    def analyze(self, **options):
        return analysis.analyze(self.cache, self.runs, self.output, device='cpu', **options)

    def assert_rejected(self, code, **options):
        with self.assertRaisesRegex(ValidationError, code):
            self.analyze(**options)
        self.assertFalse(self.output.exists())


class ValidationOnlyInferenceTest(AnalysisTestCase):
    def test_opens_validation_only_in_order_without_shuffle(self):
        loader = analysis.DataLoader
        with mock.patch.object(analysis, 'DataLoader', side_effect=loader) as spy:
            self.analyze()

        self.assertEqual(set(FakeValidationDataset.opened), {('validation', False)})
        self.assertEqual({call.kwargs['shuffle'] for call in spy.call_args_list}, {False})
        with (self.output / 'predictions_seed42.csv').open(encoding='utf-8-sig') as stream:
            ids = [row['sample_id'] for row in csv.DictReader(stream)]
        self.assertEqual(ids, [f'sample_{index:02d}' for index in range(ROWS)])

    def test_no_training_path_or_test_path_is_called(self):
        with (
            mock.patch('src.wafer_dl.evaluate.evaluate') as test_evaluation,
            mock.patch('src.wafer_dl.train.train') as training,
            mock.patch.object(torch.optim.AdamW, 'step') as step,
            mock.patch.object(torch.Tensor, 'backward') as backward,
        ):
            self.analyze()

        for called in (test_evaluation, training, step, backward):
            self.assertFalse(called.called)

    def test_model_runs_in_eval_mode_and_weights_stay_unchanged(self):
        restored = analysis._restore_model(
            {
                'model_name': 'ResNet18',
                'model_state_dict': table_state(correct_predictions()),
            },
            torch.device('cpu'),
        )
        before = restored.table.clone()

        analysis.predict_validation(
            restored,
            FakeValidationDataset('cache', 'validation'),
            4,
            torch.device('cpu'),
            self.root / 'predictions.csv',
        )

        self.assertFalse(restored.training)
        self.assertEqual(set(TableModel.modes), {False})
        self.assertTrue(torch.equal(restored.table, before))

    def test_inputs_stay_unchanged(self):
        paths = [self.cache / 'cache.json', *self.run_dir(42).iterdir()]
        before = {path: file_digest(path) for path in paths}
        self.analyze()
        self.assertEqual({path: file_digest(path) for path in paths}, before)


class ContractRejectionTest(AnalysisTestCase):
    def test_code_mismatch_is_rejected_before_prediction(self):
        files = {'train.py': 'b' * 64}
        self.write_run(43, code_fingerprint={
            'sha256': compare._digest_files(files),
            'files': files,
        })
        self.assert_rejected('VALIDATION_ERRORS_CODE_MISMATCH')
        self.assertEqual(TableModel.modes, [])
        self.assertFalse(self.output.with_name('analysis.tmp').exists())

    def test_cache_mismatch_is_rejected(self):
        self.write_run(42, cache_sha256='c' * 64)
        self.assert_rejected('VALIDATION_ERRORS_CACHE_MISMATCH')

    def test_other_split_cache_is_rejected(self):
        """실행 자체는 일관돼도 지정한 캐시의 분할이 다르면 거부한다."""
        for name, value in (
            ('protocol', 'wm_split_other'),
            ('source_split_manifest_sha256', 'd' * 64),
        ):
            with self.subTest(name=name):
                with mock.patch.object(FakeValidationDataset, name, value):
                    self.assert_rejected('VALIDATION_ERRORS_CACHE_MISMATCH')

    def test_model_mismatch_is_rejected(self):
        self.write_run(42, model_name='SmallWaferCNN')
        self.assert_rejected('COMPARE_RUN_CONTRACT_MISMATCH')

    def test_class_mapping_mismatch_is_rejected(self):
        run_dir = self.run_dir(42)
        checkpoint = torch.load(run_dir / 'best_model.pt', weights_only=True)
        checkpoint['class_mapping'] = {'0': 'Other'}
        torch.save(checkpoint, run_dir / 'best_model.pt')

        success = json.loads((run_dir / 'SUCCESS.json').read_text(encoding='utf-8'))
        success['artifacts']['best_model.pt'] = file_digest(run_dir / 'best_model.pt')
        (run_dir / 'SUCCESS.json').write_text(json.dumps(success), encoding='utf-8')

        self.assert_rejected('CHECKPOINT_CACHE_CONTRACT_MISMATCH')

    def test_changed_artifact_is_rejected(self):
        (self.run_dir(42) / 'training_report.json').write_text('{}', encoding='utf-8')
        self.assert_rejected('COMPARE_RUN_INTEGRITY_FAILED')

    def test_missing_run_is_not_replaced(self):
        shutil.rmtree(self.run_dir(44))
        self.assert_rejected('COMPARE_RUN_INCOMPLETE')

    def test_nonfinite_output_and_wrong_dimension_are_rejected(self):
        class NaNModel(TableModel):
            def forward(self, images):
                return super().forward(images) * float('nan')

        class WrongSizeModel(TableModel):
            def forward(self, images):
                return super().forward(images)[:, :8]

        for model, code in (
            (NaNModel, 'NONFINITE_LOGITS'),
            (WrongSizeModel, 'INVALID_LOGIT_SHAPE'),
        ):
            with self.subTest(code=code):
                with mock.patch.object(
                    analysis, 'build_model', lambda name, num_classes=9: model(num_classes)
                ):
                    self.assert_rejected(code)
                self.assertFalse(self.output.with_name('analysis.tmp').exists())

    def test_nonfinite_weight_is_rejected(self):
        state = table_state(correct_predictions())
        state['table'][0, 0] = float('inf')
        with self.assertRaisesRegex(ValidationError, 'NONFINITE_MODEL_WEIGHT'):
            analysis._restore_model(
                {'model_name': 'ResNet18', 'model_state_dict': state}, torch.device('cpu')
            )

    def test_strict_loading_rejects_other_state(self):
        with self.assertRaisesRegex(ValidationError, 'VALIDATION_ERRORS_CHECKPOINT_INVALID'):
            analysis._restore_model(
                {'model_name': 'ResNet18', 'model_state_dict': {'other': torch.zeros(1)}},
                torch.device('cpu'),
            )

    def test_missing_and_duplicate_samples_are_rejected(self):
        dataset = FakeValidationDataset('cache', 'validation')
        rows = [
            {'sample_id': record[3], 'actual': record[2], 'predicted': record[2]}
            for record in dataset.records
        ]
        analysis.require_all_samples(rows, dataset)

        for broken in (rows[:-1], [rows[0], *rows[:-1]]):
            with self.assertRaisesRegex(ValidationError, 'VALIDATION_ERRORS_SAMPLE_MISMATCH'):
                analysis.require_all_samples(broken, dataset)

    def test_different_labels_between_seeds_are_rejected(self):
        first = [{'sample_id': 'a', 'actual': 0, 'predicted': 0}]
        second = [{'sample_id': 'a', 'actual': 1, 'predicted': 0}]
        with self.assertRaisesRegex(ValidationError, 'VALIDATION_ERRORS_SEED_MISMATCH'):
            analysis.join_seed_rows({42: first, 43: second})

    def test_invalid_options_are_rejected(self):
        for options in (
            {'seeds': ()},
            {'seeds': (42, 42)},
            {'seeds': (1,)},
            {'batch_size': 0},
            {'max_images': analysis.MAX_IMAGES_PER_GROUP + 1},
            {'images_per_page': analysis.MAX_IMAGES_PER_PAGE + 1},
        ):
            with self.subTest(options=options):
                self.assert_rejected('INVALID_ANALYSIS', **options)

    def test_unavailable_gpu_is_not_replaced_with_cpu(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(ValidationError, 'CUDA_DEVICE_UNAVAILABLE'):
                analysis._select_device('cuda')


class ReproductionTest(AnalysisTestCase):
    def test_recomputed_confusion_matches_stored_metrics(self):
        predictions = correct_predictions()
        predictions[LOC] = SCRATCH
        self.write_run(42, predictions)

        summary = self.analyze()

        first = summary['seeds'][0]
        self.assertTrue(first['reproduction']['matched'])
        self.assertEqual(first['metrics']['confusion_matrix'], confusion(predictions))

        with (self.output / 'predictions_seed42.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        recomputed = analysis.confusion_from_rows([
            {
                'actual': CLASS_NAMES.index(row['actual_label']),
                'predicted': CLASS_NAMES.index(row['predicted_label']),
            }
            for row in rows
        ])
        self.assertEqual(recomputed.tolist(), confusion(predictions))

    def test_mismatch_keeps_diagnosis_without_completion(self):
        stored = confusion(correct_predictions())
        stored[LOC][LOC] -= 1
        stored[LOC][SCRATCH] += 1
        self.write_run(43, stored=stored)

        self.assert_rejected('VALIDATION_ERRORS_CONFUSION_MISMATCH')

        diagnosis = self.output.with_name('analysis.mismatch_1')
        self.assertFalse((diagnosis / 'SUCCESS.json').exists())
        self.assertFalse((diagnosis / 'report.md').exists())

        report = json.loads((diagnosis / 'MISMATCH.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], analysis.ANALYSIS_MISMATCH_STATUS)
        self.assertEqual(
            [item['matched'] for item in report['seeds']], [True, False, True]
        )
        self.assertEqual(len(report['seeds'][1]['differences']), 2)


class AggregationTest(unittest.TestCase):
    seeds = (42, 43, 44)

    def rows(self):
        """표본 세 개의 seed별 예측을 만든다."""
        table = (
            ('a', LOC, (SCRATCH, SCRATCH, SCRATCH)),
            ('b', LOC, (LOC, SCRATCH, NONE)),
            ('c', LOC, (LOC, LOC, LOC)),
            ('d', NONE, (SCRATCH, NONE, NONE)),
            ('e', RANDOM, (LOC, SCRATCH, RANDOM)),
        )
        seed_rows = {seed: [] for seed in self.seeds}
        for sample_id, actual, predictions in table:
            for seed, predicted in zip(self.seeds, predictions):
                seed_rows[seed].append({
                    'sample_id': sample_id,
                    'actual': actual,
                    'predicted': predicted,
                })
        return analysis.join_seed_rows(seed_rows)

    def test_union_and_intersection_count_distinct_samples(self):
        repeats = analysis.repeat_summary(self.rows(), self.seeds)
        self.assertEqual(repeats['union_wrong_samples'], 4)
        self.assertEqual(repeats['intersection_wrong_samples'], 1)
        self.assertEqual(
            repeats['samples_by_wrong_seed_count'], {0: 1, 1: 1, 2: 2, 3: 1}
        )

    def test_group_filters_and_counts(self):
        joined = self.rows()
        groups = {group['id']: group for group in analysis.PRIORITY_GROUPS}

        summary, entries = analysis.group_errors(joined, groups['loc_to_scratch'], self.seeds)
        self.assertEqual([entry['sample_id'] for entry in entries], ['a', 'b'])
        self.assertEqual(summary['actual_label_samples'], 3)
        self.assertEqual(summary['distinct_samples'], 2)
        self.assertEqual(summary['seed_error_counts'], {42: 1, 43: 2, 44: 1})
        self.assertEqual(summary['samples_in_every_seed'], 1)

        _, entries = analysis.group_errors(joined, groups['none_to_scratch'], self.seeds)
        self.assertEqual([entry['sample_id'] for entry in entries], ['d'])

        _, entries = analysis.group_errors(joined, groups['random_to_other'], self.seeds)
        self.assertEqual([entry['hit_seeds'] for entry in entries], [[42, 43]])

    def test_selection_is_deterministic(self):
        group = {'id': 'x', 'actual': 'Loc', 'predicted': 'Scratch'}
        seed_rows = {seed: [] for seed in self.seeds}
        for sample_id, hit_count in (('z', 3), ('c', 1), ('m', 3), ('a', 2), ('b', 1)):
            for number, seed in enumerate(self.seeds):
                seed_rows[seed].append({
                    'sample_id': sample_id,
                    'actual': LOC,
                    'predicted': SCRATCH if number < hit_count else LOC,
                })
        joined = analysis.join_seed_rows(seed_rows)

        _, entries = analysis.group_errors(joined, group, self.seeds)
        _, shuffled = analysis.group_errors(list(reversed(joined)), group, self.seeds)

        self.assertEqual([entry['sample_id'] for entry in entries], ['m', 'z', 'a', 'b', 'c'])
        self.assertEqual(entries, shuffled)

    def test_top_flows_keep_seed_values_separate(self):
        results = []
        for seed, count in ((42, 3), (43, 1)):
            matrix = [[0] * 9 for _ in range(9)]
            matrix[LOC][SCRATCH] = count
            matrix[LOC][LOC] = 9
            results.append({'seed': seed, 'metrics': {'confusion_matrix': matrix}})

        flows = analysis.top_flows(results)
        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0]['seed_counts'], {42: 3, 43: 1})


class ImageTest(AnalysisTestCase):
    def test_composite_map_keeps_pixels_without_transform(self):
        image = torch.zeros((2, 3, 3))
        image[1, 0:2, 0:2] = 1
        image[0, 0, 1] = 1
        np.testing.assert_array_equal(
            analysis.composite_map(image),
            np.array([[1, 2, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
        )

    def test_legend_has_three_consistent_labels(self):
        self.assertEqual(
            [handle.get_label() for handle in analysis.legend_handles()],
            ['Outside wafer', 'Valid die', 'Defect die'],
        )
        self.assertEqual(
            analysis.MAP_COLORS.N, len(analysis.MAP_LEGEND)
        )

    def test_page_limit_and_omitted_count(self):
        group = {'id': 'x', 'actual': 'Loc', 'predicted': 'Scratch'}
        entries = [
            {
                'position': index,
                'sample_id': f'id/../{index}',
                'actual_label': 'Loc',
                'predictions': {42: 'Scratch'},
            }
            for index in range(7)
        ]
        (self.root / 'images').mkdir()
        dataset = FakeValidationDataset('cache', 'validation')

        images = analysis.render_group(self.root, group, entries, dataset, (42,), 5, 2)

        self.assertEqual(images['selected_samples'], 5)
        self.assertEqual(images['omitted_samples'], 2)
        self.assertEqual(
            images['pages'],
            ['images/x_page01.png', 'images/x_page02.png', 'images/x_page03.png'],
        )
        self.assertEqual(
            sorted(path.name for path in (self.root / 'images').iterdir()),
            ['x_page01.png', 'x_page02.png', 'x_page03.png'],
        )
        for page in images['pages']:
            self.assertEqual((self.root / page).read_bytes()[:4], b'\x89PNG')

    def test_samples_are_read_only_while_drawing(self):
        group = {'id': 'x', 'actual': 'Loc', 'predicted': 'Scratch'}
        entries = [
            {
                'position': index,
                'sample_id': str(index),
                'actual_label': 'Loc',
                'predictions': {42: 'Scratch'},
            }
            for index in range(3)
        ]
        (self.root / 'images').mkdir()
        dataset = FakeValidationDataset('cache', 'validation')

        with mock.patch.object(
            FakeValidationDataset, '__getitem__', autospec=True,
            side_effect=FakeValidationDataset.__getitem__,
        ) as reads:
            analysis.render_group(self.root, group, entries, dataset, (42,), 2, 2)

        self.assertEqual(reads.call_count, 2)


class PublicationTest(AnalysisTestCase):
    def predictions_with_errors(self):
        predictions = correct_predictions()
        predictions[LOC] = SCRATCH
        predictions[NONE] = SCRATCH
        predictions[RANDOM] = LOC
        return predictions

    def test_complete_output_and_report(self):
        self.write_run(42, self.predictions_with_errors())
        self.write_run(43, self.predictions_with_errors())

        summary = self.analyze()

        success = json.loads((self.output / 'SUCCESS.json').read_text(encoding='utf-8'))
        self.assertEqual(success['status'], analysis.ANALYSIS_COMPLETE_STATUS)
        self.assertTrue(success['reproduction_matched'])
        for name, digest in success['artifacts'].items():
            self.assertEqual(file_digest(self.output / name), digest)

        for name in (
            'predictions_seed44.csv', 'metrics_seed44.json', 'joined_predictions.csv',
            'priority_errors.csv', 'analysis.json', 'report.md',
            'images/loc_to_scratch_page01.png',
        ):
            self.assertTrue((self.output / name).is_file(), name)

        report = (self.output / 'report.md').read_text(encoding='utf-8')
        for text in ('Validation 18행만', 'Outside wafer', '`None`은 데이터셋 라벨'):
            self.assertIn(text, report)
        self.assertIn('\n\n![loc_to_scratch](images/loc_to_scratch_page01.png)\n\n', report)
        self.assertIn('시각적 관찰과 가설', report)
        self.assertIn('보정된 신뢰도가 아닙니다', report)
        self.assertIn('라벨 오류나 가중치 문제를 판정하지', report)

        self.assertTrue(summary['complete_seed_set'])
        self.assertFalse(summary['ensemble'])
        self.assertEqual(summary['inference_augmentation'], 'none')

    def test_zero_errors_publish_without_pages(self):
        summary = self.analyze()

        for group in summary['priority_groups']:
            self.assertEqual(group['distinct_samples'], 0)
            self.assertEqual(group['images']['pages'], [])

        report = (self.output / 'report.md').read_text(encoding='utf-8')
        self.assertIn('이 그룹의 오류 표본은 없습니다.', report)

    def test_one_seed_is_marked_as_partial(self):
        summary = self.analyze(seeds=(42,))
        self.assertFalse(summary['complete_seed_set'])
        self.assertIn('확인용 부분 실행', (self.output / 'report.md').read_text(encoding='utf-8'))

    def test_omitted_images_stay_in_full_list(self):
        predictions = correct_predictions()
        predictions[LOC] = SCRATCH
        predictions[LOC + 9] = SCRATCH
        self.write_run(42, predictions)

        summary = self.analyze(max_images=1)

        group = summary['priority_groups'][0]
        self.assertEqual(group['distinct_samples'], 2)
        self.assertEqual(group['images']['omitted_samples'], 1)

        with (self.output / 'priority_errors.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            [row['image_selected'] for row in rows], ['True', 'False']
        )

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir(parents=True)
        marker = self.output / 'keep.txt'
        marker.write_text('기존', encoding='utf-8')

        with self.assertRaisesRegex(ValidationError, 'VALIDATION_ERRORS_OUTPUT_EXISTS'):
            self.analyze()

        self.assertEqual(marker.read_text(encoding='utf-8'), '기존')

    def test_partial_folder_is_not_accepted(self):
        partial = self.output.with_name('analysis.tmp')
        partial.mkdir(parents=True)

        self.assert_rejected('VALIDATION_ERRORS_PARTIAL_EXISTS')
        self.assertTrue(partial.exists())

    def test_failure_closes_dataset_and_keeps_partial_without_success(self):
        with mock.patch.object(
            analysis, 'render_group', side_effect=OSError('디스크 오류')
        ):
            with self.assertRaises(OSError):
                self.analyze()

        self.assertEqual(FakeValidationDataset.closed, 1)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name('analysis.tmp').exists())

        partial = self.output.with_name('analysis.incomplete_1')
        self.assertTrue((partial / 'predictions_seed42.csv').is_file())
        self.assertFalse((partial / 'SUCCESS.json').exists())

    def test_input_change_during_analysis_is_not_published(self):
        original = analysis.write_report

        def change_input(*args):
            original(*args)
            (self.cache / 'cache.json').write_text('{}', encoding='utf-8')

        with mock.patch.object(analysis, 'write_report', change_input):
            self.assert_rejected('VALIDATION_ERRORS_INPUT_CHANGED')


class FingerprintBoundaryTest(unittest.TestCase):
    def test_analysis_code_does_not_change_run_fingerprint(self):
        """새 분석 경로는 학습 경로 밖이라 기존 실행 식별값에 영향을 주지 않는다."""
        self.assertNotIn('validation_errors.py', RUN_CODE_FILES)
        self.assertIn('validation_errors.py', analysis.ANALYSIS_CODE_FILES)
        self.assertEqual(
            set(analysis.analysis_fingerprint()['files']) - set(RUN_CODE_FILES),
            {'evaluate.py', 'validation_errors.py'},
        )


if __name__ == '__main__':
    unittest.main()

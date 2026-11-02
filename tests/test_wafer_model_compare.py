"""WM-811K 모델 비교의 합성 데이터 테스트.

실제 캐시와 GPU 없이 모델 계약, D4 증강, 실행·재개 흐름, 집계 표시를 확인한다.
"""

import argparse
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from src.wafer_dl.augment import (
    D4TrainDataset,
    D4_TRANSFORM_COUNT,
    apply_d4,
    d4_transform_index,
)
from src.wafer_dl.compare import (
    PACKAGE_DIR,
    RUN_CODE_FILES,
    _add_matrix_arguments,
    _digest_files,
    code_fingerprint,
    measure_inference,
    plan_runs,
    rank_candidates,
    run_all,
    run_name,
    summarize,
    summarize_runs,
    verify_run,
)
from src.wafer_dl.evaluate import validate_checkpoint
from src.wafer_dl.models import (
    DEFAULT_COMPARE_MODELS,
    MODEL_NAMES,
    SmallWaferCNN,
    build_model,
)
from src.wafer_dl.train import load_train_config, train
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


class SyntheticDataset:
    """클래스마다 두 표본을 가진 Train/Validation 대체 Dataset."""

    protocol = 'wm_split_test'
    source_split_manifest_sha256 = 'a' * 64
    transform_config = {'transform_version': 'wm_resize_v1'}
    class_counts = (2,) * 9

    def __init__(self, cache_dir, split='train', allow_test=False):
        self.split = split

    def __len__(self):
        return 18

    def __getitem__(self, index):
        image = torch.zeros((2, 64, 64), dtype=torch.float32)
        image[1] = 1
        image[0, index % 9, :] = 1
        return {
            'image': image,
            'target': torch.tensor(index % 9, dtype=torch.int64),
            'sample_id': str(index),
        }

    def close(self):
        pass


def write_config(path, **changes):
    """1 epoch·작은 배치의 학습 설정 파일을 쓴다."""
    path.write_text(json.dumps({
        'training_version': 'wm_cnn_train_v1',
        'batch_size': 4,
        'epochs': 1,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'patience': 1,
        'seed': 42,
        'class_weighted': True,
        'max_class_weight': 10.0,
        'selection_metric': 'macro_f1',
        **changes,
    }), encoding='utf-8')


def write_cache_report(cache_dir):
    """실행 계약이 읽는 항목만 가진 합성 캐시 보고서를 쓴다."""
    cache_dir.mkdir()
    (cache_dir / 'cache.json').write_text(json.dumps({
        'status': 'cache_complete',
        'split_protocol_id': SyntheticDataset.protocol,
        'source_split_manifest_sha256': SyntheticDataset.source_split_manifest_sha256,
    }), encoding='utf-8')


def other_fingerprint(digest='b' * 64):
    """현재 코드와 다른 합성 실행 코드 식별값을 만든다."""
    files = {'train.py': digest}
    return {'sha256': _digest_files(files), 'files': files}


def asymmetric_image():
    """모든 D4 변환 결과가 서로 달라지는 2채널 이진 이미지를 만든다."""
    image = torch.zeros((2, 8, 8), dtype=torch.float32)
    image[1, 1:7, 1:7] = 1
    image[0, 1, 1:5] = 1
    image[0, 2, 1] = 1
    return image


class WaferModelFactoryTest(unittest.TestCase):
    def test_default_comparison_has_three_cnn_models_without_mlp(self):
        """기본 실험군은 CNN 계열 3개이고 MLP 계열은 선택할 때만 실행한다."""
        self.assertEqual(
            DEFAULT_COMPARE_MODELS,
            ('SmallWaferCNN', 'ResNet18', 'MobileNetV3Small'),
        )
        for name in ('WaferMLP', 'ResNet18MLP', 'MobileNetV3SmallMLP'):
            with self.subTest(name=name):
                self.assertNotIn(name, DEFAULT_COMPARE_MODELS)
                self.assertIn(name, MODEL_NAMES)

    def test_plan_run_summarize_and_cli_share_default_models(self):
        """계획·실행·집계·CLI의 기본 모델 목록이 일치한다."""
        for function in (plan_runs, run_all, summarize):
            with self.subTest(function=function.__name__):
                self.assertEqual(
                    inspect.signature(function).parameters['models'].default,
                    DEFAULT_COMPARE_MODELS,
                )

        parser = argparse.ArgumentParser()
        _add_matrix_arguments(parser)
        self.assertEqual(parser.parse_args([]).models, list(DEFAULT_COMPARE_MODELS))
        self.assertEqual(
            parser.parse_args(['--models', 'WaferMLP']).models, ['WaferMLP']
        )

    def test_all_models_accept_two_channels_and_return_nine_logits(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name).eval()
                logits = model(torch.zeros((2, 2, 128, 128), dtype=torch.float32))
                self.assertEqual(tuple(logits.shape), (2, len(CLASS_NAMES)))

    def test_parameter_counts_match_experiment_plan(self):
        """실험계획에 기록한 파라미터 수와 모델 구조가 일치한다."""
        expected = {
            'SmallWaferCNN': 288_329,
            'WaferMLP': 1_182_729,
            'ResNet18': 11_177_993,
            'ResNet18MLP': 11_307_017,
            'MobileNetV3Small': 1_526_937,
            'MobileNetV3SmallMLP': 1_076_889,
        }
        for name, count in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    sum(value.numel() for value in build_model(name).parameters()), count
                )

    def test_small_model_keeps_existing_class(self):
        self.assertIs(type(build_model('SmallWaferCNN')), SmallWaferCNN)

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, 'INVALID_MODEL_NAME'):
            build_model('ResNet50')


class WaferD4AugmentationTest(unittest.TestCase):
    def test_transforms_are_distinct_and_preserve_binary_mask_relation(self):
        """8가지 변환이 서로 다르고 0/1 값과 defect<=valid를 유지한다."""
        image = asymmetric_image()
        results = [apply_d4(image, index) for index in range(D4_TRANSFORM_COUNT)]

        self.assertEqual(len({tuple(item.flatten().tolist()) for item in results}), 8)
        for result in results:
            self.assertTrue(bool(((result == 0) | (result == 1)).all()))
            self.assertTrue(bool((result[0] <= result[1]).all()))
            self.assertEqual(float(result.sum()), float(image.sum()))

    def test_same_transform_is_applied_to_both_channels(self):
        """채널을 따로 변환한 결과와 같아 두 채널의 위치 관계가 유지된다."""
        image = asymmetric_image()
        for index in range(D4_TRANSFORM_COUNT):
            with self.subTest(index=index):
                result = apply_d4(image, index)
                first = apply_d4(image[[0, 0]], index)[0]
                second = apply_d4(image[[1, 1]], index)[0]
                self.assertTrue(torch.equal(result[0], first))
                self.assertTrue(torch.equal(result[1], second))

    def test_source_image_is_not_changed(self):
        image = asymmetric_image()
        original = image.clone()
        for index in range(D4_TRANSFORM_COUNT):
            apply_d4(image, index)
        self.assertTrue(torch.equal(image, original))

    def test_selection_is_reproducible_and_changes_by_epoch(self):
        """같은 seed·epoch·표본은 같은 변환이고 epoch가 바뀌면 달라질 수 있다."""
        self.assertEqual(d4_transform_index(42, 3, 'a'), d4_transform_index(42, 3, 'a'))
        selected = {d4_transform_index(42, epoch, 'a') for epoch in range(1, 50)}
        self.assertEqual(selected, set(range(D4_TRANSFORM_COUNT)))

    def test_dataset_keeps_label_and_sample_id(self):
        dataset = D4TrainDataset(SyntheticDataset('cache'), 42)
        dataset.set_epoch(2)
        sample = dataset[3]
        self.assertEqual(int(sample['target']), 3)
        self.assertEqual(sample['sample_id'], '3')

    def test_non_train_dataset_is_rejected(self):
        """Validation·Test는 증강 Dataset으로 감쌀 수 없다."""
        for split in ('validation', 'test'):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValidationError, 'AUGMENTATION_TRAIN_ONLY'):
                    D4TrainDataset(SyntheticDataset('cache', split), 42)

    def test_config_without_augmentation_means_none(self):
        """증강 키가 없는 기존 설정은 증강 없음으로 읽는다."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'train.json'
            write_config(path)
            self.assertEqual(load_train_config(path)['augmentation'], 'none')
            write_config(path, augmentation='d4')
            self.assertEqual(load_train_config(path)['augmentation'], 'd4')
            write_config(path, augmentation='cutmix')
            with self.assertRaisesRegex(ValidationError, 'INVALID_TRAIN_CONFIG'):
                load_train_config(path)

    def test_repository_train_config_stays_without_augmentation(self):
        self.assertEqual(
            load_train_config('configs/wm811k/train.json')['augmentation'], 'none'
        )


class WaferCompareTrainingTest(unittest.TestCase):
    def setUp(self):
        patchers = [
            mock.patch('src.wafer_dl.train.WaferDataset', SyntheticDataset),
            mock.patch('src.wafer_dl.compare.WaferDataset', SyntheticDataset),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, root, max_epochs=None, seeds=(1,), models=None, augmentations=('none',)):
        """합성 캐시로 비교 실행을 수행한다."""
        return run_all(
            root / 'cache',
            root / 'runs',
            root / 'train.json',
            'cpu',
            max_epochs,
            models=models or ('SmallWaferCNN',),
            augmentations=augmentations,
            seeds=seeds,
        )

    def _prepare(self, root):
        """학습 설정과 합성 캐시 보고서를 준비한다."""
        write_config(root / 'train.json')
        write_cache_report(root / 'cache')

    def test_each_model_trains_and_loads_from_its_checkpoint(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_config(root / 'train.json')
                output = root / 'run'

                report = train(
                    'cache',
                    output,
                    root / 'train.json',
                    'cpu',
                    model_name=name,
                    seed=7,
                    augmentation='d4',
                )
                checkpoint = torch.load(
                    output / 'best_model.pt',
                    map_location='cpu',
                    weights_only=True,
                )

                self.assertEqual(report['model_name'], name)
                self.assertEqual(report['seed'], 7)
                self.assertEqual(report['augmentation'], 'd4')
                self.assertFalse(report['test_split_used'])
                self.assertEqual(checkpoint['model_name'], name)
                self.assertEqual(checkpoint['augmentation'], 'd4')
                self.assertEqual(checkpoint['training_config']['seed'], 7)
                build_model(name).load_state_dict(checkpoint['model_state_dict'])

    def test_default_arguments_keep_small_cnn_seed_and_no_augmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_config(root / 'train.json')
            report = train('cache', root / 'run', root / 'train.json', 'cpu')
        self.assertEqual(report['model_name'], 'SmallWaferCNN')
        self.assertEqual(report['seed'], 42)
        self.assertEqual(report['augmentation'], 'none')

    def test_invalid_overrides_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_config(root / 'train.json')
            with self.assertRaisesRegex(ValidationError, 'INVALID_SEED_OVERRIDE'):
                train('cache', root / 'run', root / 'train.json', 'cpu', seed=-1)
            with self.assertRaisesRegex(ValidationError, 'INVALID_AUGMENTATION'):
                train('cache', root / 'run', root / 'train.json', 'cpu', augmentation='flip')

    def test_future_evaluation_accepts_candidates_and_rejects_unknown_model(self):
        """후보 checkpoint는 계약을 통과하고 목록 밖 모델은 거부한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_config(root / 'train.json')
            train('cache', root / 'run', root / 'train.json', 'cpu', model_name='ResNet18')
            checkpoint = torch.load(
                root / 'run' / 'best_model.pt', map_location='cpu', weights_only=True
            )

        self.assertEqual(
            validate_checkpoint(checkpoint, SyntheticDataset('cache'), [2] * 9), 0
        )
        checkpoint['model_name'] = 'ResNet50'
        with self.assertRaisesRegex(ValidationError, 'CHECKPOINT_CACHE_CONTRACT_MISMATCH'):
            validate_checkpoint(checkpoint, SyntheticDataset('cache'), [2] * 9)

    def test_inference_measurement_reports_positive_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_config(root / 'train.json')
            train('cache', root / 'run', root / 'train.json', 'cpu')
            result = measure_inference(root / 'run', 'cache', 'cpu')
        self.assertEqual(result['samples_per_pass'], 18)
        self.assertGreater(result['ms_per_sample'], 0)
        self.assertIsNone(result['peak_cuda_memory_bytes'])

    def test_final_folder_without_success_marker_stops_without_changes(self):
        """완료 표식이 없는 결과는 재사용·이동·덮어쓰기 없이 중단한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            final = root / 'runs' / run_name('SmallWaferCNN', 'none', 1)
            final.mkdir(parents=True)
            (final / 'inference.json').write_text('{}', encoding='utf-8')

            with mock.patch('src.wafer_dl.compare.train') as trained:
                with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_INCOMPLETE'):
                    self._run(root)

            trained.assert_not_called()
            self.assertEqual((final / 'inference.json').read_text(encoding='utf-8'), '{}')

    def test_interrupted_temporary_folder_is_preserved_and_retrained(self):
        """끊긴 임시 폴더는 보존하고 해당 실행만 새로 학습한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            final = root / 'runs' / run_name('SmallWaferCNN', 'none', 1)
            leftover = final.with_name(final.name + '.tmp')
            leftover.mkdir(parents=True)

            self.assertEqual(self._run(root), [])

            self.assertTrue(final.with_name(final.name + '.incomplete_1').is_dir())
            self.assertFalse(leftover.exists())
            self.assertTrue((final / 'SUCCESS.json').is_file())

    def test_different_training_config_or_cache_is_rejected(self):
        """학습 설정이나 캐시가 달라지면 기존 결과를 재사용하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])

            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CONTRACT_MISMATCH'):
                self._run(root, max_epochs=2)

            cache_file = root / 'cache' / 'cache.json'
            cache = json.loads(cache_file.read_text(encoding='utf-8'))
            cache['split_protocol_id'] = 'wm_split_other'
            cache_file.write_text(json.dumps(cache), encoding='utf-8')

            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CONTRACT_MISMATCH'):
                self._run(root)

    def test_changed_artifact_is_rejected(self):
        """완료 뒤 바뀐 보고서는 재사용과 집계에서 거부한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])

            final = root / 'runs' / run_name('SmallWaferCNN', 'none', 1)
            with (final / 'training_report.json').open('a', encoding='utf-8') as stream:
                stream.write(' ')

            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_INTEGRITY_FAILED'):
                self._run(root)
            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_INTEGRITY_FAILED'):
                verify_run(final, 'SmallWaferCNN', 'none', 1)

    def test_folder_with_other_condition_is_rejected(self):
        """다른 모델·증강의 결과를 해당 조건으로 인정하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])

            final = root / 'runs' / run_name('SmallWaferCNN', 'none', 1)
            for model_name, augmentation in (('ResNet18', 'none'), ('SmallWaferCNN', 'd4')):
                with self.subTest(model_name=model_name, augmentation=augmentation):
                    with self.assertRaisesRegex(
                        ValidationError, 'COMPARE_RUN_CONTRACT_MISMATCH'
                    ):
                        verify_run(final, model_name, augmentation, 1)

    def test_summary_rejects_runs_from_different_configs(self):
        """학습 조건이 섞인 실행은 하나의 비교로 집계하지 않는다."""
        models = ('SmallWaferCNN', 'MobileNetV3Small')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root, models=models[:1]), [])
            self.assertEqual(self._run(root, max_epochs=2, models=models[1:]), [])

            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CONTRACT_MISMATCH'):
                summarize(
                    root / 'runs',
                    root / 'report',
                    models=models,
                    augmentations=('none',),
                    seeds=(1,),
                )
            self.assertFalse((root / 'report').exists())

    def test_existing_summary_is_preserved(self):
        """이미 있는 집계 폴더는 덮어쓰지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root, seeds=(1, 2)), [])
            (root / 'report').mkdir()

            with self.assertRaisesRegex(ValidationError, 'COMPARE_REPORT_ALREADY_EXISTS'):
                summarize(
                    root / 'runs',
                    root / 'report',
                    models=('SmallWaferCNN',),
                    augmentations=('none',),
                    seeds=(1, 2),
                )
            self.assertEqual(list((root / 'report').iterdir()), [])

    def test_same_code_result_is_reused(self):
        """코드와 계약이 같으면 완료 실행을 다시 학습하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])

            with mock.patch('src.wafer_dl.compare.train') as trained:
                self.assertEqual(self._run(root), [])

            trained.assert_not_called()

    def test_changed_code_rejects_reuse_without_training(self):
        """코드가 바뀌면 완료 실행을 건너뛰지 않고 학습 전에 중단한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])
            success = root / 'runs' / run_name('SmallWaferCNN', 'none', 1) / 'SUCCESS.json'
            before = success.read_bytes()

            with mock.patch(
                'src.wafer_dl.compare.code_fingerprint', return_value=other_fingerprint()
            ), mock.patch('src.wafer_dl.compare.train') as trained:
                with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CODE_MISMATCH'):
                    self._run(root)

            trained.assert_not_called()
            self.assertEqual(success.read_bytes(), before)

    def test_mixed_code_summary_is_rejected_without_report(self):
        """서로 다른 코드의 실행은 집계하지 않고 결과도 공개하지 않는다."""
        models = ('SmallWaferCNN', 'MobileNetV3Small')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root, models=models[:1]), [])

            with mock.patch(
                'src.wafer_dl.compare.code_fingerprint', return_value=other_fingerprint()
            ):
                self.assertEqual(self._run(root, models=models[1:]), [])

            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CODE_MISMATCH'):
                summarize(
                    root / 'runs',
                    root / 'report',
                    models=models,
                    augmentations=('none',),
                    seeds=(1,),
                )
            self.assertFalse((root / 'report').exists())
            self.assertFalse((root / 'report.tmp').exists())

    def test_complete_past_code_set_is_summarized_without_loading(self):
        """한 코드로 완결된 과거 묶음은 현재 구조로 로딩하지 않고 차이를 표시한다."""
        matrix = {'models': ('SmallWaferCNN',), 'augmentations': ('none',), 'seeds': (1,)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])

            with mock.patch(
                'src.wafer_dl.compare.code_fingerprint', return_value=other_fingerprint()
            ), mock.patch('src.wafer_dl.compare.build_model') as loaded:
                result = summarize(root / 'runs', root / 'report', **matrix)

            loaded.assert_not_called()
            self.assertFalse(result['current_code_match'])
            self.assertIn(
                '현재 코드와 다름',
                (root / 'report' / 'comparison.md').read_text(encoding='utf-8'),
            )

    def test_contract_without_code_fingerprint_is_rejected_unchanged(self):
        """식별값이 없는 이전 계약은 승격하지 않고 원본을 그대로 둔다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])

            success = root / 'runs' / run_name('SmallWaferCNN', 'none', 1) / 'SUCCESS.json'
            record = json.loads(success.read_text(encoding='utf-8'))
            record['contract']['contract_version'] = 'wm_compare_run_v2'
            del record['contract']['code_fingerprint']
            success.write_text(json.dumps(record), encoding='utf-8')
            before = success.read_bytes()

            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CONTRACT_OUTDATED'):
                self._run(root)
            with self.assertRaisesRegex(ValidationError, 'COMPARE_RUN_CONTRACT_OUTDATED'):
                summarize(
                    root / 'runs',
                    root / 'report',
                    models=('SmallWaferCNN',),
                    augmentations=('none',),
                    seeds=(1,),
                )
            self.assertEqual(success.read_bytes(), before)

    def test_invalid_artifact_is_not_published(self):
        """잘못된 산출물은 완료로 공개하지 않고 임시 폴더에 남긴다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            final = root / 'runs' / run_name('SmallWaferCNN', 'none', 1)

            with mock.patch(
                'src.wafer_dl.compare.measure_inference',
                return_value={'device': 'cpu', 'ms_per_sample': 0},
            ):
                failures = self._run(root)

            self.assertEqual(failures[0]['code'], 'COMPARE_RUN_INTEGRITY_FAILED')
            self.assertFalse(final.exists())
            self.assertTrue(final.with_name(final.name + '.tmp').is_dir())

    def test_code_change_during_training_is_not_published(self):
        """학습 중 코드가 바뀐 실행은 완료로 공개하지 않는다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            final = root / 'runs' / run_name('SmallWaferCNN', 'none', 1)

            with mock.patch(
                'src.wafer_dl.compare.code_fingerprint',
                side_effect=[other_fingerprint('c' * 64), other_fingerprint('d' * 64)],
            ):
                failures = self._run(root)

            self.assertEqual(failures[0]['code'], 'COMPARE_RUN_CODE_MISMATCH')
            self.assertFalse(final.exists())

    def test_comparison_identifies_wafer_mlp_structure(self):
        """산출물에 풀링 구조를 남겨 원본 해상도 MLP와 구분한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root, models=('WaferMLP',)), [])

            summarize(
                root / 'runs',
                root / 'report',
                models=('WaferMLP',),
                augmentations=('none',),
                seeds=(1,),
            )
            comparison = (root / 'report' / 'comparison.md').read_text(encoding='utf-8')

        self.assertIn('32×32 평균 풀링', comparison)
        self.assertIn('원본 해상도 입력 MLP 아님', comparison)

    def test_plan_counts_matrix_without_training(self):
        """계획은 학습 없이 전체 조합과 남은 실행을 계산한다."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            self.assertEqual(self._run(root), [])

            with mock.patch('src.wafer_dl.compare.train') as trained:
                complete, pending = plan_runs(
                    root / 'cache', root / 'runs', root / 'train.json', 'cpu'
                )

            trained.assert_not_called()
            self.assertEqual(len(complete) + len(pending), 18)

    def test_run_all_summarizes_both_conditions_and_resumes(self):
        """재실행은 완료 실행을 건너뛰고 집계는 선정을 보류한다."""
        models = ('SmallWaferCNN', 'MobileNetV3Small')
        augmentations = ('none', 'd4')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._prepare(root)
            matrix = {'seeds': (1, 2), 'models': models, 'augmentations': augmentations}

            self.assertEqual(self._run(root, **matrix), [])

            # 재실행은 완료된 실행을 다시 학습하지 않는다.
            with mock.patch('src.wafer_dl.compare.train') as repeated:
                self.assertEqual(self._run(root, **matrix), [])
                repeated.assert_not_called()

            result = summarize(root / 'runs', root / 'report', **matrix)
            comparison = (root / 'report' / 'comparison.md').read_text(encoding='utf-8')

            self.assertEqual(result['selection_status'], '선정 보류')
            self.assertIsNone(result['selected_model'])
            self.assertEqual(len(result['candidates']), 4)
            self.assertTrue((root / 'report' / 'runs.csv').is_file())
            self.assertIn('Test는 사용하지 않았습니다', comparison)
            self.assertIn('선정 보류', comparison)
            self.assertIn('Near-Full Recall', comparison)
            self.assertIn('추론 메모리', comparison)


class WaferCodeFingerprintTest(unittest.TestCase):
    def _copy_code(self, root):
        """실제 파일을 바꾸지 않도록 대상 코드를 임시 폴더에 복사한다."""
        for name in RUN_CODE_FILES:
            (root / name).write_bytes((PACKAGE_DIR / name).read_bytes())

    def test_target_code_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._copy_code(root)
            before = code_fingerprint(root)
            self.assertEqual(before, code_fingerprint(root))

            for name in RUN_CODE_FILES:
                with self.subTest(name=name):
                    original = (root / name).read_bytes()
                    (root / name).write_bytes(original + b'\n# changed\n')
                    self.assertNotEqual(code_fingerprint(root)['sha256'], before['sha256'])
                    (root / name).write_bytes(original)

    def test_unrelated_file_does_not_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._copy_code(root)
            before = code_fingerprint(root)

            (root / 'README.md').write_text('문서 변경', encoding='utf-8')
            (root / 'evaluate.py').write_text('# Test 전용', encoding='utf-8')

            self.assertEqual(code_fingerprint(root), before)

    def test_call_path_modules_are_all_covered(self):
        """학습 경로의 패키지 내부 import가 식별 대상에서 빠지지 않는다."""
        for name in ('compare.py', 'train.py', 'dataset.py', 'models.py', 'augment.py',
                     'metrics.py', 'transform.py'):
            source = (PACKAGE_DIR / name).read_text(encoding='utf-8')
            for line in source.splitlines():
                if line.startswith('from .'):
                    module = line.split()[1].removeprefix('.')
                    with self.subTest(source=name, module=module):
                        self.assertIn(module + '.py', RUN_CODE_FILES)


class WaferCompareSummaryTest(unittest.TestCase):
    def test_lighter_model_does_not_overtake_higher_score(self):
        """0.80±0.20과 0.70±0.01에서 표준편차로 순서를 뒤집지 않는다."""
        summary = [
            {'model_name': 'Light', 'macro_f1_mean': 0.70, 'macro_f1_std': 0.01,
             'parameter_count': 1},
            {'model_name': 'Big', 'macro_f1_mean': 0.80, 'macro_f1_std': 0.20,
             'parameter_count': 100},
        ]
        self.assertEqual(
            [item['model_name'] for item in rank_candidates(summary)],
            ['Big', 'Light'],
        )

    def test_summary_rejects_missing_runs_and_mixed_splits(self):
        row = {'model_name': 'A', 'augmentation': 'none', 'seed': 1,
               'split_protocol_id': 'x', 'macro_f1': 0.5,
               'parameter_count': 1, 'hit_epoch_limit': False}
        matrix = {'models': ('A',), 'augmentations': ('none',), 'seeds': (1, 2)}

        with self.assertRaisesRegex(ValidationError, 'COMPARE_RUNS_INCOMPLETE'):
            summarize_runs([row], **matrix)
        with self.assertRaisesRegex(ValidationError, 'COMPARE_SPLIT_MISMATCH'):
            summarize_runs([row, {**row, 'seed': 2, 'split_protocol_id': 'y'}], **matrix)


if __name__ == '__main__':
    unittest.main()

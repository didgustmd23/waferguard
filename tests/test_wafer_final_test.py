"""동결 후보 최종 Test 사전점검·평가·집계의 합성 테스트.

실제 데이터·checkpoint를 쓰지 않는다. 작은 모델과 임시 폴더의 합성 캐시만 사용하며
평가는 실제 evaluate 함수를 CPU로 호출하되 실패 주입용 호출 기록기로 감싼다.
"""

import csv
import json
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.wafer_dl import compare, evaluate as evaluate_module, final_test
from src.wafer_dl.cache import file_digest
from src.wafer_dl.evaluate import evaluate, weights_digest
from src.wafer_dl.metrics import classification_metrics
from src.wafer_dl.train import class_weights
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError


PROTOCOL = 'wm_split_' + 'a' * 64
TRAIN_COUNTS = [1000, 300, 40, 350, 700, 280, 12, 60, 90]
VALIDATION_COUNTS = [10] * 9
TEST_COUNTS = [2, 1, 1, 1, 1, 1, 1, 1, 1]
MAPPING = {str(index): name for index, name in enumerate(CLASS_NAMES)}
TRANSFORM = json.loads(Path('configs/wm811k/transform.json').read_text(encoding='utf-8'))
PIN_RUNS = len(final_test.MODELS) * len(final_test.SEEDS)
TRAINING_FLAGS = []


class TinyModel(nn.Module):
    """BatchNorm 버퍼까지 포함해 eval 모드·가중치 불변을 확인하는 합성 모델."""

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(2)
        self.head = nn.Linear(2, len(CLASS_NAMES))

    def forward(self, images):
        TRAINING_FLAGS.append(self.training)
        return self.head(self.norm(images).mean(dim=(2, 3)))


def tiny_model(name, num_classes=9):
    return TinyModel()


def json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def metrics(scale=10):
    return classification_metrics(torch.eye(9, dtype=torch.int64) * scale)


class FinalTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in ('final_test', 'compare', 'evaluate'):
            patcher = mock.patch(f'src.wafer_dl.{target}.build_model', tiny_model)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('validate_check', 'print'):
            patcher = mock.patch.object(final_test, name, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.overlaps = {'lots_in_multiple_splits': 0, 'map_hashes_in_multiple_splits': 0,
                         'groups_in_multiple_splits': 0}
        patcher = mock.patch.object(final_test, 'verify_split_metadata', self.fake_split)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.split_dir = self.root / 'split'
        self.split_dir.mkdir()
        (self.split_dir / 'split_manifest.jsonl').write_text('manifest\n', encoding='utf-8')
        self.manifest_sha256 = file_digest(self.split_dir / 'split_manifest.jsonl')
        self.canonical = self.root / 'canonical'
        self.canonical.mkdir()
        json_write(self.split_dir / 'split.json', {'canonical_root': str(self.canonical)})
        self.check_dir = self.root / 'check'
        self.check_dir.mkdir()
        self.train_cache = self.root / 'train_cache'
        json_write(self.train_cache / 'cache.json', {
            'status': 'cache_complete',
            'split_protocol_id': PROTOCOL,
            'source_split_manifest_sha256': self.manifest_sha256,
            'train_class_counts': TRAIN_COUNTS,
        })
        self.cache_sha256 = file_digest(self.train_cache / 'cache.json')
        self.test_samples = []
        for label, count in enumerate(TEST_COUNTS):
            self.test_samples += [(f'sample_{label}_{n}', label) for n in range(count)]
        self.test_cache = self.write_test_cache(self.root / 'test_cache')
        self.freeze_dir = self.root / 'freeze'
        self.write_freeze()
        self.output = self.root / 'final_test_v1'
        self.calls = []
        self.fail_on_call = None
        TRAINING_FLAGS.clear()

    def fake_split(self, canonical_root, split_dir, progress=None):
        return ({
            'split_protocol_id': PROTOCOL,
            'class_mapping': MAPPING,
            'overlap_check_passed': True,
            'split_counts': {'train': sum(TRAIN_COUNTS), 'validation': sum(VALIDATION_COUNTS),
                             'test': sum(TEST_COUNTS)},
            'class_counts': {'train': TRAIN_COUNTS, 'validation': VALIDATION_COUNTS,
                             'test': TEST_COUNTS},
        }, self.overlaps, self.test_samples)

    def write_test_cache(self, root):
        (root / 'test').mkdir(parents=True)
        shard = root / 'test' / 'shard_000000.npy'
        array = np.zeros((len(self.test_samples), 128, 128), dtype=np.uint8)
        array[:, 10:100, 10:100] = 1
        array[:, 40:50, 40:50] = 2
        with shard.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        with (root / 'test' / 'index.jsonl').open('w', encoding='utf-8') as stream:
            for offset, (sample, label) in enumerate(self.test_samples):
                stream.write(json.dumps({
                    'sample_id': sample, 'class_index': label,
                    'pattern_label': CLASS_NAMES[label], 'split': 'test',
                    'shard': 'test/shard_000000.npy', 'offset': offset,
                }) + '\n')
        json_write(root / 'cache.json', {
            'status': 'cache_complete',
            'cache_version': 'wm_input_cache_v1',
            'split_protocol_id': PROTOCOL,
            'source_split_manifest_sha256': self.manifest_sha256,
            'transform_config': TRANSFORM,
            'class_mapping': MAPPING,
            'test_maps_read': True,
            'visual_review_confirmed': True,
            'train_class_counts': TRAIN_COUNTS,
            'splits': {'test': {
                'rows': len(self.test_samples),
                'class_counts': TEST_COUNTS,
                'index': 'test/index.jsonl',
                'index_sha256': file_digest(root / 'test' / 'index.jsonl'),
                'shards': [{
                    'path': 'test/shard_000000.npy', 'rows': len(self.test_samples),
                    'bytes': shard.stat().st_size, 'sha256': file_digest(shard),
                }],
            }},
        })
        json_write(root / 'SUCCESS.json', {'status': 'cache_complete',
                                           'split_protocol_id': PROTOCOL})
        return root

    def write_run(self, model_name, seed, index):
        run_path = final_test.CANDIDATES[model_name]['run_path'].format(seed=seed)
        run_dir = self.root / run_path
        run_dir.mkdir(parents=True)
        config = {
            'augmentation': 'd4', 'class_weight_alpha': 0.75, 'scratch_weight_factor': 1.0,
            'training_version': 'wm_cnn_train_v1', 'batch_size': 64, 'epochs': 30,
            'learning_rate': 0.001, 'weight_decay': 0.0001, 'patience': 6, 'seed': seed,
            'class_weighted': True, 'max_class_weight': 10.0, 'selection_metric': 'macro_f1',
        }
        if final_test.CANDIDATES[model_name]['scheduled']:
            config['lr_scheduler'] = {'type': 'ReduceLROnPlateau'}
        contract = {
            'contract_version': compare.RUN_CONTRACT_VERSION,
            'model_name': model_name, 'seed': seed, 'device': 'cpu',
            'training_config': config,
            'split_protocol_id': PROTOCOL,
            'source_split_manifest_sha256': self.manifest_sha256,
            'cache_sha256': self.cache_sha256,
            'code_fingerprint': compare.code_fingerprint(),
        }
        torch.manual_seed(index)
        weights = dict(zip(CLASS_NAMES, class_weights(TRAIN_COUNTS, 10.0, 0.75, 1.0).tolist()))
        torch.save({
            'checkpoint_version': 'wm_cnn_checkpoint_v2',
            'model_name': model_name,
            'augmentation': 'd4',
            'model_config': {'num_classes': 9, 'input_channels': 2},
            'class_mapping': MAPPING,
            'split_protocol_id': PROTOCOL,
            'source_split_manifest_sha256': self.manifest_sha256,
            'transform_config': TRANSFORM,
            'selection_metric': 'macro_f1',
            'selected_epoch': 1,
            'training_config': config,
            'class_weights': {'weights': weights},
            'model_state_dict': dict(TinyModel().state_dict()),
        }, run_dir / 'best_model.pt')
        validation = metrics()
        json_write(run_dir / 'training_report.json', {
            'status': 'training_complete', 'test_split_used': False, 'device': 'cpu',
            'model_name': model_name, 'seed': seed, 'augmentation': 'd4',
            'split_protocol_id': PROTOCOL, 'selection_metric': 'macro_f1', 'selected_epoch': 1,
            'parameter_count': 1, 'model_description': 'tiny', 'max_epochs': 30,
            'training_seconds': 1.0, 'peak_cuda_memory_bytes': None,
            'history': [{'epoch': 1, 'train_loss': 0.1, 'validation': validation}],
        })
        json_write(run_dir / 'inference.json', {
            'device': 'cpu', 'ms_per_sample': 0.1, 'peak_cuda_memory_bytes': None,
        })
        json_write(run_dir / 'SUCCESS.json', {
            'status': compare.RUN_COMPLETE_STATUS,
            'contract': contract,
            'artifacts': {name: file_digest(run_dir / name) for name in compare.RUN_ARTIFACTS},
        })
        return {
            'seed': seed, 'run_path': run_path, 'selected_epoch': 1,
            'validation_macro_f1': validation['macro_f1'], 'contract': contract,
            'model_config': {'num_classes': 9, 'input_channels': 2},
            'transform_config': TRANSFORM, 'class_mapping': MAPPING,
            'artifacts_sha256': {name: file_digest(run_dir / name)
                                 for name in final_test.RUN_FILES},
        }

    def write_freeze(self):
        candidates = []
        index = 0
        for model_name in final_test.MODELS:
            runs = []
            for seed in final_test.SEEDS:
                runs.append(self.write_run(model_name, seed, index))
                index += 1
            scores = [run['validation_macro_f1'] for run in runs]
            candidates.append({
                'model_name': model_name,
                'role': final_test.CANDIDATES[model_name]['role'],
                'runs': runs,
                'validation_macro_f1_mean': statistics.fmean(scores),
                'validation_macro_f1_std': statistics.stdev(scores),
            })
        self.store_freeze({
            'freeze_version': 'wm_candidate_freeze_v1',
            'status': 'candidates_frozen_test_not_run',
            'candidates': candidates,
            'evaluation_plan': {
                'seeds': list(final_test.SEEDS), 'checkpoint_count': PIN_RUNS,
                'same_test_partition_for_all': True, 'fit_on_test': False,
                'test_augmentation': 'none', 'test_used_for_current_selection': False,
                'test_executed_by_this_freeze': False, 'no_best_test_seed_selection': True,
                'no_test_based_retuning': True,
            },
        })

    def store_freeze(self, freeze):
        self.freeze_dir.mkdir(exist_ok=True)
        json_write(self.freeze_dir / 'freeze.json', freeze)
        (self.freeze_dir / 'README.md').write_text('동결', encoding='utf-8')
        json_write(self.freeze_dir / 'SUCCESS.json', {
            'status': 'candidates_frozen_test_not_run',
            'artifacts_sha256': {name: file_digest(self.freeze_dir / name)
                                 for name in final_test.FREEZE_FILES},
        })

    def edit_freeze(self, change):
        """SUCCESS 해시까지 다시 맞춘 위조 명세도 실제 파일과 대조돼 거부되는지 본다."""
        freeze = json.loads((self.freeze_dir / 'freeze.json').read_text(encoding='utf-8'))
        change(freeze)
        self.store_freeze(freeze)

    def preflight(self, test_cache=True, pin=None):
        return final_test.preflight(
            self.freeze_dir, self.root, self.split_dir, self.check_dir, self.train_cache,
            'configs/wm811k/transform.json', 'configs/wm811k/cache.json', None,
            self.test_cache if test_cache else None, pin,
        )

    def recording_evaluator(self, *args):
        self.calls.append(args)
        if self.fail_on_call == len(self.calls):
            args[5](4, 10)
            raise RuntimeError('합성 평가 실패')
        return evaluate(*args)

    def run_session(self, resume=False, retry=None, approved=True):
        check, runs = self.preflight()
        return final_test.evaluate_session(
            check, runs, self.test_cache, self.output, 'cpu', approved,
            resume, retry, self.recording_evaluator,
        )

    def summarize(self):
        return final_test.summarize(self.freeze_dir, self.root, self.train_cache, self.output)

    def assert_rejected(self, code, function):
        with self.assertRaises(ValidationError) as caught:
            function()
        self.assertEqual(str(caught.exception), code)


class PreflightTest(FinalTestCase):
    def test_passes_without_opening_test_arrays(self):
        # 배열이 손상돼도 통과하면 preflight가 Test shard를 읽지 않는다는 뜻이다.
        (self.test_cache / 'test' / 'shard_000000.npy').write_bytes(b'broken')
        with mock.patch.object(np, 'load') as load, \
                mock.patch.object(evaluate_module, 'WaferDataset') as dataset:
            check, runs = self.preflight()
        load.assert_not_called()
        dataset.assert_not_called()
        self.assertEqual(check['status'], 'preflight_passed')
        self.assertIs(check['test_arrays_read'], False)
        self.assertIs(check['test_evaluation_approved'], False)
        self.assertEqual(check['past_test_overlap'], 'not_established')
        self.assertEqual(len(runs), PIN_RUNS)
        self.assertEqual(check['majority_class_index'], 0)
        self.assertFalse(self.output.exists())

    def test_works_before_test_cache_exists(self):
        check, _ = self.preflight(test_cache=False)
        self.assertEqual(check['test_cache_state'], 'not_checked')

    def test_rejects_changed_freeze_files(self):
        path = self.freeze_dir / 'freeze.json'
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        self.assert_rejected('FREEZE_HASH_MISMATCH', self.preflight)
        path.write_bytes(original)
        self.assert_rejected('FREEZE_PIN_MISMATCH', lambda: self.preflight(pin='0' * 64))

    def test_rejects_resigned_freeze_changes(self):
        def run(freeze):
            return freeze['candidates'][0]['runs'][0]

        def set_seed(freeze):
            run(freeze)['seed'] = 45

        def swap_model(freeze):
            freeze['candidates'].reverse()

        def set_path(freeze):
            run(freeze)['run_path'] = 'models/other/ResNet18_d4_seed42'

        def set_epoch(freeze):
            run(freeze)['selected_epoch'] = 2

        def set_contract(freeze, key, value):
            run(freeze)['contract'][key] = value

        def set_config(freeze):
            run(freeze)['contract']['training_config']['learning_rate'] = 0.01

        def set_transform(freeze):
            run(freeze)['transform_config'] = {**TRANSFORM, 'padding_value': 1}

        def set_mapping(freeze):
            run(freeze)['class_mapping'] = {**MAPPING, '0': 'Other'}

        def set_score(freeze):
            run(freeze)['validation_macro_f1'] = 0.5

        def set_plan(freeze):
            freeze['evaluation_plan']['test_used_for_current_selection'] = True

        cases = [
            ('FREEZE_CANDIDATE_MISMATCH', set_seed),
            ('FREEZE_CANDIDATE_MISMATCH', swap_model),
            ('FREEZE_PATH_MISMATCH', set_path),
            ('FREEZE_CHECKPOINT_MISMATCH', set_epoch),
            ('FREEZE_CONTRACT_MISMATCH', lambda item: set_contract(item, 'split_protocol_id', 'x')),
            ('FREEZE_CONTRACT_MISMATCH',
             lambda item: set_contract(item, 'source_split_manifest_sha256', '0' * 64)),
            ('FREEZE_CONTRACT_MISMATCH', set_config),
            ('CHECKPOINT_CACHE_CONTRACT_MISMATCH', set_transform),
            ('FREEZE_CHECKPOINT_MISMATCH', set_mapping),
            ('FREEZE_VALIDATION_MISMATCH', set_score),
            ('FREEZE_PLAN_MISMATCH', set_plan),
        ]
        original = (self.freeze_dir / 'freeze.json').read_bytes()
        for code, change in cases:
            with self.subTest(code=code, change=getattr(change, '__name__', 'contract')):
                self.edit_freeze(change)
                self.assert_rejected(code, self.preflight)
                self.store_freeze(json.loads(original))

    def test_rejects_replaced_checkpoint_and_duplicate_candidate(self):
        run_dir = self.root / final_test.CANDIDATES['ResNet18']['run_path'].format(seed=42)
        other = self.root / final_test.CANDIDATES['ResNet18']['run_path'].format(seed=43)
        shutil.copy(other / 'best_model.pt', run_dir / 'best_model.pt')
        self.assert_rejected('FREEZE_ARTIFACT_MISMATCH', self.preflight)

    def test_rejects_changed_split_transform_and_train_inputs(self):
        manifest = self.split_dir / 'split_manifest.jsonl'
        original = manifest.read_bytes()
        manifest.write_bytes(b'other\n')
        self.assert_rejected('SPLIT_MANIFEST_MISMATCH', self.preflight)
        manifest.write_bytes(original)

        self.overlaps['lots_in_multiple_splits'] = 1
        self.assert_rejected('SPLIT_OVERLAP_DETECTED', self.preflight)
        self.overlaps['lots_in_multiple_splits'] = 0

        with mock.patch.object(final_test, 'load_transform_config',
                               return_value={**TRANSFORM, 'padding_value': 1}):
            self.assert_rejected('TRANSFORM_CONFIG_MISMATCH', self.preflight)

        cache_file = self.train_cache / 'cache.json'
        record = json.loads(cache_file.read_text(encoding='utf-8'))
        record['note'] = 'changed'
        cache_file.write_text(json.dumps(record), encoding='utf-8')
        self.assert_rejected('TRAIN_CACHE_MISMATCH', self.preflight)

    def test_rejects_wrong_train_frequency(self):
        cache_file = self.train_cache / 'cache.json'
        record = json.loads(cache_file.read_text(encoding='utf-8'))
        record['train_class_counts'] = [2000, *TRAIN_COUNTS[1:]]
        cache_file.write_text(json.dumps(record), encoding='utf-8')
        self.assert_rejected('FREEZE_TRAIN_FREQUENCY_MISMATCH', self.preflight)

    def test_inference_code_change_is_rejected_training_code_is_reported(self):
        actual = compare.code_fingerprint()

        def changed(name):
            def fingerprint(*args):
                if args:
                    return actual
                return {'sha256': 'x', 'files': {**actual['files'], name: '0' * 64}}
            return fingerprint

        with mock.patch.object(final_test, 'code_fingerprint', changed('models.py')):
            self.assert_rejected('FREEZE_INFERENCE_CODE_MISMATCH', self.preflight)
        with mock.patch.object(final_test, 'code_fingerprint', changed('train.py')):
            check, _ = self.preflight()
        self.assertEqual(check['runs'][0]['training_only_code_differences'], ['train.py'])

    def test_rejects_mixed_or_incomplete_test_cache(self):
        report_file = self.test_cache / 'cache.json'
        original = report_file.read_text(encoding='utf-8')

        def rewrite(change):
            record = json.loads(original)
            change(record)
            report_file.write_text(json.dumps(record), encoding='utf-8')

        rewrite(lambda item: item['splits'].update(train={}))
        self.assert_rejected('TEST_CACHE_NOT_PREPARED', self.preflight)
        rewrite(lambda item: item.update(split_protocol_id='wm_split_other'))
        self.assert_rejected('TEST_CACHE_CONTRACT_MISMATCH', self.preflight)
        rewrite(lambda item: item['splits']['test'].update(rows=5))
        self.assert_rejected('TEST_CACHE_COUNT_MISMATCH', self.preflight)
        report_file.write_text(original, encoding='utf-8')
        self.test_samples[0] = ('other', 0)
        self.assert_rejected('TEST_CACHE_SAMPLE_MISMATCH', self.preflight)

    def test_rejects_train_cache_used_as_test_cache(self):
        self.test_cache = self.train_cache
        self.assert_rejected('TEST_CACHE_NOT_SEPARATE', self.preflight)

    def test_cross_split_overlap_counts_from_manifest_tables(self):
        database = sqlite3.connect(':memory:')
        self.addCleanup(database.close)
        database.execute('CREATE TABLE rows(position INTEGER, lot TEXT, hash TEXT)')
        database.execute('CREATE TABLE assignments(position INTEGER, split TEXT, gid TEXT)')
        database.executemany('INSERT INTO rows VALUES(?,?,?)', [
            (0, 'lot1', 'a'), (1, 'lot1', 'b'), (2, 'lot2', 'c'), (3, 'lot3', 'c'), (4, 'lot4', 'd'),
        ])
        database.executemany('INSERT INTO assignments VALUES(?,?,?)', [
            (0, 'train', 'g1'), (1, 'test', 'g2'), (2, 'train', 'g3'), (3, 'validation', 'g3'),
            (4, 'quarantined', None),
        ])
        self.assertEqual(final_test.cross_split_overlaps(database), {
            'lots_in_multiple_splits': 1,
            'map_hashes_in_multiple_splits': 1,
            'groups_in_multiple_splits': 1,
        })


class EvaluationTest(FinalTestCase):
    def test_requires_confirmation_before_any_test_call(self):
        self.assert_rejected('FINAL_TEST_EVALUATION_CONFIRMATION_REQUIRED',
                             lambda: self.run_session(approved=False))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.output.exists())

    def test_cli_without_confirmation_skips_preflight_and_evaluation(self):
        arguments = ['final_test', 'evaluate', '--test-cache-dir', str(self.test_cache),
                     '--expected-freeze-sha256', '0' * 64]
        with mock.patch('sys.argv', arguments), \
                mock.patch.object(final_test, 'preflight') as checker, \
                mock.patch.object(final_test, 'evaluate_session') as session, \
                mock.patch('builtins.print'):
            self.assertEqual(final_test.main(), 1)
        checker.assert_not_called()
        session.assert_not_called()

    def test_preflight_requires_test_cache_before_evaluation(self):
        check, runs = self.preflight(test_cache=False)
        self.assert_rejected('FINAL_TEST_PREFLIGHT_REQUIRED', lambda: final_test.evaluate_session(
            check, runs, self.test_cache, self.output, 'cpu', True,
            evaluator=self.recording_evaluator,
        ))
        self.assertEqual(self.calls, [])

    def test_six_frozen_checkpoints_share_one_test_cache(self):
        before = {path: file_digest(path) for path in self.root.glob('models/**/best_model.pt')}
        result = self.run_session()

        self.assertEqual(result['status'], 'final_test_runs_complete')
        self.assertEqual(len(self.calls), PIN_RUNS)
        self.assertEqual({str(call[0]) for call in self.calls}, {str(self.test_cache)})
        self.assertEqual(
            [Path(call[1]).parent.name for call in self.calls],
            [f'{model}_d4_seed{seed}' for model in final_test.MODELS
             for seed in final_test.SEEDS],
        )
        self.assertTrue(TRAINING_FLAGS)
        self.assertFalse(any(TRAINING_FLAGS))
        self.assertEqual(
            {path: file_digest(path) for path in self.root.glob('models/**/best_model.pt')},
            before,
        )

        for model_name in final_test.MODELS:
            for seed in final_test.SEEDS:
                folder = self.output / model_name / f'seed{seed}'
                report = json.loads((folder / 'test_evaluation.json').read_text(encoding='utf-8'))
                checkpoint = torch.load(
                    self.root / final_test.CANDIDATES[model_name]['run_path'].format(seed=seed) /
                    'best_model.pt', weights_only=True,
                )
                model = TinyModel()
                model.load_state_dict(checkpoint['model_state_dict'])
                self.assertEqual(report['weights_sha256'], weights_digest(model))
                self.assertIs(report['weights_unchanged'], True)
                self.assertIs(report['model_training_mode'], False)
                self.assertEqual(report['test_augmentation'], 'none')
                self.assertEqual(report['class_names'], list(CLASS_NAMES))
                self.assertEqual(report['batch_size'], 64)
                self.assertEqual(report['predicted_rows'], len(self.test_samples))
                self.assertEqual(report['majority_baseline_class_name'], 'None')
                self.assertTrue((folder / final_test.RUN_RECORD).is_file())
        self.assertIn('전체 Test 접근이 1회라는 뜻이 아닙니다', final_test.SCOPE_NOTE)

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        (self.output / 'other.txt').write_text('보존', encoding='utf-8')
        self.assert_rejected('FINAL_TEST_OUTPUT_EXISTS', self.run_session)
        self.assertEqual(self.calls, [])
        self.assertEqual((self.output / 'other.txt').read_text(encoding='utf-8'), '보존')

    def test_second_session_is_rejected_and_resume_does_not_reevaluate(self):
        self.run_session()
        marker = self.output / 'ResNet18' / 'seed42' / 'test_evaluation.json'
        digest = file_digest(marker)
        self.assert_rejected('FINAL_TEST_OUTPUT_EXISTS', self.run_session)
        self.assertEqual(len(self.calls), PIN_RUNS)
        result = self.run_session(resume=True)
        self.assertEqual(result['status'], 'final_test_runs_complete')
        self.assertEqual(len(self.calls), PIN_RUNS)
        self.assertEqual(file_digest(marker), digest)

    def test_resume_requires_existing_session(self):
        self.assert_rejected('FINAL_TEST_SESSION_NOT_FOUND', lambda: self.run_session(resume=True))

    def test_failure_stops_keeps_success_and_requires_explicit_retry(self):
        self.fail_on_call = 3
        result = self.run_session()

        self.assertEqual(result['status'], 'final_test_stopped')
        self.assertEqual(result['stopped_at'], 'ResNet18:44')
        self.assertEqual(len(self.calls), 3)
        states = {item['run']: item for item in result['runs']}
        self.assertEqual(states['ResNet18:42']['status'], 'complete')
        self.assertEqual(states['ResNet18:44']['status'], 'failed')
        self.assertEqual(states['ResNet18:44']['rows_read_last_known'], 4)
        self.assertEqual(states['ResNet18:44']['last_error'], 'RuntimeError')
        self.assertEqual(states['SmallWaferCNN:42']['status'], 'pending')
        self.assertFalse((self.output / 'ResNet18' / 'seed44').exists())
        kept = file_digest(self.output / 'ResNet18' / 'seed42' / 'test_evaluation.json')

        self.assert_rejected('FINAL_TEST_RETRY_DECISION_REQUIRED',
                             lambda: self.run_session(resume=True))
        self.assertEqual(len(self.calls), 3)

        self.fail_on_call = None
        result = self.run_session(resume=True, retry=['ResNet18:44'])
        self.assertEqual(result['status'], 'final_test_runs_complete')
        self.assertEqual(len(self.calls), PIN_RUNS + 1)
        self.assertEqual(
            sum(Path(call[1]).parent.name == 'ResNet18_d4_seed42' for call in self.calls), 1
        )
        self.assertEqual(
            file_digest(self.output / 'ResNet18' / 'seed42' / 'test_evaluation.json'), kept
        )

    def test_interrupted_attempt_is_not_retried_automatically(self):
        def interrupt(*args):
            raise KeyboardInterrupt

        check, runs = self.preflight()
        with self.assertRaises(KeyboardInterrupt):
            final_test.evaluate_session(check, runs, self.test_cache, self.output, 'cpu', True,
                                        evaluator=interrupt)
        final_test._append_attempt(self.output / 'attempts.jsonl', run='ResNet18:43',
                                   event='started', rows_read=0)
        status = final_test.read_status(self.freeze_dir, self.root, self.train_cache, self.output)
        states = {item['run']: item['status'] for item in status['runs']}
        self.assertEqual(states['ResNet18:42'], 'failed')
        self.assertEqual(states['ResNet18:43'], 'interrupted')
        self.assertIs(status['test_arrays_read'], False)
        self.assertEqual(status['resume_plan']['requires_explicit_retry'],
                         ['ResNet18:42', 'ResNet18:43'])
        self.assertEqual(status['resume_plan']['flags'],
                         ['--resume', '--retry ResNet18:42', '--retry ResNet18:43'])

    def test_retry_rejects_complete_unknown_and_never_failed_runs(self):
        self.fail_on_call = 2
        self.run_session()
        count = len(self.calls)
        for code, key in (('FINAL_TEST_ALREADY_COMPLETE', 'ResNet18:42'),
                          ('FINAL_TEST_RETRY_UNKNOWN_RUN', 'ResNet18:45'),
                          ('FINAL_TEST_RETRY_NOT_FAILED', 'SmallWaferCNN:42')):
            with self.subTest(code=code):
                self.assert_rejected(code, lambda: self.run_session(resume=True, retry=[key]))
        self.assertEqual(len(self.calls), count)

    def test_resume_rejects_changed_session_binding(self):
        self.fail_on_call = 2
        self.run_session()
        self.fail_on_call = None
        session_file = self.output / 'session.json'
        session = json.loads(session_file.read_text(encoding='utf-8'))
        session['test_cache']['test_samples_sha256'] = '0' * 64
        session_file.write_text(json.dumps(session), encoding='utf-8')
        self.assert_rejected('FINAL_TEST_SESSION_MISMATCH',
                             lambda: self.run_session(resume=True, retry=['ResNet18:43']))

    def test_code_change_during_session_stops_before_next_test_call(self):
        self.fail_on_call = None
        actual = final_test.code_fingerprint
        seen = {'count': 0}

        def changing(*args):
            result = actual(*args)
            if args:
                seen['count'] += 1
                if seen['count'] > 1:
                    return {**result, 'sha256': 'changed'}
            return result

        check, runs = self.preflight()
        with mock.patch.object(final_test, 'code_fingerprint', changing):
            self.assert_rejected('FINAL_TEST_SESSION_MISMATCH', lambda: final_test.evaluate_session(
                check, runs, self.test_cache, self.output, 'cpu', True,
                evaluator=self.recording_evaluator,
            ))
        self.assertEqual(self.calls, [])

    def test_mismatched_evaluation_result_is_not_published(self):
        def wrong_model(*args):
            report = evaluate(*args)
            path = Path(args[2]) / 'test_evaluation.json'
            path.write_text(json.dumps({**report, 'selected_epoch': 9}), encoding='utf-8')

        check, runs = self.preflight()
        result = final_test.evaluate_session(check, runs, self.test_cache, self.output, 'cpu',
                                             True, evaluator=wrong_model)
        self.assertEqual(result['status'], 'final_test_stopped')
        self.assertFalse((self.output / 'ResNet18' / 'seed42').exists())
        self.assertTrue((self.output / 'ResNet18' / 'seed42.incomplete_1').is_dir())


class SummarizeTest(FinalTestCase):
    def setUp(self):
        super().setUp()
        self.run_session()

    def rewrite_run(self, model_name, seed, change):
        """완료 기록 해시까지 다시 맞춘 변조도 내용 대조로 거부되는지 본다."""
        folder = self.output / model_name / f'seed{seed}'
        path = folder / 'test_evaluation.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        change(report)
        path.write_text(json.dumps(report), encoding='utf-8')
        record = json.loads((folder / final_test.RUN_RECORD).read_text(encoding='utf-8'))
        record['artifacts']['test_evaluation.json'] = file_digest(path)
        (folder / final_test.RUN_RECORD).write_text(json.dumps(record), encoding='utf-8')

    def test_summary_reports_every_seed_without_selection(self):
        result = self.summarize()
        self.assertEqual(result['checkpoint_count'], PIN_RUNS)
        report = self.output / final_test.COMPARISON_DIR
        for name in ('comparison.md', 'comparison.json', 'runs.csv', 'summary.csv',
                     'per_class.csv', 'seed_differences.csv', 'confusion_matrices.csv',
                     'SUCCESS.json'):
            self.assertTrue((report / name).is_file(), name)

        with (report / 'runs.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), PIN_RUNS)
        scores = {}
        for row in rows:
            scores.setdefault(row['model_name'], []).append(float(row['macro_f1']))
        with (report / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            summary = {row['model_name']: row for row in csv.DictReader(stream)
                       if row['scope'] == 'all'}
        for model_name, values in scores.items():
            self.assertAlmostEqual(float(summary[model_name]['macro_f1_mean']),
                                   statistics.fmean(values))
            self.assertAlmostEqual(float(summary[model_name]['macro_f1_std']),
                                   statistics.stdev(values))

        with (report / 'seed_differences.csv').open(encoding='utf-8-sig', newline='') as stream:
            differences = list(csv.DictReader(stream))
        self.assertEqual([int(row['seed']) for row in differences], list(final_test.SEEDS))
        self.assertAlmostEqual(float(differences[0]['macro_f1']),
                               scores['ResNet18'][0] - scores['SmallWaferCNN'][0])

        with (report / 'confusion_matrices.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), PIN_RUNS * 81)
        with (report / 'per_class.csv').open(encoding='utf-8-sig', newline='') as stream:
            first = next(csv.DictReader(stream))
        self.assertIn('false_positive', first)
        self.assertIn('false_negative', first)

        text = (report / 'comparison.md').read_text(encoding='utf-8')
        self.assertIn('통계적 우월성으로 해석하지 않습니다', text)
        self.assertIn('구조만의 효과가 아닙니다', text)
        self.assertIn('표본 겹침은 미확인', text)
        details = json.loads((report / 'comparison.json').read_text(encoding='utf-8'))
        self.assertIs(details['selection_changed_by_test'], False)
        self.assertNotIn('best_seed', details)

    def test_existing_comparison_is_not_overwritten(self):
        self.summarize()
        self.assert_rejected('FINAL_TEST_REPORT_ALREADY_EXISTS', self.summarize)

    def test_rejects_missing_run(self):
        shutil.move(str(self.output / 'SmallWaferCNN' / 'seed43'),
                    str(self.output / 'SmallWaferCNN' / 'seed43.incomplete_1'))
        self.assert_rejected('FINAL_TEST_RUN_MISSING', self.summarize)

    def test_rejects_extra_seed_and_foreign_model(self):
        extra = self.output / 'ResNet18' / 'seed45'
        extra.mkdir()
        self.assert_rejected('FINAL_TEST_EXTRA_RUN', self.summarize)
        extra.rmdir()
        foreign = self.output / 'MobileNetV3Small'
        foreign.mkdir()
        self.assert_rejected('FINAL_TEST_EXTRA_RUN', self.summarize)

    def test_rejects_copied_duplicate_checkpoint_result(self):
        target = self.output / 'ResNet18' / 'seed43'
        shutil.rmtree(target)
        shutil.copytree(self.output / 'ResNet18' / 'seed42', target)
        self.assert_rejected('FINAL_TEST_RUN_INCOMPLETE', self.summarize)

    def test_rejects_changed_result_files(self):
        path = self.output / 'ResNet18' / 'seed42' / 'test_evaluation.json'
        path.write_bytes(path.read_bytes() + b' ')
        self.assert_rejected('FINAL_TEST_RUN_INTEGRITY_FAILED', self.summarize)

    def test_rejects_different_test_or_checkpoint_mixture(self):
        cases = [
            ('FINAL_TEST_MIXED_TEST', lambda item: item.update(test_samples_sha256='0' * 64)),
            ('FINAL_TEST_MIXED_TEST', lambda item: item.update(split_protocol_id='other')),
            ('FINAL_TEST_MIXED_TEST', lambda item: item.update(
                transform_config={**TRANSFORM, 'padding_value': 1})),
            ('FINAL_TEST_MIXED_TEST', lambda item: item.update(class_names=['x'] * 9)),
            ('FINAL_TEST_RUN_MISMATCH', lambda item: item.update(checkpoint_sha256='0' * 64)),
            ('FINAL_TEST_RUN_MISMATCH', lambda item: item.update(model_name='ResNet18')),
            ('FINAL_TEST_RUN_CONTRACT_MISMATCH', lambda item: item.update(test_augmentation='d4')),
        ]
        folder = self.output / 'SmallWaferCNN' / 'seed44'
        shutil.copytree(folder, self.root / 'backup')
        for code, change in cases:
            with self.subTest(code=code):
                self.rewrite_run('SmallWaferCNN', 44, change)
                self.assert_rejected(code, self.summarize)
                shutil.rmtree(folder)
                shutil.copytree(self.root / 'backup', folder)

    def test_rejects_metric_not_matching_confusion_matrix(self):
        def change(item):
            item['frozen_model']['macro_f1'] = 0.999

        self.rewrite_run('ResNet18', 42, change)
        self.assert_rejected('FINAL_TEST_METRIC_MISMATCH', self.summarize)

    def test_rejects_changed_freeze_after_session(self):
        freeze = json.loads((self.freeze_dir / 'freeze.json').read_text(encoding='utf-8'))
        freeze['note'] = 'changed'
        self.store_freeze(freeze)
        self.assert_rejected('FREEZE_PIN_MISMATCH', self.summarize)


if __name__ == '__main__':
    unittest.main()

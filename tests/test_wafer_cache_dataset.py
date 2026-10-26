"""합성 canonical에서 캐시·Tensor·DataLoader 계약을 검증한다."""

import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.wafer_dl import check_transform
from src.wafer_dl.cache import build_cache, file_digest
from src.wafer_dl.dataset import WaferDataset, check_first_batch
from src.wafer_dl.split import run_split
from src.wafer_dl.transform import load_transform_config, make_channels, transform_map
from src.wafer_dl.validate_canonical import CLASS_NAMES, ValidationError
from tests import test_wafer_split as fixtures


# ==========================================
# 캐시 저장과 Dataset의 합성 통합 검증
# - 실제 WM-811K·원본 pickle·GPU·모델 학습은 사용하지 않음
# - 작은 shard로 경계를 자주 넘어 색인과 mmap 읽기를 확인
# - 검사 통과 보고서는 합성 분할 집계로 구성한 테스트 대역
# ==========================================
class CacheDatasetTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WaferSplitTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.create_bundle()
        self.base = self.fixture.base
        self.split = self.base / 'split'
        run_split(self.fixture.root, self.split, self.fixture.config_path, approved=True)
        report = json.loads((self.split / 'split.json').read_text(encoding='utf-8'))
        self.transform_path = Path(__file__).resolve().parents[1] / 'configs/wm811k/transform.json'
        self.transform = load_transform_config(self.transform_path)
        self.check = self.base / 'check'
        self.check.mkdir()
        (self.check / 'SUCCESS.json').write_text(json.dumps({
            'status': 'transform_check_complete', 'review_required': False}), encoding='utf-8')
        (self.check / 'transform_check.json').write_text(json.dumps({
            'status': 'transform_check_passed', 'config': self.transform,
            'split_protocol_id': report['split_protocol_id'], 'test_maps_read': False,
            'rows_checked': sum(report['split_counts'][n] for n in ('train', 'validation')),
            'classes': [{'split': name, 'class': label, 'rows': report['class_counts'][name][i],
                         'failed_region_lost': 0, 'valid_region_lost': 0}
                        for name in ('train', 'validation') for i, label in enumerate(CLASS_NAMES)]}), encoding='utf-8')
        self.cache_config = self.base / 'cache_config.json'
        self.cache_config.write_text(json.dumps({'cache_version': 'wm_input_cache_v1',
            'shard_max_rows': 5, 'buffer_max_bytes': 1_048_576}), encoding='utf-8')
        self.output = self.base / 'cache'

    def build(self, **kwargs):
        return build_cache(self.split, self.check, self.output,
                           self.transform_path, self.cache_config, **kwargs)

    def test_cache_roundtrip_and_selected_reads(self):
        """개발 구간만 읽고 uint8 저장·색인·2채널을 원본 변환과 비교한다."""
        records = [json.loads(line) for line in (self.split / 'split_manifest.jsonl').read_text().splitlines()]
        allowed = {r['map_ref']['key'] + '.npy' for r in records if r['split'] in ('train', 'validation')}
        numeric_read = check_transform.read_numeric_member
        originals = {}

        def guarded_read(archive, member):
            self.assertIn(member.filename, allowed)
            array = numeric_read(archive, member)
            originals[member.filename[:-4]] = array.copy()
            return array

        before = (self.fixture.root / 'manifest.jsonl').read_bytes()
        with patch.object(check_transform, 'read_numeric_member', side_effect=guarded_read):
            self.assertEqual(self.build(approved=True)['status'], 'cache_complete')
        self.assertFalse((self.output / 'test').exists())
        self.assertEqual(before, (self.fixture.root / 'manifest.jsonl').read_bytes())
        self.assertEqual(set(originals), {name[:-4] for name in allowed})
        by_id = {r['sample_id']: r for r in records}
        for split in ('train', 'validation'):
            dataset = WaferDataset(self.output, split)
            self.addCleanup(dataset.close)
            cache_report = json.loads((self.output / 'cache.json').read_text(encoding='utf-8'))
            self.assertEqual(
                dataset.source_split_manifest_sha256,
                cache_report['source_split_manifest_sha256'],
            )
            for index in range(len(dataset)):
                sample = dataset[index]
                row = by_id[sample['sample_id']]
                transformed, _ = transform_map(originals[row['map_ref']['key']], self.transform)
                np.testing.assert_array_equal(sample['image'].numpy(), make_channels(transformed))
                self.assertEqual(sample['target'].item(), row['class_index'])
                self.assertEqual(sample['image'].dtype, torch.float32)
            self.assertLessEqual(len(dataset._maps), 2)
            self.assertEqual(check_first_batch(self.output, split, 8)['status'], 'dataset_batch_passed')

    def test_approval_and_existing_output(self):
        """시각 검토 미승인·Test 미승인·기존 출력 덮어쓰기를 거부한다."""
        with self.assertRaisesRegex(ValidationError, 'VISUAL_REVIEW_APPROVAL_REQUIRED'):
            self.build()
        with self.assertRaisesRegex(ValidationError, 'TEST_PREPARATION_APPROVAL_REQUIRED'):
            self.build(approved=True, splits=('test',))
        self.build(approved=True)
        with self.assertRaisesRegex(ValidationError, 'INVALID_CACHE_OUTPUT_DIRECTORY'):
            self.build(approved=True)

    def test_contract_mismatch_blocks_cache(self):
        """검사 설정이 다르면 완료 표식을 만들지 않는다."""
        path = self.check / 'transform_check.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        report['config']['target_size'] = [64, 64]
        path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValidationError, 'TRANSFORM_CHECK_CONTRACT_MISMATCH'):
            self.build(approved=True)
        self.assertFalse((self.output / 'SUCCESS.json').exists())

    def test_test_requires_explicit_preparation_and_read_approval(self):
        """Test 전용 준비와 Dataset 접근은 각각 명시적 승인 없이는 불가하다."""
        self.build(approved=True, splits=('test',), allow_test=True)
        self.assertFalse((self.output / 'train').exists())
        with self.assertRaisesRegex(ValidationError, 'TEST_DATASET_APPROVAL_REQUIRED'):
            WaferDataset(self.output, 'test')
        self.assertTrue(check_first_batch(self.output, 'test', 4, allow_test=True)['test_maps_read'])

    def test_shard_tampering_and_incomplete_cache(self):
        """파일 내용 손상과 완료 표식 없는 캐시를 거부한다."""
        self.build(approved=True)
        shard = next((self.output / 'train').glob('*.npy'))
        with shard.open('r+b') as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(ValidationError, 'CACHE_SHARD_INTEGRITY_FAILED'):
            WaferDataset(self.output)
        (self.output / 'SUCCESS.json').unlink()
        with self.assertRaisesRegex(ValidationError, 'MISSING_SUCCESS_FILE'):
            WaferDataset(self.output)

    def test_index_duplicates_are_rejected(self):
        """해시를 갱신해도 동일 위치를 중복 참조하는 잘못된 색인은 거부한다."""
        self.build(approved=True)
        path = self.output / 'train/index.jsonl'
        lines = path.read_text().splitlines()
        row = json.loads(lines[1])
        row['offset'] = 0
        lines[1] = json.dumps(row)
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        report_path = self.output / 'cache.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['splits']['train']['index_sha256'] = file_digest(path)
        report_path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValidationError, 'INVALID_CACHE_INDEX_ROW'):
            WaferDataset(self.output)

    def test_worker_state_and_independent_tensor(self):
        """열린 핸들을 worker 상태에서 제외하고 Tensor 수정이 캐시에 전파되지 않는다."""
        self.build(approved=True)
        dataset = WaferDataset(self.output)
        self.addCleanup(dataset.close)
        expected = dataset[0]['image'].clone()
        dataset[0]['image'].zero_()
        self.assertTrue(torch.equal(dataset[0]['image'], expected))
        # 신뢰한 자체 Dataset 객체의 worker 전달만 검사한다. 외부 pickle은 읽지 않는다.
        worker = pickle.loads(pickle.dumps(dataset))
        self.addCleanup(worker.close)
        self.assertFalse(worker._maps)
        self.assertTrue(torch.equal(worker[0]['image'], expected))
        batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
        self.assertEqual(tuple(batch['image'].shape), (4, 2, 128, 128))
        self.assertEqual(batch['target'].dtype, torch.int64)


if __name__ == '__main__':
    unittest.main()

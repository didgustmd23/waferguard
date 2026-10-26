"""완료된 숫자 캐시를 읽는 PyTorch Dataset과 첫 배치 계약 검사."""

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cache import file_digest
from .transform import make_channels
from .validate_canonical import CLASS_NAMES, decode_json, read_json, require, validation_stage


# ==========================================
# 선택한 분할의 캐시 색인 검증
# - 상대 경로·해시·행 수·클래스와 shard 내 위치를 확인
# - 다른 구간의 배열은 열지 않으며 Test는 별도 명시적 승인 필요
# - 해시는 손상·묶음 혼용 확인용이지 외부 파일의 안전성 인증이 아님
# ==========================================
class WaferDataset(Dataset):
    def __init__(self, cache_dir, split='train', allow_test=False):
        """선택한 구간의 캐시와 원본 분할 manifest 지문을 검증한다."""
        require(split in ('train', 'validation', 'test'), 'INVALID_DATASET_SPLIT')
        require(split != 'test' or allow_test is True, 'TEST_DATASET_APPROVAL_REQUIRED')
        self.root = Path(cache_dir).resolve(strict=True)
        report = read_json(self.root / 'cache.json')
        completion = read_json(self.root / 'SUCCESS.json')
        require(report.get('status') == completion.get('status') == 'cache_complete' and
                report.get('cache_version') == 'wm_input_cache_v1' and
                type(report.get('split_protocol_id')) is str and
                report['split_protocol_id'] == completion.get('split_protocol_id'), 'CACHE_NOT_COMPLETE')
        require(report.get('class_mapping') == {str(i): name for i, name in enumerate(CLASS_NAMES)},
                'INVALID_CACHE_CLASS_MAPPING')
        source_split_hash = report.get('source_split_manifest_sha256')
        require(type(source_split_hash) is str and
                re.fullmatch(r'[0-9a-f]{64}', source_split_hash) is not None,
                'INVALID_CACHE_SPLIT_MANIFEST_HASH')
        self.source_split_manifest_sha256 = source_split_hash
        config = report['transform_config']
        size = config.get('target_size')
        require(type(size) is list and len(size) == 2 and
                all(type(n) is int and 1 <= n <= 1024 for n in size) and
                config.get('channel_order') == ['defect', 'valid'] and
                config.get('transform_version') == 'wm_resize_v1' and
                config.get('resize_backend') == 'numpy_half_pixel_nearest_v1', 'INVALID_CACHE_TRANSFORM')
        self.transform_config = config
        self.size = tuple(size)
        self.split = split
        self.protocol = report['split_protocol_id']
        require(split in report['splits'], 'REQUESTED_SPLIT_NOT_CACHED')
        info = report['splits'][split]
        require(type(info['rows']) is int and 0 < info['rows'] <= 2_000_000 and
                type(info['shards']) is list and 0 < len(info['shards']) <= info['rows'], 'INVALID_CACHE_COUNTS')
        self.shards = {}
        # 파일 내용 해시까지 검사한다. 선택한 구간의 전체 파일을 순차 읽지만
        # 전체 배열을 RAM에 적재하지는 않는다. 다른 분할의 파일은 읽지 않는다.
        for number, shard in enumerate(info['shards']):
            expected = f'{split}/shard_{number:06d}.npy'
            require(shard['path'] == expected and type(shard['rows']) is int and
                    1 <= shard['rows'] <= 4096, 'INVALID_CACHE_SHARD')
            path = self._path(expected)
            expected_bytes = shard['rows'] * int(np.prod(self.size))
            require(expected_bytes <= 268_435_456 and type(shard['bytes']) is int and
                    expected_bytes < shard['bytes'] <= expected_bytes + 10_128 and
                    path.stat().st_size == shard['bytes'] and
                    file_digest(path) == shard['sha256'], 'CACHE_SHARD_INTEGRITY_FAILED')
            array = self._load(path, shard['rows'])
            array._mmap.close()
            self.shards[expected] = shard
        self.records = self._read_index(info)
        self.class_counts = tuple(info['class_counts'])
        # mmap 핸들은 프로세스별로 열며 최대 두 파일만 유지한다.
        self._maps = OrderedDict()
        self._pid = os.getpid()

    def _path(self, relative):
        require(type(relative) is str and not Path(relative).is_absolute(), 'INVALID_CACHE_PATH')
        path = (self.root / relative).resolve(strict=True)
        require(path.is_relative_to(self.root) and path.is_file(), 'INVALID_CACHE_PATH')
        return path

    def _load(self, path, rows):
        # object 배열의 pickle 복원은 허용하지 않고 숫자 전용 NPY만 mmap한다.
        array = np.load(path, mmap_mode='r', allow_pickle=False, max_header_size=10_000)
        if not (isinstance(array, np.memmap) and array.dtype == np.uint8 and
                array.shape == (rows, *self.size) and array.flags.c_contiguous):
            if isinstance(array, np.memmap):
                array._mmap.close()
            require(False, 'INVALID_CACHE_ARRAY')
        return array

    def _read_index(self, info):
        require(info['index'] == f'{self.split}/index.jsonl', 'INVALID_CACHE_INDEX')
        path = self._path(info['index'])
        require(file_digest(path) == info['index_sha256'], 'CACHE_INDEX_INTEGRITY_FAILED')
        records, seen, counts = [], set(), [0] * 9
        # writer가 기록한 순번 그대로인지 확인하여 중복 위치·누락을 거부한다.
        positions = ((name, offset) for name, shard in self.shards.items() for offset in range(shard['rows']))
        with path.open('rb') as stream:
            while True:
                line = stream.readline(8193)
                if not line:
                    break
                require(len(line) <= 8192 and len(records) < info['rows'], 'INVALID_CACHE_INDEX')
                row = decode_json(line)
                label, sample = row.get('class_index'), row.get('sample_id')
                require(type(label) is int and 0 <= label < 9 and
                        type(sample) is str and 0 < len(sample) <= 256 and sample not in seen and
                        row.get('pattern_label') == CLASS_NAMES[label] and row.get('split') == self.split and
                        type(row.get('offset')) is int and
                        (row.get('shard'), row['offset']) == next(positions, None), 'INVALID_CACHE_INDEX_ROW')
                records.append((row['shard'], row['offset'], label, sample))
                seen.add(sample)
                counts[label] += 1
        require(len(records) == info['rows'] and next(positions, None) is None and
                counts == info['class_counts'], 'CACHE_INDEX_COUNT_MISMATCH')
        return records

    def __len__(self):
        return len(self.records)

    # ==========================================
    # 한 표본의 모델 입력 구성
    # - 저장된 uint8 맵 → defect/valid 두 개의 float32 채널
    # - 원본 캐시와 별도 메모리이므로 학습 쪽 연산이 저장 맵을 변경하지 않음
    # - 라벨·표본 ID는 이미지에 섞지 않으며 ID는 추적에만 사용
    # ==========================================
    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise TypeError('표본 인덱스는 정수여야 합니다.')
        name, offset, label, sample = self.records[index]
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if name not in self._maps:
            if len(self._maps) >= 2:
                _, old = self._maps.popitem(last=False)
                old._mmap.close()
            self._maps[name] = self._load(self._path(name), self.shards[name]['rows'])
        self._maps.move_to_end(name)
        # memmap 부분 배열을 ndarray 뷰로 전달한다. make_channels가 새 배열을 만들므로
        # 저장 파일을 복사해서 RAM에 올리거나 캐시를 수정할 필요가 없다.
        array = np.asarray(self._maps[name][offset])
        require(np.isin(array, (0, 1, 2)).all() and np.any(array > 0), 'INVALID_CACHE_MAP_VALUES')
        return {'image': torch.from_numpy(make_channels(array)),
                'target': torch.tensor(label, dtype=torch.int64), 'sample_id': sample}

    def close(self):
        for array in getattr(self, '_maps', {}).values():
            array._mmap.close()
        if hasattr(self, '_maps'):
            self._maps.clear()

    def __getstate__(self):
        # Windows DataLoader worker로 객체를 전달할 때 열린 mmap은 전달하지 않는다.
        state = self.__dict__.copy()
        state['_maps'] = OrderedDict()
        state['_pid'] = None
        return state


def check_first_batch(cache_dir, split='train', batch_size=32, allow_test=False):
    require(type(batch_size) is int and 1 <= batch_size <= 4096, 'INVALID_BATCH_SIZE')
    dataset = WaferDataset(cache_dir, split, allow_test)
    try:
        # 첫 계약 검사는 CPU·단일 프로세스로 고정한다. 속도 튜닝·모델 학습은 별도다.
        batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
        image, target = batch['image'], batch['target']
        require(image.shape == (min(batch_size, len(dataset)), 2, *dataset.size) and
                image.dtype == torch.float32 and target.dtype == torch.int64 and
                target.shape == (image.shape[0],) and bool(((target >= 0) & (target < 9)).all()) and
                bool(((image == 0) | (image == 1)).all()) and
                bool((image[:, 0] <= image[:, 1]).all()), 'DATASET_BATCH_CONTRACT_FAILED')
        return {'status': 'dataset_batch_passed', 'split': split, 'image_shape': list(image.shape),
                'image_dtype': str(image.dtype), 'target_dtype': str(target.dtype),
                'split_protocol_id': dataset.protocol, 'test_maps_read': split == 'test',
                'model_trained': False, 'is_performance_evaluation': False,
                'scope': 'selected_split_integrity_and_first_batch_only'}
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description='WM-811K Dataset 첫 배치 계약 검사 (학습 없음)')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--confirm-test-preparation', action='store_true')
    args = parser.parse_args()
    try:
        with validation_stage('dataset'):
            result = check_first_batch(args.cache_dir, args.split, args.batch_size, args.confirm_test_preparation)
    except ValueError as error:
        print('Dataset 검사 실패 코드: ' + str(error))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('선택 구간 정합성과 첫 배치만 확인했습니다. 모델 학습·성능 검증 결과가 아닙니다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""첫 배치 결과의 요청 결합·최소 구조만 검사한다. Tensor나 모델 파일은 받지 않는다."""

from pathlib import Path

from .artifacts import _is_link, _root
from .wm_canonical_contract import digest, file_hash
from .wm_canonical_results import _read_small
from .worker_contract import _decode
from .wm_dataset_contract import POLICY_VERSION


FILES = {'dataset_check.json', 'dataset_check.md', 'SUCCESS.json'}


def result_files(directory, maximum):
    root = Path(directory)
    expected = FILES | ({'result.json'} if (root / 'result.json').exists() else set())

    if _is_link(root) or not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise ValueError('Dataset 결과 파일 목록이 올바르지 않습니다.')

    records, total = {}, 0

    for name in sorted(expected):
        path = root / name

        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError('Dataset 결과에는 일반 파일만 허용합니다.')

        records[name] = file_hash(path, maximum - total)
        total += records[name]['size']

    return records


def inspect_dataset(directory, request, contract, evidence):
    """출력 범위를 검증한다. CPU 실행·Tensor 연산의 독립 재계산을 뜻하지 않는다."""
    root = _root(directory)
    files = result_files(root, contract.max_output_bytes)
    result = _decode(_read_small(root / 'result.json', 65536).decode())
    expected_envelope = {
        'version': POLICY_VERSION, 'request_id': request.request_id,
        'source_id': request.source_id, 'config_id': request.config_id,
        'cache_evidence_id': digest(evidence),
        'status': 'dataset_batch_passed', 'files': {name: files[name] for name in FILES},
    }

    if result != expected_envelope:
        raise ValueError('Dataset 결과가 현재 요청·캐시 증거와 다릅니다.')

    report = _decode(_read_small(root / 'dataset_check.json', 65536).decode())
    cache = evidence['cache_report']
    expected = {
        'status': 'dataset_batch_passed', 'batch_size': contract.batch_size,
        'device': 'cpu', 'num_workers': 0, 'test_maps_read': False,
        'model_trained': False, 'is_performance_evaluation': False,
        'development_inputs_ready': True, 'scope': 'development_integrity_and_first_batch_only',
        'torch_version': contract.torch_version, 'splits': {},
    }

    for name in ('train', 'validation'):
        expected['splits'][name] = {
            'status': 'dataset_batch_passed', 'split': name,
            'image_shape': [min(contract.batch_size, cache['splits'][name]['rows']), 2,
                            *cache['transform_config']['target_size']],
            'image_dtype': 'torch.float32', 'target_dtype': 'torch.int64',
            'split_protocol_id': cache['split_protocol_id'], 'test_maps_read': False,
            'model_trained': False, 'is_performance_evaluation': False,
            'scope': 'selected_split_integrity_and_first_batch_only',
        }

    if (report != expected or report.get('development_inputs_ready') is not True or
        any(report.get(key) is not False for key in ('test_maps_read', 'model_trained', 'is_performance_evaluation')) or
        type(report.get('num_workers')) is not int or type(report.get('batch_size')) is not int):
        raise ValueError('Dataset의 구간·shape·dtype·분할 계약이 다릅니다.')

    for item in report['splits'].values():
        if (any(type(value) is not int for value in item['image_shape']) or
            any(item.get(key) is not False for key in ('test_maps_read', 'model_trained', 'is_performance_evaluation'))):
            raise ValueError('Dataset 결과에서 bool과 정수는 서로 대체할 수 없습니다.')

    marker = _decode(_read_small(root / 'SUCCESS.json', 4096).decode())

    if (marker != {'status': 'dataset_batch_passed', 'test_maps_read': False, 'model_trained': False} or
        marker.get('test_maps_read') is not False or marker.get('model_trained') is not False):
        raise ValueError('Dataset 검사 완료 표식이 올바르지 않습니다.')

    _read_small(root / 'dataset_check.md', 16384)

    if result_files(root, contract.max_output_bytes) != files:
        raise ValueError('Dataset 검사 중 결과 파일이 변경되었습니다.')

    return {'status': 'dataset_batch_passed', 'development_inputs_ready': True,
            'test_maps_read': False, 'model_trained': False, 'files': files}

"""검증된 Train·Validation 캐시의 Dataset 첫 배치만 읽는 CPU 격리 worker."""

from pathlib import Path
import sys


def execute_request(request, root):
    import json
    from dataclasses import asdict

    from src.agents.pickle_import_worker import _diagnostic
    from src.agents.wm_canonical_contract import digest
    from src.agents.wm_cache_results import cache_files
    from src.agents.wm_dataset_contract import POLICY_VERSION, WmDatasetInput
    from src.agents.wm_dataset_results import result_files
    from src.agents.wm_dataset_runtime import decode_manifest, MAX_MANIFEST_BYTES
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json

    root = Path(root).absolute()
    source = root / 'input/cache'
    stage = 'REQUEST'

    try:
        paths = tuple(str(root / ('input/' + name)) for name in ('analysis.json', 'files.json', 'runtime_files.json'))

        if (request.task_id != root.name or request.stage != 'validate_contract' or
            request.action_id != 'validate_contract' or request.policy_version != POLICY_VERSION or
            request.input_files != paths or request.dependency_directories != (str(source),) or
            request.output_directory != str(root / 'output/result') or
            request.scratch_directory != str(root / 'scratch/task')):
            raise ValueError('Dataset 검사 요청이 올바르지 않습니다.')

        with (root / 'input/analysis.json').open('rb') as stream:
            packet = _decode(stream.read(65537).decode())

        fields = {'version', 'source_id', 'code_id', 'limits', 'max_input_bytes', 'max_output_bytes',
                  'max_runtime_bytes', 'runtime_sha256', 'torch_version', 'batch_size', 'cache_report', 'cache_evidence_id'}

        if (set(packet) != fields or packet['version'] != POLICY_VERSION or
            packet['source_id'] != request.source_id or packet['limits'] != asdict(request.limits) or
            digest(packet) != request.config_id):
            raise ValueError('Dataset 검사 설정이 승인 범위와 다릅니다.')

        with (root / 'input/files.json').open('rb') as stream:
            files = decode_json(stream.read(1024 * 1024 + 1))

        with (root / 'input/runtime_files.json').open('rb') as stream:
            runtime = decode_manifest(stream.read(MAX_MANIFEST_BYTES + 1), packet['max_runtime_bytes'])

        contract = WmDatasetInput(str(source), json.dumps(files), request.limits,
                                  packet['max_input_bytes'], packet['max_output_bytes'], packet['max_runtime_bytes'],
                                  packet['runtime_sha256'], packet['torch_version'], packet['batch_size'])

        if (contract.source_id != request.source_id or digest(runtime) != contract.runtime_sha256 or
            cache_files(source, contract.max_input_bytes) != files):
            raise ValueError('Dataset 검사 캐시·런타임 명세가 변경되었습니다.')

        with (source / 'cache.json').open('rb') as stream:
            cache_report = decode_json(stream.read(1024 * 1024 + 1))

        if cache_report != packet['cache_report'] or set(cache_report['splits']) != {'train', 'validation'}:
            raise ValueError('개발 구간 캐시 보고서가 승인한 내용과 다릅니다.')

        stage = 'IMPORT_TORCH'
        _diagnostic(root, stage, 'START')
        import torch
        from src.wafer_dl.dataset import check_first_batch

        if torch.__version__ != contract.torch_version:
            raise ValueError('승인 후 PyTorch 버전이 변경되었습니다.')

        torch.set_num_threads(1)
        splits = {}

        # CPU DataLoader, shuffle=False, num_workers=0은 기존 코어가 고정한다.
        # Test 인자를 외부에서 받지 않으며 두 개발 구간만 명시적으로 순회한다.
        for name in ('train', 'validation'):
            stage = 'CHECK_TRAIN' if name == 'train' else 'CHECK_VALIDATION'
            _diagnostic(root, stage, 'START')
            splits[name] = check_first_batch(source, name, contract.batch_size, allow_test=False)

        if cache_files(source, contract.max_input_bytes) != files:
            raise ValueError('Dataset 검사 중 캐시 입력이 변경되었습니다.')

        report = {'status': 'dataset_batch_passed', 'batch_size': contract.batch_size,
                  'device': 'cpu', 'num_workers': 0, 'test_maps_read': False,
                  'model_trained': False, 'is_performance_evaluation': False,
                  'development_inputs_ready': True, 'scope': 'development_integrity_and_first_batch_only',
                  'torch_version': contract.torch_version, 'splits': splits}
        output = Path(request.output_directory) / 'dataset'
        output.mkdir()
        (output / 'dataset_check.json').write_text(json.dumps(report, allow_nan=False), encoding='utf-8')
        lines = ['# WM Dataset 첫 배치 검사', '',
                 'Train·Validation 캐시 정합성과 첫 배치만 확인했습니다. 모델 학습·성능 평가는 하지 않았습니다.', '',
                 '| 구간 | 입력 shape | 입력 dtype | 라벨 dtype |', '| --- | --- | --- | --- |']

        for name, item in splits.items():
            lines.append(f"| {name} | {item['image_shape']} | {item['image_dtype']} | {item['target_dtype']} |")

        lines += ['', 'CPU·단일 프로세스 검사입니다. Test 배열·모델·GPU 연산은 사용하지 않았습니다.',
                  '전체 표본의 Tensor 내용을 모두 검사한 결과가 아닙니다. 캐시·분할 파일은 변경하지 않았습니다.']
        (output / 'dataset_check.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (output / 'SUCCESS.json').write_text(json.dumps({'status': 'dataset_batch_passed',
                                                       'test_maps_read': False, 'model_trained': False}), encoding='utf-8')
        records = result_files(output, contract.max_output_bytes)
        payload = json.dumps({'version': POLICY_VERSION, 'request_id': request.request_id,
                              'source_id': request.source_id, 'config_id': request.config_id,
                              'cache_evidence_id': packet['cache_evidence_id'],
                              'status': 'dataset_batch_passed', 'files': records}, allow_nan=False).encode()

        if len(payload) + sum(item['size'] for item in records.values()) > contract.max_output_bytes:
            raise ValueError('Dataset 보고서 출력 상한을 초과했습니다.')

        (output / 'result.json').write_bytes(payload)
        _diagnostic(root, 'COMPLETE', 'PASSED')
        return WorkerResult(request.task_id, request.request_id, 'succeeded', 'WORKER_EXITED_OK', ('dataset',))
    except Exception as error:
        code = 'DATASET_ACCESS_DENIED' if isinstance(error, PermissionError) else 'DATASET_CHECK_FAILED'
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, 'blocked_input', 'WORKER_INPUT_BLOCKED')


def main():
    import os

    if len(sys.argv) != 3:
        return 2

    root = Path(sys.argv[1]).absolute()
    sys.path.insert(0, str(root / 'tools'))
    from src.agents.pickle_import_worker import _isolated_process

    if not _isolated_process():
        return 2

    runtime = root / 'runtime'
    sys.path.insert(1, str(runtime / 'Lib/site-packages'))
    handles = [os.add_dll_directory(str(path)) for path in
               (runtime, runtime / 'DLLs', runtime / 'Library/bin', runtime / 'Lib/site-packages/torch/lib')
               if path.is_dir()]
    from src.agents.worker_contract import WorkerRequest
    from src.agents.pickle_import_worker import _diagnostic

    try:
        with (root / 'input/request.json').open('rb') as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode())

        if request.request_id != sys.argv[2]:
            return 2

        reply = execute_request(request, root)
        (root / 'output/result/worker_result.json').write_text(reply.to_json(), encoding='utf-8')
        return 0 if reply.status == 'succeeded' else 2
    except Exception:
        _diagnostic(root, 'BOOTSTRAP', 'DATASET_BOOTSTRAP_FAILED')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

"""Train·Validation만 사용해 WM-811K 모델 비교를 실행하고 집계한다.

Test 분할은 열지 않으며, 결과는 모델 선택에 쓰인 Validation 수치이므로
일반화 성능으로 보고하지 않는다. 사전에 정한 판정 규칙이 없으므로 자동으로
최종 모델을 고르지 않고 검토 지표만 나란히 보여 준다.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from .augment import AUGMENTATIONS
from .cache import file_digest
from .dataset import WaferDataset
from .models import DEFAULT_COMPARE_MODELS, MODEL_NAMES, build_model
from .train import load_train_config, train
from .validate_canonical import CLASS_NAMES, read_json, require, validation_stage


DEFAULT_SEEDS = (42, 43, 44)

# 실행 재사용 계약. 항목이 바뀌면 버전을 올려 이전 결과와 섞이지 않게 한다.
RUN_CONTRACT_VERSION = 'wm_compare_run_v3'
RUN_COMPLETE_STATUS = 'compare_run_complete'
RUN_ARTIFACTS = ('best_model.pt', 'training_report.json', 'inference.json')

# 학습·모델·증강·입력·지표·비교 결과에 영향을 주는 실제 호출 경로의 파일이다.
# 캐시 생성 전용 코드는 cache_sha256으로, Test 전용 evaluate.py는 이 경로 밖이라 뺀다.
# 해시는 코드 혼용을 찾기 위한 값이며 인증이나 파일 잠금이 아니다.
PACKAGE_DIR = Path(__file__).resolve().parent
RUN_CODE_FILES = (
    'augment.py',
    'cache.py',
    'compare.py',
    'dataset.py',
    'metrics.py',
    'models.py',
    'train.py',
    'transform.py',
    'validate_canonical.py',
)

# 실패 코드별로 사용자가 할 수 있는 다음 행동을 안내한다.
FAILURE_GUIDANCE = {
    'COMPARE_RUN_INCOMPLETE': (
        '완료 표식이 없는 실행 폴더입니다. 기존 폴더를 보존하려면 '
        '--output-root에 새 경로를 지정해 다시 실행하세요.'
    ),
    'COMPARE_RUN_CONTRACT_MISMATCH': (
        '캐시·분할·학습 설정·증강·장치가 다른 결과입니다. 기존 결과는 그대로 두고 '
        '--output-root에 새 경로를 지정하세요.'
    ),
    'COMPARE_RUN_INTEGRITY_FAILED': (
        '완료 기록과 저장 파일이 일치하지 않습니다. 해당 폴더는 재사용할 수 없으니 '
        '보존한 채 --output-root에 새 경로를 지정하세요.'
    ),
    'COMPARE_RUN_CODE_MISMATCH': (
        '실행 코드가 다른 결과입니다. 서로 다른 코드의 결과는 섞지 않습니다. 기존 결과는 '
        '그대로 두고 --output-root에 새 경로를 지정해 같은 코드로 전체를 실행하세요.'
    ),
    'COMPARE_RUN_CONTRACT_OUTDATED': (
        '실행 코드 식별값이 없는 이전 계약의 결과입니다. 자동으로 갱신하지 않으니 기존 결과는 '
        '그대로 두고 --output-root에 새 경로를 지정하세요.'
    ),
    'COMPARE_REPORT_ALREADY_EXISTS': (
        '이전 집계를 덮어쓰지 않습니다. --report-dir에 새 경로를 지정하세요.'
    ),
}

# 판정 규칙을 실험 전에 정하지 않았으므로 자동 선정 없이 보류로 표시한다.
SELECTION_STATUS = '선정 보류'
SELECTION_REASON = (
    '사전에 정한 판정 규칙이 없습니다. Validation Macro-F1, 희소 클래스 Recall, '
    'seed 변동, 추론 시간·메모리를 함께 검토해 사람이 결정합니다.'
)

# Train 표본이 적은 순서의 희소 클래스. Recall을 따로 표시한다.
SPARSE_CLASSES = ('Near-Full', 'Donut', 'Random', 'Scratch')

# 추론 시간 측정 조건: 학습과 같은 배치 크기로 640표본을 3회 반복한다.
INFERENCE_BATCH_SIZE = 64
INFERENCE_BATCHES = 10
INFERENCE_PASSES = 3


def run_name(model_name, augmentation, seed):
    """실험군·증강·seed를 나타내는 실행 폴더 이름을 만든다."""
    return f'{model_name}_{augmentation}_seed{seed}'


def measure_inference(run_dir, cache_dir, device='cuda'):
    """디스크 읽기를 제외하고 메모리의 Validation 배치로 추론 비용을 잰다."""
    run_dir = Path(run_dir)
    selected_device = torch.device(device)
    uses_cuda = selected_device.type == 'cuda'

    # 저장된 checkpoint로 모델을 복원한다.
    checkpoint = torch.load(
        run_dir / 'best_model.pt',
        map_location='cpu',
        weights_only=True,
    )
    model = build_model(checkpoint['model_name'], num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(selected_device).eval()

    # 측정할 배치를 미리 장치에 올려 디스크 읽기 시간을 뺀다.
    dataset = WaferDataset(cache_dir, split='validation')
    try:
        rows = min(len(dataset), INFERENCE_BATCH_SIZE * INFERENCE_BATCHES)
        images = torch.stack([dataset[index]['image'] for index in range(rows)])
    finally:
        dataset.close()

    batches = [
        batch.to(selected_device)
        for batch in images.split(INFERENCE_BATCH_SIZE)
    ]

    def synchronize():
        if uses_cuda:
            torch.cuda.synchronize(selected_device)

    if uses_cuda:
        torch.cuda.reset_peak_memory_stats(selected_device)

    with torch.inference_mode():
        # 첫 실행의 초기화 비용이 섞이지 않도록 한 바퀴는 측정에서 뺀다.
        for batch in batches:
            model(batch)
        synchronize()

        started_at = time.perf_counter()
        for _ in range(INFERENCE_PASSES):
            for batch in batches:
                model(batch)
        synchronize()

    seconds = time.perf_counter() - started_at

    return {
        'device': str(selected_device),
        'batch_size': INFERENCE_BATCH_SIZE,
        'samples_per_pass': rows,
        'passes': INFERENCE_PASSES,
        'ms_per_sample': seconds * 1000 / (rows * INFERENCE_PASSES),
        'peak_cuda_memory_bytes': (
            torch.cuda.max_memory_allocated(selected_device) if uses_cuda else None
        ),
    }


def _resolve_device(device):
    """요청한 장치를 실제 사용할 장치 이름으로 바꾼다."""
    return 'cuda' if device != 'cpu' and torch.cuda.is_available() else 'cpu'


def _digest_files(file_digests):
    """파일별 해시 목록을 순서와 무관한 하나의 식별값으로 묶는다."""
    payload = json.dumps(file_digests, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def code_fingerprint(root=PACKAGE_DIR, files=RUN_CODE_FILES):
    """미커밋 변경도 구분되도록 파일 내용으로 실행 코드 식별값을 만든다."""
    file_digests = {name: file_digest(Path(root) / name) for name in sorted(files)}
    return {'sha256': _digest_files(file_digests), 'files': file_digests}


def build_run_contract(
    cache_dir,
    config_path,
    device,
    max_epochs,
    model_name,
    augmentation,
    seed,
):
    """결과를 재사용해도 되는지 판단할 입력 조건을 한 묶음으로 만든다."""
    cache_file = Path(cache_dir).resolve(strict=True) / 'cache.json'
    cache = read_json(cache_file)
    require(cache.get('status') == 'cache_complete', 'CACHE_NOT_COMPLETE')

    config = load_train_config(config_path)
    if max_epochs is not None:
        config = {**config, 'epochs': max_epochs}

    return {
        'contract_version': RUN_CONTRACT_VERSION,
        'model_name': model_name,
        'seed': seed,
        'device': _resolve_device(device),
        'training_config': {**config, 'seed': seed, 'augmentation': augmentation},
        'split_protocol_id': cache.get('split_protocol_id'),
        'source_split_manifest_sha256': cache.get('source_split_manifest_sha256'),
        'cache_sha256': file_digest(cache_file),
        'code_fingerprint': code_fingerprint(),
    }


def _shared_contract(contract):
    """실험군 사이에 같아야 하는 조건만 남긴다."""
    shared = {
        key: value for key, value in contract.items()
        if key not in ('model_name', 'seed')
    }
    shared['training_config'] = {
        key: value for key, value in contract['training_config'].items()
        if key not in ('seed', 'augmentation')
    }
    return shared


def verify_run(run_dir, model_name, augmentation, seed, contract=None):
    """완료 표식, 입력 조건, 파일 해시, 보고서와 checkpoint의 일치를 확인한다."""
    run_dir = Path(run_dir)
    require(
        run_dir.is_dir() and (run_dir / 'SUCCESS.json').is_file(),
        'COMPARE_RUN_INCOMPLETE',
    )

    success = read_json(run_dir / 'SUCCESS.json')
    stored = success.get('contract')
    artifacts = success.get('artifacts')
    require(
        success.get('status') == RUN_COMPLETE_STATUS and
        type(stored) is dict and
        type(artifacts) is dict and
        set(artifacts) == set(RUN_ARTIFACTS),
        'COMPARE_RUN_INCOMPLETE',
    )

    # 이전 계약은 자동으로 승격하지 않고 고정 코드로 거부한다.
    fingerprint = stored.get('code_fingerprint')
    require(
        stored.get('contract_version') == RUN_CONTRACT_VERSION and
        type(fingerprint) is dict,
        'COMPARE_RUN_CONTRACT_OUTDATED',
    )
    file_digests = fingerprint.get('files')
    require(
        type(file_digests) is dict and
        _digest_files(file_digests) == fingerprint.get('sha256'),
        'COMPARE_RUN_INTEGRITY_FAILED',
    )
    require(
        contract is None or fingerprint == contract['code_fingerprint'],
        'COMPARE_RUN_CODE_MISMATCH',
    )

    # 폴더 이름과 요청한 조건이 모두 맞아야 재사용한다.
    require(
        stored.get('model_name') == model_name and
        stored.get('seed') == seed and
        stored.get('training_config', {}).get('augmentation') == augmentation and
        (contract is None or stored == contract),
        'COMPARE_RUN_CONTRACT_MISMATCH',
    )

    # 내용을 읽기 전에 해시로 완료 뒤 파일이 바뀌지 않았는지 확인한다.
    for name in RUN_ARTIFACTS:
        path = run_dir / name
        require(
            path.is_file() and file_digest(path) == artifacts[name],
            'COMPARE_RUN_INTEGRITY_FAILED',
        )

    report = read_json(run_dir / 'training_report.json')
    inference = read_json(run_dir / 'inference.json')
    checkpoint = torch.load(
        run_dir / 'best_model.pt',
        map_location='cpu',
        weights_only=True,
    )

    inference_time = inference.get('ms_per_sample')
    require(
        type(inference_time) in (int, float) and inference_time > 0 and
        inference.get('device') == stored['device'],
        'COMPARE_RUN_INTEGRITY_FAILED',
    )

    require(
        report.get('status') == 'training_complete' and
        report.get('test_split_used') is False and
        report.get('device') == stored['device'] and
        report.get('model_name') == model_name and
        report.get('seed') == seed and
        report.get('augmentation') == augmentation and
        report.get('split_protocol_id') == stored['split_protocol_id'] and
        checkpoint.get('model_name') == model_name and
        checkpoint.get('augmentation') == augmentation and
        checkpoint.get('training_config') == stored['training_config'] and
        checkpoint.get('split_protocol_id') == stored['split_protocol_id'] and
        checkpoint.get('source_split_manifest_sha256') ==
        stored['source_split_manifest_sha256'] and
        checkpoint.get('selected_epoch') == report.get('selected_epoch'),
        'COMPARE_RUN_INTEGRITY_FAILED',
    )

    # 가중치가 실제 후보 구조에 맞는지 strict 로딩으로 확인한다.
    # 다른 코드로 만든 결과는 현재 구조로 재검증할 수 없으므로 로딩하지 않는다.
    if fingerprint == code_fingerprint():
        try:
            build_model(model_name, num_classes=len(CLASS_NAMES)).load_state_dict(
                checkpoint['model_state_dict'],
                strict=True,
            )
        except RuntimeError:
            require(False, 'COMPARE_RUN_INTEGRITY_FAILED')

    return stored


def _set_aside_incomplete(temporary, final):
    """끝나지 않은 임시 폴더를 지우지 않고 다른 이름으로 보존한다."""
    for number in range(1, 1000):
        target = final.with_name(f'{final.name}.incomplete_{number}')
        if not target.exists():
            temporary.rename(target)
            return

    require(False, 'TOO_MANY_INCOMPLETE_RUNS')


def _publish_run(temporary, final, contract):
    """산출물을 검증한 뒤 폴더 이름 변경으로 결과를 한 번에 공개한다."""
    # 학습 중 코드가 바뀌면 시작할 때의 계약과 달라지므로 완료로 공개하지 않는다.
    require(
        code_fingerprint() == contract['code_fingerprint'],
        'COMPARE_RUN_CODE_MISMATCH',
    )

    success = {
        'status': RUN_COMPLETE_STATUS,
        'contract': contract,
        'artifacts': {name: file_digest(temporary / name) for name in RUN_ARTIFACTS},
    }
    (temporary / 'SUCCESS.json').write_text(
        json.dumps(success, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )

    # 재개할 때와 같은 검증을 통과한 결과만 공개한다.
    verify_run(
        temporary,
        contract['model_name'],
        contract['training_config']['augmentation'],
        contract['seed'],
        contract,
    )
    temporary.rename(final)


def plan_runs(
    cache_dir,
    output_root,
    config_path,
    device='auto',
    max_epochs=None,
    models=DEFAULT_COMPARE_MODELS,
    augmentations=AUGMENTATIONS,
    seeds=DEFAULT_SEEDS,
):
    """기존 결과를 모두 검사하고 완료 실행과 남은 실행을 나눈다."""
    root = Path(output_root).resolve()
    complete = []
    pending = []

    for augmentation in augmentations:
        for seed in seeds:
            for model_name in models:
                final = root / run_name(model_name, augmentation, seed)
                contract = build_run_contract(
                    cache_dir,
                    config_path,
                    device,
                    max_epochs,
                    model_name,
                    augmentation,
                    seed,
                )

                if final.exists():
                    verify_run(final, model_name, augmentation, seed, contract)
                    complete.append(final)
                else:
                    pending.append((final, contract))

    return complete, pending


def run_all(
    cache_dir,
    output_root,
    config_path,
    device='auto',
    max_epochs=None,
    models=DEFAULT_COMPARE_MODELS,
    augmentations=AUGMENTATIONS,
    seeds=DEFAULT_SEEDS,
):
    """실험군을 순서대로 학습하고 조건이 같은 완료 실행만 건너뛴다."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 긴 학습을 시작하기 전에 기존 결과를 모두 검사해 문제가 있으면 바로 중단한다.
    complete, pending = plan_runs(
        cache_dir,
        root,
        config_path,
        device,
        max_epochs,
        models,
        augmentations,
        seeds,
    )
    for final in complete:
        print(json.dumps({'run': final.name, 'status': 'skipped_complete'}))

    failures = []

    # 시간 지표가 섞이지 않도록 동시 실행 없이 한 번에 하나씩 학습한다.
    for final, contract in pending:
        temporary = final.with_name(final.name + '.tmp')
        if temporary.exists():
            _set_aside_incomplete(temporary, final)

        try:
            with validation_stage('training'):
                train(
                    cache_dir,
                    temporary,
                    config_path,
                    device,
                    max_epochs,
                    contract['model_name'],
                    contract['seed'],
                    contract['training_config']['augmentation'],
                )
                inference = measure_inference(temporary, cache_dir, contract['device'])
                (temporary / 'inference.json').write_text(
                    json.dumps(inference, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8',
                )
                _publish_run(temporary, final, contract)

        except ValueError as error:
            # 한 실행이 실패해도 무인 실행을 멈추지 않고 끝에 모아서 알린다.
            # 공개되지 않은 .tmp 폴더는 다음 실행 때 지우지 않고 따로 보존한다.
            failures.append({'run': final.name, 'code': str(error)})
            print(json.dumps({**failures[-1], 'status': 'failed'}, ensure_ascii=False))
            continue

        print(json.dumps({'run': final.name, 'status': 'complete'}))

    return failures


def read_run(run_dir):
    """실행 보고서에서 선택 epoch의 Validation 결과를 한 행으로 정리한다."""
    report = read_json(Path(run_dir) / 'training_report.json')
    inference = read_json(Path(run_dir) / 'inference.json')

    require(
        report.get('status') == 'training_complete' and
        report.get('test_split_used') is False and
        report.get('selection_metric') == 'macro_f1',
        'INVALID_COMPARE_RUN',
    )

    # 마지막 epoch가 아니라 저장된 checkpoint와 같은 selected_epoch 값을 쓴다.
    selected = [
        epoch for epoch in report['history']
        if epoch['epoch'] == report['selected_epoch']
    ]
    require(len(selected) == 1, 'INVALID_COMPARE_RUN')

    metrics = selected[0]['validation']
    classes = {item['pattern_label']: item for item in metrics['per_class']}

    row = {
        # 실행 식별 정보
        'model_name': report['model_name'],
        'augmentation': report['augmentation'],
        'seed': report['seed'],
        'split_protocol_id': report['split_protocol_id'],
        'parameter_count': report['parameter_count'],
        'model_description': report['model_description'],

        # 학습 경과. 상한까지 갔다면 조기 종료 없이 끝나 미수렴 가능성이 있다.
        'selected_epoch': report['selected_epoch'],
        'epochs_run': len(report['history']),
        'hit_epoch_limit': len(report['history']) == report['max_epochs'],
        'selected_train_loss': selected[0]['train_loss'],
        'last_train_loss': report['history'][-1]['train_loss'],

        # 비용 지표
        'training_seconds': report['training_seconds'],
        'peak_cuda_memory_bytes': report['peak_cuda_memory_bytes'],
        'inference_ms_per_sample': inference['ms_per_sample'],
        'inference_peak_cuda_memory_bytes': inference['peak_cuda_memory_bytes'],

        # Validation 성능 지표
        'macro_f1': metrics['macro_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'accuracy': metrics['accuracy'],
    }

    for name in CLASS_NAMES:
        for key in ('precision', 'recall', 'f1'):
            row[f'{name}_{key}'] = classes[name][key]

    return row


def _mean_std(values):
    """표본 표준편차를 포함한 평균을 계산한다."""
    return {
        'mean': statistics.fmean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_runs(
    rows,
    models=DEFAULT_COMPARE_MODELS,
    augmentations=AUGMENTATIONS,
    seeds=DEFAULT_SEEDS,
):
    """모든 실행이 있는지 확인하고 모델·증강별 평균·표준편차를 계산한다."""
    # 다른 분할의 결과나 빠진 실행이 평균에 섞이면 비교가 무의미하므로 거부한다.
    protocols = {row['split_protocol_id'] for row in rows}
    require(protocols and len(protocols) == 1, 'COMPARE_SPLIT_MISMATCH')

    found = sorted((row['model_name'], row['augmentation'], row['seed']) for row in rows)
    expected = sorted(
        (model, augmentation, seed)
        for model in models
        for augmentation in augmentations
        for seed in seeds
    )
    require(found == expected, 'COMPARE_RUNS_INCOMPLETE')

    numeric = [
        key for key, value in rows[0].items()
        if type(value) in (int, float) and key not in ('seed', 'parameter_count')
    ]

    summary = []
    for model in models:
        for augmentation in augmentations:
            group = [
                row for row in rows
                if row['model_name'] == model and row['augmentation'] == augmentation
            ]
            scores = [row['macro_f1'] for row in group]
            item = {
                'model_name': model,
                'augmentation': augmentation,
                'runs': len(group),
                'parameter_count': group[0]['parameter_count'],
                'model_description': group[0].get('model_description'),
                'runs_at_epoch_limit': sum(row['hit_epoch_limit'] for row in group),
                'macro_f1_min': min(scores),
                'macro_f1_max': max(scores),
            }

            for key in numeric:
                result = _mean_std([row[key] for row in group])
                item[f'{key}_mean'] = result['mean']
                item[f'{key}_std'] = result['std']

            summary.append(item)

    return summary


def rank_candidates(summary):
    """Macro-F1 평균 순으로 정렬한다. 표시 순서일 뿐 우열 판정이 아니다."""
    return sorted(summary, key=lambda item: item['macro_f1_mean'], reverse=True)


def _write_csv(path, rows):
    """행 목록을 새 CSV로 저장한다. 엑셀 호환을 위해 BOM을 붙인다."""
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(item, key):
    """평균과 표준편차를 퍼센트 문자열로 만든다."""
    return f"{item[key + '_mean'] * 100:.2f} ± {item[key + '_std'] * 100:.2f}"


def _mebibytes(item, key):
    """바이트 평균을 MiB 문자열로 만든다. 측정값이 없으면 N/A로 표시한다."""
    value = item.get(key + '_mean')
    return 'N/A' if value is None else f'{value / 2 ** 20:.0f}'


def _write_markdown(path, ranked, code_sha256, current_code_match):
    """검토 지표 표, 선정 상태와 해석 한계를 Markdown으로 저장한다."""
    code_state = '현재 코드와 일치' if current_code_match else '현재 코드와 다름'
    sparse_headers = ''.join(f'{name} Recall (%) | ' for name in SPARSE_CLASSES)

    # 해석 범위 안내
    lines = [
        '# WM-811K 모델 비교 결과 (Validation 기준)',
        '',
        'Test는 사용하지 않았습니다. Validation은 checkpoint와 모델 선택에 쓰였으므로',
        '아래 수치는 일반화 성능이 아니며 Test 기록과 직접 비교하지 않습니다.',
        '',
        f'**선정 상태: {SELECTION_STATUS}.** {SELECTION_REASON}',
        '',
        '표는 Macro-F1 평균 순서이며 우열 판정이나 통계적 유의성을 뜻하지 않습니다.',
        '',
        f'실행 코드 식별값: `{code_sha256}` ({code_state}). 혼용 검사용이며 인증이 아닙니다.',
        '',
    ]

    # 성능 표
    lines += [
        '| 모델 | 증강 | Macro-F1 (%) | seed별 범위 (%) | ' + sparse_headers +
        'Scratch Precision (%) | Loc Recall (%) | epoch 상한 도달 |',
        '| --- | --- |' + ' ---: |' * (len(SPARSE_CLASSES) + 5),
    ]

    for item in ranked:
        sparse_values = ''.join(
            f"{_percent(item, name + '_recall')} | " for name in SPARSE_CLASSES
        )
        lines.append(
            f"| {item['model_name']} | {item['augmentation']} "
            f"| {_percent(item, 'macro_f1')} "
            f"| {item['macro_f1_min'] * 100:.2f}~{item['macro_f1_max'] * 100:.2f} "
            f'| {sparse_values}'
            f"{_percent(item, 'Scratch_precision')} "
            f"| {_percent(item, 'Loc_recall')} "
            f"| {item['runs_at_epoch_limit']}/{item['runs']} |"
        )

    # 비용 표
    lines += [
        '',
        '| 모델 | 구조 | 증강 | 파라미터 | 학습 시간(분) | 학습 메모리(MiB) '
        '| 추론(ms/표본) | 추론 메모리(MiB) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]

    for item in ranked:
        lines.append(
            f"| {item['model_name']} | {item['model_description']} "
            f"| {item['augmentation']} "
            f"| {item['parameter_count']:,} "
            f"| {item['training_seconds_mean'] / 60:.1f} "
            f"| {_mebibytes(item, 'peak_cuda_memory_bytes')} "
            f"| {item['inference_ms_per_sample_mean']:.3f} "
            f"| {_mebibytes(item, 'inference_peak_cuda_memory_bytes')} |"
        )

    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def summarize(
    output_root,
    report_dir,
    models=DEFAULT_COMPARE_MODELS,
    augmentations=AUGMENTATIONS,
    seeds=DEFAULT_SEEDS,
):
    """완료된 실행을 읽어 비교 CSV·검토 표를 새 폴더에 저장한다."""
    root = Path(output_root).resolve()
    combinations = [
        (model, augmentation, seed)
        for augmentation in augmentations
        for seed in seeds
        for model in models
    ]

    # 검증을 통과한 실행만 읽고, 실험군 간 조건이 같은지도 확인한다.
    contracts = [
        verify_run(root / run_name(*combination), *combination)
        for combination in combinations
    ]

    # 과거 코드의 결과도 한 코드로 완결된 묶음이면 집계하되 현재 코드와의 차이를 표시한다.
    code_digests = {item['code_fingerprint']['sha256'] for item in contracts}
    require(len(code_digests) == 1, 'COMPARE_RUN_CODE_MISMATCH')
    code_sha256 = code_digests.pop()
    current_code_match = code_sha256 == code_fingerprint()['sha256']

    require(
        all(_shared_contract(item) == _shared_contract(contracts[0]) for item in contracts),
        'COMPARE_RUN_CONTRACT_MISMATCH',
    )

    rows = [read_run(root / run_name(*combination)) for combination in combinations]
    ranked = rank_candidates(summarize_runs(rows, models, augmentations, seeds))

    # 이전 집계를 덮어쓰지 않도록 이미 있는 폴더는 실패시킨다.
    report = Path(report_dir).resolve()
    require(not report.exists(), 'COMPARE_REPORT_ALREADY_EXISTS')

    # 임시 폴더에 모두 쓴 뒤 이름을 바꿔 일부만 쓰인 집계가 보이지 않게 한다.
    temporary = report.with_name(report.name + '.tmp')
    if temporary.exists():
        _set_aside_incomplete(temporary, report)
    temporary.mkdir(parents=True)

    _write_csv(temporary / 'runs.csv', rows)
    _write_csv(temporary / 'summary.csv', ranked)
    _write_markdown(
        temporary / 'comparison.md', ranked, code_sha256, current_code_match
    )
    temporary.rename(report)

    return {
        'code_sha256': code_sha256,
        'current_code_match': current_code_match,
        'selection_status': SELECTION_STATUS,
        'selection_reason': SELECTION_REASON,
        'selected_model': None,
        'candidates': [
            {
                'model_name': item['model_name'],
                'augmentation': item['augmentation'],
                'macro_f1_mean': item['macro_f1_mean'],
                'macro_f1_std': item['macro_f1_std'],
            }
            for item in ranked
        ],
    }


def _add_matrix_arguments(parser):
    """실행 조합을 고르는 공통 옵션을 추가한다."""
    parser.add_argument('--models', choices=MODEL_NAMES, nargs='+',
                        default=list(DEFAULT_COMPARE_MODELS))
    parser.add_argument('--augmentations', choices=AUGMENTATIONS, nargs='+',
                        default=list(AUGMENTATIONS))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))


def _add_run_arguments(parser):
    """학습 조건을 정하는 공통 옵션을 추가한다."""
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-root', default='models/wm811k/compare_v2')
    parser.add_argument('--config', default='configs/wm811k/train.json')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int, help='설정 파일의 epoch 상한을 이번 실행에만 덮어씀')
    _add_matrix_arguments(parser)


def _print_plan(args):
    """학습 없이 전체·완료·남은 실행 수와 남은 epoch 상한을 출력한다."""
    complete, pending = plan_runs(
        args.cache_dir,
        args.output_root,
        args.config,
        args.device,
        args.epochs,
        tuple(args.models),
        tuple(args.augmentations),
        tuple(args.seeds),
    )
    print(json.dumps({
        'total_runs': len(complete) + len(pending),
        'complete_runs': len(complete),
        'pending_runs': len(pending),
        'pending_max_epochs': sum(
            contract['training_config']['epochs'] for _, contract in pending
        ),
        'pending': [final.name for final, _ in pending],
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='WM-811K 모델 비교 (Train/Validation only)')
    commands = parser.add_subparsers(dest='command', required=True)

    plan = commands.add_parser('plan', help='학습 없이 실행 수와 기존 결과를 확인')
    _add_run_arguments(plan)

    run = commands.add_parser('run', help='실험군을 순서대로 학습')
    _add_run_arguments(run)

    summary = commands.add_parser('summarize', help='완료된 실행을 집계')
    summary.add_argument('--output-root', default='models/wm811k/compare_v2')
    summary.add_argument('--report-dir', default='reports/wm811k/results/compare_v2')
    _add_matrix_arguments(summary)

    args = parser.parse_args()

    try:
        with validation_stage('compare'):
            if args.command == 'plan':
                _print_plan(args)
                return 0

            if args.command == 'run':
                failures = run_all(
                    args.cache_dir,
                    args.output_root,
                    args.config,
                    args.device,
                    args.epochs,
                    tuple(args.models),
                    tuple(args.augmentations),
                    tuple(args.seeds),
                )
                print(json.dumps({'failures': failures}, ensure_ascii=False, indent=2))
                return 1 if failures else 0

            result = summarize(
                args.output_root,
                args.report_dir,
                tuple(args.models),
                tuple(args.augmentations),
                tuple(args.seeds),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

    except ValueError as error:
        print('WM 비교 실패 코드: ' + str(error))
        if str(error) in FAILURE_GUIDANCE:
            print(FAILURE_GUIDANCE[str(error)])
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

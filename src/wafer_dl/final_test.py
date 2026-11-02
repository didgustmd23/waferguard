"""동결된 후보 6개의 최종 Test 사전점검·평가·집계 진입점.

preflight는 동결 명세·checkpoint·분할·변환 메타데이터만 대조하며 Test 배열을 열지 않는다.
evaluate는 명시적 확인 옵션이 있을 때만 기존 evaluate.evaluate를 checkpoint별로 호출한다.
summarize는 저장된 결과만 읽어 모델 2개 × seed 3개를 집계하고 모델을 다시 고르지 않는다.
해시는 변경·혼용 검사용이며 인증이나 파일 잠금이 아니다.
"""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import tempfile
import time
from types import SimpleNamespace

import torch

from .cache import file_digest, load_cache_config, validate_check
from .check_transform import index_split
from .compare import (
    PACKAGE_DIR,
    _mean_std,
    _set_aside_incomplete,
    code_fingerprint,
    read_run,
    verify_run,
)
from .evaluate import (
    _require_finite_model_weights,
    evaluate as evaluate_checkpoint,
    sample_digest,
    validate_checkpoint,
)
from .metrics import classification_metrics
from .models import build_model
from .train import class_weights
from .transform import load_transform_config
from .validate_canonical import (
    CLASS_NAMES,
    ValidationError,
    decode_json,
    read_json,
    require,
    validation_stage,
)


FREEZE_FILES = ('freeze.json', 'README.md')
RUN_FILES = ('best_model.pt', 'training_report.json', 'inference.json', 'SUCCESS.json')
SEEDS = (42, 43, 44)
AUGMENTATION = 'd4'

# 동결 후보의 역할·경로·조건. 같은 이름의 다른 checkpoint로 바꾸지 못하게 코드에도 고정한다.
CANDIDATES = {
    'ResNet18': {
        'role': 'primary_candidate',
        'run_path': 'models/wm811k/weight_v1/alpha0_75/ResNet18_d4_seed{seed}',
        'scheduled': False,
    },
    'SmallWaferCNN': {
        'role': 'comparison_candidate',
        'run_path': 'models/wm811k/lr_v1/SmallWaferCNN/plateau_alpha0_75/'
                    'SmallWaferCNN_d4_seed{seed}',
        'scheduled': True,
    },
}
MODELS = tuple(CANDIDATES)

# 동결 추론 결과에 영향을 주는 파일만 학습 당시와 같아야 한다.
# 학습 전용 파일(augment·compare·train) 차이는 숨기지 않고 보고한다.
INFERENCE_CODE_FILES = (
    'cache.py', 'dataset.py', 'metrics.py', 'models.py', 'transform.py',
    'validate_canonical.py',
)
SESSION_CODE_FILES = (*INFERENCE_CODE_FILES, 'evaluate.py', 'final_test.py')

SESSION_VERSION = 'wm_final_test_session_v1'
RUN_RECORD = 'final_run.json'
COMPARISON_DIR = 'comparison'
COMPARISON_METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')
SCOPE_NOTE = (
    '예정된 하나의 최종 평가 세션에서 동결 모델 상태 6개를 같은 Test로 평가했습니다. '
    '전체 Test 접근이 1회라는 뜻이 아닙니다.'
)
SELECTION_SCOPE_NOTE = (
    'test_used_for_model_selection=false는 이번 동결 절차의 범위를 뜻합니다. '
    '과거 Test와 현재 Test의 표본 겹침은 미확인이며 이 값이 과거 이력을 증명하지 않습니다.'
)

FAILURE_GUIDANCE = {
    'FINAL_TEST_EVALUATION_CONFIRMATION_REQUIRED': (
        'Test를 읽지 않았습니다. 평가를 실행할 때만 --confirm-final-test-evaluation을 붙이세요.'
    ),
    'FREEZE_PIN_MISMATCH': (
        '검토한 동결 명세와 다릅니다. 평가하지 말고 freeze.json 변경 경위를 확인하세요.'
    ),
    'FINAL_TEST_OUTPUT_EXISTS': (
        '기존 결과를 덮어쓰지 않습니다. status로 상태를 확인하고, 이어서 진행할 때만 --resume을 쓰세요.'
    ),
    'FINAL_TEST_SESSION_MISMATCH': (
        '동결·캐시·코드·장치가 세션 시작 때와 다릅니다. 변경을 되돌리기 전에는 이어서 평가하지 마세요.'
    ),
    'FINAL_TEST_RETRY_DECISION_REQUIRED': (
        '실패·중단된 평가가 있습니다. status로 읽은 행 수를 확인한 뒤, 다시 평가할 건만 '
        '--retry 모델:seed로 명시하세요. 완료 건은 다시 평가하지 않습니다.'
    ),
    'CACHE_PATTERN_LOSS_REVIEW_REQUIRED': (
        '해상도·전처리를 바꾸거나 표본을 지우지 말고 멈춰서 원인을 검토하세요.'
    ),
}


class FinalTestStopped(Exception):
    """한 평가가 실패해 남은 평가를 자동으로 진행하지 않고 멈췄음을 알린다."""


def _now():
    return datetime.now().astimezone().isoformat()


def _write_new_json(path, payload):
    """기존 파일을 덮어쓰지 않고 새 JSON을 저장한다."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


# ==========================================
# 동결 명세 대조
# - SUCCESS 해시 → 평가 계획 → 실행별 파일 해시·계약·checkpoint 순서로 확인
# - 현재 추론 코드가 학습 당시와 다르면 거부하고 학습 전용 차이는 보고
# ==========================================
def load_freeze(freeze_dir, expected_sha256=None):
    """동결 완료 표식과 파일 해시를 확인한 뒤 명세를 읽는다."""
    root = Path(freeze_dir).resolve(strict=True)
    success = read_json(root / 'SUCCESS.json')
    artifacts = success.get('artifacts_sha256')
    require(success.get('status') == 'candidates_frozen_test_not_run' and
            type(artifacts) is dict and set(artifacts) == set(FREEZE_FILES),
            'FREEZE_INCOMPLETE')
    for name in FREEZE_FILES:
        require((root / name).is_file() and file_digest(root / name) == artifacts[name],
                'FREEZE_HASH_MISMATCH')
    require(expected_sha256 is None or artifacts['freeze.json'] == expected_sha256,
            'FREEZE_PIN_MISMATCH')

    freeze = read_json(root / 'freeze.json')
    plan = freeze.get('evaluation_plan')
    require(freeze.get('freeze_version') == 'wm_candidate_freeze_v1' and
            freeze.get('status') == 'candidates_frozen_test_not_run' and
            type(plan) is dict and plan.get('seeds') == list(SEEDS) and
            plan.get('checkpoint_count') == len(MODELS) * len(SEEDS) and
            plan.get('same_test_partition_for_all') is True and
            plan.get('fit_on_test') is False and plan.get('test_augmentation') == 'none' and
            plan.get('test_used_for_current_selection') is False and
            plan.get('test_executed_by_this_freeze') is False and
            plan.get('no_best_test_seed_selection') is True and
            plan.get('no_test_based_retuning') is True, 'FREEZE_PLAN_MISMATCH')
    candidates = freeze.get('candidates')
    require(type(candidates) is list and
            [item.get('model_name') for item in candidates] == list(MODELS) and
            all(item.get('role') == CANDIDATES[item['model_name']]['role'] and
                [run.get('seed') for run in item.get('runs', [])] == list(SEEDS)
                for item in candidates), 'FREEZE_CANDIDATE_MISMATCH')
    return freeze, artifacts['freeze.json']


def _verify_condition(contract, model_name, seed):
    """동결된 학습 조건이 후보 정의와 같은지 확인한다."""
    config = contract.get('training_config', {})
    require(contract.get('model_name') == model_name and contract.get('seed') == seed and
            config.get('augmentation') == AUGMENTATION and
            config.get('class_weight_alpha') == 0.75 and
            config.get('max_class_weight') == 10.0 and
            config.get('scratch_weight_factor') == 1.0 and
            config.get('learning_rate') == 0.001 and
            config.get('selection_metric') == 'macro_f1' and config.get('seed') == seed,
            'FREEZE_CONDITION_MISMATCH')
    scheduler = config.get('lr_scheduler')
    if CANDIDATES[model_name]['scheduled']:
        require(type(scheduler) is dict and scheduler.get('type') == 'ReduceLROnPlateau',
                'FREEZE_CONDITION_MISMATCH')
    else:
        require(scheduler is None, 'FREEZE_CONDITION_MISMATCH')


def _verify_frozen_run(repo_root, model_name, entry, train_counts, current_code):
    """실행 폴더 하나를 동결 기록·현재 추론 코드와 대조한다."""
    seed = entry['seed']
    run_path = CANDIDATES[model_name]['run_path'].format(seed=seed)
    require(entry.get('run_path') == run_path, 'FREEZE_PATH_MISMATCH')
    run_dir = (repo_root / run_path).resolve(strict=True)
    require(run_dir.is_relative_to(repo_root), 'FREEZE_PATH_MISMATCH')

    artifacts = entry.get('artifacts_sha256')
    require(type(artifacts) is dict and set(artifacts) == set(RUN_FILES), 'FREEZE_ARTIFACT_MISMATCH')
    for name in RUN_FILES:
        require((run_dir / name).is_file() and file_digest(run_dir / name) == artifacts[name],
                'FREEZE_ARTIFACT_MISMATCH')

    contract = entry.get('contract')
    require(type(contract) is dict, 'FREEZE_CONTRACT_MISMATCH')
    require(verify_run(run_dir, model_name, AUGMENTATION, seed) == contract,
            'FREEZE_CONTRACT_MISMATCH')
    _verify_condition(contract, model_name, seed)

    frozen_code = contract['code_fingerprint']['files']
    require(all(frozen_code.get(name) == current_code[name] for name in INFERENCE_CODE_FILES),
            'FREEZE_INFERENCE_CODE_MISMATCH')

    checkpoint_file = run_dir / 'best_model.pt'
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, SimpleNamespace(
        protocol=contract['split_protocol_id'],
        source_split_manifest_sha256=contract['source_split_manifest_sha256'],
        transform_config=entry.get('transform_config'),
    ), train_counts)
    require(checkpoint['model_name'] == model_name and
            checkpoint['model_config'] == entry.get('model_config') and
            checkpoint['class_mapping'] == entry.get('class_mapping') and
            checkpoint['training_config'] == contract['training_config'] and
            checkpoint['selected_epoch'] == entry.get('selected_epoch') and
            checkpoint['selection_metric'] == 'macro_f1', 'FREEZE_CHECKPOINT_MISMATCH')

    expected = class_weights(train_counts, 10.0, 0.75, 1.0).tolist()
    recorded = checkpoint.get('class_weights', {}).get('weights')
    require(type(recorded) is dict and list(recorded) == list(CLASS_NAMES) and
            all(math.isclose(recorded[name], value, rel_tol=1e-6)
                for name, value in zip(CLASS_NAMES, expected)), 'FREEZE_TRAIN_FREQUENCY_MISMATCH')

    try:
        model = build_model(model_name, num_classes=len(CLASS_NAMES))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError:
        require(False, 'FREEZE_CHECKPOINT_MISMATCH')
    _require_finite_model_weights(model)

    row = read_run(run_dir)
    require(row['selected_epoch'] == entry['selected_epoch'] and
            row['macro_f1'] == entry.get('validation_macro_f1'), 'FREEZE_VALIDATION_MISMATCH')

    return {
        'model_name': model_name,
        'role': CANDIDATES[model_name]['role'],
        'seed': seed,
        'run_path': run_path,
        'checkpoint': checkpoint_file,
        'checkpoint_sha256': artifacts['best_model.pt'],
        'selected_epoch': entry['selected_epoch'],
        'learning_rate_schedule': (
            'ReduceLROnPlateau' if CANDIDATES[model_name]['scheduled'] else 'fixed'
        ),
        'training_only_code_differences': sorted(
            name for name, value in frozen_code.items() if current_code.get(name) != value
        ),
    }


def verify_freeze(freeze, repo_root, train_counts):
    """6개 실행과 후보 사이의 공통 입력·Validation 요약을 확인한다."""
    root = Path(repo_root).resolve(strict=True)
    current_code = code_fingerprint()['files']
    runs, entries = [], []
    for candidate in freeze['candidates']:
        items = candidate['runs']
        for entry in items:
            runs.append(_verify_frozen_run(
                root, candidate['model_name'], entry, train_counts, current_code
            ))
            entries.append(entry)
        scores = [entry['validation_macro_f1'] for entry in items]
        require(_close(statistics.fmean(scores), candidate.get('validation_macro_f1_mean')) and
                _close(statistics.stdev(scores), candidate.get('validation_macro_f1_std')),
                'FREEZE_VALIDATION_MISMATCH')

    for key in ('split_protocol_id', 'source_split_manifest_sha256', 'cache_sha256'):
        require(len({entry['contract'][key] for entry in entries}) == 1,
                'FREEZE_INPUT_MISMATCH')
    require(all(entry['transform_config'] == entries[0]['transform_config'] and
                entry['class_mapping'] == entries[0]['class_mapping'] for entry in entries),
            'FREEZE_INPUT_MISMATCH')
    require(len({run['checkpoint_sha256'] for run in runs}) == len(runs),
            'FREEZE_DUPLICATE_CHECKPOINT')
    return runs, {
        'split_protocol_id': entries[0]['contract']['split_protocol_id'],
        'source_split_manifest_sha256': entries[0]['contract']['source_split_manifest_sha256'],
        'cache_sha256': entries[0]['contract']['cache_sha256'],
        'transform_config': entries[0]['transform_config'],
        'class_mapping': entries[0]['class_mapping'],
    }


# ==========================================
# 분할·변환·캐시 메타데이터 대조
# - 기존 index_split으로 manifest 전체 행과 Lot·맵 그룹 보호를 다시 검사
# - canonical의 배열·Test 캐시의 shard는 열지 않음
# ==========================================
def cross_split_overlaps(database):
    """지도학습 세 구간에 함께 나타나는 Lot·맵·그룹 수를 manifest에서 다시 센다."""
    result = {}
    for name, field in (('lots', 'r.lot'), ('map_hashes', 'r.hash'), ('groups', 'a.gid')):
        query = (f'SELECT COUNT(*) FROM (SELECT {field} FROM rows r JOIN assignments a '
                 "USING(position) WHERE a.split IN ('train','validation','test') "
                 f'AND {field} IS NOT NULL GROUP BY {field} '
                 'HAVING COUNT(DISTINCT a.split)>1)')
        result[name + '_in_multiple_splits'] = database.execute(query).fetchone()[0]
    return result


def verify_split_metadata(canonical_root, split_dir, progress=None):
    """분할 manifest 메타데이터를 검사하고 Test 표본 ID·중복 집계를 돌려준다."""
    if progress:
        progress('분할 manifest 전체 행 검사 중 (Test 배열은 읽지 않음)')
    with tempfile.TemporaryDirectory(prefix='wm_final_test_') as temporary:
        database = sqlite3.connect(str(Path(temporary) / 'manifest.sqlite'))
        try:
            report = index_split(canonical_root, split_dir, database)
            overlaps = cross_split_overlaps(database)
            test_samples = [
                (row['sample_id'], row['class_index'])
                for row in map(decode_json, (
                    record for (record,) in database.execute(
                        "SELECT r.record FROM rows r JOIN assignments a USING(position) "
                        "WHERE a.split='test' ORDER BY r.position"
                    )
                ))
            ]
        finally:
            database.close()
    return report, overlaps, test_samples


def verify_test_cache(cache_dir, train_cache_dir, split_report, shared, test_samples, manifest_sha256):
    """Test 캐시의 보고서·색인만 대조한다. shard 배열은 평가 때 Dataset이 검사한다."""
    root = Path(cache_dir).resolve(strict=True)
    require(root != Path(train_cache_dir).resolve(strict=True), 'TEST_CACHE_NOT_SEPARATE')
    report = read_json(root / 'cache.json')
    completion = read_json(root / 'SUCCESS.json')
    require(report.get('status') == completion.get('status') == 'cache_complete' and
            report.get('test_maps_read') is True and
            report.get('visual_review_confirmed') is True and
            set(report.get('splits', {})) == {'test'}, 'TEST_CACHE_NOT_PREPARED')
    require(report.get('split_protocol_id') == shared['split_protocol_id'] and
            completion.get('split_protocol_id') == shared['split_protocol_id'] and
            report.get('source_split_manifest_sha256') == manifest_sha256 and
            report.get('transform_config') == shared['transform_config'] and
            report.get('class_mapping') == shared['class_mapping'] and
            report.get('train_class_counts') == split_report['class_counts']['train'],
            'TEST_CACHE_CONTRACT_MISMATCH')
    info = report['splits']['test']
    require(info.get('rows') == split_report['split_counts']['test'] and
            info.get('class_counts') == split_report['class_counts']['test'],
            'TEST_CACHE_COUNT_MISMATCH')
    require(info.get('index') == 'test/index.jsonl' and
            file_digest(root / info['index']) == info.get('index_sha256'),
            'TEST_CACHE_INDEX_MISMATCH')
    with (root / info['index']).open('rb') as stream:
        indexed = [(row['sample_id'], row['class_index']) for row in map(decode_json, stream)]
    require(sorted(indexed) == sorted(test_samples), 'TEST_CACHE_SAMPLE_MISMATCH')
    return {
        'test_cache_sha256': file_digest(root / 'cache.json'),
        'test_samples_sha256': sample_digest(indexed),
        'test_rows': info['rows'],
        'test_class_counts': info['class_counts'],
    }


def preflight(freeze_dir, repo_root, split_dir, check_dir, train_cache_dir,
              transform_path, cache_path, canonical_dir=None, test_cache_dir=None,
              expected_freeze_sha256=None, progress=None):
    """Test 배열·추론 없이 평가 입력을 대조하고 단계별 소요 시간을 남긴다."""
    steps = []

    def timed(name, function):
        started = time.perf_counter()
        if progress:
            progress(name)
        result = function()
        steps.append({'step': name, 'seconds': round(time.perf_counter() - started, 3)})
        return result

    split_root = Path(split_dir).resolve(strict=True)
    check_root = Path(check_dir).resolve(strict=True)
    freeze, freeze_sha256 = timed(
        '동결 표식·해시', lambda: load_freeze(freeze_dir, expected_freeze_sha256)
    )
    train_cache = Path(train_cache_dir).resolve(strict=True)
    train_report = read_json(train_cache / 'cache.json')
    require(train_report.get('status') == 'cache_complete' and
            type(train_report.get('train_class_counts')) is list, 'TRAIN_CACHE_NOT_COMPLETE')
    runs, shared = timed('checkpoint·학습 기록 6개', lambda: verify_freeze(
        freeze, repo_root, train_report['train_class_counts']
    ))
    require(file_digest(train_cache / 'cache.json') == shared['cache_sha256'] and
            train_report.get('split_protocol_id') == shared['split_protocol_id'] and
            train_report.get('source_split_manifest_sha256') ==
            shared['source_split_manifest_sha256'], 'TRAIN_CACHE_MISMATCH')

    manifest_sha256 = timed(
        '분할 manifest 해시', lambda: file_digest(split_root / 'split_manifest.jsonl')
    )
    require(manifest_sha256 == shared['source_split_manifest_sha256'],
            'SPLIT_MANIFEST_MISMATCH')

    source = read_json(split_root / 'split.json')
    canonical_root = Path(canonical_dir or source.get('canonical_root', '')).resolve(strict=True)
    split_report, overlaps, test_samples = timed(
        '분할 metadata·Lot·맵 재계산',
        lambda: verify_split_metadata(canonical_root, split_root, progress),
    )
    require(split_report['split_protocol_id'] == shared['split_protocol_id'],
            'SPLIT_PROTOCOL_MISMATCH')
    require(split_report.get('class_mapping') == shared['class_mapping'] and
            split_report['class_counts']['train'] == train_report['train_class_counts'],
            'SPLIT_TRAIN_FREQUENCY_MISMATCH')
    require(not any(overlaps.values()), 'SPLIT_OVERLAP_DETECTED')
    require(len(test_samples) == split_report['split_counts']['test'], 'SPLIT_TEST_COUNT_MISMATCH')

    transform_config = load_transform_config(transform_path)
    load_cache_config(cache_path)
    require(transform_config == shared['transform_config'] and
            transform_config.get('augmentation') == 'none', 'TRANSFORM_CONFIG_MISMATCH')
    # 계약 대조용 호출이다. 사용자 시각 확인은 cache 명령의 옵션으로 별도로 받는다.
    validate_check(check_root, split_report, transform_config, True)

    test_cache = None
    if test_cache_dir is not None:
        test_cache = timed('Test 캐시 보고서·색인', lambda: verify_test_cache(
            test_cache_dir, train_cache, split_report, shared, test_samples, manifest_sha256
        ))

    train_counts = split_report['class_counts']['train']
    return {
        'status': 'preflight_passed',
        'test_arrays_read': False,
        'test_inference_run': False,
        'test_evaluation_approved': False,
        'freeze_sha256': freeze_sha256,
        'split_protocol_id': shared['split_protocol_id'],
        'source_split_manifest_sha256': manifest_sha256,
        'transform_config': shared['transform_config'],
        'class_names': list(CLASS_NAMES),
        'train_class_counts': train_counts,
        'majority_class_index': max(range(len(train_counts)), key=train_counts.__getitem__),
        'split_counts': split_report['split_counts'],
        'split_overlap': {
            'recomputed_from_current_manifest': True,
            **overlaps,
            'stored_overlap_check_passed': split_report['overlap_check_passed'],
        },
        'transform_check_contract': 'matched',
        'visual_review': 'user_confirmation_required_on_cache_command',
        'past_test_overlap': 'not_established',
        'current_selection_used_test': False,
        'test_cache_state': 'prepared_metadata_checked' if test_cache else 'not_checked',
        'test_cache': test_cache,
        'runs': [{key: value for key, value in run.items() if key != 'checkpoint'} for run in runs],
        'steps': steps,
    }, runs


# ==========================================
# 세션 기록과 재개 판단
# - session.json: 시작 때의 입력·코드 묶음. 재개 때 같아야 함
# - attempts.jsonl: 평가 시작·진행 행 수·종료를 추가 기록. 지표는 쓰지 않음
# - 완료 건은 다시 평가하지 않고 실패 건은 사람이 지정해야 다시 평가
# ==========================================
def _session_binding(check, device):
    return {
        'session_version': SESSION_VERSION,
        'freeze_sha256': check['freeze_sha256'],
        'split_protocol_id': check['split_protocol_id'],
        'source_split_manifest_sha256': check['source_split_manifest_sha256'],
        'transform_config': check['transform_config'],
        'class_names': check['class_names'],
        'majority_class_index': check['majority_class_index'],
        'test_cache': check['test_cache'],
        'device': device,
        'code_fingerprint': code_fingerprint(PACKAGE_DIR, SESSION_CODE_FILES),
    }


def _append_attempt(path, **event):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at': _now(), **event}, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_attempts(path):
    """시도 기록을 읽는다. 전원 차단으로 잘린 마지막 줄만 무시한다."""
    if not Path(path).is_file():
        return []
    lines = Path(path).read_bytes().splitlines()
    events = []
    for number, line in enumerate(lines):
        try:
            events.append(decode_json(line))
        except ValidationError:
            require(number == len(lines) - 1, 'FINAL_TEST_ATTEMPT_LOG_INVALID')
    return events


def _run_key(model_name, seed):
    return f'{model_name}:{seed}'


def _run_dir(root, model_name, seed):
    return Path(root) / model_name / f'seed{seed}'


def _verify_report(report, run, session):
    """평가 보고서가 동결 실행·세션의 Test·변환·클래스와 같은지 확인한다."""
    cache = session['test_cache']
    require(report.get('status') == 'test_evaluation_complete' and
            report.get('model_name') == run['model_name'] and
            report.get('checkpoint_sha256') == run['checkpoint_sha256'] and
            report.get('selected_epoch') == run['selected_epoch'], 'FINAL_TEST_RUN_MISMATCH')
    require(report.get('split_protocol_id') == session['split_protocol_id'] and
            report.get('source_split_manifest_sha256') == session['source_split_manifest_sha256'] and
            report.get('transform_config') == session['transform_config'] and
            report.get('class_names') == session['class_names'] and
            report.get('evaluated_split') == 'test' and
            report.get('test_cache_sha256') == cache['test_cache_sha256'] and
            report.get('test_samples_sha256') == cache['test_samples_sha256'] and
            report.get('test_rows') == report.get('predicted_rows') == cache['test_rows'],
            'FINAL_TEST_MIXED_TEST')
    require(report.get('test_augmentation') == 'none' and
            report.get('model_training_mode') is False and
            report.get('weights_unchanged') is True and
            report.get('test_used_for_model_selection') is False and
            report.get('majority_baseline_class_index') == session['majority_class_index'],
            'FINAL_TEST_RUN_CONTRACT_MISMATCH')


def verify_published(root, run, session):
    """공개된 실행의 해시와 보고서를 확인하고 보고서를 돌려준다."""
    final = _run_dir(root, run['model_name'], run['seed'])
    record = read_json(final / RUN_RECORD)
    artifacts = record.get('artifacts')
    names = ('test_evaluation.json', 'SUCCESS.json')
    require(record.get('status') == 'final_test_run_complete' and
            record.get('model_name') == run['model_name'] and record.get('seed') == run['seed'] and
            record.get('freeze_sha256') == session['freeze_sha256'] and
            type(artifacts) is dict and set(artifacts) == set(names), 'FINAL_TEST_RUN_INCOMPLETE')
    for name in names:
        require((final / name).is_file() and file_digest(final / name) == artifacts[name],
                'FINAL_TEST_RUN_INTEGRITY_FAILED')
    report = read_json(final / 'test_evaluation.json')
    _verify_report(report, run, session)
    return report


def session_state(root, session, runs):
    """실행별 complete·failed·interrupted·pending 상태와 읽은 행 수를 정리한다."""
    events = read_attempts(Path(root) / 'attempts.jsonl')
    state = []
    for run in runs:
        key = _run_key(run['model_name'], run['seed'])
        own = [event for event in events if event.get('run') == key]
        rows = max((event.get('rows_read', 0) for event in own), default=0)
        if _run_dir(root, run['model_name'], run['seed']).exists():
            verify_published(root, run, session)
            status = 'complete'
        elif not own:
            status = 'pending'
        elif own[-1].get('event') == 'failed':
            status = 'failed'
        else:
            status = 'interrupted'
        state.append({
            'run': key,
            'status': status,
            'attempts': sum(event.get('event') == 'started' for event in own),
            'rows_read_last_known': rows,
            'last_error': next((event.get('code') for event in reversed(own)
                                if event.get('event') == 'failed'), None),
        })
    return state


def _open_session(root, binding, resume):
    """새 세션을 만들거나 재개 조건을 확인한다."""
    path = root / 'session.json'
    if path.exists():
        require(resume is True, 'FINAL_TEST_OUTPUT_EXISTS')
        session = read_json(path)
        require(session == binding, 'FINAL_TEST_SESSION_MISMATCH')
        return session
    require(resume is not True, 'FINAL_TEST_SESSION_NOT_FOUND')
    require(not root.exists(), 'FINAL_TEST_OUTPUT_EXISTS')
    root.mkdir(parents=True)
    _write_new_json(path, binding)
    return binding


def _select_runs(state, retry):
    """이번에 평가할 실행을 고른다. 완료 건은 지정해도 다시 평가하지 않는다."""
    requested = set(retry or ())
    by_run = {item['run']: item for item in state}
    require(requested <= set(by_run), 'FINAL_TEST_RETRY_UNKNOWN_RUN')
    require(all(by_run[key]['status'] != 'complete' for key in requested),
            'FINAL_TEST_ALREADY_COMPLETE')
    require(all(by_run[key]['status'] in ('failed', 'interrupted') for key in requested),
            'FINAL_TEST_RETRY_NOT_FAILED')
    undecided = {item['run'] for item in state
                 if item['status'] in ('failed', 'interrupted')} - requested
    require(not undecided, 'FINAL_TEST_RETRY_DECISION_REQUIRED')
    return {item['run'] for item in state if item['status'] != 'complete'}


def _evaluate_run(root, run, session, test_cache_dir, evaluator):
    """체크포인트 하나를 평가해 검증된 결과만 공개한다."""
    key = _run_key(run['model_name'], run['seed'])
    log = root / 'attempts.jsonl'
    final = _run_dir(root, run['model_name'], run['seed'])
    temporary = final.with_name(final.name + '.tmp')
    final.parent.mkdir(exist_ok=True)
    if temporary.exists():
        _set_aside_incomplete(temporary, final)

    require(file_digest(run['checkpoint']) == run['checkpoint_sha256'],
            'FREEZE_ARTIFACT_MISMATCH')
    require(code_fingerprint(PACKAGE_DIR, SESSION_CODE_FILES) == session['code_fingerprint'],
            'FINAL_TEST_SESSION_MISMATCH')

    read = {'rows': 0, 'logged': 0}

    def track(done, total):
        read['rows'] = done
        # 지표 없이 약 25% 단위로만 기록해 어디까지 Test를 읽었는지 남긴다.
        if done == total or done - read['logged'] >= max(total // 4, 1):
            read['logged'] = done
            _append_attempt(log, run=key, event='progress', rows_read=done, rows_total=total)
            print(f'{key}: {done}/{total}행 예측', flush=True)

    _append_attempt(log, run=key, event='started', rows_read=0)
    try:
        evaluator(test_cache_dir, run['checkpoint'], temporary, session['device'], True, track)
        report = read_json(temporary / 'test_evaluation.json')
        _verify_report(report, run, session)
        require(file_digest(run['checkpoint']) == run['checkpoint_sha256'],
                'FREEZE_ARTIFACT_MISMATCH')
        _write_new_json(temporary / RUN_RECORD, {
            'status': 'final_test_run_complete',
            'model_name': run['model_name'],
            'role': run['role'],
            'seed': run['seed'],
            'run_path': run['run_path'],
            'checkpoint_sha256': run['checkpoint_sha256'],
            'freeze_sha256': session['freeze_sha256'],
            'artifacts': {name: file_digest(temporary / name)
                          for name in ('test_evaluation.json', 'SUCCESS.json')},
            'scope_note': SCOPE_NOTE,
            'selection_scope_note': SELECTION_SCOPE_NOTE,
        })
        temporary.rename(final)
    except BaseException as error:
        code = str(error) if isinstance(error, ValidationError) else type(error).__name__
        _append_attempt(log, run=key, event='failed', code=code, rows_read=read['rows'])
        if temporary.exists():
            _set_aside_incomplete(temporary, final)
        if not isinstance(error, Exception):
            raise
        raise FinalTestStopped(key) from error
    _append_attempt(log, run=key, event='complete', rows_read=read['rows'])


def evaluate_session(check, runs, test_cache_dir, output_root, device, approved,
                     resume=False, retry=None, evaluator=evaluate_checkpoint):
    """사전점검을 통과한 6개 checkpoint를 순서대로 평가한다. 실패하면 멈춘다."""
    require(approved is True, 'FINAL_TEST_EVALUATION_CONFIRMATION_REQUIRED')
    require(device in ('cpu', 'cuda'), 'INVALID_EVALUATION_DEVICE')
    require(check.get('status') == 'preflight_passed' and check.get('test_cache') is not None,
            'FINAL_TEST_PREFLIGHT_REQUIRED')
    require(device == 'cpu' or torch.cuda.is_available(), 'CUDA_DEVICE_UNAVAILABLE')
    root = Path(output_root).resolve()
    session = _open_session(root, _session_binding(check, device), resume)
    state = session_state(root, session, runs)
    selected = _select_runs(state, retry)

    stopped = None
    for run in runs:
        if _run_key(run['model_name'], run['seed']) not in selected:
            continue
        try:
            _evaluate_run(root, run, session, test_cache_dir, evaluator)
        except FinalTestStopped as error:
            stopped = str(error)
            break
    final_state = session_state(root, session, runs)
    return {
        'status': 'final_test_runs_complete' if stopped is None else 'final_test_stopped',
        'stopped_at': stopped,
        'runs': final_state,
    }


def read_status(freeze_dir, repo_root, train_cache_dir, output_root):
    """Test를 읽지 않고 세션의 완료·실패·남은 실행과 재개 방법을 보여 준다."""
    root = Path(output_root).resolve(strict=True)
    session = read_json(root / 'session.json')
    freeze, _ = load_freeze(freeze_dir, session.get('freeze_sha256'))
    train_cache = Path(train_cache_dir).resolve(strict=True)
    train_counts = read_json(train_cache / 'cache.json')['train_class_counts']
    runs, _ = verify_freeze(freeze, repo_root, train_counts)
    state = session_state(root, session, runs)
    blocked = [item['run'] for item in state if item['status'] in ('failed', 'interrupted')]
    pending = [item['run'] for item in state if item['status'] == 'pending']
    return {
        'status': 'final_test_status',
        'test_arrays_read': False,
        'runs': state,
        'resume_plan': {
            'skip_complete': [item['run'] for item in state if item['status'] == 'complete'],
            'requires_explicit_retry': blocked,
            'pending': pending,
            'flags': (['--resume'] + [f'--retry {key}' for key in blocked]
                      if blocked or pending else []),
        },
    }


# ==========================================
# 결과 집계
# - 정확히 모델 2개 × seed 3개이고 같은 Test·프로토콜·변환·클래스인지 확인
# - 저장된 혼동행렬로 지표를 다시 계산해 대조하고 다른 폴더의 혼입을 거부
# - 표는 동결 역할 순서이며 Test 성능으로 모델·seed를 고르지 않음
# ==========================================
def _reject_extra_runs(root):
    """동결 후보 외 결과가 세션 폴더에 섞이면 거부한다."""
    allowed = {'session.json', 'attempts.jsonl', COMPARISON_DIR, *MODELS}
    for path in root.iterdir():
        require(path.name in allowed or path.name.startswith(COMPARISON_DIR + '.'),
                'FINAL_TEST_EXTRA_RUN')
    for model_name in MODELS:
        expected = {f'seed{seed}' for seed in SEEDS}
        for path in (root / model_name).iterdir():
            require(path.name in expected or '.incomplete_' in path.name, 'FINAL_TEST_EXTRA_RUN')


def _verify_metrics(report, test_counts):
    """저장 혼동행렬에서 지표를 다시 계산해 보고서·support와 대조한다."""
    for name in ('frozen_model', 'majority_baseline'):
        stored = report[name]
        confusion = torch.tensor(stored['confusion_matrix'], dtype=torch.int64)
        require(int(confusion.sum()) == report['test_rows'] and
                confusion.sum(dim=1).tolist() == test_counts, 'FINAL_TEST_CONFUSION_MISMATCH')
        computed = classification_metrics(confusion)
        require(all(_close(computed[key], stored[key]) for key in COMPARISON_METRICS) and
                len(stored['per_class']) == len(CLASS_NAMES) and
                all(left[key] == right[key] or _close(left[key], right[key])
                    for left, right in zip(computed['per_class'], stored['per_class'])
                    for key in ('support', 'precision', 'recall', 'f1')),
                'FINAL_TEST_METRIC_MISMATCH')


def _class_rows(model_name, seed, metrics):
    """클래스별 지표와 혼동행렬에서 얻은 FP·FN을 한 행씩 만든다."""
    matrix = metrics['confusion_matrix']
    rows = []
    for index, item in enumerate(metrics['per_class']):
        true_positive = matrix[index][index]
        rows.append({
            'model_name': model_name,
            'seed': seed,
            'class_name': item['pattern_label'],
            'support': item['support'],
            'precision': item['precision'],
            'recall': item['recall'],
            'f1': item['f1'],
            'true_positive': true_positive,
            'false_positive': sum(row[index] for row in matrix) - true_positive,
            'false_negative': sum(matrix[index]) - true_positive,
        })
    return rows


def _build_tables(reports):
    """검증된 보고서에서 표에 쓸 행을 만든다. 순서는 동결 역할 순서다."""
    runs_rows, class_rows, confusion_rows = [], [], []
    for (model_name, seed), report in reports.items():
        metrics = report['frozen_model']
        runs_rows.append({
            'model_name': model_name,
            'role': CANDIDATES[model_name]['role'],
            'seed': seed,
            'selected_epoch': report['selected_epoch'],
            **{key: metrics[key] for key in COMPARISON_METRICS},
            'majority_macro_f1': report['majority_baseline']['macro_f1'],
            'majority_accuracy': report['majority_baseline']['accuracy'],
            'device': report['device'],
            'batch_size': report['batch_size'],
            'test_rows': report['test_rows'],
            'inference_seconds': report['inference_seconds'],
            'checkpoint_sha256': report['checkpoint_sha256'],
        })
        class_rows += _class_rows(model_name, seed, metrics)
        for true_index, row in enumerate(metrics['confusion_matrix']):
            for predicted_index, count in enumerate(row):
                confusion_rows.append({
                    'model_name': model_name, 'seed': seed,
                    'true_class': CLASS_NAMES[true_index],
                    'predicted_class': CLASS_NAMES[predicted_index], 'count': count,
                })

    summary_rows = []
    for model_name in MODELS:
        group = [row for row in runs_rows if row['model_name'] == model_name]
        item = {'model_name': model_name, 'role': CANDIDATES[model_name]['role'],
                'scope': 'all', 'seeds': len(group)}
        for key in COMPARISON_METRICS:
            result = _mean_std([row[key] for row in group])
            item[key + '_mean'], item[key + '_std'] = result['mean'], result['std']
        summary_rows.append(item)
        for name in CLASS_NAMES:
            rows = [row for row in class_rows
                    if row['model_name'] == model_name and row['class_name'] == name]
            entry = {'model_name': model_name, 'role': CANDIDATES[model_name]['role'],
                     'scope': name, 'seeds': len(rows), 'support': rows[0]['support']}
            for key in ('precision', 'recall', 'f1', 'false_positive', 'false_negative'):
                result = _mean_std([row[key] for row in rows])
                entry[key + '_mean'], entry[key + '_std'] = result['mean'], result['std']
            summary_rows.append(entry)

    difference_rows = []
    primary, comparison = MODELS
    for seed in SEEDS:
        left = reports[primary, seed]['frozen_model']
        right = reports[comparison, seed]['frozen_model']
        row = {'seed': seed, 'difference': f'{primary} - {comparison}'}
        for key in COMPARISON_METRICS:
            row[key] = left[key] - right[key]
        for one, two in zip(left['per_class'], right['per_class']):
            for key in ('precision', 'recall', 'f1'):
                row[f"{one['pattern_label']}_{key}"] = one[key] - two[key]
        difference_rows.append(row)
    return runs_rows, summary_rows, class_rows, difference_rows, confusion_rows


def _percent(value):
    return f'{value * 100:.2f}'


def _write_csv_file(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison(path, reports, summary_rows, runs_rows, difference_rows):
    """해석 범위를 앞에 두고 평균·seed별·클래스별·혼동행렬 표를 쓴다."""
    lines = [
        '# WM-811K 동결 후보 최종 Test 결과', '',
        SCOPE_NOTE, SELECTION_SCOPE_NOTE, '',
        '- ResNet18은 고정 학습률, SmallWaferCNN은 ReduceLROnPlateau입니다. '
        '튜닝 구성 비교이며 구조만의 효과가 아닙니다.',
        '- 종합후보 지정은 Test 이전의 사용자 결정입니다. 이 결과로 모델·설정·seed를 다시 고르지 않습니다.',
        '- 같은 Test 표본을 seed별로 반복 평가했으므로 독립 표본이 늘어난 것이 아닙니다. '
        '평균 차이를 통계적 우월성으로 해석하지 않습니다.', '',
        '## 모델별 평균 ± 표본표준편차 (seed 3개, %)', '',
        '| 모델 | 역할 | Macro-F1 | Accuracy | Macro-Recall | Macro-Precision |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for item in summary_rows:
        if item['scope'] == 'all':
            cells = [f"{_percent(item[key + '_mean'])} ± {_percent(item[key + '_std'])}"
                     for key in ('macro_f1', 'accuracy', 'macro_recall', 'macro_precision')]
            lines.append(f"| {item['model_name']} | {item['role']} | " + ' | '.join(cells) + ' |')

    lines += ['', '## seed별 결과 (%)', '',
              '| 모델 | seed | Macro-F1 | Accuracy | 다수 클래스 기준 Macro-F1 | 다수 클래스 기준 Accuracy |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in runs_rows:
        lines.append(f"| {row['model_name']} | {row['seed']} | {_percent(row['macro_f1'])} "
                     f"| {_percent(row['accuracy'])} | {_percent(row['majority_macro_f1'])} "
                     f"| {_percent(row['majority_accuracy'])} |")

    lines += ['', '## 같은 seed 차이 (%p, ResNet18 − SmallWaferCNN)', '',
              '| seed | Macro-F1 | Accuracy | Macro-Recall |', '| ---: | ---: | ---: | ---: |']
    for row in difference_rows:
        lines.append(f"| {row['seed']} | {row['macro_f1'] * 100:+.2f} "
                     f"| {row['accuracy'] * 100:+.2f} | {row['macro_recall'] * 100:+.2f} |")

    lines += ['', '## 클래스별 평균 ± 표본표준편차', '',
              '| 모델 | 클래스 | support | Precision (%) | Recall (%) | F1 (%) | FP | FN |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in summary_rows:
        if item['scope'] != 'all':
            cells = [f"{_percent(item[key + '_mean'])} ± {_percent(item[key + '_std'])}"
                     for key in ('precision', 'recall', 'f1')]
            counts = [f"{item[key + '_mean']:.1f} ± {item[key + '_std']:.1f}"
                      for key in ('false_positive', 'false_negative')]
            lines.append(f"| {item['model_name']} | {item['scope']} | {item['support']} | "
                         + ' | '.join(cells + counts) + ' |')

    lines += ['', '## seed별 혼동행렬', '', '행은 정답, 열은 예측입니다.']
    for (model_name, seed), report in reports.items():
        lines += ['', f'### {model_name} seed {seed}', '',
                  '| 정답\\예측 | ' + ' | '.join(CLASS_NAMES) + ' |',
                  '| --- |' + ' ---: |' * len(CLASS_NAMES)]
        for name, row in zip(CLASS_NAMES, report['frozen_model']['confusion_matrix']):
            lines.append(f'| {name} | ' + ' | '.join(map(str, row)) + ' |')
    lines.append('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def summarize(freeze_dir, repo_root, train_cache_dir, output_root, report_dir=None):
    """6개 결과를 모두 검증한 뒤 비교표를 임시 폴더에 쓰고 한 번에 공개한다."""
    root = Path(output_root).resolve(strict=True)
    session = read_json(root / 'session.json')
    require(session.get('session_version') == SESSION_VERSION, 'FINAL_TEST_SESSION_MISMATCH')
    freeze, freeze_sha256 = load_freeze(freeze_dir, session.get('freeze_sha256'))
    train_cache = Path(train_cache_dir).resolve(strict=True)
    train_counts = read_json(train_cache / 'cache.json')['train_class_counts']
    runs, _ = verify_freeze(freeze, repo_root, train_counts)
    require(all((root / model_name).is_dir() for model_name in MODELS), 'FINAL_TEST_RUN_MISSING')
    _reject_extra_runs(root)

    reports = {}
    for run in runs:
        key = (run['model_name'], run['seed'])
        require(_run_dir(root, *key).is_dir(), 'FINAL_TEST_RUN_MISSING')
        report = verify_published(root, run, session)
        _verify_metrics(report, session['test_cache']['test_class_counts'])
        reports[key] = report
    require(len(reports) == len(MODELS) * len(SEEDS) and
            len({report['checkpoint_sha256'] for report in reports.values()}) == len(reports),
            'FINAL_TEST_DUPLICATE_CHECKPOINT')
    require(len({report['test_samples_sha256'] for report in reports.values()}) == 1,
            'FINAL_TEST_MIXED_TEST')

    report_root = Path(report_dir).resolve() if report_dir else root / COMPARISON_DIR
    require(not report_root.exists(), 'FINAL_TEST_REPORT_ALREADY_EXISTS')
    temporary = report_root.with_name(report_root.name + '.tmp')
    if temporary.exists():
        _set_aside_incomplete(temporary, report_root)
    temporary.mkdir(parents=True)

    runs_rows, summary_rows, class_rows, difference_rows, confusion_rows = _build_tables(reports)
    _write_csv_file(temporary / 'runs.csv', runs_rows)
    _write_csv_file(temporary / 'summary.csv', summary_rows)
    _write_csv_file(temporary / 'per_class.csv', class_rows)
    _write_csv_file(temporary / 'seed_differences.csv', difference_rows)
    _write_csv_file(temporary / 'confusion_matrices.csv', confusion_rows)
    _write_comparison(temporary / 'comparison.md', reports, summary_rows, runs_rows, difference_rows)
    _write_new_json(temporary / 'comparison.json', {
        'status': 'final_test_comparison_complete',
        'freeze_sha256': freeze_sha256,
        'split_protocol_id': session['split_protocol_id'],
        'test_samples_sha256': session['test_cache']['test_samples_sha256'],
        'test_rows': session['test_cache']['test_rows'],
        'checkpoint_count': len(reports),
        'selection_changed_by_test': False,
        'test_used_for_model_selection': False,
        'scope_note': SCOPE_NOTE,
        'selection_scope_note': SELECTION_SCOPE_NOTE,
        'summary': summary_rows,
        'runs': runs_rows,
        'seed_differences': difference_rows,
    })
    names = sorted(path.name for path in temporary.iterdir())
    _write_new_json(temporary / 'SUCCESS.json', {
        'status': 'final_test_comparison_complete',
        'artifacts': {name: file_digest(temporary / name) for name in names},
    })
    temporary.rename(report_root)
    return {'status': 'final_test_comparison_complete', 'report_dir': str(report_root),
            'checkpoint_count': len(reports)}


def _add_input_arguments(parser):
    parser.add_argument('--freeze-dir', default='reports/wm811k/final_freeze_v1')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--train-cache-dir',
                        default='data/processed/wm811k/input_cache/wm_resize_v1')


def _add_check_arguments(parser):
    _add_input_arguments(parser)
    parser.add_argument('--split-dir', default='data/splits/wm811k/group_v1')
    parser.add_argument('--check-dir', default='reports/wm811k/transform_v1')
    parser.add_argument('--transform-config', default='configs/wm811k/transform.json')
    parser.add_argument('--cache-config', default='configs/wm811k/cache.json')
    parser.add_argument('--canonical-dir')


def _run_preflight(args, test_cache_dir):
    return preflight(
        args.freeze_dir, args.repo_root, args.split_dir, args.check_dir,
        args.train_cache_dir, args.transform_config, args.cache_config,
        args.canonical_dir, test_cache_dir, args.expected_freeze_sha256,
        progress=lambda text: print(f'사전점검: {text}', flush=True),
    )


def main():
    parser = argparse.ArgumentParser(description='WM-811K 동결 후보 최종 Test 사전점검·평가·집계')
    commands = parser.add_subparsers(dest='command', required=True)

    check = commands.add_parser('preflight', help='Test 배열·추론 없이 메타데이터 대조')
    _add_check_arguments(check)
    check.add_argument('--test-cache-dir', help='준비된 Test 캐시의 보고서·색인만 대조')
    check.add_argument('--expected-freeze-sha256')

    run = commands.add_parser('evaluate', help='명시적 확인 후 6개 checkpoint 평가')
    _add_check_arguments(run)
    run.add_argument('--test-cache-dir', required=True)
    run.add_argument('--expected-freeze-sha256', required=True)
    run.add_argument('--output-root', default='reports/wm811k/final_test_v1')
    run.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    run.add_argument('--resume', action='store_true', help='완료 건을 건너뛰고 같은 세션을 이어 감')
    run.add_argument('--retry', action='append', default=[],
                     help='실패·중단 건을 모델:seed로 지정. 여러 번 사용 가능')
    run.add_argument('--confirm-final-test-evaluation', action='store_true')

    status = commands.add_parser('status', help='Test를 읽지 않고 세션 상태·재개 계획 표시')
    _add_input_arguments(status)
    status.add_argument('--output-root', default='reports/wm811k/final_test_v1')

    summary = commands.add_parser('summarize', help='완료된 6개 결과를 검증해 집계')
    _add_input_arguments(summary)
    summary.add_argument('--output-root', default='reports/wm811k/final_test_v1')
    summary.add_argument('--report-dir')

    args = parser.parse_args()
    try:
        with validation_stage('final_test'):
            if args.command == 'preflight':
                result, _ = _run_preflight(args, args.test_cache_dir)
            elif args.command == 'evaluate':
                require(args.confirm_final_test_evaluation is True,
                        'FINAL_TEST_EVALUATION_CONFIRMATION_REQUIRED')
                check_result, runs = _run_preflight(args, args.test_cache_dir)
                result = evaluate_session(
                    check_result, runs, args.test_cache_dir, args.output_root, args.device,
                    args.confirm_final_test_evaluation, args.resume, args.retry,
                )
            elif args.command == 'status':
                result = read_status(args.freeze_dir, args.repo_root,
                                     args.train_cache_dir, args.output_root)
            else:
                result = summarize(args.freeze_dir, args.repo_root, args.train_cache_dir,
                                   args.output_root, args.report_dir)
    except ValueError as error:
        print('최종 Test 실패 코드: ' + str(error))
        if str(error) in FAILURE_GUIDANCE:
            print(FAILURE_GUIDANCE[str(error)])
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get('status') == 'final_test_stopped' else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""D4 모델의 가중치 후보를 같은 모델 기준 실행과 같은 seed끼리 읽기 전용으로 대조한다.

학습·추론·Test 평가를 하지 않고 저장된 selected_epoch 혼동행렬만 읽는다.
후보 결과를 본 Validation 비교이므로 일반화 성능이 아니며 최종 모델을 자동으로
고르지 않는다. 기존 compare의 혼합 집계 거부는 그대로 두고, 가중치 설정만 비교
축으로 허용하는 이 경로에서 나머지 조건의 일치를 확인한다.
"""

import argparse
import json
from pathlib import Path
import statistics

import torch

from .cache import file_digest
from .compare import (
    DEFAULT_SEEDS,
    _mean_std,
    _set_aside_incomplete,
    _write_csv,
    run_name,
    verify_run,
)
from .train import OPTIONAL_TRAIN_CONFIG, class_weights
from .validate_canonical import CLASS_NAMES, read_json, require, validation_stage


DEFAULT_MODEL = 'ResNet18'
SUPPORTED_MODELS = ('ResNet18', 'SmallWaferCNN')
AUGMENTATION = 'd4'

WEIGHT_KEYS = ('max_class_weight', 'class_weight_alpha', 'scratch_weight_factor')
OVERALL = 'overall'
PERCENT_METRICS = ('macro_f1', 'accuracy', 'precision', 'recall', 'f1')
CRITERIA_KEYS = {
    'criteria_version',
    'status',
    'macro_f1_mean_difference_min_pp',
    'scratch_precision_mean_difference_min_pp',
    'scratch_precision_improved_seeds_min',
    'scratch_recall_mean_difference_min_pp',
    'random_recall_mean_difference_min_pp',
    'near_full_fn_increase_sum_max',
    'donut_fn_increase_sum_max',
}

# 판정 참고안의 경계값 비교에서 부동소수점 오차를 없애는 자릿수.
BOUNDARY_DIGITS = 9

SELECTION_STATUS = '선정 보류'
FAILURE_GUIDANCE = {
    'WEIGHT_COMPARE_REPORT_ALREADY_EXISTS': (
        '이전 대조를 덮어쓰지 않습니다. --report-dir에 새 경로를 지정하세요.'
    ),
    'WEIGHT_COMPARE_PARTIAL_EXISTS': (
        '끝나지 않은 .tmp 폴더가 있습니다. 지우지 말고 --report-dir에 새 경로를 지정하세요.'
    ),
    'WEIGHT_COMPARE_CONDITION_MISMATCH': (
        '가중치 설정 외의 조건이 다릅니다. 후보를 같은 캐시·설정으로 새 경로에 다시 학습하세요.'
    ),
    'WEIGHT_COMPARE_CODE_MIXED': (
        '한 조건 안에 서로 다른 코드의 실행이 섞였습니다. 새 --output-root에 같은 코드로 '
        '3회를 모두 실행하세요.'
    ),
    'WEIGHT_COMPARE_NO_WEIGHT_DIFFERENCE': (
        '후보의 가중치 설정이 기준과 같습니다. --candidate-root를 확인하세요.'
    ),
    'WEIGHT_COMPARE_RECORD_MISMATCH': (
        '기록된 클래스 가중치가 설정·캐시로 계산한 값과 다릅니다. 해당 실행을 비교에 쓰지 마세요.'
    ),
    'WEIGHT_COMPARE_CACHE_MISMATCH': (
        '학습에 쓴 캐시가 아닙니다. 해당 실행이 사용한 --cache-dir을 지정하세요.'
    ),
    'WEIGHT_COMPARE_MODEL_UNSUPPORTED': (
        '가중치 대조는 ResNet18과 SmallWaferCNN만 지원합니다. --model을 확인하세요.'
    ),
    'INVALID_WEIGHT_CRITERIA': (
        '판정 참고안 파일이 약속된 항목과 다릅니다. 결과를 본 뒤 고쳤다면 사후 변경으로 기록하세요.'
    ),
    'COMPARE_RUN_INCOMPLETE': (
        '완료되지 않았거나 없는 실행입니다. 해당 조건의 compare run을 끝낸 뒤 다시 실행하세요.'
    ),
    'COMPARE_RUN_CONTRACT_MISMATCH': (
        '실행 폴더의 모델·증강·seed 기록이 요청과 다릅니다. 다른 조건의 폴더를 쓰지 말고 '
        '--model, --candidate-root, --baseline-root를 확인하세요.'
    ),
    'COMPARE_RUN_INTEGRITY_FAILED': (
        '완료 기록과 저장 파일이 일치하지 않습니다. 해당 실행을 대조에 쓰지 마세요.'
    ),
}


def weight_settings(contract):
    """실행 계약의 가중치 설정을 읽는다. 이전 실행의 없는 키는 기본값이다."""
    config = contract['training_config']
    return {
        key: config.get(key, OPTIONAL_TRAIN_CONFIG.get(key))
        for key in WEIGHT_KEYS
    }


def _common_condition(contract):
    """가중치 설정·코드·seed를 뺀, 기준과 후보가 같아야 하는 조건을 남긴다."""
    common = {
        key: value for key, value in contract.items()
        if key not in ('code_fingerprint', 'seed')
    }
    common['training_config'] = {
        key: value for key, value in contract['training_config'].items()
        if key != 'seed' and key not in WEIGHT_KEYS
    }
    return common


def load_condition(root, seeds, model_name=DEFAULT_MODEL):
    """선택 모델의 한 조건을 검증하고 seed별 계약·저장 혼동행렬을 읽는다."""
    runs = {}
    for seed in seeds:
        run_dir = Path(root) / run_name(model_name, AUGMENTATION, seed)
        contract = verify_run(run_dir, model_name, AUGMENTATION, seed)

        # 코드가 다른 과거 실행도 읽을 수 있도록 모델을 만들지 않고 저장 값만 쓴다.
        checkpoint = torch.load(
            run_dir / 'best_model.pt',
            map_location='cpu',
            weights_only=True,
        )
        report = read_json(run_dir / 'training_report.json')
        selected = [
            epoch for epoch in report['history']
            if epoch['epoch'] == checkpoint['selected_epoch']
        ]
        require(
            len(selected) == 1 and
            selected[0]['validation']['confusion_matrix'] ==
            checkpoint['validation_metrics']['confusion_matrix'],
            'COMPARE_RUN_INTEGRITY_FAILED',
        )

        runs[seed] = {
            'contract': contract,
            'selected_epoch': checkpoint['selected_epoch'],
            'transform_config': checkpoint['transform_config'],
            'class_mapping': checkpoint['class_mapping'],
            'class_weights': report.get('class_weights'),
            'lr_schedule': checkpoint.get('lr_schedule'),
            'report': report,
            'confusion_matrix': selected[0]['validation']['confusion_matrix'],
            'digests': {
                name: file_digest(run_dir / name)
                for name in ('best_model.pt', 'training_report.json', 'SUCCESS.json')
            },
        }

    return runs


def require_comparable(baseline, candidate):
    """가중치와 명시적인 코드 차이 외의 조건이 같은지 확인하고 코드 차이를 돌려준다."""
    for runs in (baseline, candidate):
        require(
            len({
                run['contract']['code_fingerprint']['sha256'] for run in runs.values()
            }) == 1,
            'WEIGHT_COMPARE_CODE_MIXED',
        )
        require(
            len({
                json.dumps(weight_settings(run['contract']), sort_keys=True)
                for run in runs.values()
            }) == 1,
            'WEIGHT_COMPARE_CONDITION_MISMATCH',
        )

    reference = next(iter(baseline.values()))
    expected = _common_condition(reference['contract'])
    for runs in (baseline, candidate):
        for run in runs.values():
            require(
                _common_condition(run['contract']) == expected and
                run['transform_config'] == reference['transform_config'] and
                run['class_mapping'] == reference['class_mapping'],
                'WEIGHT_COMPARE_CONDITION_MISMATCH',
            )

    first = next(iter(candidate.values()))
    require(
        weight_settings(first['contract']) != weight_settings(reference['contract']),
        'WEIGHT_COMPARE_NO_WEIGHT_DIFFERENCE',
    )

    old_code = reference['contract']['code_fingerprint']
    new_code = first['contract']['code_fingerprint']
    names = sorted(set(old_code['files']) | set(new_code['files']))
    return {
        'baseline_code_sha256': old_code['sha256'],
        'candidate_code_sha256': new_code['sha256'],
        'same_code': old_code['sha256'] == new_code['sha256'],
        'differing_code_files': [
            name for name in names
            if old_code['files'].get(name) != new_code['files'].get(name)
        ],
    }


def compute_weights(cache_report, contract):
    """Train 빈도와 설정으로 9클래스 가중치를 계산해 이름별로 돌려준다."""
    settings = weight_settings(contract)
    weights = class_weights(
        cache_report['train_class_counts'],
        settings['max_class_weight'],
        settings['class_weight_alpha'],
        settings['scratch_weight_factor'],
    )
    return dict(zip(CLASS_NAMES, weights.tolist()))


def require_recorded_weights(candidate, expected):
    """후보 보고서의 실제 가중치가 설정으로 계산한 값과 같은지 확인한다."""
    for run in candidate.values():
        record = run['class_weights']
        require(
            type(record) is dict and record.get('weights') == expected,
            'WEIGHT_COMPARE_RECORD_MISMATCH',
        )


def matrix_metrics(matrix):
    """저장된 혼동행렬에서 클래스별 지표와 FP·FN을 계산한다."""
    size = len(CLASS_NAMES)
    require(
        type(matrix) is list and len(matrix) == size and
        all(type(row) is list and len(row) == size for row in matrix),
        'INVALID_CONFUSION_MATRIX',
    )

    classes = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        actual = sum(matrix[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        total = precision + recall
        classes[name] = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / total if total else 0.0,
            'fp': predicted - true_positive,
            'fn': actual - true_positive,
        }

    samples = sum(sum(row) for row in matrix)
    require(samples > 0, 'EMPTY_CONFUSION_MATRIX')
    return {
        OVERALL: {
            'macro_f1': statistics.fmean(item['f1'] for item in classes.values()),
            'accuracy': sum(matrix[i][i] for i in range(size)) / samples,
        },
        **classes,
    }


def paired_rows(baseline, candidate, seeds):
    """seed·범위·지표마다 기준값, 후보값, 동일 seed 차이를 한 행으로 만든다."""
    rows = []
    for seed in seeds:
        before = matrix_metrics(baseline[seed]['confusion_matrix'])
        after = matrix_metrics(candidate[seed]['confusion_matrix'])
        for scope, metrics in before.items():
            for metric, value in metrics.items():
                rows.append({
                    'seed': seed,
                    'scope': scope,
                    'metric': metric,
                    'baseline': value,
                    'candidate': after[scope][metric],
                    'difference': after[scope][metric] - value,
                })
    return rows


def summarize_rows(rows):
    """범위·지표별 3seed 평균·표준편차와 동일 seed 차이를 계산한다."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row['scope'], row['metric']), []).append(row)

    summary = []
    for (scope, metric), items in grouped.items():
        before = _mean_std([item['baseline'] for item in items])
        after = _mean_std([item['candidate'] for item in items])
        change = _mean_std([item['difference'] for item in items])
        summary.append({
            'scope': scope,
            'metric': metric,
            'baseline_mean': before['mean'],
            'baseline_std': before['std'],
            'candidate_mean': after['mean'],
            'candidate_std': after['std'],
            'difference_mean': change['mean'],
            'difference_std': change['std'],
            'difference_sum': sum(item['difference'] for item in items),
            'worse_seeds': _worse_seeds(metric, items),
        })
    return summary


def _worse_seeds(metric, items):
    """개별 seed 악화를 찾는다. FP·FN은 늘수록, 나머지는 줄수록 악화다."""
    worse = [
        item['seed'] for item in items
        if (item['difference'] > 0 if metric in ('fp', 'fn') else item['difference'] < 0)
    ]
    return ' '.join(str(seed) for seed in worse)


def load_criteria(path):
    """실행 전에 준비한 판정 참고안을 읽고 식별값과 함께 돌려준다."""
    path = Path(path).resolve(strict=True)
    criteria = read_json(path)
    require(
        set(criteria) == CRITERIA_KEYS and
        all(
            type(value) in (int, float) and not isinstance(value, bool)
            for key, value in criteria.items()
            if key not in ('criteria_version', 'status')
        ),
        'INVALID_WEIGHT_CRITERIA',
    )
    return criteria, file_digest(path)


def evaluate_criteria(summary, rows, criteria):
    """참고안 항목별 통과 여부를 계산한다. 채택 판정이 아니다."""
    items = {(item['scope'], item['metric']): item for item in summary}

    def mean_pp(scope, metric):
        difference = items[(scope, metric)]['difference_mean'] * 100
        return round(difference, BOUNDARY_DIGITS)

    def at_least(name, value, limit):
        return {'item': name, 'value': value, 'limit': limit, 'rule': '>=', 'passed': value >= limit}

    def at_most(name, value, limit):
        return {'item': name, 'value': value, 'limit': limit, 'rule': '<=', 'passed': value <= limit}

    improved = sum(
        row['difference'] > 0 for row in rows
        if row['scope'] == 'Scratch' and row['metric'] == 'precision'
    )

    return [
        at_least('평균 Macro-F1 차이(%p)', mean_pp(OVERALL, 'macro_f1'),
                 criteria['macro_f1_mean_difference_min_pp']),
        at_least('평균 Scratch Precision 차이(%p)', mean_pp('Scratch', 'precision'),
                 criteria['scratch_precision_mean_difference_min_pp']),
        at_least('Scratch Precision 개선 seed 수', improved,
                 criteria['scratch_precision_improved_seeds_min']),
        at_least('평균 Scratch Recall 차이(%p)', mean_pp('Scratch', 'recall'),
                 criteria['scratch_recall_mean_difference_min_pp']),
        at_least('평균 Random Recall 차이(%p)', mean_pp('Random', 'recall'),
                 criteria['random_recall_mean_difference_min_pp']),
        at_most('Near-Full FN 증가 합계', items[('Near-Full', 'fn')]['difference_sum'],
                criteria['near_full_fn_increase_sum_max']),
        at_most('Donut FN 증가 합계', items[('Donut', 'fn')]['difference_sum'],
                criteria['donut_fn_increase_sum_max']),
    ]


def _value(metric, value):
    """지표는 퍼센트, FP·FN은 건수로 표시한다."""
    return f'{value * 100:.2f}' if metric in PERCENT_METRICS else f'{value:g}'


def write_report(path, result, summary, rows, seeds):
    """판정표와 해석 범위를 Markdown으로 저장한다."""
    code = result['code']
    settings = result['weight_settings']
    criteria_rows = result['criteria_results']

    if code['same_code']:
        code_text = '기준과 후보의 실행 코드 식별값이 같습니다.'
    else:
        code_text = (
            '기준과 후보의 실행 코드 식별값이 **다릅니다**. 차이 파일: '
            + ', '.join(f'`{name}`' for name in code['differing_code_files']) + '. '
            '같은 코드로 학습한 비교가 아닙니다.'
        )

    lines = [
        f"# {result['model_name']}+D4 가중치 대조: {result['candidate_name']}",
        '',
        f"**선정 상태: {SELECTION_STATUS}.** 최종 채택은 사용자가 판단합니다.",
        '',
        'Test는 사용하지 않았습니다. 후보 조정에 쓴 Validation의 추가 튜닝 결과이므로 '
        '일반화 성능이나 통계적 우월성을 뜻하지 않습니다. seed별 저장 selected_epoch '
        '혼동행렬로 계산했고 재추론은 없습니다.',
        '',
        code_text,
        '',
        f"- 기준 코드: `{code['baseline_code_sha256']}`",
        f"- 후보 코드: `{code['candidate_code_sha256']}`",
        f"- 캐시: `{result['cache_sha256']}`",
        f"- 판정 참고안: `{result['criteria_sha256']}`",
        '',
        '해시는 혼용·변경 확인용이며 인증이 아닙니다.',
        '',
        '## 가중치 설정',
        '',
        '| 항목 | 기준 | 후보 |',
        '| --- | ---: | ---: |',
    ]
    for key in WEIGHT_KEYS:
        lines.append(
            f"| {key} | {settings['baseline'][key]} | {settings['candidate'][key]} |"
        )
    for name in CLASS_NAMES:
        lines.append(
            f"| 가중치 {name} | {result['class_weights']['baseline'][name]:.6f} "
            f"| {result['class_weights']['candidate'][name]:.6f} |"
        )

    lines += [
        '',
        '공통 배율은 가중 평균 손실에서 상쇄되므로 클래스 사이 비율로 해석합니다. '
        '기준의 가중치는 이전 보고서에 기록이 없어 설정과 캐시 빈도로 계산했습니다.',
        '',
        '## 판정 참고안',
        '',
        '사용자가 수치를 승인한 기준이나 통계적 유의성 기준이 아닙니다. 결과를 본 뒤 바꾸면 '
        '사후 변경으로 기록합니다. FN 증가 합계는 (후보 3seed FN 합 − 기준 3seed FN 합)이며, '
        '같은 Validation의 반복 예측이라 독립 표본 수가 아닙니다.',
        '',
        '| 항목 | 값 | 기준 | 결과 |',
        '| --- | ---: | ---: | --- |',
    ]
    for item in criteria_rows:
        lines.append(
            f"| {item['item']} | {item['value']:g} | {item['rule']} {item['limit']:g} "
            f"| {'통과' if item['passed'] else '미달'} |"
        )

    lines += [
        '',
        f"참고안 전체 통과: {'예' if result['all_reference_items_passed'] else '아니요'}. "
        '통과가 곧 채택을 뜻하지 않습니다.',
        '',
        '## 3seed 평균·표준편차와 동일 seed 차이',
        '',
        '지표는 %, 차이는 %p이고 FP·FN은 건수입니다.',
        '',
        '| 범위 | 지표 | 기준 | 후보 | 평균 차이 | 차이 표준편차 | 악화 seed |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for item in summary:
        metric = item['metric']
        lines.append(
            f"| {item['scope']} | {metric} "
            f"| {_value(metric, item['baseline_mean'])} ± {_value(metric, item['baseline_std'])} "
            f"| {_value(metric, item['candidate_mean'])} ± {_value(metric, item['candidate_std'])} "
            f"| {_value(metric, item['difference_mean'])} "
            f"| {_value(metric, item['difference_std'])} "
            f"| {item['worse_seeds'] or '-'} |"
        )

    seed_columns = ''.join(f' seed{seed} 차이 |' for seed in seeds)
    lines += [
        '',
        '## 핵심 지표의 seed별 차이',
        '',
        f'| 범위 | 지표 |{seed_columns}',
        '| --- | --- |' + ' ---: |' * len(seeds),
    ]
    for scope, metric in (
        (OVERALL, 'macro_f1'),
        ('Scratch', 'precision'),
        ('Scratch', 'recall'),
        ('Scratch', 'fp'),
        ('Scratch', 'fn'),
        ('Random', 'recall'),
        ('Random', 'fn'),
        ('Near-Full', 'recall'),
        ('Near-Full', 'fn'),
        ('Donut', 'recall'),
        ('Donut', 'fn'),
    ):
        values = {
            row['seed']: row['difference'] for row in rows
            if row['scope'] == scope and row['metric'] == metric
        }
        cells = ' | '.join(_value(metric, values[seed]) for seed in seeds)
        lines.append(f'| {scope} | {metric} | {cells} |')

    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def publish_comparison(report, temporary, rows, summary, result, write_markdown):
    """대조 산출물을 임시 폴더에 모두 쓴 뒤 이름 변경으로 한 번에 공개한다."""
    temporary.mkdir(parents=True)
    try:
        _write_csv(temporary / 'seed_differences.csv', rows)
        _write_csv(temporary / 'summary.csv', summary)
        (temporary / 'comparison.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )
        write_markdown(temporary / 'comparison.md')
        (temporary / 'SUCCESS.json').write_text(json.dumps({
            'status': result['status'],
            'artifacts': {
                path.name: file_digest(path)
                for path in sorted(temporary.iterdir())
            },
        }, indent=2), encoding='utf-8')
        require(not report.exists(), 'WEIGHT_COMPARE_REPORT_ALREADY_EXISTS')
        temporary.rename(report)
    except BaseException:
        if temporary.exists():
            _set_aside_incomplete(temporary, report)
        raise


def compare_candidate(
    candidate_root,
    report_dir,
    baseline_root='models/wm811k/compare_v2',
    cache_dir='data/processed/wm811k/input_cache/wm_resize_v1',
    criteria_path='configs/wm811k/weight_v1/decision_reference.json',
    seeds=DEFAULT_SEEDS,
    model_name=DEFAULT_MODEL,
):
    """같은 모델의 기준과 후보를 검증해 동일 seed 대조 결과를 새 폴더에 공개한다."""
    require(model_name in SUPPORTED_MODELS, 'WEIGHT_COMPARE_MODEL_UNSUPPORTED')

    report = Path(report_dir).resolve()
    temporary = report.with_name(report.name + '.tmp')
    require(not report.exists(), 'WEIGHT_COMPARE_REPORT_ALREADY_EXISTS')
    require(not temporary.exists(), 'WEIGHT_COMPARE_PARTIAL_EXISTS')

    seeds = tuple(seeds)
    require(
        len(seeds) == len(DEFAULT_SEEDS) and set(seeds) == set(DEFAULT_SEEDS),
        'WEIGHT_COMPARE_SEEDS_INCOMPLETE',
    )

    criteria, criteria_sha256 = load_criteria(criteria_path)

    candidate_path = Path(candidate_root).resolve()
    baseline = load_condition(Path(baseline_root).resolve(), seeds, model_name)
    candidate = load_condition(candidate_path, seeds, model_name)
    code = require_comparable(baseline, candidate)

    # 이미지 배열은 열지 않고 캐시 보고서의 Train 빈도만 읽는다.
    cache_file = Path(cache_dir).resolve(strict=True) / 'cache.json'
    cache_sha256 = file_digest(cache_file)
    reference = baseline[seeds[0]]['contract']
    require(reference['cache_sha256'] == cache_sha256, 'WEIGHT_COMPARE_CACHE_MISMATCH')
    cache_report = read_json(cache_file)

    weights = {
        'baseline': compute_weights(cache_report, reference),
        'candidate': compute_weights(cache_report, candidate[seeds[0]]['contract']),
    }
    require_recorded_weights(candidate, weights['candidate'])

    rows = paired_rows(baseline, candidate, seeds)
    summary = summarize_rows(rows)
    criteria_results = evaluate_criteria(summary, rows, criteria)

    result = {
        'status': 'weight_comparison_complete',
        'selection_status': SELECTION_STATUS,
        'selected_model': None,
        'split': 'validation',
        'test_split_used': False,
        'scope': 'additional_validation_tuning',
        'model_name': model_name,
        'augmentation': AUGMENTATION,
        'seeds': list(seeds),
        'candidate_name': candidate_path.name,
        'code': code,
        'cache_sha256': cache_sha256,
        'criteria': criteria,
        'criteria_sha256': criteria_sha256,
        'criteria_results': criteria_results,
        'all_reference_items_passed': all(item['passed'] for item in criteria_results),
        'weight_settings': {
            'baseline': weight_settings(reference),
            'candidate': weight_settings(candidate[seeds[0]]['contract']),
        },
        'class_weights': weights,
        'selected_epochs': {
            'baseline': {seed: baseline[seed]['selected_epoch'] for seed in seeds},
            'candidate': {seed: candidate[seed]['selected_epoch'] for seed in seeds},
        },
        'input_digests': {
            'baseline': {seed: baseline[seed]['digests'] for seed in seeds},
            'candidate': {seed: candidate[seed]['digests'] for seed in seeds},
        },
    }

    publish_comparison(
        report, temporary, rows, summary, result,
        lambda path: write_report(path, result, summary, rows, seeds),
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description='D4 모델의 가중치 후보와 같은 모델 기준의 읽기 전용 대조 (학습·추론·Test 없음)'
    )
    parser.add_argument('--candidate-root', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--baseline-root', default='models/wm811k/compare_v2')
    parser.add_argument('--cache-dir', default='data/processed/wm811k/input_cache/wm_resize_v1')
    parser.add_argument('--criteria', default='configs/wm811k/weight_v1/decision_reference.json')
    parser.add_argument('--model', default=DEFAULT_MODEL, choices=SUPPORTED_MODELS)
    args = parser.parse_args()

    try:
        with validation_stage('weight_compare'):
            result = compare_candidate(
                args.candidate_root,
                args.report_dir,
                args.baseline_root,
                args.cache_dir,
                args.criteria,
                model_name=args.model,
            )
    except ValueError as error:
        print('WM 가중치 대조 실패 코드: ' + str(error))
        if str(error) in FAILURE_GUIDANCE:
            print(FAILURE_GUIDANCE[str(error)])
        return 1

    print(json.dumps({
        'status': result['status'],
        'candidate': result['candidate_name'],
        'selection_status': result['selection_status'],
        'same_code': result['code']['same_code'],
        'differing_code_files': result['code']['differing_code_files'],
        'all_reference_items_passed': result['all_reference_items_passed'],
        'report': str(Path(args.report_dir) / 'comparison.md'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

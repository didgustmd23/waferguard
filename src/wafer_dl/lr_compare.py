"""D4·alpha0.75 모델의 학습률 스케줄 후보를 같은 모델의 고정 학습률 기준과 대조한다.

학습·추론·Test 평가를 하지 않고 저장된 보고서와 selected_epoch 혼동행렬만 읽는다.
스케줄 설정과 명시적인 코드 차이만 비교 축으로 허용하고 나머지 조건 차이는 거부한다.
후보 결과를 본 Validation 비교이므로 일반화 성능이 아니며, 합격 기준 없이 지표만
나란히 보여 주고 항상 선정 보류로 둔다.
"""

import argparse
import json
from pathlib import Path

import torch

from .cache import file_digest
from .compare import DEFAULT_SEEDS
from .train import LR_SCHEDULER_KEY, build_lr_scheduler
from .validate_canonical import CLASS_NAMES, read_json, require, validation_stage
from .weight_compare import (
    AUGMENTATION,
    FAILURE_GUIDANCE as WEIGHT_FAILURE_GUIDANCE,
    OVERALL,
    SELECTION_STATUS,
    SUPPORTED_MODELS,
    WEIGHT_KEYS,
    _value,
    load_condition,
    paired_rows,
    publish_comparison,
    summarize_rows,
    weight_settings,
)


DEFAULT_BASELINES = {
    'SmallWaferCNN': 'models/wm811k/weight_cnn_v1/alpha0_75',
    'ResNet18': 'models/wm811k/weight_v1/alpha0_75',
}
EXPECTED_WEIGHTS = {
    'max_class_weight': 10.0,
    'class_weight_alpha': 0.75,
    'scratch_weight_factor': 1.0,
}
TRAINING = 'training'
SEED_TABLE_METRICS = (
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
    (TRAINING, 'training_minutes'),
    (TRAINING, 'epochs_run'),
    (TRAINING, 'selected_epoch'),
    (TRAINING, 'lr_reductions'),
)

# 감소가 실제로 일어났는지와 감소 뒤 학습 기회가 있었는지를 구분해 표시한다.
SCHEDULE_STATES = {
    'no_reduction': '감소 없음',
    'reduced_without_training': '감소했으나 이후 학습 없음',
    'reduced_with_training': '감소 후 학습함',
}

FAILURE_GUIDANCE = {
    'LR_COMPARE_MODEL_UNSUPPORTED': (
        '학습률 대조는 ResNet18과 SmallWaferCNN만 지원합니다. --model을 확인하세요.'
    ),
    'LR_COMPARE_REPORT_ALREADY_EXISTS': (
        '이전 대조를 덮어쓰지 않습니다. --report-dir에 새 경로를 지정하세요.'
    ),
    'LR_COMPARE_PARTIAL_EXISTS': (
        '끝나지 않은 .tmp 폴더가 있습니다. 지우지 말고 --report-dir에 새 경로를 지정하세요.'
    ),
    'LR_COMPARE_SEEDS_INCOMPLETE': (
        'seed 42·43·44를 모두 대조합니다. 빠진 실행을 compare run으로 끝낸 뒤 다시 실행하세요.'
    ),
    'LR_COMPARE_BASELINE_HAS_SCHEDULER': (
        '기준은 고정 학습률 실행이어야 합니다. --baseline-root를 확인하세요.'
    ),
    'LR_COMPARE_CANDIDATE_HAS_NO_SCHEDULER': (
        '후보에 학습률 스케줄 설정이 없습니다. --candidate-root를 확인하세요.'
    ),
    'LR_COMPARE_WEIGHT_MISMATCH': (
        '상한10·alpha0.75·Scratch계수1 실행이 아닙니다. 다른 가중치 폴더를 쓰지 마세요.'
    ),
    'LR_COMPARE_CONDITION_MISMATCH': (
        '스케줄 외의 조건이 다릅니다. 후보를 같은 캐시·설정으로 새 경로에 다시 학습하세요.'
    ),
    'LR_COMPARE_CODE_MIXED': (
        '한 조건 안에 서로 다른 코드의 실행이 섞였습니다. 새 --output-root에 같은 코드로 '
        '3회를 모두 실행하세요.'
    ),
    'LR_COMPARE_RECORD_MISMATCH': (
        '학습률·조기 종료 기록이 설정과 저장 점수로 다시 계산한 값과 다릅니다. '
        '해당 실행을 대조에 쓰지 마세요.'
    ),
    'LR_COMPARE_CACHE_MISMATCH': (
        '학습에 쓴 캐시가 아닙니다. 해당 실행이 사용한 --cache-dir을 지정하세요.'
    ),
    **{
        code: WEIGHT_FAILURE_GUIDANCE[code]
        for code in (
            'COMPARE_RUN_INCOMPLETE',
            'COMPARE_RUN_CONTRACT_MISMATCH',
            'COMPARE_RUN_INTEGRITY_FAILED',
        )
    },
}


def _common_condition(contract):
    """스케줄·코드·seed를 뺀, 기준과 후보가 같아야 하는 조건을 남긴다."""
    common = {
        key: value for key, value in contract.items()
        if key not in ('code_fingerprint', 'seed')
    }
    common['training_config'] = {
        key: value for key, value in contract['training_config'].items()
        if key not in ('seed', LR_SCHEDULER_KEY)
    }
    return common


def require_comparable(baseline, candidate):
    """스케줄과 명시적인 코드 차이 외의 조건이 같은지 확인하고 코드 차이를 돌려준다."""
    for runs in (baseline, candidate):
        require(
            len({
                run['contract']['code_fingerprint']['sha256'] for run in runs.values()
            }) == 1,
            'LR_COMPARE_CODE_MIXED',
        )

    require(
        all(
            LR_SCHEDULER_KEY not in run['contract']['training_config']
            for run in baseline.values()
        ),
        'LR_COMPARE_BASELINE_HAS_SCHEDULER',
    )
    require(
        all(
            LR_SCHEDULER_KEY in run['contract']['training_config']
            for run in candidate.values()
        ),
        'LR_COMPARE_CANDIDATE_HAS_NO_SCHEDULER',
    )
    require(
        all(
            weight_settings(run['contract']) == EXPECTED_WEIGHTS
            for runs in (baseline, candidate) for run in runs.values()
        ),
        'LR_COMPARE_WEIGHT_MISMATCH',
    )

    reference = next(iter(baseline.values()))
    expected = _common_condition(reference['contract'])
    for runs in (baseline, candidate):
        for run in runs.values():
            require(
                _common_condition(run['contract']) == expected and
                run['transform_config'] == reference['transform_config'] and
                run['class_mapping'] == reference['class_mapping'],
                'LR_COMPARE_CONDITION_MISMATCH',
            )

    first = next(iter(candidate.values()))
    require(
        all(
            run['contract']['training_config'][LR_SCHEDULER_KEY] ==
            first['contract']['training_config'][LR_SCHEDULER_KEY]
            for run in candidate.values()
        ),
        'LR_COMPARE_CONDITION_MISMATCH',
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


def replay_learning_rates(history, config):
    """저장된 Validation 점수를 같은 스케줄에 넣어 epoch별 학습률을 다시 계산한다."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=float(config['learning_rate']))
    scheduler = build_lr_scheduler(optimizer, config)

    replayed = []
    for epoch in history:
        used = optimizer.param_groups[0]['lr']
        scheduler.step(epoch['validation'][config['selection_metric']])
        replayed.append((used, optimizer.param_groups[0]['lr']))
    return replayed


def schedule_summary(run):
    """후보 기록을 재계산과 대조하고 seed별 학습률 동작을 정리한다."""
    report = run['report']
    config = run['contract']['training_config']
    history = report.get('history')
    selected_epoch = run['selected_epoch']
    require(
        type(history) is list and history and
        [epoch.get('epoch') for epoch in history] == list(range(1, len(history) + 1)) and
        report.get('lr_scheduler') == config[LR_SCHEDULER_KEY],
        'LR_COMPARE_RECORD_MISMATCH',
    )

    recorded = [
        (epoch.get('learning_rate'), epoch.get('next_learning_rate')) for epoch in history
    ]
    reductions = [
        {'epoch': number, 'learning_rate': used, 'next_learning_rate': following}
        for number, (used, following) in enumerate(recorded, start=1)
        if following < used
    ]
    epochs_run = len(history)

    # 스케줄을 넣어도 epoch 상한과 조기 종료 인내 횟수는 기준과 같아야 한다.
    require(
        recorded == replay_learning_rates(history, config) and
        report.get('lr_reductions') == reductions and
        selected_epoch <= epochs_run <= config['epochs'] and
        (epochs_run == config['epochs'] or epochs_run - selected_epoch == config['patience']),
        'LR_COMPARE_RECORD_MISMATCH',
    )

    # 최고 checkpoint에는 그 epoch까지의 학습률 기록이 남아 있어야 한다.
    stored = run['lr_schedule']
    require(
        type(stored) is dict and
        stored.get('settings') == config[LR_SCHEDULER_KEY] and
        (stored.get('learning_rate'), stored.get('next_learning_rate')) ==
        recorded[selected_epoch - 1] and
        stored.get('reductions') ==
        [item for item in reductions if item['epoch'] <= selected_epoch] and
        type(stored.get('scheduler_state')) is dict,
        'LR_COMPARE_RECORD_MISMATCH',
    )

    first_epoch = reductions[0]['epoch'] if reductions else None
    if first_epoch is None:
        state = 'no_reduction'
    elif first_epoch == epochs_run:
        state = 'reduced_without_training'
    else:
        state = 'reduced_with_training'

    return {
        'state': state,
        'reductions': reductions,
        'first_reduction_epoch': first_epoch,
        'epochs_after_first_reduction': 0 if first_epoch is None else epochs_run - first_epoch,
        'selected_after_first_reduction': first_epoch is not None and selected_epoch > first_epoch,
        'selected_epoch': selected_epoch,
        'epochs_run': epochs_run,
        'last_trained_learning_rate': recorded[-1][0],
    }


def training_rows(baseline, candidate, schedules, seeds):
    """학습 시간·epoch·학습률 감소 횟수를 지표 행과 같은 형식으로 만든다."""
    rows = []
    for seed in seeds:
        before = baseline[seed]['report']
        after = candidate[seed]['report']
        values = {
            'training_minutes': (
                before['training_seconds'] / 60, after['training_seconds'] / 60
            ),
            'epochs_run': (len(before['history']), len(after['history'])),
            'selected_epoch': (baseline[seed]['selected_epoch'], candidate[seed]['selected_epoch']),
            'lr_reductions': (0, len(schedules[seed]['reductions'])),
        }
        for metric, (old, new) in values.items():
            rows.append({
                'seed': seed,
                'scope': TRAINING,
                'metric': metric,
                'baseline': old,
                'candidate': new,
                'difference': new - old,
            })
    return rows


def write_report(path, result, summary, rows, seeds):
    """seed별 값·평균·차이와 학습률 동작, 해석 범위를 Markdown으로 저장한다."""
    code = result['code']
    if code['same_code']:
        code_text = '기준과 후보의 실행 코드 식별값이 같습니다.'
    else:
        code_text = (
            '기준과 후보의 실행 코드 식별값이 **다릅니다**. 차이 파일: '
            + ', '.join(f'`{name}`' for name in code['differing_code_files']) + '. '
            '고정 학습률 경로의 계산 동등성은 합성 검증 범위이며, 실제 신구 학습의 '
            '동등성은 측정하지 않았습니다.'
        )

    lines = [
        f"# {result['model_name']}+D4 학습률 스케줄 대조: {result['candidate_name']}",
        '',
        f"**선정 상태: {SELECTION_STATUS}.** 합격 기준을 정하지 않았으므로 자동으로 채택하지 "
        '않고 지표만 나란히 보여 줍니다. 최종 판단은 사용자가 합니다.',
        '',
        'Test는 사용하지 않았습니다. 추가 튜닝에 쓴 Validation 결과이므로 일반화 성능이나 '
        '통계적 우월성을 뜻하지 않습니다. seed별 저장 selected_epoch 혼동행렬로 계산했고 '
        '재추론은 없습니다.',
        '',
        code_text,
        '',
        f"- 기준 코드: `{code['baseline_code_sha256']}`",
        f"- 후보 코드: `{code['candidate_code_sha256']}`",
        f"- 캐시: `{result['cache_sha256']}`",
        '',
        '해시는 혼용·변경 확인용이며 인증이 아닙니다. 파일별 입력 식별값은 '
        '`comparison.json`의 `input_digests`에 있습니다.',
        '',
        '## 설정',
        '',
        '| 항목 | 기준 | 후보 |',
        '| --- | ---: | ---: |',
        f"| 초기 학습률 | {result['learning_rate']} | {result['learning_rate']} |",
    ]
    for key in WEIGHT_KEYS:
        lines.append(f"| {key} | {EXPECTED_WEIGHTS[key]} | {EXPECTED_WEIGHTS[key]} |")
    lines.append('| 학습률 스케줄 | 고정 | 아래 설정 |')
    for key, value in result['lr_scheduler'].items():
        lines.append(f'| {key} | - | {value} |')

    lines += [
        '',
        '## seed별 학습률 동작',
        '',
        '감소가 없었거나 감소 뒤 학습 기회가 없었던 seed는 스케줄 효과가 없다는 근거가 아닙니다.',
        '',
        '| seed | 상태 | 감소 횟수 | 감소 epoch | 첫 감소 뒤 epoch | selected_epoch '
        '| 첫 감소 뒤 선택 | 실행 epoch | 마지막 학습률 |',
        '| ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |',
    ]
    for seed in seeds:
        item = result['schedules'][seed]
        epochs = ' '.join(str(reduction['epoch']) for reduction in item['reductions'])
        lines.append(
            f"| {seed} | {SCHEDULE_STATES[item['state']]} | {len(item['reductions'])} "
            f"| {epochs or '-'} | {item['epochs_after_first_reduction']} "
            f"| {item['selected_epoch']} "
            f"| {'예' if item['selected_after_first_reduction'] else '아니요'} "
            f"| {item['epochs_run']} | {item['last_trained_learning_rate']:g} |"
        )

    lines += [
        '',
        '## 3seed 평균·표준편차와 동일 seed 차이',
        '',
        '지표는 %, 차이는 %p이고 FP·FN은 건수입니다. `training` 범위의 시간은 분, 나머지는 '
        '횟수이며 증감의 좋고 나쁨을 판정하지 않습니다.',
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

    seed_columns = ''.join(f' seed{seed} 기준 → 후보 (차이) |' for seed in seeds)
    lines += [
        '',
        '## 핵심 지표의 seed별 값',
        '',
        f'| 범위 | 지표 |{seed_columns}',
        '| --- | --- |' + ' ---: |' * len(seeds),
    ]
    for scope, metric in SEED_TABLE_METRICS:
        values = {
            row['seed']: row for row in rows
            if row['scope'] == scope and row['metric'] == metric
        }
        cells = ' | '.join(
            f"{_value(metric, values[seed]['baseline'])} → "
            f"{_value(metric, values[seed]['candidate'])} "
            f"({_value(metric, values[seed]['difference'])})"
            for seed in seeds
        )
        lines.append(f'| {scope} | {metric} | {cells} |')

    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def compare_schedule(
    model_name,
    candidate_root,
    report_dir,
    baseline_root=None,
    cache_dir='data/processed/wm811k/input_cache/wm_resize_v1',
    seeds=DEFAULT_SEEDS,
):
    """같은 모델의 고정 학습률 기준과 스케줄 후보를 검증해 새 폴더에 대조를 공개한다."""
    require(model_name in SUPPORTED_MODELS, 'LR_COMPARE_MODEL_UNSUPPORTED')

    report = Path(report_dir).resolve()
    temporary = report.with_name(report.name + '.tmp')
    require(not report.exists(), 'LR_COMPARE_REPORT_ALREADY_EXISTS')
    require(not temporary.exists(), 'LR_COMPARE_PARTIAL_EXISTS')

    seeds = tuple(seeds)
    require(
        len(seeds) == len(DEFAULT_SEEDS) and set(seeds) == set(DEFAULT_SEEDS),
        'LR_COMPARE_SEEDS_INCOMPLETE',
    )

    candidate_path = Path(candidate_root).resolve()
    baseline_path = Path(baseline_root or DEFAULT_BASELINES[model_name]).resolve()
    baseline = load_condition(baseline_path, seeds, model_name)
    candidate = load_condition(candidate_path, seeds, model_name)
    code = require_comparable(baseline, candidate)

    cache_sha256 = file_digest(Path(cache_dir).resolve(strict=True) / 'cache.json')
    reference = baseline[seeds[0]]['contract']
    require(reference['cache_sha256'] == cache_sha256, 'LR_COMPARE_CACHE_MISMATCH')

    schedules = {seed: schedule_summary(candidate[seed]) for seed in seeds}
    rows = paired_rows(baseline, candidate, seeds)
    rows += training_rows(baseline, candidate, schedules, seeds)

    # 시간·epoch의 증감은 좋고 나쁨이 정해져 있지 않아 악화로 표시하지 않는다.
    summary = [
        {**item, 'worse_seeds': '' if item['scope'] == TRAINING else item['worse_seeds']}
        for item in summarize_rows(rows)
    ]

    first_contract = candidate[seeds[0]]['contract']
    result = {
        'status': 'lr_schedule_comparison_complete',
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
        'learning_rate': reference['training_config']['learning_rate'],
        'lr_scheduler': first_contract['training_config'][LR_SCHEDULER_KEY],
        'training_config': {
            'baseline': reference['training_config'],
            'candidate': first_contract['training_config'],
        },
        'schedules': schedules,
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
        description='학습률 스케줄 후보와 같은 모델 고정 학습률 기준의 읽기 전용 대조 '
                    '(학습·추론·Test 없음)'
    )
    parser.add_argument('--model', required=True, choices=SUPPORTED_MODELS)
    parser.add_argument('--candidate-root', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--baseline-root', help='지정하지 않으면 모델별 alpha0_75 기준을 쓴다')
    parser.add_argument('--cache-dir', default='data/processed/wm811k/input_cache/wm_resize_v1')
    args = parser.parse_args()

    try:
        with validation_stage('lr_compare'):
            result = compare_schedule(
                args.model,
                args.candidate_root,
                args.report_dir,
                args.baseline_root,
                args.cache_dir,
            )
    except ValueError as error:
        print('WM 학습률 대조 실패 코드: ' + str(error))
        if str(error) in FAILURE_GUIDANCE:
            print(FAILURE_GUIDANCE[str(error)])
        return 1

    print(json.dumps({
        'status': result['status'],
        'model_name': result['model_name'],
        'selection_status': result['selection_status'],
        'same_code': result['code']['same_code'],
        'differing_code_files': result['code']['differing_code_files'],
        'schedule_states': {
            seed: item['state'] for seed, item in result['schedules'].items()
        },
        'report': str(Path(args.report_dir) / 'comparison.md'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

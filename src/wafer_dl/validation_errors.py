"""ResNet18+D4 checkpoint를 Validation에서 재추론해 오분류 표본을 분석한다.

학습·재학습·Test 평가를 하지 않고 Validation 분할만 읽는다. 세 seed를 합쳐 새
앙상블을 만들지 않으며, 예측을 근거로 원본 라벨을 수정하지 않는다. 입력은 모두
읽기 전용이고, 결과는 저장된 혼동행렬이 재현된 경우에만 완료로 공개한다.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import torch
from torch.utils.data import DataLoader

from .cache import file_digest
from .compare import DEFAULT_SEEDS, RUN_CODE_FILES, code_fingerprint, run_name, verify_run
from .dataset import WaferDataset
from .evaluate import _require_finite_model_weights, _require_finite_tensor, validate_checkpoint
from .metrics import classification_metrics
from .models import build_model
from .validate_canonical import CLASS_NAMES, read_json, require, validation_stage


# 이번 분석 대상은 사전에 정한 조건 하나로 고정한다.
MODEL_NAME = 'ResNet18'
AUGMENTATION = 'd4'

# 이 경로의 분석 결과를 식별하는 파일이다. 혼용 검사용이며 인증이 아니다.
ANALYSIS_CODE_FILES = (*RUN_CODE_FILES, 'evaluate.py', 'validation_errors.py')

ANALYSIS_COMPLETE_STATUS = 'validation_error_analysis_complete'
ANALYSIS_MISMATCH_STATUS = 'validation_error_analysis_confusion_mismatch'

# 우선 검토 그룹. predicted가 None이면 실제 라벨 외의 모든 예측을 뜻한다.
PRIORITY_GROUPS = (
    {'id': 'loc_to_scratch', 'actual': 'Loc', 'predicted': 'Scratch'},
    {'id': 'none_to_scratch', 'actual': 'None', 'predicted': 'Scratch'},
    {'id': 'random_to_other', 'actual': 'Random', 'predicted': None},
)

MAX_BATCH_SIZE = 4096
MAX_IMAGES_PER_GROUP = 200
MAX_IMAGES_PER_PAGE = 12
PAGE_COLUMNS = 4
TOP_FLOWS = 10

# 모델 입력 두 채널을 합성한 3색 범례. 모든 페이지에 같은 색을 쓴다.
MAP_LEGEND = (
    (0, '#EFEFEF', 'Outside wafer'),
    (1, '#4C78A8', 'Valid die'),
    (2, '#F58518', 'Defect die'),
)
MAP_COLORS = ListedColormap([color for _, color, _ in MAP_LEGEND])
PAGE_NOTE = (
    '128×128 model input from the cache, not an original photo. '
    '"None" is a dataset label, not a final good-wafer verdict.'
)

# 실패 코드별로 사용자가 할 수 있는 다음 행동을 안내한다.
FAILURE_GUIDANCE = {
    'VALIDATION_ERRORS_OUTPUT_EXISTS': (
        '이전 분석을 덮어쓰지 않습니다. --output-dir에 새 경로를 지정하세요.'
    ),
    'VALIDATION_ERRORS_PARTIAL_EXISTS': (
        '끝나지 않은 .tmp 폴더가 있습니다. 내용을 지우지 않고 --output-dir에 새 경로를 지정하세요.'
    ),
    'VALIDATION_ERRORS_CODE_MISMATCH': (
        '학습 당시 코드와 현재 코드가 달라 예측하지 않았습니다. 학습 당시 코드로 되돌린 뒤 '
        '실행하세요. 분석만을 위해 재학습하거나 식별값을 고치지 마세요.'
    ),
    'VALIDATION_ERRORS_CACHE_MISMATCH': (
        '학습에 쓴 캐시가 아닙니다. 해당 실행이 사용한 --cache-dir을 지정하세요.'
    ),
    'VALIDATION_ERRORS_CONFUSION_MISMATCH': (
        '저장된 혼동행렬이 재현되지 않아 정상 결과로 공개하지 않았습니다. 출력 경로 옆 '
        '.mismatch_N 폴더의 MISMATCH.json과 장치(학습과 같은 GPU인지)를 확인하세요.'
    ),
    'VALIDATION_ERRORS_INPUT_CHANGED': (
        '분석 중 입력 파일이나 코드가 바뀌었습니다. 다른 작업이 끝난 뒤 새 경로로 다시 실행하세요.'
    ),
    'COMPARE_RUN_INCOMPLETE': (
        '완료 표식이 없는 실행입니다. 다른 checkpoint로 대체하지 않으니 --input-root를 확인하세요.'
    ),
    'CUDA_DEVICE_UNAVAILABLE': (
        'GPU를 찾을 수 없습니다. CPU는 저장 결과와 미세하게 다를 수 있으므로 GPU 환경을 확인하세요.'
    ),
}


def analysis_fingerprint():
    """실행 코드에 분석 코드를 더한 식별값을 만든다."""
    return code_fingerprint(files=ANALYSIS_CODE_FILES)


def _select_device(device):
    """요청한 장치를 정한다. GPU를 명시했는데 없으면 조용히 CPU로 바꾸지 않는다."""
    require(device in ('auto', 'cpu', 'cuda'), 'INVALID_ANALYSIS_DEVICE')
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA_DEVICE_UNAVAILABLE')
    uses_cuda = device == 'cuda' or (device == 'auto' and torch.cuda.is_available())
    return torch.device('cuda' if uses_cuda else 'cpu')


def _validate_options(seeds, batch_size, max_images, images_per_page):
    """seed 조합과 배치·이미지 상한을 확인한다."""
    require(
        len(seeds) > 0 and len(set(seeds)) == len(seeds) and
        all(seed in DEFAULT_SEEDS for seed in seeds),
        'INVALID_ANALYSIS_SEEDS',
    )
    for value, limit in (
        (batch_size, MAX_BATCH_SIZE),
        (max_images, MAX_IMAGES_PER_GROUP),
        (images_per_page, MAX_IMAGES_PER_PAGE),
    ):
        require(type(value) is int and 1 <= value <= limit, 'INVALID_ANALYSIS_LIMIT')


def _input_digests(run_dir):
    """실행 입력 파일의 해시를 모은다. 변경 확인용이며 인증이 아니다."""
    return {
        name: file_digest(run_dir / name)
        for name in ('best_model.pt', 'training_report.json', 'SUCCESS.json')
    }


def load_run(run_dir, seed, dataset, cache_report, cache_sha256):
    """실행 완료 상태와 코드·캐시·분할·선택 epoch 계약을 예측 전에 대조한다."""
    stored = verify_run(run_dir, MODEL_NAME, AUGMENTATION, seed)

    # verify_run은 과거 코드 결과 조회를 허용하지만, 현재 구조로 추론하는
    # 이 경로에서는 코드가 정확히 같을 때만 진행한다.
    require(
        stored['code_fingerprint'] == code_fingerprint(),
        'VALIDATION_ERRORS_CODE_MISMATCH',
    )
    require(
        stored['cache_sha256'] == cache_sha256 and
        stored['split_protocol_id'] == dataset.protocol and
        stored['source_split_manifest_sha256'] == dataset.source_split_manifest_sha256,
        'VALIDATION_ERRORS_CACHE_MISMATCH',
    )

    digests = _input_digests(run_dir)
    checkpoint = torch.load(
        run_dir / 'best_model.pt',
        map_location='cpu',
        weights_only=True,
    )
    validate_checkpoint(checkpoint, dataset, cache_report.get('train_class_counts'))

    report = read_json(run_dir / 'training_report.json')
    selected = [
        epoch for epoch in report['history']
        if epoch['epoch'] == checkpoint['selected_epoch']
    ]
    require(
        len(selected) == 1 and
        report.get('validation_rows') == len(dataset) and
        selected[0]['validation']['confusion_matrix'] ==
        checkpoint['validation_metrics']['confusion_matrix'],
        'VALIDATION_ERRORS_STORED_METRICS_MISMATCH',
    )

    return {
        'seed': seed,
        'run_dir': run_dir,
        'contract': stored,
        'digests': digests,
        'checkpoint': checkpoint,
        'selected_epoch': checkpoint['selected_epoch'],
        'stored_confusion': selected[0]['validation']['confusion_matrix'],
    }


def _restore_model(checkpoint, device):
    """checkpoint를 strict로 복원하고 평가 모드로 바꾼다."""
    model = build_model(checkpoint['model_name'], num_classes=len(CLASS_NAMES))
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError:
        require(False, 'VALIDATION_ERRORS_CHECKPOINT_INVALID')
    _require_finite_model_weights(model)
    return model.to(device).eval()


def predict_validation(model, dataset, batch_size, device, predictions_path):
    """고정 순서·증강 없이 추론해 표본별 예측을 CSV로 쓰고 행 목록을 돌려준다."""
    uses_cuda = device.type == 'cuda'
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=uses_cuda,
    )
    rows = []

    with predictions_path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow((
            'sample_id', 'actual_label', 'predicted_label',
            'correct', 'predicted_probability',
        ))

        with torch.inference_mode():
            for batch in loader:
                images = batch['image'].to(device, non_blocking=uses_cuda)
                logits = model(images)

                _require_finite_tensor(logits, 'NONFINITE_LOGITS')
                require(
                    logits.shape == (images.shape[0], len(CLASS_NAMES)),
                    'INVALID_LOGIT_SHAPE',
                )

                predictions = logits.argmax(dim=1)
                probabilities = torch.softmax(logits, dim=1).gather(
                    1, predictions.unsqueeze(1)
                ).squeeze(1)

                for sample_id, actual, predicted, probability in zip(
                    batch['sample_id'],
                    batch['target'].tolist(),
                    predictions.cpu().tolist(),
                    probabilities.cpu().tolist(),
                ):
                    rows.append({
                        'sample_id': sample_id,
                        'actual': actual,
                        'predicted': predicted,
                    })
                    writer.writerow((
                        sample_id,
                        CLASS_NAMES[actual],
                        CLASS_NAMES[predicted],
                        actual == predicted,
                        f'{probability:.6f}',
                    ))

    return rows


def require_all_samples(rows, dataset):
    """예측이 Validation 색인과 같은 순서로 빠짐없이 한 번씩 있는지 확인한다."""
    expected = [(record[3], record[2]) for record in dataset.records]
    found = [(row['sample_id'], row['actual']) for row in rows]
    require(found == expected, 'VALIDATION_ERRORS_SAMPLE_MISMATCH')


def confusion_from_rows(rows):
    """저장한 표본별 예측에서 혼동행렬을 다시 계산한다."""
    size = len(CLASS_NAMES)
    matrix = [[0] * size for _ in range(size)]
    for row in rows:
        matrix[row['actual']][row['predicted']] += 1
    return torch.tensor(matrix, dtype=torch.int64)


def compare_confusion(recomputed, stored):
    """재계산 값과 저장 값을 그대로 비교한다. 보정하지 않는다."""
    current = recomputed.tolist()
    differences = [
        {
            'actual_label': CLASS_NAMES[actual],
            'predicted_label': CLASS_NAMES[predicted],
            'stored': stored[actual][predicted],
            'recomputed': current[actual][predicted],
        }
        for actual in range(len(CLASS_NAMES))
        for predicted in range(len(CLASS_NAMES))
        if stored[actual][predicted] != current[actual][predicted]
    ]
    return {'matched': not differences, 'differences': differences}


def join_seed_rows(seed_rows):
    """seed별 예측을 표본 기준으로 결합한다. 합산 표본 수를 만들지 않는다."""
    seeds = list(seed_rows)
    base = seed_rows[seeds[0]]
    for seed in seeds:
        require(
            [(row['sample_id'], row['actual']) for row in seed_rows[seed]] ==
            [(row['sample_id'], row['actual']) for row in base],
            'VALIDATION_ERRORS_SEED_MISMATCH',
        )

    joined = []
    for position, row in enumerate(base):
        predictions = {
            seed: CLASS_NAMES[seed_rows[seed][position]['predicted']]
            for seed in seeds
        }
        actual = CLASS_NAMES[row['actual']]
        joined.append({
            'position': position,
            'sample_id': row['sample_id'],
            'actual_label': actual,
            'predictions': predictions,
            'wrong_seed_count': sum(label != actual for label in predictions.values()),
        })

    return joined


def repeat_summary(joined, seeds):
    """오답 seed 수별 서로 다른 표본 수와 합집합·교집합을 센다."""
    counts = {count: 0 for count in range(len(seeds) + 1)}
    for row in joined:
        counts[row['wrong_seed_count']] += 1

    return {
        'samples_by_wrong_seed_count': counts,
        'union_wrong_samples': len(joined) - counts[0],
        'intersection_wrong_samples': counts[len(seeds)],
    }


def _matches_group(actual, predicted, group):
    """한 예측이 우선 검토 그룹의 오류인지 판정한다."""
    if actual != group['actual'] or predicted == actual:
        return False
    return group['predicted'] is None or predicted == group['predicted']


def group_errors(joined, group, seeds):
    """그룹 오류를 오류 seed 수 내림차순, 동률은 sample_id 순으로 고정해 나열한다."""
    entries = []
    for row in joined:
        hit_seeds = [
            seed for seed in seeds
            if _matches_group(row['actual_label'], row['predictions'][seed], group)
        ]
        if hit_seeds:
            entries.append({**row, 'hit_seeds': hit_seeds})

    entries.sort(key=lambda entry: (-len(entry['hit_seeds']), entry['sample_id']))

    by_hit_count = {count: 0 for count in range(1, len(seeds) + 1)}
    for entry in entries:
        by_hit_count[len(entry['hit_seeds'])] += 1

    summary = {
        'group_id': group['id'],
        'actual_label': group['actual'],
        'predicted_label': group['predicted'],
        'actual_label_samples': sum(row['actual_label'] == group['actual'] for row in joined),
        'seed_error_counts': {
            seed: sum(seed in entry['hit_seeds'] for entry in entries)
            for seed in seeds
        },
        'distinct_samples': len(entries),
        'samples_by_hit_seed_count': by_hit_count,
        'samples_in_every_seed': by_hit_count[len(seeds)],
    }
    return summary, entries


def group_title(group):
    """그림 제목에 쓸 그룹 이름을 만든다."""
    predicted = group['predicted'] or 'any other class'
    return f"Actual {group['actual']} → Predicted {predicted}"


def composite_map(image):
    """두 채널을 0=바깥영역, 1=유효 다이, 2=불량 다이의 맵으로 합친다.

    보간·회전·반전 없이 캐시에서 읽은 모델 입력의 픽셀을 그대로 쓴다.
    """
    require(image.ndim == 3 and image.shape[0] == 2, 'INVALID_ANALYSIS_IMAGE')
    composed = (image[1] > 0).to(dtype=torch.uint8)
    composed[image[0] > 0] = 2
    return composed.numpy()


def _short_id(sample_id):
    """그림 안에 들어가도록 줄인다. 전체 ID는 보고서·CSV에서 번호로 찾는다."""
    return sample_id if len(sample_id) <= 26 else f'{sample_id[:16]}…{sample_id[-8:]}'


def legend_handles():
    """모든 페이지에 표시할 3색 범례를 만든다."""
    return [Patch(facecolor=color, edgecolor='#666666', label=label)
            for _, color, label in MAP_LEGEND]


def render_page(path, title, entries, first_rank, dataset, seeds):
    """한 페이지를 PNG로 저장한다. 표본은 그리는 순간에만 읽어 쌓아 두지 않는다."""
    rows = math.ceil(len(entries) / PAGE_COLUMNS)
    figure = Figure(figsize=(12, 3.3 * rows + 1.0), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, PAGE_COLUMNS, squeeze=False).flatten()

    for axis in axes:
        axis.set_axis_off()

    for number, entry in enumerate(entries):
        image = dataset[entry['position']]['image']
        predictions = '  '.join(
            f"s{seed}: {entry['predictions'][seed]}" for seed in seeds
        )
        axes[number].imshow(
            composite_map(image),
            cmap=MAP_COLORS,
            vmin=0,
            vmax=len(MAP_LEGEND) - 1,
            interpolation='nearest',
        )
        axes[number].set_title(
            f"#{first_rank + number} {_short_id(entry['sample_id'])}\n"
            f"Actual: {entry['actual_label']}\n{predictions}",
            fontsize=8,
            parse_math=False,
        )

    figure.suptitle(title, fontsize=13, parse_math=False)
    figure.supxlabel(PAGE_NOTE, fontsize=8, parse_math=False)
    figure.legend(handles=legend_handles(), loc='outside upper right', ncols=len(MAP_LEGEND))
    figure.savefig(path, format='png', dpi=120)


def render_group(output, group, entries, dataset, seeds, max_images, images_per_page):
    """선정한 표본을 페이지로 나눠 그리고 페이지 경로·생략 수를 돌려준다."""
    selected = entries[:max_images]
    pages = []

    for start in range(0, len(selected), images_per_page):
        relative = f"images/{group['id']}_page{len(pages) + 1:02d}.png"
        render_page(
            output / relative,
            group_title(group),
            selected[start:start + images_per_page],
            start + 1,
            dataset,
            seeds,
        )
        pages.append(relative)

    return {
        'selected_samples': len(selected),
        'omitted_samples': len(entries) - len(selected),
        'pages': pages,
    }


def _write_rows(path, fields, rows):
    """행이 없어도 머리글이 있는 CSV를 쓴다."""
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_joined(path, joined, seeds):
    """표본별 오답 seed 수와 seed별 예측 라벨을 쓴다."""
    prediction_fields = [f'predicted_seed{seed}' for seed in seeds]
    _write_rows(
        path,
        ['sample_id', 'actual_label', 'wrong_seed_count', *prediction_fields],
        [
            {
                'sample_id': row['sample_id'],
                'actual_label': row['actual_label'],
                'wrong_seed_count': row['wrong_seed_count'],
                **{f'predicted_seed{seed}': row['predictions'][seed] for seed in seeds},
            }
            for row in joined
        ],
    )


def write_priority_errors(path, group_results, seeds, max_images):
    """그림 생략과 무관하게 그룹별 전체 오류 목록을 쓴다."""
    prediction_fields = [f'predicted_seed{seed}' for seed in seeds]
    rows = [
        {
            'group_id': result['summary']['group_id'],
            'rank': rank,
            'sample_id': entry['sample_id'],
            'actual_label': entry['actual_label'],
            'hit_seed_count': len(entry['hit_seeds']),
            'image_selected': rank <= max_images,
            **{f'predicted_seed{seed}': entry['predictions'][seed] for seed in seeds},
        }
        for result in group_results
        for rank, entry in enumerate(result['entries'], 1)
    ]
    _write_rows(
        path,
        ['group_id', 'rank', 'sample_id', 'actual_label', 'hit_seed_count',
         'image_selected', *prediction_fields],
        rows,
    )


def top_flows(seed_results, limit=TOP_FLOWS):
    """seed별 값을 나란히 보이는 오류 흐름을 고른다. 합계는 정렬에만 쓴다."""
    size = len(CLASS_NAMES)
    flows = []
    for actual in range(size):
        for predicted in range(size):
            counts = {
                result['seed']: result['metrics']['confusion_matrix'][actual][predicted]
                for result in seed_results
            }
            if actual != predicted and any(counts.values()):
                flows.append((-sum(counts.values()), actual, predicted, counts))

    return [
        {
            'actual_label': CLASS_NAMES[actual],
            'predicted_label': CLASS_NAMES[predicted],
            'seed_counts': counts,
        }
        for _, actual, predicted, counts in sorted(flows, key=lambda item: item[:3])[:limit]
    ]


def _percent(value):
    return f'{value * 100:.2f}'


def _class_item(metrics, name):
    return metrics['per_class'][CLASS_NAMES.index(name)]


def write_report(path, summary, group_results, seeds):
    """범위·재현 대조·지표·오류 흐름·선정 이미지를 Markdown으로 저장한다."""
    seed_columns = ' | '.join(f'seed{seed}' for seed in seeds)
    seed_rule = ' ---: |' * len(seeds)
    cache = summary['inputs']['cache']

    lines = [
        '# ResNet18+D4 Validation 오분류 표본 분석',
        '',
        '## 범위',
        '',
        f"Validation {summary['validation_rows']:,}행만 재추론했습니다. Train·Test 배열은 "
        '열지 않았고 학습·재학습·앙상블은 없습니다.',
        '',
        '증강은 학습 때만 적용되었으며 재추론에는 적용하지 않았습니다. Validation은 checkpoint '
        '선택에 쓰였으므로 아래 수치는 일반화 성능이 아닙니다.',
        '',
    ]

    if not summary['complete_seed_set']:
        lines += [
            f"**확인용 부분 실행입니다.** 전체 seed가 아니라 {seeds}만 처리했습니다.",
            '',
        ]

    lines += [
        '## 입력 식별값',
        '',
        '해시는 혼용·변경 확인용이며 인증이 아닙니다.',
        '',
        f"- 분할 protocol: `{cache['split_protocol_id']}`",
        f"- 분할 manifest: `{cache['source_split_manifest_sha256']}`",
        f"- `cache.json`: `{cache['cache_sha256']}`",
        f"- 학습 코드: `{summary['inputs']['run_code_sha256']}`",
        f"- 분석 코드: `{summary['inputs']['analysis_code_sha256']}`",
        f"- 장치: `{summary['device']}`, 배치 크기: {summary['batch_size']}",
        '',
        '| seed | selected_epoch | `best_model.pt` | `training_report.json` |',
        '| ---: | ---: | --- | --- |',
    ]
    for item in summary['seeds']:
        lines.append(
            f"| {item['seed']} | {item['selected_epoch']} "
            f"| `{item['input_digests']['best_model.pt']}` "
            f"| `{item['input_digests']['training_report.json']}` |"
        )

    lines += [
        '',
        '## 재현 대조와 지표',
        '',
        '재추론 혼동행렬이 저장된 selected_epoch 혼동행렬과 정확히 같을 때만 "일치"입니다.',
        '',
        '| seed | 재현 대조 | Macro-F1 (%) | Scratch Precision (%) | Scratch Recall (%) '
        '| Random Precision (%) | Random Recall (%) |',
        '| ---: | --- |' + ' ---: |' * 5,
    ]
    for item in summary['seeds']:
        metrics = item['metrics']
        scratch = _class_item(metrics, 'Scratch')
        random_class = _class_item(metrics, 'Random')
        lines.append(
            f"| {item['seed']} | {'일치' if item['reproduction']['matched'] else '불일치'} "
            f"| {_percent(metrics['macro_f1'])} "
            f"| {_percent(scratch['precision'])} | {_percent(scratch['recall'])} "
            f"| {_percent(random_class['precision'])} | {_percent(random_class['recall'])} |"
        )

    lines += [
        '',
        '## 오류 흐름',
        '',
        f"seed 합계 상위 {TOP_FLOWS}개를 seed별로 표시합니다. 합계는 정렬에만 쓰며, 같은 표본의 "
        '반복 예측이라 독립 표본 수가 아닙니다.',
        '',
        f'| 실제 | 예측 | {seed_columns} |',
        f'| --- | --- |{seed_rule}',
    ]
    for flow in summary['top_flows']:
        counts = ' | '.join(str(flow['seed_counts'][seed]) for seed in seeds)
        lines.append(f"| {flow['actual_label']} | {flow['predicted_label']} | {counts} |")

    repeats = summary['repeats']
    lines += [
        '',
        '## 반복 오분류',
        '',
        f"한 seed 이상에서 틀린 서로 다른 표본(합집합): {repeats['union_wrong_samples']:,}개. "
        f"처리한 모든 seed에서 틀린 표본(교집합): {repeats['intersection_wrong_samples']:,}개.",
        '',
        '| 오답 seed 수 | 서로 다른 표본 수 |',
        '| ---: | ---: |',
    ]
    for count, samples in repeats['samples_by_wrong_seed_count'].items():
        lines.append(f'| {count} | {samples:,} |')

    lines += [
        '',
        '## 우선 검토 이미지',
        '',
        '선정 기준은 그룹별로 고정했습니다. 해당 오류가 발생한 seed 수 내림차순, 동률은 '
        '`sample_id` 오름차순입니다. 그림에서 생략된 표본을 포함한 전체 목록은 '
        '`priority_errors.csv`에 있습니다.',
        '',
        '이미지는 캐시의 128×128 모델 입력을 3색으로 합성한 것이며 원본 사진이 아닙니다. '
        '추가 회전·반전은 없습니다. 범례: '
        + ', '.join(f'{label} `{color}`' for _, color, label in MAP_LEGEND) + '.',
        '',
        '`None`은 데이터셋 라벨이며 최종 양품 판정이 아닙니다.',
        '',
    ]

    for result in group_results:
        group_summary = result['summary']
        images = result['images']
        lines += [
            f"### {group_title(result['group'])}",
            '',
            f"- 실제 {group_summary['actual_label']} 표본: "
            f"{group_summary['actual_label_samples']:,}개",
            f"- 서로 다른 오류 표본: {group_summary['distinct_samples']:,}개",
            f"- 모든 처리 seed에서 같은 조건으로 틀린 표본: "
            f"{group_summary['samples_in_every_seed']:,}개",
            '- seed별 오류 건수: ' + ', '.join(
                f'seed{seed} {count:,}건'
                for seed, count in group_summary['seed_error_counts'].items()
            ),
            f"- 그림 포함 {images['selected_samples']:,}개, 생략 {images['omitted_samples']:,}개",
            '',
        ]

        if not result['entries']:
            lines += ['이 그룹의 오류 표본은 없습니다.', '']
            continue

        for page in images['pages']:
            lines += [f"![{group_summary['group_id']}]({page})", '']

        lines += ['| 번호 | sample_id | 오류 seed 수 |', '| ---: | --- | ---: |']
        for rank, entry in enumerate(result['entries'][:images['selected_samples']], 1):
            lines.append(f"| {rank} | `{entry['sample_id']}` | {len(entry['hit_seeds'])} |")
        lines.append('')

    settings = summary['inputs']['training_config']
    lines += [
        '## 해석 구분',
        '',
        '### 실행 기록으로 확인된 사실',
        '',
        f"- 클래스 가중치 사용: {settings['class_weighted']}, "
        f"상한 {settings['max_class_weight']}. 오류 원인이라는 뜻은 아닙니다.",
        '- `predicted_probability`는 모델 softmax 출력이며 보정된 신뢰도가 아닙니다.',
        '',
        '### 시각적 관찰과 가설',
        '',
        '이미지를 사람이 확인한 뒤 작성합니다. 이 보고서는 라벨 오류나 가중치 문제를 판정하지 '
        '않으며, 예측을 근거로 원본 라벨을 수정하지 않습니다.',
        '',
    ]

    path.write_text('\n'.join(lines), encoding='utf-8')


def _set_aside(temporary, final, label):
    """임시 폴더를 지우지 않고 완료 경로와 구분되는 이름으로 보존한다."""
    for number in range(1, 1000):
        target = final.with_name(f'{final.name}.{label}_{number}')
        if not target.exists():
            temporary.rename(target)
            return target

    require(False, 'TOO_MANY_PARTIAL_ANALYSES')


def _run_seed(run, dataset, batch_size, device, temporary):
    """한 seed를 추론해 예측·지표·재현 대조를 저장한다."""
    seed = run['seed']
    model = _restore_model(run['checkpoint'], device)
    rows = predict_validation(
        model,
        dataset,
        batch_size,
        device,
        temporary / f'predictions_seed{seed}.csv',
    )
    require_all_samples(rows, dataset)

    recomputed = confusion_from_rows(rows)
    result = {
        'seed': seed,
        'selected_epoch': run['selected_epoch'],
        'input_digests': run['digests'],
        'metrics': classification_metrics(recomputed),
        'stored_confusion_matrix': run['stored_confusion'],
        'reproduction': compare_confusion(recomputed, run['stored_confusion']),
    }
    _write_json(temporary / f'metrics_seed{seed}.json', result)
    return result, rows


def _require_inputs_unchanged(runs, cache_file, cache_sha256, run_code, analysis_code):
    """공개 직전에 분석 중 입력·코드가 바뀌지 않았는지 다시 확인한다."""
    require(
        file_digest(cache_file) == cache_sha256 and
        all(_input_digests(run['run_dir']) == run['digests'] for run in runs),
        'VALIDATION_ERRORS_INPUT_CHANGED',
    )
    require(
        code_fingerprint() == run_code and analysis_fingerprint() == analysis_code,
        'VALIDATION_ERRORS_INPUT_CHANGED',
    )


def _publish(temporary, output, summary):
    """완료 표식을 마지막에 쓰고 폴더 이름 변경으로 한 번에 공개한다."""
    require(not output.exists(), 'VALIDATION_ERRORS_OUTPUT_EXISTS')
    _write_json(temporary / 'SUCCESS.json', {
        'status': ANALYSIS_COMPLETE_STATUS,
        'split': 'validation',
        'seeds': summary['requested_seeds'],
        'reproduction_matched': True,
        'artifacts': {
            path.relative_to(temporary).as_posix(): file_digest(path)
            for path in sorted(temporary.rglob('*')) if path.is_file()
        },
    })
    temporary.rename(output)


def analyze(
    cache_dir,
    input_root,
    output_dir,
    seeds=DEFAULT_SEEDS,
    device='auto',
    batch_size=64,
    max_images=24,
    images_per_page=MAX_IMAGES_PER_PAGE,
):
    """입력을 검증하고 seed별 재추론·재현 대조·오분류 산출물 공개까지 수행한다."""
    seeds = tuple(seeds)
    _validate_options(seeds, batch_size, max_images, images_per_page)
    selected_device = _select_device(device)

    output = Path(output_dir).resolve()
    temporary = output.with_name(output.name + '.tmp')
    require(not output.exists(), 'VALIDATION_ERRORS_OUTPUT_EXISTS')
    require(not temporary.exists(), 'VALIDATION_ERRORS_PARTIAL_EXISTS')

    cache_root = Path(cache_dir).resolve(strict=True)
    cache_file = cache_root / 'cache.json'
    cache_sha256 = file_digest(cache_file)
    cache_report = read_json(cache_file)
    require(cache_report.get('status') == 'cache_complete', 'CACHE_NOT_COMPLETE')

    run_root = Path(input_root).resolve()
    run_code = code_fingerprint()
    analysis_code = analysis_fingerprint()

    # 저장 혼동행렬과 정확히 대조하도록 학습과 같은 결정적 연산을 쓴다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = WaferDataset(cache_root, split='validation')
    try:
        # 어떤 출력도 만들기 전에 세 실행을 모두 검증한다.
        runs = [
            load_run(
                run_root / run_name(MODEL_NAME, AUGMENTATION, seed),
                seed,
                dataset,
                cache_report,
                cache_sha256,
            )
            for seed in seeds
        ]

        temporary.mkdir(parents=True)
        try:
            seed_results = []
            seed_rows = {}
            for run in runs:
                result, rows = _run_seed(run, dataset, batch_size, selected_device, temporary)
                seed_results.append(result)
                seed_rows[run['seed']] = rows

            joined = join_seed_rows(seed_rows)

            # 재현되지 않으면 그림·보고서 없이 진단 자료만 별도 이름으로 남긴다.
            if not all(item['reproduction']['matched'] for item in seed_results):
                _write_json(temporary / 'MISMATCH.json', {
                    'status': ANALYSIS_MISMATCH_STATUS,
                    'device': str(selected_device),
                    'seeds': [
                        {'seed': item['seed'], **item['reproduction']}
                        for item in seed_results
                    ],
                })
                _set_aside(temporary, output, 'mismatch')
                require(False, 'VALIDATION_ERRORS_CONFUSION_MISMATCH')

            write_joined(temporary / 'joined_predictions.csv', joined, seeds)

            (temporary / 'images').mkdir()
            group_results = []
            for group in PRIORITY_GROUPS:
                group_summary, entries = group_errors(joined, group, seeds)
                images = render_group(
                    temporary, group, entries, dataset, seeds, max_images, images_per_page
                )
                group_results.append({
                    'group': group,
                    'summary': group_summary,
                    'entries': entries,
                    'images': images,
                })

            write_priority_errors(
                temporary / 'priority_errors.csv', group_results, seeds, max_images
            )

            contract = runs[0]['contract']
            summary = {
                'status': ANALYSIS_COMPLETE_STATUS,
                'split': 'validation',
                'model_name': MODEL_NAME,
                'augmentation': AUGMENTATION,
                'inference_augmentation': 'none',
                'ensemble': False,
                'requested_seeds': list(seeds),
                'complete_seed_set': set(seeds) == set(DEFAULT_SEEDS),
                'device': str(selected_device),
                'batch_size': batch_size,
                'validation_rows': len(dataset),
                'probability_note': 'softmax 출력이며 보정된 신뢰도가 아님',
                'hash_note': '혼용·변경 확인용이며 인증이 아님',
                'inputs': {
                    'cache': {
                        'cache_sha256': cache_sha256,
                        'split_protocol_id': dataset.protocol,
                        'source_split_manifest_sha256':
                            dataset.source_split_manifest_sha256,
                    },
                    'run_code_sha256': run_code['sha256'],
                    'analysis_code_sha256': analysis_code['sha256'],
                    'training_config': {
                        key: value for key, value in contract['training_config'].items()
                        if key != 'seed'
                    },
                },
                'seeds': seed_results,
                'top_flows': top_flows(seed_results),
                'repeats': repeat_summary(joined, seeds),
                'image_selection': {
                    'rule': 'hit_seed_count_desc_then_sample_id_asc',
                    'max_images_per_group': max_images,
                    'images_per_page': images_per_page,
                },
                'priority_groups': [
                    {**item['summary'], 'images': item['images']}
                    for item in group_results
                ],
            }

            _write_json(temporary / 'analysis.json', summary)
            write_report(temporary / 'report.md', summary, group_results, seeds)
            _require_inputs_unchanged(
                runs, cache_file, cache_sha256, run_code, analysis_code
            )
            _publish(temporary, output, summary)

        except BaseException:
            # 중단·실패한 결과는 지우지 않고 완료 경로와 구분되는 이름으로 남긴다.
            if temporary.exists():
                _set_aside(temporary, output, 'incomplete')
            raise

    finally:
        dataset.close()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='ResNet18+D4 Validation 재추론·오분류 분석 (학습·Test 없음)'
    )
    parser.add_argument('--cache-dir', default='data/processed/wm811k/input_cache/wm_resize_v1')
    parser.add_argument('--input-root', default='models/wm811k/compare_v2')
    parser.add_argument('--output-dir', default='reports/wm811k/validation_errors/resnet18_d4_v1')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-images-per-group', type=int, default=24)
    parser.add_argument('--images-per-page', type=int, default=MAX_IMAGES_PER_PAGE)
    args = parser.parse_args()

    try:
        with validation_stage('validation_errors'):
            result = analyze(
                args.cache_dir,
                args.input_root,
                args.output_dir,
                args.seeds,
                args.device,
                args.batch_size,
                args.max_images_per_group,
                args.images_per_page,
            )
    except ValueError as error:
        print('WM 오분류 분석 실패 코드: ' + str(error))
        if str(error) in FAILURE_GUIDANCE:
            print(FAILURE_GUIDANCE[str(error)])
        return 1
    except (OSError, RuntimeError) as error:
        print(f'WM 오분류 분석 실패: {type(error).__name__}')
        print('출력 경로 옆 .incomplete_N 폴더는 완료 결과가 아닙니다. 원인을 해결한 뒤 '
              '--output-dir에 새 경로를 지정하세요. GPU 메모리 부족이면 --batch-size를 줄이세요.')
        return 1

    print(json.dumps({
        'status': result['status'],
        'seeds': result['requested_seeds'],
        'complete_seed_set': result['complete_seed_set'],
        'reproduction': {
            item['seed']: item['reproduction']['matched'] for item in result['seeds']
        },
        'groups': {
            item['group_id']: {
                'distinct_samples': item['distinct_samples'],
                'omitted_samples': item['images']['omitted_samples'],
            }
            for item in result['priority_groups']
        },
        'report': str(Path(args.output_dir) / 'report.md'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

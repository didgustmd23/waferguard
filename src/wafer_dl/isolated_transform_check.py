"""기존 WM 변환 알고리즘을 격리 실행의 명시적 scratch·출력 상한에 연결한다.

check_transform의 변환·지표·예시·보고서 함수를 재사용한다.
과거 승인 코드의 지문을 보존하기 위해 기존 CLI 함수와 파일은 변경하지 않는다.
"""

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from scripts.wm_sandbox.export_canonical import OutputBudget
from .check_transform import index_split, iter_development_maps, save_examples, write_check_report
from .transform import load_transform_config, make_channels, preservation_metrics, transform_map
from .validate_canonical import read_json, require


class _CsvSink:
    """CSV 문자열을 제한 writer에 UTF-8 바이트로 기록한다."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        self.stream.write(text.encode("utf-8"))
        return len(text)


def run_isolated_transform_check(split_dir, output_dir, config_path, canonical_dir=None, progress=None,
                                 *, workspace, max_output_bytes):
    split_dir, output = Path(split_dir).resolve(strict=True), Path(output_dir).resolve()
    source = read_json(split_dir / 'split.json')
    root = Path(canonical_dir or source.get('canonical_root', '')).resolve(strict=True)
    require(not output.exists() and all(not output.is_relative_to(path) and not path.is_relative_to(output)
                                       for path in (root, split_dir)), 'INVALID_TRANSFORM_OUTPUT_DIRECTORY')
    config = load_transform_config(config_path)
    workspace = Path(workspace)
    require(workspace.is_dir() and not workspace.is_symlink() and not any(workspace.iterdir())
            and not workspace.resolve().is_relative_to(root)
            and not workspace.resolve().is_relative_to(split_dir), 'INVALID_TRANSFORM_WORKSPACE')
    budget = OutputBudget(max_output_bytes)
    with nullcontext(str(workspace)) as temporary:
        database = sqlite3.connect(str(Path(temporary) / 'manifest.sqlite'))
        try:
            split_report = index_split(root, split_dir, database)
            expected = split_report['split_counts']['train'] + split_report['split_counts']['validation']
            output.mkdir(parents=True, exist_ok=False)
            aggregates, examples = {}, {}
            fields = ['sample_id', 'split', 'class', 'original_height', 'original_width', 'resized_height',
                      'resized_width', 'scale', 'top', 'left', 'bottom', 'right',
                      'original_valid_dies', 'original_failed_dies', 'transformed_valid_pixels',
                      'transformed_failed_pixels', 'original_failed_ratio', 'transformed_failed_ratio',
                      'absolute_ratio_delta', 'failed_region_lost', 'valid_region_lost']
            checked = 0
            with budget.open(output / 'preservation.csv') as stream:
                stream.write(b'\xef\xbb\xbf')
                writer = csv.DictWriter(_CsvSink(stream), fieldnames=fields)
                writer.writeheader()
                for row, split, original in iter_development_maps(root, database):
                    transformed, geometry = transform_map(original, config)
                    channels = make_channels(transformed)
                    require(channels.shape == (2, *config['target_size']) and channels.flags.c_contiguous and
                            np.all(channels[0] <= channels[1]), 'INVALID_CHANNEL_CONTRACT')
                    metrics = preservation_metrics(original, transformed)
                    writer.writerow({'sample_id': row['sample_id'], 'split': split,
                                     'class': row['pattern_label'], **geometry, **metrics})
                    key = (split, row['pattern_label'])
                    item = aggregates.setdefault(key, {'split': split, 'class': row['pattern_label'], 'rows': 0,
                        'failed_region_lost': 0, 'valid_region_lost': 0, 'delta_sum': 0., 'delta_count': 0,
                        'max_absolute_ratio_delta': None})
                    item['rows'] += 1
                    for name in ('failed_region_lost', 'valid_region_lost'):
                        item[name] += int(metrics[name])
                    delta = metrics['absolute_ratio_delta']
                    if delta is not None:
                        item['delta_sum'] += delta
                        item['delta_count'] += 1
                        item['max_absolute_ratio_delta'] = max(item['max_absolute_ratio_delta'] or 0., delta)
                    # 클래스별 첫 사례와 전체 개발 구간의 크기·종횡비 극단값을 보관한다.
                    label = row['pattern_label']
                    if label not in examples:
                        examples[label] = (0, original.copy(), transformed.copy())
                    for name, score in (('Smallest map', -original.size), ('Largest map', original.size),
                                        ('Extreme aspect', max(original.shape) / min(original.shape))):
                        if name not in examples or score > examples[name][0]:
                            examples[name] = (score, original.copy(), transformed.copy())
                    if metrics['failed_region_lost'] and 'First lost defect' not in examples:
                        examples['First lost defect'] = (0, original.copy(), transformed.copy())
                    checked += 1
                    if progress and (checked % 1000 == 0 or checked == expected):
                        progress(checked, expected)
            require(checked == expected, 'TRANSFORM_ROW_COUNT_MISMATCH')
            classes = []
            for key in sorted(aggregates):
                item = aggregates[key]
                item['mean_absolute_ratio_delta'] = item.pop('delta_sum') / item['delta_count'] if item['delta_count'] else None
                item['ratio_comparison_rows'] = item.pop('delta_count')
                classes.append(item)
            review = any(item['failed_region_lost'] or item['valid_region_lost'] for item in classes)
            report = {'status': 'transform_review_required' if review else 'transform_check_passed',
                      'config': config, 'split_protocol_id': split_report['split_protocol_id'],
                      'rows_checked': checked, 'test_maps_read': False, 'training_ready': False,
                      'visual_review_required': True, 'classes': classes}
            save_examples(examples, output)
            write_check_report(report, output)
            (output / 'SUCCESS.json').write_text(json.dumps({'status': 'transform_check_complete',
                'review_required': review, 'training_ready': False}), encoding='utf-8')
        finally:
            database.close()
    require(sum(path.stat().st_size for path in output.iterdir()) <= max_output_bytes,
            'TRANSFORM_OUTPUT_LIMIT_EXCEEDED')
    return {'status': report['status'], 'training_ready': False}

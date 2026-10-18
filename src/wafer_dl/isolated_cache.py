"""기존 캐시 writer를 재사용하되 격리 scratch와 출력 예산을 명시하는 실행 코어."""

import json
from pathlib import Path
import sqlite3

from .cache import CacheWriter, file_digest, validate_check, write_cache_report
from .check_transform import index_split, iter_split_maps
from .transform import preservation_metrics, transform_map
from .validate_canonical import CLASS_NAMES, require


def estimate_cache_bytes(split_report, transform_config):
    """픽셀·색인·헤더·보고서 여유를 포함한 보수적 상한. RAM 제한과는 별개다."""
    rows = sum(split_report['split_counts'][name] for name in ('train', 'validation'))
    height, width = transform_config['target_size']
    return rows * (height * width + 1280) + 1024 * 1024


def build_isolated_cache(split_dir, check_dir, output_dir, canonical_dir,
                         transform_config, cache_config, *, workspace, max_output_bytes, diagnostic=None):
    """확정 Train·Validation만 새 캐시에 기록한다. Test 복원·학습·원본 수정은 없다.

    tempfile의 별도 ACL을 상속하는 대신 이미 승인한 scratch의 빈 하위 폴더를 받는다.
    기존 CacheWriter의 숫자 전용 NPY 저장·자체 재검증과 보고서 형식을 유지한다.
    """
    output, scratch = Path(output_dir), Path(workspace)
    roots = tuple(Path(value).resolve(strict=True) for value in (split_dir, check_dir, canonical_dir))
    require(scratch.is_dir() and not any(scratch.iterdir()), 'CACHE_WORKSPACE_NOT_EMPTY')
    require(not output.exists() and all(not output.resolve().is_relative_to(root) and
            not root.is_relative_to(output.resolve()) for root in roots), 'INVALID_CACHE_OUTPUT_DIRECTORY')

    database = sqlite3.connect(str(scratch / 'manifest.sqlite'))
    writers = {}

    def mark(stage):
        # 행 값·경로 대신 고정 단계명만 격리 worker의 진단 기록에 전달한다.
        if diagnostic is not None:
            diagnostic(stage)

    try:
        mark('INDEX_SPLIT')
        split_report = index_split(roots[2], roots[0], database)
        mark('VALIDATE_CHECK')
        validate_check(roots[1], split_report, transform_config, approved=True)
        require(estimate_cache_bytes(split_report, transform_config) <= max_output_bytes,
                'CACHE_OUTPUT_BUDGET_TOO_SMALL')
        output.mkdir(parents=True)
        mark('OPEN_CACHE_WRITERS')

        for name in ('train', 'validation'):
            writers[name] = CacheWriter(output, name, transform_config['target_size'], cache_config)

        checked = 0
        mark('STREAM_DEVELOPMENT_MAPS')

        for row, name, original in iter_split_maps(roots[2], database, ('train', 'validation')):
            # 색상·새 보간법을 도입하지 않고 기존 최근접 변환과 영역 소실 검사를 재사용한다.
            transformed, _ = transform_map(original, transform_config)
            metrics = preservation_metrics(original, transformed)
            require(not metrics['valid_region_lost'] and not metrics['failed_region_lost'],
                    'CACHE_PATTERN_LOSS_REVIEW_REQUIRED')
            require(type(row['sample_id']) is str and len(row['sample_id'].encode('utf-8')) <= 128,
                    'CACHE_SAMPLE_ID_TOO_LONG')
            writers[name].append(row, transformed)
            checked += 1

        expected = sum(split_report['split_counts'][name] for name in writers)
        require(checked == expected, 'CACHE_ROW_COUNT_MISMATCH')
        mark('FLUSH_CACHE')
        saved = {name: writer.finish() for name, writer in writers.items()}

        for name, item in saved.items():
            require(item['class_counts'] == split_report['class_counts'][name] and
                    item['rows'] == split_report['split_counts'][name], 'CACHE_CLASS_COUNT_MISMATCH')

        counts = split_report['class_counts']['train']
        result = {
            'status': 'cache_complete', 'cache_version': cache_config['cache_version'],
            'dataset_version': split_report['dataset_version'],
            'split_protocol_id': split_report['split_protocol_id'],
            'source_split_manifest_sha256': file_digest(roots[0] / 'split_manifest.jsonl'),
            'transform_config': transform_config, 'cache_config': cache_config,
            'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
            'splits': saved, 'test_maps_read': False, 'visual_review_confirmed': True,
            'training_ready': False, 'train_class_counts': counts,
            'suggested_train_class_weights': [sum(counts) / (9 * count) for count in counts],
        }
        mark('WRITE_CACHE_REPORT')
        write_cache_report(result, output)
        (output / 'SUCCESS.json').write_text(json.dumps({
            'status': 'cache_complete', 'split_protocol_id': result['split_protocol_id'],
            'training_ready': False,
        }), encoding='utf-8')

        # 이 상한은 산출물 예산이다. 디스크 자체를 제한하는 OS quota는 아니다.
        require(sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) <= max_output_bytes,
                'CACHE_OUTPUT_BUDGET_EXCEEDED')
        return result
    finally:
        for writer in writers.values():
            writer.close()

        database.close()

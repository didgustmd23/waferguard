"""캐시의 숫자 전용 헤더·바이트·색인 연결을 검사한다. pickle·모델은 열지 않는다."""

import ast
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import struct
import tempfile

from .artifacts import _is_link, _root
from .wm_canonical_results import _read_small as _read_file
from .wm_canonical_contract import digest, file_hash
from .worker_contract import _decode


POLICY_VERSION = 'wm_development_cache_v1'
PARTITIONS = ('train', 'validation')


def cache_files(directory, maximum):
    """허용한 두 구간과 고정 관리 파일 이외의 항목을 거부한다."""
    # worker는 승인된 output 아래만 접근한다. 호스트 상위 폴더를 거슬러
    # 검사하지 않는다. 부모의 inspect_cache는 별도로 _root를 적용한다.
    root = Path(directory)

    if _is_link(root) or not root.is_dir():
        raise ValueError('캐시 결과 폴더가 올바르지 않습니다.')
    management = {'cache.json', 'cache.md', 'SUCCESS.json'}
    expected = management | set(PARTITIONS) | ({'result.json'} if (root / 'result.json').exists() else set())

    if {path.name for path in root.iterdir()} != expected:
        raise ValueError('캐시 파일 목록이 올바르지 않습니다.')

    paths = [root / name for name in sorted(expected - set(PARTITIONS))]

    for name in PARTITIONS:
        folder = root / name

        if _is_link(folder) or not folder.is_dir():
            raise ValueError('캐시 구간은 링크가 아닌 폴더여야 합니다.')

        children = list(folder.iterdir())

        if not 1 <= len(children) <= 2049 or not (folder / 'index.jsonl').is_file():
            raise ValueError('캐시 파일 개수·색인이 올바르지 않습니다.')

        for path in children:
            if path.name != 'index.jsonl' and re.fullmatch(r'shard_[0-9]{6}\.npy', path.name) is None:
                raise ValueError('캐시에 허용하지 않은 경로가 있습니다.')

        paths.extend(children)

    records, total = {}, 0

    for path in sorted(paths):
        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError('캐시에는 일반 파일만 허용합니다.')

        record = file_hash(path, maximum - total)
        total += record['size']
        records[path.relative_to(root).as_posix()] = record

    return records


def check_numeric_shard(path, rows, size):
    """NumPy import 없이 uint8 C-order NPY 헤더와 범주 바이트만 제한적으로 읽는다."""
    with path.open('rb') as stream:
        if stream.read(8) != b'\x93NUMPY\x01\x00':
            raise ValueError('숫자 캐시는 고정 NPY v1 형식이어야 합니다.')

        raw = stream.read(2)

        if len(raw) != 2:
            raise ValueError('NPY 헤더가 잘렸습니다.')

        length = struct.unpack('<H', raw)[0]

        if not 1 <= length <= 10000:
            raise ValueError('NPY 헤더 상한을 초과했습니다.')

        header = ast.literal_eval(stream.read(length).decode('ascii').strip())

        if (type(header) is not dict or set(header) != {'descr', 'fortran_order', 'shape'}
            or header['descr'] != '|u1' or header['fortran_order'] is not False
            or header['shape'] != (rows, *size)
            or any(type(value) is not int for value in header['shape'])):
            raise ValueError('NPY dtype·shape·메모리 순서가 승인 계약과 다릅니다.')

        pixels = size[0] * size[1]

        for _ in range(rows):
            data = stream.read(pixels)

            if len(data) != pixels or not set(data) <= {0, 1, 2} or not any(data):
                raise ValueError('캐시 픽셀 범주·유효 영역이 올바르지 않습니다.')

        if stream.read(1):
            raise ValueError('NPY에 허용되지 않은 추가 바이트가 있습니다.')


def _check_linkage(root, report, split_directory):
    """확정 manifest와 색인을 SQLite로 대조한다. 전체 맵·전체 색인을 RAM에 올리지 않는다."""
    with tempfile.TemporaryDirectory(prefix='wm_cache_verify_') as temporary:
        with closing(sqlite3.connect(str(Path(temporary) / 'rows.sqlite'))) as database:
            database.execute('CREATE TABLE rows(id TEXT PRIMARY KEY, partition TEXT, label INTEGER, seen INTEGER DEFAULT 0)')

            with (Path(split_directory) / 'split_manifest.jsonl').open('rb') as stream:
                while True:
                    raw = stream.readline(65537)

                    if not raw:
                        break

                    if len(raw) > 65536:
                        raise ValueError('분할 manifest의 행 크기가 상한을 초과했습니다.')

                    row = _decode(raw.decode('utf-8'))

                    if row['split'] in PARTITIONS:
                        database.execute('INSERT INTO rows(id,partition,label) VALUES(?,?,?)',
                                         (row['sample_id'], row['split'], row['class_index']))

            expected_paths = {'cache.json', 'cache.md', 'SUCCESS.json'}

            for name in PARTITIONS:
                info = report['splits'][name]
                counts, total = [0] * 9, 0
                expected_paths.add(f'{name}/index.jsonl')

                with (root / name / 'index.jsonl').open('rb') as index:
                    for number, shard in enumerate(info['shards']):
                        relative = f'{name}/shard_{number:06d}.npy'

                        if (set(shard) != {'path', 'rows', 'bytes', 'sha256'} or shard['path'] != relative
                            or type(shard['rows']) is not int or not 1 <= shard['rows'] <= report['cache_config']['shard_max_rows']):
                            raise ValueError('캐시 shard 목록이 올바르지 않습니다.')

                        expected_paths.add(relative)
                        check_numeric_shard(root / relative, shard['rows'], report['transform_config']['target_size'])

                        for offset in range(shard['rows']):
                            raw = index.readline(1025)

                            if len(raw) > 1024 or not raw.endswith(b'\n'):
                                raise ValueError('캐시 색인 행 크기·개수가 올바르지 않습니다.')

                            record = _decode(raw.decode('utf-8'))
                            label = record.get('class_index')

                            if (set(record) != {'sample_id', 'class_index', 'pattern_label', 'split', 'shard', 'offset'}
                                or type(label) is not int or not 0 <= label < 9
                                or record['pattern_label'] != report['class_mapping'][str(label)]
                                or record['split'] != name or record['shard'] != relative
                                or type(record['offset']) is not int or record['offset'] != offset
                                or type(record['sample_id']) is not str or len(record['sample_id'].encode()) > 128):
                                raise ValueError('캐시 색인의 label·구간·위치가 올바르지 않습니다.')

                            changed = database.execute('UPDATE rows SET seen=1 WHERE id=? AND partition=? AND label=? AND seen=0',
                                                       (record['sample_id'], name, label)).rowcount

                            if changed != 1:
                                raise ValueError('캐시 색인에 중복·다른 구간·라벨 변경이 있습니다.')

                            counts[label] += 1
                            total += 1

                    if index.read(1) or total != info['rows'] or counts != info['class_counts']:
                        raise ValueError('캐시 색인 집계가 보고서와 다릅니다.')

            if database.execute('SELECT COUNT(*) FROM rows WHERE seen=0').fetchone()[0]:
                raise ValueError('캐시에 확정된 개발 구간 행이 누락되었습니다.')

            return expected_paths


def inspect_cache(directory, request, contract, evidence_id, evidence):
    """요청·통계·행 연결·숫자 저장 형식만 검증한다. 원본 변환 수학을 재실행하지 않는다."""
    root = _root(directory)
    files = cache_files(root, contract.max_output_bytes)
    envelope = json.loads(_read_file(root / 'result.json', 1024 * 1024), object_pairs_hook=_unique_pairs)
    core_files = {name: value for name, value in files.items() if name != 'result.json'}

    if envelope != {'version': POLICY_VERSION, 'request_id': request.request_id, 'source_id': request.source_id,
                    'config_id': request.config_id, 'evidence_id': evidence_id, 'status': 'cache_complete', 'files': core_files}:
        raise ValueError('캐시 결과가 현재 승인 요청과 다릅니다.')

    from src.preprocessing.canonical_data import decode_json

    report = decode_json(_read_file(root / 'cache.json', 1024 * 1024))
    split = evidence['split_evidence']
    baseline = split['split_report']

    from src.wafer_dl.validate_canonical import CLASS_NAMES

    if (report.get('status') != 'cache_complete' or report.get('cache_version') != 'wm_input_cache_v1'
        or report.get('dataset_version') != baseline['dataset_version']
        or report.get('split_protocol_id') != baseline['split_protocol_id']
        or report.get('source_split_manifest_sha256') != split['split_files']['split_manifest.jsonl']['sha256']
        or report.get('transform_config') != split['transform_config']
        or report.get('cache_config') != evidence['cache_config']
        or report.get('class_mapping') != {str(i): name for i, name in enumerate(CLASS_NAMES)}
        or report.get('test_maps_read') is not False or report.get('training_ready') is not False
        or report.get('visual_review_confirmed') is not True
        or type(report.get('splits')) is not dict or set(report['splits']) != set(PARTITIONS)):
        raise ValueError('캐시 보고서의 원본·분할·변환·범위가 다릅니다.')

    counts = baseline['class_counts']['train']
    weights = [sum(counts) / (9 * count) for count in counts]

    if report.get('train_class_counts') != counts or report.get('suggested_train_class_weights') != weights:
        raise ValueError('캐시의 Train 클래스 수·가중치 참고값이 다릅니다.')

    for name in PARTITIONS:
        info = report['splits'][name]

        if (info['rows'] != baseline['split_counts'][name] or info['class_counts'] != baseline['class_counts'][name]
            or info['index'] != f'{name}/index.jsonl' or info['index_sha256'] != files[info['index']]['sha256']
            or type(info['shards']) is not list or not 1 <= len(info['shards']) <= 2048):
            raise ValueError('캐시의 행·클래스·파일 집계가 확정 분할과 다릅니다.')

        for shard in info['shards']:
            if files.get(shard['path']) != {'size': shard['bytes'], 'sha256': shard['sha256']}:
                raise ValueError('캐시 shard 해시·크기가 다릅니다.')

    completion = _decode(_read_file(root / 'SUCCESS.json', 4096).decode('utf-8'))

    if completion != {'status': 'cache_complete', 'split_protocol_id': report['split_protocol_id'], 'training_ready': False}:
        raise ValueError('캐시 완료 표식이 올바르지 않습니다.')

    expected_paths = _check_linkage(root, report, split['split_directory'])

    if set(core_files) != expected_paths or cache_files(root, contract.max_output_bytes) != files:
        raise ValueError('캐시 검증 중 파일 목록·내용이 변경되었습니다.')

    _read_file(root / 'cache.md', 65536)
    return {'status': 'cache_complete', 'test_maps_read': False, 'training_ready': False,
            'evidence_id': evidence_id, 'files': files}


def _unique_pairs(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError('캐시 메타데이터의 중복 키는 허용하지 않습니다.')

        result[key] = value

    return result

"""숫자 전용 canonical 검증기. 원본 pickle·모델 학습은 실행하지 않는다."""

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import zipfile

import numpy as np

CLASS_NAMES = ('None', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
               'Loc', 'Near-Full', 'Random', 'Scratch')
MAP_ERRORS = {'INVALID_MAP_STRUCTURE', 'INVALID_MAP_SIZE_OR_DTYPE',
              'INVALID_MAP_VALUES', 'NO_VALID_DIE'}


class ValidationError(ValueError):
    """입력 값·경로를 노출하지 않는 정형 실패 코드."""


# ==========================================
# 검증 단계와 오류 종류만 보존하는 진단 경계
# - 파일명·내용·원문 예외·traceback은 보고서에 넣지 않음
# - 동일한 읽기 실패라도 경로·권한·ZIP·배열·DB 오류를 구분
# ==========================================
@contextmanager
def validation_stage(stage):
    try:
        yield
    except ValidationError as error:
        if not hasattr(error, 'stage'):
            error.stage = stage
        raise
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error,
            zipfile.BadZipFile, EOFError, NotImplementedError) as error:
        if isinstance(error, FileNotFoundError):
            code = 'FILE_OR_DIRECTORY_NOT_FOUND'
        elif isinstance(error, PermissionError):
            code = 'FILE_ACCESS_DENIED'
        elif isinstance(error, sqlite3.Error):
            code = 'TEMP_DATABASE_ERROR'
        elif isinstance(error, zipfile.BadZipFile):
            code = 'INVALID_NPZ_ARCHIVE'
        elif isinstance(error, EOFError):
            code = 'TRUNCATED_ARRAY'
        elif isinstance(error, OSError):
            code = 'FILE_IO_ERROR'
        elif isinstance(error, NotImplementedError):
            code = 'UNSUPPORTED_ARCHIVE_COMPRESSION'
        else:
            code = 'INVALID_ARTIFACT_STRUCTURE'
        failure = ValidationError(code)
        failure.stage = stage
        raise failure from None


# ==========================================
# 검증 실패를 원문 데이터 없는 코드로 통일
# ==========================================
def require(condition, code):
    if not condition:
        raise ValidationError(code)


# ==========================================
# 비신뢰 JSON의 문법·중복 키·비유한 숫자 검사
# - 입력 객체를 실행하거나 원문을 오류에 삽입하지 않음
# ==========================================
def decode_json(data):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result

    def reject_constant(_):
        raise ValidationError('NONFINITE_JSON')

    try:
        result = json.loads(data, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
        require(type(result) is dict, 'INVALID_JSON_OBJECT')
        return result
    except (ValueError, RecursionError, UnicodeError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError('INVALID_JSON') from None


def read_json(path):
    # 관리용 문서는 크기를 제한하고 입력 루트 밖 링크를 거부한다.
    path = Path(path)
    # 정해진 관리 파일 역할만 오류 코드에 넣고 사용자 경로는 노출하지 않는다.
    role = {'SUCCESS.json': 'SUCCESS', 'conversion.json': 'CONVERSION',
            'provenance.json': 'PROVENANCE', 'validation.json': 'VALIDATION'}.get(path.name, 'JSON')
    require(path.exists(), 'MISSING_' + role + '_FILE')
    require(path.is_file(), 'NOT_A_FILE_' + role)
    require(path.resolve().is_relative_to(path.parent.resolve()), 'UNSAFE_' + role + '_PATH')
    require(path.stat().st_size <= 4_000_000, 'OVERSIZED_' + role + '_FILE')
    return decode_json(path.read_bytes())


# ==========================================
# 완료 표식·변환 단계·클래스 매핑 검증
# - 현재 exporter의 분할 전 계약만 지원
# - 원본 해시는 기록 형식만 확인하며 원본을 다시 읽지 않음
# ==========================================
def validate_metadata(root):
    success = read_json(root / 'SUCCESS.json')
    summary = read_json(root / 'conversion.json')
    provenance = read_json(root / 'provenance.json')
    for record in (success, summary):
        require(record.get('status') == 'canonical_export_complete' and
                record.get('training_ready') is False, 'INVALID_COMPLETION_CONTRACT')
    require(summary.get('split_complete') is False and
            summary.get('duplicate_audit_complete') is False, 'UNSUPPORTED_STAGE')
    require(type(summary.get('row_count')) is int and 0 < summary['row_count'] <= 2_000_000,
            'INVALID_ROW_COUNT')
    require(type(summary.get('shard_count')) is int and
            0 <= summary['shard_count'] <= summary['row_count'], 'INVALID_SHARD_COUNT')
    require(summary.get('class_mapping') == {str(i): name for i, name in enumerate(CLASS_NAMES)},
            'INVALID_CLASS_MAPPING')
    require(type(provenance.get('source_sha256')) is str and
            re.fullmatch('[0-9a-f]{64}', provenance['source_sha256']), 'INVALID_PROVENANCE')
    require(type(provenance.get('source_bytes')) is int and provenance['source_bytes'] > 0,
            'INVALID_PROVENANCE')
    for field in ('counts', 'reason_counts', 'class_counts'):
        require(type(summary.get(field)) is dict and all(type(value) is int and value >= 0
                for value in summary[field].values()), 'INVALID_SUMMARY_COUNTS')
    return summary, 'wm811k_' + provenance['source_sha256']

# ==========================================
# manifest를 한 행씩 검사하고 임시 SQLite에 색인화
# - 전체 행의 Python 객체를 한꺼번에 메모리에 보관하지 않음
# - 행 위치·ID·라벨·격리 사유·배열 참조·집계를 대조
# ==========================================
def validate_manifest(root, summary, version, database):
    database.execute('CREATE TABLE rows (position INTEGER PRIMARY KEY, shard TEXT, '
                     'map_key TEXT UNIQUE, lot TEXT, wafer TEXT, hash TEXT, label INTEGER, '
                     'accepted INTEGER, record TEXT)')
    counts, reasons, classes = Counter(), Counter(), Counter()
    path = root / 'manifest.jsonl'
    require(path.is_file() and path.resolve().is_relative_to(root), 'MISSING_OR_UNSAFE_MANIFEST')
    count = 0
    checked_shards = set()

    with path.open('rb') as stream:
        while True:
            line = stream.readline(65_537)

            if not line:
                break

            require(len(line) <= 65_536, 'OVERSIZED_MANIFEST_ROW')

            row = decode_json(line)

            require(type(row.get('source_row_position')) is int and
                    row['source_row_position'] == count, 'INVALID_ROW_POSITION')
            
            require(row.get('dataset_version') == version and
                    row.get('sample_id') == f'{version}:{count}', 'INVALID_SAMPLE_ID')
            
            status, label, index = row.get('label_status'), row.get('pattern_label'), row.get('class_index')

            require(status in ('labeled', 'unlabeled', 'invalid'), 'INVALID_LABEL_STATUS')

            if status == 'labeled':
                require(type(index) is int and 0 <= index < len(CLASS_NAMES) and
                        label == CLASS_NAMES[index], 'LABEL_MAPPING_MISMATCH')
                classes[label] += 1
            else:
                require(label is None and index is None, 'INVALID_EMPTY_LABEL')

            lot = row.get('lot_id')
            require(lot is None or (type(lot) is str and lot.strip() == lot and bool(lot)), 'INVALID_LOT')

            why = row.get('reason_codes')
            require(type(why) is list and all(type(code) is str for code in why) and
                    len(why) == len(set(why)) and
                    set(why) <= MAP_ERRORS | {'MISSING_OR_INVALID_LOT', 'INVALID_LABEL'}, 'INVALID_REASONS')
            
            require(('MISSING_OR_INVALID_LOT' in why) == (lot is None) and
                    ('INVALID_LABEL' in why) == (status == 'invalid'), 'REASON_STATE_MISMATCH')
            
            quality = 'quarantined' if why else 'accepted'
            split = 'quarantined' if why else ('unlabeled' if status == 'unlabeled' else 'pending')
            require(row.get('quality_status') == quality and row.get('split') == split, 'INVALID_ROW_STATE')

            ref, shard, key = row.get('map_ref'), None, None

            if ref is None:
                require(bool(set(why) & MAP_ERRORS) and row.get('map_hash') is None, 'MISSING_MAP_REFERENCE')
            else:
                require(type(ref) is dict and set(ref) == {'shard', 'key'}, 'INVALID_MAP_REFERENCE')
                shard, key = ref['shard'], ref['key']
                require(type(shard) is str and re.fullmatch(r'maps/shard_[0-9]{6}\.npz', shard) and
                        key == f'map_{count:09d}', 'INVALID_MAP_REFERENCE')
            
                # 같은 shard의 경로를 행마다 파일 시스템에서 다시 확인하지 않는다.
                if shard not in checked_shards:
                    target = root / shard
                    require(target.is_file() and target.resolve().is_relative_to(root), 'UNSAFE_OR_MISSING_SHARD')
                    checked_shards.add(shard)

                require(not set(why) & MAP_ERRORS, 'UNEXPECTED_MAP_REFERENCE')
                require(type(row.get('map_hash')) is str and
                        re.fullmatch('[0-9a-f]{64}', row['map_hash']), 'INVALID_MAP_HASH')
                
            wafer = row.get('wafer_index')
            require(wafer is None or type(wafer) in (str, int, float), 'INVALID_WAFER_ID')

            try:
                database.execute('INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                 (count, shard, key, lot, json.dumps(wafer) if wafer is not None else None,
                                  row.get('map_hash'), index, int(quality == 'accepted'), json.dumps(row)))
            except (sqlite3.Error, OverflowError):
                raise ValidationError('INVALID_MANIFEST_INDEX') from None
            
            counts[quality] += 1
            counts[status] += 1
            reasons.update(why)
            count += 1
            require(count <= summary['row_count'], 'ROW_COUNT_MISMATCH')

    require(count == summary['row_count'], 'ROW_COUNT_MISMATCH')

    for expected, actual in ((summary['counts'], counts), (summary['reason_counts'], reasons),
                             (summary['class_counts'], classes)):
        require(expected == dict(actual), 'SUMMARY_COUNT_MISMATCH')

    database.execute('CREATE INDEX shard_index ON rows(shard)')
    database.commit()

# ==========================================
# 배열 헤더를 검사한 뒤 숫자 배열만 복원
# - 거대한 shape·object dtype을 메모리 할당 전에 거부
# - ZIP 파일을 디스크에 풀거나 pickle을 허용하지 않음
# ==========================================
def read_numeric_member(archive, member):
    require(0 < member.file_size <= 1_010_000, 'OVERSIZED_ARRAY')
    payload = archive.read(member)
    buffer = io.BytesIO(payload)
    version = np.lib.format.read_magic(buffer)
    require(version in ((1, 0), (2, 0)), 'UNSUPPORTED_NPY_VERSION')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _, dtype = reader(buffer, max_header_size=10_000)
    require(len(shape) == 2 and all(type(length) is int and 0 < length <= 1_000_000 for length in shape),
            'INVALID_ARRAY_SHAPE')
    require(shape[0] * shape[1] <= 1_000_000 and dtype == np.dtype('uint8'), 'INVALID_ARRAY_DTYPE_OR_SIZE')
    require(len(payload) - buffer.tell() == shape[0] * shape[1], 'INVALID_ARRAY_BYTES')
    
    return np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=10_000)


# ==========================================
# shard별 맵·참조·내용 식별값 재검증
# - ZIP 구성원의 중복·누락·추가·크기·CRC 검사
# - shape·해시·다이 수·비율을 원본 크기 배열에서 재계산
# ==========================================
def validate_shards(root, summary, database, progress=None):
    expected = [item[0] for item in database.execute('SELECT DISTINCT shard FROM rows WHERE shard IS NOT NULL ORDER BY shard')]
    require(len(expected) == summary['shard_count'], 'SHARD_COUNT_MISMATCH')
    require((root / 'maps').is_dir() and (root / 'maps').resolve().is_relative_to(root), 'UNSAFE_MAP_DIRECTORY')
    actual = {path.relative_to(root).as_posix() for path in (root / 'maps').glob('*.npz')}
    require(actual == set(expected), 'UNREFERENCED_OR_MISSING_SHARD')
    
    for number, shard in enumerate(expected, 1):
        path = root / shard
        require(path.stat().st_size <= 1_100_000_000, 'OVERSIZED_SHARD')
        rows = {item[0]: decode_json(item[1]) for item in database.execute(
            'SELECT map_key, record FROM rows WHERE shard = ?', (shard,))}
        require(1 <= len(rows) <= 1000, 'INVALID_SHARD_MEMBER_COUNT')
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            require(len(members) == len(rows) and len({item.filename for item in members}) == len(members),
                    'DUPLICATE_OR_EXTRA_ARRAY')
            require({item.filename for item in members} == {key + '.npy' for key in rows}, 'ARRAY_REFERENCE_MISMATCH')
            require(sum(item.file_size for item in members) <= 1_010_000_000, 'OVERSIZED_SHARD')
            for member in members:
                require(not member.flag_bits & 1, 'ENCRYPTED_ARRAY')
                array = read_numeric_member(archive, member)
                row = rows[member.filename[:-4]]
                require(np.isin(array, (0, 1, 2)).all() and np.any(array > 0), 'INVALID_MAP_VALUES')
                digest = hashlib.sha256(np.asarray(array.shape, dtype='<u8').tobytes())
                digest.update(np.ascontiguousarray(array).tobytes())
                valid, failed = int(np.count_nonzero(array)), int(np.count_nonzero(array == 2))
                require(row.get('map_hash') == digest.hexdigest(), 'MAP_HASH_MISMATCH')
                for field, value in (('map_height', array.shape[0]), ('map_width', array.shape[1]),
                                     ('valid_die_count', valid), ('failed_die_count', failed)):
                    require(type(row.get(field)) is int and row[field] == value, 'MAP_STATISTIC_MISMATCH')
                ratio = row.get('failed_die_ratio')
                require(type(ratio) in (int, float) and abs(ratio - failed / valid) <= 1e-12, 'MAP_RATIO_MISMATCH')
        if progress:
            progress(number, len(expected))
# ==========================================
# 동일 맵·웨이퍼 ID 충돌 및 Lot 연결 그룹 진단
# - 라벨 전파·다수결·자동 제외를 하지 않음
# - 미라벨도 동일 맵 연결 검사에 포함하여 평가 Lot 유출 방지
# ==========================================
def audit_duplicates(database):
    conflicts = database.execute('SELECT COUNT(*) FROM (SELECT hash FROM rows WHERE hash IS NOT NULL '
                                 'GROUP BY hash HAVING COUNT(DISTINCT label) > 1)').fetchone()[0]
    id_conflicts = database.execute('SELECT COUNT(*) FROM (SELECT lot, wafer FROM rows WHERE lot IS NOT NULL '
                                    'AND wafer IS NOT NULL GROUP BY lot, wafer HAVING COUNT(DISTINCT hash) > 1 '
                                    'OR COUNT(DISTINCT label) > 1)').fetchone()[0]
    duplicate_groups = database.execute('SELECT COUNT(*) FROM (SELECT hash FROM rows WHERE hash IS NOT NULL '
                                        'GROUP BY hash HAVING COUNT(*) > 1)').fetchone()[0]
    parent = {}

    def find(lot):
        # 경로 압축으로 여러 Lot의 연결 관계 조회 비용을 줄인다.
        parent.setdefault(lot, lot)
        while parent[lot] != lot:
            parent[lot] = parent[parent[lot]]
            lot = parent[lot]
        return lot

    previous_hash, first_lot = None, None
    for map_hash, lot in database.execute('SELECT hash, lot FROM rows WHERE hash IS NOT NULL '
                                         'AND lot IS NOT NULL ORDER BY hash, lot'):
        if map_hash != previous_hash:
            previous_hash, first_lot = map_hash, lot
        parent[find(lot)] = find(first_lot)
    groups, class_groups = Counter(), {index: set() for index in range(9)}
    for lot, label, count in database.execute('SELECT lot, label, COUNT(*) FROM rows WHERE accepted=1 '
                                             'AND label IS NOT NULL GROUP BY lot, label'):
        group = find(lot)
        groups[group] += count
        class_groups[label].add(group)
    return {'duplicate_map_groups': duplicate_groups, 'map_label_conflicts': conflicts,
            'wafer_id_conflicts': id_conflicts}, groups, class_groups
# ==========================================
# 분할 가능성의 필요조건 진단
# - 9개 클래스별 독립 그룹 3개 이상·충돌·거대 그룹 확인
# - 실제 분할 후보는 만들지 않으므로 가능하다고 확정하지 않음
# ==========================================
def assess_split_feasibility(audit, groups, class_groups, max_group_ratio=0.2):
    total = sum(groups.values())
    largest = max(groups.values(), default=0) / total if total else 0.0
    supported = all(len(class_groups[index]) >= 3 for index in range(9))
    clear = audit['map_label_conflicts'] == 0 and audit['wafer_id_conflicts'] == 0
    ready = bool(total) and supported and clear and largest <= max_group_ratio
    return {'status': 'preconditions_passed' if ready else 'review_required',
            'split_feasible': None, 'split_complete': False,
            'largest_group_ratio': largest, 'max_group_ratio': max_group_ratio,
            'class_group_counts': {str(index): len(value) for index, value in class_groups.items()},
            'note': '필요조건 진단이며 실제 분할 후보 생성·확정은 미실시입니다.'}


# ==========================================
# 저장된 진단 집계에서 검토 사유를 정형 문장으로 생성
# - 기존 validation.json도 다시 배열 검사 없이 설명 가능
# - Lot·맵·샘플 식별값과 실제 집계 수치는 출력하지 않음
# - 원본 문자열을 출력에 사용하지 않고 고정 문장만 선택
# ==========================================
def split_review_checks(report):
    audit, assessment = report.get('duplicate_audit'), report.get('split_assessment')
    require(type(audit) is dict and type(assessment) is dict, 'INVALID_SAVED_REPORT')
    for field in ('map_label_conflicts', 'wafer_id_conflicts'):
        require(type(audit.get(field)) is int and audit[field] >= 0, 'INVALID_SAVED_REPORT')
    counts = assessment.get('class_group_counts')
    require(type(counts) is dict and set(counts) == {str(i) for i in range(9)} and
            all(type(value) is int and value >= 0 for value in counts.values()), 'INVALID_SAVED_REPORT')
    largest, limit = assessment.get('largest_group_ratio'), assessment.get('max_group_ratio')
    require(type(largest) in (int, float) and 0 <= largest <= 1 and
            type(limit) in (int, float) and 0 < limit <= 1, 'INVALID_SAVED_REPORT')
    # 희소 클래스 이름·빈도까지 콘솔에 보내지 않고 조건 충족 여부만 안내한다.
    return [
        ('MAP_LABEL_CONFLICT', audit['map_label_conflicts'] == 0,
         '동일 맵의 명시적 라벨 충돌 없음'),
        ('WAFER_ID_CONFLICT', audit['wafer_id_conflicts'] == 0,
         '동일 Lot·웨이퍼 번호의 맵 또는 라벨 충돌 없음'),
        ('NO_LABELED_GROUPS', any(counts.values()),
         '지도학습에 사용할 유효 라벨 연결 그룹 존재'),
        ('INSUFFICIENT_CLASS_GROUPS', all(value >= 3 for value in counts.values()),
         '9개 클래스 각각 독립 연결 그룹 3개 이상 확보'),
        ('LARGE_CONNECTED_GROUP', largest <= limit,
         '최대 연결 그룹 비율이 설정된 검토 상한 이하'),
    ]


def print_split_review(report):
    # 통과·검토필요를 모두 표시하여 어떤 조건만 걸렸는지 구분한다.
    print('분할 조건별 진단:')
    for code, passed, message in split_review_checks(report):
        state = '통과' if passed else '검토 필요'
        print(f'  [{state}] {message}' + ('' if passed else f' ({code})'))
    print('검토 필요는 저장 파일 손상을 뜻하지 않으며, 자동 삭제·분할은 하지 않습니다.')
# ==========================================
# 전체 검증 실행
# - 임시 DB는 종료 시 정리하고 입력 데이터는 수정하지 않음
# - 저장 정합성과 중복·분할 진단을 별도 상태로 반환
# ==========================================
def validate_bundle(input_dir, progress=None, max_group_ratio=0.2, *, workspace=None):
    """canonical을 검사한다. 격리 실행은 권한이 상속된 빈 scratch 폴더를 전달할 수 있다.

    workspace를 생략한 기존 CLI·노트북은 종전 임시 폴더 수명 관리를 유지한다.
    명시한 workspace의 DB는 호출자 소유이며 자동 삭제하지 않는다.
    에이전트·OS 권한을 이 코어에서 변경하지 않는다.
    """
    require(0 < max_group_ratio <= 1, 'INVALID_GROUP_RATIO')
    with validation_stage('input_directory'):
        root = Path(input_dir).resolve(strict=True)
        require(root.is_dir(), 'INPUT_IS_NOT_DIRECTORY')
    with validation_stage('metadata'):
        summary, version = validate_metadata(root)
    # 바깥 경계는 임시 폴더 생성·삭제 실패를 잡고 안쪽 실패 단계는 유지한다.
    with validation_stage('temporary_workspace'):
        if workspace is None:
            temporary_context = tempfile.TemporaryDirectory(prefix='wm_validate_')
        else:
            directory = Path(workspace)
            require(directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir()),
                    'INVALID_VALIDATION_WORKSPACE')
            require(not directory.resolve().is_relative_to(root), 'INVALID_VALIDATION_WORKSPACE')
            temporary_context = nullcontext(str(directory))

        with temporary_context as temporary:
            with validation_stage('temporary_database'):
                database = sqlite3.connect(str(Path(temporary) / 'manifest.sqlite'))
            try:
                with validation_stage('manifest'):
                    validate_manifest(root, summary, version, database)
                with validation_stage('shards'):
                    validate_shards(root, summary, database, progress)
                with validation_stage('duplicate_audit'):
                    audit, groups, class_groups = audit_duplicates(database)
                with validation_stage('split_assessment'):
                    feasibility = assess_split_feasibility(audit, groups, class_groups, max_group_ratio)
            finally:
                database.close()
    return {'status': 'integrity_passed', 'integrity_passed': True,
            'training_ready': False, 'source_equivalence_verified': False,
            'is_security_certification': False, 'row_count': summary['row_count'],
            'shard_count': summary['shard_count'], 'duplicate_audit': audit,
            'split_assessment': feasibility}


def write_validation_report(report, output):
    # 새 폴더에만 보고서를 기록하고 기존 결과는 덮어쓰지 않는다.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'validation.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
    with (output / 'validation.md').open('x', encoding='utf-8') as stream:
        stream.write('# Canonical 검증 결과\n\n')
        stream.write(f"상태: `{report['status']}`\n\n")
        stream.write('상세 집계는 validation.json에서 로컬로 확인하세요.\n')
        stream.write('원본과의 완전 일치·무해성·학습 준비 완료 인증은 아닙니다.\n')
        if report.get('integrity_passed'):
            stream.write('\n## 분할 조건별 진단\n\n')
            for code, passed, message in split_review_checks(report):
                state = '통과' if passed else '검토 필요'
                stream.write(f'- [{state}] {message}' + ('' if passed else f' (`{code}`)') + '\n')


def main():
    # 명시적 입력 묶음만 검사하고 결과는 입력 바깥의 새 폴더에 저장한다.
    parser = argparse.ArgumentParser(description='canonical 숫자 배열·manifest 검증')
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--show-report', help='기존 validation.json의 분할 검토 사유만 출력')
    parser.add_argument('--max-group-ratio', type=float, default=0.2)
    args = parser.parse_args()
    if args.show_report:
        if args.input_dir or args.output_dir:
            parser.error('--show-report는 --input-dir/--output-dir와 함께 사용하지 마세요.')
        try:
            report = read_json(Path(args.show_report))
            require(report.get('integrity_passed') is True and report.get('status') == 'integrity_passed',
                    'REPORT_INTEGRITY_NOT_PASSED')
            print_split_review(report)
        except (ValidationError, OSError, TypeError, ValueError):
            parser.error('완료된 정합성 검증 보고서를 읽을 수 없거나 보고서 형식이 올바르지 않습니다.')
        return 0
    if not args.input_dir or not args.output_dir:
        parser.error('--input-dir와 --output-dir를 함께 지정하세요.')
    output = Path(args.output_dir).resolve()
    root = Path(args.input_dir).resolve()
    if output.is_relative_to(root) or output.exists():
        parser.error('입력 폴더 밖의 존재하지 않는 새 보고서 폴더를 지정하세요.')
    print('manifest 검증·색인화를 시작합니다. 상세 데이터는 출력하지 않습니다.', flush=True)
    try:
        report = validate_bundle(root, progress=lambda done, total:
                                 print(f'배열 검증 진행: {done}/{total} shard', flush=True),
                                 max_group_ratio=args.max_group_ratio)
    except ValidationError as error:
        report = {'status': 'integrity_failed', 'integrity_passed': False,
                  'training_ready': False, 'error_code': str(error),
                  'failed_stage': getattr(error, 'stage', 'configuration')}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, zipfile.BadZipFile, EOFError):
        report = {'status': 'integrity_failed', 'integrity_passed': False,
                  'training_ready': False, 'error_code': 'UNREADABLE_OR_INVALID_ARTIFACT',
                  'failed_stage': 'unclassified'}
    write_validation_report(report, output)
    print(report['status'])
    if report['integrity_passed']:
        print('분할 사전 진단: ' + report['split_assessment']['status'])
        print_split_review(report)
        print('분할·학습 준비는 아직 미완료입니다.')
    else:
        print('실패 단계: ' + report['failed_stage'])
        print('실패 코드: ' + report['error_code'])
    return 0 if report['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

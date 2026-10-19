"""원본을 보존하는 충돌 맵 제외·Lot/동일 맵 연결 그룹 분할."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from contextlib import nullcontext

import numpy as np

from .validate_canonical import (
    CLASS_NAMES, ValidationError, audit_duplicates, assess_split_feasibility,
    decode_json, read_json, require, validate_manifest, validate_metadata,
    validate_shards, validation_stage,
)

SPLITS = ('train', 'validation', 'test')


# ==========================================
# 분할 설정 계약 검증
# - 알 수 없는 키·비유한 수·bool을 숫자로 받는 경우를 거부
# - 후보 수는 고정 상한 내에서만 허용하고 실패 시 자동 확대하지 않음
# ==========================================
def load_split_config(path):
    config = read_json(Path(path))
    keys = {'protocol_version', 'ratios', 'seed', 'max_candidates',
            'max_group_ratio', 'rare_class_warning_count'}
    require(set(config) == keys and config['protocol_version'] == 'wm_group_split_v1', 'INVALID_SPLIT_CONFIG')
    ratios = config['ratios']
    require(type(ratios) is list and len(ratios) == 3 and
            all(type(value) in (int, float) and np.isfinite(value) and 0 < value < 1 for value in ratios)
            and abs(sum(ratios) - 1) <= 1e-12, 'INVALID_SPLIT_RATIOS')
    require(type(config['seed']) is int and 0 <= config['seed'] <= 2**32 - 1, 'INVALID_SPLIT_SEED')
    require(type(config['max_candidates']) is int and 1 <= config['max_candidates'] <= 32, 'INVALID_CANDIDATE_COUNT')
    require(type(config['max_group_ratio']) in (int, float) and
            0 < config['max_group_ratio'] <= 1, 'INVALID_GROUP_RATIO')
    require(type(config['rare_class_warning_count']) is int and
            config['rare_class_warning_count'] >= 1, 'INVALID_WARNING_COUNT')
    return config


# ==========================================
# 충돌 맵 전체 제외와 연결 그룹 구성
# - 원본 manifest 대신 임시 DB의 active 뷰만 대상으로 연결
# - accepted 여부와 무관하게 남은 미라벨·격리 행의 Lot/맵 연결도 보호
# - 그룹 ID는 원본 Lot 이름 대신 결정적인 순번을 사용
# ==========================================
def build_groups(database):
    database.execute('CREATE TABLE conflicts AS SELECT hash FROM rows WHERE hash IS NOT NULL '
                     'GROUP BY hash HAVING COUNT(DISTINCT label)>1')
    database.execute('CREATE UNIQUE INDEX conflicts_hash ON conflicts(hash)')
    database.execute('CREATE VIEW active AS SELECT * FROM rows WHERE hash IS NULL '
                     'OR hash NOT IN (SELECT hash FROM conflicts)')
    parent = {}

    def find(lot):
        parent.setdefault(lot, lot)
        while parent[lot] != lot:
            parent[lot] = parent[parent[lot]]
            lot = parent[lot]
        return lot

    previous, first = None, None
    for digest, lot in database.execute('SELECT hash,lot FROM active WHERE hash IS NOT NULL '
                                       'AND lot IS NOT NULL ORDER BY hash,lot'):
        if digest != previous:
            previous, first = digest, lot
        parent[find(lot)] = find(first)
    lots = [row[0] for row in database.execute('SELECT DISTINCT lot FROM active WHERE lot IS NOT NULL ORDER BY lot')]
    # 루트를 원본 순서가 아닌 정렬된 Lot 순서에서 처음 만난 순번으로 고정한다.
    identifiers = {}
    database.execute('CREATE TABLE lot_groups(lot TEXT PRIMARY KEY, gid INTEGER NOT NULL)')
    for lot in lots:
        root = find(lot)
        gid = identifiers.setdefault(root, len(identifiers))
        database.execute('INSERT INTO lot_groups VALUES (?,?)', (lot, gid))
    counts = np.zeros((len(identifiers), len(CLASS_NAMES)), dtype=np.int64)
    for gid, label, count in database.execute(
            'SELECT g.gid,a.label,COUNT(*) FROM active a JOIN lot_groups g ON a.lot=g.lot '
            'WHERE a.accepted=1 AND a.label IS NOT NULL GROUP BY g.gid,a.label'):
        counts[gid, label] = count
    # 그룹 크기·클래스 지원과 웨이퍼 ID 충돌 검사는 기존 검증 계약을 재사용한다.
    database.execute('ALTER TABLE rows RENAME TO original_rows')
    try:
        database.execute('CREATE VIEW rows AS SELECT * FROM original_rows WHERE hash IS NULL '
                         'OR hash NOT IN (SELECT hash FROM conflicts)')
        audit, groups, class_groups = audit_duplicates(database)
    finally:
        database.execute('DROP VIEW rows')
        database.execute('ALTER TABLE original_rows RENAME TO rows')
    return counts, audit, groups, class_groups


# ==========================================
# 고정 후보의 결정적 그룹 배정
# - 클래스별 전체 수로 정규화하여 다수 클래스의 점수 독점 완화
# - J=표본 비율 제곱편차 합 + 클래스별 비율 제곱편차 평균
# - 각 split에 9개 클래스가 있는 후보만 채택, 동점은 후보 번호 우선
# - 전역 최적화·정확한 비율·희소 클래스 안정성을 보장하지 않음
# ==========================================
def choose_assignment(counts, config, progress=None):
    totals = counts.sum(axis=0)
    require(np.all(totals > 0) and np.all((counts > 0).sum(axis=0) >= 3), 'INSUFFICIENT_CLASS_GROUPS')
    ratios = np.asarray(config['ratios'])
    sizes = counts.sum(axis=1)
    best = None
    candidates = []

    def objective(matrix):
        return float(np.square(matrix.sum(axis=1) / totals.sum() - ratios).sum() +
                     np.square(matrix / totals - ratios[:, None]).sum() / len(CLASS_NAMES))

    for candidate in range(config['max_candidates']):
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], candidate]))
        # 희소 클래스 기여가 크거나 큰 그룹을 우선하되 제한된 변동으로 후보를 만든다.
        priority = (counts / totals).max(axis=1) * rng.uniform(0.8, 1.2, len(counts))
        order = np.argsort(-priority, kind='stable')
        matrix = np.zeros((3, len(CLASS_NAMES)), dtype=np.int64)
        assignment = np.full(len(counts), -1, dtype=np.int8)
        remaining = (counts > 0).sum(axis=0)
        abandoned = False
        for gid in order:
            if not sizes[gid]:
                continue
            remaining -= counts[gid] > 0
            scores = []
            for split_index in range(3):
                matrix[split_index] += counts[gid]
                # 아직 없는 클래스의 split 수보다 남은 지원 그룹이 적어지는 배정은 금지한다.
                possible = np.all((matrix == 0).sum(axis=0) <= remaining)
                scores.append(objective(matrix) if possible else np.inf)
                matrix[split_index] -= counts[gid]
            if not np.isfinite(scores).any():
                abandoned = True
                break
            selected = int(np.argmin(scores))
            assignment[gid] = selected
            matrix[selected] += counts[gid]
        feasible = bool(not abandoned and np.all(matrix > 0))
        score = objective(matrix)
        candidates.append({'candidate': candidate, 'objective': score, 'all_classes_present': feasible})
        if feasible and (best is None or score < best[0]):
            best = (score, candidate, assignment.copy(), matrix.copy())
        if progress:
            progress(candidate + 1, config['max_candidates'])
    require(best is not None, 'NO_FEASIBLE_SPLIT_CANDIDATE')
    return best, candidates


def verify_assignments(database):
    # 미라벨도 포함한 남은 연결 관계가 여러 평가 구간으로 나뉘지 않았는지 검사한다.
    for field in ('lot', 'hash'):
        query = (f'SELECT COUNT(*) FROM (SELECT a.{field} FROM active a '
                 'JOIN lot_groups g ON a.lot=g.lot JOIN assignments s ON g.gid=s.gid '
                 f'WHERE a.{field} IS NOT NULL AND s.split IS NOT NULL GROUP BY a.{field} '
                 'HAVING COUNT(DISTINCT s.split)>1)')
        require(database.execute(query).fetchone()[0] == 0, 'SPLIT_GROUP_OVERLAP')


# ==========================================
# 전체 실행과 파생 manifest 저장
# - NPZ를 검증하지만 복사·resize·재학습하지 않음
# - 원본 모든 행을 유지하고 충돌 맵은 파생 manifest에서만 격리
# - 미라벨의 group_split은 평가 Lot 보호용이며 학습 허가가 아님
# - 완료 표식은 산출물 검증·저장 후 마지막에만 생성
# ==========================================
def run_split(input_dir, output_dir, config_path, approved=False, progress=None, *, workspace=None):
    require(approved is True, 'SPLIT_POLICY_APPROVAL_REQUIRED')
    root, output = Path(input_dir).resolve(strict=True), Path(output_dir).resolve()
    require(not output.exists() and not output.is_relative_to(root) and
            not root.is_relative_to(output), 'INVALID_SPLIT_OUTPUT_DIRECTORY')
    config = load_split_config(config_path)
    summary, version = validate_metadata(root)
    require(summary.get('label_normalization_version') == 'wm_label_v2', 'OUTDATED_LABEL_NORMALIZATION')
    # 격리 실행에서는 승인된 scratch의 상속 ACL을 유지한다.
    # 기존 CLI는 workspace를 생략하므로 이전 임시 폴더 동작을 그대로 사용한다.
    if workspace is not None:
        workspace = Path(workspace).resolve(strict=True)
        require(workspace.is_dir() and not any(workspace.iterdir()) and
                not workspace.is_relative_to(root) and not root.is_relative_to(workspace) and
                not workspace.is_relative_to(output) and not output.is_relative_to(workspace),
                'INVALID_SPLIT_WORKSPACE')

    temporary_scope = nullcontext(workspace) if workspace is not None else tempfile.TemporaryDirectory(prefix='wm_split_')
    with temporary_scope as temporary:
        database = sqlite3.connect(str(Path(temporary) / 'manifest.sqlite'))
        try:
            validate_manifest(root, summary, version, database)
            # shard 검사에서 전체 행을 반복 검색하지 않도록 참조 열을 색인화한다.
            database.execute('CREATE INDEX rows_shard ON rows(shard)')
            validate_shards(root, summary, database, progress)
            counts, audit, groups, class_groups = build_groups(database)
            assessment = assess_split_feasibility(audit, groups, class_groups, config['max_group_ratio'])
            require(assessment['status'] == 'preconditions_passed', 'SPLIT_PRECONDITIONS_NOT_PASSED')
            best, candidates = choose_assignment(counts, config, progress)
            score, candidate, assignment, matrix = best
            database.execute('CREATE TABLE assignments(gid INTEGER PRIMARY KEY, split TEXT)')
            database.executemany('INSERT INTO assignments VALUES (?,?)',
                                 ((gid, SPLITS[int(value)] if value >= 0 else None)
                                  for gid, value in enumerate(assignment)))
            verify_assignments(database)
            # manifest와 설정으로 프로토콜을 식별하며 소스 파일에 해시를 삽입하지 않는다.
            digest = hashlib.sha256()
            with (root / 'manifest.jsonl').open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            manifest_hash = digest.hexdigest()
            digest.update(json.dumps(config, sort_keys=True).encode('utf-8'))
            protocol_id = 'wm_split_' + digest.hexdigest()
            output.mkdir(parents=True, exist_ok=False)
            report = {'status': 'split_complete', 'split_complete': True, 'training_ready': False,
                      'dataset_version': version, 'split_protocol_id': protocol_id,
                      'canonical_root': str(root), 'source_manifest_sha256': manifest_hash,
                      'config': config, 'selected_candidate': candidate, 'objective': score,
                      'candidates': candidates, 'arrays_revalidated': True,
                      'source_modified': False, 'overlap_check_passed': True,
                      'class_mapping': {str(i): name for i, name in enumerate(CLASS_NAMES)},
                      'excluded_conflict_groups': database.execute('SELECT COUNT(*) FROM conflicts').fetchone()[0],
                      'split_counts': {name: int(matrix[index].sum()) for index, name in enumerate(SPLITS)},
                      'class_counts': {name: matrix[index].tolist() for index, name in enumerate(SPLITS)}}
            report['warnings'] = [{'split': name, 'class': CLASS_NAMES[label], 'count': int(matrix[index, label])}
                                  for index, name in enumerate(SPLITS) if name != 'train'
                                  for label in range(9) if matrix[index, label] < config['rare_class_warning_count']]
            query = ('SELECT r.record,g.gid,s.split,c.hash FROM rows r '
                     'LEFT JOIN lot_groups g ON r.lot=g.lot LEFT JOIN assignments s ON g.gid=s.gid '
                     'LEFT JOIN conflicts c ON r.hash=c.hash ORDER BY r.position')
            written, excluded = 0, 0
            observed = np.zeros_like(matrix)
            with (output / 'split_manifest.jsonl').open('x', encoding='utf-8') as stream:
                for record, gid, group_split, conflict in database.execute(query):
                    row = decode_json(record)
                    row['split_protocol_id'] = protocol_id
                    row['split_group_id'] = f'group_{gid:08d}' if gid is not None and conflict is None else None
                    row['group_split'] = group_split if conflict is None else None
                    row['exclusion_reasons'] = ['MAP_LABEL_CONFLICT'] if conflict is not None else []
                    if conflict is not None:
                        row['split'] = 'quarantined'
                        excluded += 1
                    elif row['quality_status'] != 'accepted':
                        row['split'] = 'quarantined'
                    elif row['label_status'] != 'labeled':
                        row['split'] = 'unlabeled'
                    else:
                        require(group_split in SPLITS, 'MISSING_LABELED_ASSIGNMENT')
                        row['split'] = group_split
                        observed[SPLITS.index(group_split), row['class_index']] += 1
                    stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n')
                    written += 1
            require(written == summary['row_count'], 'SPLIT_ROW_COUNT_MISMATCH')
            require(np.array_equal(observed, matrix), 'SPLIT_CLASS_COUNT_MISMATCH')
            report['row_count'], report['excluded_rows'] = written, excluded
            (output / 'split.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            write_split_summary(report, output)
            (output / 'SUCCESS.json').write_text(json.dumps({'status': 'split_complete',
                'split_protocol_id': protocol_id, 'training_ready': False}), encoding='utf-8')
        finally:
            database.close()
    return {'status': 'split_complete', 'training_ready': False}


def write_split_summary(report, output):
    # 상세 수치는 로컬 보고서에만 기록하고 에이전트 반환값에는 포함하지 않는다.
    with (output / 'split_summary.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'class_index', 'pattern_label', 'samples', 'split_samples', 'split_ratio'])
        total = sum(report['split_counts'].values())
        for name in SPLITS:
            for index, label in enumerate(CLASS_NAMES):
                writer.writerow([name, index, label, report['class_counts'][name][index],
                                 report['split_counts'][name], report['split_counts'][name] / total])
    lines = ['# WM-811K 그룹 분할 결과', '',
             '원본은 수정하지 않았습니다. 충돌 맵 전체 행을 파생 manifest에서만 격리했습니다.', '',
             'NPZ 참조는 원본 canonical 루트 기준입니다. 묶음 이동 시 split.json의 루트 연결을 함께 관리하세요.', '',
             '| 클래스 | Train | Validation | Test |', '| --- | ---: | ---: | ---: |']
    for index, name in enumerate(CLASS_NAMES):
        values = [report['class_counts'][split][index] for split in SPLITS]
        lines.append(f'| {name} | {values[0]} | {values[1]} | {values[2]} |')
    total = sum(report['split_counts'].values())
    lines += ['', '## 실제 표본 비율', '']
    for name in SPLITS:
        lines.append(f'- {name}: {report["split_counts"][name]}건 / {report["split_counts"][name] / total:.2%}')
    lines += ['', f'충돌 제외 행: {report["excluded_rows"]}건',
              f'Validation/Test 소표본 경고: {len(report["warnings"])}개 클래스·구간 조합', '',
              '미라벨은 학습 대상이 아닙니다. group_split=test인 미라벨을 학습에 사용하지 마세요.',
              '분할 완료는 입력 캐시·Dataset·학습 준비 완료가 아닙니다. Test는 모델 선택에 사용하지 마세요.',
              '후보 선택은 표본·클래스 비율 편차만 사용했습니다. 상세 결과를 에이전트의 외부 LLM에 전송하지 마세요.']
    (output / 'split_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='WM canonical 충돌 제외·연결 그룹 분할')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default='configs/wm811k/split.json')
    parser.add_argument('--approve-conflict-exclusion', action='store_true')
    args = parser.parse_args()
    try:
        with validation_stage('group_split'):
            result = run_split(args.input_dir, args.output_dir, args.config,
                               approved=args.approve_conflict_exclusion,
                               progress=lambda done, total: print(f'검증·후보 진행: {done}/{total}', flush=True))
    except ValidationError as error:
        print('분할 실패 코드: ' + str(error))
        print('완료 표식이 없는 폴더는 학습에 사용하지 마세요. 자동 재시도·덮어쓰기는 하지 않습니다.')
        return 1
    print(result['status'])
    print('로컬 split_summary.md를 확인하세요. 입력 캐시·학습 준비는 아직 미완료입니다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

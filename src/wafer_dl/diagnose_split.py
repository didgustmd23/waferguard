"""manifest 기반 충돌 제외 영향 진단. 배열·라벨·분할을 수정하지 않는다."""

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import tempfile

from .validate_canonical import (
    CLASS_NAMES, ValidationError, assess_split_feasibility, audit_duplicates,
    require, validate_manifest, validate_metadata, validation_stage,
)


# ==========================================
# 시나리오별 클래스 표본·Lot·연결 그룹 집계
# - accepted이면서 명시적 라벨이 있는 행만 지도학습 표본으로 집계
# - 독립 그룹은 Lot과 동일 맵의 연결 관계 기준이며 단순 Lot 수와 다름
# ==========================================
def scenario_summary(database, max_group_ratio):
    audit, groups, class_groups = audit_duplicates(database)

    counts = {index: (samples, lots) for index, samples, lots in database.execute(
        'SELECT label, COUNT(*), COUNT(DISTINCT lot) FROM rows '
        'WHERE accepted=1 AND label IS NOT NULL GROUP BY label')}
    
    classes = []
    for index, name in enumerate(CLASS_NAMES):

        samples, lots = counts.get(index, (0, 0))

        classes.append({'class_index': index, 'pattern_label': name,
                        'labeled_samples': samples, 'lot_count': lots,
                        'independent_group_count': len(class_groups[index]),
                        'minimum_group_condition_met': len(class_groups[index]) >= 3})
        
    return {'manifest_rows': database.execute('SELECT COUNT(*) FROM rows').fetchone()[0],
            'classes': classes, 'duplicate_audit': audit,
            'split_assessment': assess_split_feasibility(audit, groups, class_groups, max_group_ratio)}


# ==========================================
# 충돌 맵 전체를 제외하는 가상 시나리오 비교
# - 같은 맵에 둘 이상의 명시적 라벨이 있는 hash를 후보로 선정
# - 해당 hash의 미라벨·격리 행도 시나리오에서는 함께 제외
# - 제외 후 Lot·동일 맵 연결 그룹을 다시 계산
# - 실제 manifest·NPZ·라벨은 수정하지 않고 임시 DB에서만 비교
# ==========================================
def compare_conflict_exclusion(database, max_group_ratio=0.2):

    require(0 < max_group_ratio <= 1, 'INVALID_GROUP_RATIO')

    database.execute('CREATE TEMP TABLE conflict_hashes AS SELECT hash FROM rows '
                     'WHERE hash IS NOT NULL GROUP BY hash HAVING COUNT(DISTINCT label) > 1')
    
    database.execute('CREATE UNIQUE INDEX conflict_hash_index ON conflict_hashes(hash)')

    before = scenario_summary(database, max_group_ratio)

    excluded = database.execute('SELECT COUNT(*) FROM rows '
                                'WHERE hash IN (SELECT hash FROM conflict_hashes)').fetchone()[0]
    
    conflict_count = database.execute('SELECT COUNT(*) FROM conflict_hashes').fetchone()[0]

    # 기존 임시 테이블은 보존하고 비교 실패 시에도 원래 이름으로 복원한다.
    database.execute('ALTER TABLE rows RENAME TO baseline_rows')

    try:
        database.execute('CREATE TABLE rows AS SELECT * FROM baseline_rows '
                         'WHERE hash IS NULL OR hash NOT IN (SELECT hash FROM conflict_hashes)')
        
        after = scenario_summary(database, max_group_ratio)
    finally:
        database.execute('DROP TABLE IF EXISTS rows')
        database.execute('ALTER TABLE baseline_rows RENAME TO rows')
        database.execute('DROP TABLE conflict_hashes')

    comparisons = []

    for original, remaining in zip(before['classes'], after['classes']):
        comparisons.append({
            'class_index': original['class_index'], 'pattern_label': original['pattern_label'],
            'samples_before': original['labeled_samples'], 'samples_after': remaining['labeled_samples'],
            'excluded_labeled_samples': original['labeled_samples'] - remaining['labeled_samples'],
            'lots_before': original['lot_count'], 'lots_after': remaining['lot_count'],
            'groups_before': original['independent_group_count'],
            'groups_after': remaining['independent_group_count'],
            'minimum_groups_met_before': original['minimum_group_condition_met'],
            'minimum_groups_met_after': remaining['minimum_group_condition_met'],
        })

    return {'status': 'diagnostic_complete', 'diagnosis_version': 'conflict_exclusion_v1',
            'data_modified': False, 'split_complete': False, 'training_ready': False,
            'arrays_revalidated': False, 'source_equivalence_verified': False,
            'candidate_conflict_hash_groups': conflict_count,
            'candidate_excluded_manifest_rows': excluded,
            'baseline': before, 'hypothetical_exclusion': after, 'class_comparison': comparisons,
            'policy': '명시적 라벨 충돌 맵 해시의 모든 행을 가상 제외. 실제 제외는 미승인.'}


# ==========================================
# 로컬 진단 보고서 저장
# - 클래스별 수치는 로컬에만 저장하고 콘솔에는 출력하지 않음
# - ID·Lot 이름·맵 해시·원본 값은 보고서에도 출력하지 않음
# - 새 폴더에만 저장하고 기존 결과는 덮어쓰지 않음
# ==========================================
def write_diagnostic_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    with (output / 'diagnosis.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)

    rows = report['class_comparison']

    with (output / 'class_comparison.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output / 'diagnosis.md').open('x', encoding='utf-8') as stream:
        stream.write('# 충돌 그룹 제외 전후 진단\n\n')
        stream.write('manifest 기반 가상 비교입니다. NPZ 내용은 재검증하지 않았으며 실제 데이터·라벨·분할은 변경하지 않았습니다.\n\n')
        stream.write('정합성 검사를 통과한 동일 묶음을 대상으로 사용하세요. 원본 동등성·안전성 인증이나 제외 정책 승인이 아닙니다.\n\n')
        stream.write(f"충돌 맵 그룹: {report['candidate_conflict_hash_groups']}개 / 가상 제외 행: {report['candidate_excluded_manifest_rows']}개\n\n")
        stream.write('| 클래스 | 표본 전 | 표본 후 | 제외 표본 | Lot 전 | Lot 후 | 그룹 전 | 그룹 후 | 후 조건 |\n')
        stream.write('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n')

        for row in rows:
            state = '충족' if row['minimum_groups_met_after'] else '부족'
            stream.write(f"| {row['pattern_label']} | {row['samples_before']} | {row['samples_after']} | "
                         f"{row['excluded_labeled_samples']} | {row['lots_before']} | {row['lots_after']} | "
                         f"{row['groups_before']} | {row['groups_after']} | {state} |\n")
            
        for key, title in (('baseline', '현재'), ('hypothetical_exclusion', '가상 제외 후')):
            scenario = report[key]
            assessment = scenario['split_assessment']
            stream.write(f"\n## {title}\n\n분할 사전 진단: `{assessment['status']}`\n\n")
            stream.write(f"최대 연결 그룹 비율: {assessment['largest_group_ratio']:.2%}, "
                         f"검토 상한: {assessment['max_group_ratio']:.2%}\n\n")
            stream.write(f"맵 라벨 충돌 그룹: {scenario['duplicate_audit']['map_label_conflicts']} / "
                         f"웨이퍼 ID 충돌 그룹: {scenario['duplicate_audit']['wafer_id_conflicts']}\n")
            
        stream.write('\n## 해석\n\n충돌 해시 제외로 연결 관계가 끊기면 독립 그룹 수는 늘 수도 있습니다. '
                     '반대로 클래스 표본·Lot 손실로 그룹 수가 줄거나 클래스가 사라질 수도 있습니다.\n\n')
        stream.write('필요조건 통과는 실제 분할 가능성을 보장하지 않습니다. 부족한 클래스나 큰 그룹이 남으면 '
                     '제외 정책·평가 방식 검토 후 별도 프로토콜로 결정하며, 자동 재라벨링·행 단위 무작위 분할을 하지 않습니다.\n\n')
        stream.write('상세 집계는 로컬에서만 확인하고 LLM으로 보내지 마세요.\n')


def main():
    parser = argparse.ArgumentParser(description='충돌 맵 제외 영향의 읽기 전용 가상 진단')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-group-ratio', type=float, default=0.2)
    args = parser.parse_args()

    root, output = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()

    if output.exists() or output.is_relative_to(root):
        parser.error('입력 바깥의 존재하지 않는 새 출력 폴더를 지정하세요.')
    try:
        require(0 < args.max_group_ratio <= 1, 'INVALID_GROUP_RATIO')

        with validation_stage('input_directory'):
            root = root.resolve(strict=True)
            require(root.is_dir(), 'INPUT_IS_NOT_DIRECTORY')

        with validation_stage('metadata'):
            summary, version = validate_metadata(root)

        print('manifest 기반 가상 진단을 시작합니다. NPZ는 다시 읽지 않습니다.', flush=True)

        with validation_stage('diagnostic_workspace'):
            with tempfile.TemporaryDirectory(prefix='wm_split_diagnosis_') as temporary:
                database = sqlite3.connect(str(Path(temporary) / 'manifest.sqlite'))
                try:
                    with validation_stage('manifest'):
                        validate_manifest(root, summary, version, database)
                    with validation_stage('conflict_exclusion'):
                        report = compare_conflict_exclusion(database, args.max_group_ratio)
                finally:
                    database.close()

        write_diagnostic_report(report, output)

    except ValidationError as error:
        print('진단 실패 단계: ' + getattr(error, 'stage', 'configuration'))
        print('실패 코드: ' + str(error))
        return 1
    
    print('diagnostic_complete')
    print('가상 제외 후 분할 사전 진단: ' + report['hypothetical_exclusion']['split_assessment']['status'])
    print('데이터는 변경하지 않았습니다. diagnosis.md를 로컬에서 확인하세요.')
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

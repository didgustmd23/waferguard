"""노트북·CLI·향후 에이전트가 공유하는 로컬 EDA 코어."""

import argparse
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from .validate_canonical import (
    CLASS_NAMES, ValidationError, decode_json, read_numeric_member, require,
    validate_metadata, validation_stage,
)


# ==========================================
# canonical metadata만 표로 적재
# - 원본 pickle·전체 맵 배열은 읽지 않음
# - 수정된 라벨 규칙의 묶음만 사용하여 이전 결과 혼동 방지
# - 행 위치·버전·관리 집계를 대조하지만 NPZ 전체 검증을 대체하지 않음
# ==========================================
def load_eda_metadata(input_dir):
    root = Path(input_dir).resolve(strict=True)
    summary, version = validate_metadata(root)

    require(summary.get('label_normalization_version') == 'wm_label_v2', 'OUTDATED_LABEL_NORMALIZATION')

    manifest = root / 'manifest.jsonl'

    require(manifest.is_file() and manifest.resolve().is_relative_to(root), 'UNSAFE_OR_MISSING_MANIFEST')

    fields = ('source_row_position', 'pattern_label', 'class_index', 'label_status',
              'quality_status', 'lot_id', 'map_height', 'map_width',
              'valid_die_count', 'failed_die_count', 'failed_die_ratio', 'map_hash')

    def records():
        with manifest.open('rb') as stream:
            position = 0

            while True:
                line = stream.readline(65_537)

                if not line:
                    break

                require(len(line) <= 65_536, 'OVERSIZED_MANIFEST_ROW')
                row = decode_json(line)

                require(row.get('source_row_position') == position and
                        row.get('sample_id') == f'{version}:{position}', 'INVALID_ROW_POSITION_OR_VERSION')
                require(position < summary['row_count'], 'ROW_COUNT_MISMATCH')

                yield tuple(row.get(field) for field in fields)

                position += 1

    frame = pd.DataFrame.from_records(records(), columns=fields)
    require(len(frame) == summary['row_count'], 'ROW_COUNT_MISMATCH')

    # 반복 문자열을 category로 저장해 전체 metadata의 추가 메모리를 줄인다.
    for column in ('pattern_label', 'label_status', 'quality_status', 'lot_id', 'map_hash'):
        frame[column] = frame[column].astype('category')

    for column in ('map_height', 'map_width', 'valid_die_count', 'failed_die_count', 'failed_die_ratio'):
        frame[column] = pd.to_numeric(frame[column], errors='raise')

    quality_counts = frame['quality_status'].value_counts().to_dict()
    label_counts = frame['label_status'].value_counts().to_dict()

    require(all(summary['counts'].get(key, 0) == count for key, count in quality_counts.items()) and
            all(summary['counts'].get(key, 0) == count for key, count in label_counts.items()), 'SUMMARY_COUNT_MISMATCH')
    
    return frame


# ==========================================
# 로컬 EDA 표 생성
# - 지도학습 대상·미라벨·격리를 구분하여 집계
# - Lot 이름·맵 해시는 집계 키로만 쓰고 결과 표에는 포함하지 않음
# - 같은 맵 내용이 반드시 같은 실물 웨이퍼 복제라는 뜻은 아님
# ==========================================
def summarize_metadata(frame):
    labeled = frame.loc[frame['label_status'].eq('labeled')]
    accepted = labeled.loc[labeled['quality_status'].eq('accepted')]
    total = labeled.groupby('pattern_label', observed=True).size().reindex(CLASS_NAMES, fill_value=0)
    usable = accepted.groupby('pattern_label', observed=True).size().reindex(CLASS_NAMES, fill_value=0)

    class_table = pd.DataFrame({'labeled_samples': total, 'accepted_labeled_samples': usable})
    class_table['accepted_lot_count'] = accepted.groupby('pattern_label', observed=True)['lot_id'].nunique().reindex(CLASS_NAMES, fill_value=0)
    class_table.index.name = 'pattern_label'

    status_table = pd.DataFrame({
        'rows': frame.groupby(['label_status', 'quality_status'], observed=True).size()})
    
    maps = frame.loc[frame['map_hash'].notna()]
    shape_table = maps.groupby(['map_height', 'map_width']).size().rename('rows').reset_index()
    shape_table = shape_table.sort_values(['rows', 'map_height', 'map_width'], ascending=[False, True, True])

    lot_sizes = frame.loc[frame['lot_id'].notna()].groupby('lot_id', observed=True).size()
    duplicate_sizes = maps.groupby('map_hash', observed=True).size()
    label_per_map = labeled.loc[labeled['map_hash'].notna()].groupby('map_hash', observed=True)['class_index'].nunique()

    statistics = {
        'row_count': int(len(frame)), 'rows_with_valid_map': int(len(maps)),
        'lot_count': int(len(lot_sizes)),
        'duplicate_map_groups': int((duplicate_sizes > 1).sum()),
        'rows_in_duplicate_groups': int(duplicate_sizes.loc[duplicate_sizes > 1].sum()),
        'map_label_conflict_groups': int((label_per_map > 1).sum()),
        'split_complete': False, 'training_ready': False,
    }

    # Lot별 분포는 크기만 반환하고 식별자를 결과에 남기지 않는다.
    return {'classes': class_table, 'statuses': status_table, 'shapes': shape_table,
            'lot_sizes': pd.Series(lot_sizes.to_numpy(), name='rows_per_lot'),
            'statistics': statistics}


# ==========================================
# 선택한 소수 예시의 숫자 맵만 읽기
# - manifest의 참조를 고정 루트 내부로 제한
# - ZIP을 풀지 않고 숫자 헤더 검사·allow_pickle=False로 복원
# - 원본 크기를 유지하고 resize·라벨 변경·보충을 하지 않음
# ==========================================
def load_example_maps(input_dir, positions):
    root = Path(input_dir).resolve(strict=True)
    wanted = set(positions)

    require(len(wanted) <= 27 and all(type(value) is int and value >= 0 for value in wanted), 'INVALID_EXAMPLE_SELECTION')

    references = []
    manifest = root / 'manifest.jsonl'

    require(manifest.resolve().is_relative_to(root), 'UNSAFE_MANIFEST')

    with manifest.open('rb') as stream:
        while wanted:
            line = stream.readline(65_537)

            if not line:
                break

            require(len(line) <= 65_536, 'OVERSIZED_MANIFEST_ROW')

            row = decode_json(line)
            position = row.get('source_row_position')

            if position in wanted:
                ref = row.get('map_ref')

                require(type(ref) is dict and type(ref.get('shard')) is str and
                        re.fullmatch(r'maps/shard_[0-9]{6}\.npz', ref['shard']) and
                        ref.get('key') == f'map_{position:09d}', 'INVALID_MAP_REFERENCE')
                
                require(row.get('pattern_label') in CLASS_NAMES and row.get('quality_status') == 'accepted', 'INVALID_EXAMPLE_STATE')

                references.append((row['pattern_label'], ref))
                wanted.remove(position)

    require(not wanted, 'MISSING_EXAMPLE')
    examples = []

    for label, ref in references:
        path = root / ref['shard']

        require(path.is_file() and path.resolve().is_relative_to(root) and
                path.stat().st_size <= 1_100_000_000, 'UNSAFE_EXAMPLE_SHARD')
        
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            require(len(members) <= 1000 and len({member.filename for member in members}) == len(members), 'INVALID_SHARD_MEMBER_COUNT')
            array = read_numeric_member(archive, archive.getinfo(ref['key'] + '.npy'))
            require(np.isin(array, (0, 1, 2)).all() and np.any(array > 0), 'INVALID_MAP_VALUES')

        examples.append((label, array))

    return examples


def select_example_positions(frame):
    # 클래스별 첫 유효 1건을 편의상 표시하며 대표성을 보장하지 않는다.
    selected = frame.loc[frame['quality_status'].eq('accepted') & frame['label_status'].eq('labeled')]
    return [int(group['source_row_position'].iloc[0]) for _, group in
            selected.groupby('pattern_label', observed=True, sort=False)]


# ==========================================
# 공통 EDA 그래프 생성
# - 노트북과 CLI에서 같은 그래프 함수를 사용
# - 상태·클래스·형태·Lot 분포를 표시하며 ID는 그리지 않음
# ==========================================
def plot_distributions(frame, tables):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    tables['classes']['accepted_labeled_samples'].plot.bar(ax=axes[0, 0], logy=True)
    axes[0, 0].set_title('Accepted labeled samples (log scale)')
    tables['statuses']['rows'].plot.bar(ax=axes[0, 1], logy=True)
    axes[0, 1].set_title('Label / quality status (log scale)')
    maps = frame.loc[frame['map_hash'].notna()]
    axes[0, 2].hexbin(maps['map_width'], maps['map_height'], gridsize=35, mincnt=1, bins='log')
    axes[0, 2].set(xlabel='Width', ylabel='Height', title='Original map dimensions')
    axes[1, 0].hist(maps['map_width'] / maps['map_height'], bins=50)
    axes[1, 0].set(xlabel='Width / height', title='Aspect ratio')
    axes[1, 1].hist(maps['failed_die_ratio'], bins=50, log=True)
    axes[1, 1].set(xlabel='Failed / valid dies', title='Failed die ratio (all valid maps)')
    axes[1, 2].hist(tables['lot_sizes'], bins=50, log=True)
    axes[1, 2].set(xlabel='Rows per lot', title='Lot size distribution')
    figure.tight_layout()

    return figure


def plot_examples(examples):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    figure, axes = plt.subplots(3, 3, figsize=(12, 12))
    lookup = dict(examples)
    cmap = ListedColormap(['#eeeeee', '#348a50', '#d94a45'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for index, (name, axis) in enumerate(zip(CLASS_NAMES, axes.flat)):
        axis.set_title(name + (' (not available)' if name not in lookup else ''))
        if name in lookup:
            axis.imshow(lookup[name], cmap=cmap, norm=norm, interpolation='nearest')
        axis.axis('off')

    figure.suptitle('First accepted example / class — original size; not representative')
    figure.tight_layout()

    return figure


# ==========================================
# 수치 metadata의 상관분석
# - accepted 유효 맵을 대상으로 하며 미라벨도 전체 집계에 포함
# - 클래스 번호·Lot·해시·행 번호는 수치 변수에서 제외
# - 모든 수치가 유한한 행만 사용하여 Pearson과 순위 기반 Spearman 비교
# - 상수 열·표본 부족은 NaN 유지: 관계가 없다는 뜻의 0으로 바꾸지 않음
# ==========================================
def correlation_data(frame):
    selected = frame.loc[frame['map_hash'].notna() & frame['quality_status'].eq('accepted')]
    numeric = selected[['map_height', 'map_width', 'valid_die_count',
                        'failed_die_count', 'failed_die_ratio']].astype(float).copy()
    
    # 높이가 0인 경우 나눗셈하지 않고 결측으로 처리한다.
    numeric['aspect_ratio'] = numeric['map_width'] / numeric['map_height'].where(numeric['map_height'] > 0)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    return numeric, selected.loc[numeric.index, 'pattern_label']


def summarize_correlations(frame):
    numeric, labels = correlation_data(frame)

    def matrices(values):
        # 완전 관측 행에 평균 순위를 부여한 뒤 Pearson을 계산하면 Spearman이다.
        return {'pearson': values.corr(method='pearson', min_periods=3),
                'spearman': values.rank(method='average').corr(method='pearson', min_periods=3)}

    result = matrices(numeric)
    result['support'] = pd.DataFrame({'rows': [len(numeric)]}, index=['all_accepted'])
    rows = []

    for name in CLASS_NAMES:
        values = numeric.loc[labels.eq(name)]

        # 같은 변수 쌍을 클래스별로 비교한다. 라벨 자체의 상관계수는 계산하지 않는다.
        for method, matrix in matrices(values).items():
            for first_index, first in enumerate(numeric.columns):
                for second in numeric.columns[first_index + 1:]:
                    rows.append({'pattern_label': name, 'method': method,
                                 'first': first, 'second': second, 'rows': len(values),
                                 'correlation': matrix.loc[first, second]})
                    
    result['by_class'] = pd.DataFrame(rows)

    return result


def plot_correlations(tables):
    # 색 범위를 고정하여 두 방법을 비교하고 계산 불가 값은 빈 칸으로 구분한다.
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(17, 7))

    for method, axis in zip(('pearson', 'spearman'), axes):
        matrix = tables[method]
        artist = axis.imshow(np.ma.masked_invalid(matrix.to_numpy()), vmin=-1, vmax=1, cmap='coolwarm')
        axis.set_xticks(range(len(matrix)), matrix.columns, rotation=90)
        axis.set_yticks(range(len(matrix)), matrix.index)
        axis.set_title(f'{method.title()} / complete rows: {tables["support"].iloc[0, 0]}')

        for row in range(len(matrix)):
            for column in range(len(matrix)):
                value = matrix.iloc[row, column]
                axis.text(column, row, f'{value:.2f}' if np.isfinite(value) else 'N/A', ha='center', va='center')

        figure.colorbar(artist, ax=axis, fraction=0.046)

    figure.tight_layout()

    return figure


def plot_correlation_scatter(frame):
    # 산점도만 고정 시드로 최대 5천 행을 추출한다. 상관계수는 전체 적격 행으로 계산한다.
    import matplotlib.pyplot as plt

    numeric, _ = correlation_data(frame)
    sample = numeric.sample(n=min(len(numeric), 5000), random_state=42)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    for axis, (first, second) in zip(axes, (
            ('valid_die_count', 'failed_die_count'),
            ('valid_die_count', 'failed_die_ratio'),
            ('aspect_ratio', 'failed_die_ratio'))):
        axis.scatter(sample[first], sample[second], s=5, alpha=0.2)
        axis.set(xlabel=first, ylabel=second)

    figure.suptitle('Display sample: at most 5,000 accepted maps (not class-balanced)')
    figure.tight_layout()

    return figure


def plot_die_statistics(frame):
    # 다이 개수와 클래스별 불량 비율을 별도로 확인한다.
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    maps = frame.loc[frame['map_hash'].notna()]
    axes[0].hist(maps['valid_die_count'], bins=50, log=True)
    axes[0].set(xlabel='Valid dies', title='Valid die counts')
    axes[1].hist(maps['failed_die_count'], bins=50, log=True)
    axes[1].set(xlabel='Failed dies', title='Failed die counts')
    labeled = maps.loc[maps['label_status'].eq('labeled') & maps['quality_status'].eq('accepted')]
    ratios = [labeled.loc[labeled['pattern_label'].eq(name), 'failed_die_ratio'].to_numpy() for name in CLASS_NAMES]
    axes[2].boxplot(ratios, showfliers=False)
    axes[2].set_xticks(range(1, 10), CLASS_NAMES, rotation=90)
    axes[2].set(ylabel='Failed / valid dies', title='Ratio by class (outliers hidden)')
    figure.tight_layout()

    return figure


# ==========================================
# 에이전트 없이도 실행 가능한 EDA 도구 경계
# - 그림·집계는 승인된 로컬 출력에만 저장
# - 반환값은 일반 실행 상태만 포함하며 LLM용 DTO가 아님
# - 외부 API·LangSmith·LLM 호출은 구현하지 않음
# ==========================================
def run_eda(input_dir, output_dir, include_examples=True):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    require(not output.exists() and not output.is_relative_to(root), 'INVALID_REPORT_DIRECTORY')

    frame = load_eda_metadata(root)
    tables = summarize_metadata(frame)
    correlations = summarize_correlations(frame)
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib.pyplot as plt

    figure = plot_distributions(frame, tables)

    try:
        figure.savefig(output / 'distributions.png', dpi=150)
    finally:
        plt.close(figure)

    figure = plot_die_statistics(frame)

    try:
        figure.savefig(output / 'die_statistics.png', dpi=150)
    finally:
        plt.close(figure)

    for name, figure in (('correlations', plot_correlations(correlations)),
                         ('correlation_scatter', plot_correlation_scatter(frame))):
        try:
            figure.savefig(output / f'{name}.png', dpi=150)
        finally:
            plt.close(figure)
            
    for name, table in correlations.items():
        table.to_csv(output / f'correlation_{name}.csv', encoding='utf-8-sig', index=name != 'by_class')
    if include_examples:
        figure = plot_examples(load_example_maps(root, select_example_positions(frame)))
        try:
            figure.savefig(output / 'examples.png', dpi=150)
        finally:
            plt.close(figure)
    for name in ('classes', 'statuses', 'shapes'):
        tables[name].to_csv(output / f'{name}.csv', encoding='utf-8-sig')
    with (output / 'eda.json').open('x', encoding='utf-8') as stream:
        json.dump(tables['statistics'], stream, indent=2, allow_nan=False)
    with (output / 'eda.md').open('x', encoding='utf-8') as stream:
        stream.write('# WM-811K 분할 전 구조·품질 EDA\n\n')
        stream.write('![분포](distributions.png)\n\n')
        stream.write('![다이 통계](die_statistics.png)\n\n')
        stream.write('![수치 상관계수](correlations.png)\n\n![산점도](correlation_scatter.png)\n\n')
        stream.write('상관분석은 accepted 유효 맵의 완전 관측 행을 사용하며 미라벨을 포함합니다. 클래스별 비교는 correlation_by_class.csv에서 확인하세요.\n\n')
        stream.write('상수 열·3건 미만은 N/A입니다. 산점도만 최대 5천 행을 고정 추출합니다. 불량 비율=불량 다이 수/유효 다이 수이므로 계산상 결합을 주의하세요.\n\n')
        stream.write('전체 상관은 클래스 구성·Lot·중복 맵의 영향을 받습니다. 인과관계·유의성 검정·자동 특징 선택 결과가 아니며 모델용 선택은 분할 후 학습 데이터에서 다시 검토하세요.\n\n')
        if include_examples:
            stream.write('![클래스별 예시](examples.png)\n\n')
        stream.write('클래스별 첫 유효 예시이며 대표 표본을 뜻하지 않습니다. 상세 집계는 로컬 CSV·JSON에서 확인하세요.\n')
        stream.write('원본 변경·충돌 제외·분할·resize·증강·모델 평가를 수행하지 않았습니다.\n')
        stream.write('전체 데이터 EDA는 구조·품질 확인용이며 모델 입력 설정 선택은 분할 후 Train/Validation에서 수행하세요.\n')
        stream.write('실제 데이터·집계·이미지·경로는 LLM·외부 추적에 전달하지 마세요.\n')
    with (output / 'SUCCESS.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'eda_complete', 'training_ready': False}, stream)
    return {'status': 'eda_complete'}


def main():
    parser = argparse.ArgumentParser(description='로컬 전용 WM canonical EDA')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--metadata-only', action='store_true', help='예시 NPZ를 읽지 않고 metadata 분포만 생성')
    args = parser.parse_args()
    try:
        with validation_stage('eda'):
            result = run_eda(args.input_dir, args.output_dir, include_examples=not args.metadata_only)
    except ValidationError as error:
        print('EDA 실패 단계: ' + getattr(error, 'stage', 'eda'))
        print('실패 코드: ' + str(error))
        return 1
    print(result['status'])
    print('분포·예시는 로컬에서만 확인하세요. 데이터와 분할은 변경하지 않았습니다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

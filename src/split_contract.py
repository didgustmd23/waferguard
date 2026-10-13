# ==========================================
# 학습·검증 split 계약 및 누수 방어
# - 예약 metadata는 모델 특징에서 제외하고 원본 행 추적에만 사용
# - 새 split의 역할·생성 계약을 검사하며 기존 CSV도 중복 행을 검사
# - 시간 검증은 Train의 최종 시각보다 뒤에 있는 구간만 허용
# ==========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

SOURCE_ROW_ID = "__source_row_id"
SPLIT_ROLE = "__split_role"
PROTOCOL_ID = "__split_protocol_id"
SPLIT_METADATA = (SOURCE_ROW_ID, SPLIT_ROLE, PROTOCOL_ID)


def validate_train_role(frame):
    """학습 입력의 split 역할·행 식별자·생성 계약 metadata를 검증한다."""
    present = set(frame.columns) & set(SPLIT_METADATA)

    if present and present != set(SPLIT_METADATA):
        raise ValueError("split metadata는 원본 ID·역할·생성 계약을 모두 포함해야 합니다.")
    
    # 새 계약의 Validation/Test를 학습 또는 OOF 입력으로 넘기면 즉시 차단한다.
    if SPLIT_ROLE in frame and (frame[SPLIT_ROLE].isna().any()
                               or set(frame[SPLIT_ROLE]) != {"train"}):
        raise ValueError("학습 입력에는 train 역할의 split만 사용할 수 있습니다.")
    
    for column in (SOURCE_ROW_ID, PROTOCOL_ID):
        if column in frame and frame[column].isna().any():
            raise ValueError(f"split metadata에 결측값이 있습니다: {column}")
        
    if SOURCE_ROW_ID in frame and frame[SOURCE_ROW_ID].duplicated().any():
        raise ValueError("학습 split의 원본 행 ID가 중복됩니다.")
    
    if PROTOCOL_ID in frame and frame[PROTOCOL_ID].nunique() != 1:
        raise ValueError("학습 split에는 하나의 생성 계약만 있어야 합니다.")


def validate_split_pair(train, validation, dataset, *, temporal=False):
    """Train/Validation 간 출처, 행·그룹 중복 및 시간 누수를 검사한다.

    split metadata가 있는 새 형식과 metadata가 없는 기존 CSV를 모두
    처리한다. ``temporal=True``이면 Train 종료 시각이 Validation 시작보다
    반드시 앞서는지도 확인한다.
    """
    if train.empty or validation.empty:
        raise ValueError("Train과 Validation split은 비어 있으면 안 됩니다.")
    
    # 같은 생성 계약의 Train과 Validation인지 먼저 확인한다.
    validate_train_role(train)
    if SPLIT_ROLE in validation and set(validation[SPLIT_ROLE]) != {"validation"}:
        raise ValueError("검증 입력에는 validation 역할의 split만 사용할 수 있습니다.")
    
    for column in SPLIT_METADATA:
        if (column in train) != (column in validation):
            raise ValueError("Train과 Validation의 split metadata 구성이 다릅니다.")
        if column in validation and validation[column].isna().any():
            raise ValueError(f"split metadata에 결측값이 있습니다: {column}")
        
    if PROTOCOL_ID in train:
        if train[PROTOCOL_ID].nunique() != 1 or validation[PROTOCOL_ID].nunique() != 1:
            raise ValueError("하나의 split에는 하나의 생성 계약만 있어야 합니다.")
        if train[PROTOCOL_ID].iloc[0] != validation[PROTOCOL_ID].iloc[0]:
            raise ValueError("Train과 Validation의 split 생성 계약이 다릅니다.")
        
    if SOURCE_ROW_ID in train:
        if validation[SOURCE_ROW_ID].duplicated().any():
            raise ValueError("검증 split의 원본 행 ID가 중복됩니다.")
        if set(train[SOURCE_ROW_ID]) & set(validation[SOURCE_ROW_ID]):
            raise ValueError("Train과 Validation에 동일한 원본 행 ID가 포함됩니다.")
    else:
        # 기존 CSV는 인덱스가 초기화되므로 컬럼 순서를 맞춘 행 해시로 중복을 검사한다.
        columns = sorted(set(train.columns) - set(SPLIT_METADATA))
        if set(columns) == set(validation.columns) - set(SPLIT_METADATA):
            train_hash = pd.util.hash_pandas_object(train[columns], index=False)
            valid_hash = pd.util.hash_pandas_object(validation[columns], index=False)
            if set(train_hash) & set(valid_hash):
                raise ValueError("Train과 Validation에 동일한 데이터 행이 포함됩니다.")
            
    if dataset.group_columns:
        groups = list(dataset.group_columns)
        if train[groups].isna().any().any() or validation[groups].isna().any().any():
            raise ValueError("split 그룹 컬럼에 결측값이 있습니다.")
        if set(map(tuple, train[groups].to_numpy())) & set(map(tuple, validation[groups].to_numpy())):
            raise ValueError("Train과 Validation에 동일한 그룹이 포함됩니다.")
        
    if temporal and dataset.timestamp_column is not None:
        # 같은 timestamp를 경계 양쪽에 두지 않으며 파싱 실패를 조용히 제거하지 않는다.
        train_time = pd.to_datetime(train[dataset.timestamp_column], format=dataset.timestamp_format, errors="coerce")
        valid_time = pd.to_datetime(validation[dataset.timestamp_column], format=dataset.timestamp_format, errors="coerce")
        if train_time.isna().any() or valid_time.isna().any():
            raise ValueError("시간 검증 입력에 timestamp 파싱 실패 또는 결측값이 있습니다.")
        if train_time.max() >= valid_time.min():
            raise ValueError("Time Train과 Validation의 시간 범위가 겹칩니다.")


# ==========================================
# Train 내부 CV fold 생성
# - 그룹이 없으면 기존 Repeated Stratified K-Fold 조건을 유지
# - 그룹 선언 시 같은 그룹이 학습·검증 fold 양쪽에 포함되지 않도록 분리
# ==========================================
def training_folds(features, labels, frame, config):
    """설정에 따라 반복 계층 fold 또는 그룹 누수 방지 fold를 생성한다.

    그룹 지정이 없으면 RepeatedStratifiedKFold를 사용하고, 그룹이 있으면
    같은 그룹이 한 fold의 양쪽에 나뉘지 않도록 StratifiedGroupKFold를 쓴다.
    반환되는 인덱스는 모든 모델·특징 실험에서 공통으로 재사용한다.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

    cv = config.experiment.cv
    if labels.nunique() < 2 or int(labels.value_counts().min()) < cv.n_splits:
        raise ValueError("정상·Fail label별 샘플 수가 CV fold 수 이상이어야 합니다.")
    
    if not config.dataset.group_columns:
        return RepeatedStratifiedKFold(n_splits=cv.n_splits, n_repeats=cv.n_repeats,
                                      random_state=cv.random_state).split(features, labels)
    
    group_frame = frame[list(config.dataset.group_columns)]

    if group_frame.isna().any().any():
        raise ValueError("CV 그룹 컬럼에 결측값이 있습니다.")
    
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(group_frame))

    for label in labels.unique():
        if len(set(groups[labels.to_numpy() == label])) < cv.n_splits:
            raise ValueError("각 label을 포함한 그룹 수가 CV fold 수보다 작습니다.")

    def grouped_folds():
        # 반복별 시드만 변경하고 모든 모델·선택 실험은 동일한 그룹 fold를 재사용한다.
        for repeat in range(cv.n_repeats):
            splitter = StratifiedGroupKFold(n_splits=cv.n_splits, shuffle=True,
                                           random_state=cv.random_state + repeat)
            for train_indices, valid_indices in splitter.split(features, labels, groups):
                if labels.iloc[train_indices].nunique() < 2 or labels.iloc[valid_indices].nunique() < 2:
                    raise ValueError("그룹 CV fold에 두 label이 모두 없습니다. fold 수 또는 그룹 구성을 검토하세요.")
                yield train_indices, valid_indices

    return grouped_folds()


# ==========================================
# timestamp 단위의 확장형 학습·평가 인덱스 생성
# - 같은 시각과 선언된 그룹·복합 ID가 경계를 넘으면 허용하지 않음
# - 입력 행 순서와 관계없이 시간 범위가 겹치지 않는지 검증
# ==========================================
def temporal_folds(frame, dataset, n_splits):
    """시간 순서를 보존하며 확장형 train/validation 인덱스를 생성한다.

    동일 timestamp의 행은 분리하지 않고, 각 평가 구간이 학습 구간보다
    뒤에 있는지 확인한다. 무작위 분할로 미래 정보가 섞이는 것을 방지한다.
    """
    if dataset.timestamp_column is None:
        raise ValueError("시간 검증에는 Dataset Profile의 timestamp 컬럼이 필요합니다.")
    times = pd.to_datetime(frame[dataset.timestamp_column],
                           format=dataset.timestamp_format, errors="coerce")
    if times.isna().any():
        raise ValueError("시간 검증 입력에 timestamp 파싱 실패 또는 결측값이 있습니다.")
    unique_times = np.sort(times.unique())
    if n_splits < 2 or len(unique_times) <= n_splits:
        raise ValueError("시간 fold 수는 2 이상이며 고유 timestamp 수보다 작아야 합니다.")
    
    folds = []

    for fit_times, eval_times in TimeSeriesSplit(n_splits=n_splits).split(unique_times):
        fit_i = np.flatnonzero(times.isin(unique_times[fit_times]).to_numpy())
        eval_i = np.flatnonzero(times.isin(unique_times[eval_times]).to_numpy())
        train = frame.iloc[fit_i]
        evaluation = frame.iloc[eval_i].copy()

        # 원본 Train의 부분집합을 내부 평가로 사용하므로 복사본의 역할만 변경한다.
        if SPLIT_ROLE in evaluation:
            evaluation[SPLIT_ROLE] = "validation"
            
        validate_split_pair(train, evaluation, dataset, temporal=True)

        if dataset.id_columns:
            ids = list(dataset.id_columns)
            if train[ids].isna().any().any() or evaluation[ids].isna().any().any():
                raise ValueError("시간 검증의 ID 컬럼에 결측값이 있습니다.")
            if set(map(tuple, train[ids].to_numpy())) & set(map(tuple, evaluation[ids].to_numpy())):
                raise ValueError("시간 검증 학습·평가 구간에 동일한 복합 ID가 포함됩니다.")
            
        folds.append((fit_i, eval_i))

    return folds

# ==========================================
# Dataset Profile 기반 범용 데이터 구조 검증
# - 컬럼명·Label 값·feature 선택 규칙을 DatasetSpec에서 받아 하드코딩을 제거
# - split·EDA·모델링 Script가 같은 feature 정의를 재사용하도록 지원
# - 원본 DataFrame은 수정하지 않고 검증된 feature 컬럼명만 반환
# ==========================================

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# `python -m src...` 실행과 `python src/<script>.py` 직접 실행을 모두 지원한다.
try:
    from src.modeling_config import DatasetSpec
    from src.split_contract import SPLIT_METADATA
except ModuleNotFoundError:
    from modeling_config import DatasetSpec
    from split_contract import SPLIT_METADATA


# ==========================================
# DatasetSpec 기준으로 모델 입력 feature 컬럼명 결정
# - prefix: 지정한 접두어를 가진 컬럼만 feature로 선택
# - all_except_metadata: Label·Timestamp·ID·Group·명시적 제외 컬럼을 제외한 모든 컬럼 선택
# - 필수 역할 컬럼 누락·중복 컬럼·feature 부재·수치형 규칙 위반을 조기에 차단
# ==========================================
def resolve_feature_columns(
    column_names: Iterable[str], dataset: DatasetSpec
) -> tuple[str, ...]:
    # Iterable 입력을 한 번만 순회할 수 있도록 불변 Tuple로 변환
    columns = tuple(column_names)

    # DataFrame의 중복 컬럼은 선택·로그·Pipeline에서 모호하므로 허용하지 않음
    if len(columns) != len(set(columns)):
        raise ValueError("데이터셋 column 이름에 중복이 있으면 안 됩니다")

    # Label·Timestamp·ID·Group·명시적 제외 컬럼이 실제 입력에 존재하는지 확인
    required_columns = [dataset.label_column]

    if dataset.timestamp_column is not None:
        required_columns.append(dataset.timestamp_column)

    required_columns.extend(dataset.id_columns)
    required_columns.extend(dataset.group_columns)
    required_columns.extend(dataset.excluded_feature_columns)
    missing_columns = [column for column in required_columns if column not in columns]

    if missing_columns:
        raise ValueError(f"데이터셋에 필수 column이 없습니다: {missing_columns}")

    # 모델 입력에서 제외할 역할 컬럼을 먼저 구성한다.
    non_feature_columns = set(required_columns) | set(SPLIT_METADATA)

    # Profile이 지정한 방식으로 feature 후보를 일관되게 선택
    if dataset.feature_selection_mode == "prefix":
        feature_columns = tuple(
            column
            for column in columns
            if column.startswith(dataset.feature_column_prefix or "")
            and column not in non_feature_columns
        )
    elif dataset.feature_selection_mode == "all_except_metadata":
        feature_columns = tuple(
            column for column in columns if column not in non_feature_columns
        )
    else:
        # load_dataset_spec이 이미 검증하지만 직접 DatasetSpec을 만들었을 때도 방어
        raise ValueError(f"지원하지 않는 feature 선택 방식입니다: {dataset.feature_selection_mode}")

    # 모델 입력 feature가 하나도 없으면 이후 Pipeline 오류 대신 명확한 원인을 제공
    if not feature_columns:
        raise ValueError("Dataset Profile에서 선택된 feature column이 없습니다")

    return feature_columns


# ==========================================
# DatasetSpec과 실제 DataFrame의 metadata·Label 정합성 검증
# - 결측·허용 외 Label을 정상 클래스로 묵시적으로 처리하지 않음
# - 반환된 feature 컬럼만 후속 split·모델링 코드에서 사용
# ==========================================
def validate_dataset_frame(dataframe: pd.DataFrame, dataset: DatasetSpec) -> tuple[str, ...]:
    # 공통 feature 선택 함수로 컬럼 구성과 필수 metadata 존재 여부를 먼저 검증
    feature_columns = resolve_feature_columns(dataframe.columns, dataset)

    # Label 결측은 클래스 비율·평가 지표를 왜곡하므로 즉시 차단
    labels = dataframe[dataset.label_column]
    if labels.isna().any():
        raise ValueError(f"데이터셋 column '{dataset.label_column}'에 결측 label이 있습니다")

    # Dataset Profile에 선언된 정상·불량 Label 외 값이 포함되었는지 확인
    allowed_labels = {dataset.negative_label, dataset.positive_label}
    invalid_labels = labels[~labels.isin(allowed_labels)].unique().tolist()
    
    if invalid_labels:
        raise ValueError(
            f"데이터셋 column '{dataset.label_column}'에 지원하지 않는 label이 있습니다: "
            f"{invalid_labels}"
        )

    # Profile에 명시된 범주형 feature만 문자열·category dtype을 허용한다.
    categorical_columns = set(dataset.categorical_feature_columns)
    unknown_categorical_columns = sorted(categorical_columns - set(feature_columns))
    
    if unknown_categorical_columns:
        raise ValueError(
            "Profile에 선언된 범주형 feature가 선택된 feature에 없습니다: "
            f"{unknown_categorical_columns}"
        )
    
    non_numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
        and not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    
    if non_numeric_columns:
        raise ValueError(
            "범주형으로 선언되지 않은 feature는 숫자형 dtype이어야 합니다: "
            f"{non_numeric_columns}"
        )

    return feature_columns


# ==========================================
# split DataFrame을 검증하고 동일한 feature 순서의 X, y로 분리
# - Profile 규칙으로 label, metadata, feature dtype을 먼저 검증
# - Validation feature의 순서가 달라도 Train feature 순서로 재정렬
# - feature 집합이 다르면 조용히 보정하지 않고 split 생성 오류로 중단
# ==========================================
def split_frame_to_xy(
    dataframe: pd.DataFrame,
    dataset: DatasetSpec,
    *,
    expected_feature_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.Series, tuple[str, ...]]:
    # Dataset Profile과 DataFrame의 label, metadata, feature 타입 일치 여부를 검증한다.
    feature_columns = validate_dataset_frame(dataframe, dataset)

    if expected_feature_columns is not None:
    
        # 순서 차이는 Train 기준으로 맞출 수 있지만, 누락 또는 추가 feature는 허용하지 않는다.
        if set(feature_columns) != set(expected_feature_columns):
            raise ValueError(
                "Train과 Validation의 feature column 구성이 일치하지 않습니다; "
                f"Train={list(expected_feature_columns)}, "
                f"Validation={list(feature_columns)}"
            )
        
        feature_columns = expected_feature_columns

    # 모델에는 검증된 feature만 전달하고 label은 원래 label 값으로 유지한다.
    return (
        dataframe.loc[:, list(feature_columns)],
        dataframe[dataset.label_column],
        feature_columns,
    )

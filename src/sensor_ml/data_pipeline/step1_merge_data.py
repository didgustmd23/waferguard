# ==========================================
# Dataset Profile 기반 원본 데이터 병합
# - Profile에 선언된 source를 label·timestamp·feature가 있는 canonical table로 변환
# - 원본 파일 경로·읽기 옵션·metadata 컬럼 순서는 Dataset Profile에서 읽음
# - 행 수·metadata 구조·Label·feature 구성을 저장 전에 검증
# - 다른 원본 형식은 별도 ingestion adapter를 추가하고, 이후 코어 단계는 재사용
# ==========================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

# 프로젝트 루트에서 모듈 실행 또는 import하여 공통 코어를 불러온다.
try:
    from src.dataset_schema import validate_dataset_frame
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH,
        DatasetSpec,
        IngestionSpec,
        load_modeling_config,
    )
except ModuleNotFoundError:
    from dataset_schema import validate_dataset_frame
    from modeling_config import (
        DEFAULT_CONFIG_PATH,
        DatasetSpec,
        IngestionSpec,
        load_modeling_config,
    )


# ==========================================
# feature·metadata pair 형식의 원본 파일 병합
# - source 이름·읽기 옵션·metadata 컬럼 순서는 adapter_options에서 읽음
# - prefix 방식이면 이름 없는 feature 열에 Profile 접두어와 순번을 부여
# - 두 파일의 행 수가 다르면 concat 전에 오류를 발생시켜 행 정렬 오류를 차단
# ==========================================
def merge_feature_metadata_pair(
    dataset: DatasetSpec, ingestion: IngestionSpec
) -> pd.DataFrame:
    # adapter가 사용할 source 이름과 metadata 컬럼 순서를 Profile에서 읽음
    options = ingestion.adapter_options
    feature_source_name = options.get("feature_source")
    metadata_source_name = options.get("metadata_source")
    metadata_columns = options.get("metadata_columns")

    if not isinstance(feature_source_name, str) or not feature_source_name:
        raise ValueError("feature_metadata_pair에는 adapter_options.feature_source 설정이 필요합니다")
    
    if not isinstance(metadata_source_name, str) or not metadata_source_name:
        raise ValueError("feature_metadata_pair에는 adapter_options.metadata_source 설정이 필요합니다")
    
    if not isinstance(metadata_columns, list) or not metadata_columns or not all(
        isinstance(column, str) and column for column in metadata_columns
    ):
        raise ValueError(
            "feature_metadata_pair에는 비어 있지 않은 adapter_options.metadata_columns 설정이 필요합니다"
        )
    
    if len(metadata_columns) != len(set(metadata_columns)):
        raise ValueError("adapter_options.metadata_columns에 중복 column이 있으면 안 됩니다")

    # source마다 Profile이 선언한 pandas read_csv 옵션을 그대로 적용한다.
    feature_source = ingestion.get_source(feature_source_name)
    metadata_source = ingestion.get_source(metadata_source_name)

    feature_frame = pd.read_csv(
        feature_source.path,
        **feature_source.read_csv_options,
    )

    metadata_frame = pd.read_csv(
        metadata_source.path,
        **metadata_source.read_csv_options,
    )

    # 원본 두 파일은 동일한 wafer 행 순서를 공유해야 하므로 행 수를 먼저 검증
    if len(feature_frame) != len(metadata_frame):
        raise ValueError(
            "feature와 metadata의 행 수가 일치해야 합니다: "
            f"{len(feature_frame)} != {len(metadata_frame)}"
        )

    # metadata 열 수와 순서가 Profile 정의와 다르면 Label·Timestamp 정렬을 신뢰할 수 없음
    if metadata_frame.shape[1] != len(metadata_columns):
        raise ValueError(
            "metadata column 수가 adapter_options.metadata_columns와 일치하지 않습니다: "
            f"{metadata_frame.shape[1]} != {len(metadata_columns)}"
        )

    # Label과 timestamp 열의 의미가 Profile 정의와 일치하는지 먼저 확인한다.
    expected_metadata_columns = [dataset.label_column]

    if dataset.timestamp_column is not None:
        expected_metadata_columns.append(dataset.timestamp_column)

    if metadata_columns != expected_metadata_columns:
        raise ValueError(
            "adapter_options.metadata_columns는 Dataset Profile의 label/timestamp "
            f"column과 일치해야 합니다: 기대값={expected_metadata_columns}, "
            f"실제값={metadata_columns}"
        )

    metadata_frame.columns = metadata_columns

    # 이름 없는 feature 행렬은 prefix 방식에서만 안전하게 canonical 컬럼명으로 변환한다.
    if dataset.feature_selection_mode == "prefix":
        if not dataset.feature_column_prefix:
            raise ValueError("prefix 방식에는 비어 있지 않은 feature column prefix가 필요합니다")
        
        feature_frame.columns = [
            f"{dataset.feature_column_prefix}{index}"
            for index in range(feature_frame.shape[1])
        ]

    elif not all(isinstance(column, str) and column for column in feature_frame.columns):
        raise ValueError(
            "all_except_metadata 방식에는 source에 이름이 있는 feature column이 필요합니다"
        )

    # metadata와 feature를 같은 행 순서로 결합해 canonical table을 생성
    merged_frame = pd.concat([metadata_frame, feature_frame], axis=1)

    # 일반 schema 검증으로 Profile의 Label·feature 정의와 병합 결과를 대조
    validate_dataset_frame(merged_frame, dataset)

    return merged_frame


# adapter 등록부를 한 곳에 모아 merge_dataset의 데이터셋별 분기를 제거한다.
IngestionAdapter = Callable[[DatasetSpec, IngestionSpec], pd.DataFrame]
INGESTION_ADAPTERS: dict[str, IngestionAdapter] = {
    "feature_metadata_pair": merge_feature_metadata_pair,
}


# ==========================================
# Profile이 지정한 ingestion adapter로 canonical merged CSV 생성
# - config.json의 Dataset Profile을 실행 시 선택할 수 있음
# - output_path를 생략하면 Dataset Profile의 input_path에 저장
# - 병합 결과는 이후 범용 EDA·split·모델링 단계의 유일한 입력으로 사용
# ==========================================
def merge_dataset(config_path: Path | str, output_path: Path | None = None) -> Path:
    # 공통 실험 설정과 활성 Dataset Profile을 함께 로드
    config = load_modeling_config(config_path)
    dataset = config.dataset
    ingestion = dataset.ingestion

    # canonical table을 만들 원본 ingestion 정의가 없으면 명확하게 중단
    if ingestion is None:
        raise ValueError(f"데이터셋 '{dataset.dataset_id}'에 ingestion 설정이 없습니다")

    # registry에서 원본 형식별 처리기를 선택한다.
    adapter = INGESTION_ADAPTERS.get(ingestion.adapter)
    if adapter is None:
        raise ValueError(f"지원하지 않는 ingestion adapter입니다: {ingestion.adapter}")
    
    merged_frame = adapter(dataset, ingestion)

    # 호출자가 경로를 주지 않으면 Profile의 canonical input_path를 사용
    destination = output_path or dataset.input_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged_frame.to_csv(destination, index=False)

    # 재실행 확인에 필요한 최소 결과를 콘솔에 기록
    print(f"데이터셋 ID: {dataset.dataset_id}")
    print(f"저장 경로: {destination}")
    print(f"데이터 크기: {merged_frame.shape}")
    print(f"레이블 분포:\n{merged_frame[dataset.label_column].value_counts()}")

    return destination


# ==========================================
# command line 실행 인자 처리
# - --config로 다른 Dataset Profile을 선택한 config.json을 전달
# - --output으로 canonical merged CSV의 저장 위치를 일시적으로 변경
# ==========================================
def parse_arguments() -> argparse.Namespace:
    # 기본 설정은 유지하되, 재현 실험에서 설정·출력 경로만 교체할 수 있게 한다.
    parser = argparse.ArgumentParser(description="Dataset Profile 기반 원본 데이터 병합")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    merge_dataset(arguments.config, arguments.output)

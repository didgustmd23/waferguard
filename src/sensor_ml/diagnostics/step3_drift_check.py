# ==========================================
# Time-based Train/Validation 입력 분포 변화 진단
# - 모델 학습, feature 제거, threshold 조정 없이 split의 입력 분포만 비교
# - 수치형 feature는 결측률, 요약 통계, 표준화 평균 차이, PSI를 기록
# - 범주형 feature는 결측률, 새 범주 비율, 범주 분포 차이를 기록
# - label 비율은 별도 요약으로 기록하되 모델 선택이나 threshold 결정에는 사용하지 않음
#
# 해석 주의:
# - 이 결과는 EDA 및 drift 후보 확인용이며 모델 feature 선택 결과가 아님
# - 실제 전처리와 feature 선택은 Train 또는 CV 학습 fold 내부에서만 fit해야 함
# ==========================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.dataset_schema import validate_dataset_frame
    from src.modeling_config import DEFAULT_CONFIG_PATH, DatasetSpec, load_modeling_config
except ModuleNotFoundError:
    from dataset_schema import validate_dataset_frame
    from modeling_config import DEFAULT_CONFIG_PATH, DatasetSpec, load_modeling_config


# ==========================================
# Train 기준 분위수 구간으로 Population Stability Index 계산
# - Train과 Validation의 구간별 비율 차이를 하나의 값으로 요약
# - Train 값이 상수이거나 유효한 값이 부족하면 안정적인 구간을 만들 수 없어 None 반환
# - PSI 값은 경보가 아니라 분포 변화 후보를 정렬하기 위한 보조 지표
# ==========================================
def calculate_population_stability_index(
    train_values: pd.Series,
    validation_values: pd.Series,
    *,
    bin_count: int = 10,
) -> float | None:
    # 결측치와 무한대는 수치 분포 비교 대상에서 제외한다.
    train_numeric = pd.to_numeric(train_values, errors="coerce").to_numpy(dtype=float)
    validation_numeric = pd.to_numeric(validation_values, errors="coerce").to_numpy(
        dtype=float
    )

    train_numeric = train_numeric[np.isfinite(train_numeric)]
    validation_numeric = validation_numeric[np.isfinite(validation_numeric)]

    # 어느 한쪽이라도 값이 없으면 두 분포의 비율을 계산할 수 없다.
    if train_numeric.size == 0 or validation_numeric.size == 0:
        return None

    # Train 분위수의 중복 경계는 제거한다. 상수 feature는 PSI를 해석하지 않는다.
    boundaries = np.unique(
        np.quantile(train_numeric, np.linspace(0.0, 1.0, bin_count + 1))
    )

    if boundaries.size < 3:
        return None

    # 양 끝을 무한대로 확장해 Validation의 범위 밖 값도 가장자리 구간에 포함한다.
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    train_counts, _ = np.histogram(train_numeric, bins=boundaries)
    validation_counts, _ = np.histogram(validation_numeric, bins=boundaries)

    # 0 비율의 log 계산을 피하기 위해 작은 값을 사용한다.
    epsilon = 1e-6
    train_ratio = np.maximum(train_counts / train_counts.sum(), epsilon)
    validation_ratio = np.maximum(validation_counts / validation_counts.sum(), epsilon)

    return float(np.sum((validation_ratio - train_ratio) * np.log(validation_ratio / train_ratio)))


# ==========================================
# Profile 기반 Train/Validation feature 구성 검증 및 정렬
# - 두 split의 feature 집합이 다르면 drift 수치가 의미 없으므로 즉시 중단
# - Validation column 순서는 Train 기준으로 맞춰 같은 feature끼리 비교
# ==========================================
def align_split_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    # 각 split이 Dataset Profile의 label, metadata, feature 타입 규칙을 만족하는지 확인한다.
    train_features = validate_dataset_frame(train_frame, dataset)
    validation_features = validate_dataset_frame(validation_frame, dataset)

    # 순서 차이는 정렬할 수 있지만 누락 또는 추가 feature는 split 생성 오류로 처리한다.
    if set(train_features) != set(validation_features):
        raise ValueError(
            "Train과 Validation의 feature column 구성이 일치하지 않습니다; "
            f"Train={list(train_features)}, Validation={list(validation_features)}"
        )

    # Train의 feature 순서를 기준으로 두 DataFrame을 만들어 비교 대상을 고정한다.
    return (
        train_frame.loc[:, list(train_features)],
        validation_frame.loc[:, list(train_features)],
        train_features,
    )


# ==========================================
# 수치형 feature의 Train/Validation 분포 변화 요약 행 생성
# - 결측률 변화와 위치 변화(mean/median), 규모가 정규화된 평균 차이를 함께 기록
# - 수치형 분포는 Train 분위수로 만든 PSI를 이용해 추가 비교
# ==========================================
def summarize_numeric_feature(
    feature_name: str,
    train_values: pd.Series,
    validation_values: pd.Series,
) -> dict[str, object]:
    # 결측률은 원본 split의 모든 행을 기준으로 계산한다.
    train_missing_ratio = float(train_values.isna().mean())
    validation_missing_ratio = float(validation_values.isna().mean())

    # pandas가 허용하는 숫자형 중 무한대는 요약 통계에서 제외한다.
    train_numeric = pd.to_numeric(train_values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    validation_numeric = pd.to_numeric(validation_values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    train_mean = train_numeric.mean()
    validation_mean = validation_numeric.mean()
    train_std = train_numeric.std(ddof=0)
    validation_std = validation_numeric.std(ddof=0)

    # 두 split의 분산을 함께 사용해 평균 차이를 scale과 무관하게 비교한다.
    pooled_std = float(np.sqrt((train_std**2 + validation_std**2) / 2))

    if pd.isna(train_mean) or pd.isna(validation_mean) or pooled_std == 0.0:
        standardized_mean_difference: float | None = None
    else:
        standardized_mean_difference = float((validation_mean - train_mean) / pooled_std)

    return {
        "feature": feature_name,
        "feature_type": "numeric",
        "train_missing_ratio": train_missing_ratio,
        "validation_missing_ratio": validation_missing_ratio,
        "missing_ratio_delta": validation_missing_ratio - train_missing_ratio,
        "train_mean": train_mean,
        "validation_mean": validation_mean,
        "train_median": train_numeric.median(),
        "validation_median": validation_numeric.median(),
        "standardized_mean_difference": standardized_mean_difference,
        "population_stability_index": calculate_population_stability_index(
            train_numeric, validation_numeric
        ),
        "unseen_category_ratio": None,
        "category_distribution_tvd": None,
    }


# ==========================================
# 범주형 feature의 Train/Validation 분포 변화 요약 행 생성
# - Validation에만 등장하는 새 범주 비율과 범주 비율의 Total Variation Distance 기록
# - 수치형 통계와 PSI는 적용하지 않아 빈 값을 명시적으로 남김
# ==========================================
def summarize_categorical_feature(
    feature_name: str,
    train_values: pd.Series,
    validation_values: pd.Series,
) -> dict[str, object]:
    # 결측은 범주 분포와 분리해 원본 split 기준으로 계산한다.
    train_missing_ratio = float(train_values.isna().mean())
    validation_missing_ratio = float(validation_values.isna().mean())
    train_non_missing = train_values.dropna()
    validation_non_missing = validation_values.dropna()

    # Train에서 보지 못한 Validation 범주는 one-hot encoder의 일반화 위험 후보로 기록한다.
    if validation_non_missing.empty:
        unseen_category_ratio: float | None = None
    else:
        unseen_category_ratio = float(
            (~validation_non_missing.isin(train_non_missing.unique())).mean()
        )

    # 두 split의 범주 확률 분포 차이를 0~1 범위의 Total Variation Distance로 계산한다.
    if train_non_missing.empty or validation_non_missing.empty:
        category_distribution_tvd: float | None = None
    else:
        train_ratio = train_non_missing.value_counts(normalize=True)
        validation_ratio = validation_non_missing.value_counts(normalize=True)
        categories = train_ratio.index.union(validation_ratio.index)
        category_distribution_tvd = float(
            0.5
            * (train_ratio.reindex(categories, fill_value=0.0) - validation_ratio.reindex(
                categories, fill_value=0.0
            ))
            .abs()
            .sum()
        )

    return {
        "feature": feature_name,
        "feature_type": "categorical",
        "train_missing_ratio": train_missing_ratio,
        "validation_missing_ratio": validation_missing_ratio,
        "missing_ratio_delta": validation_missing_ratio - train_missing_ratio,
        "train_mean": None,
        "validation_mean": None,
        "train_median": None,
        "validation_median": None,
        "standardized_mean_difference": None,
        "population_stability_index": None,
        "unseen_category_ratio": unseen_category_ratio,
        "category_distribution_tvd": category_distribution_tvd,
    }


# ==========================================
# 모든 선택 feature의 Time-based 분포 변화 보고서 생성
# - Profile의 범주형 선언을 기준으로 수치형/범주형 진단 방식을 분리
# - 명시적인 EDA 우선순위 점수로 정렬하지만 자동 제거 또는 모델 입력 변경은 수행하지 않음
# ==========================================
def build_temporal_drift_report(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: DatasetSpec,
) -> pd.DataFrame:
    # 공통 feature만 같은 순서로 맞춘 뒤 type별 요약 행을 만든다.
    train_features, validation_features, feature_columns = align_split_features(
        train_frame, validation_frame, dataset
    )

    categorical_columns = set(dataset.categorical_feature_columns)
    records: list[dict[str, object]] = []

    for column in feature_columns:
        if column in categorical_columns:
            record = summarize_categorical_feature(
                column, train_features[column], validation_features[column]
            )
        else:
            record = summarize_numeric_feature(
                column, train_features[column], validation_features[column]
            )

        records.append(record)

    report = pd.DataFrame(records)

    # 각 지표를 사람이 읽기 쉬운 후보 기준으로 나눠, 가장 큰 변화 하나를 우선순위 점수로 사용한다.
    # 10%p 결측률 변화, |SMD| 0.5, PSI 0.25, 새 범주 10%, TVD 0.2를 각각 점수 1.0으로 둔다.
    report["absolute_missing_ratio_delta"] = report["missing_ratio_delta"].abs()
    report["absolute_standardized_mean_difference"] = report[
        "standardized_mean_difference"
    ].abs()

    priority_components = pd.DataFrame(
        {
            "missing": report["absolute_missing_ratio_delta"] / 0.10,
            "standardized_mean": report["absolute_standardized_mean_difference"] / 0.50,
            "psi": report["population_stability_index"] / 0.25,
            "unseen_category": report["unseen_category_ratio"] / 0.10,
            "category_tvd": report["category_distribution_tvd"] / 0.20,
        }
    )

    report["drift_priority_score"] = priority_components.max(axis=1, skipna=True)

    report["drift_priority"] = np.select(
        [
            report["drift_priority_score"] >= 1.0,
            report["drift_priority_score"] >= 0.5,
        ],
        ["높음", "중간"],
        default="낮음",
    )

    return report.sort_values(
        by=[
            "drift_priority_score",
            "population_stability_index",
            "absolute_standardized_mean_difference",
        ],
        ascending=False,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


# ==========================================
# label 비율 변화 요약 생성
# - Profile의 정상/불량 label 순서를 유지해 Train과 Validation 비율을 비교
# - 모델 feature나 drift score에는 포함하지 않는 별도 EDA 정보
# ==========================================
def build_label_drift_summary(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset: DatasetSpec,
) -> pd.DataFrame:
    # Profile이 허용한 label만 validate_dataset_frame에서 검증된 상태로 사용한다.
    train_labels = train_frame[dataset.label_column]
    validation_labels = validation_frame[dataset.label_column]
    records: list[dict[str, object]] = []
    
    for label in (dataset.negative_label, dataset.positive_label):
        train_ratio = float((train_labels == label).mean())
        validation_ratio = float((validation_labels == label).mean())
        records.append(
            {
                "label": label,
                "train_count": int((train_labels == label).sum()),
                "validation_count": int((validation_labels == label).sum()),
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "ratio_delta": validation_ratio - train_ratio,
            }
        )

    return pd.DataFrame(records)


# ==========================================
# CSV split 파일에서 drift 보고서와 label 요약을 저장
# - 입력 경로를 명시적으로 받아 Test split을 읽거나 평가하지 않음
# - 결과 CSV는 재생성 가능한 EDA 로그이며 Git 추적 대상이 아님
# ==========================================
def run_temporal_drift_check(
    config_path: Path | str,
    train_path: Path | str,
    validation_path: Path | str,
    output_path: Path | str,
    label_output_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 설정과 지정된 Train/Validation split만 읽는다.
    config = load_modeling_config(config_path)
    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)

    # feature 구조를 검증하면서 생성한 두 보고서를 별도 CSV로 기록한다.
    report = build_temporal_drift_report(train_frame, validation_frame, config.dataset)
    label_summary = build_label_drift_summary(
        train_frame, validation_frame, config.dataset
    )

    destination = Path(output_path)
    label_destination = Path(label_output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False, encoding="utf-8-sig")
    label_summary.to_csv(label_destination, index=False, encoding="utf-8-sig")

    return report, label_summary


def _parse_arguments() -> argparse.Namespace:
    # Time Train/Validation split과 두 결과 로그의 경로를 명시적으로 받는다.
    parser = argparse.ArgumentParser(description="Time-based 분포 변화 진단")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/secom/exploratory/legacy/temporal_drift_report.csv")
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        default=Path("logs/secom/exploratory/legacy/temporal_drift_label_summary.csv"),
    )

    return parser.parse_args()


def main() -> None:
    # 지정된 Time Train/Validation split의 feature·label 분포 차이를 기록한다.
    arguments = _parse_arguments()
    config = load_modeling_config(arguments.config)
    
    report, label_summary = run_temporal_drift_check(
        arguments.config,
        arguments.train,
        arguments.validation,
        arguments.output,
        arguments.label_output,
    )

    print(f"분석 feature 수: {len(report)}")

    positive_ratio_delta = label_summary.loc[
        label_summary["label"] == config.dataset.positive_label, "ratio_delta"
    ].iloc[0]
    
    print(f"불량 label 비율 변화: {positive_ratio_delta:.4f}")
    print(f"분포 변화 보고서 저장 경로: {arguments.output}")
    print(f"label 요약 저장 경로: {arguments.label_output}")


if __name__ == "__main__":
    main()

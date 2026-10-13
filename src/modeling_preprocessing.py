# ==========================================
# Dataset Profile 기반 공통 모델 전처리
# - 수치형은 median 대치, 범주형은 최빈값 대치와 One-Hot Encoding 사용
# - 대치 전에 학습 데이터의 결측률 초과·상수 센서를 제거하고 근거 기록
# - 전처리 학습은 호출자가 생성한 Pipeline의 학습 fold에서만 수행
# ==========================================

from functools import partial
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 학습 데이터에서 센서 품질 기준을 계산하고 제거 목록 저장
# - 결측률 초과 센서를 먼저 제거한 뒤 남은 센서의 관측값 종류를 검사
# - transform에서는 fit으로 정한 센서 목록과 순서를 그대로 적용
# - 제거 사유별 목록은 중복 없이 기록하며 입력 DataFrame을 보존
# ==========================================
class SensorQualityFilter(TransformerMixin, BaseEstimator):
    """학습 구간에서 불량 센서를 찾고 동일한 열 집합을 이후 데이터에 적용한다.

    결측률 기준 초과 센서를 먼저 제거하고 남은 센서 중 관측된 고유값이
    1개 이하인 센서를 제거한다. 변환 시 평가 데이터로 기준을 재계산하지 않는다.
    """
    def __init__(self, missing_ratio_threshold=0.5, drop_zero_variance=True):
        self.missing_ratio_threshold = missing_ratio_threshold
        self.drop_zero_variance = drop_zero_variance

    def fit(self, X, y=None):
        """입력 학습 데이터의 결측률과 상수 여부를 계산해 유지 열을 기억한다."""
        # 센서 이름으로 선택하므로 이름이 중복되지 않은 DataFrame을 사용한다.
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("센서 품질 필터에는 컬럼명이 중복되지 않은 DataFrame이 필요합니다.")
        if not 0 <= self.missing_ratio_threshold <= 1:
            raise ValueError("센서 결측률 기준은 0 이상 1 이하여야 합니다.")
        
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.missing_ratios_ = X.isna().mean()
        self.high_missing_features_ = tuple(
            X.columns[self.missing_ratios_ > self.missing_ratio_threshold]
        )

        # 관측값이 하나뿐인 컬럼은 결측값 대치 이전에 상수 센서로 판정한다.
        remaining = [column for column in X if column not in self.high_missing_features_]
        self.constant_features_ = tuple(
            column for column in remaining
            if self.drop_zero_variance and X[column].nunique(dropna=True) <= 1
        )

        self.retained_features_ = tuple(
            column for column in remaining if column not in self.constant_features_
        )

        if not self.retained_features_:
            raise ValueError("센서 품질 필터 적용 후 남은 feature가 없습니다. 학습 데이터와 Profile 기준을 확인하세요.")
        return self

    def transform(self, X):
        """학습 시 유지하기로 한 열만 원래 순서대로 반환한다."""
        check_is_fitted(self, "retained_features_")
        
        # 평가 데이터의 결측률이나 상수 여부를 다시 계산하지 않는다.
        if not isinstance(X, pd.DataFrame):
            raise ValueError("센서 품질 필터에는 DataFrame이 필요합니다.")
        
        missing = sorted(set(self.retained_features_) - set(X.columns))

        if missing:
            raise ValueError(f"학습에서 유지한 feature가 입력에 없습니다: {missing}")
        
        return X.loc[:, list(self.retained_features_)].copy()

    def get_feature_names_out(self, input_features=None):
        """필터 이후 남은 특징 이름을 scikit-learn API 형식으로 돌려준다."""
        check_is_fitted(self, "retained_features_")

        return np.asarray(self.retained_features_, dtype=object)


# ==========================================
# 학습된 품질 필터의 제거 개수·센서명·기준을 CSV 저장 가능한 JSON으로 기록
# - CV 호출자는 fold 번호와 함께 각 학습 결과를 모아 기록
# ==========================================
def quality_filter_record(pipeline, *, fold=None):
    """fit된 Pipeline의 센서 제거 기준·개수·이름을 기록용 사전으로 만든다."""
    quality = pipeline.named_steps["quality_filter"]
    
    return {
        "fold": fold,
        "input_feature_count": quality.n_features_in_,
        "missing_ratio_threshold": quality.missing_ratio_threshold,
        "drop_zero_variance": quality.drop_zero_variance,
        "high_missing_removed_count": len(quality.high_missing_features_),
        "constant_removed_count": len(quality.constant_features_),
        "removed_feature_count": len(quality.high_missing_features_) + len(quality.constant_features_),
        "retained_feature_count": len(quality.retained_features_),
        "high_missing_features": list(quality.high_missing_features_),
        "constant_features": list(quality.constant_features_),
    }


def quality_filter_json(records):
    """품질 기록 목록을 한글 센서명도 보존하는 JSON 문자열로 직렬화한다."""
    # JSON 배열로 fold별 수치와 센서명을 함께 보존해 평균값의 정보 손실을 방지한다.
    return json.dumps(records, ensure_ascii=False)


def _numeric_columns(frame, *, categorical_columns):
    """Dataset 설정에서 범주형으로 지정되지 않은 변환 후 열을 반환한다."""
    # 숫자로 저장된 범주형 컬럼도 Profile 선언을 우선하여 수치형에서 제외한다.
    return [column for column in frame.columns if column not in categorical_columns]


def _categorical_columns(frame, *, categorical_columns):
    """Dataset 설정에 선언된 범주형 열 중 현재 입력에 있는 열을 반환한다."""
    # 품질 필터가 제거한 범주형 컬럼은 ColumnTransformer에 전달하지 않는다.
    return [column for column in frame.columns if column in categorical_columns]


def preprocessing_steps(dataset, *, scale=False, pandas_output=False):
    """Dataset Profile에 맞는 품질 필터·대치·인코딩 Pipeline 단계를 만든다.

    모든 변환기는 반환된 Pipeline의 fit 시점에만 통계를 학습하므로 CV
    fold별 전처리를 보장한다. ``scale``은 수치형 표준화 여부이며,
    ``pandas_output``은 선택기에서 특징 이름을 추적해야 할 때 사용한다.
    """
    quality_steps = [("quality_filter", SensorQualityFilter(
        dataset.missing_ratio_threshold, dataset.drop_zero_variance
    ))]

    # 수치형 데이터의 기존 단계 이름은 유지해 기존 실행 경로와 호환한다.
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    if not dataset.categorical_feature_columns:
        if pandas_output:
            for _, transformer in numeric_steps:
                transformer.set_output(transform="pandas")
        return quality_steps + numeric_steps

    # 범주형은 dense 출력으로 통일하여 PCA와 LightGBM도 같은 입력을 사용할 수 있게 한다.
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), partial(
            _numeric_columns, categorical_columns=dataset.categorical_feature_columns
        )),
        ("categorical", categorical, partial(
            _categorical_columns, categorical_columns=dataset.categorical_feature_columns
        )),
    ], sparse_threshold=0)

    if pandas_output:
        preprocessor.set_output(transform="pandas")
        
    return quality_steps + [("preprocessor", preprocessor)]


def fitted_feature_count(pipeline):
    """학습된 Pipeline의 selector 또는 최종 모델이 받은 특징 수를 반환한다."""
    # 선택기 또는 최종 모델이 실제로 받은 변환 후 특징 수를 기록한다.
    if "selector" in pipeline.named_steps:
        selector = pipeline.named_steps["selector"]
        if hasattr(selector, "n_components_"):
            return int(selector.n_components_)
        return int(selector.get_support().sum())
    return int(pipeline.named_steps["model"].n_features_in_)


def checked_top_k(features, *, count):
    """현재 fold에서 변환된 특징 수를 기준으로 Top-K 설정을 검증한다."""
    # One-Hot Encoding 이후의 실제 fold 입력 크기로 Top-K를 검사한다.
    if count > features.shape[1]:
        raise ValueError(f"Top-K feature 수가 변환 후 feature 수보다 큽니다: {count} > {features.shape[1]}")
    return count

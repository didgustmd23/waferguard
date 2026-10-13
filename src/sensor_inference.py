# ==========================================
# 선택된 수치형 센서만 사용하는 독립 추론 코어
# - 학습된 Pipeline에서 센서 순서·median·scaling·분류기를 가져옴
# - 학습이나 센서 재선택 없이 이름 기준으로 입력을 정렬하고 예측
# - label·timestamp·Dataset Profile·Agent는 추론 입력에 필요하지 않음
# - PCA·범주형 인코딩·센서 간 연산은 이 변환의 지원 대상이 아님
# ==========================================

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class SensorInference:
    """고정 센서의 학습 통계와 분류기를 보유한 수치형 추론 객체.

    직접 생성하기보다 ``build_sensor_inference``로 학습된 Pipeline을 변환한다.
    이 객체에는 fit 메서드가 없으며, 이후 설정 파일 변경도 적용되지 않는다.
    """

    def __init__(self, sensors, medians, model, positive_label, threshold,
                 *, mean=None, scale=None):
        # 모델을 독립 복사하여 원본 Pipeline 변경이 추론 객체에 전파되지 않게 한다.
        self.sensors = tuple(sensors)
        self._medians = np.asarray(medians, dtype=float).copy()
        self._mean = None if mean is None else np.asarray(mean, dtype=float).copy()
        self._scale = None if scale is None else np.asarray(scale, dtype=float).copy()
        self._model = deepcopy(model)
        self.positive_label = positive_label
        self.threshold = threshold
        self._validate_contract()

    def _validate_contract(self):
        """센서·통계·모델의 특징 수와 양성 label의 대응을 확인한다."""
        if (not self.sensors or len(set(self.sensors)) != len(self.sensors)
                or not all(isinstance(name, str) for name in self.sensors)):
            raise ValueError("센서 목록은 중복 없는 문자열 목록이어야 합니다.")
        
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float))
                or not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1):
            raise ValueError("추론 threshold는 0 이상 1 이하의 유한한 숫자여야 합니다.")
        
        # 각 배열의 길이가 센서 순서와 일치해야 다른 센서의 통계가 적용되지 않는다.
        for values in (self._medians, self._mean, self._scale):
            if values is not None and (values.shape != (len(self.sensors),)
                                       or not np.isfinite(values).all()):
                raise ValueError("학습 통계는 센서 수와 일치하는 유한한 1차원 배열이어야 합니다.")
            
        if self._scale is not None and (self._scale <= 0).any():
            raise ValueError("학습 scaling 값은 모두 양수여야 합니다.")
        
        check_is_fitted(self._model)

        if getattr(self._model, "n_features_in_", None) != len(self.sensors):
            raise ValueError("분류기의 학습 특징 수가 추론 센서 수와 다릅니다.")
        
        classes = list(self._model.classes_)

        if len(classes) != 2 or classes.count(self.positive_label) != 1:
            raise ValueError("분류기에 두 종류의 label과 지정한 양성 label이 필요합니다.")
        
        self._positive_index = classes.index(self.positive_label)
        self._negative_label = classes[1 - self._positive_index]

    def _prepare_input(self, frame):
        """고정 센서 입력을 검증하고 저장된 통계로만 변환한다."""
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("추론 입력은 컬럼명이 중복되지 않은 DataFrame이어야 합니다.")
        
        missing = [name for name in self.sensors if name not in frame.columns]
        extra = [name for name in frame.columns if name not in self.sensors]

        if missing or extra:
            raise ValueError(f"추론 센서 컬럼이 계약과 다릅니다. 누락: {missing}, 추가: {extra}")
        
        if frame.empty:
            raise ValueError("추론 입력에 한 행 이상 필요합니다.")
        
        # 문자열 숫자를 자동 변환하지 않아 단위·타입 오류를 조용히 감추지 않는다.
        ordered = frame.loc[:, list(self.sensors)]

        if any(not pd.api.types.is_numeric_dtype(dtype)
               or pd.api.types.is_bool_dtype(dtype)
               or pd.api.types.is_complex_dtype(dtype) for dtype in ordered.dtypes):
            raise ValueError("추론 센서 값은 실수형 또는 정수형이어야 합니다. NaN은 허용합니다.")
        
        values = ordered.to_numpy(dtype=float, na_value=np.nan, copy=True)

        if np.isinf(values).any():
            raise ValueError("추론 센서 값에 무한대가 있습니다.")
        
        if np.isnan(values).all(axis=1).any():
            raise ValueError("모든 필수 센서가 결측인 행은 예측할 수 없습니다.")
        
        # 배치의 median을 계산하지 않는다. 선택 당시 각 센서의 학습 median만 사용한다.
        values = np.where(np.isnan(values), self._medians, values)

        if self._mean is not None:
            values = values - self._mean

        if self._scale is not None:
            values = values / self._scale

        if not np.isfinite(values).all():
            raise ValueError("전처리 결과에 유한하지 않은 값이 있습니다.")
        
        # 기존 분류기가 특징 이름을 학습한 경우에만 동일한 이름을 붙여 전달한다.
        if hasattr(self._model, "feature_names_in_"):
            return pd.DataFrame(values, columns=self.sensors, index=frame.index)
        
        return values

    def predict(self, frame):
        """양성 확률·문턱 판정·원래 label을 입력 행 순서대로 반환한다."""
        values = self._prepare_input(frame)
        scores = self._model.predict_proba(values)[:, self._positive_index]
        
        # 일반 분류기의 기본 0.5 판정 대신 이 객체에 지정한 후보 문턱을 적용한다.
        positive = scores >= self.threshold

        return pd.DataFrame({
            "positive_score": scores,
            "predicted_positive": positive,
            "predicted_label": np.where(positive, self.positive_label, self._negative_label),
        }, index=frame.index)


# ==========================================
# 학습된 독립 수치형 전처리를 센서 입력 계약으로 변환
# - sklearn 객체 내부 배열을 수정하거나 새 전처리기를 fit하지 않음
# - selector의 중요도 순위가 아니라 get_support의 실제 입력 순서를 보존
# - 지원하지 않는 변환은 추측해서 처리하지 않고 오류로 중단
# max_sensors: 프로젝트에서 허용하는 원본 센서 개수 상한
# threshold: 호출자가 명시한 후보값이며 최종 운영값 확정을 뜻하지 않음
# ==========================================
def build_sensor_inference(pipeline, *, positive_label, threshold, max_sensors):
    """학습된 수치형 선택 Pipeline을 재학습 없이 축소 추론 객체로 만든다."""
    names = [name for name, _ in pipeline.steps]

    if names not in (["quality_filter", "imputer", "selector", "model"],
                     ["quality_filter", "imputer", "scaler", "selector", "model"]):
        raise ValueError("수치형 품질 필터·median 대치·선택기·분류기 경로만 지원합니다.")
    
    if isinstance(max_sensors, bool) or not isinstance(max_sensors, int) or max_sensors < 1:
        raise ValueError("센서 개수 상한은 양의 정수여야 합니다.")
    
    quality = pipeline.named_steps["quality_filter"]
    imputer = pipeline.named_steps["imputer"]
    selector = pipeline.named_steps["selector"]

    if (not isinstance(imputer, SimpleImputer) or imputer.strategy != "median"
            or imputer.add_indicator or not isinstance(selector, SelectFromModel)):
        raise ValueError("독립 median 대치와 SelectFromModel 선택기만 지원합니다.")
    
    check_is_fitted(quality, "retained_features_")
    check_is_fitted(imputer, "statistics_")
    check_is_fitted(selector, "estimator_")

    original_names = np.asarray(quality.retained_features_, dtype=object)

    # 제거·확장된 대치 출력이나 이름 불일치는 원본 센서에 일대일 대응하지 않는다.
    if (not np.array_equal(imputer.feature_names_in_, original_names)
            or not np.array_equal(imputer.get_feature_names_out(), original_names)
            or not np.isfinite(imputer.statistics_).all()):
        raise ValueError("대치 결과와 원본 센서가 일대일 대응하지 않습니다.")
    
    if not np.array_equal(pipeline[:-2].get_feature_names_out(), original_names):
        raise ValueError("선택기 이전의 특징 이름 또는 순서가 변경되었습니다.")
    
    if (hasattr(selector, "feature_names_in_")
            and not np.array_equal(selector.feature_names_in_, original_names)):
        raise ValueError("선택기가 학습한 센서 순서가 전처리 결과와 다릅니다.")
    
    mask = selector.get_support()

    if mask.shape != original_names.shape or not 0 < mask.sum() <= max_sensors:
        raise ValueError("선택된 센서 수가 비어 있거나 허용 상한을 초과합니다.")
    
    mean = scale = None

    if "scaler" in pipeline.named_steps:
        scaler = pipeline.named_steps["scaler"]
        if not isinstance(scaler, StandardScaler):
            raise ValueError("센서별 StandardScaler만 지원합니다.")
    
        check_is_fitted(scaler)
        mean = scaler.mean_[mask] if scaler.with_mean else None
        scale = scaler.scale_[mask] if scaler.with_std else None

    model = pipeline.named_steps["model"]
    selected_names = original_names[mask]

    if (hasattr(model, "feature_names_in_")
            and not np.array_equal(model.feature_names_in_, selected_names)):
        raise ValueError("분류기 학습 센서 순서가 선택기 결과와 다릅니다.")
    
    return SensorInference(selected_names, imputer.statistics_[mask], model,
                           positive_label, threshold, mean=mean, scale=scale)

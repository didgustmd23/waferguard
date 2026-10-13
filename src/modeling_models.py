# ==========================================
# 모델 정의와 공통 학습·확률 예측
# - Step별 모델 설정 중복을 없애고 동일한 실험 이름은 동일한 Pipeline으로 생성
# - 전처리·특징 선택은 Pipeline 내부에서 현재 학습 구간에만 fit
# - 불균형 가중치는 호출자가 전달한 학습 label에서만 계산
# - 외부 Validation/Test와 threshold 선택은 이 모듈에서 처리하지 않음
# ==========================================

from functools import partial

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

try:
    from src.stable_rf_selector import StableRFSelector
    from src.modeling_preprocessing import preprocessing_steps, checked_top_k
except ModuleNotFoundError:
    from stable_rf_selector import StableRFSelector
    from modeling_preprocessing import preprocessing_steps, checked_top_k


class XGBoostClassifierAdapter(ClassifierMixin, BaseEstimator):
    """문자열 또는 -1/1 label을 XGBoost의 0/1 입력으로 변환하는 분류기.

    scikit-learn의 Pipeline과 clone 호환성을 유지하고, 학습 후 ``classes_``를
    원래 label 값으로 노출하여 공통 평가 코드가 양성 class를 찾게 한다.
    XGBoost는 실제 fit 시점에만 지연 import해 미설치 오류를 명확히 전달한다.
    """

    def __init__(self, positive_label=1, negative_label=-1,
                 n_estimators=300, max_depth=3, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 scale_pos_weight=1.0, random_state=42, n_jobs=1,
                 importance_type="gain", class_weight_mode="none"):
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        # 선택 기준을 명시적으로 보관한다. gain은 평균 학습 손실 개선량이다.
        self.importance_type = importance_type
        # 새 시간순 실험에서만 사용한다. 기본 none은 기존 고정 가중치 경로를 유지한다.
        self.class_weight_mode = class_weight_mode

    def fit(self, X, y):
        """원래 label을 인코딩하고 XGBoost 분류기를 학습한다."""
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "XGBoost 후보를 사용하려면 requirements.txt의 xgboost 의존성을 설치하세요."
            ) from error

        if self.positive_label == self.negative_label:
            raise ValueError("XGBoost의 양성·음성 label은 서로 달라야 합니다.")
        
        if pd.Series(y).isna().any():
            raise ValueError("XGBoost 학습 label에 결측값이 있습니다.")
        
        observed_labels = set(pd.Series(y).unique())

        if observed_labels != {self.negative_label, self.positive_label}:
            raise ValueError("XGBoost 후보는 이진 분류 label 두 종류가 필요합니다.")
        
        # SECOM처럼 -1/1을 쓰거나 다른 문자열 label을 써도 불량은 항상 1로 학습한다.
        encoded_y = np.asarray(pd.Series(y).eq(self.positive_label), dtype=np.int32)

        if self.class_weight_mode not in {"none", "sqrt_ratio", "ratio"}:
            raise ValueError("불량 가중치 방식은 none·sqrt_ratio·ratio 중 하나여야 합니다.")
        
        # 현재 fit에 전달된 label만 사용하므로 바깥 평가 구간이나 미래 불량률이 섞이지 않는다.
        ratio = float((encoded_y == 0).sum() / (encoded_y == 1).sum())

        self.effective_scale_pos_weight_ = {
            "none": self.scale_pos_weight, "sqrt_ratio": np.sqrt(ratio), "ratio": ratio,
        }[self.class_weight_mode]

        self.classes_ = np.asarray([self.negative_label, self.positive_label])
        self.n_features_in_ = X.shape[1]

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self.estimator_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.effective_scale_pos_weight_,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            importance_type=self.importance_type,
        )

        self.estimator_.fit(X, encoded_y)
        return self

    def predict(self, X):
        """내부 0/1 예측을 원래 label 값으로 되돌려 반환한다."""
        check_is_fitted(self, "estimator_")
        encoded = self.estimator_.predict(X).astype(int)
        return np.where(encoded == 1, self.positive_label, self.negative_label)

    def predict_proba(self, X):
        """XGBoost의 두 클래스 확률을 반환한다."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        """특징 선택기가 사용할 수 있도록 내부 모델의 중요도를 노출한다."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_


# 실제 후보 이름과 기존 특징 실험 이름을 유지하여 CSV·CLI 호환성을 보존한다.
CANDIDATE_VARIANTS = (
    "logistic_regression_l1", "logistic_regression_l1_balanced", "rbf_svm",
    "random_forest", "lightgbm", "lightgbm_scale_pos_weight",
    "xgboost", "xgboost_scale_pos_weight",
)
EXPERIMENT_ALIASES = {
    "l1_balanced_all": "logistic_regression_l1_balanced",
    "l1_balanced_pca90": "logistic_regression_l1_balanced",
    "l1_balanced_l1_select": "logistic_regression_l1_balanced",
    "lightgbm_all": "lightgbm",
    "xgboost_all": "xgboost",
}
WEIGHTED_VARIANTS = {"lightgbm_scale_pos_weight", "xgboost_scale_pos_weight"}
# 명시적으로 요청한 RF 복잡도 비교에만 사용한다. 기본 후보·RF 선택기는 유지한다.
RF_LEAF_VARIANTS = {"random_forest_leaf3": 3, "random_forest_leaf5": 5}


def candidate_base_name(name):
    """가중치 변형 이름을 config.json의 기본 후보 이름으로 변환한다."""
    return name.removesuffix("_balanced").removesuffix("_scale_pos_weight")


# ==========================================
# 분류기 설정의 단일 정의
# - 특징 선택용 분류기도 동일한 생성 함수를 사용
# - 가중치 비율은 여기서 계산하지 않고 학습할 때만 결정
# ==========================================
def build_classifier(config, name, *, n_jobs=1, n_estimators=300):
    """실험의 분류기를 만들되 아직 학습하지 않는다."""
    seed = config.experiment.cv.random_state

    if name in {"logistic_regression_l1", "logistic_regression_l1_balanced"}:
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
            class_weight="balanced" if name.endswith("_balanced") else None,
        )
    
    if name == "rbf_svm":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    
    if name in {"random_forest", "random_forest_balanced"} or name in RF_LEAF_VARIANTS:
        return RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
            # 현재 fit의 클래스 빈도로 가중치를 계산한다. 깊이·leaf 설정은 유지한다.
            class_weight="balanced" if name.endswith("_balanced") else None,
            # 마지막 노드에 필요한 최소 표본 수만 바꿔 작은 표본에 대한 분할을 제한한다.
            min_samples_leaf=RF_LEAF_VARIANTS.get(name, 1),
        )
    
    if name in {"lightgbm", "lightgbm_scale_pos_weight"}:
        return LGBMClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs, verbosity=-1,
        )
    
    if name in {"xgboost", "xgboost_scale_pos_weight"}:
        return XGBoostClassifierAdapter(
            positive_label=config.dataset.positive_label,
            negative_label=config.dataset.negative_label,
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
        )
    
    raise ValueError(f"정의되지 않은 후보 모델입니다: {name}")


# ==========================================
# RF 선택기에만 복잡도 제한 적용
# - 최종 분류기 설정은 변경하지 않아 센서 선택 효과만 비교
# - 기본값은 기존 S0와 동일하며 S1·S2는 호출자가 명시
# ==========================================
def configure_rf_selectors(pipelines, *, min_samples_leaf=1, max_depth=None, stability_repeats=0):
    # bool·0·음수는 sklearn 학습 전에 읽기 쉬운 오류로 차단한다.
    for name, value in (("min_samples_leaf", min_samples_leaf), ("max_depth", max_depth)):
        if value is None and name == "max_depth":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"RF 선택기의 {name}는 양의 정수여야 합니다.")
        
    found = False

    if isinstance(stability_repeats, bool) or not isinstance(stability_repeats, int) or stability_repeats < 0 or stability_repeats == 1:
        raise ValueError("RF 반복 선택 횟수는 0(사용 안 함) 또는 2 이상의 정수여야 합니다.")
    
    for pipeline in pipelines.values():
        selector = pipeline.named_steps.get("selector")
        if selector is not None and isinstance(getattr(selector, "estimator", None), RandomForestClassifier):
            # 선택용 RF만 변경하고 pipeline의 model(XGBoost)은 건드리지 않는다.
            selector.estimator.set_params(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
            if stability_repeats:
                # 원본 센서를 재표집해야 하므로 선택기를 대치기 앞에 배치한다.
                # 내부 bootstrap 전처리는 기존 전처리 경로의 독립 복제본을 사용한다.
                preprocessing = Pipeline([(name, clone(step)) for name, step in pipeline.steps
                                          if name not in {"selector", "model"}])
                # 실제 변환 결과도 selector.fit에서 검사하므로 범주형 인코딩은 묵시적으로 허용하지 않는다.
                stable = StableRFSelector(clone(selector.estimator), preprocessing, selector.max_features,
                                          n_repeats=stability_repeats,
                                          random_state=selector.estimator.random_state)
                pipeline.steps = [("quality_filter", pipeline.named_steps["quality_filter"]),
                                  ("selector", stable)] + [
                                      (name, step) for name, step in pipeline.steps
                                      if name not in {"quality_filter", "selector", "model"}
                                  ] + [("model", pipeline.named_steps["model"])]
            found = True

    if not found and (min_samples_leaf != 1 or max_depth is not None or stability_repeats):
        raise ValueError("RF 선택기 설정을 적용할 RF 중요도 실험이 없습니다.")
    
    return pipelines


# ==========================================
# RF·gain Top-K 뒤의 최종 XGBoost에만 실험용 깊이·정규화 적용
# - 전체 센서 M0와 센서 선택용 모델은 기존 설정을 그대로 보존
# - 옵션 생략 시 기존 S0~S3 동작을 유지
# - 센서 수·데이터셋에 고정하지 않고 XGBoost Top-K 경로에 적용
# ==========================================
def configure_topk_xgb(pipelines, *, max_depth=None, reg_lambda=None):
    if max_depth is None and reg_lambda is None:
        return pipelines
    
    params = {}

    # bool도 int로 취급되므로 별도로 차단하고 학습 전에 오류를 전달한다.
    if max_depth is not None:
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("Top-K 최종 XGBoost의 최대 깊이는 양의 정수여야 합니다.")
        params["model__max_depth"] = max_depth

    # 정규화 0은 허용하지만 음수·NaN·무한대·문자열·bool은 학습 전에 차단한다.
    if reg_lambda is not None:
        if (isinstance(reg_lambda, bool) or not isinstance(reg_lambda, (int, float))
                or not np.isfinite(reg_lambda) or reg_lambda < 0):
            raise ValueError("Top-K 최종 XGBoost의 reg_lambda는 유한한 0 이상의 숫자여야 합니다.")
        params["model__reg_lambda"] = reg_lambda

    targets = [pipeline for name, pipeline in pipelines.items()
               if name.startswith(("xgboost_rf_top_", "xgboost_top_"))
               and isinstance(pipeline.named_steps.get("model"), XGBoostClassifierAdapter)]
    
    if not targets:
        raise ValueError("설정을 적용할 Top-K → XGBoost 실험이 없습니다.")
    
    for pipeline in targets:
        # 선택기와 bootstrap 설정은 건드리지 않고 최종 분류기만 변경한다.
        pipeline.set_params(**params)

    return pipelines


def build_pipeline(config, name, *, n_jobs=1):
    """모델·특징 선택 실험 이름 하나를 독립된 Pipeline으로 변환한다."""
    # 분류기와 선택기를 구분한다. RF Top-K 실험도 최종 분류기는 XGBoost다.
    # K는 기존 설정 목록을 사용하므로 특정 데이터셋이나 센서 20개에 고정하지 않는다.
    top_k_specs = {}

    for prefix, final_model, importance_model in (
        ("lightgbm_top_", "lightgbm", "lightgbm"),
        ("xgboost_top_", "xgboost", "xgboost"),
        ("xgboost_rf_top_", "xgboost", "random_forest"),
    ):
        for count in config.top_k_feature_counts:
            top_k_specs[f"{prefix}{count}"] = (final_model, importance_model, count)
            
    selection = top_k_specs.get(name)
    model_name = EXPERIMENT_ALIASES.get(name, selection[0] if selection else name)
    classifier = build_classifier(config, model_name, n_jobs=n_jobs)
    scale = model_name.startswith("logistic_regression") or model_name == "rbf_svm"
    feature_experiment = name in EXPERIMENT_ALIASES or selection is not None
    steps = preprocessing_steps(config.dataset, scale=scale, pandas_output=feature_experiment)

    # 선택기를 전처리 뒤에 넣어 해당 학습 fold의 변환된 특징만 보게 한다.
    if name == "l1_balanced_pca90":
        steps.append(("selector", PCA(
            n_components=config.experiment.pca_explained_variance,
            random_state=config.experiment.cv.random_state,
        )))
    elif name == "l1_balanced_l1_select":
        steps.append(("selector", SelectFromModel(clone(classifier))))
    elif selection is not None:
        # 전체 특징으로 학습하는 선택용 모델과 20개 등 축소 특징의 분류기는 별개다.
        importance = build_classifier(config, selection[1], n_jobs=n_jobs)
        steps.append(("selector", SelectFromModel(
            importance, threshold=-np.inf,
            max_features=partial(checked_top_k, count=selection[2]),
        )))

    return Pipeline(steps + [("model", classifier)])


def build_candidate_pipelines(config, n_jobs=1, *, names=None):
    """기본 후보 또는 명시적으로 요청한 후보만 생성한다."""
    # RF 가중치·leaf 변형은 명시적 비교에서만 사용해 기본 후보를 늘리지 않는다.
    names = CANDIDATE_VARIANTS if names is None else tuple(names)
    unknown = (set(names) - set(CANDIDATE_VARIANTS)
               - {"random_forest_balanced"} - set(RF_LEAF_VARIANTS))
    
    if unknown:
        raise ValueError(f"지원하지 않는 후보 모델입니다: {sorted(unknown)}")
    
    return {name: build_pipeline(config, name, n_jobs=n_jobs) for name in names}


def build_experiments(config, n_jobs=1):
    """기존 Step 6의 전체·PCA·L1·Top-K 실험을 생성한다."""
    names = ("l1_balanced_all", "l1_balanced_pca90", "l1_balanced_l1_select", "lightgbm_all")
    names += tuple(f"lightgbm_top_{count}" for count in config.top_k_feature_counts)
    return {name: build_pipeline(config, name, n_jobs=n_jobs) for name in names}


# ==========================================
# 학습 fold에만 의존하는 공통 fit
# - clone으로 이전 fold의 학습 상태를 재사용하지 않음
# - 병렬 수는 분류기·선택기에서 실제 지원하는 파라미터에만 적용
# - 가중치 변형은 현재 학습 label의 음성/양성 비율을 사용
# ==========================================
def fit_pipeline(pipeline, features, labels, config, experiment_name, *, n_jobs=1):
    """새 Pipeline을 학습하고 원본 미학습 Pipeline은 변경하지 않는다."""
    label_series = pd.Series(labels)
    expected = {config.dataset.negative_label, config.dataset.positive_label}

    if label_series.isna().any() or set(label_series.unique()) != expected:
        raise ValueError("학습 구간에는 결측 없이 설정된 정상·Fail label이 모두 필요합니다.")
    
    model = clone(pipeline)

    # 이름 접두어 대신 지원 파라미터를 확인해 사용자 정의 Pipeline도 처리한다.
    parallel_params = {
        key: n_jobs for key in model.get_params(deep=True)
        if key.endswith("__n_jobs")
    }

    if parallel_params:
        model.set_params(**parallel_params)

    if experiment_name in WEIGHTED_VARIANTS:
        positive_count = int(label_series.eq(config.dataset.positive_label).sum())
        negative_count = int(label_series.eq(config.dataset.negative_label).sum())
        model.set_params(model__scale_pos_weight=negative_count / positive_count)
    model.fit(features, labels)

    return model


def positive_scores(pipeline, features, dataset):
    """class 순서와 무관하게 설정된 양성 label의 확률만 반환한다."""
    # 문자열 label에서는 양성 class가 첫 번째일 수도 있으므로 [:, 1]을 가정하지 않는다.
    classes = pipeline.named_steps["model"].classes_
    positions = [index for index, label in enumerate(classes) if label == dataset.positive_label]

    if len(positions) != 1:
        raise ValueError("학습된 후보 모델에서 Profile의 Fail label을 찾을 수 없습니다.")
        
    return pipeline.predict_proba(features)[:, positions[0]]

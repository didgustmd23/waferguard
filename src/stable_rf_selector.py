# ==========================================
# 학습 fold 내부의 반복 중요도 선택으로 원본 센서 Top-K 결정
# - 클래스별 샘플 수를 유지하는 계층 bootstrap을 복원 추출
# - 반복마다 전처리와 선택 모델을 새로 fit하며 검증 데이터는 받지 않음
# - 선택 빈도 내림차순 → 평균 순위 오름차순 → 원래 열 순서로 선정
# - 최종 XGBoost는 선택된 원본 센서로 별도 학습
# ==========================================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


class StableRFSelector(TransformerMixin, BaseEstimator):
    # 기존 RF 클래스와 저장 경로를 유지하고 gain 선택기는 아래에서 같은 코어를 재사용한다.
    selection_method = "rf_stability"
    def __init__(self, estimator, preprocessing, max_features, n_repeats=10, random_state=42):
        # 생성자에서는 매개변수를 그대로 보관해 sklearn.clone과 호환되게 한다.
        self.estimator = estimator
        self.preprocessing = preprocessing
        self.max_features = max_features
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y):
        # 이름 기반 선택은 수치형 원본 센서만 지원한다. One-Hot 특징을 센서로 세지 않는다.
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("반복 센서 선택에는 이름이 중복되지 않은 DataFrame이 필요합니다.")
        
        if isinstance(self.n_repeats, bool) or not isinstance(self.n_repeats, int) or self.n_repeats < 2:
            raise ValueError("반복 센서 선택 횟수는 2 이상의 정수여야 합니다.")
        
        count = self.max_features(X) if callable(self.max_features) else self.max_features

        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= X.shape[1]:
            raise ValueError("반복 선택 센서 수는 입력 컬럼 수 이하의 양의 정수여야 합니다.")
        
        labels = np.asarray(y)

        if labels.ndim != 1 or len(labels) != len(X) or pd.isna(labels).any():
            raise ValueError("반복 선택의 label은 결측 없이 입력 행 수와 일치해야 합니다.")
        
        classes = pd.unique(labels)

        if len(classes) != 2:
            raise ValueError("반복 선택의 학습 데이터에는 두 클래스가 필요합니다.")
        
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        names_to_index = {name: index for index, name in enumerate(X.columns)}
        hits = np.zeros(X.shape[1], dtype=int)
        ranks = np.zeros(X.shape[1], dtype=float)
        importance_sum = np.zeros(X.shape[1], dtype=float)
        self.bootstrap_log_ = []

        for repeat in range(self.n_repeats):
            # 각 클래스에서 원래 클래스 행 수만큼 뽑아 소수 클래스 소멸을 막는다.
            indices = np.concatenate([
                rng.choice(np.flatnonzero(labels == value), size=int((labels == value).sum()), replace=True)
                for value in classes
            ])
            rng.shuffle(indices)
            preprocessing = clone(self.preprocessing)
            sample = preprocessing.fit_transform(X.iloc[indices], labels[indices])
            names = preprocessing.get_feature_names_out()
        
            if len(names) < count:
                raise ValueError("재표집 품질 필터 후 센서 수가 Top-K보다 적습니다.")
        
            if any(name not in names_to_index for name in names):
                raise ValueError("반복 선택은 변환 특징이 아닌 수치형 원본 센서만 지원합니다.")
            
            model = clone(self.estimator).set_params(random_state=int(rng.integers(0, 2**31)))
            model.fit(sample, labels[indices])
            order = np.argsort(-model.feature_importances_, kind="stable")

            # 재표집에서 제거된 센서는 최하위보다 낮은 순위, 중요도·선택 빈도는 0이다.
            current_ranks = np.full(X.shape[1], X.shape[1] + 1, dtype=float)
            positions = np.asarray([names_to_index[name] for name in names])
            current_ranks[positions[order]] = np.arange(1, len(names) + 1)
            ranks += current_ranks
            hits[positions[order[:count]]] += 1
            importance_sum[positions] += model.feature_importances_
            quality = preprocessing.named_steps["quality_filter"]
            
            self.bootstrap_log_.append({
                "repeat": repeat + 1, "sample_rows": len(indices),
                "unique_source_rows": len(np.unique(indices)),
                "retained_sensor_count": len(names),
                "high_missing_removed_count": len(quality.high_missing_features_),
                "constant_removed_count": len(quality.constant_features_),
            })

            if hasattr(model, "effective_scale_pos_weight_"):
                # gain 선택용 모델의 실제 bootstrap 학습 가중치를 최종 분류기와 분리한다.
                positive_count = int((labels[indices] == model.positive_label).sum())
                self.bootstrap_log_[-1].update(
                    class_weight_mode=model.class_weight_mode,
                    effective_scale_pos_weight=model.effective_scale_pos_weight_,
                    train_positive=positive_count,
                    train_negative=len(indices) - positive_count)
                
        self.selection_frequency_ = hits / self.n_repeats
        self.mean_rank_ = ranks / self.n_repeats
        self.mean_importance_ = importance_sum / self.n_repeats

        # lexsort의 마지막 키가 우선 기준이다. 동률은 입력 열 순서로 재현한다.
        self.selection_order_ = np.lexsort((np.arange(X.shape[1]), self.mean_rank_, -self.selection_frequency_))
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.selection_order_[:count]] = True

        return self

    def get_support(self, indices=False):
        # Pipeline의 특징 수 기록과 기존 선택 CSV에서 사용하는 공통 API다.
        check_is_fitted(self, "support_")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def selection_records(self, metadata):
        # 선택 순위는 빈도·평균 순위 기준이며 중요도 평균과 구분해 CSV에 남긴다.
        check_is_fitted(self, "support_")
        chosen = [index for index in self.selection_order_ if self.support_[index]]
        
        return [{**metadata, "feature": str(self.feature_names_in_[index]),
                 "selection_rank": rank, "selection_method": self.selection_method,
                 "selection_importance": float(self.mean_importance_[index]),
                 "bootstrap_selection_frequency": float(self.selection_frequency_[index]),
                 "bootstrap_mean_rank": float(self.mean_rank_[index]),
                 "bootstrap_repeats": self.n_repeats, "feature_space": "raw_sensor"}
                for rank, index in enumerate(chosen, 1)]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_].copy()

    def transform(self, X):
        # 새로운 입력으로 센서를 다시 고르거나 학습 통계를 계산하지 않는다.
        check_is_fitted(self, "support_")
        names = self.get_feature_names_out()
        
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("반복 센서 변환에는 이름이 중복되지 않은 DataFrame이 필요합니다.")
        
        if not set(names).issubset(X.columns):
            raise ValueError("반복 선택의 필수 센서 컬럼이 입력에 없습니다.")
        
        return X.loc[:, list(names)].copy()


class StableGainSelector(StableRFSelector):
    """계층 bootstrap·빈도·평균 순위 코어를 XGBoost gain 선택에 재사용한다.

    각 반복의 gain Top-K 등장 빈도를 우선하고, 동률은 평균 순위와 원본 열
    순서로 해소한다. 후보 센서를 고르는 fit에는 현재 학습 구간만 전달한다.
    """

    selection_method = "xgboost_gain_stability"

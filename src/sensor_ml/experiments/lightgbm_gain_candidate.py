# ==========================================
# 팀원 제안: LightGBM gain 선택 + 가중 XGBoost 후보
# - 선택기 가중치는 현재 fit에 전달된 label로만 계산
# - 품질 필터·대치·Top-K 선택을 하나의 Pipeline 안에서 수행
# - 전체 Train의 분석 목록을 OOF에 재사용하지 않음
# ==========================================
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from src.modeling_models import build_classifier
from src.modeling_preprocessing import preprocessing_steps, checked_top_k

EXPERIMENT = "xgboost_lightgbm_gain_top20"


class FoldGainImportance(BaseEstimator):
    """현재 학습 fold의 정상/불량 비율을 사용하는 gain 중요도 추정기."""

    def __init__(self, positive_label=1, negative_label=-1, n_estimators=200,
                 random_state=42, n_jobs=1):
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """라벨을 0/1로 명시 변환하고 현재 학습 데이터에만 LightGBM을 fit한다."""
        from lightgbm import LGBMClassifier
        labels = pd.Series(y)

        if labels.isna().any() or set(labels.unique()) != {self.negative_label, self.positive_label}:
            raise ValueError("gain 선택 학습에는 설정된 정상·불량 label이 모두 필요합니다.")
        
        # -1/1 또는 문자열 label에서도 합계가 아니라 클래스 건수를 사용한다.
        positive = labels.eq(self.positive_label).to_numpy()
        self.effective_scale_pos_weight_ = float((~positive).sum() / positive.sum())

        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs,
            importance_type="gain", scale_pos_weight=self.effective_scale_pos_weight_, verbosity=-1,
        ).fit(X, positive.astype(int))

        self.feature_importances_ = self.model_.feature_importances_
        self.n_features_in_ = X.shape[1]

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        return self


def build_gain_candidate(config, *, n_jobs=1, top_k=20):
    """fold 내부 LightGBM 선택기와 기존 가중 XGBoost 분류기를 결합한다."""
    importance = FoldGainImportance(config.dataset.positive_label, config.dataset.negative_label,
                                    random_state=config.experiment.cv.random_state, n_jobs=n_jobs)
    
    model = build_classifier(config, "xgboost", n_jobs=n_jobs)

    # adapter가 fit할 때마다 해당 fold의 클래스 비율을 계산한다.
    model.set_params(class_weight_mode="ratio")

    steps = preprocessing_steps(config.dataset, pandas_output=True)

    steps.append(("selector", SelectFromModel(
        importance, threshold=-np.inf, max_features=partial(checked_top_k, count=top_k))))
    
    return Pipeline(steps + [("model", model)])


def recall_scenarios(table, targets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Recall 조건별 최고 정밀도 후보를 비교하되 최종 문턱으로 확정하지 않는다."""
    rows = []

    for target in targets:
        # 확률이 작다는 이유로 유효한 문턱을 제외하지 않는다.
        candidates = table.loc[table.recall >= target]

        if candidates.empty:
            rows.append({"target_recall": target, "status": "infeasible"})
        else:
            best = candidates.sort_values(["precision", "f1", "threshold"],
                                           ascending=[False, False, False]).iloc[0]
            rows.append({**best.to_dict(), "target_recall": target, "status": "candidate"})
            
    return pd.DataFrame(rows)

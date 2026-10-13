# ==========================================
# 전체·Top-50·Top-20의 시간순 성능과 센서 선택 안정성 비교
# - 실험군 안에서 최종 분류기는 동일하게 유지하고 선택 중요도 모델만 비교
# - LightGBM 중요도와 원본 과제의 RF 중요도 선택을 모두 실험
# - 명시한 XGBoost 전체·Top-K도 공통 모델 정의로 같은 시간순 코어에서 평가
# - 선택기는 각 과거 학습 fold의 Pipeline 안에서만 fit
# - 최종 센서가 아닌 후보 목록을 전체 Time Train에서 별도로 준비
# - 외부 Validation·최종 Test를 읽지 않음
#
# 처리 순서:
# - 1. 전체·두 종류의 중요도 Top-50/20 Pipeline을 생성
# - 2. 기존 시간 검증 코어로 다음 구간 성능과 내부 OOF 정책 평가
# - 3. fit 직후 선택 센서를 기록하고 외부·내부 fit 빈도를 따로 계산
# - 4. 전체 Time Train에서 연구용 후보 센서 목록만 준비
# - 5. 성능·센서·설정 기록을 저장하며 최종 후보는 자동 확정하지 않음
# ==========================================

import argparse
import json
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from src.stable_rf_selector import StableGainSelector

try:
    from src.feature_reduction import assess_reduction
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_models import build_classifier, build_pipeline, fit_pipeline, XGBoostClassifierAdapter, configure_rf_selectors, configure_topk_xgb
    from src.modeling_preprocessing import checked_top_k
    from src.sensor_ml.experiments.temporal_validation import compare_temporal
    from src.split_contract import PROTOCOL_ID
except ModuleNotFoundError:
    from feature_reduction import assess_reduction
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from dataset_schema import split_frame_to_xy
    from modeling_models import build_classifier, build_pipeline, fit_pipeline, XGBoostClassifierAdapter, configure_rf_selectors, configure_topk_xgb
    from modeling_preprocessing import checked_top_k
    from src.sensor_ml.experiments.temporal_validation import compare_temporal
    from split_contract import PROTOCOL_ID


# ==========================================
# 같은 분류 모델에서 센서 수와 선택 방법만 비교하는 Pipeline 생성
# - RF selector + LightGBM 분류기로 선택 방법의 차이를 분리
# - 불균형 가중치와 scaling 등 다른 조건을 동시에 바꾸지 않음
# ==========================================
def _configure_gain_stability(pipelines, dataset, repeats):
    """gain 선택기를 원본 센서의 반복 선택 경로로 교체하고 다른 경로는 보존한다."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0 or repeats == 1:
        raise ValueError("gain 반복 선택 횟수는 0 또는 2 이상의 정수여야 합니다.")
    if repeats == 0:
        return
    if dataset.categorical_feature_columns:
        raise ValueError("반복 gain 선택은 수치형 원본 센서 Profile만 지원합니다.")
    found = False
    for pipeline in pipelines.values():
        selector = pipeline.named_steps.get("selector")
        if not isinstance(getattr(selector, "estimator", None), XGBoostClassifierAdapter):
            continue
        # 재표집마다 독립적인 품질 필터·대치기를 fit한다. 평가 데이터는 받지 않는다.
        preprocessing = Pipeline([(name, clone(step)) for name, step in pipeline.steps
                                  if name not in {"selector", "model"}])
        stable = StableGainSelector(clone(selector.estimator), preprocessing, selector.max_features,
                                    n_repeats=repeats, random_state=selector.estimator.random_state)
        # 선택은 대치 전에 원본 센서 이름으로 수행하고, 최종 대치는 학습 Fold에서 별도 fit한다.
        pipeline.steps = [("quality_filter", pipeline.named_steps["quality_filter"]),
                          ("selector", stable)] + [(name, step) for name, step in pipeline.steps
                                                  if name not in {"quality_filter", "selector"}]
        found = True
    if not found:
        raise ValueError("반복 선택을 적용할 XGBoost gain 선택기가 없습니다.")


def build_feature_pipelines(config, *, n_jobs=1, n_estimators=300, experiment_names=None,
                            rf_min_samples_leaf=1, rf_max_depth=None, rf_stability_repeats=0,
                            topk_xgb_max_depth=None, topk_xgb_reg_lambda=None,
                            xgb_weight_mode="none", xgb_selector_weight_mode="none",
                            xgb_stability_repeats=0):
    
    # V2 비교에서도 기존 기본값은 유지하고 요청한 가중치만 적용한다.
    if xgb_weight_mode not in {"none", "sqrt_ratio", "ratio"}:
        raise ValueError("XGBoost 가중치 방식은 none·sqrt_ratio·ratio 중 하나여야 합니다.")
    
    if xgb_weight_mode != "none" and not experiment_names:
        raise ValueError("XGBoost 가중치를 적용할 특징 실험을 명시하세요.")
    
    if xgb_selector_weight_mode not in {"none", "sqrt_ratio", "ratio"}:
        raise ValueError("XGBoost 선택기 가중치 방식은 none·sqrt_ratio·ratio 중 하나여야 합니다.")
    
    if xgb_selector_weight_mode != "none" and not experiment_names:
        raise ValueError("XGBoost 선택기 가중치를 적용할 특징 실험을 명시하세요.")
    
    if rf_stability_repeats and config.dataset.categorical_feature_columns:
        raise ValueError("S3 반복 RF 선택은 수치형 원본 센서 Profile만 지원합니다.")
    
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("트리 수는 양의 정수여야 합니다.")
    
    # 모든 최종 분류기의 설정이 같아야 센서 축소에 따른 차이를 해석할 수 있다.
    classifier = build_classifier(config, "lightgbm", n_jobs=n_jobs, n_estimators=n_estimators)
    steps = preprocessing_steps(config.dataset, pandas_output=True)
    pipelines = {"lightgbm_all": Pipeline(steps + [("model", clone(classifier))])}

    for method in ("lightgbm", "rf_importance"):
        for count in (50, 20):
            # 매 Pipeline마다 독립된 전처리기·선택기·분류기를 사용한다.
            importance = clone(classifier) if method == "lightgbm" else build_classifier(config, "random_forest", n_jobs=n_jobs, n_estimators=n_estimators)
            selector = SelectFromModel(importance, threshold=-np.inf,
                max_features=partial(checked_top_k, count=count))
            
            # -inf는 중요도 자체의 최소값 제한을 없애고 max_features로 정확히 K개 선택한다.
            # checked_top_k는 대치·One-Hot 후 해당 fold의 실제 특징 수를 검사한다.
            name = f"lightgbm_top_{count}" if method == "lightgbm" else f"lightgbm_rf_top_{count}"
            pipelines[name] = Pipeline(preprocessing_steps(config.dataset, pandas_output=True) +
                                        [("selector", selector), ("model", clone(classifier))])
    if experiment_names is None:
        # 옵션을 생략하면 기존 LightGBM 전체·Top-50/20 실험을 그대로 유지한다.
        _configure_gain_stability(pipelines, config.dataset, xgb_stability_repeats)
        configure_topk_xgb(pipelines, max_depth=topk_xgb_max_depth, reg_lambda=topk_xgb_reg_lambda)
        return configure_rf_selectors(pipelines, min_samples_leaf=rf_min_samples_leaf, max_depth=rf_max_depth,
                                      stability_repeats=rf_stability_repeats)
    
    if not experiment_names or len(set(experiment_names)) != len(experiment_names):
        raise ValueError("시간순 특징 실험 이름은 중복 없이 하나 이상 지정해야 합니다.")
    
    xgboost_names = {"xgboost_all"} | {
        f"{prefix}{count}" for prefix in ("xgboost_top_", "xgboost_rf_top_")
        for count in config.top_k_feature_counts
    }

    unknown = set(experiment_names) - set(pipelines) - xgboost_names

    if unknown:
        raise ValueError(f"지원하지 않는 시간순 특징 실험입니다: {sorted(unknown)}")
    
    chosen = {}

    for name in experiment_names:
        pipeline = pipelines[name] if name in pipelines else build_pipeline(config, name, n_jobs=n_jobs)

        # 선택용 모델과 최종 분류기에 같은 트리 수를 적용해 CLI 설정을 일관되게 반영한다.
        params = {key: n_estimators for key in pipeline.get_params(deep=True)
                  if key.endswith("__n_estimators")}
        
        pipeline.set_params(**params)
        chosen[name] = pipeline

    xgb_models = [pipeline.named_steps["model"] for pipeline in chosen.values()
                  if isinstance(pipeline.named_steps["model"], XGBoostClassifierAdapter)]
    
    if xgb_weight_mode != "none" and not xgb_models:
        raise ValueError("가중치를 적용할 최종 XGBoost 분류기가 없습니다.")
    
    for model in xgb_models:
        # 선택용 RF는 변경하지 않는다. 가중치는 각 실제 fit의 label에서 계산된다.
        model.set_params(class_weight_mode=xgb_weight_mode)
    xgb_selectors = [pipeline.named_steps["selector"].estimator for pipeline in chosen.values()
                     if isinstance(getattr(pipeline.named_steps.get("selector"), "estimator", None),
                                   XGBoostClassifierAdapter)]
    
    if xgb_selector_weight_mode != "none" and not xgb_selectors:
        raise ValueError("가중치를 적용할 XGBoost gain 선택기가 없습니다.")
    
    for selector in xgb_selectors:
        # 최종 분류기·RF 선택기는 건드리지 않고 gain 계산용 학습에만 적용한다.
        selector.set_params(class_weight_mode=xgb_selector_weight_mode)

    # 전체 센서 기준과 RF는 유지하고 축소 경로의 최종 분류기에만 M1~M3을 반영한다.
    configure_topk_xgb(chosen, max_depth=topk_xgb_max_depth, reg_lambda=topk_xgb_reg_lambda)
    _configure_gain_stability(chosen, config.dataset, xgb_stability_repeats)
    return configure_rf_selectors(chosen, min_samples_leaf=rf_min_samples_leaf, max_depth=rf_max_depth,
                                  stability_repeats=rf_stability_repeats)


# ==========================================
# 학습된 Pipeline에서 실제 선택 특징명·중요도·순위 추출
# - One-Hot을 쓴 데이터에서는 원본 센서가 아니라 변환 특징일 수 있음
# - 최종 분류기의 중요도가 아니라 선택에 사용한 모델의 중요도 기록
# - 선택기의 get_support 순서는 모델 입력 순서이며 중요도 순위와 다름
# ==========================================
def selection_rows(pipeline, metadata):

    quality = pipeline.named_steps["quality_filter"]

    if hasattr(pipeline.named_steps.get("selector"), "selection_records"):
        return pipeline.named_steps["selector"].selection_records(metadata)
    
    if "selector" in pipeline.named_steps:
        # 선택기 전까지의 특징명에 get_support 마스크를 적용해 실제 입력 특징을 찾는다.
        names = pipeline[:-2].get_feature_names_out()
        selector = pipeline.named_steps["selector"]
        mask = selector.get_support()
        selected = np.asarray(names)[mask]
        importance = selector.estimator_.feature_importances_[mask]

        if isinstance(selector.estimator_, RandomForestClassifier):
            method = "rf_importance"
        elif isinstance(selector.estimator_, XGBoostClassifierAdapter):
            method = f"xgboost_{selector.estimator_.importance_type}"
        else:
            method = "lightgbm_importance"
    else:
        # 전체 모델에는 중요도 순위 선택이 없으므로 순위는 입력 순서이고 중요도는 결측이다.
        selected = pipeline[:-1].get_feature_names_out()
        importance = np.full(len(selected), np.nan)
        method = "all_after_quality_filter"

    # 중요도가 같으면 원래 특징 순서를 유지해 같은 결과를 재현한다.
    order = np.argsort(-importance, kind="stable") if method != "all_after_quality_filter" else np.arange(len(selected))

    return [{**metadata, "feature": str(selected[index]), "selection_method": method,
             "selection_rank": rank, "selection_importance": float(importance[index]),
             "feature_space": "raw_sensor" if str(selected[index]) in quality.retained_features_ else "transformed_feature"}
            for rank, index in enumerate(order, 1)]


# ==========================================
# 학습별 선택 목록에서 특징별 선택 빈도 계산
# - 빈도 = 특징을 선택한 fit 수 / 해당 모델·역할의 전체 fit 수
# - outer_fit의 분모는 3, inner_oof의 분모는 6
# - 학습 구간이 겹치므로 빈도는 안정성 참고값이며 독립 검정 결과가 아님
# ==========================================
def feature_frequency(selected):
    # 외부 fit·내부 OOF fit은 분모를 구분하며 겹치는 학습을 독립 실험으로 보지 않는다.
    counts = selected.groupby(["model_name", "fit_role"])["fit_id"].nunique()
    frequency = selected.groupby(["model_name", "fit_role", "feature", "feature_space"], as_index=False).agg(
        selected_fit_count=("fit_id", "nunique"), mean_selection_rank=("selection_rank", "mean"))
    frequency["total_fit_count"] = [counts.loc[(name, role)] for name, role in zip(frequency.model_name, frequency.fit_role)]
    frequency["selection_frequency"] = frequency.selected_fit_count / frequency.total_fit_count

    return frequency.sort_values(["model_name", "fit_role", "selection_frequency", "mean_selection_rank", "feature"],
                                 ascending=[True, True, False, True, True]).reset_index(drop=True)


# ==========================================
# 같은 시간순 코어로 평가하고 각 fit의 선택 센서 기록
# - 기록용 hook은 fit 결과를 조회할 뿐 예측 계산을 바꾸지 않음
# - 외부 fit 3회·내부 OOF fit 6회의 센서 목록과 빈도를 별도로 저장
# ==========================================
def compare_feature_time(frame, config, *, n_jobs=1, n_estimators=300, experiment_names=None,
                         rf_min_samples_leaf=1, rf_max_depth=None, rf_stability_repeats=0,
                         topk_xgb_max_depth=None, topk_xgb_reg_lambda=None,
                         xgb_weight_mode="none", xgb_selector_weight_mode="none",
                         xgb_stability_repeats=0):
    
    pipelines = build_feature_pipelines(config, n_jobs=n_jobs, n_estimators=n_estimators,
                                       experiment_names=experiment_names,
                                       rf_min_samples_leaf=rf_min_samples_leaf, rf_max_depth=rf_max_depth,
                                       rf_stability_repeats=rf_stability_repeats,
                                       topk_xgb_max_depth=topk_xgb_max_depth, topk_xgb_reg_lambda=topk_xgb_reg_lambda,
                                       xgb_weight_mode=xgb_weight_mode,
                                       xgb_selector_weight_mode=xgb_selector_weight_mode,
                                       xgb_stability_repeats=xgb_stability_repeats)
    
    selected, bootstrap_records, weights = [], [], []

    # 시간 검증 코어에 기록용 hook을 전달해 같은 학습을 중복 실행하지 않는다.
    def record_selection(model, train, evaluation, outer, name, mode, inner):

        # outer:mode:inner 조합으로 fit을 구분해 같은 특징이 중복 집계되지 않게 한다.
        # 평가 데이터는 hook에서 읽지 않아 선택 목록에 미래 정보가 섞이지 않는다.
        selected.extend(selection_rows(model, {"outer_fold": outer, "model_name": name,
            "fit_role": "outer_fit" if inner is None else "inner_oof", "inner_fold": inner,
            "fit_id": f"{outer}:{mode}:{inner}", "train_samples": len(train)}))
        selector = model.named_steps.get("selector")
        classifier = model.named_steps["model"]

        if isinstance(classifier, XGBoostClassifierAdapter):
            # 추정 설정이 아니라 학습된 모델의 실제 가중치를 과거 학습 건수와 함께 기록한다.
            labels = train[config.dataset.label_column]
            weights.append({"outer_fold": outer, "model_name": name, "mode": mode,
                            "inner_fold": inner, "train_samples": len(train),
                            "train_positive": int(labels.eq(config.dataset.positive_label).sum()),
                            "train_negative": int(labels.eq(config.dataset.negative_label).sum()),
                            "class_weight_mode": classifier.class_weight_mode,
                            "effective_scale_pos_weight": classifier.effective_scale_pos_weight_})
            # SelectFromModel이 실제 학습한 선택기를 조회해 최종 분류기와 따로 기록한다.
            fitted_selector = getattr(selector, "estimator_", None)

            if isinstance(fitted_selector, XGBoostClassifierAdapter):
                weights[-1].update(
                    selector_class_weight_mode=fitted_selector.class_weight_mode,
                    selector_effective_scale_pos_weight=fitted_selector.effective_scale_pos_weight_)
                
        bootstrap_records.extend({"outer_fold": outer, "model_name": name, "inner_fold": inner,
                                  "fit_role": "outer_fit" if inner is None else "inner_oof", **record}
                                 for record in getattr(selector, "bootstrap_log_", []))
        
    results = compare_temporal(frame, config, model_names=tuple(pipelines), pipelines=pipelines,
                              oof_modes=("temporal_oof",), n_jobs=n_jobs, fit_observer=record_selection)
    
    results["selected_features"] = pd.DataFrame(selected)

    if weights:
        results["fit_weights"] = pd.DataFrame(weights)

    results["feature_frequency"] = feature_frequency(results["selected_features"])

    # 시간 구간 AP는 threshold와 무관하므로 기본 문턱 결과에서 구간당 한 번만 읽는다.
    evaluated = results["fold_results"]
    mode = evaluated["mode"].iloc[0]

    reduction_folds = evaluated[evaluated["mode"].eq(mode)].rename(columns={
        "model_name": "experiment", "outer_fold": "fold", "average_precision": "ap",
    }).copy()

    outer = results["selected_features"].query("fit_role == 'outer_fit'")

    sensor_info = outer.groupby(["model_name", "outer_fold"]).agg(
        sensor_count=("feature", "nunique"),
        is_raw_sensor=("feature_space", lambda values: values.eq("raw_sensor").all()),
    ).reset_index().rename(columns={"model_name": "experiment", "outer_fold": "fold"})

    reduction_folds = reduction_folds.merge(sensor_info, on=["experiment", "fold"], validate="one_to_one")

    results["reduction_assessment"] = assess_reduction(reduction_folds, config.reduction_policy)

    # 전체 Train 후보 fit은 평가 fold가 아니다. 이 목록으로 과거 점수를 재계산하지 않는다.
    features, labels, _ = split_frame_to_xy(frame, config.dataset)
    candidates, quality = [], []

    for name, pipeline in pipelines.items():
        # clone으로 CV의 학습 상태를 버린 뒤 전체 Time Train에서 새로 fit한다.
        # 최종 모델·학습 범위를 아직 합의하지 않았으므로 is_final=False로 저장한다.
        model = fit_pipeline(pipeline, features, labels, config, name, n_jobs=n_jobs)
        selector = model.named_steps.get("selector")
        bootstrap_records.extend({"model_name": name, "fit_role": "candidate_full_train", **record}
                                 for record in getattr(selector, "bootstrap_log_", []))
        candidates.extend(selection_rows(model, {"model_name": name, "fit_role": "candidate_full_train",
                           "train_samples": len(frame), "is_final": False}))
        quality.append({"model_name": name, **quality_filter_record(model)})

    results["candidate_features"] = pd.DataFrame(candidates)
    results["candidate_quality_filter"] = pd.DataFrame(quality)

    if bootstrap_records:
        results["bootstrap_records"] = pd.DataFrame(bootstrap_records)
    return results


# ==========================================
# 명시한 Train에서 실험 실행 후 새 폴더에 로그 저장
# - CSV는 노트북에서 읽고 feature_run.json은 재현 조건을 보관
# - 후보 센서 목록을 최종 모델·threshold로 자동 연결하지 않음
# ==========================================
def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="전체·Top-50·Top-20 시간순 성능 및 센서 선택 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--xgb-weight-mode", choices=("none", "sqrt_ratio", "ratio"), default="none",
                        help="최종 XGBoost 전체·축소 경로의 불량 가중치. ratio는 각 fit의 정상/불량 비율입니다.")
    parser.add_argument("--xgb-selector-weight-mode", choices=("none", "sqrt_ratio", "ratio"), default="none",
                        help="gain 선택용 XGBoost만 변경합니다. 최종 분류기와 RF 선택기는 유지합니다.")
    parser.add_argument("--xgb-stability-repeats", type=int, default=0,
                        help="gain 선택기의 학습 내부 계층 bootstrap 횟수. 0은 단일 선택, 2 이상은 반복 선택입니다.")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1,
                        help="RF 선택기 leaf의 최소 샘플 수. S0=1, S1·S2=2입니다.")
    parser.add_argument("--rf-max-depth", type=int, default=None,
                        help="RF 선택기의 최대 깊이. 생략하면 제한 없음, S2=8입니다.")
    parser.add_argument("--rf-stability-repeats", type=int, default=0,
                        help="S3의 학습 내부 반복 선택 횟수. 0은 기존 단일 선택, 2 이상은 반복 선택입니다.")
    parser.add_argument("--topk-xgb-max-depth", type=int, default=None,
                        help="RF·gain Top-K 뒤 최종 XGBoost의 깊이만 변경합니다. 전체 센서 모델과 선택기는 유지합니다.")
    parser.add_argument("--topk-xgb-reg-lambda", type=float, default=None,
                        help="RF·gain Top-K 뒤 최종 XGBoost의 L2 정규화만 변경합니다. 전체 센서 모델과 선택기는 유지합니다.")
    parser.add_argument("--experiments", nargs="+",
                        help="실행할 특징 실험 이름. 생략하면 기존 LightGBM 전체·Top-50/20을 비교합니다.")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    return args


def save_results(args, config, train, results):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        # 센서 목록의 JSON 변환은 저장용 복사본에만 적용한다.

        if name in {"quality_filter", "candidate_quality_filter"}:
            table = table.copy()

            for column in ("high_missing_features", "constant_features"):
                table[column] = table[column].map(quality_filter_json)
        table.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        
    record = {"config": asdict(config), "train_path": str(args.train.resolve()), "n_estimators": args.n_estimators,
              "xgb_weight_mode": args.xgb_weight_mode,
              "xgb_selector_weight_mode": args.xgb_selector_weight_mode,
              "xgb_stability_repeats": args.xgb_stability_repeats,
              "n_jobs": args.n_jobs, "outer_splits": 3, "inner_temporal_splits": 2,
              "training_protocol_id": train[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train else None,
              "sklearn": sklearn.__version__, "candidate_lists_are_final": False,
              "models": results["summary"].model_name.unique().tolist(),
              "topk_xgb_parameters": {"max_depth_override": args.topk_xgb_max_depth,
                                    "reg_lambda_override": args.topk_xgb_reg_lambda,
                                      "targets": [name for name in results["summary"].model_name.unique()
                                                  if name.startswith(("xgboost_rf_top_", "xgboost_top_"))]
                                      if args.topk_xgb_max_depth is not None or args.topk_xgb_reg_lambda is not None else []},
              "rf_selector_parameters": {"min_samples_leaf": args.rf_min_samples_leaf,
                                         "max_depth": args.rf_max_depth,
                                         "stability_repeats": args.rf_stability_repeats,
                                         "resampling": "stratified_bootstrap" if args.rf_stability_repeats else None},
              "versions": {name: version(name) for name in ("scikit-learn", "lightgbm")}}
    
    # 기존 LightGBM 실험만 실행할 때는 선택하지 않은 XGBoost 설치를 강제하지 않는다.
    if any(name.startswith("xgboost") for name in record["models"]):
        record["versions"]["xgboost"] = version("xgboost")
        
    (args.output_dir / "feature_run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    train = pd.read_csv(args.train)
    results = compare_feature_time(train, config, n_jobs=args.n_jobs, n_estimators=args.n_estimators,
                                   experiment_names=args.experiments,
                                   rf_min_samples_leaf=args.rf_min_samples_leaf, rf_max_depth=args.rf_max_depth,
                                   rf_stability_repeats=args.rf_stability_repeats,
                                   topk_xgb_max_depth=args.topk_xgb_max_depth, topk_xgb_reg_lambda=args.topk_xgb_reg_lambda,
                                   xgb_weight_mode=args.xgb_weight_mode,
                                   xgb_selector_weight_mode=args.xgb_selector_weight_mode,
                                   xgb_stability_repeats=args.xgb_stability_repeats)
    save_results(args, config, train, results)
    print(results["summary"].to_string(index=False))
    if not results["reduction_assessment"].empty:
        print("센서 축소 AP 판정 (최종 후보는 자동 확정하지 않음)")
        print(results["reduction_assessment"].to_string(index=False))
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

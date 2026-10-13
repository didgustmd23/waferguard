# ==========================================
# CV fold 내부 feature 선택 비교
# - Train만 사용하며 외부 Validation/Test는 읽지 않음
# - PCA·L1·Importance 선택은 공통 모델 모듈의 Pipeline 안에서 fit
# - 실험별 성능 평균·편차와 실제 특징 수·센서 제거 기록을 저장
# - 지정한 실험만 실행하고 fold별 지표·선택 센서를 별도 기록 가능
# ==========================================
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from src.feature_reduction import assess_reduction
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.modeling_models import build_experiments, build_pipeline, fit_pipeline, positive_scores, configure_rf_selectors, configure_topk_xgb
    from src.modeling_preprocessing import (
        fitted_feature_count, quality_filter_record, quality_filter_json,
        checked_top_k as _checked_top_k,
    )
    from src.split_contract import PROTOCOL_ID, validate_train_role, training_folds
except ModuleNotFoundError:
    from feature_reduction import assess_reduction
    from dataset_schema import split_frame_to_xy
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from modeling_models import build_experiments, build_pipeline, fit_pipeline, positive_scores, configure_rf_selectors, configure_topk_xgb
    from modeling_preprocessing import (
        fitted_feature_count, quality_filter_record, quality_filter_json,
        checked_top_k as _checked_top_k,
    )
    from split_contract import PROTOCOL_ID, validate_train_role, training_folds


# build_experiments와 _checked_top_k는 기존 import 경로와의 호환을 위해 노출한다.
# 신규 호출자는 modeling_models / modeling_preprocessing에서 직접 가져온다.


# ==========================================
# 현재 학습 fold에서 선택한 특징의 근거를 기록
# - 선택기 입력의 열 이름과 support mask를 같은 위치 기준으로 대응
# - 트리 중요도 또는 L1 계수 절댓값을 기록하며 PCA는 제외
# - 센서명·순위는 연구용 기록이며 최종 고정 센서 집합이 아님
# ==========================================
def _selected_rows(model, name, fold):
    """학습된 중요도 선택기의 센서명·순위·점수를 기록한다."""
    selector = model.named_steps.get("selector")
    # PCA는 원본 센서 선택이 아니라 새로운 축 생성이므로 이 목록에 넣지 않는다.
    if selector is None or not hasattr(selector, "get_support"):
        return []
    
    if hasattr(selector, "selection_records"):
        return selector.selection_records({"experiment": name, "cv_fold": fold})
    
    names = model[:-2].get_feature_names_out()
    mask = selector.get_support()
    estimator = selector.estimator_

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_[mask]
    else:
        importance = np.abs(estimator.coef_).sum(axis=0)[mask]

    selected = np.asarray(names)[mask]

    # 같은 중요도는 입력 특징 순서를 유지해 기록 순위가 재현되도록 한다.
    order = np.argsort(-importance, kind="stable")
    retained = set(model.named_steps["quality_filter"].retained_features_)

    return [
        {"experiment": name, "cv_fold": fold, "feature": str(selected[index]),
         "selection_rank": rank, "selection_importance": float(importance[index]),
         "feature_space": "raw_sensor" if selected[index] in retained else "transformed_feature"}
        for rank, index in enumerate(order, 1)
    ]


def compare_features(frame, config, n_jobs: int = 1, *, experiment_names=None,
                     rf_min_samples_leaf=1, rf_max_depth=None, rf_stability_repeats=0,
                     topk_xgb_max_depth=None, topk_xgb_reg_lambda=None):
    """같은 Train CV fold에서 특징 선택 방식별 성능과 특징 수를 비교한다."""
    validate_train_role(frame)

    # Profile 검증을 통과한 특징과 label만 학습에 사용한다.
    features, labels, _ = split_frame_to_xy(frame, config.dataset)

    if labels.nunique() < 2:
        raise ValueError("특징 선택 비교를 위해 Train split에는 정상과 Fail label이 모두 있어야 합니다.")
    
    smallest_class_count = int(labels.value_counts().min())

    if smallest_class_count < config.experiment.cv.n_splits:
        raise ValueError(
            "가장 적은 class의 샘플 수가 CV fold 수보다 작습니다: "
            f"최소 class 샘플={smallest_class_count}, fold={config.experiment.cv.n_splits}"
        )
    
    # 범주형 입력은 One-Hot 이후 특징 수가 달라지므로 실제 fold 안에서 검사한다.
    too_large_top_k = (
        [count for count in config.top_k_feature_counts if count > features.shape[1]]
        if experiment_names is None and not config.dataset.categorical_feature_columns else []
    )

    if too_large_top_k:
        raise ValueError(
            "Top-K feature 수가 입력 feature 수보다 큽니다: "
            f"Top-K={too_large_top_k}, feature 수={features.shape[1]}"
        )

    if experiment_names is not None and (
        not experiment_names or len(set(experiment_names)) != len(experiment_names)
    ):
        raise ValueError("특징 실험 이름은 중복 없이 하나 이상 지정해야 합니다.")
    
    experiments = (
        build_experiments(config, n_jobs) if experiment_names is None else
        {name: build_pipeline(config, name, n_jobs=n_jobs) for name in experiment_names}
    )

    if rf_stability_repeats and config.dataset.categorical_feature_columns:
        raise ValueError("S3 반복 RF 선택은 수치형 원본 센서 Profile만 지원합니다.")
    
    configure_rf_selectors(experiments, min_samples_leaf=rf_min_samples_leaf, max_depth=rf_max_depth,
                           stability_repeats=rf_stability_repeats)
    
    # M1~M3은 RF 선택을 유지하고 축소 경로의 최종 분류기만 변경한다.
    configure_topk_xgb(experiments, max_depth=topk_xgb_max_depth, reg_lambda=topk_xgb_reg_lambda)

    # 기본 실험 목록은 그대로 두고 명시한 실험만 실행할 수 있도록 한다.
    rows, fold_rows, selected_rows, bootstrap_rows = [], [], [], []

    for name, pipeline in experiments.items():
        fold_metrics, fit_times, feature_counts, quality_records = [], [], [], []

        for fold, (fit_indices, validation_indices) in enumerate(
            training_folds(features, labels, frame, config), 1,
        ):
            # 매 fold마다 독립된 Pipeline을 현재 학습 데이터로만 fit한다.
            started_at = perf_counter()

            model = fit_pipeline(
                pipeline, features.iloc[fit_indices], labels.iloc[fit_indices],
                config, name, n_jobs=n_jobs,
            )

            fit_times.append(perf_counter() - started_at)
            feature_counts.append(fitted_feature_count(model))
            quality_records.append(quality_filter_record(model, fold=len(quality_records) + 1))

            # 설정된 양성 label의 확률을 공통 평가 함수에 전달한다.
            scores = positive_scores(model, features.iloc[validation_indices], config.dataset)

            metrics = evaluate_binary_scores(
                labels.iloc[validation_indices], scores,
                positive_label=config.dataset.positive_label,
                negative_label=config.dataset.negative_label,
                threshold=config.experiment.default_threshold,
            )

            fold_metrics.append(metrics)

            # 서로 같은 fold의 전체·축소 모델 차이를 나중에 직접 비교할 수 있게 보존한다.
            fold_rows.append({
                "experiment": name, "cv_fold": fold,
                "repeat": (fold - 1) // config.experiment.cv.n_splits + 1,
                "fold_in_repeat": (fold - 1) % config.experiment.cv.n_splits + 1,
                "train_samples": len(fit_indices), "validation_samples": len(validation_indices),
                "selected_feature_count": feature_counts[-1],
                "fit_time_seconds": fit_times[-1], **asdict(metrics),
            })

            selected_rows.extend(_selected_rows(model, name, fold))

            # 재표집별 품질 제거 개수와 실제 사용 행 수를 센서 순위 표와 따로 보관한다.
            selector = model.named_steps.get("selector")
            bootstrap_rows.extend({"experiment": name, "cv_fold": fold, **record}
                                  for record in getattr(selector, "bootstrap_log_", []))

        protocol_id = frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in frame else None
        
        row = {
            "dataset_id": config.dataset.dataset_id,
            "split_strategy": "time_train_cv" if str(protocol_id).startswith("time:") else "random_cv",
            "experiment": name,
            "n_splits": config.experiment.cv.n_splits,
            "n_repeats": config.experiment.cv.n_repeats,
            "threshold": config.experiment.default_threshold,
            "selected_feature_count_mean": float(np.mean(feature_counts)),
            "fit_time_mean_seconds": float(np.mean(fit_times)),
            "training_protocol_id": protocol_id,
            "quality_filter_log": quality_filter_json(quality_records),
        }

        # ROC-AUC가 정의되지 않는 fold의 None은 기존과 같이 집계에서 제외한다.
        for key in ("average_precision", "recall", "precision", "f1", "roc_auc"):
            values = [getattr(metric, key) for metric in fold_metrics if getattr(metric, key) is not None]
            row[f"{key}_mean"] = float(np.mean(values)) if values else None
            row[f"{key}_std"] = float(np.std(values)) if values else None

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("average_precision_mean", ascending=False)

    # 기존 요약 CSV 열을 바꾸지 않고 상세 표는 별도 산출물로 전달한다.
    result.attrs["fold_results"] = pd.DataFrame(fold_rows)
    result.attrs["selected_features"] = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(columns=[
        "experiment", "cv_fold", "feature", "selection_rank",
        "selection_importance", "feature_space",
    ])

    # AP 평균은 동일한 CV fold에서 비교한다. PCA·One-Hot 특징은 원본 센서로 세지 않는다.
    reduction_folds = result.attrs["fold_results"].rename(columns={
        "cv_fold": "fold", "average_precision": "ap", "selected_feature_count": "sensor_count",
    }).copy()

    raw_experiments = set()

    for name, group in result.attrs["selected_features"].groupby("experiment"):
        if group.feature_space.eq("raw_sensor").all():
            raw_experiments.add(name)

    reduction_folds["is_raw_sensor"] = reduction_folds.experiment.isin(raw_experiments)
    result.attrs["reduction_assessment"] = assess_reduction(reduction_folds, config.reduction_policy)
    result.attrs["bootstrap_records"] = pd.DataFrame(bootstrap_rows)

    return result


def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="CV 내부 feature 선택 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/secom/exploratory/legacy/feature_compare.csv"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1,
                        help="RF 선택기 leaf의 최소 샘플 수. S0=1, S1·S2=2입니다.")
    parser.add_argument("--rf-max-depth", type=int, default=None,
                        help="RF 선택기의 최대 깊이. 생략하면 제한 없음, S2=8입니다.")
    parser.add_argument("--rf-stability-repeats", type=int, default=0,
                        help="S3의 학습 내부 반복 선택 횟수. 0은 기존 단일 선택, 2 이상은 반복 선택입니다.")
    parser.add_argument("--topk-xgb-max-depth", type=int, default=None,
                        help="RF Top-K 뒤 최종 XGBoost만 변경합니다. M1=2, 전체 센서 M0·RF는 유지합니다.")
    parser.add_argument("--topk-xgb-reg-lambda", type=float, default=None,
                        help="RF Top-K 뒤 최종 XGBoost의 L2 정규화. M2·M3=5, 전체 센서 M0·RF는 유지합니다.")
    parser.add_argument("--experiments", nargs="+",
                        help="비교할 실험 이름. 생략하면 기존 PCA·L1·LightGBM 목록을 실행합니다.")
    parser.add_argument("--details-dir", type=Path,
                        help="fold별 지표·선택 센서·실험 설정을 저장할 비어 있는 폴더입니다.")
    args = parser.parse_args(argv)

    if args.details_dir is not None and args.details_dir.exists() and any(args.details_dir.iterdir()):
        raise ValueError("상세 결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    
    return args


def save_results(args, config, result):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.details_dir is not None:
        args.details_dir.mkdir(parents=True, exist_ok=True)

        for name in ("fold_results", "selected_features", "reduction_assessment"):
            result.attrs[name].to_csv(args.details_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

        if not result.attrs["bootstrap_records"].empty:
            result.attrs["bootstrap_records"].to_csv(args.details_dir / "bootstrap_records.csv", index=False, encoding="utf-8-sig")

        # 코드 변경 감지용 해시는 추가하지 않고 실제 실행의 설정·버전을 보관한다.
        record = {
            "config": asdict(config), "train_path": str(args.train.resolve()),
            "experiments": result.experiment.tolist(), "n_jobs": args.n_jobs,
            "training_protocol_id": result.training_protocol_id.iloc[0],
            "versions": {name: version(name) for name in ("scikit-learn", "lightgbm", "xgboost")},
            "is_final_selection": False,
            "topk_xgb_parameters": {"max_depth_override": args.topk_xgb_max_depth,
                                    "reg_lambda_override": args.topk_xgb_reg_lambda,
                                    "targets": [name for name in result.experiment
                                                if name.startswith("xgboost_rf_top_")]
                                    if args.topk_xgb_max_depth is not None or args.topk_xgb_reg_lambda is not None else []},
            "rf_selector_parameters": {"min_samples_leaf": args.rf_min_samples_leaf,
                                       "max_depth": args.rf_max_depth,
                                       "stability_repeats": args.rf_stability_repeats,
                                       "resampling": "stratified_bootstrap" if args.rf_stability_repeats else None},
        }

        (args.details_dir / "feature_run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
        )


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    
    result = compare_features(
        pd.read_csv(args.train), config, args.n_jobs, experiment_names=args.experiments,
        rf_min_samples_leaf=args.rf_min_samples_leaf, rf_max_depth=args.rf_max_depth,
        rf_stability_repeats=args.rf_stability_repeats,
        topk_xgb_max_depth=args.topk_xgb_max_depth, topk_xgb_reg_lambda=args.topk_xgb_reg_lambda,
    )

    # 최신 실행 결과를 저장하지만 Test와 외부 Validation은 입력받지 않는다.
    save_results(args, config, result)
    print(result.drop(columns="quality_filter_log").to_string(index=False))
    if not result.attrs["reduction_assessment"].empty:
        print("센서 축소 AP 판정 (최종 선정·threshold 정책과 별도)")
        print(result.attrs["reduction_assessment"].to_string(index=False))


if __name__ == "__main__":
    main()

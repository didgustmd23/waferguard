# ==========================================
# 초기 과거에서 한 번 선택한 센서 목록의 시간순 유지 검증
# - 지정한 외부 학습 구간에서만 반복 gain 선택기를 fit (기본 첫 구간)
# - 이후 모든 구간은 같은 센서명·순서를 사용
# - 대치 통계와 분류기는 각 구간의 과거 학습 데이터로 fit
# - 매 구간 재선택 경로와 같은 평가 행·분류기 설정으로 비교
# - 문턱 선택·외부 Validation/Test·모델 저장은 하지 않음
# ==========================================
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.dataset_schema import split_frame_to_xy
from src.sensor_ml.experiments.feature_time_compare import build_feature_pipelines
from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_metrics import evaluate_binary_scores
from src.modeling_models import fit_pipeline, positive_scores
from src.split_contract import SOURCE_ROW_ID, temporal_folds, validate_train_role


def compare_fixed_sensors(frame, config, *, n_jobs=1, n_estimators=300,
                          n_splits=3, repeats=5, selection_fold=1, evaluate_from_fold=None):
    """초기 센서 고정과 구간별 재선택을 동일한 미래 행에서 비교한다."""
    validate_train_role(frame)
    if config.reduction_policy is None or config.dataset.categorical_feature_columns:
        raise ValueError("고정 센서 검증에는 센서 축소 정책과 수치형 센서 Profile이 필요합니다.")
    
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("트리 수는 양의 정수여야 합니다.")
    
    # 1부터 시작하는 구간 번호를 사용한다. 선택 이전의 과거를 소급 평가하지 않는다.
    start_fold = selection_fold if evaluate_from_fold is None else evaluate_from_fold

    for value in (selection_fold, start_fold):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= n_splits:
            raise ValueError("선택·평가 시작 구간은 전체 구간 수 이하의 양의 정수여야 합니다.")
        
    if start_fold < selection_fold:
        raise ValueError("평가는 센서 선택 구간보다 앞에서 시작할 수 없습니다.")
    
    count = config.reduction_policy.max_sensor_count

    # 센서 수는 Profile과 공통 정책에서 가져오며 데이터셋 이름에 고정하지 않는다.
    name = f"xgboost_top_{count}"

    adaptive = build_feature_pipelines(
        config, n_jobs=n_jobs, n_estimators=n_estimators, experiment_names=(name,),
        xgb_weight_mode="ratio", xgb_selector_weight_mode="ratio",
        xgb_stability_repeats=repeats, topk_xgb_max_depth=2,
        topk_xgb_reg_lambda=1)[name]
    
    folds = list(temporal_folds(frame, config.dataset, n_splits))
    selection_indices = folds[selection_fold - 1][0]
    initial = frame.iloc[selection_indices]
    initial_x, initial_y, feature_names = split_frame_to_xy(initial, config.dataset)

    # 초기 과거만 전달한다. 미래 label과 전체 Train 후보 목록은 사용하지 않는다.
    selection = clone(adaptive[:-1]).fit(initial_x, initial_y)
    fixed_names = selection.get_feature_names_out().tolist()

    if len(fixed_names) != count or len(set(fixed_names)) != count:
        raise ValueError("초기 선택 결과가 정책의 센서 수와 일치하지 않습니다.")
    
    # 고정 경로에는 품질 제거·선택기를 다시 넣지 않는다. 입력은 고정 목록으로 슬라이싱한다.
    fixed = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)
         .set_output(transform="pandas")),
        ("model", clone(adaptive.named_steps["model"])),
    ])

    rows, predictions, selected, weights = [], [], [], []

    for fold, (train_i, eval_i) in enumerate(folds, 1):

        # 551건에서 선택했다면 더 이른 평가 행을 예측 결과에 넣지 않는다.
        if fold < start_fold:
            continue

        train, evaluation = frame.iloc[train_i], frame.iloc[eval_i]

        train_x, train_y, _ = split_frame_to_xy(
            train, config.dataset, expected_feature_columns=feature_names)
        
        eval_x, eval_y, _ = split_frame_to_xy(
            evaluation, config.dataset, expected_feature_columns=feature_names)
        
        # 초기 선택에 사용한 행은 모든 학습 범위에 포함되고 평가 범위와 겹치지 않는다.
        if not set(selection_indices).issubset(set(train_i)) or set(selection_indices) & set(eval_i):
            raise ValueError("고정 센서 선택 구간과 이후 검증 구간의 시간 계약이 잘못됐습니다.")
        
        for role, template in (("adaptive_topk", adaptive), ("fixed_topk", fixed)):
            used_x = train_x.loc[:, fixed_names] if role == "fixed_topk" else train_x
            used_eval = eval_x.loc[:, fixed_names] if role == "fixed_topk" else eval_x

            # 목록은 고정하되 대치·모델의 재학습은 과거 데이터 증가에 따라 허용하는 실험이다.
            model = fit_pipeline(template, used_x, train_y, config, name, n_jobs=n_jobs)
            names = model[:-1].get_feature_names_out().tolist()
            scores = positive_scores(model, used_eval, config.dataset)

            metrics = evaluate_binary_scores(
                eval_y, scores, positive_label=config.dataset.positive_label,
                negative_label=config.dataset.negative_label,
                threshold=config.experiment.default_threshold)
            
            rows.append({"outer_fold": fold, "model_name": role,
                         "train_samples": len(train), "selected_feature_count": len(names),
                         "selection_train_samples": len(initial) if role == "fixed_topk" else len(train),
                         "selection_fold": selection_fold if role == "fixed_topk" else fold,
                         "empty_input_sensor_count": int(used_x.isna().all().sum()),
                         "classifier_max_depth": model.named_steps["model"].max_depth,
                         "classifier_reg_lambda": model.named_steps["model"].reg_lambda,
                         **asdict(metrics)})
            
            selected.extend({"outer_fold": fold, "model_name": role,
                             "sensor_order": order, "feature": sensor}
                            for order, sensor in enumerate(names))
            
            predictions.extend({"outer_fold": fold, "model_name": role,
                                "source_row_index": index,
                                "source_row_id": evaluation[SOURCE_ROW_ID].iloc[pos]
                                if SOURCE_ROW_ID in evaluation else index,
                                "label": eval_y.iloc[pos], "positive_score": float(score)}
                               for pos, (index, score) in enumerate(zip(evaluation.index, scores)))
            
            classifier = model.named_steps["model"]

            weights.append({"outer_fold": fold, "model_name": role,
                            "train_positive": int(train_y.eq(config.dataset.positive_label).sum()),
                            "train_negative": int(train_y.eq(config.dataset.negative_label).sum()),
                            "effective_scale_pos_weight": classifier.effective_scale_pos_weight_})
            
    results = pd.DataFrame(rows)

    summary = results.groupby("model_name", sort=False).agg(
        ap_mean=("average_precision", "mean"),
        # 기존 시간순 실험과 같은 모집단 표준편차(ddof=0)를 사용한다.
        ap_std=("average_precision", lambda values: float(np.std(values, ddof=0))),
        roc_auc_mean=("roc_auc", "mean"), evaluated_folds=("outer_fold", "count"))
    
    paired = results[results.model_name.eq("adaptive_topk")][["outer_fold", "average_precision"]].merge(
        results[results.model_name.eq("fixed_topk")][["outer_fold", "average_precision"]],
        on="outer_fold", suffixes=("_adaptive", "_fixed"), validate="one_to_one")
    
    paired["ap_delta"] = paired.average_precision_fixed - paired.average_precision_adaptive

    return {
        "summary": summary.reset_index(), "fold_results": results,
        "paired_ap": paired, "predictions": pd.DataFrame(predictions),
        "selected_features": pd.DataFrame(selected), "fit_weights": pd.DataFrame(weights),
        "fixed_sensors": pd.DataFrame({"sensor_order": range(count), "feature": fixed_names}),
        "selection_rows": pd.DataFrame({"source_row_index": initial.index,
                                         "source_row_id": initial[SOURCE_ROW_ID].to_numpy()
                                         if SOURCE_ROW_ID in initial else initial.index}),
        "selection_bootstrap": pd.DataFrame(selection.named_steps["selector"].bootstrap_log_),
    }


def parse_args(argv=None):
    """학습 입력과 새 결과 폴더만 받는다."""
    parser = argparse.ArgumentParser(description="초기 과거의 고정 센서와 재선택 경로 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--selection-fold", type=int, default=1,
                        help="센서를 한 번 선택할 외부 학습 구간 번호. 기본 1입니다.")
    parser.add_argument("--evaluate-from-fold", type=int, default=None,
                        help="평가 시작 구간 번호. 생략하면 선택 구간부터 평가합니다.")
    args = parser.parse_args(argv)

    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("비어 있는 새 결과 폴더를 지정하세요.")
    
    return args


def _json_value(value):
    """설정의 Path만 문자열로 변환하고 예상하지 못한 타입은 오류로 남긴다."""
    # WindowsPath/PosixPath는 json 기본 지원 타입이 아니다. 경로의 의미는 유지한다.
    if isinstance(value, Path):
        return str(value)
    
    raise TypeError(f"실행 기록에서 JSON으로 변환할 수 없는 타입입니다: {type(value).__name__}")


def execution_json(args, config):
    """중첩 설정의 경로까지 포함해 실행 기록을 학습 전에 직렬화한다."""
    # 실제 선택 출처와 모델 설정을 기록한다. 고정 모델·최종 문턱의 동결 기록은 아니다.
    record = {"config": asdict(config), "train_path": str(args.train.resolve()),
              "selection_scope": "selected_outer_train_only", "n_splits": args.n_splits,
              "selection_fold": args.selection_fold,
              "evaluate_from_fold": args.evaluate_from_fold if args.evaluate_from_fold is not None else args.selection_fold,
              "n_estimators": args.n_estimators, "repeats": args.repeats, "n_jobs": args.n_jobs,
              "classifier_max_depth": 2, "classifier_reg_lambda": 1,
              "selector_max_depth": 3, "selector_reg_lambda": 1,
              "class_weight_mode": "ratio", "threshold_selection_performed": False,
              "external_validation_used": False, "test_used": False,
              "is_final_model": False, "model_saved": False}
    
    return json.dumps(record, ensure_ascii=False, indent=2, default=_json_value)


def main():
    """설정 검증 후 결과를 저장하고 문턱 진단의 범위를 표시한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    # 저장 형식 오류를 긴 학습이 끝난 뒤에 발견하지 않도록 먼저 확인한다.
    record_text = execution_json(args, config)
    frame = pd.read_csv(args.train)
    results = compare_fixed_sensors(frame, config, n_jobs=args.n_jobs,
                                    n_estimators=args.n_estimators, n_splits=args.n_splits,
                                    repeats=args.repeats, selection_fold=args.selection_fold,
                                    evaluate_from_fold=args.evaluate_from_fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "execution.json").write_text(record_text, encoding="utf-8")
    print(results["summary"].to_string(index=False))
    print("같은 초기 센서 목록을 유지한 개발 검증입니다. 문턱을 선택하거나 모델을 동결하지 않았습니다.")
    print("Recall은 기본 문턱 진단입니다. OOF Recall 90% 시나리오의 결과가 아닙니다.")
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

# ==========================================
# 저장된 고정 센서 후보의 선택 이후 시간순 OOF 문턱 검증
# - 센서 선택에 사용한 과거 행을 OOF 평가에서 제외
# - 초기 선택 과거를 학습에 사용하고 이후 블록마다 학습 범위를 확장
# - 동일 timestamp는 같은 블록에 유지
# - 과거 OOF에서 문턱을 선택한 뒤 마지막 개발 구간에 적용
# - 센서 재선택·최종 모델 저장·외부 Validation/Test는 하지 않음
# ==========================================
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.dataset_schema import split_frame_to_xy
from src.sensor_ml.experiments.fixed_sensor_compare import _json_value
from src.sensor_ml.experiments.step8_threshold_oof import compare_thresholds
from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_metrics import evaluate_binary_scores
from src.modeling_models import build_classifier, fit_pipeline, positive_scores
from src.split_contract import SOURCE_ROW_ID, temporal_folds, validate_train_role


def validate_candidate(frame, config, candidate_dir):
    """후보의 설정·선택 행·센서 순서를 현재 학습 계약과 대조한다."""
    record = json.loads((candidate_dir / "execution.json").read_text(encoding="utf-8"))
    current = json.loads(json.dumps(asdict(config), default=_json_value))

    if record["config"] != current:
        raise ValueError("현재 설정이 센서 후보 선정 당시와 다릅니다. 같은 설정을 사용하세요.")
    
    if record["selection_fold"] != 2 or record["n_splits"] != 3:
        raise ValueError("이번 검증은 두 번째 과거에서 선택한 3구간 후보만 지원합니다.")
    
    if (record["classifier_max_depth"], record["classifier_reg_lambda"],
            record["class_weight_mode"]) != (2, 1, "ratio"):
        raise ValueError("후보의 최종 분류기 설정이 현재 주후보와 다릅니다.")
    
    names = pd.read_csv(candidate_dir / "fixed_sensors.csv").sort_values("sensor_order").feature.tolist()
    rows = pd.read_csv(candidate_dir / "selection_rows.csv")["source_row_id"].tolist()
    expected = frame.iloc[list(temporal_folds(frame, config.dataset, 3))[1][0]][SOURCE_ROW_ID].tolist()

    if rows != expected:
        raise ValueError("센서 선택 행이 현재 Train의 두 번째 과거 구간과 일치하지 않습니다.")
    
    return names, record


def compare_fixed_oof(frame, config, names, *, n_jobs=1, n_estimators=300,
                      inner_splits=2, targets=(0.8, 0.9)):
    """선택 이후 과거 OOF와 마지막 개발 평가의 역할을 분리한다."""
    validate_train_role(frame)

    if SOURCE_ROW_ID not in frame:
        raise ValueError("고정 후보 OOF 검증에는 원본 행 ID가 필요합니다.")
    
    if config.reduction_policy is None:
        raise ValueError("고정 센서 검증에는 센서 축소 정책이 필요합니다.")
    
    if not names or len(set(names)) != len(names) or len(names) != config.reduction_policy.max_sensor_count:
        raise ValueError("고정 센서 목록은 정책 센서 수와 같고 중복이 없어야 합니다.")
    
    if isinstance(inner_splits, bool) or not isinstance(inner_splits, int) or inner_splits < 2:
        raise ValueError("시간순 OOF 블록 수는 2 이상의 정수여야 합니다.")
    
    if not targets or any(isinstance(t, bool) or not np.isfinite(t) or not 0 < t <= 1 for t in targets):
        raise ValueError("Recall 목표는 0 초과 1 이하의 유한한 값이어야 합니다.")
    
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("트리 수는 양의 정수여야 합니다.")
    
    features, labels, _ = split_frame_to_xy(frame, config.dataset)

    if any(name not in features for name in names):
        raise ValueError("고정 센서 컬럼이 Train에 없습니다.")
    
    features = features.loc[:, names]
    folds = list(temporal_folds(frame, config.dataset, 3))

    selection_i = folds[1][0]
    train_i, future_i = folds[2]

    selection_set = set(selection_i)
    post_i = [i for i in train_i if i not in selection_set]

    times = pd.to_datetime(frame[config.dataset.timestamp_column], format=config.dataset.timestamp_format)
    unique_times = np.sort(times.iloc[post_i].unique())

    if len(unique_times) < inner_splits:
        raise ValueError("선택 이후 OOF를 나누기에 timestamp 수가 부족합니다.")
    
    classifier = build_classifier(config, "xgboost", n_jobs=n_jobs, n_estimators=n_estimators)
    classifier.set_params(max_depth=2, reg_lambda=1, class_weight_mode="ratio")

    template = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
        ("model", classifier),
    ])

    past_i = list(selection_i)
    oof, blocks = [], []

    for block, dates in enumerate(np.array_split(unique_times, inner_splits), 1):
        eval_i = [i for i in post_i if times.iloc[i] in dates]

        # 목록은 외부 후보 그대로 사용하고 대치·가중치·모델만 앞선 데이터에서 fit한다.
        if times.iloc[past_i].max() >= times.iloc[eval_i].min():
            raise ValueError("OOF 학습과 평가의 시간 범위가 겹칩니다.")
        
        model = fit_pipeline(template, features.iloc[past_i], labels.iloc[past_i],
                             config, "fixed_topk", n_jobs=n_jobs)
        
        scores = positive_scores(model, features.iloc[eval_i], config.dataset)

        oof.extend({"inner_fold": block, "source_row_id": frame[SOURCE_ROW_ID].iloc[i],
                    "label": labels.iloc[i], "oof_positive_score": float(score)}
                   for i, score in zip(eval_i, scores))
        
        blocks.append({"inner_fold": block, "train_samples": len(past_i),
                       "evaluation_samples": len(eval_i),
                       "evaluation_fail": int(labels.iloc[eval_i].eq(config.dataset.positive_label).sum()),
                       "train_end": str(times.iloc[past_i].max()),
                       "evaluation_start": str(times.iloc[eval_i].min()),
                       "effective_scale_pos_weight": model.named_steps["model"].effective_scale_pos_weight_})
        
        past_i.extend(eval_i)

    oof = pd.DataFrame(oof)

    if oof.source_row_id.duplicated().any():
        raise ValueError("OOF에서 원본 행이 중복 예측됐습니다.")
    
    table = compare_thresholds(oof, config, experiment_name="fixed_topk")
    table["alarm_ratio"] = (table.true_positive + table.false_positive) / table.support

    # 평가 label은 문턱 선택 함수에 전달하지 않는다. 미래 성능은 문턱 선택 후 계산한다.
    choices = []

    for target in targets:
        eligible = table[table.recall >= target]

        if eligible.empty:
            raise ValueError(f"과거 OOF Recall 목표 {target}를 만족하는 후보가 없습니다.")
        
        choice = eligible.sort_values(["alarm_ratio", "precision", "threshold"],
                                      ascending=[True, False, False]).iloc[0]
        choices.append((target, choice))

    final = fit_pipeline(template, features.iloc[train_i], labels.iloc[train_i],
                         config, "fixed_topk", n_jobs=n_jobs)
    
    future_scores = positive_scores(final, features.iloc[future_i], config.dataset)
    
    scenarios = []

    for target, choice in choices:

        metrics = evaluate_binary_scores(labels.iloc[future_i], future_scores,
                  positive_label=config.dataset.positive_label, negative_label=config.dataset.negative_label,
                  threshold=float(choice.threshold))
        
        scenarios.append({"target_recall": target, "oof_recall": float(choice.recall),
                          "oof_alarm_ratio": float(choice.alarm_ratio), **asdict(metrics),
                          "alarm_ratio": (metrics.true_positive + metrics.false_positive) / metrics.support})
        
    return {"scenario_results": pd.DataFrame(scenarios), "threshold_compare": table,
            "oof_predictions": oof, "inner_folds": pd.DataFrame(blocks),
            "future_predictions": pd.DataFrame({"source_row_id": frame[SOURCE_ROW_ID].iloc[future_i].to_numpy(),
                                                 "label": labels.iloc[future_i].to_numpy(),
                                                 "positive_score": future_scores}),
            "fixed_sensors": pd.DataFrame({"sensor_order": range(len(names)), "feature": names})}


def main():
    """후보 계약을 확인한 뒤 선택 이후 OOF 개발 검증을 새 폴더에 저장한다."""
    parser = argparse.ArgumentParser(description="고정 센서의 선택 이후 OOF 문턱 개발 검증")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("비어 있는 새 결과 폴더를 지정하세요.")
    config = load_modeling_config(args.config)
    frame = pd.read_csv(args.train)
    validate_train_role(frame)
    names, candidate = validate_candidate(frame, config, args.candidate_dir)
    record = {"config": asdict(config), "candidate_dir": str(args.candidate_dir.resolve()),
              "candidate_execution": candidate, "train_path": str(args.train.resolve()),
              "n_jobs": args.n_jobs, "inner_splits": 2, "recall_targets": [0.8, 0.9],
              "oof_scope": "after_selection_only", "alarm_limit_applied": False,
              "test_used": False, "is_final_model": False, "model_saved": False}
    record_text = json.dumps(record, ensure_ascii=False, indent=2, default=_json_value)
    results = compare_fixed_oof(frame, config, names, n_jobs=args.n_jobs,
                                n_estimators=candidate["n_estimators"])
    # 같은 후보의 재학습 확률이 기존 결과와 일치하는지 확인해 설정·입력 변경을 탐지한다.
    previous = pd.read_csv(args.candidate_dir / "predictions.csv", float_precision="round_trip")
    previous = previous.loc[previous.model_name.eq("fixed_topk") & previous.outer_fold.eq(3)]
    paired = results["future_predictions"].merge(
        previous[["source_row_id", "label", "positive_score"]], on="source_row_id",
        suffixes=("_new", "_previous"), validate="one_to_one")
    if (len(paired) != len(previous) or len(paired) != len(results["future_predictions"])
            or not paired.label_new.eq(paired.label_previous).all()
            or not np.allclose(paired.positive_score_new, paired.positive_score_previous,
                               rtol=0, atol=1e-7)):
        raise ValueError("기존 고정 후보의 평가 행·확률과 일치하지 않습니다. 입력·환경·설정을 확인하세요.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "execution.json").write_text(record_text, encoding="utf-8")
    print(results["scenario_results"].to_string(index=False))
    print("선택 이후 과거 OOF의 문턱을 마지막 개발 구간에 적용했습니다. 최종 Test·배포 승인 결과가 아닙니다.")
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

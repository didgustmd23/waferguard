# ==========================================
# 저장 V2 고정 Top-20과 비교할 전체 센서 기준 모델 준비
# - 기존 V2의 학습 행과 선택 이후 OOF 블록을 그대로 사용
# - 센서 품질 제거·중앙값 대치는 각 과거 학습 Pipeline 안에서 fit
# - 최종 XGBoost 설정은 Top-20과 동일하고 중요도 센서 선택만 생략
# - 80%·90% 문턱은 OOF만으로 선택한 뒤 모델과 함께 보존
# - 이 단계는 Test 파일을 읽거나 평가하지 않음
# ==========================================
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.dataset_schema import split_frame_to_xy
from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_models import build_classifier, fit_pipeline, positive_scores
from src.modeling_preprocessing import SensorQualityFilter, quality_filter_record
from src.sensor_ml.evaluation.evaluate_v2_test import dataset_contract
from src.sensor_ml.experiments.step8_threshold_oof import compare_thresholds
from src.split_contract import SOURCE_ROW_ID, PROTOCOL_ID, temporal_folds, validate_train_role


def reference_blocks(frame, config, reference_dir):
    """기존 고정 모델의 OOF 행·정답·시간 경계와 학습 범위를 대조한다."""
    validate_train_role(frame)
    if SOURCE_ROW_ID not in frame or frame[SOURCE_ROW_ID].duplicated().any():
        raise ValueError("Train의 원본 행 ID가 없거나 중복되어 있습니다.")
    
    execution = json.loads((reference_dir / "experiment/execution.json").read_text(encoding="utf-8"))
    current = json.loads(json.dumps(asdict(config), default=str))

    if (dataset_contract(current["dataset"]) != dataset_contract(execution["config"]["dataset"])
            or current["experiment"] != execution["config"]["experiment"]):
        raise ValueError("현재 데이터 정의·실험 설정이 저장 V2와 다릅니다.")
    
    candidate = execution["candidate_execution"]

    if (execution["oof_scope"] != "after_selection_only" or execution["inner_splits"] != 2
            or candidate["selection_fold"] != 2 or candidate["n_splits"] != 3
            or (candidate["classifier_max_depth"], candidate["classifier_reg_lambda"],
                candidate["class_weight_mode"]) != (2, 1, "ratio")):
        raise ValueError("지원하는 V2 고정 센서 비교 계약이 아닙니다.")

    # 저장된 OOF 블록의 ID를 사용하므로 표본 수·정답·평가 행이 정확히 같다.
    expected = pd.read_csv(reference_dir / "experiment/oof_predictions.csv")
    folds = list(temporal_folds(frame, config.dataset, 3))
    fit_i, future_i = folds[2]
    initial = set(folds[1][0])
    post_i = [i for i in fit_i if i not in initial]

    actual = frame.iloc[post_i][[SOURCE_ROW_ID, config.dataset.label_column]].copy()
    actual.columns = ["source_row_id", "label"]

    previous = expected[["source_row_id", "label"]]

    if not actual.reset_index(drop=True).equals(previous.reset_index(drop=True)):
        raise ValueError("Train의 OOF 행·정답이 저장 V2 기록과 다릅니다.")
    
    if expected.inner_fold.tolist() != sorted(expected.inner_fold.tolist()) or set(expected.inner_fold) != {1, 2}:
        raise ValueError("기존 OOF 블록 순서·개수가 잘못되어 있습니다.")
    
    future = pd.read_csv(reference_dir / "experiment/future_predictions.csv")

    if frame.iloc[future_i][SOURCE_ROW_ID].tolist() != future.source_row_id.tolist():
        raise ValueError("마지막 개발 구간이 저장 V2와 다릅니다.")
    
    times = pd.to_datetime(frame[config.dataset.timestamp_column], format=config.dataset.timestamp_format)
    past = list(folds[1][0])
    blocks = []

    for number, group in expected.groupby("inner_fold", sort=True):
        ids = set(group.source_row_id)
        eval_i = [i for i in post_i if frame[SOURCE_ROW_ID].iloc[i] in ids]

        if not eval_i or times.iloc[past].max() >= times.iloc[eval_i].min():
            raise ValueError("OOF 학습·평가의 시간 범위가 겹칩니다.")
        
        blocks.append((int(number), past.copy(), eval_i))
        past.extend(eval_i)

    if past != list(fit_i):
        raise ValueError("OOF 확장 후 학습 범위가 저장 V2와 다릅니다.")
    
    return blocks, fit_i, execution


def choose_scenarios(oof, config):
    """기존 V2와 동일하게 목표 달성 후보 중 선별 부담이 가장 작은 문턱을 고른다."""
    table = compare_thresholds(oof, config, experiment_name="v2_all_matched_depth2")
    table["alarm_ratio"] = (table.true_positive + table.false_positive) / table.support
    choices = []

    for target in (0.8, 0.9):
        eligible = table.loc[table.recall >= target].sort_values(
            ["alarm_ratio", "precision", "threshold"], ascending=[True, False, False])
        
        if eligible.empty:
            raise ValueError(f"OOF Recall {target:.0%}를 만족하는 문턱 후보가 없습니다.")
        
        choices.append({"target_recall": target, **eligible.iloc[0].to_dict()})

    return table, pd.DataFrame(choices)


def prepare_reference(frame, config, reference_dir, n_jobs):
    """Test에 접근하지 않고 전체 센서 OOF와 학습 824행 기준 분류기를 준비한다."""
    blocks, fit_i, execution = reference_blocks(frame, config, reference_dir)
    features, labels, _ = split_frame_to_xy(frame, config.dataset)
    classifier = build_classifier(config, "xgboost", n_jobs=n_jobs,
                                  n_estimators=execution["candidate_execution"]["n_estimators"])
    
    classifier.set_params(max_depth=2, reg_lambda=1, class_weight_mode="ratio")

    template = Pipeline([
        ("quality_filter", SensorQualityFilter(config.dataset.missing_ratio_threshold,
                                                config.dataset.drop_zero_variance)),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
        ("model", classifier),
    ])

    rows, quality = [], []

    for number, past_i, eval_i in blocks:
        # fit_pipeline이 매번 clone하므로 앞 블록의 전처리 통계가 섞이지 않는다.
        fitted = fit_pipeline(template, features.iloc[past_i], labels.iloc[past_i],
                              config, "v2_all_matched_depth2", n_jobs=n_jobs)
        
        scores = positive_scores(fitted, features.iloc[eval_i], config.dataset)

        rows.extend({"inner_fold": number, "source_row_id": frame[SOURCE_ROW_ID].iloc[i],
                     "label": labels.iloc[i], "oof_positive_score": float(score)}
                    for i, score in zip(eval_i, scores))
        
        quality.append({**quality_filter_record(fitted, fold=number), "train_samples": len(past_i)})

    oof = pd.DataFrame(rows)
    thresholds, scenarios = choose_scenarios(oof, config)

    # 문턱을 확정한 뒤 같은 과거 824행으로 한 번 학습한다. Test는 이후 별도 평가한다.
    final = fit_pipeline(template, features.iloc[fit_i], labels.iloc[fit_i],
                         config, "v2_all_matched_depth2", n_jobs=n_jobs)
    
    quality.append({**quality_filter_record(final, fold="final"), "train_samples": len(fit_i)})

    return final, oof, thresholds, scenarios, quality, fit_i, execution


def main():
    """새 비교 폴더에 OOF 근거·동결 문턱·모델을 저장하며 기존 V2는 수정하지 않는다."""
    parser = argparse.ArgumentParser(description="V2와 동일 학습·OOF 조건의 전체 센서 비교 모델 준비 (Test 미사용)")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reference-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("비어 있는 새 출력 폴더를 지정하세요.")
    
    if args.n_jobs == 0 or args.n_jobs < -1:
        raise ValueError("병렬 수는 -1 또는 양의 정수여야 합니다.")
    
    config = load_modeling_config(args.config)

    frame = pd.read_csv(args.train)

    final, oof, thresholds, scenarios, quality, fit_i, execution = prepare_reference(
        frame, config, args.reference_bundle_dir, args.n_jobs)
    
    record = {"status": "prepared", "model_name": "v2_all_matched_depth2",
              "config": asdict(config), "reference_execution": execution,
              "reference_bundle_dir": str(args.reference_bundle_dir.resolve()),
              "train_path": str(args.train.resolve()), "model_fit_rows": len(fit_i),
              "training_protocol_id": str(frame[PROTOCOL_ID].iloc[0]),
              "model_parameters": final.named_steps["model"].get_params(),
              "retained_sensors": list(final.named_steps["quality_filter"].retained_features_),
              "scenarios": scenarios.to_dict(orient="records"), "quality_filter": quality,
              "test_used": False, "is_final_model": False,
              "versions": {name: version(name) for name in ("scikit-learn", "xgboost", "joblib")}}
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(args.output_dir / "threshold_compare.csv", index=False, encoding="utf-8-sig")
    scenarios.to_csv(args.output_dir / "scenario_thresholds.csv", index=False, encoding="utf-8-sig")

    frame.iloc[fit_i][[SOURCE_ROW_ID]].to_csv(args.output_dir / "fit_rows.csv", index=False)

    joblib.dump(final, args.output_dir / "pipeline.joblib")

    (args.output_dir / "execution.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    
    print(scenarios[["target_recall", "threshold", "recall", "alarm_ratio", "true_positive",
                     "false_positive", "false_negative"]].to_string(index=False))
    print(f"학습 행 수: {len(fit_i)}, 품질 필터 후 센서 수: {len(record['retained_sensors'])}")
    print(f"전체 센서 비교 모델 준비 경로: {args.output_dir}")
    print("Test는 읽지 않았습니다. 다음 별도 명령으로 같은 기존 Test를 평가합니다.")


if __name__ == "__main__":
    main()

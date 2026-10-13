# ==========================================
# 동결된 전체 센서 모델의 기존 V2 Test 비교 평가
# - 저장 모델을 한 번 복원·예측하고 두 OOF 문턱을 적용
# - 기존 고정 Top-20과 평가 행·정답·학습 범위를 대조
# - Test에서 fit·센서 재선택·문턱 탐색을 하지 않음
# - 원본 V2 결과를 수정하지 않고 새 비교 폴더에 저장
# ==========================================
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_metrics import evaluate_binary_scores
from src.modeling_models import positive_scores
from src.sensor_ml.diagnostics.plot_v2_test import read_evaluation, validate_pair
from src.sensor_ml.evaluation.evaluate_frozen_model import (
    reserve_evaluation, save_evaluation_results, validate_final_pair,
)
from src.sensor_ml.evaluation.evaluate_v2_test import dataset_contract, validate_v1_rows
from src.sensor_ml.experiments.all_sensor_reference import reference_blocks, choose_scenarios
from src.sensor_ml.inference.predict_cli import read_sensor_csv
from src.split_contract import SOURCE_ROW_ID


def validate_reference(directory, record, train, config):
    """현재 설정·학습 행·OOF 근거를 동결 기록과 대조하고 두 문턱을 반환한다."""
    if (record.get("status") != "prepared" or record.get("test_used") is not False
            or record.get("model_name") != "v2_all_matched_depth2"):
        raise ValueError("준비 완료된 전체 센서 비교 모델 기록이 필요합니다.")
    
    current = json.loads(json.dumps(asdict(config), default=str))

    if (dataset_contract(current["dataset"]) != dataset_contract(record["config"]["dataset"])
            or current["experiment"] != record["config"]["experiment"]):
        raise ValueError("현재 데이터 정의·실험 설정이 모델 준비 당시와 다릅니다.")
    
    # 역직렬화 전에 버전을 확인한다. 출처가 불명확한 joblib 파일은 사용하지 않는다.
    for name, saved in record["versions"].items():
        if version(name) != saved:
            raise ValueError(f"저장 모델과 현재 패키지 버전이 다릅니다: {name}")
        
    blocks, fit_i, execution = reference_blocks(
        train, config, Path(record["reference_bundle_dir"]))
    
    if execution != record["reference_execution"]:
        raise ValueError("기준 V2의 OOF 실행 기록이 준비 당시와 다릅니다.")
    
    fit_rows = pd.read_csv(directory / "fit_rows.csv")[SOURCE_ROW_ID].tolist()

    if (train.iloc[fit_i][SOURCE_ROW_ID].tolist() != fit_rows
            or len(fit_rows) != record["model_fit_rows"]):
        raise ValueError("저장 모델의 학습 행이 기준 V2와 다릅니다.")
    
    oof = pd.read_csv(directory / "oof_predictions.csv", float_precision="round_trip")

    # 문턱을 바꾸지 않는다. 보존된 OOF에서 같은 선택 규칙의 결과인지 검증만 한다.
    expected_ids = [train[SOURCE_ROW_ID].iloc[i] for _, _, indices in blocks for i in indices]

    if oof.source_row_id.tolist() != expected_ids:
        raise ValueError("전체 센서의 OOF 평가 행이 기준 V2와 다릅니다.")
    
    expected_labels = train.set_index(SOURCE_ROW_ID).loc[expected_ids, config.dataset.label_column]

    if not np.array_equal(oof.label, expected_labels):
        raise ValueError("전체 센서의 OOF 정답이 Train과 다릅니다.")
    
    _, selected = choose_scenarios(oof, config)

    frozen = pd.DataFrame(record["scenarios"]).sort_values("target_recall").reset_index(drop=True)
    saved = pd.read_csv(directory / "scenario_thresholds.csv", float_precision="round_trip")
    columns = ["target_recall", "threshold", "recall", "alarm_ratio"]

    for other in (selected, saved):
        other = other.sort_values("target_recall").reset_index(drop=True)
    
        if len(other) != 2 or not np.array_equal(frozen[columns].to_numpy(), other[columns].to_numpy()):
            raise ValueError("동결 문턱·OOF 선택 근거가 서로 다릅니다.")
        
    return frozen


def validate_pipeline(pipeline, record):
    """복원한 품질 필터·대치기·분류기의 구조와 저장된 설정을 확인한다."""
    if list(pipeline.named_steps) != ["quality_filter", "imputer", "model"]:
        raise ValueError("전체 센서 비교 Pipeline 구조가 잘못되었습니다.")
    
    if list(pipeline.named_steps["quality_filter"].retained_features_) != record["retained_sensors"]:
        raise ValueError("복원 모델의 유지 센서가 저장 기록과 다릅니다.")
    
    parameters = json.loads(json.dumps(pipeline.named_steps["model"].get_params(), default=str))

    if parameters != record["model_parameters"]:
        raise ValueError("복원 분류기 설정이 동결 기록과 다릅니다.")
    
    dataset = record["config"]["dataset"]
    quality = pipeline.named_steps["quality_filter"]

    if (quality.missing_ratio_threshold != dataset["missing_ratio_threshold"]
            or quality.drop_zero_variance != dataset["drop_zero_variance"]
            or pipeline.named_steps["imputer"].strategy != "median"):
        raise ValueError("복원 전처리 설정이 동결 기록과 다릅니다.")


def evaluate_scenarios(pipeline, features, labels, config, scenarios):
    
    """모델 확률은 한 번만 구하고 각 동결 문턱의 지표·판정을 반환한다."""
    scores = positive_scores(pipeline, features, config.dataset)
    results = []

    for row in scenarios.itertuples(index=False):
        threshold = float(row.threshold)

        metrics = asdict(evaluate_binary_scores(
            labels, scores, positive_label=config.dataset.positive_label,
            negative_label=config.dataset.negative_label, threshold=threshold,
        ))

        selected = scores >= threshold

        metrics.update(target_recall=float(row.target_recall),
                       recall_target_met=metrics["recall"] >= row.target_recall,
                       reinspection_ratio=float(selected.mean()))
        
        predictions = pd.DataFrame({"positive_score": scores, "predicted_positive": selected,
                                    "predicted_label": np.where(selected, config.dataset.positive_label,
                                                                config.dataset.negative_label)})
        
        results.append((float(row.target_recall), metrics, predictions))

    return results


def main():
    """사전 검사 후 한 번 평가하고 기존 Top-20과의 비교표를 저장한다."""
    parser = argparse.ArgumentParser(description="전체 센서 동결 모델의 기존 V2 Test 비교 (재학습 없음)")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--top20-recall80-dir", type=Path, required=True)
    parser.add_argument("--top20-recall90-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trusted-local-model", action="store_true")
    parser.add_argument("--confirm-existing-test-comparison", action="store_true")
    args = parser.parse_args()

    if not args.trusted_local_model or not args.confirm_existing_test_comparison:
        parser.error("신뢰한 로컬 모델과 기존 Test 비교를 두 확인 옵션으로 승인하세요.")

    if (args.reference_dir / "evaluation_started.json").exists():
        raise ValueError("이 전체 센서 모델에는 평가 실행 기록이 이미 있습니다.")
    
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("비어 있는 새 평가 출력 폴더를 지정하세요.")
    
    record = json.loads((args.reference_dir / "execution.json").read_text(encoding="utf-8"))
    config = load_modeling_config(args.config)
    train = read_sensor_csv(args.train)
    scenarios = validate_reference(args.reference_dir, record, train, config)
    test = read_sensor_csv(args.test)
    features, labels = validate_final_pair(train, test, config.dataset, record["training_protocol_id"])
    old80, meta80 = read_evaluation(args.top20_recall80_dir, 0.8)
    old90, meta90 = read_evaluation(args.top20_recall90_dir, 0.9)

    validate_pair(old80, old90)

    for previous, metadata in ((old80, meta80), (old90, meta90)):
        validate_v1_rows(test, previous)
        if (metadata["model_fit_rows"] != record["model_fit_rows"]
                or metadata["training_protocol_id"] != record["training_protocol_id"]):
            raise ValueError("고정 Top-20 결과와 학습 범위·분할 계약이 다릅니다.")
        
    pipeline = joblib.load(args.reference_dir / "pipeline.joblib")
    validate_pipeline(pipeline, record)

    # 품질 필터 변환은 입력 컬럼 검사만 하며 학습된 유지 목록을 바꾸지 않는다.
    pipeline.named_steps["quality_filter"].transform(features)

    start = {"status": "started", "evaluation_scope": "existing_v1_test_comparison",
             "historical_nonuse_guaranteed": False, "refit": False,
             "model_name": record["model_name"], "model_fit_rows": record["model_fit_rows"],
             "training_protocol_id": record["training_protocol_id"],
             "reference_dir": str(args.reference_dir.resolve()),
             "train_path": str(args.train.resolve()), "test_path": str(args.test.resolve())}
    
    marker = reserve_evaluation(args.reference_dir, args.output_dir, start)

    # 시작 표식 이후 실패해도 자동 재실행하지 않는다. 기존 V2 평가에는 접근하지 않는다.
    results = evaluate_scenarios(pipeline, features, labels, config, scenarios)
    comparison = []

    for target, metrics, predictions in results:
        directory = args.output_dir / f"recall{int(target * 100)}"
        directory.mkdir()

        scenario_record = {**start, "target_recall": target, "threshold": metrics["threshold"],
                           "sensors": record["retained_sensors"]}
        
        save_evaluation_results(directory, directory / "evaluation_started.json", scenario_record,
                                metrics, predictions, test, labels,
                                {"min_recall": target, "max_reinspection_ratio": None})
        
        comparison.append({"model_name": "all_sensors", "sensor_count": len(record["retained_sensors"]), **metrics})
        previous = meta80 if target == 0.8 else meta90

        comparison.append({"model_name": "fixed_top20", "sensor_count": len(previous["sensors"]),
                           **previous["metrics"]})
        
    summary = pd.DataFrame(comparison)
    summary.to_csv(args.output_dir / "comparison.csv", index=False, encoding="utf-8-sig")

    report = ["# 전체 센서·고정 Top-20의 동일 기존 Test 비교", "",
              "학습 824행·깊이 2·규제 1·300 trees·비율 가중치 조건의 비교입니다.",
              "문턱은 각 모델의 동일 Train 내부 OOF 구간에서 선택했습니다. 같은 숫자의 문턱을 공유하지 않습니다.",
              "기존 Test 재사용 비교이며 독립 미사용 Test가 아닙니다. 평가 중 재학습·문턱 변경은 없습니다.", "",
              "```text", summary.to_string(index=False), "```", ""]
    
    (args.output_dir / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    
    start.update(status="completed", targets=[0.8, 0.9], output_dir=str(args.output_dir.resolve()))
    marker.write_text(json.dumps(start, ensure_ascii=False, indent=2), encoding="utf-8")

    (args.output_dir / "evaluation.json").write_text(
        json.dumps(start, ensure_ascii=False, indent=2), encoding="utf-8")
    
    columns = ["model_name", "target_recall", "sensor_count", "threshold", "recall", "precision",
               "reinspection_ratio", "average_precision", "roc_auc", "true_positive", "false_positive", "false_negative"]
    
    print(summary[columns].to_string(index=False))
    print(f"동일 Test 비교 보고서: {args.output_dir / 'comparison.md'}")
    print("기존 V2는 변경하지 않았습니다. 기존 Test 재사용 비교이며 문턱을 재조정하지 않습니다.")


if __name__ == "__main__":
    main()

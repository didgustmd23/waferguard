# ==========================================
# 저장된 V2 고정 센서 후보의 기존 Test 구간 평가
# - 90%·80% 묶음의 모델과 문턱을 그대로 사용하며 fit하지 않음
# - 저장 OOF·센서 목록·개발 구간과 V1 평가 행을 대조
# - 폴더 이동 전 경로는 보존하고 현재 경로를 별도로 기록
# - 이미 사용한 Test의 비교 평가이며 독립 미사용 평가로 표시하지 않음
# ==========================================
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.sensor_ml.evaluation.evaluate_frozen_model import (
    evaluate_saved_predictions, reserve_evaluation, save_evaluation_results,
    validate_final_pair,
)
from src.sensor_ml.inference.predict_cli import read_sensor_csv
from src.sensor_ml.inference.save_v2_scenario import scenario_source
from src.sensor_ml.inference.sensor_bundle import load_sensor_bundle
from src.split_contract import SOURCE_ROW_ID, temporal_folds


def dataset_contract(snapshot):
    """파일 위치만 제외하고 라벨·센서·변환 정의를 비교용으로 추출한다."""
    
    # 원본 절대 경로는 바꾸지 않는다. 비교용 복사본에서 위치만 제외한다.
    contract = json.loads(json.dumps(snapshot, default=str))
    contract.pop("input_path", None)

    for source in contract.get("ingestion", {}).get("sources", []):
        source.pop("path", None)

    return contract


def validate_v1_rows(test, previous):
    """V1에서 평가한 원본 ID·label과 현재 Test가 정확히 같은지 검사한다."""
    current = test[[SOURCE_ROW_ID, "label"]].rename(columns={SOURCE_ROW_ID: "source_row_id"})
    expected = previous[["source_row_id", "label"]]

    # 순서 변경은 허용하되 중복·누락·추가·정답 변경은 허용하지 않는다.
    for table in (current, expected):
        if table.isna().any().any() or table.source_row_id.duplicated().any():
            raise ValueError("V1 비교 입력의 원본 ID·label에 결측 또는 중복이 있습니다.")
        
    if not current.sort_values("source_row_id").reset_index(drop=True).equals(
        expected.sort_values("source_row_id").reset_index(drop=True)
    ):
        raise ValueError("현재 Test의 행·label이 V1 평가 기록과 다릅니다.")


def validate_v2_bundle(manifest, inference, directory, config):
    """저장 시나리오와 보존 OOF를 연결하며 현재 정책으로 문턱을 바꾸지 않는다."""
    provenance = manifest["provenance"]

    if provenance.get("version") != "v2" or provenance.get("training_scope") != "third_outer_train_only":
        raise ValueError("지원하는 V2 고정 센서 저장 묶음이 아닙니다.")
    
    target = provenance["target_recall"]

    if target not in (0.8, 0.9):
        raise ValueError("V2 목표 시나리오는 80% 또는 90%여야 합니다.")
    
    threshold, names, execution, _ = scenario_source(directory / "experiment", target)

    if (execution != provenance["oof_execution"] or names != manifest["sensors"]
            or names != list(inference.sensors) or threshold != manifest["threshold"]
            or threshold != inference.threshold or inference.positive_label != manifest["positive_label"]):
        raise ValueError("저장 객체·manifest·보존 OOF 계약이 다릅니다.")
    
    if dataset_contract(asdict(config.dataset)) != dataset_contract(execution["config"]["dataset"]):
        raise ValueError("현재 데이터 정의가 저장 OOF의 데이터 정의와 다릅니다.")
    
    return execution, target


def main():
    """사전 검증 후 한 번 평가하고 원래 묶음의 실행 표식과 결과를 보존한다."""
    parser = argparse.ArgumentParser(description="V2의 기존 V1 Test 구간 비교 평가 (재학습 없음)")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--v1-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trusted-local-bundle", action="store_true")
    parser.add_argument("--confirm-frozen-evaluation", action="store_true")
    args = parser.parse_args()

    if not args.confirm_frozen_evaluation:
        parser.error("--confirm-frozen-evaluation으로 기존 Test 비교 평가를 확인하세요.")

    if (args.bundle_dir / "evaluation_started.json").exists():
        raise ValueError("이 묶음의 평가 기록이 이미 있습니다. 자동 재실행하지 않습니다.")
    
    payload = load_sensor_bundle(args.bundle_dir, trusted=args.trusted_local_bundle)
    inference = payload["inference"]
    
    manifest = json.loads((args.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    config = load_modeling_config(args.config)
    
    execution, target = validate_v2_bundle(manifest, inference, args.bundle_dir, config)
    train, test = read_sensor_csv(args.train), read_sensor_csv(args.test)

    v1 = json.loads((args.v1_result_dir / "evaluation.json").read_text(encoding="utf-8"))
    
    if v1.get("status") != "completed":
        raise ValueError("완료된 V1 평가 기록이 필요합니다.")
    
    protocol = v1["training_protocol_id"]
    features, labels = validate_final_pair(train, test, config.dataset, protocol)

    validate_v1_rows(test, pd.read_csv(args.v1_result_dir / "test_predictions.csv"))
    
    # V2 저장 당시의 마지막 개발 구간 ID도 대조해 다른 Train을 조용히 허용하지 않는다.
    fit_i, valid_i = temporal_folds(train, config.dataset, 3)[2]
    previous = pd.read_csv(args.bundle_dir / "experiment" / "future_predictions.csv")

    if train.iloc[valid_i][SOURCE_ROW_ID].tolist() != previous.source_row_id.tolist():
        raise ValueError("Train의 개발 구간이 V2 보존 기록과 다릅니다.")
    
    inference._prepare_input(features.loc[:, list(inference.sensors)])

    record = {"status": "started", "evaluation_scope": "existing_v1_test_comparison",
              "historical_nonuse_guaranteed": False, "refit": False,
              "target_recall": target, "threshold": inference.threshold,
              "sensors": list(inference.sensors), "training_protocol_id": protocol,
              "train_path": str(args.train.resolve()), "test_path": str(args.test.resolve()),
              "original_train_path": execution["train_path"],
              "bundle_dir": str(args.bundle_dir.resolve()), "model_fit_rows": len(fit_i),
              "v1_result_dir": str(args.v1_result_dir.resolve())}
    
    marker = reserve_evaluation(args.bundle_dir, args.output_dir, record)
    metrics, predictions = evaluate_saved_predictions(inference, features, labels)

    # 80%/90%는 OOF 선택 목표다. 과거의 40% 선별 상한 정책을 승계하지 않는다.
    metrics["target_recall"] = target
    metrics["recall_target_met"] = metrics["recall"] >= target

    save_evaluation_results(args.output_dir, marker, record, metrics, predictions, test, labels,
                            {"min_recall": target, "max_reinspection_ratio": None})
    
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"기존 Test 비교 보고서: {args.output_dir / 'evaluation.md'}")
    print("V1과 같은 평가 행입니다. 독립 미사용 Test가 아니며 재학습·문턱 변경을 하지 않았습니다.")


if __name__ == "__main__":
    main()

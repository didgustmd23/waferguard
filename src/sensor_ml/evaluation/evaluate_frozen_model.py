# ==========================================
# 동결한 연구·데모 모델의 후속 시간 구간 평가
# - 저장 모델을 복원하고 fit·센서 재선택·문턱 탐색 없이 평가
# - Train/Test 역할·원본 ID·생성 계약·시간 경계를 먼저 확인
# - 과거 완전한 미사용 여부는 보증하지 않는다고 결과에 명시
# - 묶음별 실행 표식을 남겨 새 출력 경로를 통한 반복 실행도 차단
# ==========================================

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.dataset_schema import split_frame_to_xy
from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_metrics import evaluate_binary_scores
from src.sensor_ml.inference.predict_cli import read_sensor_csv
from src.sensor_ml.inference.sensor_bundle import load_sensor_bundle
from src.split_contract import SPLIT_METADATA, SPLIT_ROLE, PROTOCOL_ID, validate_split_pair


def validate_final_pair(train, test, dataset, protocol):
    """새 계약의 Train과 Test만 허용하고 시간·원본 ID를 대조한다."""

    for frame in (train, test):
        if not set(SPLIT_METADATA).issubset(frame.columns):
            raise ValueError("최종 평가에는 split metadata를 포함한 Train과 Test가 필요합니다.")
        
        if frame.empty or frame[PROTOCOL_ID].isna().any() or set(frame[PROTOCOL_ID]) != {protocol}:
            raise ValueError("최종 평가의 생성 계약이 저장 모델과 다릅니다.")
        
    if test[SPLIT_ROLE].isna().any() or set(test[SPLIT_ROLE]) != {"test"}:
        raise ValueError("평가 입력에는 test 역할의 split만 사용하세요.")
    
    # 원본 역할은 바꾸지 않고 검증기에 전달할 복사본만 validation으로 만든다.
    checking = test.copy()
    checking[SPLIT_ROLE] = "validation"

    validate_split_pair(train, checking, dataset, temporal=True)
    _, _, columns = split_frame_to_xy(train, dataset)
    features, labels, _ = split_frame_to_xy(test, dataset, expected_feature_columns=columns)

    if labels.nunique() != 2:
        raise ValueError("최종 평가에는 정상·불량 label이 모두 필요합니다.")
    
    return features, labels


def evaluate_saved_predictions(inference, features, labels):
    
    """저장된 센서와 문턱으로 전체 행을 평가하고 선별 비율을 기록한다."""
    missing = set(inference.sensors) - set(features.columns)

    if missing:
        raise ValueError(f"평가 입력에 필수 센서가 없습니다: {sorted(missing)}")
    
    predictions = inference.predict(features.loc[:, list(inference.sensors)])
    
    metrics = evaluate_binary_scores(labels, predictions.positive_score,
                                    positive_label=inference.positive_label,
                                    negative_label=inference._negative_label,
                                    threshold=inference.threshold).to_dict()
    
    metrics["reinspection_ratio"] = (metrics["true_positive"] + metrics["false_positive"]) / metrics["support"]

    return metrics, predictions


def reserve_evaluation(bundle_dir, output_dir, record):
    
    """저장 모델별 실행 시작을 배타적으로 기록하고 반복 실행을 거부한다."""
    marker = Path(bundle_dir) / "evaluation_started.json"

    if marker.exists():
        raise ValueError("이 묶음에는 평가 실행 기록이 이미 있습니다. 자동 재실행하지 않습니다.")
    
    directory = Path(output_dir)

    if directory.exists() and any(directory.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 기존 결과를 덮어쓰지 않습니다.")
    
    directory.mkdir(parents=True, exist_ok=True)

    # x 모드는 동시에 실행된 프로세스도 같은 묶음을 평가하지 못하게 한다.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        
    return marker


def parse_args(argv=None):
    """평가 경로와 명시적 승인 옵션을 읽고 실행 전 조건을 확인한다."""
    parser = argparse.ArgumentParser(description="동결 모델의 연구·데모 시간 구간 평가 (재학습 없음)")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="동결 후보 설정과 연결된 기존 시간 검증 execution.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trusted-local-bundle", action="store_true")
    parser.add_argument("--confirm-frozen-evaluation", action="store_true",
                        help="동결 조건과 과거 미사용 여부 보증 불가를 확인하고 한 번 평가합니다.")
    args = parser.parse_args(argv)
    
    if not args.confirm_frozen_evaluation:
        raise ValueError("동결 평가 승인 확인 옵션이 필요합니다. 아직 평가하지 않았습니다.")
    
    if (args.bundle_dir / "evaluation_started.json").exists():
        raise ValueError("이 후보 묶음은 평가 실행 기록이 이미 있어 재실행하지 않습니다.")
    
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("출력 폴더가 비어 있지 않습니다.")
    
    return args


def validate_frozen_context(config, manifest, reference):
    """동결 당시 설정 연결을 항목별로 대조하고 불일치 원인을 안내한다."""
    dataset_snapshot = json.loads(json.dumps(asdict(config.dataset), default=str))
    provenance = manifest["provenance"]

    # 현재 설정을 저장 객체에 적용하지 않는다. 비교 대상과 기존 검사 순서는 유지한다.
    checks = (
        ("데이터 정의", dataset_snapshot, reference["config"]["dataset"]),
        ("모델 설정", provenance["model_parameters"], reference["model_parameters"]),
        ("RF 선택기 설정", provenance["rf_selector_parameters"], reference["rf_selector_parameters"]),
        ("학습 생성 계약", provenance["training_protocol_id"], reference["training_protocol_id"]),
        ("문턱", manifest["threshold"], reference["threshold"]),
        ("양성 label", manifest["positive_label"], config.dataset.positive_label),
        ("선택 센서", manifest["sensors"], provenance["selected_sensors"]),
    )

    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"동결 후보와 기존 실행 기록의 연결이 다릅니다: {name}")


def load_evaluation_context(args):
    """신뢰한 저장 묶음과 기준 실행 기록을 읽고 동결 설정을 검증한다."""
    
    # 역직렬화 승인·버전·묶음 계약 검사는 기존 loader에서 그대로 수행한다.
    payload = load_sensor_bundle(args.bundle_dir, trusted=args.trusted_local_bundle)
    manifest = json.loads((args.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    reference = json.loads(args.reference_run.read_text(encoding="utf-8"))
    config = load_modeling_config(args.config)

    validate_frozen_context(config, manifest, reference)

    return payload["inference"], manifest["provenance"], reference, config


def prepare_evaluation_input(args, inference, provenance, dataset):
    
    """경로·행 수·split·추론 입력을 확인하며 실행 표식 생성 전에 실패를 차단한다."""
    if args.train.resolve() != Path(provenance["train_path"]).resolve():
        raise ValueError("Train 경로가 저장 후보의 학습 출처와 다릅니다.")
    
    if args.test.resolve() == args.train.resolve():
        raise ValueError("Train과 평가 입력 경로는 달라야 합니다.")
    
    train, test = read_sensor_csv(args.train), read_sensor_csv(args.test)

    if len(train) != provenance["input_rows"]:
        raise ValueError("Train 행 수가 저장 후보의 학습 기록과 다릅니다.")
    
    features, labels = validate_final_pair(train, test, dataset, provenance["training_protocol_id"])

    # 전체 결측 등 입력 계약 오류는 실행 시작 전에 차단한다. 행을 제외하지 않는다.
    inference._prepare_input(features.loc[:, list(inference.sensors)])
    
    return features, labels, test


def build_evaluation_record(args, inference, provenance):
    """평가 시작 시점의 출처·고정 문턱·센서와 평가 범위를 기록한다."""
    return {"status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "bundle_dir": str(args.bundle_dir.resolve()), "train_path": str(args.train.resolve()),
              "test_path": str(args.test.resolve()), "output_dir": str(args.output_dir.resolve()),
              "threshold": inference.threshold, "sensors": list(inference.sensors),
              "training_protocol_id": provenance["training_protocol_id"],
              "evaluation_scope": "existing_data_later_time_period",
              "historical_nonuse_guaranteed": False, "refit": False}


def save_evaluation_results(output_dir, marker, record, metrics, predictions, test, labels, policy):
    """CSV·완료 기록·보고서를 기존 이름과 순서로 저장한다."""
    
    # 파일 형식·열 순서·완료 표식 갱신 시점은 리팩토링 전 동작을 유지한다.
    pd.DataFrame([metrics]).to_csv(output_dir / "test_result.csv", index=False, encoding="utf-8-sig")
    
    predictions.insert(0, "source_row_id", test["__source_row_id"].to_numpy())
    predictions.insert(1, "label", labels.to_numpy())
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    record.update(status="completed", metrics=metrics, threshold_policy=policy,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    
    content = json.dumps(record, ensure_ascii=False, indent=2)

    (output_dir / "evaluation.json").write_text(content, encoding="utf-8")

    marker.write_text(content, encoding="utf-8")

    report = ["# 동결 모델의 후속 시간 구간 평가", "",
              "기존 데이터의 후속 시간 구간 평가입니다. 과거 완전한 미사용 여부는 보증하지 못합니다.",
              "현장 배포 승인이 아니며 재학습·센서 재선택·threshold 탐색을 하지 않았습니다.", "",
              "```text", pd.DataFrame([metrics]).to_string(index=False), "```", "",
              "목표 미달도 그대로 기록하며 결과에 맞춰 동결한 모델·문턱·센서를 변경하지 않습니다."]
    
    (output_dir / "evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    """검증 → 실행 예약 → 고정 평가 → 저장 순서로 한 번 평가한다."""
    args = parse_args()
    inference, provenance, reference, config = load_evaluation_context(args)
    features, labels, test = prepare_evaluation_input(args, inference, provenance, config.dataset)
    record = build_evaluation_record(args, inference, provenance)
    marker = reserve_evaluation(args.bundle_dir, args.output_dir, record)
    
    # 표식 이후 오류가 발생해도 삭제하거나 자동 재시도하지 않는다.
    metrics, predictions = evaluate_saved_predictions(inference, features, labels)
    
    # 현재 정책이 아닌 기준 실행 기록의 동결 정책으로 판정한다.
    policy = reference["config"]["threshold_policy"]
    metrics["policy_feasible"] = (metrics["recall"] >= policy["min_recall"]
                                  and metrics["reinspection_ratio"] <= policy["max_reinspection_ratio"])
    
    save_evaluation_results(args.output_dir, marker, record, metrics, predictions, test, labels, policy)
    
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"연구·데모 평가 보고서: {args.output_dir / 'evaluation.md'}")
    print("과거 완전한 미사용 여부 보증 불가. 결과에 맞춘 재조정·자동 재실행은 하지 않습니다.")


if __name__ == "__main__":
    main()

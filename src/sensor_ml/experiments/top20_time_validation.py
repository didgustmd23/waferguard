# ==========================================
# 기존 S0·M3 단일 OOF 정책 후보를 시간순 Validation에 적용
# - OOF는 다시 학습하지 않고 저장된 확률에서 현재 정책으로 문턱을 확인
# - 실행 설정·Train 행·모델 파라미터가 맞을 때만 전체 Train으로 학습
# - Validation에서 문턱을 재탐색하지 않으며 Test 입력은 제공하지 않음
# - 기존 결과를 보호하기 위해 새 출력 폴더만 사용
# ==========================================

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_models import build_pipeline
    from src.dataset_schema import split_frame_to_xy
    from src.split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role
    from src.sensor_ml.experiments.step8_threshold_oof import compare_thresholds
    from src.sensor_ml.experiments.step7_time_validation import compare_time_validation
    from src.threshold_policy import select_policy_threshold, validate_threshold_report
    from src.sensor_ml.diagnostics.score_diagnostics import write_score_diagnostics
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_models import build_pipeline
    from dataset_schema import split_frame_to_xy
    from split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role
    from src.sensor_ml.experiments.step8_threshold_oof import compare_thresholds
    from src.sensor_ml.experiments.step7_time_validation import compare_time_validation
    from threshold_policy import select_policy_threshold, validate_threshold_report
    from diagnostics.score_diagnostics import write_score_diagnostics


# ==========================================
# 저장된 OOF와 현재 Train으로 평가 후보 복원
# ==========================================
def prepare_candidate(train, oof, record, config, *, threshold, n_jobs=1):
    
    """저장된 설정·행을 확인하고 학습 전 Pipeline과 후보 표를 반환한다."""
    validate_train_role(train)
    
    # 정책 변경만 허용하며 데이터 정의·전처리·난수 등의 변경은 OOF 재생성이 필요하다.
    # JSON에 저장될 때 tuple은 list가 되므로 같은 직렬화 형식으로 비교한다.
    current = json.loads(json.dumps(asdict(config), default=str))
    previous = dict(record["config"])
    current.pop("threshold_policy", None)
    previous.pop("threshold_policy", None)

    if current != previous or record.get("oof_score_method") != "single" or record.get("n_repeats") != 1:
        raise ValueError("OOF 실행 설정이 현재 설정과 다릅니다. 같은 설정으로 OOF를 다시 생성하세요.")
    
    if PROTOCOL_ID not in train or record.get("training_protocol_id") != train[PROTOCOL_ID].iloc[0]:
        raise ValueError("OOF 실행 기록과 Train의 생성 계약이 일치하지 않습니다.")
    
    models = [row for row in record["models"] if row["model_variant"] == "M3"]

    if len(models) != 1:
        raise ValueError("OOF 실행 기록에는 M3 후보가 정확히 하나 있어야 합니다.")
    
    model_record = models[0]
    experiment = model_record["experiment"]
    model_params = model_record["model_parameters"]
    rf_params = model_record["rf_selector_parameters"]
    count = config.reduction_policy.max_sensor_count

    if (experiment != f"s0_m3_rf_top_{count}" or model_params.get("max_depth") != 2
            or model_params.get("reg_lambda") != 5 or rf_params.get("min_samples_leaf") != 1
            or rf_params.get("max_depth") is not None):
        raise ValueError("OOF 실행 기록의 S0·M3 이름과 선택기·모델 설정이 일치하지 않습니다.")
    
    rows = oof.loc[oof.model_variant.eq("M3")].reset_index(drop=True)

    _, labels, _ = split_frame_to_xy(train, config.dataset)

    # 같은 생성 계약이라도 행 순서·label 변경이나 중복 예측이 있으면 재사용하지 않는다.
    if (len(rows) != len(train) or not rows.source_row_index.eq(np.arange(len(train))).all()
            or not np.array_equal(rows.source_row_id.to_numpy(), train[SOURCE_ROW_ID].to_numpy())
            or not np.array_equal(rows.label.to_numpy(), labels.to_numpy())
            or not rows.oof_prediction_count.eq(1).all()
            or not rows.experiment.eq(experiment).all()):
        raise ValueError("저장된 OOF와 현재 Train 행·label·모델이 일치하지 않습니다.")
    
    # 저장된 확률에서 출처 포함 후보 표를 재계산하며 과거 정책 충족 여부는 사용하지 않는다.
    table = compare_thresholds(rows, config, experiment_name=experiment,
                               thresholds=np.unique(np.r_[0., rows.oof_positive_score, 1.]))
    
    chosen = select_policy_threshold(table, config.threshold_policy)

    if chosen is None:
        raise ValueError("현재 정책을 만족하는 OOF threshold 후보가 없습니다.")
    
    if not np.isfinite(threshold) or threshold != chosen:
        raise ValueError(f"지정한 threshold가 현재 OOF 정책 선택값과 다릅니다. 선택값: {chosen!r}")
    
    validate_threshold_report(table, train, config, experiment, threshold)
    pipeline = build_pipeline(config, f"xgboost_rf_top_{count}", n_jobs=n_jobs)

    # OOF 당시 파라미터를 복원하며, 계산 병렬 수만 이번 실행 값으로 변경한다.
    pipeline.named_steps["model"].set_params(**model_params)
    pipeline.named_steps["selector"].estimator.set_params(**rf_params)
    pipeline.set_params(model__n_jobs=n_jobs, selector__estimator__n_jobs=n_jobs)

    return experiment, pipeline, table


# ==========================================
# 전체 센서 경로에 같은 M3 분류기 설정을 적용
# - 품질 필터·중앙값 대치는 유지하고 RF Top-K 선택만 제외
# - 문턱은 Top-20 OOF에서 가져온 공통 진단값이며 전체 모델의 운영 문턱이 아님
# - 순위 지표인 AP·ROC-AUC로 센서 축소 영향을 먼저 비교
# ==========================================
def compare_all_sensors(train, validation, config, top20_pipeline, *, threshold, n_jobs=1):
    
    """선택기 없는 M3 전체 특징 모델을 학습하고 시간 검증 결과·설정을 반환한다."""
    pipeline = build_pipeline(config, "xgboost_all", n_jobs=n_jobs)

    # 깊이·정규화뿐 아니라 모든 분류기 파라미터를 복사해 모델 설정 차이를 없앤다.
    pipeline.named_steps["model"].set_params(**top20_pipeline.named_steps["model"].get_params())

    if "selector" in pipeline.named_steps:
        raise ValueError("전체 센서 비교 Pipeline에는 특징 선택기가 없어야 합니다.")
    
    name = "m3_all_sensors"

    result = compare_time_validation(train, validation, config, experiment_names=(name,),
                                     threshold=threshold, n_jobs=n_jobs,
                                     pipeline_templates={name: pipeline})
    
    result["reinspection_ratio"] = (result.true_positive + result.false_positive) / result.support
    result["threshold_source"] = "top20_oof_diagnostic_only"

    return result, pipeline.named_steps["model"].get_params()


def save_sensor_comparison(top20, whole, output_dir):
    """두 경로의 시간 검증 지표를 저장하고 문턱 해석 한계를 문서에 명시한다."""
    
    # attrs의 행별 목록은 따로 저장한다. pandas concat의 목록 동등 비교는 피한다.
    frames = []

    for result, role in ((whole, "all_sensors"), (top20, "rf_top20")):
        frame = result.copy()
        frame.attrs = {}
        frame["comparison_role"] = role
        frames.append(frame)

    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(output_dir / "sensor_compare.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(whole.attrs["predictions"]).to_csv(output_dir / "all_validation_predictions.csv",
                                                 index=False, encoding="utf-8-sig")
    
    all_ap, top_ap = float(whole.average_precision.iloc[0]), float(top20.average_precision.iloc[0])

    loss = f"{(all_ap - top_ap) / all_ap:.2%}" if all_ap > 0 else "계산 불가: 전체 모델 AP가 0"
    
    columns = ["comparison_role", "selected_feature_count", "average_precision", "roc_auc",
               "threshold", "recall", "reinspection_ratio", "true_positive", "false_positive", "false_negative"]
    
    report = ["# 동일 M3 설정의 전체 센서·RF Top-20 시간 검증 비교", "",
              "```text", comparison[columns].to_string(index=False), "```", "",
              f"전체 모델 대비 Top-20 AP 상대 하락률: {loss}. 음수면 Top-20 AP가 더 높다는 뜻입니다.", "",
              "- 전체 센서 모델도 Train 품질 필터를 통과한 센서만 사용합니다. 원본 590개를 그대로 사용하는 뜻은 아닙니다.",
              "- 동일 Train·Validation·분류기 설정에서 RF Top-20 선택 여부만 비교합니다.",
              "- AP·ROC-AUC는 문턱과 무관합니다. 센서 축소 영향을 확인할 우선 지표입니다.",
              "- 공통 문턱은 Top-20 OOF에서 선택한 진단값입니다. 전체 센서 모델의 OOF에서 선택한 운영 문턱이 아니므로 Recall·양성 비율의 공정한 운영 비교로 해석하지 않습니다.",
              "- 전체 센서 모델도 저하되면 센서 축소만으로 설명할 수 없습니다. 반대로 전체 모델만 유지되면 선택 센서의 일반화 문제를 추가 확인합니다.",
              "- Validation으로 문턱을 재탐색하지 않았으며 최종 Test는 사용하지 않았습니다."]
    
    (output_dir / "sensor_compare.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    return comparison


def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="S0·M3 OOF 문턱의 시간순 Validation 확인")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true", help="확률 분포·센서 교집합·진단 문서를 생성합니다.")
    parser.add_argument("--compare-all", action="store_true", help="같은 M3 설정의 전체 센서 모델도 시간 검증합니다.")
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/secom/figures/v1/m3_policy80_40_diagnostics"))
    args = parser.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    
    if args.diagnostics and (args.figure_dir / "score_distribution.png").exists():
        raise ValueError("진단 그림이 이미 있습니다. 새 --figure-dir을 지정하세요.")
    
    return args


def save_results(args, config, record, experiment, pipeline, result, whole_params):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "time_validation.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(result.attrs["selected_features"]).to_csv(args.output_dir / "selected_features.csv",
                                                         index=False, encoding="utf-8-sig")
    
    # 기존 실행에는 없던 행별 예측을 보존하여 후속 분석에서 다시 학습할 필요를 줄인다.
    predictions = pd.DataFrame(result.attrs["predictions"])
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    execution = {"config": asdict(config), "threshold": args.threshold, "experiment": experiment,
                 "train_path": str(args.train.resolve()), "validation_path": str(args.validation.resolve()),
                 "oof_dir": str(args.oof_dir.resolve()), "training_protocol_id": record["training_protocol_id"],
                 "model_parameters": pipeline.named_steps["model"].get_params(),
                 "rf_selector_parameters": pipeline.named_steps["selector"].estimator.get_params(),
                 "all_model_parameters": whole_params,
                 "compare_all": args.compare_all,
                 "is_final_test": False, "threshold_source": "train_single_oof_policy"}
    
    (args.output_dir / "execution.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    
    return predictions


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    train = pd.read_csv(args.train)
    record = json.loads((args.oof_dir / "oof_run.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(args.oof_dir / "oof_predictions.csv", float_precision="round_trip")
    
    experiment, pipeline, table = prepare_candidate(train, oof, record, config,
                                                    threshold=args.threshold, n_jobs=args.n_jobs)
    
    # 진단에 필요한 과거 센서 기록은 학습 전에 읽어 누락으로 인한 불필요한 재학습을 피한다.
    oof_sensors = pd.read_csv(args.oof_dir / "selected_features.csv") if args.diagnostics else None

    # 시간 순서·split 역할·행 중복은 기존 평가 코어에서 학습 전에 검사한다.
    validation = pd.read_csv(args.validation)
    result = compare_time_validation(train, validation, config,
                                    experiment_names=(experiment,), threshold=args.threshold,
                                    threshold_report=table, n_jobs=args.n_jobs,
                                    pipeline_templates={experiment: pipeline})
    
    result["reinspection_ratio"] = (result.true_positive + result.false_positive) / result.support

    result["policy_feasible"] = ((result.recall >= config.threshold_policy.min_recall)
                                 & (result.reinspection_ratio <= config.threshold_policy.max_reinspection_ratio))
    
    result["threshold_source"] = "train_single_oof_policy"

    # 추가 모델 역시 OOF 재생성 없이 같은 분할에서만 학습·평가한다.
    whole, whole_params = compare_all_sensors(train, validation, config, pipeline,
                                             threshold=args.threshold, n_jobs=args.n_jobs) if args.compare_all else (None, None)
    
    predictions = save_results(args, config, record, experiment, pipeline, result, whole_params)

    if args.diagnostics:
        write_score_diagnostics(oof.loc[oof.model_variant.eq("M3")], predictions,
                                oof_sensors.loc[oof_sensors.model_variant.eq("M3")],
                                pd.DataFrame(result.attrs["selected_features"]), config, args.threshold,
                                args.output_dir, args.figure_dir)
        
    if whole is not None:
        comparison = save_sensor_comparison(result, whole, args.output_dir)
        print(comparison[["comparison_role", "selected_feature_count", "average_precision", "roc_auc",
                          "recall", "reinspection_ratio", "true_positive", "false_positive", "false_negative"]].to_string(index=False))
        print("전체 모델의 문턱은 Top-20 OOF 값을 공유한 진단값입니다. 운영 문턱을 확정하지 않았습니다.")
    else:
        print(result[["threshold", "recall", "reinspection_ratio", "average_precision",
                      "true_positive", "false_positive", "false_negative", "policy_feasible"]].to_string(index=False))
        
    print(f"시간 검증 결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

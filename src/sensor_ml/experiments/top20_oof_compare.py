# ==========================================
# 같은 S0 센서 선택기로 M0·M3 단일 OOF와 문턱 시나리오 비교
# - 각 Train 샘플은 단일 5-fold에서 한 번만 미학습 예측을 받음
# - 품질 필터·대치·RF Top-K 선택은 학습 fold 안에서만 수행
# - 같은 Recall 목표의 최소 양성 비율과 같은 상한의 최대 Recall 비교
# - 시나리오 문턱은 비교용 후보이며 최종 정책을 자동 변경하지 않음
# - 외부 Validation·Test를 읽지 않고 전체 Train 모델도 학습하지 않음
# ==========================================

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.modeling_models import build_pipeline, configure_topk_xgb
    from src.modeling_preprocessing import quality_filter_json
    from src.sensor_ml.experiments.step8_threshold_oof import generate_oof_scores, compare_thresholds
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from modeling_models import build_pipeline, configure_topk_xgb
    from modeling_preprocessing import quality_filter_json
    from src.sensor_ml.experiments.step8_threshold_oof import generate_oof_scores, compare_thresholds


# ==========================================
# 문턱 후보에서 공통 시나리오별 대표 비교 행 선택
# - Recall 목표: 목표 이상 중 양성 비율 최소 → Recall 최대 → 문턱 최대
# - 양성 상한: 상한 이하 중 Recall 최대 → 양성 비율 최소 → 문턱 최대
# - 동률 규칙을 기록하고 미충족을 이웃 후보로 대체하지 않음
# ==========================================
def scenario_rows(table, recall_targets, alarm_limit):
    rows = []
    scenarios = [("recall_target", target) for target in recall_targets]
    scenarios.append(("alarm_limit", alarm_limit))

    for kind, target in scenarios:
        if kind == "recall_target":
            feasible = table.loc[table.recall >= target]
            order, ascending = ["reinspection_ratio", "recall", "threshold"], [True, False, False]
        else:
            feasible = table.loc[table.reinspection_ratio <= target]
            order, ascending = ["recall", "reinspection_ratio", "threshold"], [False, True, False]

        metadata = {"scenario": kind, "target": target, "is_final_threshold": False}

        if feasible.empty:
            rows.append({**metadata, "status": "infeasible", "threshold": None})
        else:
            chosen = feasible.sort_values(order, ascending=ascending).iloc[0].to_dict()
            rows.append({**chosen, **metadata, "status": "candidate"})

    return rows


# ==========================================
# M0·M3의 단일 OOF와 같은 목표별 비교 자료 생성
# - max_sensor_count는 설정에서 읽어 다른 데이터셋에서도 센서 수를 변경 가능
# - 문턱 후보는 모든 고유 OOF 확률을 사용해 분위수 격자 누락을 피함
# - 0·1도 후보에 포함하되 동점 확률이 있으면 목표 Recall을 초과할 수 있음
# - 선택 후보의 문턱을 비교 대상 모델에 서로 복사하지 않음
# ==========================================
def compare_top20_oof(frame, config, *, n_jobs=1, recall_targets=(0.7, 0.8, 0.9), alarm_limit=0.2):
    for value in (*recall_targets, alarm_limit):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Recall 목표와 양성 비율 상한은 0~1의 유한한 숫자여야 합니다.")
        
    if not recall_targets or len(set(recall_targets)) != len(recall_targets):
        raise ValueError("Recall 목표는 중복 없이 하나 이상 지정해야 합니다.")
    
    if config.reduction_policy is None:
        raise ValueError("센서 수를 결정하려면 feature_selection.reduction_policy 설정이 필요합니다.")
    
    count = config.reduction_policy.max_sensor_count

    if count not in config.top_k_feature_counts:
        raise ValueError("센서 수 상한이 top_k_feature_counts에 포함되어야 합니다.")
    
    pipeline_name = f"xgboost_rf_top_{count}"
    results = {name: [] for name in ("oof_predictions", "threshold_compare", "scenario_compare",
                                    "quality_filter", "selected_features")}
    
    run_records = []
    reference = None
    reference_sensors = None

    for variant, depth, reg_lambda in (("M0", 3, 1.0), ("M3", 2, 5.0)):
    
        # 두 경로 모두 새 S0 선택기를 만들고 최종 XGBoost의 설정만 바꾼다.
        pipeline = build_pipeline(config, pipeline_name, n_jobs=n_jobs)
        configure_topk_xgb({pipeline_name: pipeline}, max_depth=depth, reg_lambda=reg_lambda)
        experiment = f"s0_{variant.lower()}_rf_top_{count}"

        oof, elapsed = generate_oof_scores(frame, config, n_jobs=n_jobs, score_method="single",
                                          experiment_name=experiment, pipeline_template=pipeline)
        
        # 예측 행·정답·fold가 일치해야 같은 표본의 공정한 비교가 된다.
        identity = oof[["source_row_index", "source_row_id", "label", "cv_fold"]]

        if reference is None:
            reference = identity.copy()
        elif not identity.equals(reference):
            raise RuntimeError("M0·M3의 OOF 행·label·fold가 일치하지 않습니다.")
        
        # 같은 S0 선택기를 썼는지 실제 학습 결과도 비교해 모델 설정 효과와 구분한다.
        sensor_rows = pd.DataFrame(oof.attrs["selected_features"])

        if sensor_rows.empty or not sensor_rows.groupby("cv_fold").size().eq(count).all():
            raise RuntimeError("각 OOF 학습 fold의 선택 센서 수가 설정과 일치하지 않습니다.")
        
        sensor_rows = sensor_rows.sort_values(["cv_fold", "feature"]).reset_index(drop=True)
        
        if reference_sensors is None:
            reference_sensors = sensor_rows.copy()
        elif not sensor_rows.equals(reference_sensors):
            raise RuntimeError("M0·M3의 S0 선택 센서 목록이 일치하지 않습니다.")
        
        thresholds = np.unique(np.r_[0.0, oof.oof_positive_score.to_numpy(), 1.0])
        table = compare_thresholds(oof, config, thresholds=thresholds, experiment_name=experiment)
        table["model_variant"] = variant

        # 기존 정책의 충족 여부만 덧붙이며 설정값을 바꾸거나 문턱을 확정하지 않는다.
        table["policy_feasible"] = ((table.recall >= config.threshold_policy.min_recall)
                                     & (table.reinspection_ratio <= config.threshold_policy.max_reinspection_ratio))
        
        metadata = {"experiment": experiment, "model_variant": variant}

        results["scenario_compare"].extend({**row, **metadata} for row in scenario_rows(table, recall_targets, alarm_limit))
        results["quality_filter"].extend({**row, **metadata} for row in oof.attrs["quality_filter_records"])
        results["selected_features"].extend({**row, **metadata} for row in oof.attrs["selected_features"])

        # pandas의 attrs에는 목록이 있어 concat 비교가 모호해질 수 있으므로 저장용 복사본에서 제거한다.
        stored_oof = oof.copy()
        stored_oof.attrs = {}

        results["oof_predictions"].append(stored_oof.assign(**metadata))
        results["threshold_compare"].append(table)

        run_records.append({**metadata, "elapsed_seconds": elapsed,
                            "model_parameters": pipeline.named_steps["model"].get_params(),
                            "rf_selector_parameters": pipeline.named_steps["selector"].estimator.get_params()})
        
    tables = {key: pd.concat(value, ignore_index=True) 
              if key in {"oof_predictions", "threshold_compare"}
              else pd.DataFrame(value) for key, value in results.items()}
    
    # 단일 OOF AP는 전체 행의 예측을 합친 값이며 반복 CV의 평균 AP와 구분한다.
    tables["oof_summary"] = tables["threshold_compare"].groupby("model_variant", as_index=False).first()[
        ["model_variant", "support", "positive_support", "average_precision", "roc_auc"]]
    
    return tables, run_records


# ==========================================
# 같은 Train의 비교 자료를 새 결과 폴더에 저장
# - 비어 있지 않은 폴더에 쓰지 않아 기존 실험을 보호
# - 실행 설정과 실제 모델 파라미터를 JSON에 저장
# ==========================================
def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="S0 RF 센서 선택을 유지한 M0·M3 단일 OOF 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--recall-targets", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--alarm-limit", type=float, default=0.2)
    args = parser.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("결과 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    
    return args


def save_results(args, config, tables, records):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        stored = table.copy()
        if name == "quality_filter":
            for column in ("high_missing_features", "constant_features"):
                stored[column] = stored[column].map(quality_filter_json)
        stored.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    protocol = tables["oof_predictions"].training_protocol_id.iloc[0]

    record = {"config": asdict(config), "train_path": str(args.train.resolve()),
              "training_protocol_id": protocol, "n_jobs": args.n_jobs,
              "oof_score_method": "single", "n_splits": config.experiment.cv.n_splits, "n_repeats": 1,
              "recall_targets": args.recall_targets, "alarm_limit": args.alarm_limit,
              "is_final_threshold": False, "models": records,
              "versions": {name: version(name) for name in ("scikit-learn", "xgboost")}}
    
    (args.output_dir / "oof_run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    tables, records = compare_top20_oof(pd.read_csv(args.train), config, n_jobs=args.n_jobs,
                                      recall_targets=tuple(args.recall_targets), alarm_limit=args.alarm_limit)
    
    save_results(args, config, tables, records)

    columns = ["model_variant", "scenario", "target", "status", "threshold", "recall", "reinspection_ratio",
               "true_positive", "false_positive", "false_negative", "policy_feasible"]
    
    print(tables["scenario_compare"].reindex(columns=columns).to_string(index=False))
    print("시나리오 문턱은 비교 후보이며 최종 threshold를 확정하지 않았습니다.")
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

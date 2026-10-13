# ==========================================
# 팀원 LightGBM 선택 후보와 기존 V2의 시간순 비교
# - 같은 Time Train의 외부 3구간·내부 시간순 OOF 2구간 사용
# - 기존 전체 센서 기준과 V2 반복 gain Top-20을 함께 재현
# - 두 Top-20 분류기는 깊이 2·규제 1·비율 가중치로 통일
# - 모든 센서 선택은 해당 학습 fold 안에서만 fit
# - 기존 Test·외부 Validation·저장 V2 모델은 사용하거나 변경하지 않음
# ==========================================
import argparse
import json
from pathlib import Path

import pandas as pd

from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.sensor_ml.experiments.feature_time_compare import build_feature_pipelines
from src.sensor_ml.experiments.lightgbm_gain_candidate import build_gain_candidate
from src.sensor_ml.experiments.temporal_validation import compare_temporal, save_results

DEFAULT_PRESET = Path(__file__).resolve().parents[3] / "configs/experiments/team_gain_time_compare.json"


def build_candidates(config, preset, *, n_jobs=1):
    """기존 후보를 유지하고 같은 최종 분류기 조건의 LightGBM 선택 후보를 추가한다."""

    # 기존 V2의 반복 weighted gain과 전체 기준을 공통 생성기로 만든다.
    previous = build_feature_pipelines(
        config, n_jobs=n_jobs, n_estimators=preset["n_estimators"],
        experiment_names=("xgboost_all", f"xgboost_top_{preset['top_k']}"),
        xgb_weight_mode=preset["class_weight_mode"],
        xgb_selector_weight_mode=preset["class_weight_mode"],
        xgb_stability_repeats=preset["stable_gain_repeats"],
        topk_xgb_max_depth=preset["topk_classifier_max_depth"],
        topk_xgb_reg_lambda=preset["topk_classifier_reg_lambda"],
    )

    team = build_gain_candidate(config, n_jobs=n_jobs, top_k=preset["top_k"])

    # 앞선 랜덤 OOF의 기본 깊이 3 대신 기존 V2 Top-20과 같은 깊이 2를 사용한다.
    team.set_params(model__n_estimators=preset["n_estimators"],
                    model__max_depth=preset["topk_classifier_max_depth"],
                    model__reg_lambda=preset["topk_classifier_reg_lambda"],
                    model__class_weight_mode=preset["class_weight_mode"],
                    selector__estimator__n_estimators=preset["lightgbm_selector_estimators"])
    
    return {"v2_all": previous["xgboost_all"],
            "v2_stable_gain_top20": previous[f"xgboost_top_{preset['top_k']}"],
            "team_lgbm_gain_top20": team}


def main():
    """설정 검증 후 Train 내부 시간순 비교 결과를 새 폴더에 저장한다."""
    parser = argparse.ArgumentParser(description="팀원 gain 후보와 기존 V2의 Train 내부 시간순 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("비어 있는 새 출력 폴더를 지정하세요.")
    
    config = load_modeling_config(args.config)
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    pipelines = build_candidates(config, preset, n_jobs=args.n_jobs)
    frame = pd.read_csv(args.train)

    # temporal_oof 행의 Recall은 기존 실행기와 같은 F1 문턱 진단값이다.
    # Recall 목표 시나리오를 자동 확정하거나 저장 모델의 문턱을 변경하지 않는다.
    results = compare_temporal(frame, config, outer_splits=preset["outer_splits"],
                               inner_splits=preset["inner_splits"], model_names=tuple(pipelines),
                               pipelines=pipelines, oof_modes=("temporal_oof",), n_jobs=args.n_jobs)
    
    args.outer_splits = preset["outer_splits"]
    args.inner_splits = preset["inner_splits"]
    args.models = list(pipelines)
    args.oof_modes = ["temporal_oof"]

    save_results(args, config, frame, results)

    # 절대 원본 경로는 그대로 보존하고 실제 Pipeline 파라미터도 함께 기록한다.
    record = {"preset": preset, "test_used": False, "saved_bundles_changed": False,
              "pipeline_parameters": {name: pipe.get_params(deep=True) for name, pipe in pipelines.items()}}
    (args.output_dir / "candidate_settings.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    
    print(results["summary"].to_string(index=False))
    print("temporal_oof Recall은 F1 문턱 진단값이며 80%·90% 목표 시나리오가 아닙니다.")
    print(f"시간순 비교 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

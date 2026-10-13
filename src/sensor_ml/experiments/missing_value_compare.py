# ==========================================
# V2 모델을 유지한 결측 처리 방식 시간순 비교
# - P0: 중앙값 대치 / P1: 중앙값 + 결측 indicator
# - P2: NaN을 유지하여 XGBoost 자체 결측 처리 사용
# - 품질 필터·모델·시간 경계·OOF 정책은 동일하게 유지
# - 기존 실행 경로와 저장 모델은 변경하지 않음
# ==========================================

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.split_contract import PROTOCOL_ID, validate_train_role
from src.sensor_ml.experiments.time_weight_compare import DEFAULT_PRESET, build_weight_candidates
from src.sensor_ml.experiments.temporal_validation import compare_temporal

MODES = {"median": "P0", "median_indicator": "P1", "native_nan": "P2"}


def build_missing_candidates(config, preset, model_name, modes, *, n_jobs=1):
    """기존 V2 후보를 복제하여 결측 처리 단계만 변경한다."""
    if config.dataset.categorical_feature_columns:
        raise ValueError("이 결측 처리 비교는 수치형 센서 전용입니다. 범주형 Profile은 별도 비교가 필요합니다.")
    
    if not modes or len(set(modes)) != len(modes) or not set(modes).issubset(MODES):
        raise ValueError("결측 처리 방식은 중복 없이 median, median_indicator, native_nan 중 지정하세요.")
    
    candidates = build_weight_candidates(config, preset, n_jobs=n_jobs)

    if model_name not in candidates:
        raise ValueError(f"알 수 없는 V2 후보입니다: {model_name}")
    
    result = {}

    for mode in modes:
        pipeline = clone(candidates[model_name])
        # indicator는 해당 학습 Fold에서 결측이 관측된 센서에만 생성된다.
        # 센서 수와 파생 특징 수가 다르므로 두 개수를 따로 기록한다.

        if mode == "median_indicator":
            pipeline.set_params(imputer__add_indicator=True)
        elif mode == "native_nan":
            # 학습 통계를 이용한 대치를 생략한다. 품질 필터는 여전히 Fold 내부에서 fit한다.
            pipeline.set_params(imputer="passthrough")

        result[f"{MODES[mode]}_{model_name}"] = pipeline

    return result


def compare_missing(frame, config, preset, model_name, modes, *, n_jobs=1):
    """같은 시간 경계에서 내부 OOF와 미래 구간 평가를 수행한다."""
    validate_train_role(frame)
    if PROTOCOL_ID not in frame or frame[PROTOCOL_ID].isna().any() or frame[PROTOCOL_ID].nunique() != 1:
        raise ValueError("단일 생성 계약을 포함하는 Train split이 필요합니다.")
    
    pipelines = build_missing_candidates(config, preset, model_name, modes, n_jobs=n_jobs)
    fit_records = []

    def record_fit(model, train, evaluation, outer_fold, candidate, mode, inner_fold):
        # 실제 각 fit의 입력 특징 수와 동적 불량 가중치를 기록한다.
        classifier = model.named_steps["model"]
        fit_records.append({
            "model_name": candidate, "outer_fold": outer_fold, "mode": mode,
            "inner_fold": inner_fold, "train_samples": len(train),
            "retained_sensor_count": len(model.named_steps["quality_filter"].retained_features_),
            "model_input_feature_count": classifier.n_features_in_,
            "effective_scale_pos_weight": classifier.effective_scale_pos_weight_,
        })

    tables = compare_temporal(
        frame, config, outer_splits=preset["outer_splits"], inner_splits=preset["inner_splits"],
        model_names=tuple(pipelines), pipelines=pipelines, oof_modes=("temporal_oof",),
        n_jobs=n_jobs, fit_observer=record_fit,
    )

    tables["fit_preprocessing"] = pd.DataFrame(fit_records)

    return tables


def save_results(args, config, preset, frame, tables):
    """실험 설정·Fold별 결과와 해석 주의사항을 새 폴더에 저장한다."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        stored = table.copy()

        if name == "quality_filter":

            for column in ("high_missing_features", "constant_features"):
                stored[column] = stored[column].map(lambda value: json.dumps(value, ensure_ascii=False))

        stored.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    execution = {"config": asdict(config), "preset": preset, "base_model": args.model,
                 "preprocessing_modes": args.modes, "train_path": str(args.train.resolve()),
                 "training_protocol_id": frame[PROTOCOL_ID].iloc[0], "n_jobs": args.n_jobs,
                 "external_validation_used": False, "test_used": False,
                 "final_model_selected": False, "oof_modes": ["temporal_oof"]}
    
    (args.output_dir / "execution.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    
    report = ["# V2 결측 처리 비교", "", "```text",
              tables["summary"].to_string(index=False), "```", "",
              "- 동일한 모델 설정에서 결측 처리만 변경했습니다.",
              "- P0 중앙값, P1 중앙값+결측 indicator, P2 XGBoost 자체 NaN 처리입니다.",
              "- indicator는 학습 시 결측이 있었던 센서만 포함합니다. 센서 수와 파생 특징 수를 구분하세요.",
              "- temporal_oof Recall은 내부 OOF F1 기준 문턱의 진단값입니다.",
              "- 운영 정책 충족 여부는 policy_selection.csv, 구간별 성능은 fold_results.csv에서 확인하세요.",
              "- 외부 Validation/Test는 읽지 않았고 최종 모델·문턱을 확정하지 않았습니다."]
    
    (args.output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args(argv=None):
    """결측 처리 후보와 기존 결과를 덮어쓰지 않는 출력 경로를 확인한다."""
    parser = argparse.ArgumentParser(description="V2 결측 처리 시간순 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model", default="v2_m0_ratio")
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=["median", "median_indicator"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.n_jobs < 1:
        parser.error("병렬 수는 1 이상의 정수여야 합니다.")

    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error(f"비어 있는 새 출력 폴더를 지정하세요: {args.output_dir}")
        
    return args


def main():
    """설정 확인 → 시간순 비교 → 결과 저장 순서로 실행한다."""
    args = parse_args()
    config = load_modeling_config(args.config)
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.train)
    tables = compare_missing(frame, config, preset, args.model, args.modes, n_jobs=args.n_jobs)
    save_results(args, config, preset, frame, tables)
    print(tables["summary"].to_string(index=False))
    print(f"결측 처리 비교 보고서: {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()

# ==========================================
# 전체 센서에서 M0·M3와 불량 가중치 3수준을 시간순 비교
# - Train 내부의 동일한 외부 시간 구간에서 6개 후보 평가
# - 문턱은 각 과거 학습 구간의 시간순 OOF에서만 선택
# - 불량 가중치는 각 fit의 정상/불량 비율로 다시 계산
# - 기존 모델·외부 Validation·Test는 읽지 않음
# ==========================================

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
from src.modeling_models import build_pipeline
from src.split_contract import PROTOCOL_ID, validate_train_role
from src.sensor_ml.experiments.temporal_validation import compare_temporal

DEFAULT_PRESET = Path(__file__).resolve().parents[3] / "configs/experiments/time_weight_v2.json"


def build_weight_candidates(config, preset, *, n_jobs=1):
    """동일한 전처리를 사용하고 M0/M3와 가중치만 바꾼 6개 후보를 생성한다."""
    if set(preset["model_variants"]) != {"M0", "M3"} or preset["weight_modes"] != ["none", "sqrt_ratio", "ratio"]:
        raise ValueError("이 실험에는 M0·M3와 none·sqrt_ratio·ratio의 6개 조합이 필요합니다.")
    
    pipelines = {}

    for variant, params in preset["model_variants"].items():
        for mode in preset["weight_modes"]:
            name = f"v2_{variant.lower()}_{mode}"
            pipeline = build_pipeline(config, "xgboost_all", n_jobs=n_jobs)
            # RF 선택기를 넣지 않는다. 각 fold의 품질 제거 후 남은 전체 센서를 사용한다.
            parameters = {**preset["base_parameters"], **params, "class_weight_mode": mode}
            pipeline.named_steps["model"].set_params(**parameters)
            pipelines[name] = pipeline

    return pipelines


def compare_time_weights(frame, config, preset, *, n_jobs=1):

    """시간순 OOF·미래 평가를 분리하고 실제 학습 가중치도 기록한다."""
    validate_train_role(frame)

    if PROTOCOL_ID not in frame or frame[PROTOCOL_ID].isna().any() or frame[PROTOCOL_ID].nunique() != 1:
        raise ValueError("새 실험에는 생성 계약을 포함한 Train split이 필요합니다.")
    
    pipelines = build_weight_candidates(config, preset, n_jobs=n_jobs)
    weights = []

    def record_weight(model, train, evaluation, outer_fold, model_name, mode, inner_fold):
    
        # 추정 설정이 아니라 실제 fit된 분류기의 가중치와 해당 학습 label 수를 저장한다.
        classifier = model.named_steps["model"]
        labels = train[config.dataset.label_column]

        weights.append({"model_name": model_name, "outer_fold": outer_fold,
                        "mode": mode, "inner_fold": inner_fold,
                        "train_samples": len(train),
                        "train_positive": int(labels.eq(config.dataset.positive_label).sum()),
                        "train_negative": int(labels.eq(config.dataset.negative_label).sum()),
                        "class_weight_mode": classifier.class_weight_mode,
                        "effective_scale_pos_weight": classifier.effective_scale_pos_weight_})

    tables = compare_temporal(frame, config, outer_splits=preset["outer_splits"],
                              inner_splits=preset["inner_splits"],
                              model_names=tuple(pipelines), pipelines=pipelines,
                              oof_modes=("temporal_oof",), n_jobs=n_jobs,
                              fit_observer=record_weight)
    
    tables["fit_weights"] = pd.DataFrame(weights)
    
    return tables


def parse_args(argv=None):
    """실험 옵션과 기존 출력 경로 조건을 확인하고 실행 인자를 반환한다."""
    parser = argparse.ArgumentParser(description="전체 센서 M0·M3 × 가중치 3수준 시간순 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    if args.n_jobs < 1:
        raise ValueError("병렬 수는 1 이상의 정수여야 합니다.")
    
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("출력 폴더가 비어 있지 않습니다. 새 폴더를 지정하세요.")
    
    return args


def save_results(args, config, preset, frame, tables):
    """계산된 결과와 실행 설정을 기존 파일 이름·형식으로 저장한다."""
    
    # 저장 단계에서는 모델을 학습하거나 문턱을 다시 선택하지 않는다.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, table in tables.items():
    
        # 리스트·사전형 품질 기록은 JSON으로 직렬화하여 CSV 안에서도 구조를 보존한다.
        stored = table.copy()
    
        if name == "quality_filter":
            for column in ("high_missing_features", "constant_features"):
                stored[column] = stored[column].map(lambda value: json.dumps(value, ensure_ascii=False))

        stored.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    execution = {"config": asdict(config), "preset": preset,
                 "train_path": str(args.train.resolve()),
                 "training_protocol_id": frame[PROTOCOL_ID].iloc[0],
                 "n_jobs": args.n_jobs, "oof_modes": ["temporal_oof"],
                 "external_validation_used": False, "test_used": False,
                 "feature_scope": "all_after_fold_quality_filter",
                 "model_parameters": {name: pipe.named_steps["model"].get_params()
                                      for name, pipe in build_weight_candidates(config, preset, n_jobs=args.n_jobs).items()}}
    
    (args.output_dir / "execution.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    
    report = ["# V2 전체 센서 시간순 가중치 비교", "",
              "```text", tables["summary"].to_string(index=False), "```", "",
              "- temporal_oof 요약의 Recall은 F1 최대 진단 문턱 결과입니다. 운영 정책 달성을 뜻하지 않습니다.",
              "- 80%·40% 등 config 정책 충족 여부는 policy_selection.csv에서 확인하세요.",
              "- 내부 OOF에서 정책 후보가 없더라도 AP·ROC-AUC·F1 진단 결과는 남습니다.",
              "- 구간별 AP·미검·오탐과 fit_weights.csv의 실제 가중치를 함께 확인하세요.",
              "- 3개 외부 fold는 학습 구간이 겹치며 독립 반복 실험이 아닙니다.",
              "- 원본 센서 20개 제한은 아직 적용하지 않았습니다. 선택기 없는 전체 센서 개선 실험입니다.",
              "- 기존 저장 모델·외부 Validation·Test는 사용하지 않았으며 최종 후보를 확정하지 않았습니다."]
    
    (args.output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    """인자 검증 → 기존 실험 코어 실행 → 결과 저장 → 콘솔 요약을 수행한다."""
    args = parse_args()
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    config = load_modeling_config(args.config)
    frame = pd.read_csv(args.train)
    tables = compare_time_weights(frame, config, preset, n_jobs=args.n_jobs)
    save_results(args, config, preset, frame, tables)
    print(tables["summary"].to_string(index=False))
    print("시간순 OOF Recall은 F1 진단값입니다. 정책 결과는 policy_selection.csv에서 확인하세요.")
    print(f"결과 저장 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

# ==========================================
# Repeated CV 기반 후보 모델 비교
# - 지정된 Train split만 사용하며 Validation/Test 파일은 읽지 않음
# - 결측 처리와 scaling은 매 CV 학습 fold 내부 Pipeline에서 fit
# - AP와 Fail Recall을 중심으로 후보 모델의 평균·표준편차·학습 시간을 기록
# ==========================================

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config, SUPPORTED_CANDIDATE_MODELS,
    )
    from src.modeling_metrics import evaluate_binary_scores
    from src.modeling_models import (build_candidate_pipelines, candidate_base_name,
                                     fit_pipeline, positive_scores)
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_preprocessing import (quality_filter_record,
                                           quality_filter_json)
    from src.split_contract import PROTOCOL_ID, validate_train_role, training_folds
except ModuleNotFoundError:
    from modeling_config import (
        DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config, SUPPORTED_CANDIDATE_MODELS,
    )
    from modeling_metrics import evaluate_binary_scores
    from modeling_models import (build_candidate_pipelines, candidate_base_name,
                                 fit_pipeline, positive_scores)
    from dataset_schema import split_frame_to_xy
    from modeling_preprocessing import (quality_filter_record,
                                       quality_filter_json)
    from split_contract import PROTOCOL_ID, validate_train_role, training_folds


# ==========================================
# Train split 하나에서 후보 모델의 반복 계층 CV 실행
# - 후보 비교 threshold는 config.json의 0.50 고정값만 사용
# - Test 및 외부 Validation의 정보는 모델 선택 과정에 사용하지 않음
# ==========================================
def compare_candidates(train_frame: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    """Train 데이터에서 반복 교차검증으로 후보 모델을 비교한다.

    Validation/Test 데이터는 입력받지 않는다. 각 fold의 학습 구간으로
    Pipeline을 새로 학습하고, 보류한 fold에서 확률 기반 지표를 계산한다.
    결과에는 지표의 평균·표준편차와 학습 시간, 센서 제거 기록을 포함한다.
    """
    validate_train_role(train_frame)

    # 직접 생성한 설정도 파일 로더와 동일하게 후보 이름을 검사한다.
    supported = SUPPORTED_CANDIDATE_MODELS
    unknown = sorted(set(config.candidate_models) - supported)

    if unknown:
        raise ValueError(f"지원하지 않는 후보 모델입니다: {unknown}")
    
    train_x, train_y, _ = split_frame_to_xy(train_frame, config.dataset)

    # 계층 CV와 모든 분류 모델은 정상·Fail label이 모두 있어야 학습할 수 있다.
    if train_y.nunique() < 2:
        raise ValueError("후보 모델 비교를 위해 Train split에는 정상과 Fail label이 모두 있어야 합니다.")
    
    # 각 fold validation에 두 class가 포함되도록 가장 적은 class 수를 먼저 확인한다.
    smallest_class_count = int(train_y.value_counts().min())

    if smallest_class_count < config.experiment.cv.n_splits:
        raise ValueError(
            "가장 적은 class의 샘플 수가 CV fold 수보다 작습니다: "
            f"최소 class 샘플={smallest_class_count}, fold={config.experiment.cv.n_splits}"
        )
    
    records: list[dict[str, object]] = []

    for model_name, pipeline in build_candidate_pipelines(config).items():
        base_model_name = candidate_base_name(model_name)

        if base_model_name not in config.candidate_models:
            continue

        fold_metrics: list[dict[str, float]] = []
        fit_times: list[float] = []
        quality_records = []

        for fit_index, valid_index in training_folds(train_x, train_y, train_frame, config):
            # 복제·불균형 가중치는 현재 학습 fold에서만 적용한다.
            started_at = perf_counter()

            model = fit_pipeline(
                pipeline, train_x.iloc[fit_index], train_y.iloc[fit_index],
                config, model_name,
            )

            fit_times.append(perf_counter() - started_at)
            
            # 제거 개수와 센서명은 해당 fold 학습 결과에서 가져온다.
            quality_records.append(quality_filter_record(model, fold=len(quality_records) + 1))
            scores = positive_scores(model, train_x.iloc[valid_index], config.dataset)

            metrics = evaluate_binary_scores(
                train_y.iloc[valid_index], scores,
                positive_label=config.dataset.positive_label,
                negative_label=config.dataset.negative_label,
                threshold=config.experiment.default_threshold,
            )

            fold_metrics.append({key: float(value) for key, value in metrics.to_dict().items()
                                 if key in {"average_precision", "recall", "precision", "f1", "roc_auc"} and value is not None})

        record: dict[str, object] = {
            "dataset_id": config.dataset.dataset_id,
            "split_strategy": "time_train_cv" if PROTOCOL_ID in train_frame and str(train_frame[PROTOCOL_ID].iloc[0]).startswith("time:") else "random_cv",
            "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train_frame else None,
            "train_samples": len(train_frame),
            "model_name": model_name,
            "n_splits": config.experiment.cv.n_splits,
            "n_repeats": config.experiment.cv.n_repeats,
            "threshold": config.experiment.default_threshold,
            "fit_time_mean_seconds": float(np.mean(fit_times)),
            "fit_time_std_seconds": float(np.std(fit_times)),
            "quality_filter_log": quality_filter_json(quality_records),
        }

        for metric_name in ("average_precision", "recall", "precision", "f1", "roc_auc"):
            values = [fold[metric_name] for fold in fold_metrics if metric_name in fold]
            record[f"{metric_name}_mean"] = float(np.mean(values)) if values else None
            record[f"{metric_name}_std"] = float(np.std(values)) if values else None

        records.append(record)

    if not records:
        raise ValueError("models.candidates에 지원하는 후보 모델이 없습니다.")
    
    return pd.DataFrame(records).sort_values("average_precision_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    """명령행 인자를 읽어 후보 비교 결과를 CSV로 저장한다."""
    # Train 파일만 받아 후보 비교 결과를 재생성 가능한 CSV로 기록한다.
    parser = argparse.ArgumentParser(description="반복 계층 CV 후보 모델 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/secom/exploratory/legacy/model_compare.csv"))
    arguments = parser.parse_args()
    config = load_modeling_config(arguments.config)
    results = compare_candidates(pd.read_csv(arguments.train), config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(arguments.output, index=False, encoding="utf-8-sig")
    print(results.drop(columns="quality_filter_log").to_string(index=False))


if __name__ == "__main__":
    main()

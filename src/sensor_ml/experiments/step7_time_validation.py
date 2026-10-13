# ==========================================
# 시간 기반 Validation에서 사전 선택 후보를 비교
# - Time Train 내부 CV 결과로 미리 정한 후보만 동일한 Time Train으로 학습한다.
# - Time Validation은 시간 순서 일반화 확인용이며 threshold 탐색에는 사용하지 않는다.
# - Test split은 이 단계에서 읽거나 평가하지 않는다.
# ==========================================

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_models import build_pipeline, fit_pipeline, positive_scores
    from src.modeling_preprocessing import fitted_feature_count, quality_filter_record, quality_filter_json
    from src.split_contract import PROTOCOL_ID, SOURCE_ROW_ID, validate_split_pair
    from src.threshold_policy import validate_threshold_report
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from dataset_schema import split_frame_to_xy
    from modeling_models import build_pipeline, fit_pipeline, positive_scores
    from modeling_preprocessing import fitted_feature_count, quality_filter_record, quality_filter_json
    from split_contract import PROTOCOL_ID, SOURCE_ROW_ID, validate_split_pair
    from threshold_policy import validate_threshold_report


# 기존 탐색 단계의 비교 후보 목록이다. 새 평가에서도 후보를 Train 내부에서 사전 선정한다.
# 이 목록을 시간 검증 결과에 따라 늘리면 Validation set에 과적합될 수 있다.
PRESELECTED_EXPERIMENTS = (
    "lightgbm_all",
    "l1_balanced_l1_select",
)

AVAILABLE_EXPERIMENTS = PRESELECTED_EXPERIMENTS + (
    "xgboost",
    "xgboost_scale_pos_weight",
)


# ==========================================
# Time Train/Validation으로 후보 모델을 학습·평가
# - Train feature 구성을 기준으로 Validation feature 구성을 엄격히 확인한다.
# - 각 Pipeline의 imputation, scaling, feature selection은 Time Train에서만 fit한다.
# - 비교 기본 threshold 또는 사전에 선택한 OOF threshold를 사용한다.
# ==========================================
def compare_time_validation(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    experiment_names: tuple[str, ...] = PRESELECTED_EXPERIMENTS,
    threshold: float | None = None,
    n_jobs: int = 1,
    threshold_report: pd.DataFrame | None = None,
    pipeline_templates: dict | None = None,
) -> pd.DataFrame:
    """사전 선택된 실험을 시간 순서 Validation 구간에서 평가한다.

    전처리·특징 선택·모델 학습은 과거 구간인 Train에서만 수행한다.
    threshold는 OOF에서 미리 정한 값을 그대로 쓰며, Validation으로
    threshold를 다시 고르지 않아 평가 자료에 맞춘 낙관 편향을 방지한다.
    """
    if not experiment_names:
        raise ValueError("시간 검증할 후보 모델을 하나 이상 지정해야 합니다.")

    validate_split_pair(train_frame, validation_frame, config.dataset, temporal=True)

    # Train의 feature 집합과 순서를 기준으로 Validation을 검증한다.
    train_x, train_y, feature_columns = split_frame_to_xy(train_frame, config.dataset)

    validation_x, validation_y, _ = split_frame_to_xy(
        validation_frame,
        config.dataset,
        expected_feature_columns=feature_columns,
    )

    if train_y.nunique() < 2:
        raise ValueError("Time Train split에는 정상과 Fail label이 모두 있어야 합니다.")

    # 별도 threshold가 없으면 후보 모델 비교 때 사용한 config 기본값을 쓴다.
    # 최종 후보 threshold는 CLI 인자로만 주입해 config.json의 비교 기준을 바꾸지 않는다.
    evaluation_threshold = (
        config.experiment.default_threshold if threshold is None else threshold
    )

    if not 0.0 <= evaluation_threshold <= 1.0:
        raise ValueError("Time Validation threshold는 0.0 이상 1.0 이하여야 합니다.")
    
    if threshold_report is not None:
        if threshold is None or len(experiment_names) != 1:
            raise ValueError("threshold 비교표를 적용할 때는 threshold와 후보 모델 하나를 명시하세요.")
        
        validate_threshold_report(threshold_report, train_frame, config,
                                  experiment_names[0], evaluation_threshold)

    # 후보·특징 실험 모두 동일한 이름 해석 규칙으로 필요한 Pipeline만 만든다.
    if pipeline_templates is not None and set(pipeline_templates) != set(experiment_names):
        raise ValueError("주입한 Pipeline 이름과 평가 후보 이름이 일치해야 합니다.")
    
    # OOF와 같은 설정의 후보를 전달받을 수 있으며 실제 fit은 공통 코어가 복제하여 수행한다.
    available_experiments = pipeline_templates if pipeline_templates is not None else {
        name: build_pipeline(config, name, n_jobs=n_jobs) for name in experiment_names
    }

    rows: list[dict[str, object]] = []
    selected_features = []
    prediction_records = []

    for experiment_name in experiment_names:
        # 공통 fit은 전처리·선택·가중치를 Time Train에서만 학습한다.
        started_at = perf_counter()
        
        pipeline = fit_pipeline(
            available_experiments[experiment_name], train_x, train_y,
            config, experiment_name, n_jobs=n_jobs,
        )

        fit_seconds = perf_counter() - started_at
        positive_scores_array = positive_scores(pipeline, validation_x, config.dataset)

        # 행별 확률을 남겨 문턱을 다시 탐색하지 않고 확률 분포와 미검출을 진단한다.
        prediction_records.extend({"experiment": experiment_name, "source_row_index": index,
                                   "source_row_id": validation_frame[SOURCE_ROW_ID].iloc[position]
                                   if SOURCE_ROW_ID in validation_frame else None,
                                   "label": validation_y.iloc[position], "positive_score": float(score),
                                   "predicted_positive": bool(score >= evaluation_threshold)}
                                  for position, (index, score) in enumerate(
                                      zip(validation_frame.index, positive_scores_array)))
        
        # 전체 Train에서 다시 학습한 선택 결과를 남긴다. OOF fold별 센서와 같다고 가정하지 않는다.
        selector = pipeline.named_steps.get("selector")
        if selector is not None and hasattr(selector, "get_support"):
            names = pipeline[:-2].get_feature_names_out()[selector.get_support()]
            selected_features.extend({"experiment": experiment_name, "feature": str(name)} for name in names)

        # OOF에서 정한 threshold를 그대로 적용해 시간 구간의 일반화만 확인한다.
        # 이 함수는 Time Validation 결과로 threshold를 다시 탐색하지 않는다.
        metrics = evaluate_binary_scores(
            validation_y,
            positive_scores_array,
            positive_label=config.dataset.positive_label,
            negative_label=config.dataset.negative_label,
            threshold=evaluation_threshold,
        )

        rows.append(
            {
                "dataset_id": config.dataset.dataset_id,
                "split_strategy": "time_validation",
                "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train_frame else None,
                "experiment": experiment_name,
                "train_samples": len(train_frame),
                "validation_samples": len(validation_frame),
                "selected_feature_count": fitted_feature_count(pipeline),
                "fit_time_seconds": fit_seconds,
                "quality_filter_log": quality_filter_json([quality_filter_record(pipeline)]),
                **asdict(metrics),
            }
        )

    result = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    result.attrs["selected_features"] = selected_features
    result.attrs["predictions"] = prediction_records

    return result


# ==========================================
# CSV 파일을 입력받아 Time Validation 결과를 기록
# - 실행 결과는 재생성 가능한 로그이므로 기존 파일은 최신 실행 결과로 교체한다.
# - Test 경로 인자는 제공하지 않아 최종 평가 데이터 사용을 구조적으로 막는다.
# ==========================================
def compare_time_validation_from_files(
    config_path: Path | str,
    train_path: Path | str,
    validation_path: Path | str,
    output_path: Path | str,
    *,
    experiment_names: tuple[str, ...] = PRESELECTED_EXPERIMENTS,
    threshold: float | None = None,
    n_jobs: int = 1,
    threshold_report_path: Path | str | None = None,
) -> pd.DataFrame:
    """설정·분할 CSV를 읽어 시간 검증을 수행하고 결과 CSV를 저장한다."""
    # 파일을 읽는 경계에서만 I/O를 수행하고, 실제 평가는 순수 함수에 위임한다.
    config = load_modeling_config(config_path)

    result = compare_time_validation(
        pd.read_csv(train_path),
        pd.read_csv(validation_path),
        config,
        experiment_names=experiment_names,
        threshold=threshold,
        n_jobs=n_jobs,
        threshold_report=pd.read_csv(threshold_report_path) if threshold_report_path is not None else None,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")

    return result


def _parse_arguments() -> argparse.Namespace:
    """Time Validation CLI 옵션을 정의하고 파싱한다."""
    # Test 경로 인자를 제공하지 않아 CLI 단계에서 Test 오용을 막는다.
    parser = argparse.ArgumentParser(description="사전 선택 모델의 Time Validation 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--threshold-report", type=Path,
                        help="새 단일 OOF threshold 비교표. 학습 계약·모델·방식을 검사합니다.")
    parser.add_argument(
        "--output", type=Path, default=Path("logs/secom/exploratory/legacy/time_validation_compare.csv")
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="트리 모델 내부 병렬 처리 수입니다. 로컬 실행은 -1을 사용할 수 있습니다.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "OOF에서 미리 정한 평가 threshold입니다. 생략하면 config.json의 "
            "후보 비교용 기본값을 사용합니다. Time Validation 결과로 이 값을 재탐색하지 않습니다."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=AVAILABLE_EXPERIMENTS,
        default=list(PRESELECTED_EXPERIMENTS),
        help="Time Validation에 평가할 사전 선택 후보 이름입니다.",
    )
    
    return parser.parse_args()


def main() -> None:
    """CLI 진입점: 평가를 수행하고 결과를 터미널에 요약한다."""
    # OOF에서 정한 threshold를 그대로 전달해 Time Validation 결과만 출력한다.
    arguments = _parse_arguments()
    result = compare_time_validation_from_files(
        arguments.config,
        arguments.train,
        arguments.validation,
        arguments.output,
        experiment_names=tuple(arguments.experiments),
        threshold=arguments.threshold,
        n_jobs=arguments.n_jobs,
        threshold_report_path=arguments.threshold_report,
    )
    print(result.to_string(index=False))
    print(f"결과 저장 경로: {arguments.output}")


if __name__ == "__main__":
    main()

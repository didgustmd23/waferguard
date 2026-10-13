# ==========================================
# 선택한 트리 후보의 OOF 확률과 threshold별 지표 비교
# - 시간 holdout을 먼저 제외한 Time Train 내부에서만 OOF 확률을 생성한다.
# - Time Validation과 Test split은 읽거나 threshold 결정에 사용하지 않는다.
# - 이 파일은 후보별 trade-off를 기록할 뿐, 최종 threshold를 자동 확정하지 않는다.
# ==========================================

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from src.modeling_metrics import evaluate_binary_scores
    from src.threshold_policy import validate_threshold_report
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_models import build_pipeline, fit_pipeline, positive_scores
    from src.modeling_preprocessing import quality_filter_record, quality_filter_json
    from src.split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role, training_folds
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, ModelingConfig, load_modeling_config
    from modeling_metrics import evaluate_binary_scores
    from threshold_policy import validate_threshold_report
    from dataset_schema import split_frame_to_xy
    from modeling_models import build_pipeline, fit_pipeline, positive_scores
    from modeling_preprocessing import quality_filter_record, quality_filter_json
    from split_contract import SOURCE_ROW_ID, PROTOCOL_ID, validate_train_role, training_folds


EXPERIMENT_NAME = "lightgbm_all"


# ==========================================
# 단일 모델 threshold 비교용 OOF 확률 생성
# - 한 repeat에서 각 샘플은 정확히 한 번 validation fold에 속한다.
# - 기본 single 모드는 반복 평균 없이 샘플당 한 번의 fold 모델 예측을 사용한다.
# - repeated_mean은 기존 방식의 분석용이며 최종 threshold 근거로 혼동하지 않는다.
# - 단일 OOF도 fold 학습과 전체 Train 재학습의 확률 척도 일치를 보장하지 않는다.
# ==========================================
def generate_oof_scores(
    train_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    n_jobs: int = 1,
    score_method: str = "single",
    experiment_name: str = EXPERIMENT_NAME,
    pipeline_template=None,
) -> tuple[pd.DataFrame, float]:
    """Train 내부 교차검증으로 OOF 점수와 실행 시간을 만든다.

    ``single``은 각 샘플이 한 번만 검증 fold에 나타나도록 반복 횟수를
    1로 제한하여 threshold 후보를 만든다. ``repeated_mean``은 반복별
    예측을 평균한 분석용 결과다. 두 방식은 결과 열에 구분해 기록한다.
    """
    validate_train_role(train_frame)

    if score_method not in {"single", "repeated_mean"}:
        raise ValueError("OOF score_method는 single 또는 repeated_mean이어야 합니다.")
    
    # 모델 비교의 5×5 CV 정책은 유지하고 threshold OOF에만 별도 반복 정책을 적용한다.
    oof_config = config if score_method == "repeated_mean" else replace(
        config, experiment=replace(config.experiment, cv=replace(config.experiment.cv, n_repeats=1))
    )

    features, labels, _ = split_frame_to_xy(train_frame, config.dataset)

    if labels.nunique() < 2:
        raise ValueError("OOF를 생성하려면 Train에 정상과 Fail label이 모두 있어야 합니다.")

    # 외부에서 준비한 M0·M3도 같은 OOF 코어를 사용한다.
    # fit_pipeline이 매 fold마다 clone하므로 전달한 객체의 학습 상태는 재사용하지 않는다.
    if experiment_name == "xgboost_lightgbm_gain_top20" and pipeline_template is None:

        # 팀원 제안은 명시적으로 선택할 때만 사용한다. 외부 센서 JSON은 읽지 않는다.
        from src.sensor_ml.experiments.lightgbm_gain_candidate import build_gain_candidate

        pipeline_template = build_gain_candidate(config, n_jobs=n_jobs)

    experiment_pipeline = (build_pipeline(config, experiment_name, n_jobs=n_jobs)
                           if pipeline_template is None else pipeline_template)

    score_sum = np.zeros(len(train_frame), dtype=float)
    prediction_count = np.zeros(len(train_frame), dtype=int)
    quality_records = []
    fold_ids = np.zeros(len(train_frame), dtype=int)
    selected_records = []
    started_at = perf_counter()

    for fit_indices, validation_indices in training_folds(features, labels, train_frame, oof_config):
        
        # fold별 Pipeline을 새로 만들어 imputing과 model fit이 validation에 닿지 않게 한다.
        pipeline = fit_pipeline(
            experiment_pipeline, features.iloc[fit_indices], labels.iloc[fit_indices],
            config, experiment_name, n_jobs=n_jobs,
        )

        fold = len(quality_records) + 1
        quality_records.append(quality_filter_record(pipeline, fold=fold))

        # 각 샘플의 미학습 예측 fold와 학습 fold에서 고른 센서 목록을 별도로 기록한다.
        fold_ids[validation_indices] = fold
        selector = pipeline.named_steps.get("selector")

        if selector is not None and hasattr(selector, "get_support"):
            names = pipeline[:-2].get_feature_names_out()[selector.get_support()]
            selected_records.extend({"cv_fold": fold, "feature": str(name)} for name in names)

        fold_scores = positive_scores(pipeline, features.iloc[validation_indices], config.dataset)
        score_sum[validation_indices] += fold_scores
        prediction_count[validation_indices] += 1

    elapsed_seconds = perf_counter() - started_at
    expected_count = oof_config.experiment.cv.n_repeats

    if not np.all(prediction_count == expected_count):
        raise RuntimeError(
            "모든 Train 샘플의 OOF 예측 횟수가 반복 횟수와 일치하지 않습니다. "
            f"기대값={expected_count}, 실제 범위={prediction_count.min()}~{prediction_count.max()}"
        )

    # split 파일의 행 위치와 원본 ID를 구분한다. 기존 CSV의 원본 ID는 알 수 없다.
    oof_frame = pd.DataFrame(
        {
            "source_row_index": train_frame.index,
            "source_row_id": train_frame[SOURCE_ROW_ID].to_numpy() if SOURCE_ROW_ID in train_frame else None,
            "label": labels.to_numpy(),
            "oof_positive_score": score_sum / prediction_count,
            "oof_prediction_count": prediction_count,
            "training_protocol_id": train_frame[PROTOCOL_ID].iloc[0] if PROTOCOL_ID in train_frame else None,
            "oof_score_method": score_method,
            "threshold_use": "candidate" if score_method == "single" else "analysis_only",
            # 반복 평균은 여러 fold 예측이 섞이므로 단일 fold 번호를 노출하지 않는다.
            "cv_fold": fold_ids if score_method == "single" else None,
        }
    )
    
    # fold별 제거 기록은 샘플별 점수와 별도로 보존해 같은 로그의 중복 저장을 방지한다.
    oof_frame.attrs["quality_filter_records"] = quality_records
    oof_frame.attrs["selected_features"] = selected_records
    return oof_frame, elapsed_seconds


# ==========================================
# OOF 확률을 threshold 후보별 지표로 변환
# - threshold 범위는 0~1 내부에서만 허용한다.
# - Recall, Precision, FN, FP를 모두 저장해 운영 기준으로 사람이 선택한다.
# ==========================================
def compare_thresholds(
    oof_frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    thresholds: np.ndarray | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> pd.DataFrame:
    """주어진 OOF 점수에 threshold 후보를 적용하고 정책 지표를 계산한다.

    기본 후보는 OOF 점수 분위수로 구성하고 기본 threshold도 포함한다.
    반환표는 후보별 Recall, Precision, 오분류 건수와 재검사 비율을 담아
    사람이 운영 제약과 함께 threshold를 검토할 수 있도록 한다.
    """
    required_columns = {"label", "oof_positive_score"}
    missing_columns = sorted(required_columns - set(oof_frame.columns))

    if missing_columns:
        raise ValueError(f"OOF 결과에 필수 column이 없습니다: {missing_columns}")
    
    if oof_frame.empty:
        raise ValueError("OOF 결과는 비어 있으면 안 됩니다.")
    
    # 분위수 계산 전에도 공통 metric 방어 코드로 결측 label·확률 범위를 검사한다.
    evaluate_binary_scores(oof_frame["label"], oof_frame["oof_positive_score"],
                           positive_label=config.dataset.positive_label,
                           negative_label=config.dataset.negative_label,
                           threshold=config.experiment.default_threshold)
    
    for column in ("training_protocol_id", "oof_score_method", "threshold_use"):
        if column in oof_frame and oof_frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"서로 다른 OOF 실행 결과가 섞여 있습니다: {column}")

    # 모델의 확률 보정 상태에 따라 유효 threshold 범위가 크게 달라질 수 있다.
    # 따라서 고정 격자 대신 OOF 확률의 분위수를 기본 후보로 사용한다.
    if thresholds is None:
        thresholds = np.quantile(
            oof_frame["oof_positive_score"], np.linspace(0.0, 1.0, 101)
        )

    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("threshold 후보는 비어 있지 않은 1차원 배열이어야 합니다.")
    
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("모든 threshold 후보는 0.0 이상 1.0 이하의 유한한 값이어야 합니다.")

    rows: list[dict[str, object]] = []

    for threshold in np.unique(np.append(thresholds, config.experiment.default_threshold)):
        metrics = evaluate_binary_scores(
            oof_frame["label"],
            oof_frame["oof_positive_score"],
            positive_label=config.dataset.positive_label,
            negative_label=config.dataset.negative_label,
            threshold=float(threshold),
        )

        rows.append(
            {
                "dataset_id": config.dataset.dataset_id,
                "split_strategy": "time_train_oof" if "training_protocol_id" in oof_frame and str(oof_frame["training_protocol_id"].iloc[0]).startswith("time:") else "legacy_train_oof",
                "training_protocol_id": oof_frame["training_protocol_id"].iloc[0] if "training_protocol_id" in oof_frame else None,
                "experiment": experiment_name,
                "oof_score_method": oof_frame["oof_score_method"].iloc[0] if "oof_score_method" in oof_frame else "unknown",
                "threshold_use": oof_frame["threshold_use"].iloc[0] if "threshold_use" in oof_frame else "analysis_only",
                **asdict(metrics),
                "reinspection_ratio": (metrics.true_positive + metrics.false_positive) / metrics.support,
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


# ==========================================
# Train 파일에서 OOF·threshold 비교 결과를 저장
# - OOF 결과와 threshold 표를 별도 파일로 저장해 재현성과 오류 분석을 지원한다.
# - Test 경로 인자는 제공하지 않아 봉인된 Test 사용을 막는다.
# ==========================================
def run_threshold_oof_from_file(
    config_path: Path | str,
    train_path: Path | str,
    oof_output_path: Path | str,
    threshold_output_path: Path | str,
    *,
    n_jobs: int = 1,
    score_method: str = "single",
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """입력 파일을 읽어 OOF 점수·threshold 표·품질 필터 로그를 저장한다."""
    
    # OOF 원본과 threshold 비교표를 분리 저장해 이후 오류 사례 분석에 재사용한다.
    config = load_modeling_config(config_path)
    
    oof_frame, elapsed_seconds = generate_oof_scores(
        pd.read_csv(train_path), config, n_jobs=n_jobs, score_method=score_method,
        experiment_name=experiment_name,
    )

    threshold_frame = compare_thresholds(oof_frame, config, experiment_name=experiment_name)

    for frame, output_path in (
        (oof_frame, oof_output_path),
        (threshold_frame, threshold_output_path),
    ):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    # OOF 점수 파일 옆에 fold별 센서 제거 개수와 JSON 센서 목록을 함께 저장한다.
    quality_log = pd.DataFrame(oof_frame.attrs["quality_filter_records"])

    for column in ("high_missing_features", "constant_features"):
        quality_log[column] = quality_log[column].map(quality_filter_json)

    quality_path = Path(oof_output_path).with_name(Path(oof_output_path).stem + "_quality_filter.csv")
    quality_log.to_csv(quality_path, index=False, encoding="utf-8-sig")

    # fold별 목록은 전체 Train에서 고정한 센서 목록과 구분해 보존한다.
    selected = pd.DataFrame(oof_frame.attrs["selected_features"], columns=["cv_fold", "feature"])

    if not selected.empty:
        selected.to_csv(Path(oof_output_path).with_name(Path(oof_output_path).stem + "_selected_features.csv"),
                        index=False, encoding="utf-8-sig")

    return oof_frame, threshold_frame, elapsed_seconds


def _parse_arguments() -> argparse.Namespace:
    """OOF 생성 및 threshold 분석 CLI의 인자를 정의하고 파싱한다."""
    # 새 평가 경로에서는 integrated/time_train.csv만 입력한다.
    parser = argparse.ArgumentParser(description="후보 모델 OOF threshold 비교")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, default=Path("logs/secom/exploratory/legacy/lightgbm_oof_predictions.csv"))
    parser.add_argument("--threshold-output", type=Path, default=Path("logs/secom/exploratory/legacy/threshold_compare.csv"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--score-method", choices=("single", "repeated_mean"), default="single",
                        help="single은 threshold 후보용, repeated_mean은 반복 평균 분석용입니다.")
    parser.add_argument("--experiment", choices=("lightgbm_all", "xgboost",
                                                    "xgboost_scale_pos_weight", "xgboost_lightgbm_gain_top20"),
                        default=EXPERIMENT_NAME,
                        help="OOF threshold를 생성할 사전 선택 후보입니다.")
    return parser.parse_args()


def main() -> None:
    """CLI 실행 진입점으로 비교 요약과 생성된 파일 경로를 출력한다."""
    arguments = _parse_arguments()
    _, threshold_frame, elapsed_seconds = run_threshold_oof_from_file(
        arguments.config,
        arguments.train,
        arguments.oof_output,
        arguments.threshold_output,
        n_jobs=arguments.n_jobs,
        score_method=arguments.score_method,
        experiment_name=arguments.experiment,
    )
    # 전체 비교표는 CSV에 저장하고, 콘솔에는 판단에 필요한 대표 후보만 표시한다.
    default_row = threshold_frame.loc[
        np.isclose(threshold_frame["threshold"], 0.5)
    ]

    console_summary = pd.concat(
        [
            threshold_frame.nlargest(5, "f1"),
            threshold_frame.nlargest(5, "recall"),
            default_row,
        ]
    ).drop_duplicates(subset=["threshold"]).sort_values("threshold")

    print(console_summary.to_string(index=False))
    
    # 기존 요약을 없애지 않고 팀원 제안의 Recall별 정밀도 비교를 추가한다.
    from src.sensor_ml.experiments.lightgbm_gain_candidate import recall_scenarios
    print("\nRecall 목표별 정밀도 비교 — 최종 문턱 확정 아님")
    print(recall_scenarios(threshold_frame).to_string(index=False))
    print(f"OOF 생성 시간(초): {elapsed_seconds:.3f}")
    print(f"OOF 저장 경로: {arguments.oof_output}")
    print(f"Threshold 비교 저장 경로: {arguments.threshold_output}")


if __name__ == "__main__":
    main()

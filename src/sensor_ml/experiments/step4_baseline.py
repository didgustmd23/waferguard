# ==========================================
# Train/Validation split 기반 Logistic Regression Baseline
# - A 담당자가 생성한 Train/Validation split만 입력으로 사용
# - Test split은 이 단계에서 읽거나 평가하지 않아 최종 성능 평가 누수를 방지
# - 수치형 feature는 median imputation → scaling, 선언된 범주형 feature는
#   most-frequent imputation → one-hot encoding으로 범용 처리
# - Fail 확률과 config.json의 기본 threshold로 공통 지표 및 학습 시간을 기록
#
# split 파일 계약:
# - Train과 Validation은 모두 Dataset Profile의 label 및 feature column을 포함
# - metadata column은 포함할 수 있으나 Dataset Profile 규칙에 따라 모델 입력에서 제외
# - 실제 Test 평가는 모델과 threshold가 모두 확정된 뒤 별도 단계에서만 수행
# ==========================================

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from src.dataset_schema import split_frame_to_xy
    from src.modeling_config import DEFAULT_CONFIG_PATH, DatasetSpec, ModelingConfig, load_modeling_config
    from src.modeling_metrics import BinaryMetrics, evaluate_binary_scores
    from src.modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from src.split_contract import validate_split_pair
except ModuleNotFoundError:
    from dataset_schema import validate_dataset_frame
    from modeling_config import DEFAULT_CONFIG_PATH, DatasetSpec, ModelingConfig, load_modeling_config
    from modeling_metrics import BinaryMetrics, evaluate_binary_scores
    from modeling_preprocessing import preprocessing_steps, quality_filter_record, quality_filter_json
    from split_contract import validate_split_pair


# ==========================================
# Baseline 한 번의 Train/Validation 실행 결과
# - 지표와 함께 feature 수, split 표본 수, 모델 fit 시간도 동일한 행에 기록
# - test 관련 필드는 두지 않아 이 단계의 역할을 Train/Validation 검증으로 제한
# ==========================================
@dataclass(frozen=True)
class BaselineResult:
    dataset_id: str
    model_name: str
    train_samples: int
    validation_samples: int
    feature_count: int
    training_seconds: float
    metrics: BinaryMetrics
    split_strategy: str | None = None
    train_path: str | None = None
    validation_path: str | None = None
    quality_filter_log: str = "[]"

    def to_dict(self) -> dict[str, object]:
        # metrics 객체를 평탄화하여 CSV의 한 행으로 바로 저장할 수 있게 만든다.
        return {
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "feature_count": self.feature_count,
            "quality_filter_log": self.quality_filter_log,
            "training_seconds": self.training_seconds,
            "split_strategy": self.split_strategy,
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            **asdict(self.metrics),
        }


# ==========================================
# Dataset Profile의 feature 타입에 맞는 Baseline Pipeline 생성
# - 센서 품질 필터를 먼저 적용한 뒤 median → scaler → logistic 순서로 학습
# - 범주형 feature가 선언된 경우에만 ColumnTransformer를 사용해 타입별 전처리를 분리
# - 모든 전처리기는 train split의 fit 단계에서만 학습되어 validation 정보가 섞이지 않음
# ==========================================
def build_baseline_pipeline(
    dataset: DatasetSpec,
    feature_columns: tuple[str, ...],
    *,
    random_state: int,
) -> Pipeline:
    # Profile에 선언된 범주형 column이 현재 선택 feature에 모두 포함되는지 확인한다.
    unknown_categorical = sorted(set(dataset.categorical_feature_columns) - set(feature_columns))

    if unknown_categorical:
        raise ValueError(
            "Profile에 선언된 범주형 feature가 선택된 feature에 없습니다: "
            f"{unknown_categorical}"
        )

    # Baseline과 후보·특징 선택 실험이 동일한 타입별 전처리 코어를 재사용한다.
    return Pipeline(preprocessing_steps(dataset, scale=True) + [
        ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


# ==========================================
# Train/Validation split으로 Baseline 학습 및 검증
# - split 생성은 수행하지 않으며, 주입된 Train/Validation만 사용
# - Fail 확률 열을 명시적으로 찾아 공통 metric 함수에 전달
# - Validation에는 positive label이 있어야 AP와 Fail Recall을 해석할 수 있음
# ==========================================
def run_baseline(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ModelingConfig,
) -> BaselineResult:
    
    # 현재 계획의 Baseline은 Logistic Regression 하나로 한정한다.
    if config.baseline_model != "logistic_regression":
        raise ValueError(
            "step4_baseline은 'logistic_regression' Baseline만 지원합니다; "
            f"입력값: '{config.baseline_model}'"
        )

    # split의 역할·원본 행 중복을 모델 학습 전에 검사한다.
    validate_split_pair(train_frame, validation_frame, config.dataset)

    # Train feature를 기준으로 Validation feature의 구성과 순서를 검증한다.
    train_x, train_y, feature_columns = split_frame_to_xy(train_frame, config.dataset)

    validation_x, validation_y, _ = split_frame_to_xy(
        validation_frame,
        config.dataset,
        expected_feature_columns=feature_columns,
    )

    # Logistic Regression 학습에는 두 class가 모두 있어야 한다.
    if train_y.nunique() < 2:
        raise ValueError("Train split에는 정상 label과 Fail label이 모두 있어야 합니다")

    # Pipeline 내부의 imputer와 scaler도 train split에서만 fit되도록 학습 시간을 측정한다.
    pipeline = build_baseline_pipeline(
        config.dataset,
        feature_columns,
        random_state=config.experiment.cv.random_state,
    )

    started_at = perf_counter()
    pipeline.fit(train_x, train_y)
    training_seconds = perf_counter() - started_at

    # class 순서를 가정하지 않고 Profile의 Fail label에 해당하는 확률 열을 찾는다.
    model = pipeline.named_steps["model"]

    class_positions = [
        index for index, label in enumerate(model.classes_) if label == config.dataset.positive_label
    ]

    if len(class_positions) != 1:
        raise ValueError("학습된 Baseline에 Profile의 Fail label이 없습니다")
    
    positive_scores = pipeline.predict_proba(validation_x)[:, class_positions[0]]

    # 후보 Baseline 비교에서는 config.json에 둔 고정 threshold를 명시적으로 전달한다.
    metrics = evaluate_binary_scores(
        validation_y,
        positive_scores,
        positive_label=config.dataset.positive_label,
        negative_label=config.dataset.negative_label,
        threshold=config.experiment.default_threshold,
    )

    return BaselineResult(
        dataset_id=config.dataset.dataset_id,
        model_name=config.baseline_model,
        train_samples=len(train_frame),
        validation_samples=len(validation_frame),
        feature_count=len(pipeline.named_steps["quality_filter"].retained_features_),
        quality_filter_log=quality_filter_json([quality_filter_record(pipeline)]),
        training_seconds=training_seconds,
        metrics=metrics,
    )


# ==========================================
# Baseline 결과를 재현 가능한 한 행 CSV로 저장
# - 결과 폴더가 없으면 생성하고 UTF-8-SIG로 저장해 Excel에서도 한글 header를 읽음
# - 이 함수는 결과를 누적하지 않고 실행 단위의 최신 한 행을 기록
# ==========================================
def write_baseline_result(result: BaselineResult, output_path: Path | str) -> Path:
    # 문자열 경로도 Path로 통일한 뒤 상위 로그 폴더를 준비한다.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 평탄화한 결과를 index 없이 한 행 CSV로 기록한다.
    pd.DataFrame([result.to_dict()]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ==========================================
# CSV split 파일을 읽어 Baseline을 실행하는 진입점
# - Train/Validation 경로를 필수 인자로 받아 원본 데이터나 Test를 임의로 사용하지 않음
# - config.json 경로를 바꾸면 다른 Dataset Profile에도 같은 코드를 재사용 가능
# ==========================================
def run_baseline_from_files(
    config_path: Path | str,
    train_path: Path | str,
    validation_path: Path | str,
    output_path: Path | str,
    *,
    split_strategy: str,
) -> BaselineResult:
    # 결과 로그에 남길 split 방식은 비어 있지 않은 문자열이어야 한다.
    if not split_strategy.strip():
        raise ValueError("split_strategy는 비어 있지 않은 문자열이어야 합니다")

    # 공통 설정과 두 split 파일을 명시적으로 읽는다.
    config = load_modeling_config(config_path)
    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)

    if split_strategy == "time":
        validate_split_pair(train_frame, validation_frame, config.dataset, temporal=True)

    # 실행에 사용한 split 방식과 상대 경로를 결과에도 남겨 파일명에만 의존하지 않게 한다.
    result = replace(
        run_baseline(train_frame, validation_frame, config),
        split_strategy=split_strategy,
        train_path=Path(train_path).as_posix(),
        validation_path=Path(validation_path).as_posix(),
    )

    # Train/Validation 결과만 저장하고 Test 성능은 이 단계에서 계산하지 않는다.
    write_baseline_result(result, output_path)
    return result


def _parse_arguments() -> argparse.Namespace:
    # split 생성 단계가 정한 파일을 명시적으로 받도록 CLI 인자를 정의한다.
    parser = argparse.ArgumentParser(description="Train/Validation Baseline 평가")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--split-strategy", type=str, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/secom/exploratory/legacy/baseline_result.csv"))
    return parser.parse_args()


def main() -> None:
    # 명령행에서 받은 Train/Validation split으로 Baseline을 실행하고 핵심 결과만 출력한다.
    arguments = _parse_arguments()
    
    result = run_baseline_from_files(
        arguments.config,
        arguments.train,
        arguments.validation,
        arguments.output,
        split_strategy=arguments.split_strategy,
    )

    print(f"데이터셋 ID: {result.dataset_id}")
    print(f"모델: {result.model_name}")
    print(f"Split 방식: {result.split_strategy}")
    print(f"Train 표본 수: {result.train_samples}")
    print(f"Validation 표본 수: {result.validation_samples}")
    print(f"Fail Recall: {result.metrics.recall:.4f}")
    print(f"Average Precision: {result.metrics.average_precision:.4f}")
    print(f"결과 저장 경로: {arguments.output}")


if __name__ == "__main__":
    main()

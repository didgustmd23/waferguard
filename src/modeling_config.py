# ==========================================
# Dataset Profile + config.json 기반 모델링 실험 설정 로더
# - 범용 코어는 특정 데이터셋의 컬럼명·Label 값에 직접 의존하지 않음
# - Dataset Profile에는 데이터 구조와 품질 규칙을, config.json에는 실험 조건을 분리
# - SECOM은 configs/datasets/secom.json으로 제공되는 첫 번째 Dataset Profile
# - 설정 파일 형식과 범위를 검증해 잘못된 실험 조건을 조기에 차단
# ==========================================

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final


# src/ 기준 상위 프로젝트 루트의 공통 실험 설정 경로
DEFAULT_CONFIG_PATH: Final = Path(__file__).resolve().parents[1] / "config.json"

# 설정 검증과 모델 생성이 공유하는 기본 후보 이름; 무거운 모델 모듈은 import하지 않는다.
SUPPORTED_CANDIDATE_MODELS = frozenset({
    "logistic_regression_l1", "rbf_svm", "random_forest", "lightgbm", "xgboost",
})


# ==========================================
# 원본 데이터 source 하나의 위치와 CSV 읽기 옵션
# - source 이름과 파일 경로를 Profile에서 선언해 파일 역할을 코드에 고정하지 않음
# - read_csv_options는 구분자, header, encoding 등 원본 형식별 읽기 조건을 보관
# - 여러 source를 조합하는 방식은 adapter가 선택하고, 범용 코어는 source 목록만 관리
# ==========================================
@dataclass(frozen=True)
class IngestionSourceSpec:
    name: str
    path: Path
    read_csv_options: dict[str, Any]


# ==========================================
# 데이터셋 전용 원본 파일을 canonical table로 변환하는 ingestion 설정
# - adapter는 source 결합 규칙을 제공하고, source 개수·이름·읽기 방식은 Profile에서 관리
# - adapter_options는 adapter에 필요한 추가 규칙만 보관해 특정 데이터셋 필드를 코어에서 제거
# - 범용 split·모델링 코어는 canonical table 이후 단계만 담당
# ==========================================
@dataclass(frozen=True)
class IngestionSpec:
    adapter: str
    sources: tuple[IngestionSourceSpec, ...]
    adapter_options: dict[str, Any]

    def get_source(self, name: str) -> IngestionSourceSpec:
        """Profile에 선언된 이름으로 원본 source를 조회한다."""
        # source 이름은 adapter_options에서 참조하므로 순서가 아닌 이름으로 조회한다.
        for source in self.sources:
            if source.name == name:
                return source
        # 잘못된 source 이름은 병합 전에 명확한 Profile 오류로 중단한다.
        raise ValueError(f"ingestion source가 정의되어 있지 않습니다: {name}")


# ==========================================
# 데이터셋별 구조·Label·품질 규칙 Profile
# - 데이터 경로, Label, Timestamp, feature 선택 규칙을 데이터셋별 JSON으로 관리
# - feature_selection_mode는 prefix 또는 all_except_metadata 중 하나를 사용
# - Profile만 추가·교체하면 코어 코드 변경 없이 다른 데이터셋을 지원
# ==========================================
@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    input_path: Path
    label_column: str
    positive_label: int | float | str
    negative_label: int | float | str
    timestamp_column: str | None
    timestamp_format: str | None
    feature_selection_mode: str
    feature_column_prefix: str | None
    missing_ratio_threshold: float
    drop_zero_variance: bool
    id_columns: tuple[str, ...] = ()
    group_columns: tuple[str, ...] = ()
    excluded_feature_columns: tuple[str, ...] = ()
    categorical_feature_columns: tuple[str, ...] = ()
    ingestion: IngestionSpec | None = None


# ==========================================
# 후보 모델 비교용 Repeated Stratified Cross Validation 설정
# - 데이터셋의 Label 값은 DatasetSpec이 담당하고, CV 횟수만 실험 설정으로 관리
# - 같은 random_state로 모델 비교를 재현
# ==========================================
@dataclass(frozen=True)
class CrossValidationConfig:
    n_splits: int
    n_repeats: int
    random_state: int


# ==========================================
# 데이터셋과 독립적인 모델링 실험 Protocol
# - Threshold, PCA, CV처럼 모델 비교에 사용하는 정책만 보관
# - Label 값·컬럼명·결측 기준은 DatasetSpec에서 읽음
# ==========================================
@dataclass(frozen=True)
class ExperimentProtocol:
    default_threshold: float
    pca_explained_variance: float
    cv: CrossValidationConfig


# ==========================================
# OOF threshold 선택용 평가 정책
# - 최소 Recall과 최대 재검사 비율을 데이터셋과 독립적으로 관리
# - 미충족 시 임의 후보를 선택하지 않고 결과만 기록
# ==========================================
@dataclass(frozen=True)
class ThresholdPolicy:
    # 목표값은 사람이 설정하고 실제 threshold는 OOF 후보에서 계산한다.
    min_recall: float = 0.7
    max_reinspection_ratio: float = 0.2
    selection_rule: str = "min_reinspection"
    on_infeasible: str = "report_only"


# ==========================================
# Dataset Profile·실험 조건·평가 정책을 묶은 공통 설정
# ==========================================
@dataclass(frozen=True)
class FeatureReductionPolicy:
    # 비율은 0~1로 저장한다. AP 손실 정책과 OOF threshold 정책은 독립적이다.
    max_sensor_count: int
    target_ap_loss_ratio: float
    max_ap_loss_ratio: float
    comparison_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ModelingConfig:
    dataset: DatasetSpec
    experiment: ExperimentProtocol
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    top_k_feature_counts: tuple[int, ...]
    baseline_model: str
    candidate_models: tuple[str, ...]
    threshold_policy: ThresholdPolicy = ThresholdPolicy()
    # 정책을 생략한 다른 데이터셋에는 SECOM의 센서 수·손실 기준을 강제하지 않는다.
    reduction_policy: FeatureReductionPolicy | None = None


# ==========================================
# JSON 설정에서 필수 Object를 안전하게 읽기
# - 누락된 Key와 Object가 아닌 값을 명확한 오류로 변환
# - 프로젝트 설정과 Dataset Profile에서 공통으로 사용
# ==========================================
def _require_mapping(source: dict[str, Any], key: str) -> dict[str, Any]:
    # 지정한 Key가 존재하는지 먼저 확인
    if key not in source:
        raise ValueError(f"설정에 필수 key가 없습니다: {key}")

    # 하위 설정은 JSON Object여야 하므로 List·문자열·숫자는 차단
    value = source[key]
    if not isinstance(value, dict):
        raise ValueError(f"설정 key '{key}'는 object여야 합니다")

    return value


# ==========================================
# JSON 설정에서 필수 Scalar·List 값을 안전하게 읽기
# - Python의 bool은 int의 하위 타입이므로 정수 설정에서는 별도 차단
# - 타입 오류를 설정 파일 위치와 분리해 읽기 쉬운 오류로 제공
# ==========================================
def _require_value(
    source: dict[str, Any], key: str, expected_type: type | tuple[type, ...]
) -> Any:
    # 지정한 Key가 존재하는지 확인
    if key not in source:
        raise ValueError(f"설정에 필수 key가 없습니다: {key}")

    value = source[key]

    # bool은 random seed·fold 수처럼 정수를 기대하는 항목에 허용하지 않음
    integer_allowed = expected_type is int or (
        isinstance(expected_type, tuple) and int in expected_type
    )
    if integer_allowed and isinstance(value, bool):
        raise ValueError(f"설정 key '{key}'는 정수여야 합니다")

    # 설정값의 실제 타입이 기대한 JSON 타입과 같은지 확인
    if not isinstance(value, expected_type):
        expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        expected_name = " or ".join(item.__name__ for item in expected_types)
        raise ValueError(f"설정 key '{key}'는 {expected_name} 자료형이어야 합니다")

    return value


# ==========================================
# JSON 설정의 선택 문자열 목록을 tuple로 로드·검증
# - 선언하지 않은 역할 목록은 빈 tuple로 처리해 최소 Profile도 지원
# - 빈 값·중복·문자열 외 값을 조기에 차단해 컬럼 역할 모호성을 제거
# ==========================================
def _load_optional_column_names(source: dict[str, Any], key: str) -> tuple[str, ...]:
    # Profile에 해당 역할을 선언하지 않으면 빈 목록으로 처리한다.
    value = source.get(key, [])

    # 컬럼 역할은 실제 DataFrame 컬럼명과 비교하므로 비어 있지 않은 문자열만 허용한다.
    if not isinstance(value, list) or not all(
        isinstance(column, str) and column for column in value
    ):
        raise ValueError(f"설정 key '{key}'는 비어 있지 않은 문자열 목록이어야 합니다")
    
    # 같은 역할을 두 번 선언하면 feature 제외 규칙이 모호해지므로 차단한다.
    if len(value) != len(set(value)):
        raise ValueError(f"설정 key '{key}'에 중복값이 있으면 안 됩니다")
    
    return tuple(value)


# ==========================================
# JSON 파일을 읽어 최상위 Object인지 검증
# - 공통 실험 설정과 Dataset Profile에서 동일한 파일 오류 처리를 사용
# - UTF-8 JSON만 허용해 운영체제 기본 인코딩에 영향받지 않음
# ==========================================
def _load_json_object(path: Path) -> dict[str, Any]:
    # UTF-8 JSON 파일을 읽고 문법 오류를 명확하게 전달
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"설정 파일이 올바른 JSON 형식이 아닙니다: {path}") from error

    # 최상위 구조는 이름 기반 설정을 위한 JSON Object여야 함
    if not isinstance(raw_config, dict):
        raise ValueError(f"설정 파일의 최상위 구조는 object여야 합니다: {path}")

    return raw_config


# ==========================================
# Dataset Profile을 DatasetSpec 객체로 로드·검증
# - 데이터셋 전용 가정을 코어 코드가 아닌 Profile 파일에 격리
# - 다른 데이터셋은 같은 형식의 JSON Profile만 추가하면 됨
# ==========================================
def load_dataset_spec(
    profile_path: Path | str, *, project_root: Path | None = None
) -> DatasetSpec:
    # 문자열 또는 Path 입력을 일관된 경로 객체로 변환
    path = Path(profile_path)
    raw_profile = _load_json_object(path)

    # Dataset Profile의 구조 영역을 분리
    columns = _require_mapping(raw_profile, "columns")
    labels = _require_mapping(raw_profile, "labels")
    feature_columns = _require_mapping(raw_profile, "feature_columns")
    quality_rules = _require_mapping(raw_profile, "quality_rules")

    # 데이터셋 식별자와 Profile에 기록된 입력 경로를 읽음
    dataset_id = _require_value(raw_profile, "dataset_id", str)
    input_path = Path(_require_value(raw_profile, "input_path", str))
    # config.json이 있는 디렉터리를 기준으로 입력 경로를 절대 경로로 변환
    if project_root is not None and not input_path.is_absolute():
        input_path = project_root / input_path

    # Label과 Timestamp 컬럼 정의를 읽음
    label_column = _require_value(columns, "label", str)
    timestamp_column = columns.get("timestamp")
    timestamp_format = columns.get("timestamp_format")

    if timestamp_column is not None and not isinstance(timestamp_column, str):
        raise ValueError("설정 key 'columns.timestamp'는 문자열 또는 null이어야 합니다")
    
    if timestamp_format is not None and not isinstance(timestamp_format, str):
        raise ValueError("설정 key 'columns.timestamp_format'는 문자열 또는 null이어야 합니다")
    
    if (timestamp_column is None) != (timestamp_format is None):
        raise ValueError("columns.timestamp와 columns.timestamp_format은 함께 설정해야 합니다")

    # ID·그룹·명시적 제외 컬럼은 모델 입력에서 반드시 제외할 Dataset Profile 역할
    raw_column_roles = raw_profile.get("column_roles", {})
    if not isinstance(raw_column_roles, dict):
        raise ValueError("설정 key 'column_roles'는 object여야 합니다")
    
    id_columns = _load_optional_column_names(raw_column_roles, "id_columns")
    group_columns = _load_optional_column_names(raw_column_roles, "group_columns")

    excluded_feature_columns = _load_optional_column_names(
        raw_column_roles, "excluded_feature_columns"
    )

    role_columns = id_columns + group_columns + excluded_feature_columns

    if len(role_columns) != len(set(role_columns)):
        raise ValueError("column 역할 목록에 같은 column을 두 번 지정하면 안 됩니다")

    # Label·timestamp는 이미 모델 입력에서 제외되므로 역할 목록에 중복 선언하지 않음
    reserved_columns = {label_column}

    if timestamp_column is not None:
        reserved_columns.add(timestamp_column)

    duplicated_roles = [column for column in role_columns if column in reserved_columns]

    if duplicated_roles:
        raise ValueError(
            "column 역할 목록에 label 또는 timestamp column을 중복 지정하면 안 됩니다: "
            f"{duplicated_roles}"
        )

    # 범주형 feature는 명시적으로 선언하고, 나머지 선택 feature는 수치형으로 검증한다.
    raw_feature_types = raw_profile.get("feature_types", {})

    if not isinstance(raw_feature_types, dict):
        raise ValueError("설정 key 'feature_types'는 object여야 합니다")
    
    categorical_feature_columns = _load_optional_column_names(
        raw_feature_types, "categorical_columns"
    )

    invalid_categorical_roles = [
        column for column in categorical_feature_columns if column in reserved_columns
    ]

    if invalid_categorical_roles:
        raise ValueError(
            "feature_types.categorical_columns에 label 또는 timestamp를 포함하면 안 됩니다: "
            f"{invalid_categorical_roles}"
        )
    
    overlapping_roles = [
        column for column in categorical_feature_columns if column in role_columns
    ]

    if overlapping_roles:
        raise ValueError(
            "feature_types.categorical_columns에 feature가 아닌 column을 포함하면 안 됩니다: "
            f"{overlapping_roles}"
        )

    # 원본 Label 값은 숫자 또는 문자열만 허용하고 bool은 제외
    label_types = (int, float, str)
    positive_label = _require_value(labels, "positive", label_types)
    negative_label = _require_value(labels, "negative", label_types)
    
    if isinstance(positive_label, bool) or isinstance(negative_label, bool):
        raise ValueError("labels.positive와 labels.negative는 bool 값이면 안 됩니다")
    
    if positive_label == negative_label:
        raise ValueError("labels.positive와 labels.negative는 서로 달라야 합니다")

    # Feature 선택 방법을 읽고 prefix 방식일 때만 prefix 값을 요구
    feature_selection_mode = _require_value(feature_columns, "selection", str)
    feature_column_prefix = feature_columns.get("prefix")

    if feature_selection_mode not in {"prefix", "all_except_metadata"}:
        raise ValueError(
            "feature_columns.selection은 'prefix' 또는 'all_except_metadata'여야 합니다"
        )
    
    if feature_selection_mode == "prefix":
        if not isinstance(feature_column_prefix, str) or not feature_column_prefix:
            raise ValueError("prefix 방식의 feature_columns.prefix는 비어 있지 않은 문자열이어야 합니다")
    elif feature_column_prefix is not None:
        raise ValueError("all_except_metadata 방식의 feature_columns.prefix는 null이어야 합니다")

    # 결측률·분산 규칙은 Dataset Profile별 데이터 품질 정책으로 관리
    missing_ratio_threshold = float(
        _require_value(quality_rules, "missing_ratio_threshold", (int, float))
    )

    drop_zero_variance = _require_value(quality_rules, "drop_zero_variance", bool)

    if not 0.0 <= missing_ratio_threshold <= 1.0:
        raise ValueError("quality_rules.missing_ratio_threshold는 0 이상 1 이하여야 합니다")

    # 원본 파일 형식이 정의된 경우 source 목록과 adapter 설정을 함께 검증
    ingestion: IngestionSpec | None = None
    raw_ingestion = raw_profile.get("ingestion")

    if raw_ingestion is not None:
        if not isinstance(raw_ingestion, dict):
            raise ValueError("설정 key 'ingestion'은 object 또는 null이어야 합니다")

        # adapter는 source를 canonical table로 결합하는 규칙만 식별
        adapter = _require_value(raw_ingestion, "adapter", str)

        # 각 원본 source의 이름·경로·읽기 옵션을 Profile에서 독립적으로 읽음
        raw_sources = _require_value(raw_ingestion, "sources", list)

        if not raw_sources:
            raise ValueError("ingestion.sources에는 source가 하나 이상 있어야 합니다")
        
        sources: list[IngestionSourceSpec] = []

        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                raise ValueError("각 ingestion source는 object여야 합니다")
            
            source_name = _require_value(raw_source, "name", str)
            source_path = Path(_require_value(raw_source, "path", str))
            read_csv_options = _require_mapping(raw_source, "read_csv_options")

            if project_root is not None and not source_path.is_absolute():
                source_path = project_root / source_path

            sources.append(
                IngestionSourceSpec(
                    name=source_name,
                    path=source_path,
                    read_csv_options=read_csv_options,
                )
            )

        if len({source.name for source in sources}) != len(sources):
            raise ValueError("ingestion.sources의 name은 중복되면 안 됩니다")

        # adapter_options는 source 결합 규칙처럼 adapter 전용 설정을 담는다.
        adapter_options = raw_ingestion.get("adapter_options", {})

        if not isinstance(adapter_options, dict):
            raise ValueError("ingestion.adapter_options는 object여야 합니다")

        ingestion = IngestionSpec(
            adapter=adapter,
            sources=tuple(sources),
            adapter_options=adapter_options,
        )

    # 검증된 값을 변경 불가능한 DatasetSpec으로 반환
    return DatasetSpec(
        dataset_id=dataset_id,
        input_path=input_path,
        label_column=label_column,
        positive_label=positive_label,
        negative_label=negative_label,
        timestamp_column=timestamp_column,
        timestamp_format=timestamp_format,
        feature_selection_mode=feature_selection_mode,
        feature_column_prefix=feature_column_prefix,
        missing_ratio_threshold=missing_ratio_threshold,
        drop_zero_variance=drop_zero_variance,
        id_columns=id_columns,
        group_columns=group_columns,
        excluded_feature_columns=excluded_feature_columns,
        categorical_feature_columns=categorical_feature_columns,
        ingestion=ingestion,
    )


# ==========================================
# config.json과 선택된 Dataset Profile을 ModelingConfig로 로드·검증
# - config.json의 dataset_profile 경로는 config.json 위치 기준으로 해석
# - 실험 설정과 데이터셋 설정을 함께 반환해 호출부의 하드코딩을 제거
# ==========================================
def load_modeling_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> ModelingConfig:
    # 문자열 또는 Path 입력을 일관된 경로 객체로 변환
    path = Path(config_path)
    raw_config = _load_json_object(path)

    # Dataset Profile 경로를 공통 실험 설정 위치를 기준으로 해석
    profile_reference = _require_value(raw_config, "dataset_profile", str)
    profile_path = Path(profile_reference)

    if not profile_path.is_absolute():
        profile_path = path.parent / profile_path

    dataset = load_dataset_spec(profile_path, project_root=path.parent)

    # 실험·평가·특징 선택·모델 설정 영역을 분리
    experiment = _require_mapping(raw_config, "experiment")
    metrics = _require_mapping(raw_config, "metrics")
    feature_selection = _require_mapping(raw_config, "feature_selection")
    models = _require_mapping(raw_config, "models")
    cv = _require_mapping(experiment, "cross_validation")

    # 센서 축소 정책은 선택 사항이며, 명시한 전체·축소 비교 쌍에만 적용한다.
    reduction_policy = None

    if "reduction_policy" in feature_selection:
        reduction = _require_mapping(feature_selection, "reduction_policy")
        sensor_count = _require_value(reduction, "max_sensor_count", int)
        target_loss = float(_require_value(reduction, "target_ap_loss_ratio", (int, float)))
        max_loss = float(_require_value(reduction, "max_ap_loss_ratio", (int, float)))
        pairs = _require_value(reduction, "comparison_pairs", list)
    
        if sensor_count < 1 or not 0 <= target_loss <= max_loss <= 1:
            raise ValueError("센서 축소 정책은 센서 수 >= 1, 0 <= 목표 AP 하락률 <= 최대 AP 하락률 <= 1이어야 합니다")
        
        if not pairs or any(
            not isinstance(pair, list) or len(pair) != 2
            or not all(isinstance(name, str) and name.strip() for name in pair)
            or pair[0] == pair[1] for pair in pairs
        ):
            raise ValueError("comparison_pairs에는 서로 다른 전체·축소 실험 이름 두 개씩을 지정해야 합니다")
        
        parsed_pairs = tuple(tuple(pair) for pair in pairs)

        if len(set(parsed_pairs)) != len(parsed_pairs):
            raise ValueError("comparison_pairs에는 중복 비교 쌍을 지정할 수 없습니다")
        
        reduction_policy = FeatureReductionPolicy(sensor_count, target_loss, max_loss, parsed_pairs)

    # 기존 설정 파일도 로드하되, 정책이 있으면 필수 항목·범위를 엄격히 검증한다.
    policy = raw_config.get("threshold_policy", asdict(ThresholdPolicy()))

    if not isinstance(policy, dict):
        raise ValueError("threshold_policy는 object여야 합니다")
    
    min_recall = float(_require_value(policy, "min_recall", (int, float)))
    max_reinspection = float(_require_value(policy, "max_reinspection_ratio", (int, float)))
    selection_rule = _require_value(policy, "selection_rule", str)
    on_infeasible = _require_value(policy, "on_infeasible", str)

    if not 0 <= min_recall <= 1 or not 0 <= max_reinspection <= 1:
        raise ValueError("threshold_policy의 목표 비율은 0 이상 1 이하의 유한한 값이어야 합니다")
    
    if selection_rule != "min_reinspection":
        raise ValueError("threshold_policy.selection_rule은 min_reinspection만 지원합니다")
    
    if on_infeasible != "report_only":
        raise ValueError("threshold_policy.on_infeasible은 report_only만 지원합니다")

    # 데이터셋과 독립적인 수치 기반 실험 설정을 읽음
    random_state = _require_value(experiment, "random_state", int)
    default_threshold = float(_require_value(experiment, "default_threshold", (int, float)))

    pca_explained_variance = float(
        _require_value(experiment, "pca_explained_variance", (int, float))
    )

    n_splits = _require_value(cv, "n_splits", int)
    n_repeats = _require_value(cv, "n_repeats", int)

    # List 기반 설정은 실행 중 수정되지 않도록 Tuple로 변환
    primary_metrics = tuple(_require_value(metrics, "primary", list))
    secondary_metrics = tuple(_require_value(metrics, "secondary", list))

    top_k_feature_counts = tuple(
        _require_value(feature_selection, "top_k_feature_counts", list)
    )

    baseline_model = _require_value(models, "baseline", str)
    candidate_models = tuple(_require_value(models, "candidates", list))

    # 실험 수치의 허용 범위를 검증
    if not 0.0 < default_threshold < 1.0:
        raise ValueError("experiment.default_threshold는 0 초과 1 미만이어야 합니다")
    if not 0.0 < pca_explained_variance < 1.0:
        raise ValueError("experiment.pca_explained_variance는 0 초과 1 미만이어야 합니다")
    if not 0 <= random_state < 2**32:
        raise ValueError("experiment.random_state는 0 이상 2**32 미만이어야 합니다")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("cross_validation은 n_splits >= 2 및 n_repeats >= 1이어야 합니다")

    # 문자열 목록과 Top-K 목록의 내용까지 검증
    if not primary_metrics or not all(isinstance(metric, str) for metric in primary_metrics):
        raise ValueError("metrics.primary에는 지표 이름이 하나 이상 있어야 합니다")
    if not secondary_metrics or not all(isinstance(metric, str) for metric in secondary_metrics):
        raise ValueError("metrics.secondary에는 지표 이름이 있어야 합니다")
    if not baseline_model:
        raise ValueError("models.baseline은 비어 있으면 안 됩니다")
    if not candidate_models or not all(isinstance(model, str) for model in candidate_models):
        raise ValueError("models.candidates에는 모델 이름이 하나 이상 있어야 합니다")
    
    # 이름 오타가 후보를 조용히 제외하거나 빈 결과의 KeyError로 이어지지 않게 차단한다.
    supported_models = SUPPORTED_CANDIDATE_MODELS
    unknown_models = sorted(set(candidate_models) - supported_models)
    
    if unknown_models:
        raise ValueError(f"models.candidates에 지원하지 않는 모델이 있습니다: {unknown_models}")
    if baseline_model != "logistic_regression":
        raise ValueError("models.baseline은 logistic_regression만 지원합니다")
    if not top_k_feature_counts or not all(
        isinstance(count, int) and not isinstance(count, bool) and count > 0
        for count in top_k_feature_counts
    ):
        raise ValueError("feature_selection.top_k_feature_counts에는 양의 정수만 포함되어야 합니다")

    # Dataset Profile과 실험 설정을 하나의 범용 모델링 설정 객체로 반환
    return ModelingConfig(
        dataset=dataset,
        experiment=ExperimentProtocol(
            default_threshold=default_threshold,
            pca_explained_variance=pca_explained_variance,
            cv=CrossValidationConfig(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            ),
        ),
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics,
        top_k_feature_counts=top_k_feature_counts,
        baseline_model=baseline_model,
        candidate_models=candidate_models,
        threshold_policy=ThresholdPolicy(min_recall, max_reinspection, selection_rule, on_infeasible),
        reduction_policy=reduction_policy,
    )


# config.json과 활성 Dataset Profile을 한 번 로드해 다른 Script에서 공통으로 import
MODELING_CONFIG: Final = load_modeling_config()

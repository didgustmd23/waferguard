"""사용자 분석 모듈의 참조 예제: 행별 결측 특징 수의 빈도표."""

import pandas as pd

from ..contracts import TabularInput
from ..module_contracts import ModuleContract, ModulePort
from ..module_registry import RegisteredModule


def validate_settings(settings):
    """완전 관측 행을 포함할지만 선택한다. 임의 경로·실행 옵션은 받지 않는다."""
    if set(settings) - {"include_complete"}:
        raise ValueError("행 결측 분포 모듈에 허용되지 않은 설정입니다.")

    include_complete = settings.get("include_complete", True)

    if type(include_complete) is not bool:
        raise ValueError("완전 관측 행 포함 여부는 bool이어야 합니다.")

    return {"include_complete": include_complete}


def calculate(data, *, include_complete=True):
    """선택 특징만 분석한다. 라벨·ID·시간은 제외하고 원본을 변경하지 않는다."""
    if not isinstance(data, TabularInput):
        raise TypeError("검증할 표형 입력 계약이 필요합니다.")

    settings = validate_settings({"include_complete": include_complete})
    data.validate()
    features = data.frame.loc[:, list(data.roles.feature_columns)]
    counts = features.isna().sum(axis=1).value_counts().sort_index()

    if not settings["include_complete"]:
        counts = counts.loc[counts.index != 0]

    return pd.DataFrame({"missing_feature_count": counts.index.to_numpy(dtype="int64"),
                         "row_count": counts.to_numpy(dtype="int64")})


def validate_result(result):
    """결과의 정수 빈도·축 구조를 검사한다. 원본 대비 통계 정답 인증은 아니다."""
    if not isinstance(result, pd.DataFrame) or list(result.columns) != ["missing_feature_count", "row_count"]:
        return False

    for column in result:
        values = result[column]

        if not pd.api.types.is_integer_dtype(values) or pd.api.types.is_bool_dtype(values) or values.isna().any():
            return False

    return bool(
        result["missing_feature_count"].is_unique
        and result["missing_feature_count"].is_monotonic_increasing
        and (result["missing_feature_count"] >= 0).all()
        and (result["row_count"] > 0).all()
    )


def build_module():
    """프로젝트의 신뢰된 등록 초기화 코드에서만 호출한다. 승인 기능은 없다."""
    contract = ModuleContract(
        module_id="user_missing_row_histogram", version="1.0.0", kind="analysis",
        inputs=(ModulePort("data", "tabular_data"),),
        outputs=(ModulePort("result", "missing_row_histogram_result"),),
        implementation_id="reviewed_missing_row_histogram_v1",
        result_validator_id="missing_row_histogram_result_v1",
        settings_fields=("include_complete",),
    )

    return RegisteredModule(contract, calculate, validate_settings, validate_result)

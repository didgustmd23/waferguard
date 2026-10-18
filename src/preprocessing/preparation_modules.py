"""변환 모듈의 고정 등록부. 직접 호출은 로컬 코어이며 승인 실행이 아니다."""

from dataclasses import fields

import numpy as np
import pandas as pd

from .contracts import TabularInput
from .module_contracts import ModuleContract, ModulePort
from .module_registry import ModuleRegistry, RegisteredModule
from .transform import PreparationRules, build_preparation, _validate_state


def _rules(settings):
    """기존 규칙 객체로 기본값·타입·조합 검증을 재사용한다."""
    return {"rules": PreparationRules(**settings)}


def _no_settings(settings):
    if settings:
        raise ValueError("적용 단계에서 저장된 전처리 규칙을 변경할 수 없습니다.")

    return {}


def derive_candidate(data, *, rules):
    """호출자가 검증한 Train에서만 기준을 산출한다. 승인·분할 증거는 실행기 책임이다."""
    if not isinstance(data, TabularInput) or not isinstance(rules, PreparationRules):
        raise TypeError("표형 입력 계약과 전처리 규칙이 필요합니다.")

    data.validate()
    pipeline = build_preparation(data.roles, rules)

    # 특징만 전달한다. 라벨·ID·시간을 통계 산출에 사용하지 않는다.
    pipeline.fit(data.frame.loc[:, list(data.roles.feature_columns)])

    return pipeline


def validate_candidate(pipeline):
    """메모리 상태의 구조 검사다. 저장 파일·승인·출처 검증을 대체하지 않는다."""
    try:
        transformer = pipeline.named_steps["preprocess"]
        _validate_state(transformer.state_, transformer.roles, transformer.rules)
        expected = [
            f"feature_{index}__{record['column']}__{offset}"
            for index, record in enumerate(transformer.state_["records"])
            for offset in range(record["width"])
        ]

        return list(pipeline.get_feature_names_out()) == expected
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def apply_candidate(data, candidate):
    """저장 기준만 적용한다. fit 또는 배치 통계 재산출을 호출하지 않는다."""
    if not isinstance(data, TabularInput) or not validate_candidate(candidate):
        raise ValueError("검증된 입력과 전처리 후보가 필요합니다.")

    data.validate()

    if data.roles != candidate.named_steps["preprocess"].roles:
        raise ValueError("입력 역할이 후보 산출 시의 역할과 다릅니다.")

    return candidate.transform(data.frame.loc[:, list(data.roles.feature_columns)])


def validate_prepared(result):
    """수치 출력 구조만 확인한다. 허용 NaN·행 연결은 기존 파일 검증기가 확인한다."""
    if not isinstance(result, pd.DataFrame) or not result.columns.is_unique or result.empty:
        return False

    try:
        return not np.isinf(result.to_numpy(dtype=float)).any()
    except (ValueError, TypeError):
        return False


def build_preparation_registry():
    """별도 변환 등록부를 봉인한다. 기존 EDA worker 허용 목록을 확장하지 않는다."""
    registry = ModuleRegistry()
    common = {
        "version": "1.0.0", "kind": "transformation",
        "preconditions": ("snapshot_verified", "tabular_roles_validated", "confirmed_partition"),
    }
    propose = ModuleContract(
        **common, module_id="tabular_prepare_propose",
        inputs=(ModulePort("data", "tabular_data"),),
        outputs=(ModulePort("candidate", "preparation_candidate"),),
        implementation_id="builtin_preparation_propose_v1",
        result_validator_id="preparation_candidate_object_v1",
        settings_fields=tuple(field.name for field in fields(PreparationRules)),
        criteria_policy="derive_on_train", derivation_partition="train",
    )
    apply = ModuleContract(
        **common, module_id="tabular_prepare_apply",
        inputs=(ModulePort("data", "tabular_data"), ModulePort("candidate", "preparation_candidate")),
        outputs=(ModulePort("prepared", "prepared_tabular_data"),),
        implementation_id="builtin_preparation_apply_v1",
        result_validator_id="prepared_tabular_object_v1", criteria_policy="apply_saved",
    )
    registry.register(RegisteredModule(propose, derive_candidate, _rules, validate_candidate))
    registry.register(RegisteredModule(apply, apply_candidate, _no_settings, validate_prepared))

    return registry.seal()

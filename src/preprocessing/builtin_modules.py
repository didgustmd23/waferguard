"""초기 내장 모듈 정의. 고정 코어를 재사용하며 등록 중 데이터를 읽거나 실행하지 않는다."""

from dataclasses import fields
from functools import partial

import numpy as np
import pandas as pd

from . import eda
from .module_contracts import ModuleContract, ModulePort
from .module_registry import ModuleRegistry, RegisteredModule
from .quality import QualityFinding, QualityReport, QualityRules, inspect_quality


def _empty_settings(settings):
    """초기 EDA 계산 모듈에 임의 인자를 추가하지 않는다."""
    if settings:
        raise ValueError("이 분석 모듈은 추가 설정을 받지 않습니다.")

    return {}


def _quality_settings(settings):
    """기존 QualityRules를 설정 스키마로 재사용한다."""
    labels = settings.get("allowed_labels")

    # JSON 배열을 불변 tuple로 변환한다. 원래 설정 객체는 변경하지 않는다.
    if isinstance(labels, list):
        settings["allowed_labels"] = tuple(labels)

    rules = QualityRules(**settings)

    return {"rules": rules}


def _validate_table(result, *, columns):
    """계산 결과의 타입·컬럼명·중복을 검사한다. 수치의 정답 인증은 아니다."""
    structure_valid = (
        isinstance(result, pd.DataFrame)
        and result.columns.is_unique and result.index.is_unique
        and tuple(result.columns) == columns
    )

    if not structure_valid:
        return False

    for name in ("count", "missing_count", "observed_unique", "missing_ratio"):
        if name not in result:
            continue

        series = result[name]

        # 빈 범주/라벨 표는 object dtype일 수 있다. 값이 없으면 헤더만 검사한다.
        if series.empty:
            continue

        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return False

        values = series.to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values < 0).any():
            return False

        if name == "missing_ratio":
            if (values > 1).any():
                return False
        elif (values != np.floor(values)).any():
            return False

    return True


def _validate_numeric(result):
    if not isinstance(result, pd.DataFrame):
        return False

    # 수치 특징이 없는 입력은 기존 코어가 빈 0×0 표로 표현한다.
    if result.shape == (0, 0):
        return True

    return _validate_table(result, columns=("count", "mean", "std", "min", "25%", "50%", "75%", "max"))


def _validate_correlations(result):
    """세 행렬의 형태·축·범위를 확인한다. 원본은 다시 읽지 않는다."""
    if type(result) is not dict or set(result) != {"pearson", "spearman", "correlation_support"}:
        return False

    axes = None

    for name, matrix in result.items():
        if not isinstance(matrix, pd.DataFrame) or not matrix.index.is_unique or not matrix.columns.is_unique:
            return False

        if not matrix.index.equals(matrix.columns):
            return False

        if axes is not None and not matrix.index.equals(axes):
            return False

        axes = matrix.index

        try:
            values = matrix.to_numpy(dtype=float)
        except (TypeError, ValueError):
            return False

        if name == "correlation_support":
            if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
                return False
        elif np.isinf(values).any() or (np.abs(values[~np.isnan(values)]) > 1.000000001).any():
            return False

    return True


def _validate_quality(result):
    """검사 수행 성공과 품질 통과를 구분하고 통지 수준·상태의 일관성을 확인한다."""
    if not isinstance(result, QualityReport) or type(result.findings) is not tuple:
        return False

    for finding in result.findings:
        if not isinstance(finding, QualityFinding) or finding.level not in ("review", "block"):
            return False

        if type(finding.code) is not str or not finding.code:
            return False

        if finding.column is not None and type(finding.column) is not str:
            return False

        if finding.count is not None and (type(finding.count) is not int or finding.count < 0):
            return False

    levels = {finding.level for finding in result.findings}
    expected = "blocked_input" if "block" in levels else "review_required" if "review" in levels else "passed"

    return result.status == expected


def build_builtin_registry():
    """검토된 내장 6개 기능을 구성·봉인한다. import 문자열·사용자 코드는 받지 않는다."""
    registry = ModuleRegistry()
    definitions = (
        ("eda_missing", eda.analyze_missing, partial(_validate_table, columns=("missing_count", "missing_ratio", "observed_unique"))),
        ("eda_numeric", eda.analyze_numeric, _validate_numeric),
        ("eda_categories", eda.analyze_categories, partial(_validate_table, columns=("feature", "value", "count"))),
        ("eda_target", eda.analyze_target, partial(_validate_table, columns=("label", "count"))),
        ("eda_correlations", eda.analyze_correlations, _validate_correlations),
    )

    for module_id, calculate, validator in definitions:
        contract = ModuleContract(
            module_id=module_id, version="1.0.0", kind="analysis",
            inputs=(ModulePort("data", "tabular_data"),),
            outputs=(ModulePort("result", module_id + "_result"),),
            implementation_id="builtin_" + module_id + "_v1",
            result_validator_id=module_id + "_result_v1",
        )
        registry.register(RegisteredModule(contract, calculate, _empty_settings, validator))

    quality = ModuleContract(
        module_id="tabular_quality", version="1.0.0", kind="validation",
        inputs=(ModulePort("data", "tabular_data"),),
        outputs=(ModulePort("report", "quality_report"),),
        implementation_id="builtin_tabular_quality_v1", result_validator_id="quality_object_v1",
        settings_fields=tuple(field.name for field in fields(QualityRules)),
        # 내용이 잘못돼도 품질 보고서를 만들도록 EDA와 같은 엄격 내용 검증을 선행하지 않는다.
        preconditions=("snapshot_verified", "safe_loader_completed"),
    )
    registry.register(RegisteredModule(quality, inspect_quality, _quality_settings, _validate_quality))

    return registry.seal()

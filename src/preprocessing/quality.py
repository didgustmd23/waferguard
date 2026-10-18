"""데이터를 수정하지 않는 범용 품질 검사와 로컬 보고서."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ColumnRoles, TabularInput


def _label_key(value):
    """문자열·숫자·bool 라벨을 구분하고 수치 1과 1.0은 같은 값으로 취급한다."""
    if isinstance(value, np.generic):
        value = value.item()

    if type(value) is bool:
        return ("bool", value)

    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return ("number", value)

    if type(value) is str and value.strip():
        return ("string", value)

    raise ValueError("허용 라벨은 비어 있지 않은 문자열·유한 숫자·bool이어야 합니다.")


# ==========================================
# 프로파일별 품질 기준
# - 기준 초과는 검토 대상으로 기록하며 특징을 자동 삭제하지 않는다.
# - optional target이 없으면 라벨 검사를 생략한다.
# - ignore는 해당 데이터 기준의 통지 수준이며 구조 오류를 우회하지 않는다.
# ==========================================
@dataclass(frozen=True)
class QualityRules:
    missing_ratio_threshold: float = 0.5
    allowed_labels: tuple | None = None
    high_missing: str = "review"
    constant_feature: str = "review"
    all_missing_feature: str = "review"
    target_missing: str = "review"
    duplicate_id: str = "review"
    identifier_missing: str = "review"
    group_missing: str = "review"
    timestamp_missing: str = "review"
    timestamp_format: str | None = None

    def __post_init__(self):
        threshold = self.missing_ratio_threshold

        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("결측률 검토 기준은 0~1 사이 유한 숫자여야 합니다.")

        levels = (
            self.high_missing, self.constant_feature, self.all_missing_feature,
            self.target_missing, self.duplicate_id, self.identifier_missing,
            self.group_missing, self.timestamp_missing,
        )

        if any(level not in ("ignore", "review", "block") for level in levels):
            raise ValueError("품질 통지 수준은 ignore·review·block 중 하나여야 합니다.")

        if self.timestamp_format is not None:
            if not isinstance(self.timestamp_format, str) or not self.timestamp_format.strip():
                raise ValueError("시간 형식은 문자열 또는 null이어야 합니다.")

        if self.allowed_labels is not None:
            if not isinstance(self.allowed_labels, tuple) or not self.allowed_labels:
                raise ValueError("허용 라벨은 비어 있지 않은 tuple 또는 None이어야 합니다.")

            keys = [_label_key(value) for value in self.allowed_labels]

            if len(keys) != len(set(keys)):
                raise ValueError("허용 라벨을 중복 지정할 수 없습니다.")


@dataclass(frozen=True)
class QualityFinding:
    """컬럼명과 건수는 로컬 보고서용이며 LLM에 전달하지 않는다."""

    code: str
    level: str
    column: str | None = None
    count: int | None = None


@dataclass(frozen=True)
class QualityReport:
    """검사 결과. 원본 행·라벨 값·표본 목록을 포함하지 않는다."""

    status: str
    findings: tuple[QualityFinding, ...]

    def to_dict(self):
        return {"status": self.status, "findings": [asdict(finding) for finding in self.findings]}


def _missing(series):
    """NaN·None·공백 문자열을 결측으로 판단하되 실제 값을 바꾸지 않는다."""
    blank = series.astype(object).map(lambda value: isinstance(value, str) and not value.strip())

    return series.isna() | blank


def _add(findings, code, level, column=None, count=None):
    """ignore 정책은 기록하지 않고 나머지는 고정 코드로 추가한다."""
    if level != "ignore":
        findings.append(QualityFinding(code, level, column, count))


def _finish(findings):
    levels = {finding.level for finding in findings}
    status = "passed"

    if "block" in levels:
        status = "blocked_input"
    elif "review" in levels:
        status = "review_required"

    return QualityReport(status, tuple(findings))


def _check_structure(data, findings):
    """후속 컬럼 검사가 가능한 구조인지 먼저 확인한다."""
    if not isinstance(data, TabularInput) or not isinstance(data.frame, pd.DataFrame):
        _add(findings, "INVALID_TABLE", "block")

        return False

    if not isinstance(data.roles, ColumnRoles):
        _add(findings, "INVALID_COLUMN_ROLES", "block")

        return False

    frame = data.frame

    if not frame.columns.is_unique or any(not isinstance(name, str) for name in frame.columns):
        _add(findings, "INVALID_COLUMNS", "block")

        return False

    if frame.empty:
        _add(findings, "EMPTY_TABLE", "block")

        return False

    missing = set(data.roles.required_columns) - set(frame.columns)

    for name in sorted(missing):
        _add(findings, "REQUIRED_COLUMN_MISSING", "block", name)

    return not missing


def _check_features(data, rules, findings):
    """결측률·상수 후보와 수치형·유한값 조건을 검사한다."""
    for name in data.roles.feature_columns:
        series = data.frame[name]
        missing = _missing(series)
        count = int(missing.sum())

        if count / len(series) > rules.missing_ratio_threshold:
            _add(findings, "HIGH_MISSING_RATIO", rules.high_missing, name, count)

        try:
            unique = series.loc[~missing].nunique()
        except TypeError:
            _add(findings, "UNSUPPORTED_FEATURE_VALUES", "block", name)

            continue

        if unique == 0:
            _add(findings, "ALL_MISSING_FEATURE", rules.all_missing_feature, name, count)
        elif unique == 1:
            _add(findings, "CONSTANT_FEATURE", rules.constant_feature, name)

        if name not in data.roles.numeric:
            continue

        numeric_type = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        if not numeric_type:
            _add(findings, "INVALID_NUMERIC_TYPE", "block", name)

            continue

        values = series.dropna().to_numpy(dtype=float)
        invalid = int((~np.isfinite(values)).sum())

        if invalid:
            _add(findings, "NONFINITE_NUMERIC_VALUE", "block", name, invalid)


def _check_target(data, rules, findings):
    if data.roles.target is None:
        return

    name = data.roles.target
    series = data.frame[name]
    missing = _missing(series)

    if missing.any():
        _add(findings, "TARGET_MISSING", rules.target_missing, name, int(missing.sum()))

    allowed = None if rules.allowed_labels is None else {_label_key(value) for value in rules.allowed_labels}
    invalid = 0
    invalid_type = 0

    for value in series.loc[~missing]:
        try:
            key = _label_key(value)

            if allowed is not None:
                invalid += key not in allowed
        except ValueError:
            invalid_type += 1

    if invalid_type:
        _add(findings, "INVALID_TARGET_TYPE", "block", name, invalid_type)

    if invalid:
        _add(findings, "UNKNOWN_TARGET_LABEL", "block", name, invalid)


def _check_identifiers(data, rules, findings):
    names = data.roles.identifiers

    if not names:
        return

    identifiers = data.frame.loc[:, list(names)]

    if any(not identifiers[name].map(pd.api.types.is_scalar).all() for name in names):
        _add(findings, "INVALID_IDENTIFIER_TYPE", "block")

        return

    missing = pd.concat([_missing(identifiers[name]) for name in names], axis=1).any(axis=1)

    if missing.any():
        _add(findings, "IDENTIFIER_MISSING", rules.identifier_missing, count=int(missing.sum()))

    # 여러 ID 컬럼은 하나의 복합 ID로 검사하며 결측 ID는 중복 계산에서 제외한다.
    try:
        duplicates = int(identifiers.loc[~missing].duplicated(keep=False).sum())
    except TypeError:
        _add(findings, "INVALID_IDENTIFIER_TYPE", "block")

        return

    if duplicates:
        _add(findings, "DUPLICATE_IDENTIFIER", rules.duplicate_id, count=duplicates)


def _check_group_and_time(data, rules, findings):
    if data.roles.group is not None:
        name = data.roles.group
        missing = _missing(data.frame[name])

        if not data.frame[name].map(pd.api.types.is_scalar).all():
            _add(findings, "INVALID_GROUP_TYPE", "block", name)

        if missing.any():
            _add(findings, "GROUP_MISSING", rules.group_missing, name, int(missing.sum()))

    if data.roles.timestamp is None:
        return

    name = data.roles.timestamp
    series = data.frame[name]
    missing = _missing(series)

    if missing.any():
        _add(findings, "TIMESTAMP_MISSING", rules.timestamp_missing, name, int(missing.sum()))

    if pd.api.types.is_datetime64_any_dtype(series) or (~missing).sum() == 0:
        return

    if rules.timestamp_format is None:
        _add(findings, "TIMESTAMP_FORMAT_REQUIRED", "review", name)

        return

    # 문자열 형식을 명시한 경우만 파싱한다. 결과는 검사에만 사용한다.
    try:
        parsed = pd.to_datetime(series.loc[~missing], format=rules.timestamp_format, errors="coerce", utc=True)
    except (TypeError, ValueError):
        _add(findings, "INVALID_TIMESTAMP", "block", name)

        return
    invalid = int(parsed.isna().sum())

    if invalid:
        _add(findings, "INVALID_TIMESTAMP", "block", name, invalid)


def inspect_quality(data: TabularInput, rules=None):
    """원본을 수정하지 않고 통과·검토 필요·차단 상태를 계산한다."""
    rules = QualityRules() if rules is None else rules

    if not isinstance(rules, QualityRules):
        raise TypeError("품질 기준은 QualityRules여야 합니다.")

    findings = []

    if not _check_structure(data, findings):
        return _finish(findings)

    _check_features(data, rules, findings)
    _check_target(data, rules, findings)
    _check_identifiers(data, rules, findings)
    _check_group_and_time(data, rules, findings)

    return _finish(findings)


def write_quality_report(report, output_dir):
    """실제 상세 근거를 로컬 JSON·Markdown에만 저장한다."""
    if not isinstance(report, QualityReport):
        raise TypeError("품질 보고서 객체가 필요합니다.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "quality.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    lines = ["# 품질 검사\n", f"검사 상태: `{report.status}`\n", ""]

    for finding in report.findings:
        lines.append(f"- `{finding.code}` / {finding.level} / 컬럼: {finding.column} / 건수: {finding.count}")

    lines.append("\n검토·차단 근거는 로컬 전용입니다. 데이터·라벨·특징을 자동 수정하지 않았습니다.")
    (output / "quality.md").write_text("\n".join(lines), encoding="utf-8")

    return {"status": report.status}

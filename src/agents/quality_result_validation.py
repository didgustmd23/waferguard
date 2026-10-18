"""원본 재파싱 없이 품질 결과의 필드·역할·정책·판정 일관성을 검사한다."""

import json

from src.preprocessing.profiles import _unique_pairs
from .artifacts import _root
from .isolated_eda import _read_file


FIXED_BLOCK_CODES = frozenset({
    "INVALID_TABLE", "INVALID_COLUMN_ROLES", "INVALID_COLUMNS", "EMPTY_TABLE",
    "INPUT_LOAD_CONTRACT_INVALID", "REQUIRED_COLUMN_MISSING", "UNSUPPORTED_FEATURE_VALUES",
    "INVALID_NUMERIC_TYPE", "NONFINITE_NUMERIC_VALUE", "INVALID_TARGET_TYPE",
    "UNKNOWN_TARGET_LABEL", "INVALID_IDENTIFIER_TYPE", "INVALID_GROUP_TYPE", "INVALID_TIMESTAMP",
})
POLICY_CODES = {
    "HIGH_MISSING_RATIO": "high_missing", "ALL_MISSING_FEATURE": "all_missing_feature",
    "CONSTANT_FEATURE": "constant_feature", "TARGET_MISSING": "target_missing",
    "IDENTIFIER_MISSING": "identifier_missing", "DUPLICATE_IDENTIFIER": "duplicate_id",
    "GROUP_MISSING": "group_missing", "TIMESTAMP_MISSING": "timestamp_missing",
}
FEATURE_CODES = frozenset({"HIGH_MISSING_RATIO", "ALL_MISSING_FEATURE", "CONSTANT_FEATURE", "UNSUPPORTED_FEATURE_VALUES", "INVALID_NUMERIC_TYPE", "NONFINITE_NUMERIC_VALUE"})
TARGET_CODES = frozenset({"TARGET_MISSING", "INVALID_TARGET_TYPE", "UNKNOWN_TARGET_LABEL"})
TIME_CODES = frozenset({"TIMESTAMP_MISSING", "TIMESTAMP_FORMAT_REQUIRED", "INVALID_TIMESTAMP"})
GLOBAL_CODES = frozenset({"INVALID_TABLE", "INVALID_COLUMN_ROLES", "INVALID_COLUMNS", "EMPTY_TABLE", "INPUT_LOAD_CONTRACT_INVALID", "INVALID_IDENTIFIER_TYPE", "IDENTIFIER_MISSING", "DUPLICATE_IDENTIFIER"})


def _invalid_number(value):
    raise ValueError("품질 결과에 비표준 숫자를 사용할 수 없습니다.")


def read_quality_report(directory, profile):
    """검증된 보고서 객체만 반환한다. 잘못된 결과는 None이며 자동 통과시키지 않는다."""
    try:
        root = _root(directory)

        if {path.name for path in root.iterdir()} != {"quality.json", "quality.md"}:
            return None

        payload = _read_file(root / "quality.json", 1024 * 1024)
        actual = json.loads(payload, object_pairs_hook=_unique_pairs, parse_constant=_invalid_number)

        if not isinstance(actual, dict) or set(actual) != {"status", "findings"} or not isinstance(actual["findings"], list):
            return None

        if len(actual["findings"]) > 6 * len(profile.roles.required_columns) + 20:
            return None

        roles = profile.roles
        levels = set()
        seen = set()
        lines = ["# 품질 검사\n", f"검사 상태: `{actual['status']}`\n", ""]

        for finding in actual["findings"]:
            if not isinstance(finding, dict) or set(finding) != {"code", "level", "column", "count"}:
                return None

            code, level, column, count = (finding[key] for key in ("code", "level", "column", "count"))

            if not isinstance(code, str) or not isinstance(level, str) or (column is not None and not isinstance(column, str)):
                return None

            if code in FIXED_BLOCK_CODES:
                expected = "block"
            elif code in POLICY_CODES:
                expected = getattr(profile.quality_rules, POLICY_CODES[code])
            elif code == "TIMESTAMP_FORMAT_REQUIRED":
                expected = "review"
            else:
                return None

            if level != expected or level not in {"review", "block"}:
                return None

            if code in GLOBAL_CODES:
                valid_column = column is None
            elif code == "REQUIRED_COLUMN_MISSING":
                valid_column = column in roles.required_columns
            elif code in FEATURE_CODES:
                valid_column = column in roles.feature_columns
            elif code in TARGET_CODES:
                valid_column = roles.target is not None and column == roles.target
            elif code in TIME_CODES:
                valid_column = roles.timestamp is not None and column == roles.timestamp
            else:
                valid_column = code in {"GROUP_MISSING", "INVALID_GROUP_TYPE"} and roles.group is not None and column == roles.group

            if not valid_column or (count is not None and (type(count) is not int or not 0 < count <= 10**12)):
                return None

            key = (code, column)

            if key in seen:
                return None

            seen.add(key)
            levels.add(level)
            lines.append(f"- `{code}` / {level} / 컬럼: {column} / 건수: {count}")

        expected_status = "blocked_input" if "block" in levels else "review_required" if "review" in levels else "passed"

        if actual["status"] != expected_status:
            return None

        lines.append("\n검토·차단 근거는 로컬 전용입니다. 데이터·라벨·특징을 자동 수정하지 않았습니다.")
        markdown = _read_file(root / "quality.md", 1024 * 1024).decode("utf-8").replace("\r\n", "\n")

        if markdown != "\n".join(lines):
            return None

        return actual
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return None

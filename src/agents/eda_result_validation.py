"""원본을 다시 파싱하지 않는 부모의 EDA 결과 계약 검사. 통계 정답 인증은 아니다."""

import csv
from io import StringIO
import math

from src.preprocessing.eda_report import report_markdown
from .adapters.tabular import REPORT_FILES
from .artifacts import _root
from .isolated_eda import _read_file, _valid_png
from .worker_contract import _decode


MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_TABLE_ROWS = 100000


def _rows(payload):
    """표준 CSV 파서로 결과를 읽고 행 수·열 수를 제한한다."""
    result = []

    for row in csv.reader(StringIO(payload.decode("utf-8-sig")), strict=True):
        if len(result) >= MAX_TABLE_ROWS or len(row) > 1025:
            raise ValueError("EDA 결과 표의 행·열 제한을 초과했습니다.")

        result.append(row)

    if not result or any(len(row) != len(result[0]) for row in result):
        raise ValueError("EDA 결과 표의 구조가 올바르지 않습니다.")

    return result


def _count(value, maximum):
    if len(value) > 20 or not value.isascii() or not value.isdigit():
        raise ValueError("집계 값은 제한된 음이 아닌 정수여야 합니다.")

    number = int(value)

    if number > maximum:
        raise ValueError("집계 값이 표본 수를 초과했습니다.")

    return number


def _number(value, *, missing=False):
    if value == "" and missing:
        return None

    number = float(value)

    if not math.isfinite(number):
        raise ValueError("집계 값에 비유한 수를 사용할 수 없습니다.")

    return number


def validate_eda_report(directory, profile):
    """고정 파일·헤더·범위·역할·내부 일관성을 검사하며 원본 데이터는 읽지 않는다."""
    try:
        root = _root(directory)
        paths = tuple(root.iterdir())

        if {path.name for path in paths} != REPORT_FILES:
            return False

        # 크기를 먼저 제한하고 파일마다 링크·하드 링크·일반 파일 여부를 재검사한다.
        if sum(path.lstat().st_size for path in paths) > MAX_TOTAL_BYTES:
            return False

        payloads = {path.name: _read_file(path, MAX_FILE_BYTES) for path in paths}
        summary = _decode(payloads["summary.json"].decode("utf-8"))
        roles = profile.roles
        expected_keys = {"row_count", "feature_count", "numeric_feature_count", "categorical_feature_count", "repeated_feature_rows", "group_count", "target_present"}

        if set(summary) != expected_keys:
            return False

        for key in ("row_count", "feature_count", "numeric_feature_count", "categorical_feature_count", "repeated_feature_rows"):
            if type(summary[key]) is not int or not 0 <= summary[key] <= 10**12:
                return False

        count = summary["row_count"]

        if (
            count == 0 or summary["feature_count"] != len(roles.feature_columns)
            or summary["numeric_feature_count"] != len(roles.numeric)
            or summary["categorical_feature_count"] != len(roles.categorical)
            or summary["repeated_feature_rows"] > count
            or summary["target_present"] is not (roles.target is not None)
        ):
            return False

        groups = summary["group_count"]

        if roles.group is None:
            if groups is not None:
                return False
        elif type(groups) is not int or not 0 <= groups <= count:
            return False

        if _decode(payloads["SUCCESS.json"].decode()) != {"status": "eda_complete"}:
            return False

        if payloads["eda.md"].decode("utf-8").replace("\r\n", "\n") != report_markdown():
            return False

        if any(not _valid_png(payloads[name]) for name in REPORT_FILES if name.endswith(".png")):
            return False

        missing = _rows(payloads["missing.csv"])

        if missing[0] != ["feature", "missing_count", "missing_ratio", "observed_unique"] or [row[0] for row in missing[1:]] != list(roles.feature_columns):
            return False

        for _, raw_count, ratio, unique in missing[1:]:
            absent = _count(raw_count, count)

            if not math.isclose(_number(ratio), absent / count, abs_tol=1e-12) or _count(unique, count) > count - absent:
                return False

        statistics = _rows(payloads["numeric_statistics.csv"])

        if roles.numeric:
            if statistics[0] != ["", "count", "mean", "std", "min", "25%", "50%", "75%", "max"] or [row[0] for row in statistics[1:]] != list(roles.numeric):
                return False

            for row in statistics[1:]:
                if not 0 <= _number(row[1]) <= count:
                    return False

                for value in row[2:]:
                    _number(value, missing=True)
        elif statistics != [[""]]:
            return False

        for name, header in (("categorical_distribution.csv", ["", "feature", "value", "count"]), ("target_distribution.csv", ["", "label", "count"])):
            table = _rows(payloads[name])

            if table[0] != header:
                return False

            totals = {}

            for row in table[1:]:
                key = row[1] if name.startswith("categorical") else "target"

                if name.startswith("categorical") and key not in roles.categorical:
                    return False

                totals[key] = totals.get(key, 0) + _count(row[-1], count)

            expected = set(roles.categorical) if name.startswith("categorical") else ({"target"} if roles.target else set())

            if set(totals) != expected or any(total != count for total in totals.values()):
                return False

        for name in ("pearson.csv", "spearman.csv", "correlation_support.csv"):
            table = _rows(payloads[name])

            if table[0] != ["", *roles.numeric] or [row[0] for row in table[1:]] != list(roles.numeric):
                return False

            for row in table[1:]:
                for value in row[1:]:
                    if name == "correlation_support.csv":
                        _count(value, count)
                    else:
                        number = _number(value, missing=True)

                        if number is not None and not -1.000000001 <= number <= 1.000000001:
                            return False

        return True
    except (OSError, ValueError, TypeError, KeyError, csv.Error, OverflowError, RecursionError):
        return False

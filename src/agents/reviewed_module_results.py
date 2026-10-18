"""참조 모듈의 결과를 표준 파서로 검사한다. 원본 파싱·확장 모듈 import 없음."""

import csv

from .artifacts import _root
from .eda_result_validation import _rows, _count
from .isolated_eda import _read_file
from .quality_result_validation import read_quality_report
from .reviewed_module_contract import demo_profile, module_identity
from .worker_contract import _decode


def read_result(directory, config_id, include_complete):
    """품질 중단·정수 범위·합성 정답을 확인한다. 일반 데이터의 수학적 인증은 아니다."""
    old_limit = csv.field_size_limit(1024 * 1024)

    try:
        root = _root(directory)
        result = _decode(_read_file(root / "pipeline_result.json", 65536).decode("utf-8"))

        if (set(result) != {"version", "config_id", "module", "row_count", "status", "completed_steps"}
                or result["version"] != "reviewed_module_result_v1" or result["config_id"] != config_id
                or result["module"] != module_identity(include_complete)
                or type(result["row_count"]) is not int or result["row_count"] != 8):
            return None

        quality = read_quality_report(root / "quality", demo_profile())

        if quality is None or quality["status"] != result["status"]:
            return None

        passed = result["status"] == "passed"
        expected_steps = ["quality", "row_missing"] if passed else ["quality"]

        if result["completed_steps"] != expected_steps or {path.name for path in root.iterdir()} != {"pipeline_result.json", *expected_steps}:
            return None

        if passed:
            child = _root(root / "row_missing")

            if {path.name for path in child.iterdir()} != {"missing_row_histogram.csv"}:
                return None

            rows = _rows(_read_file(child / "missing_row_histogram.csv", 65536))

            if rows[0] != ["missing_feature_count", "row_count"]:
                return None

            histogram = [(_count(row[0], 3), _count(row[1], 8)) for row in rows[1:] if len(row) == 2]
            expected = [(0, 7), (1, 1)] if include_complete else [(1, 1)]

            if len(histogram) != len(rows) - 1 or histogram != expected:
                return None

        return result
    except (OSError, ValueError, TypeError, KeyError, IndexError, csv.Error):
        return None
    finally:
        csv.field_size_limit(old_limit)

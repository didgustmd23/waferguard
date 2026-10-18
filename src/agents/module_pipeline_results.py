"""모듈 worker의 파일 결과를 제한된 표준 파서로 검사한다. 원본 CSV는 읽지 않는다."""

from dataclasses import replace
import csv
import json
import math

from src.preprocessing.quality import QualityRules
from .artifacts import _root
from .eda_result_validation import _rows, _count, _number
from .isolated_eda import _read_file
from .module_pipeline_contract import MODULE_FILES
from .quality_result_validation import read_quality_report
from .worker_contract import _decode


MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024


def _validate_table(payload, module_id, filename, roles, count):
    """역할·헤더·범위·집계의 내부 일관성을 확인한다. 실제 통계 정답은 인증하지 않는다."""
    rows = _rows(payload)

    if module_id == "eda_missing":
        if rows[0] != ["feature", "missing_count", "missing_ratio", "observed_unique"] or [row[0] for row in rows[1:]] != list(roles.feature_columns):
            return False

        for _, missing, ratio, unique in rows[1:]:
            absent = _count(missing, count)

            if not math.isclose(_number(ratio), absent / count, abs_tol=1e-12) or _count(unique, count) > count - absent:
                return False

    elif module_id == "eda_numeric":
        if not roles.numeric:
            return rows == [[""]]

        if rows[0] != ["", "count", "mean", "std", "min", "25%", "50%", "75%", "max"] or [row[0] for row in rows[1:]] != list(roles.numeric):
            return False

        for row in rows[1:]:
            observed = _number(row[1])

            if not 0 <= observed <= count or not observed.is_integer():
                return False

            for value in row[2:]:
                _number(value, missing=True)

    elif module_id in ("eda_categories", "eda_target"):
        categorical = module_id == "eda_categories"
        header = ["", "feature", "value", "count"] if categorical else ["", "label", "count"]

        if rows[0] != header:
            return False

        totals = {}

        for row in rows[1:]:
            key = row[1] if categorical else "target"
            totals[key] = totals.get(key, 0) + _count(row[-1], count)

        expected = set(roles.categorical) if categorical else ({"target"} if roles.target else set())

        if set(totals) != expected or any(value != count for value in totals.values()):
            return False

    elif module_id == "eda_correlations":
        if rows[0] != ["", *roles.numeric] or [row[0] for row in rows[1:]] != list(roles.numeric):
            return False

        for row in rows[1:]:
            for value in row[1:]:
                if filename == "correlation_support.csv":
                    _count(value, count)
                else:
                    number = _number(value, missing=True)

                    if number is not None and not -1.000000001 <= number <= 1.000000001:
                        return False
    else:
        return False

    return True


def read_pipeline_result(directory, plan, profile):
    """선택한 단계의 정확한 파일만 허용한다. 검토/차단 보고서는 마지막 단계에서 멈춰야 한다."""
    try:
        root = _root(directory)
        result = _decode(_read_file(root / "pipeline_result.json", 65536).decode("utf-8"))

        if set(result) != {"version", "pipeline_signature", "row_count", "status", "completed_steps"}:
            return None

        if result["version"] != "module_pipeline_result_v1" or result["pipeline_signature"] != plan.signature:
            return None

        count, ids = result["row_count"], result["completed_steps"]

        if type(count) is not int or not 1 <= count <= 10**12 or type(ids) is not list or not 1 <= len(ids) <= len(plan.steps):
            return None

        if ids != [step.step_id for step in plan.steps[:len(ids)]]:
            return None

        if {path.name for path in root.iterdir()} != {"pipeline_result.json", *ids}:
            return None

        total_bytes = (root / "pipeline_result.json").lstat().st_size
        quality_status = "passed"

        for index, step in enumerate(plan.steps[:len(ids)]):
            child = _root(root / step.step_id)
            files = MODULE_FILES[step.module_id]

            if {path.name for path in child.iterdir()} != set(files):
                return None

            total_bytes += sum((child / name).lstat().st_size for name in files)

            if total_bytes > MAX_TOTAL_BYTES:
                return None

            if step.module_id == "tabular_quality":
                effective = json.loads(step.effective_settings_json)
                rules = effective["rules"]

                if rules["allowed_labels"] is not None:
                    rules["allowed_labels"] = tuple(rules["allowed_labels"])

                report = read_quality_report(child, replace(profile, quality_rules=QualityRules(**rules)))

                if report is None:
                    return None

                quality_status = report["status"]

                if quality_status != "passed" and index != len(ids) - 1:
                    return None
            else:
                for name in files:
                    if not _validate_table(_read_file(child / name, MAX_FILE_BYTES), step.module_id, name, profile.roles, count):
                        return None

        if result["status"] != quality_status or (quality_status == "passed" and len(ids) != len(plan.steps)):
            return None

        return result
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError, csv.Error):
        return None

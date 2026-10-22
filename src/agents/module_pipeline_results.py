"""모듈 worker의 파일 결과를 제한된 표준 파서로 검사한다. 원본 CSV는 읽지 않는다."""

from dataclasses import replace
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from src.preprocessing.input_profile import (
    DEDUPE_INPUT, DEDUPE_MAP, DEDUPE_RULE, DEDUPE_VERSION, DTYPE_GROUPS, HEADER_DEDUPE_ID, IQR_MULTIPLIER,
    MAX_HEADER_BYTES, MIN_IQR_VALUES, MODULE_ID as INPUT_PROFILE_MODULE_ID, OUTPUT_FILES, PROFILE_COLUMNS,
    SUMMARY_VERSION, UTF8_BOM, dedupe_names, parse_header, render_header, split_header,
)
from src.preprocessing.quality import QualityRules
from .artifacts import _root
from .eda_result_validation import _rows, _count, _number
from .isolated_eda import _read_file
from .module_pipeline_contract import MODULE_FILES
from .quality_result_validation import read_quality_report
from .input_snapshot import PAYLOAD_NAME
from .worker_contract import _decode
from .worker_paths import _path


MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
# 진단 표의 실수 칸은 pandas 기본 표기만 받는다. 공백·부호 장식으로 값을 숨기지 못하게 한다.
_PLAIN_NUMBER = re.compile(r"-?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?")


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


def _validate_profile_row(row, count):
    """한 컬럼의 무한대·유한값 집계 항등식과 통계 순서를 확인한다. 통계 정답은 인증하지 않는다."""
    values = dict(zip(PROFILE_COLUMNS, row))
    group = values["dtype_group"]

    for name in ("missing_ratio", "mean", "std", "min", "q25", "median", "q75", "max", "iqr_lower", "iqr_upper"):
        if values[name] != "" and _PLAIN_NUMBER.fullmatch(values[name]) is None:
            return False

    if not values["column"] or group not in DTYPE_GROUPS:
        return False

    absent = _count(values["missing_count"], count)
    present = count - absent

    if not math.isclose(_number(values["missing_ratio"]), absent / count, abs_tol=1e-12) or _count(values["observed_unique"], present) > present:
        return False

    if group == "text":
        _count(values["blank_string_count"], present)
    elif values["blank_string_count"] != "":
        return False

    numeric_fields = PROFILE_COLUMNS[6:]

    if group != "numeric":
        return all(values[name] == "" for name in numeric_fields)

    finite = _count(values["finite_count"], present)

    if finite + _count(values["pos_inf_count"], present) + _count(values["neg_inf_count"], present) != present:
        return False

    stats = {name: _number(values[name], missing=True) for name in ("mean", "std", "min", "q25", "median", "q75", "max")}
    iqr = [values[name] for name in ("iqr_lower", "iqr_upper", "outlier_candidate_count")]

    if finite == 0:
        return all(value is None for value in stats.values()) and iqr == ["", "", ""]

    ordered = [stats[name] for name in ("min", "q25", "median", "q75", "max")]

    if None in ordered or stats["mean"] is None or ordered != sorted(ordered) or (stats["std"] is None) != (finite == 1):
        return False

    if stats["std"] is not None and stats["std"] < 0:
        return False

    tolerance = 1e-9 * max(1.0, abs(ordered[0]), abs(ordered[-1]))

    if not ordered[0] - tolerance <= stats["mean"] <= ordered[-1] + tolerance:
        return False

    if finite < MIN_IQR_VALUES:
        return iqr == ["", "", ""]

    lower, upper = _number(iqr[0]), _number(iqr[1])
    margin = IQR_MULTIPLIER * (stats["q75"] - stats["q25"])

    return (
        math.isclose(lower, stats["q25"] - margin, rel_tol=1e-9, abs_tol=tolerance)
        and math.isclose(upper, stats["q75"] + margin, rel_tol=1e-9, abs_tol=tolerance)
        and lower <= upper and _count(iqr[2], finite) <= finite
    )


def read_input_profile_result(directory, plan, columns):
    """고정 진단 계획의 파일 2개만 허용한다. 일반 모듈 결과 검증과 분리된 읽기 함수다.

    columns는 부모가 승인된 snapshot 헤더에서 읽은 컬럼명 순서다. 결과 행은 이 순서와 같아야 한다.
    """
    try:
        root = _root(directory)

        if len(plan.steps) != 1 or plan.steps[0].module_id != INPUT_PROFILE_MODULE_ID:
            return None

        step_id = plan.steps[0].step_id
        result = _decode(_read_file(root / "pipeline_result.json", 65536).decode("utf-8"))

        if set(result) != {"version", "pipeline_signature", "row_count", "status", "completed_steps"}:
            return None

        count = result["row_count"]

        if (
            result["version"] != "module_pipeline_result_v1" or result["pipeline_signature"] != plan.signature
            or result["status"] != "passed" or result["completed_steps"] != [step_id]
            or type(count) is not int or not 1 <= count <= 10**12
        ):
            return None

        child = _root(root / step_id)

        if {path.name for path in root.iterdir()} != {"pipeline_result.json", step_id} or {path.name for path in child.iterdir()} != set(OUTPUT_FILES):
            return None

        if sum((child / name).lstat().st_size for name in OUTPUT_FILES) > MAX_TOTAL_BYTES:
            return None

        summary = _decode(_read_file(child / "input_summary.json", 65536).decode("utf-8"))

        if set(summary) != {"version", "row_count", "column_count", "repeated_rows"} or summary["version"] != SUMMARY_VERSION:
            return None

        width, repeated = summary["column_count"], summary["repeated_rows"]

        if summary["row_count"] != count or width != len(columns) or type(repeated) is not int or not 0 <= repeated <= count:
            return None

        rows = _rows(_read_file(child / "input_profile.csv", MAX_FILE_BYTES))

        if tuple(rows[0]) != PROFILE_COLUMNS or len(rows) - 1 != width or width < 1:
            return None

        if [row[0] for row in rows[1:]] != list(columns) or not all(_validate_profile_row(row, count) for row in rows[1:]):
            return None

        return result
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError, csv.Error):
        return None


def _same_tail(derived, original, offset_derived, offset_original):
    """두 파일의 지정 위치 이후 바이트가 같은지 스트리밍으로 비교하고 파생 파일 해시를 함께 계산한다."""
    digest = hashlib.sha256()

    with open(derived, "rb") as left, open(original, "rb") as right:
        digest.update(left.read(offset_derived))
        right.seek(offset_original)

        while True:
            chunk = left.read(1024 * 1024)

            if chunk != right.read(len(chunk)) or (not chunk and right.read(1)):
                return None

            if not chunk:
                return digest.hexdigest()

            digest.update(chunk)


def read_header_dedupe_result(directory, plan, snapshot_directory, snapshot):
    """중복 컬럼명 정비 결과를 원본 snapshot과 대조한다. 부모는 CSV를 다시 쓰지 않고 읽어 비교만 한다.

    - 대응표는 원본 헤더에서 같은 규칙으로 다시 계산한 결과와 정확히 같아야 한다.
    - 파생 입력은 "정비한 헤더 + 원본 헤더 뒤의 바이트"와 바이트 단위로 같아야 한다.
    - 바뀐 이름이 없으면 파생 입력 파일이 없어야 한다.
    """
    try:
        root = _root(directory)

        if len(plan.steps) != 1 or plan.steps[0].module_id != HEADER_DEDUPE_ID:
            return None

        step_id = plan.steps[0].step_id
        result = _decode(_read_file(root / "pipeline_result.json", 65536).decode("utf-8"))

        if result != {"version": "header_dedupe_result_v1", "pipeline_signature": plan.signature,
                      "status": "passed", "completed_steps": [step_id]}:
            return None

        if {path.name for path in root.iterdir()} != {"pipeline_result.json", step_id}:
            return None

        child = _root(root / step_id)
        mapping = _decode(_read_file(child / DEDUPE_MAP, 4 * 1024 * 1024).decode("utf-8"))
        original = _path(Path(snapshot_directory) / PAYLOAD_NAME, directory=False)

        with open(original, "rb") as stream:
            bom, header, terminator = split_header(stream.read(MAX_HEADER_BYTES))

        names = parse_header(header)
        renamed = dedupe_names(names)
        changed = [index for index, (old, new) in enumerate(zip(names, renamed)) if old != new]
        derived_header = render_header(renamed, terminator) if changed else None
        expected = {
            "version": DEDUPE_VERSION, "rule": DEDUPE_RULE, "original_sha256": snapshot.sha256,
            "bom": bom, "line_terminator": terminator, "original_header_bytes": len(header),
            "derived_header_bytes": None if derived_header is None else len(derived_header),
            "column_count": len(names), "changed_positions": changed,
            "columns": [{"position": index, "original": old, "renamed": new} for index, (old, new) in enumerate(zip(names, renamed))],
        }

        if mapping != expected:
            return None

        files = {path.name for path in child.iterdir()}

        if not changed:
            return {"status": "passed", "changed": False, "mapping": mapping} if files == {DEDUPE_MAP} else None

        if files != {DEDUPE_MAP, DEDUPE_INPUT}:
            return None

        derived = _path(child / DEDUPE_INPUT, directory=False)
        prefix = (UTF8_BOM if bom else b"") + derived_header
        skipped = len(UTF8_BOM) * bom + len(header)

        if derived.stat().st_size != snapshot.size_bytes - skipped + len(prefix):
            return None

        with open(derived, "rb") as stream:
            if stream.read(len(prefix)) != prefix:
                return None

        digest = _same_tail(derived, original, len(prefix), skipped)

        if digest is None:
            return None

        return {"status": "passed", "changed": True, "mapping": mapping,
                "derived_sha256": digest, "derived_size": derived.stat().st_size}
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError, csv.Error):
        return None

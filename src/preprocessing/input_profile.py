"""역할 지정 전 입력 진단. 기존 EDA 계산을 재사용하며 역할 확정·값 변경·파일 외 출력은 하지 않는다."""

from collections import Counter
import csv
from io import StringIO
import json
import shutil

import numpy as np
import pandas as pd

from . import eda
from .builtin_modules import _empty_settings
from .contracts import ColumnRoles, TabularInput
from .module_contracts import ModuleContract, ModulePort
from .module_registry import ModuleRegistry, RegisteredModule
from .pipeline import validate_pipeline


MODULE_ID = "input_profile"
VERSION = "1.0.0"
SUMMARY_VERSION = "input_profile_v1"
OUTPUT_FILES = ("input_profile.csv", "input_summary.json")
IQR_MULTIPLIER = 1.5
MIN_IQR_VALUES = 4
DTYPE_GROUPS = ("numeric", "boolean", "text", "other")
PROFILE_COLUMNS = (
    "column", "dtype_group", "missing_count", "missing_ratio", "observed_unique", "blank_string_count",
    "pos_inf_count", "neg_inf_count", "finite_count", "mean", "std", "min", "q25", "median", "q75", "max",
    "iqr_lower", "iqr_upper", "outlier_candidate_count",
)
COUNT_COLUMNS = (
    "missing_count", "observed_unique", "blank_string_count", "pos_inf_count", "neg_inf_count",
    "finite_count", "outlier_candidate_count",
)
# 검토된 단일 진단 계획. worker에도 복사되어 부모와 같은 상수로 전체 대조·서명 재계산을 한다.
_PLAN_JSON = json.dumps({
    "schema_version": "module_pipeline_v1", "pipeline_id": "input_profile_diagnostic",
    "input": {"contract_id": "tabular_data", "contract_version": 1, "partition": "unsplit"},
    "steps": [{"step_id": "input_profile", "module_id": MODULE_ID, "version": VERSION, "inputs": {"data": "source.data"}}],
}, sort_keys=True)


# ==========================================
# 중복 컬럼명 자동 정비 (사용자가 허용한 경우만)
# - 원본·기존 snapshot은 바꾸지 않고 새 파생 입력 파일만 만든다.
# - 헤더 레코드의 중복 이름만 바꾸고, 헤더 뒤의 바이트는 그대로 복사한다.
# - 빈 이름·깨진 CSV는 고치지 않고 거부한다.
# ==========================================
HEADER_DEDUPE_ID = "header_dedupe"
DEDUPE_RULE = "number_duplicates_v1"
DEDUPE_VERSION = "header_dedupe_v1"
DEDUPE_MAP = "header_map.json"
DEDUPE_INPUT = "derived_input.csv"
MAX_HEADER_BYTES = 1024 * 1024
UTF8_BOM = bytes((0xEF, 0xBB, 0xBF))
NEWLINE = bytes((0x0A,))
CRLF = bytes((0x0D, 0x0A))
TERMINATORS = {"crlf": CRLF.decode("ascii"), "lf": NEWLINE.decode("ascii")}
_DEDUPE_PLAN_JSON = json.dumps({
    "schema_version": "module_pipeline_v1", "pipeline_id": "input_header_dedupe",
    "input": {"contract_id": "tabular_data", "contract_version": 1, "partition": "unsplit"},
    "steps": [{"step_id": "header_dedupe", "module_id": HEADER_DEDUPE_ID, "version": VERSION, "inputs": {"data": "source.data"}}],
}, sort_keys=True)


def header_dedupe_pipeline():
    """중복 컬럼명 정비 고정 계획의 새 복사본을 반환한다."""
    return json.loads(_DEDUPE_PLAN_JSON)


def split_header(prefix):
    """앞부분 바이트에서 첫 CSV 레코드의 끝을 찾는다. 따옴표 안 줄바꿈은 레코드 안으로 본다."""
    bom = prefix.startswith(UTF8_BOM)
    data = prefix[len(UTF8_BOM):] if bom else prefix
    start = quotes = 0

    while True:
        newline = data.find(NEWLINE, start)

        if newline < 0:
            raise ValueError("헤더 레코드의 끝을 상한 안에서 찾지 못했습니다.")

        quotes += data.count(b'"', start, newline)

        if quotes % 2 == 0:
            end = newline + 1
            return bom, data[:end], "crlf" if data[end - 2:end] == CRLF else "lf"

        start = newline + 1


def parse_header(header_bytes):
    """헤더 바이트가 정확히 한 CSV 레코드일 때만 이름 목록을 반환한다."""
    try:
        # strict: 닫히지 않은 따옴표·따옴표 뒤 문자를 오류로 본다. 헤더 끝 탐색과 csv 해석이 어긋나는
        # 입력(예: 인용되지 않은 필드 안의 따옴표)은 한 레코드로 맞지 않아 아래 검사에서 거부된다.
        rows = list(csv.reader(StringIO(header_bytes.decode("utf-8"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as error:
        raise ValueError("CSV 헤더를 읽지 못했습니다.") from error

    if len(rows) != 1 or not rows[0]:
        raise ValueError("헤더는 정확히 한 레코드여야 합니다.")

    return rows[0]


def render_header(names, terminator):
    """정비한 이름을 원래 줄 끝 표기(crlf·lf)로 한 CSV 레코드로 만든다."""
    stream = StringIO()
    csv.writer(stream, lineterminator=TERMINATORS[terminator]).writerow(names)
    return stream.getvalue().encode("utf-8")


def dedupe_names(names):
    """중복 이름만 이름_1~이름_n으로 바꾼다. 기존 이름과 충돌하면 구분자를 늘려 연속 번호를 유지한다."""
    if any(type(name) is not str or not name.strip() for name in names):
        raise ValueError("빈 컬럼명은 자동 정비 범위가 아닙니다.")

    counts = Counter(names)
    taken = set(names)
    renamed = list(names)

    for base in dict.fromkeys(name for name in names if counts[name] > 1):
        for separator in ("_", "__", "___", "____"):
            candidates = [f"{base}{separator}{index}" for index in range(1, counts[base] + 1)]

            if not taken.intersection(candidates):
                break
        else:
            raise ValueError("기존 이름과 충돌하지 않는 번호 이름을 만들 수 없습니다.")

        taken.update(candidates)
        numbers = iter(candidates)
        renamed = [next(numbers) if name == base else name for name in renamed]

    return renamed


def build_header_dedupe(source, directory, original_sha256):
    """격리 worker에서만 호출한다. 대응표를 쓰고, 바뀐 이름이 있을 때만 파생 입력을 새로 만든다."""
    directory.mkdir(exist_ok=False)

    with open(source, "rb") as stream:
        bom, header, terminator = split_header(stream.read(MAX_HEADER_BYTES))
        names = parse_header(header)
        renamed = dedupe_names(names)
        changed = [index for index, (old, new) in enumerate(zip(names, renamed)) if old != new]
        derived_header = None

        if changed:
            derived_header = render_header(renamed, terminator)

            if parse_header(derived_header) != renamed:
                raise ValueError("정비한 헤더를 다시 읽은 결과가 다릅니다.")

            with open(directory / DEDUPE_INPUT, "xb") as target:
                target.write(UTF8_BOM if bom else b"")
                target.write(derived_header)
                stream.seek(len(UTF8_BOM) * bom + len(header))
                shutil.copyfileobj(stream, target, 1024 * 1024)

    mapping = {
        "version": DEDUPE_VERSION, "rule": DEDUPE_RULE, "original_sha256": original_sha256,
        "bom": bom, "line_terminator": terminator, "original_header_bytes": len(header),
        "derived_header_bytes": None if derived_header is None else len(derived_header),
        "column_count": len(names), "changed_positions": changed,
        "columns": [{"position": index, "original": old, "renamed": new} for index, (old, new) in enumerate(zip(names, renamed))],
    }

    with open(directory / DEDUPE_MAP, "x", encoding="utf-8") as target:
        target.write(_canonical(mapping))

    return mapping


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_profile_pipeline():
    """고정 진단 계획의 새 복사본을 반환한다. 호출자가 바꿔도 상수에는 반영되지 않는다."""
    return json.loads(_PLAN_JSON)


def _group(series):
    """자료형 기준 내부 계산 그룹. 사용자 역할로 저장하지 않는다."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "text"

    return "other"


def _value(number):
    return None if pd.isna(number) else float(number)


def profile_input(frame):
    """원본 갈래(결측·고유값·무한대·공백)와 유한값 갈래(기초 통계·IQR 후보)를 나눠 계산한다."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("DataFrame이 필요합니다.")

    columns = tuple(frame.columns)
    # 원본 갈래: categorical은 dtype·무한대 검사를 받지 않으므로 값을 바꾸지 않고 그대로 센다.
    original = TabularInput(frame, ColumnRoles(categorical=columns))
    summary = eda.analyze_summary(original)
    missing = eda.analyze_missing(original)
    groups = {name: _group(frame[name]) for name in columns}
    numeric = tuple(name for name in columns if groups[name] == "numeric")
    finite = None

    if numeric:
        # 유한값 갈래: 복사본에서만 무한대를 비우고 기존 수치 통계를 호출한다.
        values = frame.loc[:, list(numeric)].astype(float).replace([np.inf, -np.inf], np.nan)
        finite = eda.analyze_numeric(TabularInput(values, ColumnRoles(numeric=numeric)))

    rows = []

    for name in columns:
        series = frame[name]
        row = dict.fromkeys(PROFILE_COLUMNS)
        row.update(
            column=name, dtype_group=groups[name],
            missing_count=int(missing.at[name, "missing_count"]),
            missing_ratio=float(missing.at[name, "missing_ratio"]),
            observed_unique=int(missing.at[name, "observed_unique"]),
        )

        if groups[name] == "text":
            row["blank_string_count"] = int(series.map(lambda value: isinstance(value, str) and not value.strip()).sum())

        if groups[name] == "numeric":
            values = series.to_numpy(dtype=float)
            stats = finite.loc[name]
            row.update(
                pos_inf_count=int(np.isposinf(values).sum()), neg_inf_count=int(np.isneginf(values).sum()),
                finite_count=int(stats["count"]), mean=_value(stats["mean"]), std=_value(stats["std"]),
                min=_value(stats["min"]), q25=_value(stats["25%"]), median=_value(stats["50%"]),
                q75=_value(stats["75%"]), max=_value(stats["max"]),
            )
            observed = values[np.isfinite(values)]

            # transform.py의 iqr_clip과 같은 공식이다. 후보 수만 세고 값은 바꾸지 않는다.
            if len(observed) >= MIN_IQR_VALUES:
                q1, q3 = np.quantile(observed, [0.25, 0.75])
                margin = IQR_MULTIPLIER * (q3 - q1)
                lower, upper = float(q1 - margin), float(q3 + margin)
                row.update(
                    iqr_lower=lower, iqr_upper=upper,
                    outlier_candidate_count=int(((observed < lower) | (observed > upper)).sum()),
                )

        rows.append(row)

    table = pd.DataFrame(rows, columns=list(PROFILE_COLUMNS))
    # 빈칸이 섞인 건수 열도 CSV에서 정수 표기를 유지한다.
    table[list(COUNT_COLUMNS)] = table[list(COUNT_COLUMNS)].astype("Int64")

    return {
        "columns": table,
        "summary": {
            "version": SUMMARY_VERSION, "row_count": summary["row_count"],
            "column_count": len(columns), "repeated_rows": summary["repeated_feature_rows"],
        },
    }


def _validate_profile(result):
    """메모리 결과의 형식과 유한성만 확인한다. 상세 일관성은 부모 파일 검증이 다시 본다."""
    if type(result) is not dict or set(result) != {"columns", "summary"}:
        return False

    table, summary = result["columns"], result["summary"]

    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != PROFILE_COLUMNS or type(summary) is not dict:
        return False

    if len(table) != summary.get("column_count") or not table["dtype_group"].isin(DTYPE_GROUPS).all():
        return False

    numbers = table.loc[:, list(PROFILE_COLUMNS[2:])].astype(float).to_numpy()

    return bool(np.isfinite(numbers[~np.isnan(numbers)]).all())


def write_input_profile(result, directory):
    """고정 파일 2개만 새 폴더에 쓴다. 범주 원본 값은 결과에 포함되지 않는다."""
    directory.mkdir(exist_ok=False)
    result["columns"].to_csv(directory / OUTPUT_FILES[0], index=False, encoding="utf-8-sig")
    (directory / OUTPUT_FILES[1]).write_text(_canonical(result["summary"]), encoding="utf-8")


def build_diagnostic_registry():
    """진단 모듈 하나만 봉인 등록한다. 내장 등록부·모듈 목록·LLM 제안 목록에는 넣지 않는다."""
    registry = ModuleRegistry()
    contract = ModuleContract(
        module_id=MODULE_ID, version=VERSION, kind="analysis",
        inputs=(ModulePort("data", "tabular_data"),),
        outputs=(ModulePort("result", "input_profile_result"),),
        implementation_id="builtin_input_profile_v1", result_validator_id="input_profile_result_v1",
        # 역할 지정 전 단계이므로 tabular_roles_validated는 요구하지 않는다.
        preconditions=("snapshot_verified", "safe_loader_completed"),
    )
    registry.register(RegisteredModule(contract, profile_input, _empty_settings, _validate_profile))
    dedupe = ModuleContract(
        module_id=HEADER_DEDUPE_ID, version=VERSION, kind="transformation",
        inputs=(ModulePort("data", "tabular_data"),),
        outputs=(ModulePort("derived", "tabular_data"),),
        implementation_id="builtin_header_dedupe_v1", result_validator_id="header_dedupe_result_v1",
        # CSV 내용 적재 전에 헤더만 다루므로 snapshot 검증만 선행한다.
        preconditions=("snapshot_verified",),
    )
    registry.register(RegisteredModule(dedupe, build_header_dedupe, _empty_settings, lambda result: type(result) is dict))

    return registry.seal()


def bind_input_profile_plan(config):
    """고정 진단·정비 계획 중 하나와 정확히 같은 설정만 받아 서명을 다시 계산한다."""
    fixed = next((plan for plan in (input_profile_pipeline(), header_dedupe_pipeline())
                  if _canonical(plan) == _canonical(config)), None)

    if fixed is None:
        raise ValueError("고정 입력 진단 계획과 다른 설정입니다.")

    bound = json.loads(_canonical(config))
    plan = validate_pipeline(bound, build_diagnostic_registry())

    if plan.signature != validate_pipeline(fixed, build_diagnostic_registry()).signature:
        raise ValueError("고정 입력 진단 계획의 서명이 일치하지 않습니다.")

    return bound, plan

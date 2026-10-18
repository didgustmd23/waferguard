"""표·숫자 배열의 독립 canonical 변환 코어. pickle이나 에이전트를 호출하지 않는다."""

from datetime import date, datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MAX_ROWS = 2_000_000
MAX_CELLS = 100_000_000
MAX_JSON_BYTES = 1024 * 1024
TABLE_DTYPES = frozenset({
    "bool", "boolean", "object", "string", "datetime64[ns]",
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
    "float16", "float32", "float64", "Float32", "Float64",
})
NUMERIC_SCALARS = frozenset(np.dtype(name).type for name in (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
))


def decode_json(payload):
    """중복 키·비표준 숫자·과대 JSON을 거부한다."""
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("canonical JSON 크기 상한을 초과했습니다.")

    def unique(items):
        result = {}

        for key, value in items:
            if key in result:
                raise ValueError("canonical JSON에 중복 키가 있습니다.")

            result[key] = value

        return result

    def invalid(_):
        raise ValueError("canonical JSON에 비표준 숫자가 있습니다.")

    return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)


def encode_scalar(value):
    """일반 스칼라만 직렬화하며 결측·날짜를 명시적 태그로 보존한다."""
    if value is pd.NA:
        return {"$scalar": "pd_na"}

    if value is pd.NaT:
        return {"$scalar": "nat"}

    if type(value) in NUMERIC_SCALARS:
        value = value.item()

    if value is None or type(value) in (str, int, bool):
        return value

    if type(value) is float:
        if math.isnan(value):
            return {"$scalar": "nan"}

        if math.isinf(value):
            return {"$scalar": "positive_inf" if value > 0 else "negative_inf"}

        return value

    if type(value) in (pd.Timestamp, datetime, date):
        return {"$scalar": type(value).__name__, "value": value.isoformat()}

    raise ValueError("표 셀 또는 인덱스에 지원하지 않는 객체가 있습니다.")


def decode_scalar(value):
    """고정 태그만 복원한다. 임의 클래스·함수·import 경로는 처리하지 않는다."""
    if type(value) is float and not math.isfinite(value):
        raise ValueError("비유한 숫자는 명시적 결측·무한대 태그가 필요합니다.")

    if value is None or type(value) in (str, int, bool, float):
        return value

    if type(value) is not dict or "$scalar" not in value:
        raise ValueError("canonical 스칼라 형식이 올바르지 않습니다.")

    tag = value["$scalar"]
    simple = {"pd_na": pd.NA, "nat": pd.NaT, "nan": float("nan"),
              "positive_inf": float("inf"), "negative_inf": -float("inf")}

    if type(tag) is str and tag in simple and set(value) == {"$scalar"}:
        return simple[tag]

    if set(value) != {"$scalar", "value"} or type(value["value"]) is not str or len(value["value"]) > 128:
        raise ValueError("canonical 날짜 스칼라 형식이 올바르지 않습니다.")

    loaders = {"Timestamp": pd.Timestamp, "datetime": datetime.fromisoformat, "date": date.fromisoformat}

    if type(tag) is not str or tag not in loaders:
        raise ValueError("지원하지 않는 canonical 스칼라 태그입니다.")

    return loaders[tag](value["value"])


def table_schema(frame):
    """컬럼·자료형·원본 인덱스를 명시한다. label·group 등의 의미를 추측하지 않는다."""
    if type(frame) is not pd.DataFrame or frame.empty or len(frame) > MAX_ROWS:
        raise ValueError("표의 구조 또는 행 수가 지원 범위를 벗어났습니다.")

    if len(frame.columns) > 4096 or frame.size > MAX_CELLS or not frame.columns.is_unique:
        raise ValueError("표의 컬럼·셀 개수 또는 중복 컬럼이 지원 범위를 벗어났습니다.")

    if any(type(name) is not str or not name or len(name) > 256 for name in frame.columns):
        raise ValueError("표 컬럼에는 길이 256 이하의 문자열 이름이 필요합니다.")

    dtypes = [str(dtype) for dtype in frame.dtypes]

    if any(dtype not in TABLE_DTYPES for dtype in dtypes) or type(frame.index) is pd.MultiIndex:
        raise ValueError("표 자료형 또는 다중 인덱스는 별도 변환 규칙이 필요합니다.")

    index_dtype = str(frame.index.dtype)

    if index_dtype not in TABLE_DTYPES or frame.index.name is not None and type(frame.index.name) is not str:
        raise ValueError("표 인덱스 자료형 또는 이름이 지원 범위를 벗어났습니다.")

    index = {"kind": "index", "dtype": index_dtype, "name": frame.index.name}

    if type(frame.index) is pd.RangeIndex:
        index.update(kind="range", start=frame.index.start, stop=frame.index.stop, step=frame.index.step)

    return {"version": "canonical_table_v1", "kind": "tabular", "rows": len(frame),
            "columns": list(frame.columns), "dtypes": dtypes, "index": index}


def supported_table(frame):
    """object 컬럼의 일반 문자열은 허용하고 중첩 배열·사용자 객체는 구분한다."""
    try:
        table_schema(frame)

        for row in frame.itertuples(index=True, name=None):
            for value in row:
                encode_scalar(value)

        return True
    except (ValueError, TypeError, OverflowError):
        return False


def supported_array(array):
    return (type(array) is np.ndarray and array.dtype.kind in "biufc" and not array.dtype.hasobject
            and array.ndim <= 8 and array.size <= MAX_CELLS)


def export_table(frame, output, budget):
    """바이너리 UTF-8 JSONL로 셀 타입·결측 태그·원본 인덱스를 보존한다."""
    schema = table_schema(frame)
    budget.write_json(output / "schema.json", schema)

    with budget.open(output / "data.jsonl") as stream:
        for row in frame.itertuples(index=True, name=None):
            record = {"index": encode_scalar(row[0]), "values": [encode_scalar(value) for value in row[1:]]}
            payload = (json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")

            if len(payload) > MAX_JSON_BYTES:
                raise ValueError("표 한 행의 canonical 크기 상한을 초과했습니다.")

            stream.write(payload)

    return ("schema.json", "data.jsonl")


def export_array(array, output, budget):
    """숫자 dtype과 shape를 유지하며 object pickle 없이 NPY로 기록한다."""
    if not supported_array(array):
        raise ValueError("지원하지 않는 배열 dtype·차원·원소 개수입니다.")

    budget.write_json(output / "schema.json", {
        "version": "canonical_array_v1", "kind": "numeric_array",
        "dtype": array.dtype.str, "shape": list(array.shape),
    })

    with budget.open(output / "array.npy") as stream:
        np.save(stream, array, allow_pickle=False)

    return ("schema.json", "array.npy")


def read_schema(directory):
    with (Path(directory) / "schema.json").open("rb") as stream:
        schema = decode_json(stream.read(MAX_JSON_BYTES + 1))

    if type(schema) is not dict:
        raise ValueError("canonical 스키마는 JSON 객체여야 합니다.")

    return schema


def _check_dtype(value, dtype):
    """스키마와 실제 셀 타입·정수 범위를 대조하고 묵시적 타입 변환을 허용하지 않는다."""
    if dtype == "object":
        return

    nullable = dtype in {"boolean", "string", "Int8", "Int16", "Int32", "Int64",
                         "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"}

    if value is pd.NA and nullable or value is pd.NaT and dtype == "datetime64[ns]":
        return

    if dtype in {"bool", "boolean"}:
        valid = type(value) is bool
    elif dtype == "string":
        valid = type(value) is str
    elif dtype == "datetime64[ns]":
        valid = type(value) is pd.Timestamp and value.tzinfo is None
    elif dtype.lower().startswith(("int", "uint")):
        bounds = np.iinfo(dtype.lower())
        valid = type(value) is int and bounds.min <= value <= bounds.max
    else:
        valid = type(value) is float

    if not valid:
        raise ValueError("canonical 셀 자료형이 저장 스키마와 다릅니다.")


def iter_table(directory, schema):
    """행별로 제한된 JSON만 읽어 자료형·컬럼 수·인덱스·행 수를 확인한다."""
    if set(schema) != {"version", "kind", "rows", "columns", "dtypes", "index"}:
        raise ValueError("표 스키마 필드가 올바르지 않습니다.")

    columns, dtypes, index = schema["columns"], schema["dtypes"], schema["index"]

    if (schema["version"] != "canonical_table_v1" or schema["kind"] != "tabular"
        or type(schema["rows"]) is not int or not 0 < schema["rows"] <= MAX_ROWS
        or type(columns) is not list or not 0 < len(columns) <= 4096
        or any(type(name) is not str or not name or len(name) > 256 for name in columns)
        or len(set(columns)) != len(columns) or type(dtypes) is not list or len(dtypes) != len(columns)
        or any(type(dtype) is not str or dtype not in TABLE_DTYPES for dtype in dtypes)
        or schema["rows"] * len(columns) > MAX_CELLS or type(index) is not dict):
        raise ValueError("표 스키마의 컬럼·자료형·행 수가 올바르지 않습니다.")

    expected_index = {"kind", "dtype", "name"}

    if index.get("kind") == "range":
        expected_index.update(("start", "stop", "step"))

    if (set(index) != expected_index or index["kind"] not in {"range", "index"}
        or type(index["dtype"]) is not str or index["dtype"] not in TABLE_DTYPES
        or index["name"] is not None and type(index["name"]) is not str):
        raise ValueError("표 인덱스 스키마가 올바르지 않습니다.")

    if index["kind"] == "range":
        if any(type(index[key]) is not int for key in ("start", "stop", "step")) or index["step"] == 0:
            raise ValueError("표 RangeIndex 설정이 올바르지 않습니다.")

        if len(range(index["start"], index["stop"], index["step"])) != schema["rows"]:
            raise ValueError("표 RangeIndex 길이가 행 수와 다릅니다.")

    count = 0

    with (Path(directory) / "data.jsonl").open("rb") as stream:
        while True:
            payload = stream.readline(MAX_JSON_BYTES + 1)

            if not payload:
                break

            count += 1
            row = decode_json(payload)

            if (count > schema["rows"] or type(row) is not dict or set(row) != {"index", "values"}
                or type(row["values"]) is not list or len(row["values"]) != len(columns)):
                raise ValueError("canonical 표의 행 구조 또는 개수가 올바르지 않습니다.")

            source_index = decode_scalar(row["index"])
            _check_dtype(source_index, index["dtype"])

            if index["kind"] == "range" and source_index != index["start"] + (count - 1) * index["step"]:
                raise ValueError("canonical 표의 원본 인덱스가 스키마와 다릅니다.")

            values = [decode_scalar(value) for value in row["values"]]

            for value, dtype in zip(values, dtypes):
                _check_dtype(value, dtype)

            yield source_index, values

    if count != schema["rows"]:
        raise ValueError("canonical 표의 행 수가 스키마와 다릅니다.")


def validate_data(directory):
    """pickle 복원 없이 표 또는 숫자 배열 계약을 검사한다."""
    root = Path(directory)
    schema = read_schema(root)

    if schema.get("kind") == "tabular":
        for _ in iter_table(root, schema):
            pass

        return {"kind": "tabular", "rows": schema["rows"], "columns": len(schema["columns"])}

    if set(schema) != {"version", "kind", "dtype", "shape"} or schema.get("version") != "canonical_array_v1":
        raise ValueError("지원하지 않는 canonical 데이터 종류입니다.")

    shape = schema["shape"]

    if (schema["kind"] != "numeric_array" or type(shape) is not list or len(shape) > 8
        or any(type(length) is not int or length < 0 for length in shape)
        or math.prod(shape) > MAX_CELLS or type(schema["dtype"]) is not str or len(schema["dtype"]) > 32):
        raise ValueError("숫자 배열 스키마가 올바르지 않습니다.")

    dtype = np.dtype(schema["dtype"])

    if dtype.kind not in "biufc" or dtype.hasobject:
        raise ValueError("숫자 배열에 object 자료형을 사용할 수 없습니다.")

    with (root / "array.npy").open("rb") as stream:
        version = np.lib.format.read_magic(stream)

        if version not in ((1, 0), (2, 0)):
            raise ValueError("지원하지 않는 NPY 헤더 버전입니다.")

        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        header_shape, _, header_dtype = reader(stream, max_header_size=10000)

        if (list(header_shape) != shape or header_dtype != dtype
            or (root / "array.npy").stat().st_size != stream.tell() + math.prod(shape) * dtype.itemsize):
            raise ValueError("NPY 헤더·파일 크기가 스키마와 다릅니다.")

    return {"kind": "numeric_array", "shape": shape, "dtype": dtype.str}


def load_table(directory):
    """후속 EDA·전처리에 사용할 DataFrame을 명시적 canonical 입력에서 복원한다."""
    schema = read_schema(directory)
    rows = list(iter_table(directory, schema))
    data = {name: pd.Series([values[column] for _, values in rows], dtype=schema["dtypes"][column])
            for column, name in enumerate(schema["columns"])}
    frame = pd.DataFrame(data)
    index = schema["index"]

    if index["kind"] == "range":
        frame.index = pd.RangeIndex(index["start"], index["stop"], index["step"], name=index["name"])
    else:
        frame.index = pd.Index([value for value, _ in rows], dtype=index["dtype"], name=index["name"])

    return frame

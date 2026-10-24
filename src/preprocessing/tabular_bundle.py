"""표의 자료형과 원래 행 위치를 보존하고 v2 파일 묶음을 엄격하게 검증한다."""

from dataclasses import asdict, dataclass
from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

import numpy as np
import pandas as pd

from .canonical_data import TABLE_DTYPES, encode_scalar, decode_scalar


VERSION = "canonical_table_v2"
MAX_ROWS, MAX_COLUMNS, MAX_CELLS = 2_000_000, 4096, 100_000_000
MAX_JSON_BYTES = 1024 * 1024
DATA_FILES = ("schema.json", "data.jsonl", "provenance.json")
FILES = frozenset((*DATA_FILES, "manifest.json", "SUCCESS.json"))
_WORKER_ROOT = ContextVar("tabular_worker_root", default=None)


@contextmanager
def worker_filesystem(root):
    """부모가 검증한 작업 경계 아래만 조회한다. 경계 자체와 조상에는 worker 권한이 없다."""
    root = Path(os.path.abspath(root))
    token = _WORKER_ROOT.set(root)
    try:
        yield
    finally:
        _WORKER_ROOT.reset(token)


def _require(condition):
    if not condition:
        raise ValueError("표형 묶음의 형식·자료형 또는 정합성이 올바르지 않습니다.")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _require(len(payload) <= MAX_JSON_BYTES)
    return payload


def _decode(payload):
    _require(len(payload) <= MAX_JSON_BYTES)

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result)
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("비표준 JSON 숫자는 허용하지 않습니다.")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    _require(type(value) is dict)
    return value


def _safe(path, *, directory):
    path = Path(os.path.abspath(path))
    boundary = _WORKER_ROOT.get()
    _require(boundary is None or path.is_relative_to(boundary))
    for component in (path, *path.parents):
        if component == boundary:
            break
        info = component.lstat()
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400)
    _require(path.is_dir() if directory else path.is_file() and path.stat().st_nlink == 1)
    return path


def _read(root, name):
    with _safe(root / name, directory=False).open("rb") as stream:
        return _decode(stream.read(MAX_JSON_BYTES + 1))


def _fingerprint(path, maximum):
    path = _safe(path, directory=False)
    before = path.stat()
    size, digest = 0, hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(min(1024 * 1024, maximum + 1)):
            size += len(chunk)
            _require(size <= maximum)
            digest.update(chunk)
    after = path.stat()
    _require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, size, after.st_mtime_ns))
    return {"sha256": digest.hexdigest(), "size_bytes": size}


@dataclass(frozen=True)
class BundleRef:
    """부모가 외부에 고정하는 파일 지문이며 실행 승인이나 원본 동등성 증명은 아니다."""

    manifest_sha256: str
    completion_sha256: str
    size_bytes: int
    rows: int
    columns: int
    schema_version: str = VERSION

    def __post_init__(self):
        _require(self.schema_version == VERSION)
        for value in (self.manifest_sha256, self.completion_sha256):
            _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
        for value, limit in ((self.rows, MAX_ROWS), (self.columns, MAX_COLUMNS), (self.size_bytes, 2**63 - 1)):
            _require(type(value) is int and 0 < value <= limit)
        _require(self.rows * self.columns <= MAX_CELLS)

    @classmethod
    def from_dict(cls, value):
        """알 수 없는 필드·버전과 bool을 정수로 가장한 값을 거부한다."""
        _require(type(value) is dict and set(value) == set(cls.__dataclass_fields__))
        return cls(**value)


def _encode(value):
    result = encode_scalar(value)
    if type(value) is pd.Timestamp and value.tz is not None:
        result["timezone"] = str(value.tz)
    return result


def _scalar(value):
    if type(value) is dict and "timezone" in value:
        _require(set(value) == {"$scalar", "value", "timezone"} and value["$scalar"] == "Timestamp")
        _require(type(value["timezone"]) is str and len(value["timezone"]) <= 128)
        timestamp = decode_scalar({key: item for key, item in value.items() if key != "timezone"})
        _require(timestamp.tz is not None)
        restored = timestamp.tz_convert(value["timezone"])
        _require(restored.isoformat() == value["value"])
        return restored
    return decode_scalar(value)


def _describe(dtype):
    if isinstance(dtype, pd.CategoricalDtype):
        return {"kind": "category", "ordered": dtype.ordered,
                "categories": [_encode(value) for value in dtype.categories],
                "categories_dtype": _describe(dtype.categories.dtype)}
    if isinstance(dtype, pd.StringDtype):
        _require(dtype.storage == "python")
        return {"kind": "plain", "dtype": "string"}
    if isinstance(dtype, pd.DatetimeTZDtype):
        return {"kind": "timestamp", "unit": dtype.unit, "timezone": str(dtype.tz)}
    name = str(dtype)
    if re.fullmatch(r"datetime64\[(s|ms|us|ns)\]", name):
        return {"kind": "timestamp", "unit": name[11:-1], "timezone": None}
    _require(name in TABLE_DTYPES and name != "datetime64[ns]")
    return {"kind": "plain", "dtype": name}


def _dtype(spec):
    _require(type(spec) is dict)
    kind = spec.get("kind")
    if kind == "plain":
        _require(set(spec) == {"kind", "dtype"} and type(spec["dtype"]) is str and spec["dtype"] in TABLE_DTYPES and spec["dtype"] != "datetime64[ns]")
        return pd.StringDtype(storage="python") if spec["dtype"] == "string" else spec["dtype"]
    if kind == "timestamp":
        _require(set(spec) == {"kind", "unit", "timezone"} and spec["unit"] in ("s", "ms", "us", "ns"))
        zone = spec["timezone"]
        _require(zone is None or type(zone) is str and 0 < len(zone) <= 128)
        return pd.DatetimeTZDtype(unit=spec["unit"], tz=zone) if zone is not None else f"datetime64[{spec['unit']}]"
    _require(kind == "category" and set(spec) == {"kind", "categories", "categories_dtype", "ordered"})
    _require(type(spec["ordered"]) is bool and type(spec["categories"]) is list)
    _require(type(spec["categories_dtype"]) is dict and spec["categories_dtype"].get("kind") != "category")
    values = [_scalar(value) for value in spec["categories"]]
    categories = pd.Index(_series(values, spec["categories_dtype"]))
    _require(categories.is_unique and not categories.hasnans)
    return pd.CategoricalDtype(categories=categories, ordered=spec["ordered"])


def _valid_scalar(value, spec, category_keys=None):
    kind = spec["kind"]
    if kind == "category":
        if value is pd.NA or value is None or type(value) is float and math.isnan(value):
            return
        _valid_scalar(value, spec["categories_dtype"])
        tokens = category_keys if category_keys is not None else {_json({"v": item}) for item in spec["categories"]}
        _require(_json({"v": _encode(value)}) in tokens)
        return
    if kind == "timestamp":
        if value is pd.NaT:
            return
        _require(type(value) is pd.Timestamp)
        _require((value.tz is None) == (spec["timezone"] is None))
        _require(value.tz is None or str(value.tz) == spec["timezone"])
        converted = pd.Timestamp(value).as_unit(spec["unit"], round_ok=False)
        _require(converted == value)
        return
    name = spec["dtype"]
    if name == "object":
        _encode(value)
        return
    nullable = name in {"string", "boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"}
    if value is pd.NA and nullable:
        return
    if name == "string":
        _require(type(value) is str)
    elif name in ("bool", "boolean"):
        _require(type(value) is bool)
    elif name.lower().startswith(("int", "uint")):
        bounds = np.iinfo(name.lower())
        _require(type(value) is int and bounds.min <= value <= bounds.max)
    else:
        _require(type(value) is float)
        if math.isfinite(value):
            converted = np.dtype(name.lower()).type(value).item()
            _require(converted == value)


def _series(values, spec):
    dtype = _dtype(spec)
    tokens = {_json({"v": item}) for item in spec["categories"]} if spec["kind"] == "category" else None
    for value in values:
        _valid_scalar(value, spec, tokens)
    if spec.get("dtype") in ("Float32", "Float64"):
        mask = np.array([value is pd.NA for value in values], dtype=bool)
        data = np.array([np.nan if missing else value for value, missing in zip(values, mask)], dtype=spec["dtype"].lower())
        return pd.Series(pd.arrays.FloatingArray(data, mask))
    return pd.Series(values, dtype=dtype)


def _schema(frame):
    _require(type(frame) is pd.DataFrame and 0 < len(frame) <= MAX_ROWS and 0 < len(frame.columns) <= MAX_COLUMNS)
    _require(frame.size <= MAX_CELLS and frame.columns.is_unique and not isinstance(frame.index, pd.MultiIndex))
    _require(all(type(name) is str and 0 < len(name) <= 256 for name in frame.columns))
    _require(frame.columns.name is None or type(frame.columns.name) is str and len(frame.columns.name) <= 256)
    _require(frame.index.name is None or type(frame.index.name) is str and len(frame.index.name) <= 256)
    index = {"kind": "index", "dtype": _describe(frame.index.dtype), "name": frame.index.name}
    if type(frame.index) is pd.RangeIndex:
        index.update(kind="range", start=frame.index.start, stop=frame.index.stop, step=frame.index.step)
    elif type(frame.index) is pd.DatetimeIndex:
        index["frequency"] = frame.index.freqstr
    return {"version": VERSION, "scalar_encoding": "tagged_scalar_v2", "rows": len(frame),
            "columns_name": frame.columns.name,
            "columns": [{"id": f"c{index}", "name": name, "dtype": _describe(frame[name].dtype)}
                        for index, name in enumerate(frame.columns)], "index": index}


def _validate_schema(schema):
    _require(set(schema) == {"version", "scalar_encoding", "rows", "columns", "columns_name", "index"})
    _require(schema["columns_name"] is None or type(schema["columns_name"]) is str and len(schema["columns_name"]) <= 256)
    _require(schema["version"] == VERSION and schema["scalar_encoding"] == "tagged_scalar_v2")
    count, columns = schema["rows"], schema["columns"]
    _require(type(count) is int and 0 < count <= MAX_ROWS and type(columns) is list and 0 < len(columns) <= MAX_COLUMNS)
    _require(count * len(columns) <= MAX_CELLS)
    names = set()
    for index, column in enumerate(columns):
        _require(type(column) is dict and set(column) == {"id", "name", "dtype"})
        _require(column["id"] == f"c{index}" and type(column["name"]) is str and 0 < len(column["name"]) <= 256)
        _require(column["name"] not in names)
        names.add(column["name"])
        _dtype(column["dtype"])
    index = schema["index"]
    _require(type(index) is dict and index.get("kind") in ("range", "index"))
    keys = {"kind", "dtype", "name"} | ({"start", "stop", "step"} if index["kind"] == "range" else set())
    if "frequency" in index:
        _require(index["kind"] == "index" and index["dtype"].get("kind") == "timestamp")
        _require(index["frequency"] is None or type(index["frequency"]) is str and 0 < len(index["frequency"]) <= 64)
        keys.add("frequency")
    _require(set(index) == keys and (index["name"] is None or type(index["name"]) is str and len(index["name"]) <= 256))
    _dtype(index["dtype"])
    if index["kind"] == "range":
        _require(all(type(index[key]) is int for key in ("start", "stop", "step")) and index["step"] != 0)
        _require(len(range(index["start"], index["stop"], index["step"])) == count)


def _rows(root, schema, *, verified_stream=None):
    count, previous = 0, -1
    tokens = [{_json({"v": item}) for item in column["dtype"]["categories"]}
              if column["dtype"]["kind"] == "category" else None for column in schema["columns"]]
    context = (_safe(root / "data.jsonl", directory=False).open("rb")
               if verified_stream is None else nullcontext(verified_stream))
    with context as stream:
        while payload := stream.readline(MAX_JSON_BYTES + 1):
            row = _decode(payload)
            _require(set(row) == {"position", "index", "values"})
            position = row["position"]
            _require(type(position) is int and previous < position < MAX_ROWS and count < schema["rows"])
            _require(type(row["values"]) is list and len(row["values"]) == len(schema["columns"]))
            index = _scalar(row["index"])
            _valid_scalar(index, schema["index"]["dtype"])
            if schema["index"]["kind"] == "range":
                _require(index == schema["index"]["start"] + count * schema["index"]["step"])
            values = [_scalar(item) for item in row["values"]]
            for value, column, choices in zip(values, schema["columns"], tokens):
                _valid_scalar(value, column["dtype"], choices)
            count, previous = count + 1, position
            yield position, index, values
    _require(count == schema["rows"])


def _files(root, maximum):
    _require(type(maximum) is int and 0 < maximum <= 2**63 - 1)
    _require({path.name for path in root.iterdir()} == FILES)
    records, total = {}, 0
    for name in sorted(FILES):
        records[name] = _fingerprint(root / name, maximum - total)
        total += records[name]["size_bytes"]
        _require(total <= maximum)
    return records, total


def inspect_bundle(directory, *, expected, max_bytes):
    """묶음 전체를 스트리밍 검증한다. 생산 중 외에는 외부 기대 참조가 필요하다."""
    _require(expected is None or type(expected) is BundleRef)
    root = _safe(directory, directory=True)
    records, total = _files(root, max_bytes)
    manifest, marker = _read(root, "manifest.json"), _read(root, "SUCCESS.json")
    _require(manifest == {"version": "tabular_bundle_manifest_v2", "files": {name: records[name] for name in DATA_FILES}})
    _require(marker == {"version": "tabular_bundle_complete_v2", "manifest_sha256": records["manifest.json"]["sha256"]})
    schema = _read(root, "schema.json")
    _validate_schema(schema)
    reference = BundleRef(records["manifest.json"]["sha256"], records["SUCCESS.json"]["sha256"], total,
                          schema["rows"], len(schema["columns"]))
    _require(expected is None or reference == expected)
    _read(root, "provenance.json")
    for _ in _rows(root, schema):
        pass
    _require(_files(root, max_bytes) == (records, total))
    return reference


def read_bundle_schema(directory, *, expected):
    """외부 기대 지문에 묶인 스키마만 반환한다."""
    _require(type(expected) is BundleRef)
    inspect_bundle(directory, expected=expected, max_bytes=expected.size_bytes)
    schema = _read_bound_metadata(_safe(directory, directory=True), "schema.json", expected)
    _validate_schema(schema)
    return schema


def read_bundle_provenance(directory, *, expected):
    """승인된 출처와 대조할 검증된 로컬 출처 정보만 반환한다."""
    _require(type(expected) is BundleRef)
    inspect_bundle(directory, expected=expected, max_bytes=expected.size_bytes)
    return _read_bound_metadata(_safe(directory, directory=True), "provenance.json", expected)


def _read_bound_metadata(root, name, expected):
    manifest = _read_bound_manifest(root, expected)
    with _safe(root / name, directory=False).open("rb") as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    _require(manifest["files"][name] == {"sha256": _sha(payload), "size_bytes": len(payload)})
    return _decode(payload)


def _read_bound_manifest(root, expected):
    with _safe(root / "manifest.json", directory=False).open("rb") as stream:
        manifest_bytes = stream.read(MAX_JSON_BYTES + 1)
    _require(_sha(manifest_bytes) == expected.manifest_sha256)
    return _decode(manifest_bytes)


def iter_bundle_rows(directory, *, expected):
    """실제 소비 바이트를 먼저 검증한 비공개 임시 스트림에서만 행을 반환한다."""
    schema = read_bundle_schema(directory, expected=expected)
    root = _safe(directory, directory=True)
    manifest = _read_bound_manifest(root, expected)
    record = manifest["files"]["data.jsonl"]
    # 전후 경로 해시만으로는 ABA 교체를 잡을 수 없다. 실제 읽은 바이트를 고정한 뒤 첫 행을 공개한다.
    with tempfile.TemporaryFile(mode="w+b") as verified:
        size, digest = 0, hashlib.sha256()
        with _safe(root / "data.jsonl", directory=False).open("rb") as source:
            while payload := source.read(min(1024 * 1024, record["size_bytes"] - size + 1)):
                size += len(payload)
                _require(size <= record["size_bytes"])
                digest.update(payload)
                verified.write(payload)
        _require(size == record["size_bytes"] and digest.hexdigest() == record["sha256"])
        verified.seek(0)
        yield from _rows(root, schema, verified_stream=verified)
    inspect_bundle(directory, expected=expected, max_bytes=expected.size_bytes)


def _write(directory, schema, rows, provenance, maximum):
    _require(type(provenance) is dict and type(maximum) is int and 0 < maximum <= 2**63 - 1)
    _validate_schema(schema)
    # 폴더를 만들기 전에 메타데이터 크기·객체를 검사한다.
    schema_bytes, provenance_bytes = _json(schema), _json(provenance)
    parent = _safe(Path(directory).absolute().parent, directory=True)
    root = parent / Path(directory).name
    root.mkdir(exist_ok=False)
    used = 0

    def save(name, parts):
        nonlocal used
        with (root / name).open("xb") as stream:
            for payload in parts:
                _require(used + len(payload) <= maximum)
                stream.write(payload)
                used += len(payload)
            stream.flush()
            os.fsync(stream.fileno())

    save("schema.json", (schema_bytes,))
    save("provenance.json", (provenance_bytes,))
    save("data.jsonl", (_json({"position": position, "index": _encode(index), "values": [_encode(value) for value in values]}) + b"\n"
                       for position, index, values in rows))
    # 잘못된 행에는 완료 표식을 만들지 않는다.
    for _ in _rows(root, schema):
        pass
    files = {name: _fingerprint(root / name, maximum) for name in DATA_FILES}
    manifest = _json({"version": "tabular_bundle_manifest_v2", "files": files})
    save("manifest.json", (manifest,))
    completion = _json({"version": "tabular_bundle_complete_v2", "manifest_sha256": _sha(manifest)})
    _require(used + len(completion) <= maximum)
    save(".complete.tmp", (completion,))
    os.replace(root / ".complete.tmp", root / "SUCCESS.json")
    return inspect_bundle(root, expected=None, max_bytes=maximum)


def write_bundle(frame, directory, *, provenance, max_output_bytes, row_positions=None):
    """원본 표를 수정하지 않고 새 폴더에만 v2 묶음을 만든다."""
    schema = _schema(frame)
    positions = range(len(frame)) if row_positions is None else row_positions
    _require(hasattr(positions, "__len__") and len(positions) == len(frame))
    _require(all(type(value) is int and 0 <= value < MAX_ROWS for value in positions))
    _require(all(left < right for left, right in zip(positions, positions[1:])))
    rows = ((position, record[0], record[1:]) for position, record in zip(positions, frame.itertuples(index=True, name=None)))
    return _write(directory, schema, rows, provenance, max_output_bytes)


def load_bundle(directory, *, expected, columns=None):
    """행 객체 전체 목록 대신 컬럼별 작은 청크로 자료형을 검증하며 표를 복원한다."""
    schema = read_bundle_schema(directory, expected=expected)
    names = [column["name"] for column in schema["columns"]]
    selected = names if columns is None else list(columns)
    _require(selected and all(type(name) is str and name in names for name in selected) and len(set(selected)) == len(selected))
    indexes, index_parts = [], []
    buffers, parts = {name: [] for name in selected}, {name: [] for name in selected}
    offsets = {column["name"]: (index, column["dtype"]) for index, column in enumerate(schema["columns"])}

    def flush():
        if not indexes:
            return
        index_parts.append(_series(indexes, schema["index"]["dtype"]))
        indexes.clear()
        for name in selected:
            parts[name].append(_series(buffers[name], offsets[name][1]))
            buffers[name].clear()

    for _, index, values in iter_bundle_rows(directory, expected=expected):
        indexes.append(index)
        for name in selected:
            buffers[name].append(values[offsets[name][0]])
        if len(indexes) >= 1024:
            flush()
    flush()
    frame = pd.DataFrame({name: pd.concat(parts[name], ignore_index=True) for name in selected})
    description = schema["index"]
    if description["kind"] == "range":
        frame.index = pd.RangeIndex(description["start"], description["stop"], description["step"], name=description["name"])
    else:
        labels = pd.concat(index_parts, ignore_index=True)
        frame.index = (pd.DatetimeIndex(labels, name=description["name"], freq=description["frequency"])
                       if "frequency" in description else pd.Index(labels, name=description["name"]))
    frame.columns.name = schema["columns_name"]
    return frame


def write_subset(directory, positions, destination, *, expected, provenance, max_output_bytes):
    """원래 행 위치로 선택한 구간을 순서·자료형·인덱스와 함께 새 묶음에 기록한다."""
    _require(type(positions) in (list, tuple) and 0 < len(positions) <= MAX_ROWS)
    _require(all(type(value) is int and 0 <= value < MAX_ROWS for value in positions))
    _require(all(left < right for left, right in zip(positions, positions[1:])))
    schema = read_bundle_schema(directory, expected=expected)
    schema["rows"] = len(positions)
    schema["index"] = {key: value for key, value in schema["index"].items() if key in {"kind", "dtype", "name"}}
    schema["index"]["kind"] = "index"
    selected = set(positions)

    def rows():
        count = 0
        for position, index, values in iter_bundle_rows(directory, expected=expected):
            if position in selected:
                count += 1
                yield position, index, values
        _require(count == len(positions))

    return _write(destination, schema, rows(), provenance, max_output_bytes)

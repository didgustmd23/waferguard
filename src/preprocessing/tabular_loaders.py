"""고정 CSV 해석을 전수 검사하고 동일 snapshot만 v2 표형 묶음으로 생성한다."""

from contextlib import contextmanager, ExitStack
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from io import StringIO, TextIOWrapper
import json
import math
from pathlib import Path
import re
import tempfile
import time

import pandas as pd

from .canonical_data import decode_json
from .input_profile import dedupe_names
from .tabular_bundle import MAX_ROWS, MAX_COLUMNS, MAX_CELLS, _safe, write_bundle


INFERENCE_POLICY = "csv_lossless_inference_v1"
POLICY_VERSION = "tabular_input_v2"
MAX_RECORD_BYTES = 1024 * 1024
MAX_INSPECTION_BYTES = 16 * 1024 * 1024
_INTEGER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z")
_FLOAT = re.compile(r"[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)\Z")


class CsvLoadError(ValueError):
    """원본 값·경로·파서 예외 대신 고정 코드만 전달한다."""

    def __init__(self, code):
        super().__init__("CSV 입력의 해석·자원 또는 검사 결과가 일치하지 않습니다: " + code)
        self.code = code


def _require(condition, code):
    if not condition:
        raise CsvLoadError(code)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CsvReadSpec:
    """허용된 해석만 고정하며 파서 함수나 pandas 자유 인자를 받지 않는다."""

    loader: str = "delimited_v1"
    encoding: str = "utf-8-sig"
    delimiter: str = ","
    header: str = "first_row"
    empty_field: str = "missing"
    header_repair: str = "reject"

    def __post_init__(self):
        _require(all(type(value) is str for value in asdict(self).values()), "CSV_SPEC_INVALID")
        _require((self.loader, self.encoding, self.delimiter, self.header)
                 == ("delimited_v1", "utf-8-sig", ",", "first_row"), "CSV_SPEC_UNSUPPORTED")
        _require(self.empty_field in ("missing", "preserve")
                 and self.header_repair in ("reject", "number_duplicates"), "CSV_SPEC_INVALID")

    def to_dict(self):
        """검사·승인·생성에서 공유할 명시적 해석 값을 반환한다."""
        return asdict(self)

    def to_json(self):
        """해석을 결정적 JSON으로 직렬화한다."""
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        """추가 옵션과 누락된 설정을 거부한다."""
        _require(type(value) is dict and set(value) == set(cls.__dataclass_fields__), "CSV_SPEC_INVALID")
        return cls(**value)

    @classmethod
    def from_json(cls, payload):
        """중복 키·비표준 숫자·과대 설정을 거부한다."""
        _require(type(payload) is str and len(payload.encode("utf-8")) <= 65536, "CSV_SPEC_INVALID")
        return cls.from_dict(decode_json(payload.encode("utf-8")))


def _limits(value):
    value = asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
    bounds = {"wall_seconds": 86400, "memory_mib": 32768, "processes": 8}
    _require(type(value) is dict and set(value) == set(bounds), "CSV_LIMITS_INVALID")
    _require(all(type(value[key]) is int and 1 <= value[key] <= upper for key, upper in bounds.items()), "CSV_LIMITS_INVALID")
    return dict(value)


def _check_time(deadline):
    _require(time.monotonic() <= deadline, "CSV_TIME_LIMIT")


@contextmanager
def _snapshot_stream(source, maximum, deadline):
    _require(type(maximum) is int and 0 < maximum <= 2**63 - 1, "CSV_LIMITS_INVALID")
    with ExitStack() as resources:
        try:
            stream = resources.enter_context(tempfile.TemporaryFile(mode="w+b"))
            snapshot = _copy_snapshot(source, maximum, deadline, stream)
        except PermissionError:
            raise CsvLoadError("CSV_INPUT_ACCESS_DENIED") from None
        except OSError:
            raise CsvLoadError("CSV_INPUT_UNREADABLE") from None
        # 소비자/출력 writer의 예외까지 입력 오류로 오분류하지 않는다.
        yield stream, snapshot


def _copy_snapshot(source, maximum, deadline, stream):
    path = Path(source)
    _require(path.is_absolute() and not path.drive.startswith("\\"), "CSV_INPUT_UNSAFE")
    try:
        path = _safe(path, directory=False)
    except ValueError:
        raise CsvLoadError("CSV_INPUT_UNSAFE") from None
    before = path.stat()
    _require(before.st_size <= maximum, "CSV_INPUT_TOO_LARGE")
    _require(before.st_size > 0, "CSV_EMPTY_INPUT")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as original:
        while chunk := original.read(min(1024 * 1024, maximum - size + 1)):
            _check_time(deadline)
            size += len(chunk)
            _require(size <= maximum, "CSV_INPUT_TOO_LARGE")
            digest.update(chunk)
            stream.write(chunk)
    after = path.stat()
    _require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, size, after.st_mtime_ns), "CSV_SNAPSHOT_CHANGED")
    stream.seek(0)
    return {"sha256": digest.hexdigest(), "size_bytes": size, "max_input_bytes": maximum}


def _records(binary, deadline):
    """따옴표 경계와 레코드 상한을 검사하며 CSV 전역 필드 제한을 바꾸지 않는다."""
    text = TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    field, row, state, used, pending = StringIO(), [], "start", 0, False

    def finish_field():
        row.append(field.getvalue())
        _require(len(row) <= MAX_COLUMNS, "CSV_TOO_MANY_COLUMNS")
        field.seek(0)
        field.truncate(0)

    try:
        while line := text.readline(MAX_RECORD_BYTES - used + 1):
            _check_time(deadline)
            used += len(line.encode("utf-8"))
            _require(used <= MAX_RECORD_BYTES, "CSV_RECORD_TOO_LARGE")
            skip_lf = False
            for offset, char in enumerate(line):
                if offset % 4096 == 0:
                    _check_time(deadline)
                _require(char != "\x00", "CSV_MALFORMED")
                if skip_lf and char == "\n":
                    skip_lf = False
                    continue
                skip_lf, pending = False, True
                if state == "quoted":
                    if char == '"':
                        state = "after_quote"
                    else:
                        field.write(char)
                    continue
                if state == "after_quote" and char == '"':
                    field.write('"')
                    state = "quoted"
                    continue
                if char in (",", "\r", "\n"):
                    finish_field()
                    state = "start"
                    if char != ",":
                        yield row
                        row, used, pending = [], 0, False
                        skip_lf = char == "\r"
                    continue
                _require(state != "after_quote", "CSV_MALFORMED")
                if char == '"':
                    _require(state == "start", "CSV_MALFORMED")
                    state = "quoted"
                else:
                    field.write(char)
                    state = "bare"
        _require(state != "quoted", "CSV_MALFORMED")
        if pending:
            finish_field()
            yield row
    finally:
        text.detach()


def _number(token):
    if token in ("inf", "+inf", "-inf"):
        return "float", float(token), False
    if _INTEGER.fullmatch(token):
        digits = token.lstrip("+-")
        if len(digits) > 20:
            return "text", token, True
        value = int(token)
        return ("integer", value, False) if -(2**63) <= value <= 2**64 - 1 else ("text", token, True)
    if _FLOAT.fullmatch(token):
        integer_part = re.split(r"[.eE]", token.lstrip("+-"), maxsplit=1)[0]
        if len(integer_part) > 1 and integer_part.startswith("0"):
            return "text", token, False
        try:
            value = float(token)
            if math.isfinite(value) and Decimal(str(value)) == Decimal(token):
                return "float", value, False
        except (ValueError, InvalidOperation, OverflowError):
            pass
        return "text", token, True
    return "text", token, False


class _ColumnStats:
    def __init__(self, name):
        self.name, self.missing, self.success, self.failure = name, 0, 0, 0
        self.precision, self.float_seen, self.float_unsafe_integer = 0, False, 0
        self.minimum, self.maximum = 0, 0
        self.has_integer = False

    def add(self, token, spec):
        if token == "" and spec.empty_field == "missing":
            self.missing += 1
            return
        kind, value, precision = _number(token)
        self.precision += int(precision)
        if kind == "text":
            self.failure += 1
        else:
            self.success += 1
            self.float_seen |= kind == "float"
            if kind == "integer":
                self.minimum = min(self.minimum, value) if self.has_integer else value
                self.maximum = max(self.maximum, value) if self.has_integer else value
                self.has_integer = True
                self.float_unsafe_integer += int(int(float(value)) != value)

    def result(self, position):
        precision = self.precision + (self.float_unsafe_integer if self.float_seen else 0)
        if self.failure or precision:
            dtype = "string"
        elif self.success == 0:
            dtype = "object"
        elif self.float_seen:
            dtype = "Float64" if self.missing else "float64"
        elif self.minimum >= 0 and self.maximum > 2**63 - 1:
            dtype = "UInt64" if self.missing else "uint64"
        elif self.maximum <= 2**63 - 1:
            dtype = "Int64" if self.missing else "int64"
        else:
            dtype, precision = "string", self.success
        return {"id": f"c{position}", "name": self.name, "dtype": dtype, "missing_count": self.missing,
                "numeric_success_count": self.success, "numeric_failure_count": self.failure,
                "precision_risk_count": precision}


def _base(spec, limits, maximum):
    return {"schema_version": "csv_inspection_v1", "policy_version": POLICY_VERSION, "inference_policy": INFERENCE_POLICY,
            "loader_version": "1.0.0", "spec": spec.to_dict(), "limits": limits, "max_input_bytes": maximum,
            "snapshot": None, "status": "blocked_input", "code": "CSV_INPUT_UNREADABLE",
            "rows": 0, "columns": [], "header_mapping": [], "detail": {}, "candidates": []}


def _seal(report):
    payload = _json(report)
    _require(len(payload.encode("utf-8")) <= MAX_INSPECTION_BYTES, "CSV_INSPECTION_TOO_LARGE")
    return {**report, "inspection_id": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def _analyze(stream, spec, report, deadline):
    stream.seek(0)
    records = _records(stream, deadline)
    try:
        header = next(records, None)
        _require(header is not None, "CSV_EMPTY_INPUT")
        _require(all(name.strip() for name in header), "EMPTY_COLUMN_NAME")
        _require(all(len(name) <= 256 for name in header), "CSV_HEADER_TOO_LONG")
        renamed = header
        if len(set(header)) != len(header):
            if spec.header_repair == "reject":
                counts = Counter(header)
                report.update(status="review_required", code="DUPLICATE_COLUMN_NAMES",
                              detail={"duplicated_names": sum(count > 1 for count in counts.values()),
                                      "affected_columns": sum(count for count in counts.values() if count > 1)},
                              candidates=["number_duplicates"])
                return report
            try:
                renamed = dedupe_names(header)
            except ValueError:
                raise CsvLoadError("CSV_HEADER_REPAIR_CONFLICT") from None
        mapping = [{"position": index, "original": old, "renamed": new}
                   for index, (old, new) in enumerate(zip(header, renamed)) if old != new]
        _require(all(len(name) <= 256 for name in renamed), "CSV_HEADER_TOO_LONG")
        stats = [_ColumnStats(name) for name in renamed]
        count = 0
        for row in records:
            count += 1
            _require(count <= MAX_ROWS and count * len(header) <= MAX_CELLS, "CSV_TABLE_TOO_LARGE")
            _require(len(row) == len(header), "CSV_ROW_WIDTH_MISMATCH")
            for column, token in zip(stats, row):
                column.add(token, spec)
        _require(count > 0, "CSV_EMPTY_DATA")
        report.update(status="ready", code="READY", rows=count,
                      columns=[column.result(index) for index, column in enumerate(stats)], header_mapping=mapping)
        return report
    finally:
        records.close()


def inspect_csv(snapshot_file, spec, *, limits, max_input_bytes):
    """원본 행/실패 값 없이 전체 snapshot의 형식·변환 가능 건수·자료형을 검사한다."""
    _require(type(spec) is CsvReadSpec, "CSV_SPEC_INVALID")
    normalized = _limits(limits)
    _require(type(max_input_bytes) is int and 0 < max_input_bytes <= 2**63 - 1, "CSV_LIMITS_INVALID")
    report = _base(spec, normalized, max_input_bytes)
    deadline = time.monotonic() + normalized["wall_seconds"]
    try:
        with _snapshot_stream(snapshot_file, max_input_bytes, deadline) as (stream, snapshot):
            report["snapshot"] = snapshot
            _analyze(stream, spec, report, deadline)
    except UnicodeError:
        report.update(status="blocked_input", code="CSV_ENCODING_INVALID")
    except CsvLoadError as error:
        report.update(status="blocked_input", code=error.code)
    except OSError:
        report.update(status="blocked_input", code="CSV_INPUT_UNREADABLE")
    return _seal(report)


def _frame(stream, spec, report, deadline):
    stream.seek(0)
    records = _records(stream, deadline)
    columns = report["columns"]
    buffers = [[] for _ in columns]
    chunks = [[] for _ in columns]

    def flush():
        for buffer, parts, column in zip(buffers, chunks, columns):
            if buffer:
                parts.append(pd.Series(buffer, dtype=column["dtype"]))
                buffer.clear()

    try:
        next(records)
        count = 0
        for row in records:
            _require(len(row) == len(columns), "CSV_ROW_WIDTH_MISMATCH")
            for buffer, token, column in zip(buffers, row, columns):
                dtype = column["dtype"]
                value = (pd.NA if token == "" and spec.empty_field == "missing" else token if dtype in ("string", "object")
                         else int(token) if dtype.lower().startswith(("int", "uint")) else float(token))
                buffer.append(value)
            count += 1
            if count % 1024 == 0:
                flush()
        _require(count == report["rows"], "CSV_INSPECTION_MISMATCH")
        flush()
        return pd.DataFrame({column["name"]: pd.concat(parts, ignore_index=True) for column, parts in zip(columns, chunks)})
    finally:
        records.close()


def materialize_csv(snapshot_file, spec, directory, *, expected_snapshot, expected_inspection, max_output_bytes):
    """같은 바이트·해석·검사 결과를 확인한 뒤 새 폴더에만 canonical 표를 생성한다."""
    _require(type(spec) is CsvReadSpec, "CSV_SPEC_INVALID")
    _require(type(max_output_bytes) is int and 0 < max_output_bytes <= 2**63 - 1, "CSV_LIMITS_INVALID")
    snapshot = asdict(expected_snapshot) if is_dataclass(expected_snapshot) and not isinstance(expected_snapshot, type) else expected_snapshot
    _require(type(snapshot) is dict and set(snapshot) == {"sha256", "size_bytes", "max_input_bytes"}, "CSV_SNAPSHOT_INVALID")
    _require(type(snapshot["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", snapshot["sha256"]) is not None, "CSV_SNAPSHOT_INVALID")
    _require(type(snapshot["size_bytes"]) is int and type(snapshot["max_input_bytes"]) is int
             and 0 < snapshot["size_bytes"] <= snapshot["max_input_bytes"] <= 2**63 - 1, "CSV_SNAPSHOT_INVALID")
    _require(type(expected_inspection) is dict and expected_inspection.get("status") == "ready", "CSV_INSPECTION_INVALID")
    try:
        expected_json = _json(expected_inspection)
    except (ValueError, TypeError, RecursionError):
        raise CsvLoadError("CSV_INSPECTION_INVALID") from None
    _require(len(expected_json.encode("utf-8")) <= MAX_INSPECTION_BYTES + 128, "CSV_INSPECTION_TOO_LARGE")
    limits = _limits(expected_inspection.get("limits"))
    deadline = time.monotonic() + limits["wall_seconds"]
    with _snapshot_stream(snapshot_file, snapshot["max_input_bytes"], deadline) as (stream, actual):
        _require(actual == snapshot, "CSV_SNAPSHOT_CHANGED")
        report = _base(spec, limits, snapshot["max_input_bytes"])
        report["snapshot"] = actual
        try:
            report = _seal(_analyze(stream, spec, report, deadline))
        except UnicodeError:
            raise CsvLoadError("CSV_ENCODING_INVALID") from None
        _require(_json(report) == expected_json, "CSV_INSPECTION_MISMATCH")
        frame = _frame(stream, spec, report, deadline)
        provenance = {"source_snapshot_sha256": actual["sha256"], "source_snapshot_size_bytes": actual["size_bytes"],
                      "loader_id": spec.loader, "loader_version": "1.0.0", "read_spec": spec.to_dict(),
                      "inspection_id": report["inspection_id"], "policy_version": POLICY_VERSION,
                      "inspection_sha256": _digest(report),
                      "inference_policy": INFERENCE_POLICY, "header_mapping": report["header_mapping"],
                      "row_identity": "logical_csv_record_0_based"}
        _check_time(deadline)
        return write_bundle(frame, directory, provenance=provenance, max_output_bytes=max_output_bytes)

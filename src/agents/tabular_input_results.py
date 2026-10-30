"""원본 CSV를 다시 읽지 않고 입력 worker 결과의 파일·형식·요청 계보를 검증한다."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from src.preprocessing.tabular_bundle import BundleRef, inspect_bundle, read_bundle_schema, read_bundle_provenance, _safe
from src.preprocessing.tabular_loaders import INFERENCE_POLICY
from . import strict_json
from .input_snapshot import Snapshot
from .tabular_input_worker import build_packet, POLICY_VERSION


MAX_REPORT_BYTES = 16 * 1024 * 1024
CODES = frozenset({"READY", "DUPLICATE_COLUMN_NAMES", "CSV_EMPTY_INPUT", "CSV_EMPTY_DATA", "EMPTY_COLUMN_NAME",
                   "CSV_TOO_MANY_COLUMNS", "CSV_HEADER_TOO_LONG", "CSV_HEADER_REPAIR_CONFLICT", "CSV_ENCODING_INVALID",
                   "CSV_MALFORMED", "CSV_ROW_WIDTH_MISMATCH", "CSV_RECORD_TOO_LARGE", "CSV_INPUT_TOO_LARGE",
                   "CSV_INPUT_UNSAFE", "CSV_INPUT_UNREADABLE", "CSV_INPUT_ACCESS_DENIED", "CSV_TIME_LIMIT",
                   "CSV_TABLE_TOO_LARGE", "CSV_SNAPSHOT_CHANGED", "CSV_INSPECTION_TOO_LARGE"})


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _require(value):
    if not value:
        raise ValueError("입력 worker 결과의 형식·지문 또는 계보가 일치하지 않습니다.")


def _int(value, low=0, high=2_000_000):
    return type(value) is int and low <= value <= high


def _read(root, name, maximum):
    path = _safe(root / name, directory=False)
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    _require(len(payload) <= maximum)
    parsed = strict_json.loads(payload.decode("utf-8"), maximum)
    _require(payload == _json(parsed).encode("utf-8"))
    return parsed, hashlib.sha256(payload).hexdigest()


def _inspection(report, packet):
    keys = {"schema_version", "policy_version", "inference_policy", "loader_version", "spec", "limits", "max_input_bytes",
            "snapshot", "status", "code", "rows", "columns", "header_mapping", "detail", "candidates", "inspection_id"}
    _require(type(report) is dict and set(report) == keys)
    _require(report["schema_version"] == "csv_inspection_v1" and report["policy_version"] == POLICY_VERSION
             and report["inference_policy"] == INFERENCE_POLICY and report["loader_version"] == "1.0.0")
    _require(_json(report["spec"]) == _json(packet["read_spec"]) and _json(report["limits"]) == _json(packet["limits"]))
    _require(type(report["max_input_bytes"]) is int and report["max_input_bytes"] == packet["snapshot"]["max_input_bytes"])
    _require(report["snapshot"] is None or _json(report["snapshot"]) == _json(packet["snapshot"]))
    _require(report["inspection_id"] == _hash({key: value for key, value in report.items() if key != "inspection_id"}))
    _require(report["status"] in ("ready", "blocked_input", "review_required") and report["code"] in CODES)
    _require(type(report["columns"]) is list and type(report["header_mapping"]) is list
             and type(report["detail"]) is dict and type(report["candidates"]) is list)
    _require(set(report["detail"]) <= {"duplicated_names", "affected_columns"}
             and all(_int(value, high=4096) for value in report["detail"].values()))
    _require(report["candidates"] in ([], ["number_duplicates"]))
    if report["status"] != "ready":
        _require(report["code"] != "READY" and type(report["rows"]) is int and report["rows"] == 0
                 and report["columns"] == [] and report["header_mapping"] == [])
        return
    _require(report["code"] == "READY" and report["snapshot"] is not None and _int(report["rows"], 1))
    _require(0 < len(report["columns"]) <= 4096 and report["rows"] * len(report["columns"]) <= 100_000_000)
    names = []
    for index, column in enumerate(report["columns"]):
        _require(type(column) is dict and set(column) == {"id", "name", "dtype", "missing_count", "numeric_success_count", "numeric_failure_count", "precision_risk_count"})
        _require(column["id"] == f"c{index}" and type(column["name"]) is str and 0 < len(column["name"]) <= 256)
        _require(column["dtype"] in ("string", "object", "int64", "Int64", "uint64", "UInt64", "float64", "Float64"))
        _require(all(_int(column[key], high=report["rows"]) for key in ("missing_count", "numeric_success_count", "numeric_failure_count", "precision_risk_count")))
        _require(sum(column[key] for key in ("missing_count", "numeric_success_count", "numeric_failure_count")) == report["rows"])
        names.append(column["name"])
    _require(len(set(names)) == len(names))
    positions = set()
    for entry in report["header_mapping"]:
        _require(type(entry) is dict and set(entry) == {"position", "original", "renamed"})
        _require(_int(entry["position"], high=len(names) - 1) and entry["position"] not in positions)
        _require(type(entry["original"]) is str and 0 < len(entry["original"]) <= 256
                 and entry["renamed"] == names[entry["position"]] and entry["original"] != entry["renamed"])
        positions.add(entry["position"])
    _require(not positions or packet["read_spec"]["header_repair"] == "number_duplicates")


def _binding(directory, request, action, committed):
    root = _safe(directory, directory=True)
    names = {"inspection.json", "binding.json"} | ({"table"} if action == "materialize" else set())
    if committed:
        names.add("AGENT_COMPLETE.json")
    _require({path.name for path in root.iterdir()} == names)
    binding, _ = _read(root, "binding.json", 65536)
    _require(set(binding) == {"version", "request_id", "packet", "bundle", "inspection_sha256"})
    _require(binding["version"] == "tabular_input_result_v1" and binding["request_id"] == request.request_id)
    packet = binding["packet"]
    _require(type(packet) is dict)
    expected = build_packet(action, packet["snapshot"], packet["read_spec"], packet["limits"], packet["code_id"],
                            runtime_id=packet["runtime_id"], inspection_id=packet["inspection_id"], inspection_sha256=packet["inspection_sha256"], max_output_bytes=packet["max_output_bytes"])
    _require(_json(expected) == _json(packet) and _hash(packet) == request.config_id and request.policy_version == POLICY_VERSION
             and request.stage == "ingest" and request.action_id == "run_ingest" and Snapshot(**packet["snapshot"]).source_id == request.source_id
             and asdict(request.limits) == packet["limits"])
    total = sum(_safe(path, directory=False).stat().st_size for path in root.rglob("*") if path.is_file() and path.name != "AGENT_COMPLETE.json")
    _require(total <= packet["max_output_bytes"])
    report, digest = _read(root, "inspection.json", MAX_REPORT_BYTES)
    _require(digest == binding["inspection_sha256"])
    _inspection(report, packet)
    return root, binding, report, digest


def read_inspection(directory, expected_request, *, committed=False):
    """고정 요청에 묶인 구조 검사 결과만 반환한다. 원본 대비 통계 동등성 증명은 아니다."""
    root, binding, report, digest = _binding(directory, expected_request, "inspect", committed)
    _require(binding["bundle"] is None)
    return {"inspection": report, "inspection_sha256": digest}


def read_materialization(directory, expected_request, *, committed=False):
    """검증된 생성 묶음과 선행 검사·확정 해석의 연결을 대조한다."""
    root, binding, report, digest = _binding(directory, expected_request, "materialize", committed)
    packet = binding["packet"]
    _require(report["status"] == "ready" and digest == packet["inspection_sha256"] and report["inspection_id"] == packet["inspection_id"])
    reference = BundleRef.from_dict(binding["bundle"])
    inspect_bundle(root / "table", expected=reference, max_bytes=packet["max_output_bytes"])
    schema = read_bundle_schema(root / "table", expected=reference)
    _require(reference.rows == report["rows"] and reference.columns == len(report["columns"]))
    _require([(item["name"], item["dtype"]) for item in schema["columns"]]
             == [(item["name"], {"kind": "plain", "dtype": item["dtype"]}) for item in report["columns"]])
    provenance = read_bundle_provenance(root / "table", expected=reference)
    for key, value in {"source_snapshot_sha256": packet["snapshot"]["sha256"], "source_snapshot_size_bytes": packet["snapshot"]["size_bytes"],
                       "loader_id": packet["read_spec"]["loader"], "read_spec": packet["read_spec"],
                       "inspection_id": packet["inspection_id"], "inspection_sha256": digest, "policy_version": POLICY_VERSION}.items():
        _require(_json(provenance.get(key)) == _json(value))
    return {"inspection": report, "inspection_sha256": digest, "bundle": asdict(reference)}

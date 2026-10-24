"""CSV 전달 묶음의 안전한 형식·해시·행 연결 검사. 모델·pickle을 불러오지 않는다."""

import csv
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import re
import tempfile

from src.preprocessing.contracts import ColumnRoles
from src.preprocessing.transform import PreparationRules, _validate_state
from src.preprocessing.split import _digest
from .artifacts import _root, _fingerprint
from .partition_csv import iter_csv_rows
from .split_evidence import read_metadata
from .worker_paths import _path


PREPARED_FILES = ("train_features.csv", "validation_features.csv", "train_rows.json",
                  "validation_rows.json", "application.json", "application.md")
PAYLOAD_FILES = tuple("prepared/" + name for name in PREPARED_FILES) + (
    "preprocessor.json", "feature_mapping.json", "provenance.json", "README.md",
)
MAX_BUNDLE_BYTES = 2 * 1024 ** 3
PROVENANCE_FIELDS = {"schema_version", "pipeline_name", "pipeline_signature", "candidate_id", "apply_run_id",
                     "apply_plan_id", "preprocessor_sha256", "application_binding", "test_used", "model_trained"}
# 완료 기록 형식별 고정 규칙. 묶음이 선언한 형식은 workflow 검증에서 기대 버전과 다시 대조한다.
VERSIONS = {
    "csv_preparation_handoff_v1": {"provenance": "csv_handoff_provenance_v1", "fields": PROVENANCE_FIELDS, "typed": False,
                                   "applications": {("preparation_application_v1", False), ("preparation_application_v2", True)}},
    "tabular_preparation_handoff_v2": {"provenance": "tabular_handoff_provenance_v2", "typed": True,
                                       "fields": PROVENANCE_FIELDS | {"source_contract_id"},
                                       "applications": {("preparation_application_v3", True)}},
}


def feature_mapping(state, columns):
    """원본 컬럼을 이름 파싱으로 추측하지 않고 저장된 기록 순서로 연결한다."""
    rows = []

    for record in state["records"]:
        for offset in range(record["width"]):
            rows.append({"feature": columns[len(rows)], "source_column": record["column"],
                         "kind": record["kind"], "output_index": offset})

    return rows


def _saved_state(root):
    """JSON 구조 검증만 재사용한다. Pipeline 생성·fit·transform을 하지 않는다."""
    payload = read_metadata(root / "preprocessor.json")

    if set(payload) != {"schema_version", "roles", "rules", "state", "output_columns"} or payload["schema_version"] != "fitted_tabular_prepare_v1":
        raise ValueError("전달할 저장 기준의 형식이 올바르지 않습니다.")

    values = dict(payload["roles"])

    for key in ("numeric", "categorical", "identifiers"):
        if type(values.get(key)) is not list:
            raise ValueError("저장 기준의 컬럼 역할이 올바르지 않습니다.")

        values[key] = tuple(values[key])

    roles = ColumnRoles(**values)
    rules = PreparationRules(**payload["rules"])
    state = _validate_state(payload["state"], roles, rules)
    columns = [f"feature_{index}__{record['column']}__{offset}"
               for index, record in enumerate(state["records"]) for offset in range(record["width"])]

    if payload["output_columns"] != columns:
        raise ValueError("출력 특징 순서가 저장 기준과 다릅니다.")

    return roles, rules, state, columns


def _partition(root, name, record, roles, rules, state, columns, *, typed=False):
    """CSV 행과 JSON 메타데이터를 순서대로 연결한다. 라벨 값은 변환하지 않는다."""
    if type(record) is not dict or set(record) != {"rows", "features", "row_contract"}:
        raise ValueError("적용 구간 기록이 올바르지 않습니다.")

    count = record["rows"]

    if type(count) is not int or count < 1 or count * len(columns) > rules.max_output_cells:
        raise ValueError("전달 구간의 행 수가 올바르지 않습니다.")

    for suffix, key in (("features.csv", "features"), ("rows.json", "row_contract")):
        expected = record[key]

        if type(expected) is not dict or set(expected) != {"size_bytes", "sha256"}:
            raise ValueError("적용 파일의 해시 기록이 올바르지 않습니다.")

        if _fingerprint(root / "prepared" / f"{name}_{suffix}") != {"size": expected["size_bytes"], "sha256": expected["sha256"]}:
            raise ValueError("적용 파일이 원래 기록과 다릅니다.")

    row_data = read_metadata(root / "prepared" / f"{name}_rows.json")
    expected_metadata = list(dict.fromkeys(value for value in (roles.target, roles.group, roles.timestamp, *roles.identifiers) if value is not None))

    if typed:
        from .tabular_preparation_worker import load_row_metadata

        # 원래 자료형 스키마로 실제 복원되는지 확인한다. 문자열로 낮춘 메타데이터는 거부한다.
        try:
            frame = load_row_metadata(root / "prepared" / f"{name}_rows.json")
        except (TypeError, KeyError, IndexError, OverflowError) as error:
            raise ValueError("typed 행 메타데이터를 복원할 수 없습니다.") from error

        if row_data["metadata_columns"] != expected_metadata or list(frame.columns) != expected_metadata or len(frame) != count:
            raise ValueError("행 연결의 메타데이터 역할이 다릅니다.")
    elif set(row_data) != {"row_positions", "metadata_columns", "metadata"} or row_data["metadata_columns"] != expected_metadata:
        raise ValueError("행 연결의 메타데이터 역할이 다릅니다.")

    positions = row_data["row_positions"]
    metadata = [] if typed else row_data["metadata"]

    if typed and (type(positions) is not list or len(positions) != count or positions != sorted(set(positions))
                  or any(type(value) is not int or value < 0 for value in positions)):
        raise ValueError("행 위치 연결이 올바르지 않습니다.")
    elif not typed and (type(positions) is not list or len(positions) != count
            or any(type(value) is not int or value < 0 for value in positions)
            or positions != sorted(set(positions)) or type(metadata) is not list or len(metadata) != count
            or any(type(row) is not list or len(row) != len(expected_metadata) or any(type(value) is not str for value in row) for row in metadata)):
        raise ValueError("행 위치 또는 메타데이터 연결이 올바르지 않습니다.")

    allowed_missing = [record["kind"] == "numeric" and rules.numeric_imputation == "none" and record["edges"] is None
                       for record in state["records"] for _ in range(record["width"])]
    old_limit = csv.field_size_limit(1024 * 1024)

    try:
        with (root / "prepared" / f"{name}_features.csv").open(encoding="utf-8", newline="") as stream:
            reader = iter_csv_rows(stream, max_columns=len(columns))

            if next(reader, None) != columns:
                raise ValueError("특징 CSV의 컬럼 순서가 다릅니다.")

            observed = 0

            for row in reader:
                observed += 1

                if observed > count or len(row) != len(columns):
                    raise ValueError("특징 CSV의 행 수 또는 폭이 다릅니다.")

                for value, allowed in zip(row, allowed_missing):
                    if value == "" and allowed:
                        continue

                    if not value or not math.isfinite(float(value)):
                        raise ValueError("특징 값의 수치 형식이 올바르지 않습니다.")

            if observed != count:
                raise ValueError("특징 CSV와 행 연결 JSON의 행 수가 다릅니다.")
    finally:
        csv.field_size_limit(old_limit)

    return positions


def verify_bundle(directory):
    """출처 로그 없이 묶음 자체를 검사한다. 승인 이력·수학적 정답·진본 인증은 아니다."""
    root = _root(directory)
    manifest_path = _path(root / "HANDOFF_COMPLETE.json", directory=False)

    if manifest_path.stat().st_size > 65536:
        raise ValueError("전달 완료 기록은 64 KiB 이하여야 합니다.")

    manifest = read_metadata(manifest_path)

    if (set(manifest) != {"schema_version", "files", "payload_bytes", "max_bundle_bytes"}
            or type(manifest["schema_version"]) is not str or manifest["schema_version"] not in VERSIONS
            or type(manifest["files"]) is not dict or set(manifest["files"]) != set(PAYLOAD_FILES)
            or type(manifest["max_bundle_bytes"]) is not int or not 0 < manifest["max_bundle_bytes"] <= MAX_BUNDLE_BYTES
            or type(manifest["payload_bytes"]) is not int):
        raise ValueError("전달 묶음의 완료 기록이 올바르지 않습니다.")

    if {path.name for path in root.iterdir()} != {"prepared", "preprocessor.json", "feature_mapping.json", "provenance.json", "README.md", "HANDOFF_COMPLETE.json"}:
        raise ValueError("전달 묶음에 누락되거나 허용되지 않은 파일이 있습니다.")

    prepared = _root(root / "prepared")

    if {path.name for path in prepared.iterdir()} != set(PREPARED_FILES):
        raise ValueError("전달할 적용 파일 목록이 다릅니다.")

    total = 0

    for relative, expected in manifest["files"].items():
        path = _path(root / relative, directory=False)

        if type(expected) is not dict or set(expected) != {"size", "sha256"} or type(expected["size"]) is not int or expected["size"] < 1:
            raise ValueError("전달 파일의 해시 기록이 올바르지 않습니다.")

        total += path.stat().st_size

        if total > manifest["max_bundle_bytes"] or _fingerprint(path) != expected:
            raise ValueError("전달 파일의 크기 또는 해시가 다릅니다.")

    if total != manifest["payload_bytes"] or total + manifest_path.stat().st_size > manifest["max_bundle_bytes"]:
        raise ValueError("전달 파일의 합계 크기가 다릅니다.")

    roles, rules, state, columns = _saved_state(root)

    if read_metadata(root / "feature_mapping.json") != {"schema_version": "feature_mapping_v1", "features": feature_mapping(state, columns)}:
        raise ValueError("출력 특징과 원본 컬럼의 연결 기록이 다릅니다.")

    provenance = read_metadata(root / "provenance.json")
    version = VERSIONS[manifest["schema_version"]]
    report = read_metadata(root / "prepared/application.json")

    if version["typed"] and (type(provenance.get("source_contract_id")) is not str
                             or re.fullmatch(r"tabular_source:[0-9a-f]{64}", provenance["source_contract_id"]) is None
                             or any(type(part) is not dict or part.get("parent_source_id") != provenance["source_contract_id"]
                                    for part in ((report.get("binding") or {}).get("partitions") or {"": None}).values())):
        raise ValueError("typed 전달의 출처 ID가 적용 기록과 다릅니다.")

    if (set(provenance) != version["fields"] or provenance["schema_version"] != version["provenance"]
            or provenance["test_used"] is not False or provenance["model_trained"] is not False
            or provenance["preprocessor_sha256"] != manifest["files"]["preprocessor.json"]["sha256"]
            # v2는 적용 검증기가 대조한 구간별 영향 합계(impact)를 더한다. v1 전달 묶음도 그대로 검증한다.
            or set(report) - {"impact"} != {"schema_version", "status", "binding", "partitions", "transformed", "refit", "test_used", "model_trained"}
            or type(report["schema_version"]) is not str
            or (report["schema_version"], "impact" in report) not in version["applications"]
            or report["status"] != "applied"
            or report["transformed"] is not True or report["refit"] is not False or report["test_used"] is not False
            or report["model_trained"] is not False or report["binding"] != provenance["application_binding"]
            or set(report["partitions"]) != {"train", "validation"}
            or report["binding"].get("roles") != json_roles(roles)
            or report["binding"].get("rules") != asdict(rules)
            or report["binding"].get("candidate_id") != provenance["candidate_id"]
            or report["binding"].get("preprocessor_sha256") != provenance["preprocessor_sha256"]
            or report["binding"].get("output_feature_count") != len(columns)
            or report["binding"].get("output_columns_sha256") != _digest(columns)):
        raise ValueError("출처와 적용 기록이 일치하지 않습니다.")

    observed = {}

    for name in ("train", "validation"):
        observed[name] = _partition(root, name, report["partitions"][name], roles, rules, state, columns, typed=version["typed"])

        part = report["binding"]["partitions"][name]

        if part["rows"] != len(observed[name]) or part["positions_sha256"] != _digest(observed[name]):
            raise ValueError("구간 행 위치가 적용 계약과 다릅니다.")

    if set(observed["train"]) & set(observed["validation"]) or state["train_rows"] != len(observed["train"]):
        raise ValueError("구간 중복 또는 Train 기준 산출 행 수가 다릅니다.")

    # 검사 도중 변경도 거부한다. 결과에는 상세 통계·라벨을 출력하지 않는다.
    for relative, expected in manifest["files"].items():
        if _fingerprint(_path(root / relative, directory=False)) != expected:
            raise ValueError("전달 검사 중 파일이 변경되었습니다.")

    return {"status": "handoff_verified", "test_used": False, "model_trained": False,
            "schema_version": manifest["schema_version"], "scope": "bundle_integrity_structure_and_row_linkage_only"}


def _snapshot_bundle(root, target):
    """완료 기록 바이트를 한 번 읽어 고정하고, 그 기록이 가리키는 파일만 해시를 확인하며 비공개 사본으로 복사한다.

    이후 검사와 해석은 모두 이 사본에서 수행하므로 검사한 기록·파일과 소비하는 바이트가 같다.
    """
    manifest_path = _path(root / "HANDOFF_COMPLETE.json", directory=False)

    with manifest_path.open("rb") as stream:
        sealed = stream.read(65537)

    if len(sealed) > 65536:
        raise ValueError("전달 완료 기록은 64 KiB 이하여야 합니다.")

    (target / "HANDOFF_COMPLETE.json").write_bytes(sealed)
    manifest = read_metadata(target / "HANDOFF_COMPLETE.json")
    files = manifest.get("files") if type(manifest) is dict else None
    limit = manifest.get("max_bundle_bytes") if type(manifest) is dict else None

    if (type(files) is not dict or set(files) != set(PAYLOAD_FILES) or type(limit) is not int or not 0 < limit <= MAX_BUNDLE_BYTES
            or any(type(item) is not dict or type(item.get("size")) is not int or item["size"] < 1
                   or type(item.get("sha256")) is not str or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                   for item in files.values())):
        raise ValueError("전달 묶음의 완료 기록이 올바르지 않습니다.")

    # 원본 폴더의 추가·누락 파일도 사본 검사와 같은 기준으로 거부한다.
    if ({path.name for path in root.iterdir()} != {"prepared", "preprocessor.json", "feature_mapping.json", "provenance.json", "README.md", "HANDOFF_COMPLETE.json"}
            or {path.name for path in _root(root / "prepared").iterdir()} != set(PREPARED_FILES)):
        raise ValueError("전달 묶음에 누락되거나 허용되지 않은 파일이 있습니다.")

    (target / "prepared").mkdir()
    total = 0

    for relative in PAYLOAD_FILES:
        expected = files[relative]
        digest = hashlib.sha256()
        size = 0

        with _path(root / relative, directory=False).open("rb") as source, (target / relative).open("xb") as copy:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                total += len(chunk)

                if size > expected["size"] or total > limit:
                    raise ValueError("전달 파일의 크기가 완료 기록과 다릅니다.")

                digest.update(chunk)
                copy.write(chunk)

        if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError("전달 파일의 크기 또는 해시가 완료 기록과 다릅니다.")


def load_handoff_partition(directory, partition):
    """typed 묶음을 검사한 뒤 특징 표와 원래 자료형의 행 메타데이터를 읽는다. 모델·승인 호출은 없다."""
    import pandas as pd
    from .tabular_preparation_worker import load_row_metadata

    if partition not in ("train", "validation"):
        raise ValueError("Train 또는 Validation 구간만 읽을 수 있습니다.")

    root = _root(directory)

    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch)
        _snapshot_bundle(root, target)

        if verify_bundle(target)["schema_version"] != "tabular_preparation_handoff_v2":
            raise ValueError("자료형 보존 v2 전달 묶음이 아닙니다.")

        _, _, _, columns = _saved_state(target)

        # %.17g로 쓴 float64 값을 추가 반올림 없이 같은 값으로 읽는다.
        with (target / "prepared" / f"{partition}_features.csv").open(encoding="utf-8", newline="") as stream:
            features = pd.read_csv(stream, dtype="float64", float_precision="round_trip")

        rows = load_row_metadata(target / "prepared" / f"{partition}_rows.json")

    if list(features.columns) != columns or len(features) != len(rows):
        raise ValueError("특징 표와 행 메타데이터의 연결이 다릅니다.")

    # i번째 특징 행과 i번째 메타데이터 행이 대응한다. 인덱스는 원래 입력의 인덱스다.
    return features, rows


def json_roles(roles):
    """JSON 배열 표현과 불변 tuple 역할을 같은 표현으로 비교한다."""
    values = asdict(roles)

    for key in ("numeric", "categorical", "identifiers"):
        values[key] = list(values[key])

    return values

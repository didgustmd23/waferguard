"""typed 전처리의 고정 구간 참조·입력 뷰·스칼라 메타데이터. 승인 모듈은 import하지 않는다."""

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from src.preprocessing.split import _digest
from src.preprocessing.tabular_bundle import (BundleRef, FILES, inspect_bundle, iter_bundle_rows, load_bundle,
    read_bundle_schema, read_bundle_provenance, worker_filesystem, _encode, _scalar, _series)


CANDIDATE_POLICY = "preparation_candidate_v2"
APPLY_POLICY = "preparation_apply_v2"


def input_files(root, *, applying=False):
    """참조할 두 구간과 설정 파일만 고정한다. Test 경로는 없다."""
    names = [f"{part}/{name}" for part in (("train", "validation") if applying else ("train",)) for name in sorted(FILES)]
    names += ["preprocessor.json", "positions.json", "apply_task.json"] if applying else ["candidate_task.json"]
    return tuple(str(Path(root) / "input" / name) for name in names)


def check_part(directory, packet, *, partition=None):
    """기대 BundleRef·구간·행 위치·생성 계보를 검사한다. 승인 DB 접근은 부모 책임이다."""
    keys = {"schema_version", "partition", "contract_id", "source_id", "bundle", "rows", "positions_sha256",
            "parent_source_id", "split_signature", "export_marker_sha256"}
    if (type(packet) is not dict or set(packet) != keys or packet["schema_version"] != "tabular_partition_input_v2"
            or packet["partition"] not in ("train", "validation") or partition is not None and packet["partition"] != partition
            or packet["source_id"] != "tabular_partition:" + packet["contract_id"]):
        raise ValueError("고정 v2 구간 참조가 필요합니다.")
    for key in ("contract_id", "positions_sha256", "split_signature", "export_marker_sha256"):
        value = packet[key]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("구간 지문이 올바르지 않습니다.")
    parent = packet["parent_source_id"]
    import re
    # 원래 생성 표 또는 시간 복원 파생 표 출처(tabular_source_contract.SOURCE_ID_PATTERN과 같은 형식).
    if type(parent) is not str or re.fullmatch(r"tabular_(?:time_)?source:[0-9a-f]{64}", parent) is None:
        raise ValueError("부모 생성 출처가 필요합니다.")
    reference = BundleRef.from_dict(packet["bundle"])
    if type(packet["rows"]) is not int or reference.rows != packet["rows"] or not 1 <= reference.rows <= 1_000_000:
        raise ValueError("구간 행 수가 다릅니다.")
    inspect_bundle(directory, expected=reference, max_bytes=reference.size_bytes)
    provenance = read_bundle_provenance(directory, expected=reference)
    expected = {"parent_source_id": packet["parent_source_id"], "split_signature": packet["split_signature"],
                "row_positions_sha256": packet["positions_sha256"]}
    if provenance != expected:
        raise ValueError("구간의 생성 계보가 다릅니다.")
    positions = [row[0] for row in iter_bundle_rows(directory, expected=reference)]
    if len(positions) != reference.rows or _digest(positions) != packet["positions_sha256"]:
        raise ValueError("구간 행 위치가 다릅니다.")
    return reference


def load_part(root, packet, name):
    """worker가 허용된 구간 폴더 아래에서만 typed 표를 복원한다."""
    with worker_filesystem(root):
        directory = Path(root) / "input" / name
        reference = check_part(directory, packet, partition=name)
        return load_bundle(directory, expected=reference)


def feature_view(frame, roles):
    """범주 사전 dtype만 값이 같은 object 뷰로 바꿔 기존 변환기에 전달한다."""
    selected = frame.loc[:, list(roles.feature_columns)].copy()
    for name in roles.categorical:
        if isinstance(selected[name].dtype, pd.CategoricalDtype):
            selected[name] = selected[name].astype(object)
    return selected


def row_metadata(directory, reference, roles):
    """안전한 스칼라와 원래 스키마로 라벨·그룹·ID·원래 인덱스를 보존한다."""
    schema = read_bundle_schema(directory, expected=reference)
    names = list(dict.fromkeys(v for v in (roles.target, roles.group, roles.timestamp, *roles.identifiers) if v is not None))
    offsets = {item["name"]: i for i, item in enumerate(schema["columns"])}
    metadata, positions, indices = [], [], []
    for position, index, values in iter_bundle_rows(directory, expected=reference):
        positions.append(position)
        indices.append(_encode(index))
        metadata.append([_encode(values[offsets[name]]) for name in names])
    return {"schema_version": "typed_row_metadata_v2", "scalar_encoding": schema["scalar_encoding"],
            "row_positions": positions, "metadata_columns": names,
            "metadata_schema": [schema["columns"][offsets[name]] for name in names],
            "index_schema": schema["index"], "index": indices, "columns_name": schema["columns_name"], "metadata": metadata}


def load_row_metadata(path):
    """검증된 로컬 부속 파일을 원래 dtype으로 복원한다. 생성·승인·모델 호출은 하지 않는다."""
    from .strict_json import loads
    from src.preprocessing.tabular_bundle import _safe
    path = _safe(path, directory=False)
    with path.open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    value = loads(raw.decode("utf-8"), 8 * 1024 * 1024)
    keys = {"schema_version", "scalar_encoding", "row_positions", "metadata_columns", "metadata_schema", "index_schema", "index", "columns_name", "metadata"}
    if type(value) is not dict or set(value) != keys or value["schema_version"] != "typed_row_metadata_v2" or value["scalar_encoding"] != "tagged_scalar_v2":
        raise ValueError("typed 메타데이터 형식이 다릅니다.")
    positions, names = value["row_positions"], value["metadata_columns"]
    if type(positions) is not list or any(type(p) is not int or not 0 <= p < 2_000_000 for p in positions) or positions != sorted(set(positions)):
        raise ValueError("typed 메타데이터 행 위치가 올바르지 않습니다.")
    count = len(positions)
    if (not 1 <= count <= 1_000_000 or len(value["index"]) != count or len(value["metadata"]) != count
            or type(names) is not list or any(type(name) is not str or not 0 < len(name) <= 256 for name in names)
            or names != [item["name"] for item in value["metadata_schema"]] or len(set(names)) != len(names)
            or any(type(row) is not list or len(row) != len(names) for row in value["metadata"])):
        raise ValueError("typed 메타데이터 행·열 수가 다릅니다.")
    frame = pd.DataFrame({name: _series([_scalar(row[i]) for row in value["metadata"]], value["metadata_schema"][i]["dtype"])
                          for i, name in enumerate(names)})
    if not names:
        frame = pd.DataFrame(index=range(count))
    index = _series([_scalar(v) for v in value["index"]], value["index_schema"]["dtype"])
    frame.index = pd.Index(index, name=_scalar(value["index_schema"]["name"]))
    frame.columns.name = value["columns_name"]
    return frame

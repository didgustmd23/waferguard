"""typed 구간의 전처리 사전 검사와 기존 도구 staging 연결. 변환 계산은 하지 않는다."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from src.preprocessing.tabular_bundle import read_bundle_schema, iter_bundle_rows, inspect_bundle, scoped_result
from src.preprocessing.transform import _category
from .artifacts import _root, _fingerprint
from .tabular_source_contract import TabularPartitionInputContract
from .tabular_preparation_worker import input_files
from .worker_paths import WorkerPathPolicy
from .snapshot_eda import _digest


def partition_packet(part):
    """구간의 고정 참조만 IPC에 넣는다. 승인 DB·외부 경로·전체 배정표는 넣지 않는다."""
    if type(part) is not TabularPartitionInputContract:
        raise TypeError("v2 구간 계약이 필요합니다.")
    part.check()
    return checked_partition_packet(part)


def checked_partition_packet(part):
    """같은 호출 안에서 part.check()를 이미 마친 구간의 참조만 만든다. 검사를 대신하지 않는다."""
    if type(part) is not TabularPartitionInputContract:
        raise TypeError("v2 구간 계약이 필요합니다.")
    return {"schema_version": part.schema_version, "partition": part.partition, "contract_id": part.contract_id,
            "source_id": part.source_id, "bundle": asdict(part.bundle), "rows": len(part.row_positions),
            "positions_sha256": _digest(list(part.row_positions)), "parent_source_id": part.parent_source_id,
            "split_signature": part.split.signature, "export_marker_sha256": part.export.marker_sha256}


def check_feature_types(part):
    """이미 검증된 묶음의 타입만 대조한다. 통계·대치값·특징 선택은 계산하지 않는다."""
    part.check()
    return checked_feature_types(part)


def checked_feature_types(part):
    """같은 호출 안에서 part.check()를 이미 마친 구간의 타입만 대조한다. 묶음 해시는 읽기 함수가 다시 확인한다."""
    schema = read_bundle_schema(part.bundle_directory, expected=part.bundle)
    columns = {item["name"]: item for item in schema["columns"]}
    warnings, categorical_features = [], []
    for name in part.roles.feature_columns:
        descriptor = columns[name]["dtype"]
        dtype = descriptor.get("dtype", "")
        numeric = dtype.lower().startswith(("int", "uint", "float")) and descriptor["kind"] == "plain"
        if name in part.roles.numeric and not numeric:
            raise ValueError("수치 특징은 bool·문자열·시각을 제외한 수치 자료형이어야 합니다.")
        if name in part.roles.categorical:
            selected = descriptor.get("categories_dtype", descriptor)
            if selected["kind"] == "timestamp":
                raise ValueError("시각 자료형은 현재 범주 특징 변환기의 지원 범위가 아닙니다.")
            categorical_features.append(name)
        if dtype in ("int64", "uint64", "Int64", "UInt64") or descriptor.get("categories_dtype", {}).get("dtype") in ("int64", "uint64", "Int64", "UInt64"):
            warnings.append({"code": "FLOAT64_PRECISION_POSSIBLE", "column": name})
    if categorical_features:
        offsets = {item["name"]: i for i, item in enumerate(schema["columns"])}

        def categories_pass():
            for _, _, values in iter_bundle_rows(part.bundle_directory, expected=part.bundle):
                for name in categorical_features:
                    _category(values[offsets[name]])
            return True

        # 같은 공개 호출 안에서는 같은 바이트·역할·검사 컬럼일 때만 범주 값 검사 통과를 공유한다(행은 보관하지 않음).
        scoped_result(part.bundle_directory, expected=part.bundle,
                      condition=("categories", part.roles_json, tuple(categorical_features)), compute=categories_pass)
    return warnings


def stage_parts(tool, request, *, applying=False):
    """승인 소비 후에만 고정 구간과 정확한 후보를 새 작업 폴더로 복사한다."""
    tool.contract.check()
    root = tool._work_root(request.task_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    _root(root.parent)
    root.mkdir(exist_ok=False)
    for name in ("runtime", "input", "tools", "output", "scratch"):
        (root / name).mkdir()
    parts = [("train", tool.contract.train)]
    if applying:
        parts.append(("validation", tool.contract.validation))
    for name, part in parts:
        shutil.copytree(part.bundle_directory, root / "input" / name)
        inspect_bundle(root / "input" / name, expected=part.bundle, max_bytes=part.bundle.size_bytes)
    if applying:
        shutil.copyfile(tool.contract.candidate.directory / "preprocessor.json", root / "input/preprocessor.json")
        if _fingerprint(root / "input/preprocessor.json")["sha256"] != tool.contract.candidate.state_sha256:
            raise ValueError("복사한 기준 파일이 승인 후보와 다릅니다.")
        positions = json.dumps(tool.contract.positions()).encode("utf-8")
        if len(positions) > 8 * 1024 * 1024:
            raise ValueError("행 위치 전달 상한을 초과했습니다.")
        (root / "input/positions.json").write_bytes(positions)
    settings = "apply_task.json" if applying else "candidate_task.json"
    (root / "input" / settings).write_text(json.dumps(tool.arguments, ensure_ascii=False), encoding="utf-8")
    (root / "input/request.json").write_text(request.to_json(), encoding="utf-8")
    Path(request.output_directory).mkdir()
    Path(request.scratch_directory).mkdir()
    request.check_policy(WorkerPathPolicy(tuple(root / n for n in ("runtime", "input", "tools")), root / "output", root / "scratch",
        protected_roots=(tool.states.directory, Path(tool.contract.train.export.run_directory), Path(tool.contract.train.split.artifact.run_directory))
                        + ((Path(tool.contract.candidate.artifact.run_directory),) if applying else ())), tool.limits)
    tool.contract.check()
    return root

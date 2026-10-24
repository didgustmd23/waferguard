"""구간 내보내기의 부모 파일 검증. 원본 CSV는 파싱하지 않는다."""

from dataclasses import asdict
import csv
import json

from src.preprocessing.split import _digest
from .artifacts import _root
from .input_snapshot import SNAPSHOT_VERSION, Snapshot, verify_snapshot
from .partition_export_contract import PartitionExportContract
from .partition_csv import iter_csv_rows
from .split_evidence import read_metadata
from .worker_paths import _path


def read_partition_export(directory, contract, *, committed=False):
    """예상 구간·출처·배정·역할·바이트·CSV 구조를 확인한다. 추출 정답의 독립 인증은 아니다."""
    from .partition_export_contract import TabularExportContract
    if type(contract) is TabularExportContract:
        return read_tabular_export(directory, contract, committed=committed)
    if not isinstance(contract, PartitionExportContract):
        raise TypeError("검증할 PartitionExportContract가 필요합니다.")

    old_field_limit = csv.field_size_limit(1024 * 1024)

    try:
        contract.check()
        root = _root(directory)
        expected_files = {"partition_exports.json", "train", "validation"}

        if committed:
            expected_files.add("AGENT_COMPLETE.json")

        if {path.name for path in root.iterdir()} != expected_files:
            return None

        receipt = read_metadata(root / "partition_exports.json")
        required = {"schema_version", "split_evidence_id", "source_id", "proposal_id", "roles", "partitions"}
        roles = contract.source.profile.roles

        if set(receipt) != required or receipt["schema_version"] != "partition_exports_v1":
            return None

        if (
            receipt["split_evidence_id"] != contract.split.signature
            or receipt["source_id"] != contract.split.artifact.scope.source_id
            or receipt["proposal_id"] != contract.split.proposal["proposal_id"]
            or receipt["roles"] != json.loads(json.dumps(asdict(roles)))
            or type(receipt["partitions"]) is not dict or set(receipt["partitions"]) != {"train", "validation"}
        ):
            return None

        expected_header = None

        for name in ("train", "validation"):
            record = receipt["partitions"][name]
            positions = contract.split.positions(name)

            if type(record) is not dict or set(record) != {"row_count", "row_positions_sha256", "snapshot"}:
                return None

            if type(record["row_count"]) is not int or record["row_count"] != len(positions) or record["row_positions_sha256"] != _digest(list(positions)):
                return None

            if type(record["snapshot"]) is not dict or set(record["snapshot"]) != set(Snapshot.__dataclass_fields__):
                return None

            snapshot = Snapshot(**record["snapshot"])

            if snapshot.max_input_bytes != contract.max_partition_bytes:
                return None

            marker = read_metadata(root / name / "snapshot.json")

            if marker != {"version": SNAPSHOT_VERSION, **asdict(snapshot)} or not verify_snapshot(root / name, snapshot):
                return None

            source = _path(root / name / "input.bin", directory=False)

            with source.open(encoding="utf-8", newline="") as stream:
                reader = iter_csv_rows(stream)
                header = next(reader, [])

                if not header or any(not value for value in header) or len(header) != len(set(header)) or not set(roles.required_columns).issubset(header):
                    return None

                if expected_header is not None and header != expected_header:
                    return None

                expected_header = header
                count = 0

                for row in reader:
                    count += 1

                    if count > len(positions) or len(row) != len(header):
                        return None

                if count != len(positions):
                    return None

            # 구조 검사 중 변경된 파일도 완료로 인정하지 않는다.
            if not verify_snapshot(root / name, snapshot):
                return None

        contract.check()

        return receipt
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, csv.Error, OverflowError, RecursionError):
        return None
    finally:
        csv.field_size_limit(old_field_limit)


def read_tabular_export(directory, contract, *, committed=False):
    """v2 구간의 스키마·내용 지문·원래 행 위치·생성 계보를 대조한다."""
    from src.preprocessing.tabular_bundle import BundleRef, inspect_bundle, read_bundle_schema, read_bundle_provenance, iter_bundle_rows
    try:
        contract.check()
        root = _root(directory)
        names = {"partition_exports.json", "train", "validation"} | ({"AGENT_COMPLETE.json"} if committed else set())
        if {p.name for p in root.iterdir()} != names:
            return None
        receipt = read_metadata(root / "partition_exports.json")
        source = contract.source.source
        if (set(receipt) != {"schema_version", "source_id", "split_evidence_id", "roles", "partitions"}
                or receipt["schema_version"] != "partition_exports_v2" or receipt["source_id"] != source.source_id
                or receipt["split_evidence_id"] != contract.split.signature
                or _digest(receipt["roles"]) != _digest(asdict(contract.source.profile.roles))
                or type(receipt["partitions"]) is not dict or set(receipt["partitions"]) != {"train", "validation"}):
            return None
        schema = read_bundle_schema(source.bundle_directory, expected=source.bundle)
        for name in ("train", "validation"):
            record = receipt["partitions"][name]
            positions = contract.split.positions(name)
            if type(record) is not dict or set(record) != {"bundle", "row_positions_sha256"}:
                return None
            reference = BundleRef.from_dict(record["bundle"])
            if reference.rows != len(positions) or reference.columns != source.bundle.columns or record["row_positions_sha256"] != _digest(list(positions)):
                return None
            inspect_bundle(root / name, expected=reference, max_bytes=contract.max_partition_bytes)
            actual_schema = read_bundle_schema(root / name, expected=reference)
            if actual_schema["columns"] != schema["columns"] or actual_schema["columns_name"] != schema["columns_name"]:
                return None
            if tuple(row[0] for row in iter_bundle_rows(root / name, expected=reference)) != positions:
                return None
            provenance = read_bundle_provenance(root / name, expected=reference)
            if provenance != {"parent_source_id": source.source_id, "split_signature": contract.split.signature,
                              "row_positions_sha256": record["row_positions_sha256"]}:
                return None
        contract.check()
        return receipt
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        return None

"""원본 파싱·fit·transform 없이 적용 출력의 파일·행·숫자·출처 계약을 확인한다."""

import csv
import json
import math

from .artifacts import _root, _fingerprint
from .isolated_eda import _read_file
from .partition_csv import iter_csv_rows
from .preparation_apply_io import application_markdown
from .split_evidence import read_metadata
from .worker_paths import _path


def _file_record(path, record, maximum):
    if type(record) is not dict or set(record) != {"size_bytes", "sha256"} or type(record["size_bytes"]) is not int or not 0 < record["size_bytes"] <= maximum:
        return False

    path = _path(path, directory=False)

    if path.stat().st_size != record["size_bytes"]:
        return False

    return _fingerprint(path) == {"size": record["size_bytes"], "sha256": record["sha256"]}


def read_application(directory, contract, code_id):
    """검증은 형식·일관성 검사이며 원본 대비 변환 수치의 독립 인증은 아니다."""
    old_limit = csv.field_size_limit(1024 * 1024)

    try:
        contract.check()
        root = _root(directory)
        expected_files = {"train_features.csv", "validation_features.csv", "train_rows.json", "validation_rows.json", "application.json", "application.md"}

        if {path.name for path in root.iterdir()} != expected_files:
            return None

        report = json.loads(_read_file(root / "application.json", 65536).decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid)
        expected_fields = {"schema_version", "status", "binding", "partitions", "transformed", "refit", "test_used", "model_trained"}

        if (type(report) is not dict or set(report) != expected_fields or report["schema_version"] != "preparation_application_v1"
                or report["status"] != "applied" or report["transformed"] is not True or report["refit"] is not False
                or report["test_used"] is not False or report["model_trained"] is not False
                or report["binding"] != json.loads(json.dumps(contract.packet(code_id)))
                or type(report["partitions"]) is not dict or set(report["partitions"]) != {"train", "validation"}):
            return None

        expected_columns = contract.candidate.output_columns
        state = read_metadata(contract.candidate.directory / "preprocessor.json")["state"]
        rules = contract.candidate.derivation.rules
        allow_missing = [record["kind"] == "numeric" and rules.numeric_imputation == "none" and record["edges"] is None
                         for record in state["records"] for _ in range(record["width"])]
        roles = contract.train.roles
        metadata_columns = list(dict.fromkeys(value for value in (roles.target, roles.group, roles.timestamp, *roles.identifiers) if value is not None))

        for name, part in (("train", contract.train), ("validation", contract.validation)):
            record = report["partitions"][name]
            count = len(part.row_positions)

            if type(record) is not dict or set(record) != {"rows", "features", "row_contract"} or type(record["rows"]) is not int or record["rows"] != count:
                return None

            features = root / f"{name}_features.csv"
            row_path = root / f"{name}_rows.json"

            if not _file_record(features, record["features"], contract.max_partition_bytes) or not _file_record(row_path, record["row_contract"], min(contract.max_partition_bytes, 8 * 1024 * 1024)):
                return None

            rows = read_metadata(row_path)

            if (set(rows) != {"row_positions", "metadata_columns", "metadata"} or rows["row_positions"] != list(part.row_positions)
                    or any(type(value) is not int for value in rows["row_positions"])
                    or rows["metadata_columns"] != metadata_columns or type(rows["metadata"]) is not list or len(rows["metadata"]) != count
                    or any(type(row) is not list or len(row) != len(metadata_columns) or any(type(value) is not str for value in row) for row in rows["metadata"])):
                return None

            with features.open(encoding="utf-8", newline="") as stream:
                reader = iter_csv_rows(stream, max_columns=len(expected_columns))

                if next(reader, None) != expected_columns:
                    return None

                observed = 0

                for row in reader:
                    observed += 1

                    if observed > count or len(row) != len(expected_columns):
                        return None

                    for value, permitted in zip(row, allow_missing):
                        if value == "" and permitted:
                            continue

                        if not value or not math.isfinite(float(value)):
                            return None

                if observed != count:
                    return None

            # 검사 중 파일 변경도 완료로 인정하지 않는다.
            if not _file_record(features, record["features"], contract.max_partition_bytes) or not _file_record(row_path, record["row_contract"], min(contract.max_partition_bytes, 8 * 1024 * 1024)):
                return None

        if _read_file(root / "application.md", 65536).decode("utf-8") != application_markdown():
            return None

        contract.check()
        return report
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, csv.Error, OverflowError, RecursionError):
        return None
    finally:
        csv.field_size_limit(old_limit)


def _pairs(items):
    result = {}

    for key, value in items:
        if key in result:
            raise ValueError("적용 보고서에 중복 JSON 키가 있습니다.")

        result[key] = value

    return result


def _invalid(value):
    raise ValueError("적용 보고서에 비표준 숫자를 사용할 수 없습니다.")

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
        # 후보·두 구간을 한 번 검사하고 같은 결과로 기대 binding과 Validation 계약을 얻는다. 끝에서 다시 검사한다.
        binding, validation = contract.verified_packet(code_id)
        root = _root(directory)
        expected_files = {"train_features.csv", "validation_features.csv", "train_rows.json", "validation_rows.json", "application.json", "application.md"}

        if {path.name for path in root.iterdir()} != expected_files:
            return None

        report = json.loads(_read_file(root / "application.json", 65536).decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid)
        expected_fields = {"schema_version", "status", "binding", "partitions", "transformed", "refit", "test_used", "model_trained"}
        # v2는 구간별 실제 적용 영향 합계를 더한다. v1(영향 집계 이전 적용)은 이전 코드로 실행한 기록일 때만 받는다.
        typed = contract.schema_version == "preparation_apply_v2"
        impact_version = "preparation_application_v3" if typed else "preparation_application_v2"
        versioned = type(report) is dict and report.get("schema_version") == impact_version

        if versioned:
            expected_fields = expected_fields | {"impact"}
        elif typed or code_id == _current_code_id():
            return None

        if (type(report) is not dict or set(report) != expected_fields
                or report["schema_version"] != (impact_version if versioned else "preparation_application_v1")
                or report["status"] != "applied" or report["transformed"] is not True or report["refit"] is not False
                or report["test_used"] is not False or report["model_trained"] is not False
                or report["binding"] != json.loads(json.dumps(binding))
                or type(report["partitions"]) is not dict or set(report["partitions"]) != {"train", "validation"}
                or (versioned and (type(report["impact"]) is not dict or set(report["impact"]) != {"train", "validation"}))):
            return None

        expected_columns = contract.candidate.output_columns
        state = read_metadata(contract.candidate.directory / "preprocessor.json")["state"]
        rules = contract.candidate.derivation.rules
        allow_missing = [record["kind"] == "numeric" and rules.numeric_imputation == "none" and record["edges"] is None
                         for record in state["records"] for _ in range(record["width"])]
        roles = contract.train.roles
        metadata_columns = list(dict.fromkeys(value for value in (roles.target, roles.group, roles.timestamp, *roles.identifiers) if value is not None))

        for name, part in (("train", contract.train), ("validation", validation)):
            record = report["partitions"][name]
            count = len(part.row_positions)

            if type(record) is not dict or set(record) != {"rows", "features", "row_contract"} or type(record["rows"]) is not int or record["rows"] != count:
                return None

            features = root / f"{name}_features.csv"
            row_path = root / f"{name}_rows.json"

            if not _file_record(features, record["features"], contract.max_partition_bytes) or not _file_record(row_path, record["row_contract"], min(contract.max_partition_bytes, 8 * 1024 * 1024)):
                return None

            rows = read_metadata(row_path)

            if typed:
                from .tabular_preparation_worker import row_metadata, load_row_metadata
                from src.preprocessing.split import _digest
                if _digest(rows) != _digest(row_metadata(part.bundle_directory, part.bundle, roles)):
                    return None
                load_row_metadata(row_path)
            elif (set(rows) != {"row_positions", "metadata_columns", "metadata"} or rows["row_positions"] != list(part.row_positions)
                    or any(type(value) is not int for value in rows["row_positions"])
                    or rows["metadata_columns"] != metadata_columns or type(rows["metadata"]) is not list or len(rows["metadata"]) != count
                    or any(type(row) is not list or len(row) != len(metadata_columns) or any(type(value) is not str for value in row) for row in rows["metadata"])):
                return None

            with features.open(encoding="utf-8", newline="") as stream:
                reader = iter_csv_rows(stream, max_columns=len(expected_columns))

                if next(reader, None) != expected_columns:
                    return None

                observed = 0
                tally = _ImpactTally(state["records"]) if versioned else None

                for row in reader:
                    observed += 1

                    if observed > count or len(row) != len(expected_columns):
                        return None

                    for value, permitted in zip(row, allow_missing):
                        if value == "" and permitted:
                            continue

                        if not value or not math.isfinite(float(value)):
                            return None

                    if tally is not None:
                        tally.add(row)

                if observed != count:
                    return None

            if tally is not None and not tally.matches(report["impact"][name], state["records"], count):
                return None

        # 같은 Train·같은 기준이므로 후보가 센 Train 영향 합계와 적용 Train 합계가 같아야 한다. Train에는 새 범주가 없다.
        if versioned and (report["impact"]["train"]["unknown_categories"]
                          or not _matches_candidate(contract.candidate.directory / "candidate.json", report["impact"]["train"])):
            return None

        # 검사 중 파일 변경도 완료로 인정하지 않는다.
        for name in ("train", "validation"):
            record = report["partitions"][name]
            features, row_path = root / f"{name}_features.csv", root / f"{name}_rows.json"
            if not _file_record(features, record["features"], contract.max_partition_bytes) or not _file_record(row_path, record["row_contract"], min(contract.max_partition_bytes, 8 * 1024 * 1024)):
                return None

        if _read_file(root / "application.md", 65536).decode("utf-8") != application_markdown(typed=typed):
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


def _current_code_id():
    from .preparation_apply_transport import code_manifest
    from .snapshot_eda import _digest

    return _digest(code_manifest())


def _matches_candidate(path, totals):
    """후보 v2의 Train 영향 합계와 대조한다. 후보가 v1(이전 코드)이면 대조할 값이 없어 통과한다."""
    from src.preprocessing.transform import impact_totals

    candidate = read_metadata(path)

    if candidate.get("schema_version") not in ("preparation_candidate_report_v2", "preparation_candidate_report_v3"):
        return candidate.get("schema_version") == "preparation_candidate_report_v1"

    return impact_totals(candidate["impact"]) == totals


class _ImpactTally:
    """이미 전수 읽는 적용 출력에서 worker가 기록한 영향 합계 중 출력만으로 정해지는 값을 센다.

    출력 열이 모두 한 값인 특징 수, 원핫 묶음이 모두 0인 행(Train에 없던 범주) 수, 빈 칸(결측) 수는 정확히 대조한다.
    clipping 변경 수는 출력만으로 다시 셀 수 없어 범위와 일관성만 확인한다.
    """

    def __init__(self, records):
        self.first = None
        self.varies = None
        self.blocks = []
        offset = 0

        for record in records:
            self.blocks.append((offset, record["width"], record["kind"]))
            offset += record["width"]

        self.unknown = 0
        self.missing = 0

    def add(self, row):
        self.missing += sum(1 for value in row if value == "")
        # worker와 같게 숫자로 비교한다(-0과 0은 같은 값). 빈 칸은 빈 칸끼리만 같다.
        values = [None if value == "" else float(value) for value in row]

        if self.first is None:
            self.first, self.varies = values, [False] * len(values)
        else:
            for index, value in enumerate(values):
                if not self.varies[index] and value != self.first[index]:
                    self.varies[index] = True

        for offset, width, kind in self.blocks:
            if kind == "categorical" and all(float(value) == 0 for value in row[offset:offset + width]):
                self.unknown += 1

    def matches(self, totals, records, count):
        fields = {"clipped_values", "clipped_features", "constant_features", "unknown_categories", "missing_values"}

        if type(totals) is not dict or set(totals) != fields or any(type(value) is not int or value < 0 for value in totals.values()):
            return False

        clip_features = sum(1 for record in records if record["kind"] == "numeric" and record.get("clip") is not None)
        constant = sum(1 for offset, width, _ in self.blocks
                       if self.varies is None or not any(self.varies[offset:offset + width]))

        return (totals["constant_features"] == constant and totals["unknown_categories"] == self.unknown
                and totals["missing_values"] == self.missing
                and totals["clipped_features"] <= clip_features and totals["clipped_values"] <= count * totals["clipped_features"]
                and (totals["clipped_values"] == 0) == (totals["clipped_features"] == 0))

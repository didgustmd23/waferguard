"""가상 충돌 제외 후보의 요청 결합·선택 파일·집계 구조를 부모에서 확인한다."""

import csv
from itertools import islice
import re
from pathlib import Path

from src.wafer_dl.validate_canonical import CLASS_NAMES, split_review_checks
from .artifacts import _is_link, _root
from .wm_canonical_contract import file_hash
from .wm_canonical_results import _read_small
from .worker_contract import _decode


POLICY_VERSION = "wm_conflict_candidate_v1"
FILES = frozenset({"conflict_hashes.jsonl", "excluded_rows.csv", "diagnosis/diagnosis.json",
                   "diagnosis/diagnosis.md", "diagnosis/class_comparison.csv"})


def candidate_files(directory, maximum):
    """파일 개수·깊이를 고정하고 링크·추가 항목·크기 초과를 거부한다."""
    root = Path(directory)

    if _is_link(root) or {p.name for p in root.iterdir()} not in (
        {"conflict_hashes.jsonl", "excluded_rows.csv", "diagnosis"},
        {"conflict_hashes.jsonl", "excluded_rows.csv", "diagnosis", "candidate.json"},
    ):
        raise ValueError("충돌 후보 결과 목록이 올바르지 않습니다.")

    diagnosis = root / "diagnosis"

    if _is_link(diagnosis) or not diagnosis.is_dir() or {p.name for p in diagnosis.iterdir()} != {
        "diagnosis.json", "diagnosis.md", "class_comparison.csv",
    }:
        raise ValueError("충돌 진단 파일 목록이 올바르지 않습니다.")

    names = set(FILES)

    if (root / "candidate.json").exists():
        names.add("candidate.json")

    total = 0
    files = {}

    for name in sorted(names):
        path = root / name

        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("충돌 후보에는 일반 결과 파일만 허용합니다.")

        record = file_hash(path, maximum - total)
        files[name] = record
        total += record["size"]

    return files


def inspect_candidate(directory, request, contract, evidence_id, evidence):
    """원본 배열을 읽거나 수치를 다시 계산하지 않고 출력 계약과 일관성만 검사한다."""
    root = _root(directory)
    files = candidate_files(root, contract.max_output_bytes)
    record = _decode(_read_small(root / "candidate.json", 65536).decode("utf-8"))

    if (set(record) != {"version", "request_id", "source_id", "config_id", "quality_evidence_id", "status", "files"}
        or record["version"] != POLICY_VERSION or record["request_id"] != request.request_id
        or record["source_id"] != request.source_id or record["config_id"] != request.config_id
        or record["quality_evidence_id"] != evidence_id or record["status"] != "candidate_ready"
        or set(files) != FILES | {"candidate.json"}
        or record["files"] != {name: files[name] for name in FILES}):
        raise ValueError("충돌 후보가 승인된 입력·품질 결과와 다릅니다.")

    report = _decode(_read_small(root / "diagnosis/diagnosis.json", 65536).decode("utf-8"))

    if (report.get("status") != "diagnostic_complete" or report.get("diagnosis_version") != "conflict_exclusion_v1"
        or any(report.get(key) is not False for key in ("data_modified", "split_complete", "training_ready",
                                                      "arrays_revalidated", "source_equivalence_verified"))):
        raise ValueError("가상 진단의 처리 범위가 올바르지 않습니다.")

    before, after = report.get("baseline"), report.get("hypothetical_exclusion")

    if type(before) is not dict or type(after) is not dict:
        raise ValueError("가상 진단에 전후 집계가 필요합니다.")

    # worker가 다른 기준 집계를 꾸며 넣지 못하도록 이전 승인 결과와 직접 대조한다.
    if (before.get("manifest_rows") != evidence["row_count"]
        or before.get("duplicate_audit") != evidence["duplicate_audit"]
        or before.get("split_assessment") != evidence["split_assessment"]):
        raise ValueError("가상 제외 전 집계가 결합된 품질 검사 결과와 다릅니다.")

    for scenario in (before, after):
        split_review_checks(scenario)

        if (scenario["split_assessment"]["max_group_ratio"] != contract.max_group_ratio
            or type(scenario.get("manifest_rows")) is not int
            or not 0 <= scenario["manifest_rows"] <= 2_000_000):
            raise ValueError("가상 진단의 행 수·검토 상한이 올바르지 않습니다.")

    group_count, excluded_count = report.get("candidate_conflict_hash_groups"), report.get("candidate_excluded_manifest_rows")

    if (type(group_count) is not int or not 1 <= group_count <= 1_000_000
        or type(excluded_count) is not int or not 1 <= excluded_count <= before["manifest_rows"]
        or before["manifest_rows"] - after["manifest_rows"] != excluded_count
        or before["duplicate_audit"]["map_label_conflicts"] != group_count
        or after["duplicate_audit"]["map_label_conflicts"] != 0):
        raise ValueError("충돌 후보 개수와 전후 진단 집계가 일치하지 않습니다.")

    previous = ""
    count = 0

    with (root / "conflict_hashes.jsonl").open("rb") as stream:
        while True:
            line = stream.readline(129)

            if not line:
                break

            value = _decode(line.decode("ascii"))
            hash_value = value.get("map_hash")

            if (len(line) > 128 or set(value) != {"map_hash"} or type(hash_value) is not str
                or re.fullmatch(r"[a-f0-9]{64}", hash_value) is None or hash_value <= previous):
                raise ValueError("충돌 해시 후보는 정렬된 중복 없는 SHA-256 목록이어야 합니다.")

            previous = hash_value
            count += 1

            if count > group_count:
                raise ValueError("충돌 해시 후보 개수가 진단과 다릅니다.")

    if count != group_count:
        raise ValueError("충돌 해시 후보가 누락되었습니다.")

    previous = -1
    count = 0

    with (root / "excluded_rows.csv").open("r", encoding="ascii", newline="") as stream:
        if stream.readline(128) != "source_row_position\n":
            raise ValueError("제외 후보 행 위치 헤더가 올바르지 않습니다.")

        for line in iter(lambda: stream.readline(17), ""):
            if len(line) > 16 or re.fullmatch(r"[0-9]+\n", line) is None:
                raise ValueError("제외 후보에는 정수 행 위치만 허용합니다.")

            position = int(line)

            if not previous < position < before["manifest_rows"]:
                raise ValueError("제외 후보 행 위치의 정렬·범위가 올바르지 않습니다.")

            previous = position
            count += 1

            if count > excluded_count:
                raise ValueError("제외 후보 행 수가 진단과 다릅니다.")

    if count != excluded_count:
        raise ValueError("제외 후보 행 위치가 누락되었습니다.")

    comparisons = report.get("class_comparison")

    if type(comparisons) is not list or len(comparisons) != 9:
        raise ValueError("9개 클래스의 제외 영향 비교가 필요합니다.")

    for index, row in enumerate(comparisons):
        if row.get("class_index") != index or row.get("pattern_label") != CLASS_NAMES[index]:
            raise ValueError("클래스 순서·정의가 기존 WM 계약과 다릅니다.")

        for key in ("samples_before", "samples_after", "excluded_labeled_samples", "lots_before", "lots_after",
                    "groups_before", "groups_after"):
            if type(row.get(key)) is not int or not 0 <= row[key] <= before["manifest_rows"]:
                raise ValueError("클래스별 집계는 제한된 음이 아닌 정수여야 합니다.")

        if (row["samples_before"] - row["samples_after"] != row["excluded_labeled_samples"]
            or row["minimum_groups_met_before"] is not (row["groups_before"] >= 3)
            or row["minimum_groups_met_after"] is not (row["groups_after"] >= 3)):
            raise ValueError("클래스별 제외 집계·독립 그룹 조건이 일치하지 않습니다.")

    with (root / "diagnosis/class_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        expected = [{key: str(value) for key, value in row.items()} for row in comparisons]

        if list(islice(reader, 10)) != expected:
            raise ValueError("클래스별 CSV와 진단 JSON이 일치하지 않습니다.")

    _read_small(root / "diagnosis/diagnosis.md", 65536).decode("utf-8")

    if candidate_files(root, contract.max_output_bytes) != files:
        raise ValueError("검증 중 충돌 후보가 변경되었습니다.")

    return {"status": "candidate_ready", "data_modified": False, "exclusion_approved": False,
            "quality_evidence_id": evidence_id, "files": files}

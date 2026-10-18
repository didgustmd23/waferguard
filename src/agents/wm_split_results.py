"""WM 그룹 분할안의 파일 계약·집계·그룹 배정 구조를 부모에서 확인한다."""

import csv
import math
from pathlib import Path

from .artifacts import _is_link, _root
from .wm_canonical_contract import file_hash
from .wm_canonical_results import _read_small
from .worker_contract import _decode


POLICY_VERSION = "wm_group_proposal_v1"
FILES = {"proposal.json", "groups.csv", "proposal.md"}
SPLITS = ("train", "validation", "test")


def proposal_files(directory, maximum):
    root = Path(directory)
    expected = FILES | ({"result.json"} if (root / "result.json").exists() else set())

    if _is_link(root) or {path.name for path in root.iterdir()} != expected:
        raise ValueError("분할안 파일 목록이 올바르지 않습니다.")

    records, total = {}, 0

    for name in sorted(expected):
        path = root / name

        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("분할안에는 일반 파일만 허용합니다.")

        records[name] = file_hash(path, maximum - total)
        total += records[name]["size"]

    return records


def inspect_proposal(directory, request, contract, evidence_id, evidence):
    """통계 기반 후보만 확인한다. 확정 분할·배열 재파싱·성능 평가는 하지 않는다."""
    root = _root(directory)
    files = proposal_files(root, contract.max_output_bytes)
    result = _decode(_read_small(root / "result.json", 65536).decode("utf-8"))

    if (set(result) != {"version", "request_id", "source_id", "config_id", "evidence_id", "files", "status"}
        or result["version"] != POLICY_VERSION or result["request_id"] != request.request_id
        or result["source_id"] != request.source_id or result["config_id"] != request.config_id
        or result["evidence_id"] != evidence_id or result["status"] != "proposal_ready"
        or result["files"] != {name: files[name] for name in FILES}):
        raise ValueError("분할안이 승인한 입력·EDA·설정과 다릅니다.")

    report = _decode(_read_small(root / "proposal.json", 65536).decode("utf-8"))
    config = evidence["split_config"]

    if (report.get("status") != "proposal_ready" or report.get("split_complete") is not False
        or report.get("training_ready") is not False or report.get("source_modified") is not False
        or report.get("arrays_read") is not False or report.get("overlap_check_passed") is not True
        or report.get("source_id") != request.source_id or report.get("config") != config
        or report.get("row_count") != evidence["quality"]["row_count"]
        or report.get("excluded_conflict_groups") != 0):
        raise ValueError("분할안의 처리 범위·설정이 올바르지 않습니다.")

    class_counts = report.get("class_counts")

    if (type(class_counts) is not dict or set(class_counts) != set(SPLITS)
        or any(type(values) is not list or len(values) != 9
               or any(type(value) is not int or value <= 0 for value in values) for values in class_counts.values())):
        raise ValueError("각 분할에 9개 클래스의 양의 표본 수가 필요합니다.")

    totals = [sum(class_counts[name][index] for name in SPLITS) for index in range(9)]
    all_samples = sum(totals)
    score = sum((sum(class_counts[name]) / all_samples - config["ratios"][index]) ** 2
                for index, name in enumerate(SPLITS))
    score += sum((class_counts[name][label] / totals[label] - config["ratios"][index]) ** 2
                 for label in range(9) for index, name in enumerate(SPLITS)) / 9

    if (all_samples > report["row_count"] or report.get("split_counts") != {name: sum(class_counts[name]) for name in SPLITS}
        or type(report.get("objective")) not in (int, float) or not math.isclose(score, report["objective"], abs_tol=1e-12)):
        raise ValueError("분할안의 표본 집계·목적함수가 일치하지 않습니다.")

    candidates = report.get("candidates")
    selected = report.get("selected_candidate")

    if (type(candidates) is not list or len(candidates) != config["max_candidates"]
        or type(selected) is not int or not 0 <= selected < len(candidates)):
        raise ValueError("고정 후보 수·선택 번호가 올바르지 않습니다.")

    for index, row in enumerate(candidates):
        if (type(row) is not dict or set(row) != {"candidate", "objective", "all_classes_present"}
            or row["candidate"] != index or type(row["all_classes_present"]) is not bool
            or type(row["objective"]) not in (int, float) or not math.isfinite(row["objective"])):
            raise ValueError("분할 후보 비교 기록이 올바르지 않습니다.")

    feasible = [row for row in candidates if row["all_classes_present"]]

    if (not feasible or min(feasible, key=lambda row: (row["objective"], row["candidate"]))["candidate"] != selected
        or not math.isclose(candidates[selected]["objective"], score, abs_tol=1e-12)):
        raise ValueError("선택한 분할안이 고정 후보 비교 기록과 다릅니다.")

    observed = {name: [0] * 9 for name in SPLITS}
    count = 0

    # 그룹 하나당 한 행만 허용한다. 그룹의 Lot·동일 맵 연결 검증은 고정 worker가 수행한다.
    with (root / "groups.csv").open(encoding="ascii", newline="") as stream:
        reader = csv.reader(stream, strict=True)

        if next(reader, None) != ["group_id", "split"] + [f"class_{index}" for index in range(9)]:
            raise ValueError("분할안 그룹 목록 헤더가 올바르지 않습니다.")

        for row in reader:
            if len(row) != 11 or row[0] != f"group_{count:08d}" or row[1] not in (*SPLITS, ""):
                raise ValueError("그룹은 중복 없는 연속 ID와 단일 배정을 가져야 합니다.")

            if any(not value.isascii() or not value.isdecimal() or len(value) > 7 for value in row[2:]):
                raise ValueError("그룹별 표본 수가 올바르지 않습니다.")

            values = list(map(int, row[2:]))

            if (sum(values) > report["row_count"] or (bool(sum(values)) != bool(row[1]))):
                raise ValueError("미라벨 전용 그룹을 지도학습 분할로 사용하지 않습니다.")

            if row[1]:
                observed[row[1]] = [old + value for old, value in zip(observed[row[1]], values)]

            count += 1

            if count > report["row_count"]:
                raise ValueError("그룹 수가 입력 행 수를 초과했습니다.")

    if observed != class_counts or count != report.get("group_count"):
        raise ValueError("그룹 배정과 분할안의 집계가 다릅니다.")

    _read_small(root / "proposal.md", 65536).decode("utf-8")

    if proposal_files(root, contract.max_output_bytes) != files:
        raise ValueError("검증 중 분할안이 변경되었습니다.")

    return {"status": "proposal_ready", "split_complete": False, "training_ready": False,
            "evidence_id": evidence_id, "files": files}

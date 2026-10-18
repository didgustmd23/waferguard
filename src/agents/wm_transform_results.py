"""Train·Validation 변환 검사 결과의 제한된 파일·집계·CSV·PNG 계약 검사."""

import csv
import math
from pathlib import Path
import struct
import zlib

from src.wafer_dl.validate_canonical import CLASS_NAMES
from .artifacts import _is_link, _root
from .wm_canonical_contract import file_hash
from .wm_canonical_results import _read_small
from .worker_contract import _decode


POLICY_VERSION = "wm_transform_check_v1"
FILES = {"preservation.csv", "examples.png", "transform_check.json", "transform_check.md", "SUCCESS.json"}
FIELDS = ["sample_id", "split", "class", "original_height", "original_width", "resized_height", "resized_width",
          "scale", "top", "left", "bottom", "right", "original_valid_dies", "original_failed_dies",
          "transformed_valid_pixels", "transformed_failed_pixels", "original_failed_ratio",
          "transformed_failed_ratio", "absolute_ratio_delta", "failed_region_lost", "valid_region_lost"]


def check_files(directory, maximum):
    root = Path(directory)
    expected = FILES | ({"result.json"} if (root / "result.json").exists() else set())

    if _is_link(root) or {path.name for path in root.iterdir()} != expected:
        raise ValueError("WM 변환 검사 파일 목록이 올바르지 않습니다.")

    records, total = {}, 0

    for name in sorted(expected):
        path = root / name

        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("변환 검사에는 일반 파일만 허용합니다.")

        records[name] = file_hash(path, maximum - total)
        total += records[name]["size"]

    return records


def valid_comparison_png(data):
    """그림을 복원하지 않고 고정 비교 그림의 PNG 청크·CRC·크기를 검사한다.

    기존 예시 함수는 최대 13행, 높이 4,680px를 생성한다. 일반 EDA의 4,096px 제한을
    전역 변경하지 않고 이 그림에만 2,048×5,120px·8MiB 판독 상한을 적용한다.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False

    offset, first, image_data = 8, True, False

    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + size + 12

        if end > len(data):
            return False

        kind, payload = data[offset + 4:offset + 8], data[offset + 8:end - 4]

        if zlib.crc32(kind + payload) & 0xFFFFFFFF != struct.unpack_from(">I", data, end - 4)[0]:
            return False

        if first:
            if kind != b"IHDR" or size != 13:
                return False

            width, height = struct.unpack_from(">II", payload)

            if not 1 <= width <= 2048 or not 1 <= height <= 5120:
                return False

            first = False

        image_data = image_data or kind == b"IDAT"

        if kind == b"IEND":
            return size == 0 and end == len(data) and image_data

        offset = end

    return False


def inspect_check(directory, request, contract, evidence_id, evidence):
    """원본 이미지 대신 구조·요청 결합·검사 범위·행별/클래스별 집계만 확인한다."""
    root = _root(directory)
    files = check_files(root, contract.max_output_bytes)
    result = _decode(_read_small(root / "result.json", 65536).decode("utf-8"))

    if (set(result) != {"version", "request_id", "source_id", "config_id", "evidence_id", "files", "status"}
        or result["version"] != POLICY_VERSION or result["request_id"] != request.request_id
        or result["source_id"] != request.source_id or result["config_id"] != request.config_id
        or result["evidence_id"] != evidence_id or result["status"] not in {"transform_check_passed", "transform_review_required"}
        or result["files"] != {name: files[name] for name in FILES}):
        raise ValueError("변환 검사 결과가 승인한 확정 분할·변환 설정과 다릅니다.")

    report = _decode(_read_small(root / "transform_check.json", 65536).decode("utf-8"))
    split = evidence["split_report"]
    expected = {(name, label): split["class_counts"][name][index]
                for name in ("train", "validation") for index, label in enumerate(CLASS_NAMES)}
    expected_rows = sum(expected.values())

    if (report.get("status") != result["status"] or report.get("config") != evidence["transform_config"]
        or report.get("split_protocol_id") != split["split_protocol_id"] or report.get("rows_checked") != expected_rows
        or report.get("test_maps_read") is not False or report.get("training_ready") is not False
        or report.get("visual_review_required") is not True or type(report.get("classes")) is not list
        or len(report["classes"]) != 18):
        raise ValueError("변환 검사의 대상·설정·시각 검토 요구가 올바르지 않습니다.")

    summaries = {}

    for item in report["classes"]:
        key = (item.get("split"), item.get("class"))

        if key in summaries or key not in expected or item.get("rows") != expected[key]:
            raise ValueError("변환 검사의 구간·클래스가 중복되거나 누락되었습니다.")

        for field in ("failed_region_lost", "valid_region_lost", "ratio_comparison_rows"):
            if type(item.get(field)) is not int or not 0 <= item[field] <= item["rows"]:
                raise ValueError("패턴 보존 집계의 정수 범위가 올바르지 않습니다.")

        for field in ("mean_absolute_ratio_delta", "max_absolute_ratio_delta"):
            value = item.get(field)

            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("패턴 비율 변화 집계가 올바르지 않습니다.")

        summaries[key] = item

    review = any(item["failed_region_lost"] or item["valid_region_lost"] for item in summaries.values())
    status = "transform_review_required" if review else "transform_check_passed"
    completion = _decode(_read_small(root / "SUCCESS.json", 65536).decode("utf-8"))

    if result["status"] != status or completion != {"status": "transform_check_complete", "review_required": review,
                                                   "training_ready": False}:
        raise ValueError("소실 집계와 검토 필요 판정이 다릅니다.")

    observed = {key: [0, 0, 0] for key in expected}
    sample_ids = set()

    with (root / "preservation.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)

        if reader.fieldnames != FIELDS:
            raise ValueError("행별 패턴 보존 CSV 헤더가 올바르지 않습니다.")

        for index, row in enumerate(reader, 1):
            key = (row.get("split"), row.get("class"))
            sample = row.get("sample_id")

            if (index > expected_rows or key not in expected or set(row) != set(FIELDS)
                or type(sample) is not str or not 1 <= len(sample) <= 256 or sample in sample_ids
                or row["failed_region_lost"] not in {"True", "False"} or row["valid_region_lost"] not in {"True", "False"}):
                raise ValueError("행별 변환 CSV의 구간·표본 중복·형식이 올바르지 않습니다.")

            sample_ids.add(sample)
            observed[key][0] += 1
            observed[key][1] += row["failed_region_lost"] == "True"
            observed[key][2] += row["valid_region_lost"] == "True"

    if any(observed[key] != [item["rows"], item["failed_region_lost"], item["valid_region_lost"]]
           for key, item in summaries.items()):
        raise ValueError("행별 CSV와 클래스별 소실 집계가 일치하지 않습니다.")

    if not valid_comparison_png(_read_small(root / "examples.png", 8 * 1024 * 1024)):
        raise ValueError("원본·변환 비교 PNG의 구조·크기가 올바르지 않습니다.")

    _read_small(root / "transform_check.md", 65536).decode("utf-8")

    if check_files(root, contract.max_output_bytes) != files:
        raise ValueError("검증 중 변환 검사 결과가 변경되었습니다.")

    return {"status": status, "test_maps_read": False, "training_ready": False,
            "visual_review_required": True, "evidence_id": evidence_id, "files": files}

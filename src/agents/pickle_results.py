"""pickle을 열지 않고 표·배열·WM 결과의 파일·출처·구조를 검증한다."""

import re

from src.wafer_dl.validate_canonical import validate_bundle
from .artifacts import _fingerprint, _is_link, _root
from .isolated_eda import _read_file
from .pickle_contract import PickleInput, digest
from src.preprocessing.canonical_data import decode_json, validate_data
from .worker_contract import _decode


def _bundle_files(root, maximum):
    """링크·추가 파일을 거부하고 실제 파일 크기·해시를 제한된 형식으로 수집한다."""
    expected = {"maps", "manifest.jsonl", "conversion.json", "provenance.json", "SUCCESS.json"}

    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("canonical 결과 파일 목록이 올바르지 않습니다.")

    maps = root / "maps"

    if _is_link(maps) or not maps.is_dir():
        raise ValueError("canonical 맵 폴더가 올바르지 않습니다.")

    paths = [root / name for name in expected - {"maps"}]

    for path in maps.iterdir():
        if re.fullmatch(r"shard_[0-9]{6,7}\.npz", path.name) is None:
            raise ValueError("canonical shard 파일명이 올바르지 않습니다.")

        paths.append(path)

        if len(paths) > 2_000_004:
            raise ValueError("canonical 파일 개수 상한을 초과했습니다.")

    total = 0
    hashes = {}

    for path in paths:
        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("canonical 결과에 일반 파일만 허용합니다.")

        total += path.stat().st_size

        if total > maximum:
            raise ValueError("canonical 결과가 승인된 출력 상한을 초과했습니다.")

        hashes[path.relative_to(root).as_posix()] = _fingerprint(path)["sha256"]

    return total, hashes


def _inspect_wm(directory, contract):
    """종료한 worker의 결과만 검사한다. 원본 동등성·무해성 인증은 하지 않는다."""
    if not isinstance(contract, PickleInput):
        raise TypeError("검증할 pickle 입력 계약이 필요합니다.")

    root = _root(directory)
    size, hashes = _bundle_files(root, contract.max_output_bytes)
    provenance = _decode(_read_file(root / "provenance.json", 65536).decode("utf-8"))
    _check_selection(provenance, contract)

    if provenance["converter_id"] != "wm811k_v1":
        raise ValueError("WM 결과에 다른 변환기 ID를 사용할 수 없습니다.")

    if (
        provenance.get("source_sha256") != contract.snapshot.sha256
        or provenance.get("source_bytes") != contract.snapshot.size_bytes
        or provenance.get("execution_mode") != "appcontainer"
    ):
        raise ValueError("canonical 결과의 출처가 승인된 pickle 복사본과 다릅니다.")

    # 관리 JSON·manifest의 한 행 크기를 제한한 뒤 기존 숫자 배열 검증기로 전달한다.
    for name in ("conversion.json", "SUCCESS.json"):
        _decode(_read_file(root / name, 65536).decode("utf-8"))

    with (root / "manifest.jsonl").open("rb") as stream:
        count = 0

        while True:
            line = stream.readline(65537)

            if not line:
                break

            count += 1

            if len(line) > 65536 or count > 2_000_000:
                raise ValueError("canonical manifest 행 크기 또는 개수 상한을 초과했습니다.")

    report = validate_bundle(root)

    if report.get("status") != "integrity_passed" or report.get("integrity_passed") is not True:
        raise ValueError("canonical 결과의 숫자 배열·manifest 정합성이 올바르지 않습니다.")

    if _bundle_files(root, contract.max_output_bytes) != (size, hashes):
        raise ValueError("검증 중 canonical 결과가 변경되었습니다.")

    return {
        "status": "integrity_passed", "bundle_id": digest(hashes), "output_bytes": size,
        "row_count": report["row_count"], "shard_count": report["shard_count"],
        "split_status": report["split_assessment"]["status"],
        "training_ready": False, "source_equivalence_verified": False,
    }


def _check_selection(source, contract, request_id=None):
    """승인된 변환기·경로와 결과의 출처 결합을 확인한다."""
    converter = source.get("converter_id")
    path = source.get("selected_path")

    if converter not in {"tabular_v1", "numeric_array_v1", "wm811k_v1"}:
        raise ValueError("등록되지 않은 변환기입니다.")

    if contract.converter_id != "auto" and converter != contract.converter_id:
        raise ValueError("승인된 변환기와 다릅니다.")

    if (type(path) is not list or len(path) > 4
        or any(not (type(key) is str and 0 < len(key) <= 256
                   or type(key) is int and 0 <= key <= 2_000_000) for key in path)
        or path[:len(contract.select_path)] != list(contract.select_path)):
        raise ValueError("승인된 선택 경로와 다릅니다.")

    if (source.get("source_sha256") != contract.snapshot.sha256
        or source.get("source_bytes") != contract.snapshot.size_bytes
        or source.get("execution_mode") != "appcontainer"
        or source.get("selection_scope") != "selected_object"
        or source.get("unselected_original") != "input_snapshot"
        or request_id is not None and source.get("request_id") != request_id):
        raise ValueError("변환 결과의 출처가 승인 범위와 다릅니다.")


def _generic_files(root, maximum):
    """추가 파일·링크를 거부하고 전체 출력 크기와 지문을 수집한다."""
    paths = list(root.iterdir())
    names = {path.name for path in paths}
    common = {"schema.json", "source.json", "manifest.json", "SUCCESS.json"}

    if names not in (common | {"data.jsonl"}, common | {"array.npy"}):
        raise ValueError("범용 변환 결과 파일 목록이 올바르지 않습니다.")

    total = 0
    fingerprints = {}

    for path in paths:
        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("일반 결과 파일만 허용합니다.")

        total += path.stat().st_size

        if total > maximum:
            raise ValueError("승인된 출력 크기를 초과했습니다.")

        fingerprints[path.name] = _fingerprint(path)

    return total, fingerprints


def inspect_canonical(directory, contract, request_id=None):
    """부모에서는 pickle을 읽지 않고 제한된 JSON·숫자 결과만 검증한다."""
    if not isinstance(contract, PickleInput):
        raise TypeError("pickle 입력 계약이 필요합니다.")

    root = _root(directory)

    if not (root / "manifest.json").exists():
        report = _inspect_wm(root, contract)
        source = _decode(_read_file(root / "provenance.json", 65536).decode("utf-8"))
        _check_selection(source, contract, request_id)
        return {**report, "converter_id": source["converter_id"], "selected_path": source["selected_path"]}

    size, fingerprints = _generic_files(root, contract.max_output_bytes)
    read = lambda name: decode_json(_read_file(root / name, 1024 * 1024))
    source = read("source.json")
    manifest = read("manifest.json")
    _check_selection(source, contract, request_id)
    converter = source["converter_id"]
    expected = {"schema.json", "source.json", "data.jsonl" if converter == "tabular_v1" else "array.npy"}

    if (type(manifest) is not dict or set(manifest) != {"version", "converter_id", "files"}
        or manifest["version"] != "canonical_bundle_v1"
        or manifest["converter_id"] != converter
        or manifest["files"] != {name: fingerprints[name] for name in expected}
        or read("SUCCESS.json") != {"status": "canonical_export_complete", "training_ready": False}):
        raise ValueError("변환 manifest 또는 완료 기록이 올바르지 않습니다.")

    detail = validate_data(root)

    if ((converter == "tabular_v1") != (detail["kind"] == "tabular")
        or converter == "wm811k_v1"
        or _generic_files(root, contract.max_output_bytes) != (size, fingerprints)):
        raise ValueError("변환 결과 형식 또는 검증 중 파일 변경이 발견되었습니다.")

    return {"status": "integrity_passed", "bundle_id": digest(fingerprints), "output_bytes": size,
            "converter_id": converter, "selected_path": source["selected_path"], **detail,
            "training_ready": False, "source_equivalence_verified": False}


def read_structure_review(directory, request, maximum=1024 * 1024):
    """worker의 검토 결과도 요청 결합과 제한된 구조 계약으로 검증한다."""
    from .pickle_converters import CONVERTER_IDS, REVIEW_CODES, ROOT_KINDS, REVIEW_MARKDOWN

    root = _root(directory)

    if {path.name for path in root.iterdir()} != {"structure.json", "structure.md"}:
        raise ValueError("구조 검토 파일 목록이 올바르지 않습니다.")

    total = 0

    for path in root.iterdir():
        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("구조 검토에는 일반 파일만 허용합니다.")

        total += path.stat().st_size

        if total > maximum:
            raise ValueError("구조 검토 보고서가 승인된 출력 상한을 초과했습니다.")

    report = decode_json(_read_file(root / "structure.json", 1024 * 1024))

    if (type(report) is not dict
        or set(report) != {"version", "status", "root_kind", "reason", "candidates", "selected_path", "converter_id", "request_id", "source_id"}
        or report["version"] != "pickle_structure_v1" or report["status"] != "review_required"
        or report["root_kind"] not in ROOT_KINDS or report["reason"] not in REVIEW_CODES
        or report["request_id"] != request.request_id or report["source_id"] != request.source_id
        or report["selected_path"] is not None or report["converter_id"] is not None
        or type(report["candidates"]) is not list or len(report["candidates"]) > 64
        or _read_file(root / "structure.md", 65536).decode("utf-8") != REVIEW_MARKDOWN):
        raise ValueError("구조 검토 보고서가 현재 요청과 일치하지 않습니다.")

    for item in report["candidates"]:
        if type(item) is not dict or item.get("converter_id") not in CONVERTER_IDS:
            raise ValueError("구조 후보가 올바르지 않습니다.")

        path = item.get("path")

        if (type(path) is not list or len(path) > 4
            or any(not (type(key) is str and 0 < len(key) <= 256
                       or type(key) is int and 0 <= key <= 2_000_000) for key in path)):
            raise ValueError("구조 후보 경로가 올바르지 않습니다.")

        if item.get("kind") == "dataframe":
            valid = (set(item) == {"path", "converter_id", "kind", "rows", "columns"}
                     and type(item["rows"]) is int and 0 <= item["rows"] <= 2_000_000
                     and type(item["columns"]) is int and 0 <= item["columns"] <= 4096)
        else:
            shape = item.get("shape")
            valid = (set(item) == {"path", "converter_id", "kind", "shape", "dtype"}
                     and item.get("kind") == "ndarray" and type(shape) is list and len(shape) <= 8
                     and all(type(size) is int and 0 <= size <= 100_000_000 for size in shape)
                     and type(item.get("dtype")) is str and len(item["dtype"]) <= 32)

        if not valid:
            raise ValueError("구조 후보의 크기 또는 형식이 올바르지 않습니다.")

    return report

"""새 파생 canonical의 요청·출력 지문·검증 보고서 계약을 제한적으로 확인한다."""

from pathlib import Path

from src.preprocessing.canonical_data import decode_json
from src.wafer_dl.validate_canonical import split_review_checks, validate_metadata
from .artifacts import _is_link, _root
from .wm_canonical_contract import digest, file_hash, inventory
from .wm_canonical_results import _read_small
from .worker_contract import _decode


POLICY_VERSION = "wm_conflict_exclusion_v1"
MAX_RESULT_RECORD_BYTES = 1024 * 1024


def read_result_record(path):
    """파일 목록·영수증은 제한된 1MiB 계약으로 읽는다. IPC의 64KiB 상한은 유지한다.

    최대 4,096개 canonical 파일의 지문 목록은 실제 데이터에서 64KiB를 넘는다.
    중복 키·비표준 숫자·객체 타입은 그대로 거부하며 무제한 JSON 읽기는 하지 않는다.
    """
    record = decode_json(_read_small(path, MAX_RESULT_RECORD_BYTES))

    if type(record) is not dict:
        raise ValueError("파생 결과 관리 기록은 JSON 객체여야 합니다.")

    return record


def derived_files(directory, maximum):
    """정해진 canonical·검증 JSON 외의 추가 파일과 링크는 받지 않는다."""
    root = Path(directory)
    expected = {"canonical", "validation.json"}

    if (root / "result.json").exists():
        expected.add("result.json")

    if _is_link(root) or {path.name for path in root.iterdir()} != expected:
        raise ValueError("파생 canonical 결과 목록이 올바르지 않습니다.")

    records = {"canonical/" + name: value for name, value in inventory(root / "canonical", maximum, staged=True).items()}
    total = sum(value["size"] for value in records.values())

    for name in sorted(expected - {"canonical"}):
        path = root / name

        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("파생 결과에는 일반 파일만 허용합니다.")

        records[name] = file_hash(path, maximum - total)
        total += records[name]["size"]

    return records


def inspect_derived(directory, request, contract, evidence_id, evidence):
    """배열 파싱 없이 요청 결합·파일 지문·검증 집계·출처 연결을 확인한다."""
    root = _root(directory)
    files = derived_files(root, contract.max_output_bytes)
    result = read_result_record(root / "result.json")

    if (set(result) != {"version", "request_id", "source_id", "config_id", "evidence_id", "files", "status"}
        or result["version"] != POLICY_VERSION or result["request_id"] != request.request_id
        or result["source_id"] != request.source_id or result["config_id"] != request.config_id
        or result["evidence_id"] != evidence_id or result["status"] != "derived_ready"
        or result["files"] != {name: value for name, value in files.items() if name != "result.json"}):
        raise ValueError("파생 canonical이 승인한 후보·요청과 다릅니다.")

    report = _decode(_read_small(root / "validation.json", 65536).decode("utf-8"))
    expected = evidence["hypothetical"]

    if (report.get("status") != "integrity_passed" or report.get("integrity_passed") is not True
        or report.get("training_ready") is not False or report.get("source_equivalence_verified") is not False
        or report.get("is_security_certification") is not False
        or report.get("row_count") != expected["manifest_rows"]
        or report.get("duplicate_audit") != expected["duplicate_audit"]
        or report.get("split_assessment") != expected["split_assessment"]
        or not all(passed for _, passed, _ in split_review_checks(report))):
        raise ValueError("파생 결과의 재검증 집계가 검토한 가상 제외 결과와 다릅니다.")

    summary, _version = validate_metadata(root / "canonical")
    provenance = _decode(_read_small(root / "canonical/provenance.json", 65536).decode("utf-8"))

    if (summary["row_count"] != report["row_count"] or summary["shard_count"] != report["shard_count"]
        or provenance.get("parent_source_id") != request.source_id
        or provenance.get("candidate_evidence_id") != evidence["candidate_evidence_id"]
        or provenance.get("operation") != "exclude_all_conflicting_map_rows_v1"
        or provenance.get("source_kind") != "parent_canonical_manifest"
        or provenance.get("source_sha256") != contract.files["manifest.jsonl"]["sha256"]
        or provenance.get("source_bytes") != contract.files["manifest.jsonl"]["size"]):
        raise ValueError("파생 canonical의 출처·개수가 재검증 결과와 다릅니다.")

    if derived_files(root, contract.max_output_bytes) != files:
        raise ValueError("검증 중 파생 canonical이 변경되었습니다.")

    return {"status": "derived_ready", "original_modified": False, "training_ready": False,
            "split_status": report["split_assessment"]["status"], "evidence_id": evidence_id, "files": files}

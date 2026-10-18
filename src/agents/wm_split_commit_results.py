"""WM 분할 확정 산출물의 제한된 파일·출처·고정 배정 검증 계약."""

import re
from pathlib import Path

from .artifacts import _is_link, _root
from .wm_canonical_contract import file_hash
from .wm_canonical_results import _read_small
from .worker_contract import _decode


POLICY_VERSION = "wm_group_commit_v1"
FILES = {"split_manifest.jsonl", "split.json", "split_summary.csv", "split_summary.md", "SUCCESS.json", "validation.json"}


def split_files(directory, maximum):
    root = Path(directory)
    expected = FILES | ({"result.json"} if (root / "result.json").exists() else set())

    if _is_link(root) or {path.name for path in root.iterdir()} != expected:
        raise ValueError("WM 확정 분할 파일 목록이 올바르지 않습니다.")

    records, total = {}, 0

    for name in sorted(expected):
        path = root / name

        if _is_link(path) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("확정 분할에는 일반 파일만 허용합니다.")

        records[name] = file_hash(path, maximum - total)
        total += records[name]["size"]

    return records


def inspect_split(directory, request, contract, evidence_id, evidence):
    """큰 manifest는 해싱하고 작은 보고서만 파싱한다. 원본 분석은 worker에서 한다."""
    root = _root(directory)
    files = split_files(root, contract.max_output_bytes)
    result = _decode(_read_small(root / "result.json", 65536).decode("utf-8"))

    if (set(result) != {"version", "request_id", "source_id", "config_id", "evidence_id", "files", "status"}
        or result["version"] != POLICY_VERSION or result["request_id"] != request.request_id
        or result["source_id"] != request.source_id or result["config_id"] != request.config_id
        or result["evidence_id"] != evidence_id or result["status"] != "split_complete"
        or result["files"] != {name: files[name] for name in FILES}):
        raise ValueError("WM 확정 분할이 승인한 제안과 다릅니다.")

    report = _decode(_read_small(root / "split.json", 65536).decode("utf-8"))
    proposal = evidence["proposal"]
    source_hash = contract.files["manifest.jsonl"]["sha256"]
    # hashlib의 내부 상태를 파일 해시 문자열에서 복원할 수 없으므로 프로토콜은 형식·완료 표식과 결합한다.
    # 실제 공식 및 행 대응은 worker의 기존 index_split이 원본 manifest로 재검증한다.
    protocol = report.get("split_protocol_id")

    if (type(protocol) is not str or re.fullmatch(r"wm_split_[a-f0-9]{64}", protocol) is None
        or report.get("status") != "split_complete" or report.get("split_complete") is not True
        or report.get("training_ready") is not False or report.get("source_modified") is not False
        or report.get("arrays_revalidated") is not False or report.get("resampled") is not False
        or report.get("overlap_check_passed") is not True or report.get("excluded_rows") != 0
        or report.get("excluded_conflict_groups") != 0 or report.get("source_manifest_sha256") != source_hash
        or report.get("canonical_root") != contract.snapshot_directory
        or report.get("array_validation_source") != "completed_eda_same_inventory"):
        raise ValueError("WM 확정 분할의 처리 범위·출처·검증 상태가 올바르지 않습니다.")

    for key in ("row_count", "split_counts", "class_counts", "config", "selected_candidate", "objective", "candidates"):
        if report.get(key) != proposal[key]:
            raise ValueError("WM 확정 분할이 검토한 고정 배정의 집계와 다릅니다.")

    success = _decode(_read_small(root / "SUCCESS.json", 65536).decode("utf-8"))
    verification = _decode(_read_small(root / "validation.json", 65536).decode("utf-8"))

    if (success != {"status": "split_complete", "split_protocol_id": protocol, "training_ready": False}
        or verification != {"status": "split_verified", "source_id": request.source_id,
                             "proposal_evidence_id": evidence_id, "split_protocol_id": protocol,
                             "row_count": report["row_count"], "row_linkage_verified": True,
                             "overlap_check_passed": True, "arrays_read": False, "resampled": False}):
        raise ValueError("WM 분할 완료 표식·재검증 보고서가 일치하지 않습니다.")

    _read_small(root / "split_summary.csv", 65536).decode("utf-8-sig")
    _read_small(root / "split_summary.md", 65536).decode("utf-8")

    if split_files(root, contract.max_output_bytes) != files:
        raise ValueError("검증 중 WM 확정 분할이 변경되었습니다.")

    return {"status": "split_complete", "split_complete": True, "training_ready": False,
            "split_protocol_id": protocol, "evidence_id": evidence_id, "files": files}

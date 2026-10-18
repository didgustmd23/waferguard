"""검토된 WM 그룹 배정만 적용하고 원본 행 대응을 재검증하는 격리 worker."""

from pathlib import Path
import sys


def execute_request(request, root):
    from dataclasses import asdict
    import json

    from src.agents.pickle_import_worker import _diagnostic
    from src.agents.wm_canonical_contract import WmCanonicalInput, digest, inventory
    from src.agents.wm_split_results import proposal_files
    from src.agents.wm_split_commit_results import POLICY_VERSION, split_files
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json
    from src.wafer_dl.commit_split import commit_reviewed_split

    root = Path(root).absolute()
    source, proposal_dir = root / "input/canonical", root / "input/proposal"
    settings, manifest = root / "input/analysis.json", root / "input/files.json"
    stage = "REQUEST"

    try:
        if (request.task_id != root.name or request.stage != "commit_split" or request.action_id != "commit_split"
            or request.policy_version != POLICY_VERSION or request.input_files != (str(settings), str(manifest))
            or request.dependency_directories != (str(source), str(proposal_dir))
            or request.output_directory != str(root / "output/result")
            or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("WM 분할 확정 요청이 올바르지 않습니다.")

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if (set(packet) != {"version", "source_id", "limits", "max_input_bytes", "max_output_bytes",
                            "include_examples", "max_group_ratio", "code_id", "commit_evidence"}
            or packet["version"] != POLICY_VERSION or packet["source_id"] != request.source_id
            or packet["limits"] != asdict(request.limits) or digest(packet) != request.config_id
            or packet["include_examples"] is not False):
            raise ValueError("WM 분할 확정 설정이 승인 범위와 다릅니다.")

        with manifest.open("rb") as stream:
            files = decode_json(stream.read(1024 * 1024 + 1))

        contract = WmCanonicalInput(str(source), json.dumps(files), request.limits,
                                    packet["max_input_bytes"], packet["max_output_bytes"], False, packet["max_group_ratio"])
        evidence = packet["commit_evidence"]

        if (contract.source_id != request.source_id or evidence["source_id"] != request.source_id
            or evidence["evidence_id"] != digest({key: value for key, value in evidence.items() if key != "evidence_id"})
            or inventory(source, contract.max_input_bytes, staged=True) != files
            or proposal_files(proposal_dir, contract.max_output_bytes) != evidence["proposal_files"]):
            raise ValueError("확정할 원본·분할안 지문이 검토한 증거와 다릅니다.")

        with (proposal_dir / "proposal.json").open("rb") as stream:
            proposal = _decode(stream.read(65537).decode("utf-8"))

        if proposal != evidence["proposal"] or proposal["source_id"] != request.source_id:
            raise ValueError("확정할 분할안의 내용이 검토한 제안과 다릅니다.")

        stage = "COMMIT_SPLIT"
        output = Path(request.output_directory) / "split"
        report = commit_reviewed_split(source, output, root / "scratch/task", proposal, proposal_dir / "groups.csv",
                                       evidence["canonical_root"], contract.max_output_bytes - 131072)
        verification = {"status": "split_verified", "source_id": request.source_id,
                        "proposal_evidence_id": digest(evidence), "split_protocol_id": report["split_protocol_id"],
                        "row_count": report["row_count"], "row_linkage_verified": True,
                        "overlap_check_passed": True, "arrays_read": False, "resampled": False}
        (output / "validation.json").write_text(json.dumps(verification, allow_nan=False), encoding="utf-8")

        if (inventory(source, contract.max_input_bytes, staged=True) != files
            or proposal_files(proposal_dir, contract.max_output_bytes) != evidence["proposal_files"]):
            raise ValueError("분할 확정 중 입력·제안 복사본이 변경되었습니다.")

        records = split_files(output, contract.max_output_bytes)
        payload = json.dumps({"version": POLICY_VERSION, "request_id": request.request_id,
                              "source_id": request.source_id, "config_id": request.config_id,
                              "evidence_id": digest(evidence), "status": "split_complete", "files": records},
                             allow_nan=False).encode("utf-8")

        if len(payload) + sum(value["size"] for value in records.values()) > contract.max_output_bytes:
            raise ValueError("확정 분할 결과 크기가 출력 상한을 초과했습니다.")

        (output / "result.json").write_bytes(payload)
        _diagnostic(root, "COMPLETE", "PASSED")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("split",))
    except Exception:
        _diagnostic(root, stage, "WM_SPLIT_COMMIT_FAILED")
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")


def main():
    """기존 AppContainer 토큰·Job 검사를 통과한 뒤에만 요청을 읽는다."""
    import os

    if len(sys.argv) != 3:
        return 2

    root = Path(sys.argv[1]).absolute()
    sys.path.insert(0, str(root / "tools"))
    from src.agents.pickle_import_worker import _isolated_process

    if not _isolated_process():
        return 2

    runtime = root / "runtime"
    sys.path.insert(1, str(runtime / "Lib/site-packages"))
    handles = [os.add_dll_directory(str(path)) for path in
               (runtime, runtime / "DLLs", runtime / "Library/bin") if path.is_dir()]
    from src.agents.worker_contract import WorkerRequest

    try:
        with (root / "input/request.json").open("rb") as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))

        if request.request_id != sys.argv[2]:
            return 2

        reply = execute_request(request, root)
        (root / "output/result/worker_result.json").write_text(reply.to_json(), encoding="utf-8")
        return 0 if reply.status == "succeeded" else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

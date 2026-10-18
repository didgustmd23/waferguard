"""승인된 충돌 제외 후보를 새 canonical에 적용하고 전체 재검증하는 격리 worker."""

from pathlib import Path
import sys


def execute_request(request, root):
    """원본은 읽기 전용으로 유지하고 허용된 출력 폴더에만 파생 결과를 만든다."""
    from dataclasses import asdict
    import json

    from src.agents.pickle_import_worker import _diagnostic
    from src.agents.wm_canonical_contract import WmCanonicalInput, digest, inventory
    from src.agents.wm_exclusion_results import POLICY_VERSION, derived_files
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json
    from src.wafer_dl.exclude_conflicts import derive_canonical
    from src.wafer_dl.validate_canonical import validate_bundle

    root = Path(root).absolute()
    source = root / "input/canonical"
    settings, manifest = root / "input/analysis.json", root / "input/files.json"
    stage = "REQUEST"

    try:
        if (request.task_id != root.name or request.stage != "derive_preparation"
            or request.action_id != "derive_preparation" or request.policy_version != POLICY_VERSION
            or request.input_files != (str(settings), str(manifest)) or request.dependency_directories != (str(source),)
            or request.output_directory != str(root / "output/result")
            or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("충돌 제외 worker 요청이 올바르지 않습니다.")

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if (set(packet) != {"version", "source_id", "limits", "max_input_bytes", "max_output_bytes",
                            "include_examples", "max_group_ratio", "code_id", "candidate_evidence"}
            or packet["version"] != POLICY_VERSION or packet["source_id"] != request.source_id
            or packet["limits"] != asdict(request.limits) or digest(packet) != request.config_id
            or packet["include_examples"] is not False):
            raise ValueError("충돌 제외 설정이 승인 범위와 다릅니다.")

        with manifest.open("rb") as stream:
            files = decode_json(stream.read(1024 * 1024 + 1))

        contract = WmCanonicalInput(str(source), json.dumps(files), request.limits,
                                    packet["max_input_bytes"], packet["max_output_bytes"], False, packet["max_group_ratio"])
        evidence = packet["candidate_evidence"]

        if (contract.source_id != request.source_id or evidence["source_id"] != request.source_id
            or evidence["candidate_evidence_id"] != digest({key: value for key, value in evidence.items()
                                                           if key != "candidate_evidence_id"})
            or evidence["hypothetical"]["split_assessment"]["status"] != "preconditions_passed"
            or inventory(source, contract.max_input_bytes, staged=True) != files):
            raise ValueError("충돌 제외 입력·후보 증거가 다릅니다.")

        stage = "DERIVE"
        derived = Path(request.output_directory) / "derived"
        derived.mkdir(exist_ok=False)
        derive_canonical(source, derived / "canonical", root / "scratch/task", evidence, contract.max_output_bytes)
        stage = "VALIDATION"
        workspace = root / "scratch/task/validation"
        workspace.mkdir()
        report = validate_bundle(derived / "canonical", max_group_ratio=contract.max_group_ratio, workspace=workspace)
        expected = evidence["hypothetical"]

        if (report["row_count"] != expected["manifest_rows"] or report["duplicate_audit"] != expected["duplicate_audit"]
            or report["split_assessment"] != expected["split_assessment"]):
            raise ValueError("파생 결과 재검증이 승인한 가상 제외 결과와 다릅니다.")

        (derived / "validation.json").write_text(json.dumps(report, allow_nan=False), encoding="utf-8")

        if inventory(source, contract.max_input_bytes, staged=True) != files:
            raise ValueError("충돌 제외 중 원본 복사본이 변경되었습니다.")

        records = derived_files(derived, contract.max_output_bytes)
        payload = json.dumps({"version": POLICY_VERSION, "request_id": request.request_id,
                              "source_id": request.source_id, "config_id": request.config_id,
                              "evidence_id": digest(evidence), "status": "derived_ready", "files": records},
                             allow_nan=False).encode("utf-8")

        if len(payload) + sum(record["size"] for record in records.values()) > contract.max_output_bytes:
            raise ValueError("파생 canonical 관리 기록이 출력 상한을 초과했습니다.")

        (derived / "result.json").write_bytes(payload)
        _diagnostic(root, "COMPLETE", "PASSED")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("derived",))
    except Exception:
        # 상세 경로·맵·라벨·원문 예외는 worker 응답으로 노출하지 않는다.
        _diagnostic(root, stage, "EXCLUSION_FAILED")
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")


def main():
    """일반 프로세스에서 직접 호출한 경우 입력을 읽지 않고 거부한다."""
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

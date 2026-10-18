"""확정 Train·Validation의 숫자 전용 캐시만 생성하는 격리 worker."""

from pathlib import Path
import sys


def execute_request(request, root):
    from dataclasses import asdict
    import json

    from src.agents.pickle_import_worker import _diagnostic
    from src.agents.wm_canonical_contract import WmCanonicalInput, digest, inventory
    from src.agents.wm_split_commit_results import split_files
    from src.agents.wm_cache_results import POLICY_VERSION, cache_files
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json
    from src.wafer_dl.isolated_cache import build_isolated_cache
    from src.agents.wm_transform_results import check_files

    root = Path(root).absolute()
    source, split_dir = root / "input/canonical", root / "input/split"
    check_dir = root / "input/check"
    settings, manifest = root / "input/analysis.json", root / "input/files.json"
    stage = "REQUEST"

    try:
        if (request.task_id != root.name or request.stage != "prepare_data"
            or request.action_id != "prepare_data" or request.policy_version != POLICY_VERSION
            or request.input_files != (str(settings), str(manifest))
            or request.dependency_directories != (str(source), str(split_dir), str(check_dir))
            or request.output_directory != str(root / "output/result")
            or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("WM 캐시 생성 요청이 올바르지 않습니다.")

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if (set(packet) != {"version", "source_id", "limits", "max_input_bytes", "max_output_bytes",
                            "include_examples", "max_group_ratio", "code_id", "cache_evidence"}
            or packet["version"] != POLICY_VERSION or packet["source_id"] != request.source_id
            or packet["limits"] != asdict(request.limits) or digest(packet) != request.config_id
            or packet["include_examples"] is not False):
            raise ValueError("WM 캐시 생성 설정이 승인 범위와 다릅니다.")

        with manifest.open("rb") as stream:
            files = decode_json(stream.read(1024 * 1024 + 1))

        contract = WmCanonicalInput(str(source), json.dumps(files), request.limits,
                                    packet["max_input_bytes"], packet["max_output_bytes"], False, packet["max_group_ratio"])
        evidence = packet["cache_evidence"]

        split = evidence["split_evidence"]

        # 분할 manifest는 캐시 출력보다 클 수 있다. 입력 증거의 정확한 크기 합계로 제한한다.
        split_budget = sum(value["size"] for value in split["split_files"].values())

        if (contract.source_id != request.source_id or evidence["source_id"] != request.source_id
            or evidence["evidence_id"] != digest({key: value for key, value in evidence.items() if key != "evidence_id"})
            or inventory(source, contract.max_input_bytes, staged=True) != files
            or split_files(split_dir, split_budget) != split["split_files"]
            or check_files(check_dir, sum(item["size"] for item in evidence["check_files"].values())) != evidence["check_files"]):
            raise ValueError("변환 검사 입력·확정 분할 증거가 다릅니다.")

        with (split_dir / "split.json").open("rb") as stream:
            split_report = _decode(stream.read(65537).decode("utf-8"))

        if split_report != split["split_report"]:
            raise ValueError("변환할 분할 보고서가 승인한 결과와 다릅니다.")

        workspace = root / "scratch/task/validation"
        workspace.mkdir()
        stage = "BUILD_CACHE"
        output = Path(request.output_directory) / "cache"

        def mark_stage(value):
            # 알려진 처리 단계만 기록하며 입력 값·예외 원문은 기록하지 않는다.
            nonlocal stage
            stage = value
            _diagnostic(root, stage, "START")

        result = build_isolated_cache(
            split_dir, check_dir, output, source, split["transform_config"], evidence["cache_config"],
            workspace=workspace, max_output_bytes=contract.max_output_bytes - 1024 * 1024,
            diagnostic=mark_stage,
        )

        if (inventory(source, contract.max_input_bytes, staged=True) != files
            or split_files(split_dir, split_budget) != split["split_files"]):
            raise ValueError("변환 검사 중 입력·확정 분할 복사본이 변경되었습니다.")

        records = cache_files(output, contract.max_output_bytes)
        payload = json.dumps({"version": POLICY_VERSION, "request_id": request.request_id,
                              "source_id": request.source_id, "config_id": request.config_id,
                              "evidence_id": digest(evidence), "status": result["status"], "files": records},
                             allow_nan=False).encode("utf-8")

        if len(payload) + sum(value["size"] for value in records.values()) > contract.max_output_bytes:
            raise ValueError("변환 검사 결과 크기가 출력 상한을 초과했습니다.")

        (output / "result.json").write_bytes(payload)
        _diagnostic(root, "COMPLETE", "PASSED")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("cache",))
    except Exception as error:
        code = "CACHE_ACCESS_DENIED" if isinstance(error, PermissionError) else "WM_CACHE_BUILD_FAILED"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")


def main():
    """AppContainer 토큰·Job 검사를 통과한 프로세스만 입력을 읽는다."""
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

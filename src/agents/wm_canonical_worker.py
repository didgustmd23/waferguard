"""승인된 canonical 복사본의 기존 WM 검증·EDA만 실행하는 격리 진입점."""

from pathlib import Path
import sys


def execute_request(request, root):
    """파일 정합성·분할 사전조건을 검사하고 통과한 입력만 비파괴 EDA에 전달한다."""
    import json
    from dataclasses import asdict

    from src.agents.wm_canonical_contract import POLICY_VERSION, check_files, digest, inventory
    from src.agents.wm_canonical_results import result_files
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json
    from src.wafer_dl.validate_canonical import validate_bundle, write_validation_report

    root = Path(root).absolute()
    source = root / "input/canonical"
    settings = root / "input/analysis.json"
    manifest = root / "input/files.json"
    output = root / "output/result"
    stage = "REQUEST"
    from src.agents.pickle_import_worker import _diagnostic

    try:
        if (request.task_id != root.name or request.stage != "run_eda" or request.action_id != "run_eda"
            or request.policy_version != POLICY_VERSION
            or request.input_files != (str(settings), str(manifest))
            or request.dependency_directories != (str(source),)
            or request.output_directory != str(output) or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("WM worker 요청 경로가 올바르지 않습니다.")

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if (set(packet) != {"version", "source_id", "limits", "max_input_bytes", "max_output_bytes",
                            "include_examples", "max_group_ratio", "code_id"}
            or packet["version"] != POLICY_VERSION or packet["source_id"] != request.source_id
            or packet["limits"] != asdict(request.limits) or digest(packet) != request.config_id
            or type(packet["include_examples"]) is not bool
            or type(packet["max_group_ratio"]) not in (int, float) or not 0 < packet["max_group_ratio"] <= 1):
            raise ValueError("WM worker 설정이 현재 승인 범위와 다릅니다.")

        for key in ("max_input_bytes", "max_output_bytes"):
            if type(packet[key]) is not int or not 1 <= packet[key] <= 2 ** 63 - 1:
                raise ValueError("WM worker 입출력 상한이 올바르지 않습니다.")

        with manifest.open("rb") as stream:
            files = decode_json(stream.read(1024 * 1024 + 1))

        stage = "FILES"
        check_files(files, packet["max_input_bytes"])

        if ("wm_canonical:" + digest(files) != request.source_id
            or inventory(source, packet["max_input_bytes"], staged=True) != files):
            raise ValueError("WM canonical 복사본이 승인된 내용과 다릅니다.")

        analysis = output / "analysis"
        analysis.mkdir(exist_ok=False)
        stage = "QUALITY"
        # Python 임시 폴더의 별도 권한 설정 대신 승인된 scratch의 상속 ACL을 유지한다.
        workspace = root / "scratch/task/validation"
        workspace.mkdir(exist_ok=False)
        report = validate_bundle(source, max_group_ratio=packet["max_group_ratio"], workspace=workspace)
        write_validation_report(report, analysis / "quality")
        quality = "passed" if report["split_assessment"]["status"] == "preconditions_passed" else "review_required"

        if quality == "passed":
            # 알고리즘을 복제하지 않는다. 기존 노트북·CLI와 동일한 WM EDA 코어를 호출한다.
            # 격리 worker에서는 GUI·사용자 화면 접근 없이 파일 출력 백엔드만 사용한다.
            import matplotlib
            matplotlib.use("Agg", force=True)
            from src.wafer_dl.eda import run_eda

            stage = "EDA"
            run_eda(source, analysis / "eda", include_examples=packet["include_examples"])

        if inventory(source, packet["max_input_bytes"], staged=True) != files:
            raise ValueError("분석 중 canonical 복사본이 변경되었습니다.")

        stage = "RESULT"
        records = result_files(analysis, packet["max_output_bytes"])
        result = {"version": POLICY_VERSION, "request_id": request.request_id, "source_id": request.source_id,
                  "config_id": request.config_id, "quality_status": quality, "files": records}
        payload = json.dumps(result, allow_nan=False).encode("utf-8")

        if len(payload) + sum(record["size"] for record in records.values()) > packet["max_output_bytes"]:
            raise ValueError("WM 분석 결과가 승인된 출력 상한을 초과했습니다.")

        (analysis / "result.json").write_bytes(payload)
        _diagnostic(root, "COMPLETE", "PASSED")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("analysis",))
    except (ValueError, TypeError, OSError, KeyError, EOFError) as error:
        # 원문 예외 대신 기존 검증기의 고정된 하위 단계·오류 분류만 허용한다.
        known_stages = {"input_directory", "metadata", "temporary_workspace", "temporary_database",
                        "manifest", "shards", "duplicate_audit", "split_assessment"}
        known_codes = {"FILE_ACCESS_DENIED", "FILE_OR_DIRECTORY_NOT_FOUND", "TEMP_DATABASE_ERROR",
                       "INVALID_METADATA_FILE", "UNREADABLE_OR_INVALID_ARTIFACT", "INVALID_PROVENANCE"}
        substage = getattr(error, "stage", None)
        code = str(error) if str(error) in known_codes else "INPUT_INVALID"
        _diagnostic(root, stage + "_" + substage.upper() if substage in known_stages else stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception as error:
        # 타입 분류만 기록한다. 예외 메시지·컬럼·경로·원본 값은 기록하지 않는다.
        classification = {ImportError: "DEPENDENCY_IMPORT_FAILED", ModuleNotFoundError: "DEPENDENCY_IMPORT_FAILED",
                          AttributeError: "LIBRARY_ATTRIBUTE_FAILED", RuntimeError: "LIBRARY_RUNTIME_FAILED",
                          MemoryError: "MEMORY_EXHAUSTED", AssertionError: "LIBRARY_ASSERTION_FAILED",
                          NotImplementedError: "LIBRARY_NOT_IMPLEMENTED", IndexError: "LIBRARY_INDEX_FAILED",
                          RecursionError: "LIBRARY_RECURSION_FAILED", SystemError: "LIBRARY_SYSTEM_FAILED"}
        _diagnostic(root, stage, classification.get(type(error), "RUNTIME_FAILED"))
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")


def main():
    """실제 AppContainer·capability 부재·Job 소속 확인 후에만 데이터를 읽는다."""
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

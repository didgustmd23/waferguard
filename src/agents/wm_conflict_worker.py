"""승인된 WM 품질 결과와 같은 canonical의 가상 충돌 제외 후보만 산출한다."""

from pathlib import Path
import sys


def execute_request(request, root):
    """NPZ·pickle을 파싱하지 않고 새 scratch DB의 manifest에서 제외 영향을 진단한다."""
    from dataclasses import asdict
    import json
    import sqlite3

    from src.agents.pickle_import_worker import _diagnostic
    from src.agents.wm_canonical_contract import WmCanonicalInput, digest, inventory
    from src.agents.wm_conflict_results import POLICY_VERSION, candidate_files
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.canonical_data import decode_json
    from src.wafer_dl.diagnose_split import compare_conflict_exclusion, write_diagnostic_report
    from src.wafer_dl.validate_canonical import validate_manifest, validate_metadata

    root = Path(root).absolute()
    source = root / "input/canonical"
    settings = root / "input/analysis.json"
    manifest = root / "input/files.json"
    stage = "REQUEST"

    try:
        if (request.task_id != root.name or request.stage != "derive_preparation"
            or request.action_id != "derive_preparation" or request.policy_version != POLICY_VERSION
            or request.input_files != (str(settings), str(manifest)) or request.dependency_directories != (str(source),)
            or request.output_directory != str(root / "output/result")
            or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("충돌 후보 worker 요청이 올바르지 않습니다.")

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if (set(packet) != {"version", "source_id", "limits", "max_input_bytes", "max_output_bytes",
                            "include_examples", "max_group_ratio", "code_id", "quality_evidence"}
            or packet["version"] != POLICY_VERSION or packet["source_id"] != request.source_id
            or packet["limits"] != asdict(request.limits) or digest(packet) != request.config_id
            or packet["include_examples"] is not False):
            raise ValueError("충돌 후보 설정이 승인 범위와 다릅니다.")

        with manifest.open("rb") as stream:
            files = decode_json(stream.read(1024 * 1024 + 1))

        contract = WmCanonicalInput(str(source), json.dumps(files), request.limits,
                                    packet["max_input_bytes"], packet["max_output_bytes"], False, packet["max_group_ratio"])
        evidence = packet["quality_evidence"]

        if (contract.source_id != request.source_id or evidence["source_id"] != request.source_id
            or inventory(source, contract.max_input_bytes, staged=True) != files):
            raise ValueError("충돌 후보 입력과 이전 품질 검사의 묶음이 다릅니다.")

        stage = "MANIFEST"
        summary, version = validate_metadata(source)
        database = sqlite3.connect(str(root / "scratch/task/conflict.sqlite"))
        candidate = Path(request.output_directory) / "candidate"
        candidate.mkdir(exist_ok=False)

        try:
            validate_manifest(source, summary, version, database)
            # 기존 진단 함수의 임시 conflict_hashes와 충돌하지 않는 별도 선택 테이블이다.
            database.execute("CREATE TEMP TABLE selected_hashes AS SELECT hash FROM rows WHERE hash IS NOT NULL "
                             "GROUP BY hash HAVING COUNT(DISTINCT label)>1")
            database.execute("CREATE UNIQUE INDEX selected_hash_index ON selected_hashes(hash)")
            written = 0

            def write_bounded(stream, payload):
                """후보 행·해시 파일이 커져도 요청된 출력 상한을 쓰기 전에 확인한다."""
                nonlocal written
                written += len(payload)

                if written > contract.max_output_bytes:
                    raise ValueError("충돌 후보 출력 상한을 초과했습니다.")

                stream.write(payload)

            with (candidate / "conflict_hashes.jsonl").open("xb") as stream:
                for (hash_value,) in database.execute("SELECT hash FROM selected_hashes ORDER BY hash"):
                    write_bounded(stream, (json.dumps({"map_hash": hash_value}) + "\n").encode("ascii"))

            with (candidate / "excluded_rows.csv").open("xb") as stream:
                write_bounded(stream, b"source_row_position\n")

                for (position,) in database.execute(
                    "SELECT position FROM rows WHERE hash IN (SELECT hash FROM selected_hashes) ORDER BY position",
                ):
                    write_bounded(stream, (str(position) + "\n").encode("ascii"))

            stage = "DIAGNOSIS"
            report = compare_conflict_exclusion(database, contract.max_group_ratio)
        finally:
            database.close()

        before = report["baseline"]

        if (before["manifest_rows"] != evidence["row_count"]
            or before["duplicate_audit"] != evidence["duplicate_audit"]
            or before["split_assessment"] != evidence["split_assessment"]
            or report["candidate_conflict_hash_groups"] <= 0):
            raise ValueError("현재 manifest 진단과 기존 품질 검사 결과가 일치하지 않습니다.")

        write_diagnostic_report(report, candidate / "diagnosis")

        if inventory(source, contract.max_input_bytes, staged=True) != files:
            raise ValueError("가상 진단 중 canonical 복사본이 변경되었습니다.")

        records = candidate_files(candidate, contract.max_output_bytes)
        result = {"version": POLICY_VERSION, "request_id": request.request_id, "source_id": request.source_id,
                  "config_id": request.config_id, "quality_evidence_id": digest(evidence),
                  "status": "candidate_ready", "files": records}
        payload = json.dumps(result, allow_nan=False).encode("utf-8")

        if len(payload) + sum(value["size"] for value in records.values()) > contract.max_output_bytes:
            raise ValueError("충돌 후보 관리 기록이 출력 상한을 초과했습니다.")

        (candidate / "candidate.json").write_bytes(payload)
        _diagnostic(root, "COMPLETE", "PASSED")
        return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("candidate",))
    except (ValueError, TypeError, OSError, KeyError, EOFError, sqlite3.Error):
        _diagnostic(root, stage, "INPUT_INVALID")
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")


def main():
    """기존 격리 토큰·Job 확인을 재사용하며 일반 프로세스의 직접 실행은 거부한다."""
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

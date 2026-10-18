"""고정 합성 입력의 품질 검사 후 검토된 참조 모듈만 실행하는 worker."""

from pathlib import Path
import sys


def execute_request(request, root):
    import json
    import hashlib
    from dataclasses import asdict
    from src.agents.input_snapshot import Snapshot
    from src.agents.reviewed_module_contract import POLICY_VERSION, check_demo, module_identity
    from src.agents.snapshot_eda_worker import _verify_staged_snapshot, _diagnostic
    from src.agents.synthetic_eda_data import CSV_BYTES, fixture_input
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.profiles import profile_from_dict
    from src.preprocessing.quality import inspect_quality, write_quality_report

    root = Path(root)
    source = root / "input/snapshot/input.bin"
    settings = root / "input/eda.json"
    output = root / "output/result"
    stage = "REQUEST_PATHS"

    try:
        if (request.task_id != root.name or request.stage != "run_eda" or request.action_id != "run_eda"
                or request.policy_version != POLICY_VERSION or request.dependency_directories
                or request.input_files != (str(source), str(source.parent / "snapshot.json"), str(settings))
                or request.output_directory != str(output) or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("참조 모듈의 고정 실행 요청이 다릅니다.")

        stage = "SETTINGS"

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if set(packet) != {"contract_id", "snapshot", "profile", "limits", "code_id", "module"}:
            raise ValueError("참조 모듈 설정 필드가 다릅니다.")

        expected = Snapshot(**packet["snapshot"])
        profile = profile_from_dict(packet["profile"])
        include_complete = packet["module"]["settings"]["include_complete"]
        check_demo(expected, profile, include_complete, staged=True)

        config_id = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

        if (packet["module"] != module_identity(include_complete) or request.config_id != config_id
                or request.source_id != expected.source_id or packet["limits"] != asdict(request.limits)):
            raise ValueError("승인한 모듈·입력·설정·자원이 다릅니다.")

        stage = "SNAPSHOT"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("합성 복사본이 변경되었습니다.")

        with source.open("rb") as stream:
            payload = stream.read(len(CSV_BYTES) + 1)

        data = fixture_input(payload)
        target = output / "pipeline"
        target.mkdir()
        stage = "INPUT_VALIDATION"
        report = inspect_quality(data, profile.quality_rules)
        write_quality_report(report, target / "quality")
        completed = ["quality"]

        if report.status == "passed":
            stage = "EDA"
            # 확장 코드 import·계산은 승인된 격리 worker 안에서만 수행한다.
            from src.agents.module_catalogue import build_reviewed_registry, MODULE_ID, VERSION

            module = build_reviewed_registry(staged=True).resolve(MODULE_ID, VERSION)
            table = module.calculate(data, **module.validate_settings({"include_complete": include_complete}))

            if not module.result_validator(table):
                raise ValueError("모듈 결과 계약이 다릅니다.")

            (target / "row_missing").mkdir()
            table.to_csv(target / "row_missing/missing_row_histogram.csv", index=False, encoding="utf-8")
            completed.append("row_missing")

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("실행 중 합성 입력이 변경되었습니다.")

        (target / "pipeline_result.json").write_text(json.dumps({
            "version": "reviewed_module_result_v1", "config_id": request.config_id,
            "module": module_identity(include_complete), "row_count": len(data.frame),
            "status": report.status, "completed_steps": completed,
        }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    except (ValueError, TypeError, KeyError, UnicodeError, OSError):
        _diagnostic(root, stage, "INPUT_INVALID")
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("pipeline",))


def main():
    """기존 복사형 runtime 부팅 절차를 사용하고 원문 예외는 출력하지 않는다."""
    import os
    root = Path(sys.argv[1]).resolve()
    runtime = root / "runtime"
    sys.path[:0] = [str(root / "tools"), str(runtime / "Lib/site-packages")]
    handles = [os.add_dll_directory(str(path)) for path in (runtime, runtime / "DLLs", runtime / "Library/bin") if path.is_dir()]
    from src.agents.worker_contract import WorkerRequest

    try:
        with (root / "input/request.json").open("rb") as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))

        if request.request_id != sys.argv[2]:
            return 2

        result = execute_request(request, root)
        (root / "output/result/worker_result.json").write_text(result.to_json(), encoding="utf-8")
        return 0 if result.status == "succeeded" else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""승인된 CSV 복사본에서 고정 등록 모듈만 순차 실행하는 격리 worker."""

from pathlib import Path
import sys


def execute_request(request, root):
    """요청·설정·snapshot을 대조하고 검토 필요 시 후속 모듈을 실행하지 않는다."""
    import csv
    from dataclasses import asdict
    import hashlib
    import json

    import pandas as pd
    from src.agents.input_snapshot import Snapshot
    from src.agents.module_pipeline_contract import POLICY_VERSION, bind_pipeline, MODULE_FILES
    from src.agents.snapshot_eda_worker import _diagnostic, _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.builtin_modules import build_builtin_registry
    from src.preprocessing.contracts import TabularInput
    from src.preprocessing.profiles import profile_from_dict
    from src.preprocessing.quality import write_quality_report

    root = Path(root).resolve()
    source = root / "input" / "snapshot" / "input.bin"
    settings = root / "input" / "eda.json"
    output = root / "output" / "result"
    stage = "REQUEST_PATHS"

    try:
        if (
            request.task_id != root.name or request.stage != "run_eda" or request.action_id != "run_eda"
            or request.policy_version != POLICY_VERSION or request.dependency_directories
            or request.input_files != (str(source), str(source.parent / "snapshot.json"), str(settings))
            or request.output_directory != str(output) or request.scratch_directory != str(root / "scratch" / "task")
        ):
            raise ValueError("모듈 작업의 고정 요청 계약이 일치하지 않습니다.")

        stage = "SETTINGS"

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if set(packet) != {"contract_id", "snapshot", "profile", "limits", "code_id", "pipeline", "pipeline_signature"}:
            raise ValueError("모듈 작업 설정 필드가 올바르지 않습니다.")

        config_id = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        expected = Snapshot(**packet["snapshot"])
        profile = profile_from_dict(packet["profile"])
        _, plan = bind_pipeline(packet["pipeline"], profile)

        if (
            request.config_id != config_id or request.source_id != expected.source_id
            or packet["limits"] != asdict(request.limits) or profile.adapter_id != "tabular_csv_v1"
            or packet["pipeline_signature"] != plan.signature
        ):
            raise ValueError("모듈 계획·프로파일·입력이 승인 범위와 다릅니다.")

        stage = "SNAPSHOT"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("모듈 작업의 입력 복사본이 변경되었습니다.")

        stage = "CSV_LOAD"

        with source.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])

        if not header or len(header) != len(set(header)):
            raise ValueError("CSV 헤더 계약이 올바르지 않습니다.")

        frame = pd.read_csv(source, encoding="utf-8-sig")

        if frame.empty:
            raise ValueError("실행할 입력 행이 없습니다.")

        data = TabularInput(frame, profile.roles)
        registry = build_builtin_registry()
        completed = []
        status = "passed"
        target = output / "pipeline"
        target.mkdir(exist_ok=False)

        for step in plan.steps:
            stage = "MODULE_EXECUTION"

            if not set(step.depends_on).issubset(completed):
                raise ValueError("선행 단계가 완료되지 않았습니다.")

            module = registry.resolve(step.module_id, step.version)
            # 품질 모듈은 내용 오류를 보고해야 하므로 EDA의 엄격 검증은 EDA 코어에 맡긴다.
            result = module.calculate(data, **module.validate_settings(json.loads(step.settings_json)))

            if not module.result_validator(result):
                raise ValueError("모듈의 메모리 결과 계약 검증에 실패했습니다.")

            directory = target / step.step_id

            if step.module_id == "tabular_quality":
                write_quality_report(result, directory)
                status = result.status
            else:
                directory.mkdir(exist_ok=False)
                tables = result if step.module_id == "eda_correlations" else {Path(MODULE_FILES[step.module_id][0]).stem: result}

                for name, table in tables.items():
                    table.to_csv(directory / f"{name}.csv", encoding="utf-8-sig")

            completed.append(step.step_id)

            # 검사가 실행됐어도 품질 통과와는 다르다. 후속 분석을 호출하지 않고 중단한다.
            if status != "passed":
                break

        stage = "SNAPSHOT_RECHECK"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("모듈 실행 중 입력이 변경되었습니다.")

        (target / "pipeline_result.json").write_text(json.dumps({
            "version": "module_pipeline_result_v1", "pipeline_signature": plan.signature,
            "row_count": len(frame), "status": status, "completed_steps": completed,
        }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    except (ValueError, TypeError, KeyError, UnicodeError, OSError, csv.Error) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("pipeline",))


def main():
    """복사된 고정 코드·런타임으로만 부팅하며 원문 예외·값은 출력하지 않는다."""
    import os

    root = Path(sys.argv[1]).resolve()
    runtime = root / "runtime"
    sys.path[:0] = [str(root / "tools"), str(runtime / "Lib" / "site-packages")]
    handles = [os.add_dll_directory(str(path)) for path in (runtime, runtime / "DLLs", runtime / "Library" / "bin") if path.is_dir()]

    from src.agents.worker_contract import WorkerRequest

    try:
        with (root / "input" / "request.json").open("rb") as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))

        if request.request_id != sys.argv[2]:
            return 2

        result = execute_request(request, root)
        (root / "output" / "result" / "worker_result.json").write_text(result.to_json(), encoding="utf-8")
        return 0 if result.status == "succeeded" else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

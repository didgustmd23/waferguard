"""CSV snapshot의 비파괴 품질 검사 worker. EDA·학습·자동 보정은 하지 않는다."""

from pathlib import Path
import sys


def execute_request(request, root):
    """입력·설정을 대조하고 기존 inspect_quality의 결과를 로컬 파일로 기록한다."""
    import csv
    from dataclasses import asdict
    import hashlib
    import json

    import pandas as pd
    from src.agents.input_snapshot import Snapshot
    from src.agents.snapshot_eda_worker import _diagnostic, _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.contracts import TabularInput
    from src.preprocessing.profiles import profile_from_dict
    from src.preprocessing.quality import QualityFinding, QualityReport, inspect_quality, write_quality_report

    root = Path(root).resolve()
    directory = root / "input" / "snapshot"
    settings = root / "input" / "audit.json"
    output = root / "output" / "result"
    stage = "REQUEST_PATHS"

    try:
        if (
            request.task_id != root.name or request.stage != "audit" or request.action_id != "run_audit"
            or request.dependency_directories
            or request.input_files != (str(directory / "input.bin"), str(directory / "snapshot.json"), str(settings))
            or request.output_directory != str(output)
            or request.scratch_directory != str(root / "scratch" / "task")
        ):
            raise ValueError("품질 검사 요청의 고정 경로가 일치하지 않습니다.")

        stage = "SETTINGS"
        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if set(packet) != {"contract_id", "snapshot", "profile", "limits", "code_id"}:
            raise ValueError("품질 검사 설정 필드가 올바르지 않습니다.")

        config_id = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        expected = Snapshot(**packet["snapshot"])
        stage = "PROFILE"
        profile = profile_from_dict(packet["profile"])

        if (
            request.config_id != config_id or request.source_id != expected.source_id
            or packet["limits"] != asdict(request.limits) or profile.adapter_id != "tabular_csv_v1"
        ):
            raise ValueError("품질 검사 입력·설정이 승인 계약과 다릅니다.")

        stage = "SNAPSHOT"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("품질 검사 snapshot이 일치하지 않습니다.")

        # 품질 검사에서는 잘못된 타입·필수 컬럼 누락도 근거로 남겨야 하므로
        # TabularInput.validate()를 먼저 호출하지 않는다. 구조 검사는 inspect_quality가 수행한다.
        try:
            source = directory / "input.bin"
            stage = "CSV_HEADER"

            with source.open(encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])

            if not header or len(header) != len(set(header)):
                raise ValueError("CSV 헤더 계약 위반")

            stage = "CSV_LOAD"
            frame = pd.read_csv(source, encoding="utf-8-sig")
        except (ValueError, UnicodeError, csv.Error):
            report = QualityReport("blocked_input", (QualityFinding("INPUT_LOAD_CONTRACT_INVALID", "block"),))
        else:
            stage = "INPUT_VALIDATION"
            report = inspect_quality(TabularInput(frame, profile.roles), profile.quality_rules)

        stage = "SNAPSHOT_RECHECK"
        if not _verify_staged_snapshot(root, expected):
            raise ValueError("검사 중 snapshot이 변경되었습니다.")

        write_quality_report(report, output / "quality")
    except (ValueError, TypeError, KeyError, OSError) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    # 프로세스 작업 성공과 데이터 품질 통과는 별개다. 부모가 quality.json의 상태를 판정한다.
    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("quality",))


def main():
    """복사한 런타임·고정 코드만 사용하며 상세 예외를 외부 응답에 복사하지 않는다."""
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

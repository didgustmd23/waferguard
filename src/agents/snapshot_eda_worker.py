"""등록 CSV 로더로 snapshot을 읽는 격리 진입점. pickle·임의 모듈은 실행하지 않는다."""

from pathlib import Path
import sys


DIAGNOSTIC_STAGES = frozenset({
    "REQUEST_PATHS", "SETTINGS", "PROFILE", "SNAPSHOT", "CSV_HEADER", "CSV_LOAD",
    "INPUT_VALIDATION", "SNAPSHOT_RECHECK", "EDA", "COMPLETE",
})


def _diagnostic(root, stage, code):
    """고정 단계·오류 분류만 로컬 scratch에 기록한다. 경로·원문 예외·값은 저장하지 않는다."""
    import json

    if stage not in DIAGNOSTIC_STAGES or code not in {"PASSED", "ACCESS_DENIED", "INPUT_INVALID", "IO_FAILED", "RUNTIME_FAILED"}:
        return

    try:
        (root / "scratch" / "task" / "diagnostic.json").write_text(
            json.dumps({"stage": stage, "code": code}), encoding="utf-8",
        )
    except OSError:
        # 진단 기록 실패를 성공으로 바꾸거나 기존 오류를 숨기지 않는다.
        pass


def _verify_staged_snapshot(root, expected, *, name="snapshot"):
    """부모가 검증·복사·읽기 전용으로 허용한 input 내부만 검사한다.

    부모의 전체 경로 검사를 대체하는 범용 API가 아니다. worker는 작업 요청으로
    고정된 input 아래의 디렉터리·두 파일만 확인하며 허용 밖의 조상은 조회하지 않는다.
    """
    from dataclasses import asdict
    import os
    import stat

    from src.agents.input_snapshot import PAYLOAD_NAME, SNAPSHOT_VERSION, Snapshot, _scan
    from src.agents.worker_contract import _decode

    root = Path(root)

    if not root.is_absolute() or ".." in root.parts or not isinstance(expected, Snapshot) or name not in {"snapshot", "train", "validation"}:
        raise ValueError("고정 작업 경로와 기대 snapshot 계약이 필요합니다.")

    allowed = root / "input"
    # 기존 단일 snapshot과 승인된 구간 이름만 받는다. 임의 상대 경로는 허용하지 않는다.
    directory = allowed / name

    # 전체 조상 검사는 부모에서 이미 수행했다. 여기서는 허용된 경계 아래만 조회한다.
    for path in (allowed, directory):
        info = path.lstat()

        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return False

    if {path.name for path in directory.iterdir()} != {PAYLOAD_NAME, "snapshot.json"}:
        return False

    for name in (PAYLOAD_NAME, "snapshot.json"):
        info = (directory / name).lstat()

        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or getattr(info, "st_file_attributes", 0) & 0x400:
            return False

    # manifest도 크기를 제한하며 해시·크기·파일 상한 모두 기대 계약과 대조한다.
    with (directory / "snapshot.json").open("rb") as stream:
        metadata = _decode(stream.read(4097).decode("utf-8"))

    if metadata != {"version": SNAPSHOT_VERSION, **asdict(expected)}:
        return False

    with (directory / PAYLOAD_NAME).open("rb") as stream:
        info = os.fstat(stream.fileno())

        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != expected.size_bytes:
            return False

        return _scan(stream, expected.max_input_bytes) == (expected.size_bytes, expected.sha256)


def execute_request(request, root):
    """요청·입력 해시·설정을 확인하고 공통 EDA를 호출한다. 상세 오류는 반환하지 않는다."""
    import csv
    from dataclasses import asdict
    import hashlib
    import json

    import pandas as pd
    from src.agents.input_snapshot import Snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.contracts import TabularInput
    from src.preprocessing.profiles import profile_from_dict
    from src.preprocessing.eda_report import generate_eda_report

    root = Path(root).resolve()
    snapshot_dir = root / "input" / "snapshot"
    settings = root / "input" / "eda.json"
    output = root / "output" / "result"
    stage = "REQUEST_PATHS"

    try:
        # CLI가 전달한 request_id는 main에서 대조한다. 여기서는 고정 작업 경로도 검사한다.
        if (
            request.task_id != root.name or request.action_id != "run_eda" or request.stage != "run_eda"
            or request.dependency_directories
            or request.input_files != (str(snapshot_dir / "input.bin"), str(snapshot_dir / "snapshot.json"), str(settings))
            or request.output_directory != str(output)
            or request.scratch_directory != str(root / "scratch" / "task")
        ):
            raise ValueError("작업 요청 경로가 일치하지 않습니다.")

        # 설정 크기를 제한한 뒤 중복 키·비표준 숫자를 거부하는 JSON 파서를 사용한다.
        stage = "SETTINGS"
        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if set(packet) != {"contract_id", "snapshot", "profile", "limits", "code_id"}:
            raise ValueError("EDA 설정 필드가 올바르지 않습니다.")

        config_id = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        expected = Snapshot(**packet["snapshot"])
        stage = "PROFILE"
        profile = profile_from_dict(packet["profile"])

        if (
            request.config_id != config_id or request.source_id != expected.source_id
            or packet["limits"] != asdict(request.limits)
            or profile.adapter_id != "tabular_csv_v1"
        ):
            raise ValueError("EDA 입력·설정 식별값이 일치하지 않습니다.")

        stage = "SNAPSHOT"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("작업용 snapshot이 일치하지 않습니다.")

        # 확장자가 아니라 등록 ID로 CSV 로더를 선택한다. 다른 로더는 아직 등록하지 않는다.
        source = snapshot_dir / "input.bin"
        stage = "CSV_HEADER"

        with source.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])

        if not header or len(header) != len(set(header)):
            raise ValueError("CSV 헤더가 비어 있거나 중복됩니다.")

        stage = "CSV_LOAD"
        frame = pd.read_csv(source, encoding="utf-8-sig")
        stage = "INPUT_VALIDATION"
        data = TabularInput(frame, profile.roles).validate()

        # 로더가 읽은 동안 바뀐 입력도 완료 처리하지 않는다.
        stage = "SNAPSHOT_RECHECK"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("읽는 동안 snapshot이 변경되었습니다.")

        stage = "EDA"
        result = generate_eda_report(data, output / "eda", max_plot_features=profile.max_plot_features)

        if result != {"status": "eda_complete"}:
            raise RuntimeError("EDA 결과 상태가 올바르지 않습니다.")

    except (ValueError, TypeError, KeyError, UnicodeError, OSError) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("eda",))


def main():
    """복사된 런타임에서만 부팅하고 요청 식별값이 다르면 실행하지 않는다."""
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

        import matplotlib
        matplotlib.use("Agg")
        result = execute_request(request, root)
        (root / "output" / "result" / "worker_result.json").write_text(result.to_json(), encoding="utf-8")
        return 0 if result.status == "succeeded" else 2
    except Exception:
        # 원문 예외·값·절대 경로를 stdout/stderr에 전달하지 않는다.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

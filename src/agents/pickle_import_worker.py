"""AppContainer에서 pickle 구조를 판별해 지원하는 canonical 변환기를 실행한다."""

from pathlib import Path
import sys


def _isolated_process():
    """진입점 실행 전에 실제 AppContainer 토큰·capability 부재·Job 소속을 확인한다."""
    import ctypes
    from ctypes import wintypes
    import os

    if os.name != "nt":
        return False

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    process = kernel.GetCurrentProcess()

    if not advapi.OpenProcessToken(process, 8, ctypes.byref(token)):
        return False

    try:
        flag = wintypes.DWORD()
        needed = wintypes.DWORD()

        if not advapi.GetTokenInformation(token, 29, ctypes.byref(flag), ctypes.sizeof(flag), ctypes.byref(needed)) or flag.value != 1:
            return False

        # capability 조회 실패를 네트워크 권한 없음으로 간주하지 않는다.
        needed = wintypes.DWORD()
        advapi.GetTokenInformation(token, 30, None, 0, ctypes.byref(needed))

        if not 4 <= needed.value <= 65536:
            return False

        buffer = ctypes.create_string_buffer(needed.value)

        if not advapi.GetTokenInformation(token, 30, buffer, len(buffer), ctypes.byref(needed)):
            return False

        if wintypes.DWORD.from_buffer(buffer).value != 0:
            return False

        in_job = wintypes.BOOL()

        return bool(kernel.IsProcessInJob(process, None, ctypes.byref(in_job)) and in_job.value)
    finally:
        kernel.CloseHandle(token)


def _diagnostic(root, stage, code):
    """단계·고정 오류 분류만 남겨 원본 값이나 예외 상세가 기록되지 않게 한다."""
    import json

    try:
        (root / "scratch/task/diagnostic.json").write_text(
            json.dumps({"stage": stage, "code": code}), encoding="utf-8",
        )
    except OSError:
        pass


def execute_request(request, root):
    """고정 요청·복사본을 확인한 뒤 기존 변환기로 새 묶음을 만든다."""
    from dataclasses import asdict
    import json

    from src.agents.input_snapshot import Snapshot
    from src.agents.snapshot_eda_worker import _verify_staged_snapshot
    from src.agents.pickle_contract import ADAPTER_ID, POLICY_VERSION, PickleInput, digest
    from src.agents.worker_contract import WorkerResult, _decode

    root = Path(root).absolute()
    source = root / "input/snapshot/input.bin"
    settings = root / "input/import.json"
    output = root / "output/result"
    stage = "REQUEST"

    try:
        # 경로를 요청에서 자유롭게 받지 않고 부모가 준비한 고정 작업 영역만 사용한다.
        if (
            request.task_id != root.name or request.stage != "ingest"
            or request.action_id != "run_ingest" or request.policy_version != POLICY_VERSION
            or request.dependency_directories
            or request.input_files != (str(source), str(source.parent / "snapshot.json"), str(settings))
            or request.output_directory != str(output)
            or request.scratch_directory != str(root / "scratch/task")
        ):
            raise ValueError("pickle worker 요청 경로가 올바르지 않습니다.")

        stage = "SETTINGS"

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if set(packet) != {
            "version", "adapter_id", "snapshot", "limits", "max_output_bytes", "shard_rows", "code_id",
            "converter_id", "select_path",
        }:
            raise ValueError("pickle worker 설정 필드가 올바르지 않습니다.")

        expected = Snapshot(**packet["snapshot"])
        if type(packet["select_path"]) is not list:
            raise ValueError("선택 경로는 JSON 배열이어야 합니다.")

        PickleInput(str(source.parent), expected, request.limits, packet["max_output_bytes"],
                    packet["shard_rows"], packet["converter_id"], tuple(packet["select_path"]))

        if (
            packet["version"] != POLICY_VERSION or packet["adapter_id"] != ADAPTER_ID
            or digest(packet) != request.config_id or packet["limits"] != asdict(request.limits)
            or request.source_id != expected.source_id or not _verify_staged_snapshot(root, expected)
        ):
            raise ValueError("pickle 복사본·설정·자원 요청이 일치하지 않습니다.")

        import pandas as pd
        from src.agents.pickle_converters import analyze_structure, export_review, export_selected

        stage = "PICKLE_LOAD"
        # 이 역직렬화 호출은 전용 AppContainer 진입점에만 둔다.
        # 부모 실행기·승인 CLI·검증기는 원본 pickle을 읽지 않는다.
        payload = pd.read_pickle(source, compression=None)

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("변환 중 pickle 복사본이 변경되었습니다.")

        stage = "STRUCTURE"
        value, report = analyze_structure(payload, packet["converter_id"], tuple(packet["select_path"]))

        if report["status"] == "review_required":
            export_review(output / "review", report, request, packet["max_output_bytes"])
            _diagnostic(root, "STRUCTURE", "REVIEW_REQUIRED")
            return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("review",))

        stage = "EXPORT"
        export_selected(value, output / "canonical", packet, report, request)

    except (ValueError, TypeError, KeyError, UnicodeError, OSError, EOFError) as error:
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(
        request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", ("canonical",),
    )


def main():
    """복사한 런타임·허용 소스에서 부팅하고 요청 식별값을 대조한다."""
    import os

    # 사용자가 진입점을 일반 Python 프로세스에서 직접 실행해도 원본은 읽지 않는다.
    try:
        isolated = len(sys.argv) == 3 and _isolated_process()
    except (OSError, ValueError, AttributeError):
        return 2

    if not isolated:
        return 2

    root = Path(sys.argv[1]).absolute()
    runtime = root / "runtime"
    sys.path[:0] = [str(root / "tools"), str(runtime / "Lib/site-packages")]
    handles = [os.add_dll_directory(str(path)) for path in (
        runtime, runtime / "DLLs", runtime / "Library/bin",
    ) if path.is_dir()]

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
